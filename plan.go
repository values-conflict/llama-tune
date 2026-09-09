package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// fitProbe is the result of one `llama fit-params` call for a kv type.
// The printed args give the placement; the trace log gives the memory
// projection at the requested ctx, which anchors the local memory model.
type fitProbe struct {
	KV        string
	Requested int
	Fitted    int // -c from the printed args; equals Requested when it fits
	Fits      bool
	NGl       string
	OT        string
	SizeMiB   int64 // "projected to use N MiB" at the requested ctx
	FreeMiB   int64 // "vs. N MiB of free/total memory"
	Err       error
}

var fitLineRe = regexp.MustCompile(`projected to use (\d+) MiB of (?:device|host) memory vs\. (\d+) MiB of (?:free|total)(?: (?:device|host))? memory`)

// runFitParams asks llama.cpp to fit ctx on the local hardware with the
// given kv cache type and returns the placement plus the memory projection.
func runFitParams(llamaBin, modelRef string, modelLocal bool, kv string, ctx, marginMiB int) fitProbe {
	var args []string
	if modelLocal {
		args = append(args, "fit-params", "-m", modelRef)
	} else {
		args = append(args, "fit-params", "-hf", modelRef)
	}
	args = append(args,
		"-c", strconv.Itoa(ctx),
		"-ctk", kv, "-ctv", kv,
		"--fit-target", strconv.Itoa(marginMiB),
		"-lv", "4",
	)
	out, err := exec.Command(llamaBin, args...).CombinedOutput()
	if err != nil {
		return fitProbe{KV: kv, Requested: ctx, Err: fmt.Errorf("fit-params: %w\n%s", err, string(out))}
	}
	fp := fitProbe{KV: kv, Requested: ctx}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "-c ") {
			fields := strings.Fields(line)
			if len(fields) >= 4 {
				fp.Fitted, _ = strconv.Atoi(fields[1])
				fp.NGl = fields[3]
				for i, f := range fields {
					if f == "-ot" && i+1 < len(fields) {
						fp.OT = strings.Trim(strings.Join(fields[i+1:], " "), "\"")
					}
				}
				fp.Fits = fp.Fitted >= ctx
			}
			continue
		}
		if m := fitLineRe.FindStringSubmatch(line); m != nil {
			fp.SizeMiB, _ = strconv.ParseInt(m[1], 10, 64)
			fp.FreeMiB, _ = strconv.ParseInt(m[2], 10, 64)
		}
	}
	if fp.Fitted == 0 {
		fp.Err = fmt.Errorf("fit-params printed no args:\n%s", string(out))
	}
	return fp
}

// memModel answers "does (kv, ctx, parallel) fit, and how much is left".
// Anchor: fit-params' projection at (kv, trainedCtx, 1 slot) minus the KV
// part gives weights+compute, which is constant for the model.
type memModel struct {
	probes  map[string]fitProbe // per kv type
	res     *Resources
	margin  int64
	osResv  int64
	envMiB  int64 // planning envelope
	kvMiB   map[string]float64 // MiB per token per kv type
	wPlusC  map[string]int64   // weights + compute, per kv type
	anchor  int                // ctx the projections were measured at
}

func newMemModel(res *Resources, m *ggufModel, probes []fitProbe, marginMiB, osReserveMiB int64, anchorCtx int) *memModel {
	mm := &memModel{
		probes: map[string]fitProbe{},
		res:    res,
		margin: marginMiB,
		osResv: osReserveMiB,
		kvMiB:  map[string]float64{},
		wPlusC: map[string]int64{},
		anchor: anchorCtx,
	}
	var devFree int64
	for _, p := range probes {
		mm.probes[p.KV] = p
		if p.Err != nil || p.SizeMiB <= 0 {
			continue
		}
		kt, _ := m.kvBytesPerToken(p.KV, p.KV)
		mm.kvMiB[p.KV] = kt / (1024 * 1024)
		mm.wPlusC[p.KV] = p.SizeMiB - int64(mm.kvMiB[p.KV]*float64(anchorCtx))
		if devFree == 0 || p.FreeMiB < devFree {
			devFree = p.FreeMiB
		}
	}
	// envelope: the smaller of what the device reports and what the
	// container may actually use (cgroup limit), minus an OS reserve
	env := res.MemAvailMiB - osReserveMiB
	if devFree > 0 && devFree < env {
		env = devFree
	}
	mm.envMiB = env
	return mm
}

func (mm *memModel) usable(kv string) bool {
	_, ok := mm.wPlusC[kv]
	return ok
}

// usageMiB estimates total memory for (kv, ctx, parallel slots).
func (mm *memModel) usageMiB(kv string, ctx, parallel int) int64 {
	return mm.wPlusC[kv] + int64(mm.kvMiB[kv]*float64(ctx)*float64(parallel))
}

// fits reports whether (kv, ctx, parallel) fits and the headroom in MiB.
func (mm *memModel) fits(kv string, ctx, parallel int) (bool, int64) {
	head := mm.envMiB - mm.margin - mm.usageMiB(kv, ctx, parallel)
	return head >= 0, head
}

// cacheRamMiB is the projected room for the server prompt cache.
func (mm *memModel) cacheRamMiB(kv string, ctx, parallel int) int64 {
	ok, head := mm.fits(kv, ctx, parallel)
	if !ok {
		return 0
	}
	v := head / 256 * 256 // round down to a readable step
	if v > 16384 {
		v = 16384
	}
	return v
}

func ladder(max, step int) []int {
	var out []int
	for d := step; d <= max; d *= 2 {
		out = append(out, d)
	}
	return out
}

func joinInts(v []int) string {
	s := make([]string, len(v))
	for i, n := range v {
		s[i] = strconv.Itoa(n)
	}
	return strings.Join(s, ", ")
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// cmdPlan emits a sweep config derived from the model and the local
// hardware. The user reviews and tweaks it before running.
func cmdPlan(args []string) error {
	fs := flag.NewFlagSet("plan", flag.ExitOnError)
	model := fs.String("hf", "", "hf ref or local gguf path (required)")
	llamaBin := fs.String("llama-bin", "llama", "path to the unified llama binary")
	ctx := fs.Int("ctx", 0, "target context size (default: the model's trained ctx)")
	margin := fs.Int64("margin", 1024, "memory margin to keep free, MiB")
	osReserve := fs.Int64("os-reserve", 2560, "memory reserved for the OS/container bookkeeping, MiB")
	minTradeoffCtx := fs.Int("min-ctx", 32768, "smallest ctx to offer in parallel trade-off sections")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" {
		return fmt.Errorf("-hf is required")
	}
	local := !isHFRef(*model)
	modelPath, err := resolveModel(*llamaBin, *model)
	if err != nil {
		return err
	}
	m, err := scanGGUF(modelPath)
	if err != nil {
		return err
	}
	target := *ctx
	if target <= 0 {
		target = int(m.NContext)
	}
	if target <= 0 {
		return fmt.Errorf("no context size: pass -ctx")
	}
	res, err := probeResources()
	if err != nil {
		return err
	}

	var probes []fitProbe
	for _, kv := range []string{"q4_0", "q8_0", "f16"} {
		fmt.Fprintf(os.Stderr, "fit-params: probing kv %s at ctx %d ...\n", kv, target)
		fp := runFitParams(*llamaBin, *model, local, kv, target, int(*margin))
		probes = append(probes, fp)
		if fp.Err != nil {
			fmt.Fprintf(os.Stderr, "  skipped: %v\n", fp.Err)
			continue
		}
		fmt.Fprintf(os.Stderr, "  -> fits=%v (fitted ctx %d, ngl %s, %d MiB used of %d MiB)\n",
			fp.Fits, fp.Fitted, fp.NGl, fp.SizeMiB, fp.FreeMiB)
	}
	if len(probes) == 0 {
		return fmt.Errorf("no kv type could be probed")
	}
	mm := newMemModel(res, m, probes, *margin, *osReserve, target)
	if mm.envMiB <= 0 {
		return fmt.Errorf("could not determine a memory envelope; pass --margin or -os-reserve")
	}

	// primary = first kv type (smallest first) that fits the target ctx;
	// otherwise the one that fits the most context
	primaryKV := probes[0].KV
	primaryFitted := 0
	for _, p := range probes {
		if p.Err != nil {
			continue
		}
		if p.Fits {
			primaryKV = p.KV
			primaryFitted = p.Fitted
			break
		}
		if p.Fitted > primaryFitted {
			primaryKV, primaryFitted = p.KV, p.Fitted
		}
	}
	if probes[0].Err != nil && primaryFitted == 0 {
		primaryFitted = target
	}
	ctxSize := primaryFitted
	if ctxSize <= 0 {
		ctxSize = target
	}
	if probes[0].Err == nil && mm.probes[primaryKV].Fitted < target {
		fmt.Fprintf(os.Stderr, "note: %d does not fully fit with %s kv; using ctx %d\n", target, primaryKV, ctxSize)
	}

	baseOK, _ := mm.fits(primaryKV, ctxSize, 1)
	cacheRam := mm.cacheRamMiB(primaryKV, ctxSize, 1)
	if !baseOK {
		fmt.Fprintf(os.Stderr, "note: baseline itself does not fit in the envelope; see the trade-off sections\n")
	}

	var b strings.Builder
	c := func(format string, a ...any) { fmt.Fprintf(&b, format+"\n", a...) }

	c("# generated by llama-tune plan")
	c("# model: %s", *model)
	c("# arch: %s, %d layers, embd %d, trained ctx %d, gqa %d kv heads, mtp %v, moe %v",
		m.Arch, m.BlockCount, m.NEmbd, m.NContext, m.NKVHeads, m.MTP, m.NExperts > 0)
	c("# hardware: %s", res.String())
	c("# memory model: envelope %d MiB (margin %d, os reserve %d)", mm.envMiB, *margin, *osReserve)
	for _, p := range probes {
		if p.Err != nil {
			c("# fit kv=%s: probe failed", p.KV)
			continue
		}
		c("# fit kv=%s: requested %d -> fitted %d, %d MiB of %d MiB, ngl %s",
			p.KV, p.Requested, p.Fitted, p.SizeMiB, p.FreeMiB, p.NGl)
	}
	c("#")
	c("# active candidates benchmark speed (one knob at a time from baseline)")
	c("# commented sections are memory/capacity trade-offs: same t/s per token,")
	c("# different context size, parallelism, and prompt cache - pick consciously")
	c("")
	if local {
		c("model = %s", *model)
	} else {
		c("hf = %s", *model)
	}
	c("ctx-size = %d", ctxSize)
	c("jinja = true")
	c("parallel = 1")
	c("kv-unified = true")
	c("cache-type-k = %s", primaryKV)
	c("cache-type-v = %s", primaryKV)
	c("# prompt cache sized from the projected headroom (0 = no room left)")
	c("cache-ram = %d", cacheRam)
	c("")
	c("[tier1]")
	c("pp = 4096")
	c("gen = 64")
	c("repeats = 2")
	c("depths = %s", joinInts(ladder(minInt(ctxSize, 65536), 1024)))
	c("")
	c("[tier2]")
	c("gen = 64")
	c("repeats = 1")
	c("warmup = true")
	c("depths = %s", joinInts(ladder(ctxSize, 4096)))
	c("")

	c("[baseline]")
	p := mm.probes[primaryKV]
	if p.OT != "" {
		c("n-gpu-layers = %s", p.NGl)
		c("override-tensor = %s", p.OT)
	} else {
		c("n-gpu-layers = all")
	}
	c("threads = %d", res.PhysicalCPUs)
	c("flash-attn = true")
	c("batch-size = 4096")
	c("ubatch-size = 1024")
	if m.MTP {
		c("spec-type = draft-mtp")
		c("spec-draft-n-max = 2")
	} else {
		c("spec-type = ngram-mod")
		c("spec-ngram-mod-n-min = 48")
		c("spec-ngram-mod-n-max = 64")
		c("spec-ngram-mod-n-match = 24")
	}
	c("")
	c("[baseline-nospec]")
	c("inherit = baseline")
	c("spec-type = none")
	c("")
	c("[no-fa]")
	c("inherit = baseline-nospec")
	c("flash-attn = false")
	c("spec-type = none")
	c("depths = %s", joinInts(ladder(minInt(ctxSize, 65536), 4096)))
	c("")

	// kv cache variants, only when they fit at all
	for _, p := range probes {
		if p.KV == primaryKV || p.Err != nil || !mm.usable(p.KV) {
			continue
		}
		ok, head := mm.fits(p.KV, p.Fitted, 1)
		depthCap := p.Fitted
		if !ok {
			depthCap = p.Fitted / 2
		}
		c("# kv cache variant: %s (fits ctx %d, headroom %d MiB)", p.KV, p.Fitted, head)
		c("[kv-" + p.KV + "]")
		c("inherit = baseline-nospec")
		c("cache-type-k = %s", p.KV)
		c("cache-type-v = %s", p.KV)
		c("ctx-size = %d", p.Fitted)
		c("spec-type = none")
		c("depths = %s", joinInts(ladder(minInt(depthCap, 131072), 4096)))
		c("")
	}

	// thread counts from the container's actual CPU view
	threadSet := []int{res.PhysicalCPUs}
	if res.EffectiveCPUs != res.PhysicalCPUs {
		threadSet = append(threadSet, res.EffectiveCPUs)
	}
	for _, t := range threadSet {
		if t == res.PhysicalCPUs {
			continue
		}
		c("[threads-%d]", t)
		c("inherit = baseline-nospec")
		c("threads = %d", t)
		c("spec-type = none")
		c("depths = %s", joinInts(ladder(minInt(ctxSize, 65536), 4096)))
		c("")
	}

	// performance-core pin: only useful when CPU inference carries weight,
	// i.e. the fit probe left layers on the CPU (and only exists on P/E parts).
	// Threads = physical P-cores: one thread per core, SMT off. The plain
	// threads-* candidates above already cover full-width SMT on/off.
	if len(res.FastCPUs) > 0 && p.NGl != "all" && p.OT == "" {
		th := res.FastPhysical
		if th < 1 {
			th = len(res.FastCPUs)
		}
		c("# P-core cluster: %s (%d physical). One thread per P-core, SMT off.", joinInts(res.FastCPUs), th)
		c("[fast-cpus]")
		c("inherit = baseline-nospec")
		c("threads = %d", th)
		c("pin-cpus = %s", joinInts(res.FastCPUs))
		c("spec-type = none")
		c("depths = %s", joinInts(ladder(minInt(ctxSize, 65536), 4096)))
		c("")
	}

	// prompt processing: ubatch ladder, cheap enough to run in tier1 only
	for _, ub := range []int{512, 2048} {
		c("# prompt processing only: llama-bench pp tests, no server needed")
		c("[ubatch-" + strconv.Itoa(ub) + "]")
		c("inherit = baseline-nospec")
		c("ubatch-size = %d", ub)
		c("spec-type = none")
		c("tier2 = no")
		c("")
	}

	if m.MTP {
		for _, n := range []int{1, 3} {
			c("[spec" + strconv.Itoa(n) + "]")
			c("inherit = baseline")
			c("spec-draft-n-max = %d", n)
			c("tier1 = no")
			c("")
		}
	}
	c("[spec-ngram]")
	c("inherit = baseline-nospec")
	c("spec-type = ngram-mod")
	c("spec-ngram-mod-n-min = 48")
	c("spec-ngram-mod-n-max = 64")
	c("spec-ngram-mod-n-match = 24")
	c("tier1 = no")
	c("")

	// memory/capacity trade-offs, commented out: each section is a
	// (ctx, parallel) point that fits, with the prompt cache it leaves room for
	if mm.usable(primaryKV) {
		c("# ---- trade-offs (uncomment to benchmark; none of these change t/s per token) ----")
		levels := []int{ctxSize}
		for l := ctxSize / 2; l >= *minTradeoffCtx; l /= 2 {
			levels = append(levels, l)
		}
		any := false
		for _, n := range []int{2, 3, 4} {
			for _, l := range levels {
				ok, _ := mm.fits(primaryKV, l, n)
				if !ok {
					continue
				}
				cr := mm.cacheRamMiB(primaryKV, l, n)
				if !any {
					c("# each section below fits with the stated prompt cache headroom:")
					any = true
					c("")
				}
				c("# [parallel-%d-%s]", n, depthLabel(l))
				c("# inherit = baseline-nospec")
				c("# ctx-size = %d", l)
				c("# parallel = %d", n)
				c("# cache-ram = %d", cr)
				c("# spec-type = none")
				c("# depths = %s", joinInts(ladder(l, 4096)))
				c("#")
			}
		}
		if !any {
			c("# no (ctx, parallel) trade-off fits at or above %d ctx", *minTradeoffCtx)
		}
		c("")
	}

	c("# escape hatch: move heavy tensors around by hand, e.g.")
	c("# [ot-host-io]")
	c("# inherit = baseline-nospec")
	c("# override-tensor = output=CPU,token_embd=CPU")
	c("# spec-type = none")

	fmt.Fprint(os.Stdout, b.String())
	return nil
}
