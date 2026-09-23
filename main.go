package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

const toolVersion = "0.1.0"

func usage() {
	fmt.Fprintf(os.Stderr, `llama-tune %s - benchmark llama.cpp serving configs

usage:
  llama-tune scan <model> [model...]      scan gguf files (or hf refs) for model intel
  llama-tune plan [flags]                 generate a sweep config from model + hardware
  llama-tune run [flags]                   run the sweep from a config file
  llama-tune report -dir <run>             re-render/summarize a run's results
  llama-tune emit [flags]                 print the run's winner as a models.ini section
  llama-tune compare <runA> <runB>         compare two run directories

run flags:
  -config      config file (required)
  -dir         output directory for results (default: run-<timestamp>)
  -llama-bin   path to the unified llama binary (default: llama)
  -only        comma-separated candidate names to run (default: all)
  -phases      comma-separated list of t1,t2 (default: both)
  -force       re-measure everything, even depths already in -dir
  -t1-timeout  wall-clock bound for each 'llama bench' process (default: 8h,
               0 = no bound); a hung GPU driver otherwise blocks the sweep
               indefinitely. Progress markers are streamed to the tier1 log
               (-progress), so a log that stops growing is a hang.

re-runs into the same -dir resume: already-measured (candidate, phase,
depth) rows are kept and only the missing ones are measured. Results are
written after every candidate, so an interrupted run keeps what it had.

llama-tune drives the unified 'llama' binary: 'llama bench' for tier 1,
'llama serve' for tier 2, 'llama download' to get models on disk.
`, toolVersion)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "scan":
		err = cmdScan(os.Args[2:])
	case "plan":
		err = cmdPlan(os.Args[2:])
	case "run":
		err = cmdRun(os.Args[2:])
	case "report":
		err = cmdReport(os.Args[2:])
	case "emit":
		err = cmdEmit(os.Args[2:])
	case "compare":
		err = cmdCompare(os.Args[2:])
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

// resolveModel downloads (if needed) via 'llama download' and returns the
// local gguf path. Local paths are returned as-is.
func resolveModel(llamaBin, ref string) (string, error) {
	if !isHFRef(ref) {
		return ref, nil
	}
	cmd := exec.Command(llamaBin, "download", "-hf", ref)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%s download %s: %w\n%s", llamaBin, ref, err, string(out))
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasSuffix(line, ".gguf") {
			return line, nil
		}
	}
	return "", fmt.Errorf("could not find gguf path in output of %q:\n%s", llamaBin+" download", string(out))
}

func cmdScan(args []string) error {
	fs := flag.NewFlagSet("scan", flag.ExitOnError)
	llamaBin := fs.String("llama-bin", "llama", "path to the llama binary (for downloads)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() == 0 {
		return fmt.Errorf("usage: llama-tune scan <model> [model...]")
	}
	for _, ref := range fs.Args() {
		path, err := resolveModel(*llamaBin, ref)
		if err != nil {
			fmt.Printf("%s: %v\n", ref, err)
			continue
		}
		m, err := scanGGUF(path)
		if err != nil {
			fmt.Printf("%s: %v\n", ref, err)
			continue
		}
		printModel(m)
	}
	return nil
}

func printModel(m *ggufModel) {
	fmt.Printf("model:       %s\n", m.Path)
	fmt.Printf("arch:        %s\n", m.Arch)
	fmt.Printf("layers:      %d\n", m.BlockCount)
	fmt.Printf("embd:        %d\n", m.NEmbd)
	if m.NHeads > 0 {
		kv := m.NKVHeads
		if kv <= 0 {
			kv = m.NHeads
		}
		fmt.Printf("attention:   %d heads, %d kv heads (gqa 1:%d)\n", m.NHeads, kv, m.NHeads/kv)
	}
	fmt.Printf("trained ctx: %d\n", m.NContext)
	if m.SlidingWin > 0 {
		fmt.Printf("sliding win: %d\n", m.SlidingWin)
	}
	if m.NExperts > 0 {
		fmt.Printf("moe:         %d experts, %d active\n", m.NExperts, m.NExpertSel)
	}
	if m.FileType != "" {
		fmt.Printf("file type:   %s\n", m.FileType)
	}
	if m.MTP {
		fmt.Printf("MTP:         yes (spec-type draft-mtp supported)\n")
	} else if m.HasNextN {
		fmt.Printf("MTP:         no (nextn tensors present but no complete module)\n")
	} else {
		fmt.Printf("MTP:         no\n")
	}
	// kv cache footprint for common ctx sizes and cache types
	fmt.Printf("kv cache:    ")
	for i, ctx := range []int{32768, 131072, 262144} {
		if i > 0 {
			fmt.Print(", ")
		}
		miB := m.kvMiBFor(ctx, "q4_0", "q4_0")
		if miB < 0 {
			fmt.Printf("%dk: n/a", ctx/1000)
		} else {
			fmt.Printf("%dk: %.0f MiB (q4_0 kv)", ctx/1000, miB)
		}
	}
	fmt.Println()
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	configPath := fs.String("config", "", "config file (required)")
	dir := fs.String("dir", "", "output directory (default: run-<timestamp>)")
	llamaBin := fs.String("llama-bin", "llama", "path to the unified llama binary")
	only := fs.String("only", "", "comma-separated candidate names (default: all)")
	skip := fs.String("skip", "", "comma-separated candidate names to exclude (inverse of -only)")
	t1Timeout := fs.Duration("t1-timeout", 8*time.Hour, "wall-clock bound for each llama bench process (0 = no bound)")
	phases := fs.String("phases", "t1,t2", "phases to run: t1 (llama-bench), t2 (live server)")
	force := fs.Bool("force", false, "re-measure everything, even depths already in -dir; also retries candidates skipped after a load failure")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *configPath == "" {
		return fmt.Errorf("-config is required")
	}
	cfg, err := parseConfig(*configPath)
	if err != nil {
		return err
	}

	doT1 := strings.Contains(*phases, "t1")
	doT2 := strings.Contains(*phases, "t2")
	if doT2 && !doT1 && cfg.T1.Enabled {
		// fine: tier2-only runs are useful for spec candidates
	}

	modelPath, err := resolveModel(*llamaBin, cfg.Model)
	if err != nil {
		return err
	}
	modelInfo, err := scanGGUF(modelPath)
	if err != nil {
		fmt.Printf("warning: gguf scan failed: %v\n", err)
		modelInfo = nil
	}

	if *dir == "" {
		*dir = fmt.Sprintf("run-%s", time.Now().Format("20060102-150405"))
	}
	if err := os.MkdirAll(*dir, 0755); err != nil {
		return err
	}
	host, _ := os.Hostname()
	manifest := &RunManifest{
		Tool:       "llama-tune " + toolVersion,
		Time:       time.Now().Format(time.RFC3339),
		Hostname:   host,
		Model:      cfg.Model,
		ModelPath:  modelPath,
		ConfigFile: *configPath,
		ConfigText: cfg.ConfigText,
		LlamaBin:   *llamaBin,
		ModelInfo:  modelInfo,
	}

	var onlySet, skipSet map[string]bool
	if *only != "" {
		onlySet = map[string]bool{}
		for _, n := range strings.Split(*only, ",") {
			onlySet[strings.TrimSpace(n)] = true
		}
	}
	if *skip != "" {
		skipSet = map[string]bool{}
		for _, n := range strings.Split(*skip, ",") {
			skipSet[strings.TrimSpace(n)] = true
		}
	}

	ctx := context.Background()
	benchMeta := &benchMeta{}

	// resume: a previous run in the same dir keeps its measured rows; only
	// missing (candidate, phase, depth) combinations are measured again
	var prev *RunManifest
	if !*force {
		if p, err := loadRun(*dir); err == nil {
			if p.Model != cfg.Model {
				return fmt.Errorf("%s already holds results for model %s; use -force or a fresh -dir", *dir, p.Model)
			}
			prev = p
			manifest.Candidates = p.Candidates
			if manifest.Build == "" {
				manifest.Build = p.Build
			}
			if manifest.GPUInfo == "" {
				manifest.GPUInfo, manifest.Backends = p.GPUInfo, p.Backends
			}
			if manifest.ModelInfo == nil {
				manifest.ModelInfo = p.ModelInfo
			}
			if manifest.Hostname == "" {
				manifest.Hostname = p.Hostname
			}
			fmt.Printf("resuming: %d candidate(s) already in %s\n", len(p.Candidates), *dir)
		}
	}

	for i := range cfg.Candidates {
		cand := &cfg.Candidates[i]
		if onlySet != nil && !onlySet[cand.Name] {
			continue
		}
		if skipSet != nil && skipSet[cand.Name] {
			fmt.Printf("== candidate %s: excluded by -skip\n", cand.Name)
			continue
		}
		cr := &CandidateResult{Name: cand.Name}
		if prev != nil {
			if old := prev.result(cand.Name); old != nil {
				*cr = *old
				cr.Errors = append([]string{}, old.Errors...)
				// migrate load failures recorded before T2Skip existed so they
				// stop being retried: no rows plus a load error is a dead server
				if cr.T2Skip == "" && len(cr.T2) == 0 {
					for _, e := range cr.Errors {
						if strings.Contains(e, "server did not become healthy") ||
							strings.Contains(e, "server failed to load") {
							cr.T2Skip = "server failed to load"
							break
						}
					}
				}
			}
		}
		replaceCandidate(manifest, cr)

		fmt.Printf("== candidate %s\n", cand.Name)

		flags := cfg.resolved(cand, 0)
		if flags == nil {
			cr.Errors = append(cr.Errors, "broken inherit chain")
			replaceCandidate(manifest, cr)
			continue
		}
		resolveAutoSpec(flags, modelInfo != nil && modelInfo.MTP)

		// fail fast on combinations llama.cpp refuses to load, so the
		// candidate does not burn twenty minutes reaching the same wall
		if reason := infeasible(flags); reason != "" {
			fmt.Printf("  skipped: %s\n", reason)
			cr.Errors = append(cr.Errors, "skipped: "+reason)
			replaceCandidate(manifest, cr)
			continue
		}

		if doT1 && !cand.NoTier1 && cfg.T1.Enabled {
			if cand.hasSpec(flags) {
				cr.T1Skip = "spec decoding not supported by llama-bench"
				cr.T1 = nil
			} else if missing := missingDepths(cfg.T1.Depths, cr.T1); len(missing) == 0 {
				fmt.Printf("  tier1: %d depth(s) already measured, skipping (resume)\n", len(cfg.T1.Depths))
			} else {
				fmt.Printf("  tier1: running llama-bench (%d new depth(s): %s)\n", len(missing), joinInts(missing))
				t1, meta, err := runTier1(cfg, cand, *llamaBin, *dir, missing, *t1Timeout)
				if meta != nil && benchMeta.BuildCommit == "" {
					*benchMeta = *meta
				}
				cr.T1 = mergePhase(cr.T1, t1)
				for _, d := range t1 {
					fmt.Printf("  [%s] tier1 depth %6d: pp %7.1f t/s, tg %7.2f t/s\n", cand.Name, d.Depth, d.PP, d.TG)
				}
				if err != nil {
					cr.Errors = append(cr.Errors, "tier1: "+err.Error())
				}
			}
		}
		if doT2 && !cand.NoTier2 {
			wl := cfg.T2
			if len(cand.DepthOverride) > 0 {
				wl.Depths = cand.DepthOverride
			}
			if cr.T2Skip != "" {
				// a prior run recorded a load failure for this candidate; re-running
				// reaches the same wall. -force clears this by not loading prev.
				fmt.Printf("  tier2: skipped (prior load failure, use -force to retry)\n")
			} else if missing := missingDepths(wl.Depths, cr.T2); len(missing) == 0 {
				fmt.Printf("  tier2: %d depth(s) already measured, skipping (resume)\n", len(wl.Depths))
			} else {
				fmt.Printf("  tier2: running server (%d new depth(s): %s)\n", len(missing), joinInts(missing))
				wl.Depths = missing
				t2, err := runTier2(ctx, cand, flags, wl, modelPath, *llamaBin, *dir)
				cr.T2 = mergePhase(cr.T2, t2)
				if err != nil {
					cr.Errors = append(cr.Errors, "tier2: "+err.Error())
					if errors.Is(err, errLoadFailed) {
						// deterministic for this config+box: remember it so resume
						// does not burn minutes re-reaching the same load failure
						cr.T2Skip = "server failed to load"
					}
				}
			}
			if manifest.Build == "" {
				manifest.Build = extractBuild(serverLogPath(*dir, cand.Name))
			}
		}
		clearSatisfiedErrors(cr, "tier1", cfg.T1.Depths, cr.T1)
		clearSatisfiedErrors(cr, "tier2", cfg.T2.Depths, cr.T2)
		cr.Errors = dedupStrings(cr.Errors)
		replaceCandidate(manifest, cr)
		// incremental: an interrupted run keeps everything measured so far
		if err := writeRun(*dir, manifest); err != nil {
			return err
		}
	}

	if benchMeta.BuildCommit != "" {
		manifest.Build = fmt.Sprintf("%s (%s)", benchMeta.BuildNumber, benchMeta.BuildCommit)
	}
	if benchMeta.GPUInfo != "" {
		manifest.GPUInfo = benchMeta.GPUInfo
		manifest.Backends = benchMeta.Backends
	}

	if err := writeRun(*dir, manifest); err != nil {
		return err
	}
	fmt.Printf("\nresults written to %s (run.json, results.tsv, results.md)\n\n", *dir)
	fmt.Println()
	fmt.Println(manifest.markdown())
	return nil
}

// infeasible returns a reason when the flag set is one llama.cpp refuses
// to load, or "" when it may work.
func infeasible(flags map[string]string) string {
	fa := strings.ToLower(flags["flash-attn"])
	if fa != "false" && fa != "off" && fa != "no" && fa != "0" {
		return ""
	}
	quantKV := func(t string) bool {
		t = strings.ToLower(t)
		return t != "" && t != "f16" && t != "bf16"
	}
	if quantKV(flags["cache-type-k"]) || quantKV(flags["cache-type-v"]) {
		return "quantized kv cache requires flash-attn (llama refuses to load this)"
	}
	return ""
}

// resolveAutoSpec expands spec-type = auto using the model's MTP capability.
func resolveAutoSpec(flags map[string]string, mtp bool) {
	if flags["spec-type"] != "auto" {
		return
	}
	if mtp {
		flags["spec-type"] = "draft-mtp"
		if _, ok := flags["spec-draft-n-max"]; !ok {
			flags["spec-draft-n-max"] = "3"
		}
	} else {
		// same values as the built-in --spec-default
		flags["spec-type"] = "ngram-mod"
		flags["spec-ngram-mod-n-min"] = "48"
		flags["spec-ngram-mod-n-max"] = "64"
		flags["spec-ngram-mod-n-match"] = "24"
	}
}

func serverLogPath(dir, cand string) string {
	return dir + "/server-" + cand + ".log"
}
