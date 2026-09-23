package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// benchMeta is filled from any llama-bench json row (identical for all rows).
type benchMeta struct {
	BuildCommit string `json:"build_commit"`
	BuildNumber string `json:"build_number"`
	GPUInfo     string `json:"gpu_info"`
	Backends    string `json:"backends"`
}

type benchRow struct {
	NPrompt int     `json:"n_prompt"`
	NGen    int     `json:"n_gen"`
	NDepth  int     `json:"n_depth"`
	AvgTS   float64 `json:"avg_ts"`
	StdTS   float64 `json:"stddev_ts"`
}

// benchBenign are flags that llama-bench does not accept but that cannot
// affect its single-sequence, no-server measurement - reporting them as
// "unmapped" would be noise on every candidate.
var benchBenign = map[string]bool{
	"kv-unified": true, // layout only matters with multiple sequences
	"jinja":      true,
	"parallel":   true,
	"cache-ram":  true,
	"ctx-size":   true, // bench contexts come from -d depths
	"warmup":     true,
}

// benchArgs maps a candidate's resolved flags to llama-bench CLI args.
// Returns the args and the list of flags that could not be mapped.
func benchArgs(flags map[string]string, t1 Tier1) ([]string, []string) {
	var args []string
	var unmapped []string
	add := func(short, val string) { args = append(args, short, val) }
	specOff := flags["spec-type"] == "" || flags["spec-type"] == "none"

	ngl := func(v string) string {
		if v == "all" || v == "auto" || v == "-1" {
			return "9999"
		}
		return v
	}

	for k, v := range flags {
		switch k {
		case "hf", "model":
			// handled by caller
		case "pin-cpus":
			// consumed by the harness (process affinity)
		case "n-gpu-layers":
			add("-ngl", ngl(v))
		case "threads":
			add("-t", v)
		case "batch-size":
			add("-b", v)
		case "ubatch-size":
			add("-ub", v)
		case "cache-type-k":
			add("-ctk", v)
		case "cache-type-v":
			add("-ctv", v)
		case "split-mode":
			add("-sm", v)
		case "n-cpu-moe":
			add("-ncmoe", v)
		case "cpu-mask":
			add("-C", v)
		case "flash-attn":
			switch strings.ToLower(v) {
			case "true", "on", "yes", "1":
				add("-fa", "on")
			case "false", "off", "no", "0":
				add("-fa", "off")
			default:
				add("-fa", v)
			}
		case "kv-unified":
			// server-only flag; llama-bench does not accept -kvu
		default:
			if benchBenign[k] {
				continue
			}
			// spec flags are inert when spec-type is none (leftover inheritance)
			if strings.HasPrefix(k, "spec-") && specOff {
				continue
			}
			unmapped = append(unmapped, k)
		}
	}

	// tests
	add("-p", fmt.Sprintf("%d", t1.PP))
	add("-n", fmt.Sprintf("%d", t1.Gen))
	if len(t1.Depths) > 0 {
		var ds []string
		for _, d := range t1.Depths {
			ds = append(ds, fmt.Sprintf("%d", d))
		}
		args = append(args, "-d", strings.Join(ds, ","))
	}
	args = append(args, "-r", fmt.Sprintf("%d", t1.Repeats))
	args = append(args, "-o", "json")
	// progress markers go to stderr (the tier1 log): a log that stops
	// growing is a hung bench, even with no timeout configured
	args = append(args, "--progress")
	return args, unmapped
}

// runTier1 executes 'llama bench' for one candidate and returns per-depth
// results. depths overrides cfg.T1.Depths (resume: measure only what is
// missing). timeout bounds the whole bench process (a hung GPU driver
// otherwise blocks the sweep indefinitely); 0 disables the bound.
func runTier1(cfg *Config, cand *Candidate, llamaBin string, runDir string, depths []int, timeout time.Duration) ([]DepthResult, *benchMeta, error) {
	resolved := cfg.resolved(cand, 0)
	if resolved == nil {
		return nil, nil, fmt.Errorf("candidate %s: broken inherit chain", cand.Name)
	}
	if cand.hasSpec(resolved) {
		return nil, nil, fmt.Errorf("skip: spec-type %s, llama-bench does not support speculative decoding (tier2 only)", resolved["spec-type"])
	}

	t1 := cfg.T1
	if len(depths) > 0 {
		t1.Depths = depths
	}
	flagArgs, unmapped := benchArgs(resolved, t1)
	args := append([]string{"bench"}, flagArgs...)
	if len(unmapped) > 0 {
		fmt.Printf("  tier1: note: flags not mappable to llama-bench (ignored): %s\n", strings.Join(unmapped, ", "))
	}
	if isHFRef(cfg.Model) {
		args = append(args, "-hf", cfg.Model)
	} else {
		args = append(args, "-m", cfg.Model)
	}

	logPath := filepath.Join(runDir, fmt.Sprintf("tier1-%s.log", cand.Name))
	logF, err := os.Create(logPath)
	if err != nil {
		return nil, nil, err
	}
	defer logF.Close()

	var cmd *exec.Cmd
	var ctx context.Context
	var cancel context.CancelFunc
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(context.Background(), timeout)
		cmd = exec.CommandContext(ctx, llamaBin, args...)
	} else {
		cmd = exec.Command(llamaBin, args...)
	}
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = logF
	logF2, _ := os.Create(logPath + ".cmd")
	if logF2 != nil {
		fmt.Fprintf(logF2, "$ %s %s\n", llamaBin, strings.Join(args, " "))
		logF2.Close()
	}
	err = withAffinity(pinCPUs(resolved), cmd.Run)
	if cancel != nil {
		cancel()
	}
	if err != nil {
		if ctx != nil && ctx.Err() == context.DeadlineExceeded {
			return nil, nil, fmt.Errorf("llama bench timed out after %s (hung?), see %s", timeout, logPath)
		}
		return nil, nil, fmt.Errorf("llama bench failed (%v), see %s", err, logPath)
	}

	var rows []map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &rows); err != nil {
		return nil, nil, fmt.Errorf("parsing llama-bench json: %w", err)
	}
	if len(rows) == 0 {
		return nil, nil, fmt.Errorf("llama-bench produced no rows")
	}

	meta := &benchMeta{
		BuildCommit: jsonStr(rows[0]["build_commit"]),
		BuildNumber: jsonStr(rows[0]["build_number"]),
		GPUInfo:     jsonStr(rows[0]["gpu_info"]),
		Backends:    jsonStr(rows[0]["backends"]),
	}
	if meta.BuildNumber == "0" {
		meta.BuildNumber = ""
	}

	// one entry per context depth: pp and tg rows at the same depth merge
	type agg struct {
		pp, tg     float64
		promptN    int
		havePP, TG bool
	}
	results := map[int]*agg{}
	for _, raw := range rows {
		br := benchRow{
			NPrompt: int(num(raw["n_prompt"])),
			NGen:    int(num(raw["n_gen"])),
			NDepth:  int(num(raw["n_depth"])),
			AvgTS:   num(raw["avg_ts"]),
		}
		if br.NGen == 0 && br.NPrompt == 0 {
			continue
		}
		a := results[br.NDepth]
		if a == nil {
			a = &agg{}
			results[br.NDepth] = a
		}
		switch {
		case br.NGen > 0 && br.NPrompt > 0:
			// combined pg test: record both
			a.tg, a.TG = br.AvgTS, true
			a.pp, a.havePP = br.AvgTS, true
			a.promptN = br.NPrompt
		case br.NGen > 0:
			a.tg, a.TG = br.AvgTS, true
		case br.NPrompt > 0:
			a.pp, a.havePP = br.AvgTS, true
			a.promptN = br.NPrompt
		}
	}

	var out []DepthResult
	for d, a := range results {
		r := DepthResult{Depth: d, PromptN: a.promptN}
		if a.TG {
			r.TG = a.tg
		}
		if a.havePP {
			r.PP = a.pp
		}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Depth < out[j].Depth })
	if len(out) == 0 {
		return nil, meta, fmt.Errorf("no usable rows in llama-bench output")
	}
	return out, meta, nil
}

// jsonStr renders a json value (string or number) as a string.
func jsonStr(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	}
	return ""
}

func num(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	}
	return 0
}
