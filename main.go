package main

import (
	"context"
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
  llama-tune run [flags]                   run the sweep from a config file
  llama-tune compare <runA> <runB>         compare two run directories

run flags:
  -config      config file (required)
  -dir         output directory for results (default: run-<timestamp>)
  -bench-bin   path to llama-bench (default: llama-bench)
  -server-bin  path to llama-server (default: llama-server)
  -llama-bin   path to llama (for 'llama download', default: llama)
  -only        comma-separated candidate names to run (default: all)
  -phases      comma-separated list of t1,t2 (default: both)
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
	case "run":
		err = cmdRun(os.Args[2:])
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
	benchBin := fs.String("bench-bin", "llama-bench", "path to llama-bench")
	serverBin := fs.String("server-bin", "llama-server", "path to llama-server")
	llamaBin := fs.String("llama-bin", "llama", "path to llama (for model downloads)")
	only := fs.String("only", "", "comma-separated candidate names (default: all)")
	phases := fs.String("phases", "t1,t2", "phases to run: t1 (llama-bench), t2 (live server)")
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
		BenchBin:   *benchBin,
		ServerBin:  *serverBin,
		ModelInfo:  modelInfo,
	}

	var onlySet map[string]bool
	if *only != "" {
		onlySet = map[string]bool{}
		for _, n := range strings.Split(*only, ",") {
			onlySet[strings.TrimSpace(n)] = true
		}
	}

	ctx := context.Background()
	benchMeta := &benchMeta{}
	for i := range cfg.Candidates {
		cand := &cfg.Candidates[i]
		if onlySet != nil && !onlySet[cand.Name] {
			continue
		}
		cr := &CandidateResult{Name: cand.Name}
		manifest.Candidates = append(manifest.Candidates, *cr)

		fmt.Printf("== candidate %s\n", cand.Name)

		flags := cfg.resolved(cand, 0)
		if flags == nil {
			cr.Errors = append(cr.Errors, "broken inherit chain")
			manifest.Candidates[len(manifest.Candidates)-1] = *cr
			continue
		}
		resolveAutoSpec(flags, modelInfo != nil && modelInfo.MTP)

		if doT1 && !cand.NoTier1 && cfg.T1.Enabled {
			if cand.hasSpec(flags) {
				cr.T1Skip = "spec decoding not supported by llama-bench"
			} else {
				fmt.Printf("  tier1: running llama-bench\n")
				t1, meta, err := runTier1(cfg, cand, *benchBin, *dir)
				if meta != nil && benchMeta.BuildCommit == "" {
					*benchMeta = *meta
				}
				if err != nil {
					cr.Errors = append(cr.Errors, "tier1: "+err.Error())
				} else {
					cr.T1 = t1
					for _, d := range t1 {
						fmt.Printf("  [%s] tier1 depth %6d: pp %7.1f t/s, tg %7.2f t/s\n", cand.Name, d.Depth, d.PP, d.TG)
					}
				}
			}
		}
		if doT2 && !cand.NoTier2 {
			wl := cfg.T2
			if len(cand.DepthOverride) > 0 {
				wl.Depths = cand.DepthOverride
			}
			cr.T2, err = runTier2(ctx, cand, flags, wl, modelPath, *serverBin, *dir)
			if err != nil {
				cr.Errors = append(cr.Errors, "tier2: "+err.Error())
			}
			if manifest.Build == "" {
				manifest.Build = extractBuild(serverLogPath(*dir, cand.Name))
			}
		}
		manifest.Candidates[len(manifest.Candidates)-1] = *cr
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
	fmt.Printf("\nresults written to %s (run.json, results.tsv, results.md)\n", *dir)
	fmt.Println()
	fmt.Println(manifest.markdown())
	return nil
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
