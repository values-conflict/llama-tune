package main

import (
	"flag"
	"fmt"
	"os"
)

// cmdEmit prints the winning candidate of a run as a models.ini section in
// the router's config format, ready to paste.
func cmdEmit(args []string) error {
	fs := flag.NewFlagSet("emit", flag.ExitOnError)
	run := fs.String("run", "", "run directory (required)")
	depth := fs.Int("depth", 0, "context depth to rank by (default: deepest measured)")
	metric := fs.String("metric", "tg", "tg or pp")
	candName := fs.String("candidate", "", "force a candidate instead of picking the winner")
	section := fs.String("section", "", "section name (default: the model ref)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *run == "" {
		return fmt.Errorf("-run is required")
	}
	m, err := loadRun(*run)
	if err != nil {
		return err
	}
	cfg, err := parseConfigString(m.ConfigText)
	if err != nil {
		return err
	}

	type entry struct {
		cand  *Candidate
		flags map[string]string
		res   []DepthResult
	}
	var entries []entry
	var deepest int
	for i := range cfg.Candidates {
		cand := &cfg.Candidates[i]
		res := candRes(m, cand.Name)
		phase := res.T2
		if len(phase) == 0 {
			phase = res.T1
		}
		if len(phase) == 0 {
			continue
		}
		for _, d := range phase {
			if d.Depth > deepest {
				deepest = d.Depth
			}
		}
		flags := cfg.resolved(cand, 0)
		resolveAutoSpec(flags, m.ModelInfo != nil && m.ModelInfo.MTP)
		entries = append(entries, entry{cand, flags, phase})
	}
	if len(entries) == 0 {
		return fmt.Errorf("no candidates with results in %s", *run)
	}
	if *depth <= 0 {
		*depth = deepest
	}

	// best metric value at the deepest measured depth at or below -depth
	type scored struct {
		e   entry
		val float64
		at  int
		row *DepthResult
	}
	var best *scored
	for _, e := range entries {
		if *candName != "" && e.cand.Name != *candName {
			continue
		}
		var row *DepthResult
		for i := range e.res {
			if e.res[i].Depth <= *depth && (row == nil || e.res[i].Depth > row.Depth) {
				row = &e.res[i]
			}
		}
		if row == nil {
			continue
		}
		v := row.TG
		if *metric == "pp" {
			v = row.PP
		}
		if v <= 0 {
			continue
		}
		if best == nil || v > best.val {
			best = &scored{e, v, row.Depth, row}
		}
	}
	if best == nil {
		return fmt.Errorf("no candidate has %s results at or below depth %d", *metric, *depth)
	}

	name := *section
	if name == "" {
		name = m.Model
	}
	fmt.Fprintf(os.Stdout, "[%s]\n", name)
	seen := map[string]bool{}
	printKey := func(k, v string) {
		if seen[k] {
			return
		}
		seen[k] = true
		fmt.Fprintf(os.Stdout, "%s = %s\n", k, v)
	}
	for _, k := range cfg.DefaultKeys {
		printKey(k, best.e.flags[k])
	}
	// keys from the inherit chain (parents first), skipping defaults
	for _, k := range chainFlagKeys(cfg, best.e.cand) {
		if _, ok := cfg.Defaults[k]; !ok {
			printKey(k, best.e.flags[k])
		}
	}
	// rank by -metric, but show both axes: the pp/tg winner can differ
	fmt.Fprintf(os.Stderr, "\nwinner by %s: %s (tg %.2f, pp %.1f t/s at depth %d, from %s)\n",
		*metric, best.e.cand.Name, best.row.TG, best.row.PP, best.at, *run)
	return nil
}

func candRes(m *RunManifest, name string) *CandidateResult {
	return m.result(name)
}

// chainFlagKeys returns the candidate's flags (parents first) in order.
func chainFlagKeys(cfg *Config, cand *Candidate) []string {
	seen := map[string]bool{}
	var out []string
	var walk func(c *Candidate, d int)
	walk = func(c *Candidate, d int) {
		if d > 8 {
			return
		}
		if c.Inherit != "" {
			for i := range cfg.Candidates {
				if cfg.Candidates[i].Name == c.Inherit {
					walk(&cfg.Candidates[i], d+1)
					break
				}
			}
		}
		for _, k := range c.FlagKeys {
			if !seen[k] {
				seen[k] = true
				out = append(out, k)
			}
		}
	}
	walk(cand, 0)
	return out
}
