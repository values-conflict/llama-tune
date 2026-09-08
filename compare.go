package main

import (
	"fmt"
	"sort"
)

// cmdCompare aligns the results of two runs (e.g. the same sweep compiled
// with -march=x86-64-v3 vs x86-64-v4) and prints the per-cell deltas.
func cmdCompare(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: llama-tune compare <runA> <runB>")
	}
	a, err := loadRun(args[0])
	if err != nil {
		return fmt.Errorf("run A: %w", err)
	}
	b, err := loadRun(args[1])
	if err != nil {
		return fmt.Errorf("run B: %w", err)
	}

	nameOf := func(m *RunManifest) string {
		if m.Build != "" {
			return m.Model + " " + m.Build
		}
		return m.Model
	}
	fmt.Printf("A: %s (%s)\nB: %s (%s)\n\n", nameOf(a), a.Time, nameOf(b), b.Time)
	if a.Model != b.Model {
		fmt.Println("warning: runs use different models")
	}

	for _, ph := range []struct {
		name string
		get  func(*CandidateResult) []DepthResult
	}{
		{"tier1", func(c *CandidateResult) []DepthResult { return c.T1 }},
		{"tier2", func(c *CandidateResult) []DepthResult { return c.T2 }},
	} {
		var names []string
		seen := map[string]bool{}
		for _, c := range a.Candidates {
			if len(ph.get(&c)) > 0 && !seen[c.Name] {
				seen[c.Name] = true
				names = append(names, c.Name)
			}
		}
		if len(names) == 0 {
			continue
		}
		sort.Strings(names)

		fmt.Printf("## %s\n\n", ph.name)
		fmt.Println("| candidate | depth | A tg | B tg | delta t/s | % | A pp | B pp |")
		fmt.Println("|---|---|---|---|---|---|---|---|")
		for _, n := range names {
			ca, cb := a.result(n), b.result(n)
			if ca == nil || cb == nil {
				continue
			}
			am := depthMap(ph.get(ca))
			bm := depthMap(ph.get(cb))
			var depths []int
			for d := range am {
				if _, ok := bm[d]; ok {
					depths = append(depths, d)
				}
			}
			sort.Ints(depths)
			for _, d := range depths {
				da, db := am[d], bm[d]
				fmt.Printf("| %s | %d | %s | %s | %+.3f | %+.1f%% | %s | %s |\n",
					n, d, f3(da.TG), f3(db.TG), db.TG-da.TG, pct(da.TG, db.TG), f3(da.PP), f3(db.PP))
			}
		}
		fmt.Println()
	}
	return nil
}

func depthMap(rs []DepthResult) map[int]DepthResult {
	m := map[int]DepthResult{}
	for _, r := range rs {
		m[r.Depth] = r
	}
	return m
}

func pct(a, b float64) float64 {
	if a == 0 {
		return 0
	}
	return (b - a) / a * 100
}
