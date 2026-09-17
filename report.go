package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// DepthResult is one measurement at a given context depth.
type DepthResult struct {
	Depth    int     `json:"depth"`          // requested context depth
	PromptN  int     `json:"prompt_n"`       // actual prompt tokens processed
	PP       float64 `json:"pp_tps"`         // prompt processing t/s
	TG       float64 `json:"tg_tps"`         // generation t/s
	DraftN   int     `json:"draft_n"`        // speculative tokens drafted (0 = no spec)
	DraftAcc int     `json:"draft_accepted"` // speculative tokens accepted
}

func (d DepthResult) acceptRate() float64 {
	if d.DraftN == 0 {
		return -1
	}
	return float64(d.DraftAcc) / float64(d.DraftN)
}

type CandidateResult struct {
	Name   string        `json:"name"`
	T1     []DepthResult `json:"tier1,omitempty"`
	T2     []DepthResult `json:"tier2,omitempty"`
	T1Skip string        `json:"tier1_skip,omitempty"`
	T2Skip string        `json:"tier2_skip,omitempty"`
	Errors []string      `json:"errors,omitempty"`
}

type RunManifest struct {
	Tool       string            `json:"tool"`
	Time       string            `json:"time"`
	Hostname   string            `json:"hostname,omitempty"`
	Model      string            `json:"model"` // hf ref or path from config
	ModelPath  string            `json:"model_path,omitempty"`
	Build      string            `json:"build,omitempty"` // llama.cpp "build: N (commit)"
	GPUInfo    string            `json:"gpu_info,omitempty"`
	Backends   string            `json:"backends,omitempty"`
	ConfigFile string            `json:"config_file"`
	ConfigText string            `json:"config_text"`
	LlamaBin   string            `json:"llama_bin"`
	ModelInfo  *ggufModel        `json:"model_info,omitempty"`
	Candidates []CandidateResult `json:"candidates"`
}

func (m *RunManifest) result(name string) *CandidateResult {
	for i := range m.Candidates {
		if m.Candidates[i].Name == name {
			return &m.Candidates[i]
		}
	}
	return nil
}

func writeRun(dir string, m *RunManifest) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "run.json"), data, 0644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "results.tsv"), []byte(m.tsv()), 0644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "results.md"), []byte(m.markdown()), 0644)
}

func loadRun(dir string) (*RunManifest, error) {
	data, err := os.ReadFile(filepath.Join(dir, "run.json"))
	if err != nil {
		return nil, err
	}
	var m RunManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// replaceCandidate upserts cr into the manifest by name.
func replaceCandidate(m *RunManifest, cr *CandidateResult) {
	for i := range m.Candidates {
		if m.Candidates[i].Name == cr.Name {
			m.Candidates[i] = *cr
			return
		}
	}
	m.Candidates = append(m.Candidates, *cr)
}

// missingDepths returns the wanted depths that have no measured row yet.
// rowValid reports whether a measured row is a real result. With ignore_eos a
// successful measurement always yields pp>0 and tg>0; a zero in either is a
// failed depth (EOS bail, server hiccup) that must be re-measured, not kept.
func rowValid(d DepthResult) bool {
	return d.PP > 0 && d.TG > 0
}

func missingDepths(want []int, have []DepthResult) []int {
	seen := map[int]bool{}
	for _, d := range have {
		if rowValid(d) {
			seen[d.Depth] = true
		}
	}
	var out []int
	for _, d := range want {
		if !seen[d] {
			out = append(out, d)
		}
	}
	return out
}

// mergePhase appends fresh rows onto old ones; fresh wins per depth.
func mergePhase(old, fresh []DepthResult) []DepthResult {
	byDepth := map[int]DepthResult{}
	var order []int
	for _, d := range append(append([]DepthResult{}, old...), fresh...) {
		if _, ok := byDepth[d.Depth]; !ok {
			order = append(order, d.Depth)
		}
		byDepth[d.Depth] = d
	}
	sort.Ints(order)
	out := make([]DepthResult, 0, len(order))
	for _, d := range order {
		out = append(out, byDepth[d])
	}
	return out
}

// clearSatisfiedErrors drops a phase's old errors once every wanted depth
// has a valid row (a re-measured timeout should stop haunting notes).
func clearSatisfiedErrors(cr *CandidateResult, prefix string, want []int, have []DepthResult) {
	seen := map[int]bool{}
	for _, d := range have {
		if rowValid(d) {
			seen[d.Depth] = true
		}
	}
	for _, d := range want {
		if !seen[d] {
			return
		}
	}
	var out []string
	for _, e := range cr.Errors {
		if strings.HasPrefix(e, prefix+":") {
			continue
		}
		out = append(out, e)
	}
	cr.Errors = out
}

func dedupStrings(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// cmdReport re-renders results.md/tsv from a run's run.json (e.g. after an
// interrupted run) and prints the summary.
func cmdReport(args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	dir := fs.String("dir", "", "run directory (required)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dir == "" {
		return fmt.Errorf("-dir is required")
	}
	m, err := loadRun(*dir)
	if err != nil {
		return err
	}
	if err := writeRun(*dir, m); err != nil {
		return err
	}
	fmt.Println(m.markdown())
	return nil
}

func f3(v float64) string {
	return fmt.Sprintf("%.3f", v)
}

func depthLabel(d int) string {
	if d >= 1000 && d%1000 == 0 {
		return fmt.Sprintf("%dk", d/1000)
	}
	return fmt.Sprintf("%d", d)
}

func (m *RunManifest) tsv() string {
	var b strings.Builder
	b.WriteString("phase\tcandidate\tdepth\tprompt_n\tpp_tps\ttg_tps\tdraft_acc\n")
	for _, c := range m.Candidates {
		for _, phase := range []struct {
			name string
			rs   []DepthResult
		}{{"tier1", c.T1}, {"tier2", c.T2}} {
			for _, d := range phase.rs {
				acc := ""
				if d.acceptRate() >= 0 {
					acc = fmt.Sprintf("%.3f", d.acceptRate())
				}
				b.WriteString(fmt.Sprintf("%s\t%s\t%d\t%d\t%s\t%s\t%s\n",
					phase.name, c.Name, d.Depth, d.PromptN, f3(d.PP), f3(d.TG), acc))
			}
		}
	}
	return b.String()
}

func (m *RunManifest) markdown() string {
	var b strings.Builder
	b.WriteString("# llama-tune run\n\n")
	b.WriteString(fmt.Sprintf("- model: `%s`\n", m.Model))
	if m.ModelPath != "" {
		b.WriteString(fmt.Sprintf("- model path: `%s`\n", m.ModelPath))
	}
	if m.Build != "" {
		b.WriteString(fmt.Sprintf("- build: %s\n", m.Build))
	}
	if m.GPUInfo != "" {
		b.WriteString(fmt.Sprintf("- gpu: %s\n", m.GPUInfo))
	}
	if m.ModelInfo != nil {
		mi := m.ModelInfo
		b.WriteString(fmt.Sprintf("- arch: %s, %d layers, embd %d, trained ctx %d, ftype %s\n",
			mi.Arch, mi.BlockCount, mi.NEmbd, mi.NContext, mi.FileType))
		if mi.MTP {
			b.WriteString("- MTP: yes (draft-mtp capable)\n")
		} else {
			b.WriteString("- MTP: no\n")
		}
	}
	b.WriteString(fmt.Sprintf("- time: %s\n\n", m.Time))

	depths := []int{}
	seen := map[int]bool{}
	for _, c := range m.Candidates {
		for _, d := range append(append([]DepthResult{}, c.T1...), c.T2...) {
			if !seen[d.Depth] {
				seen[d.Depth] = true
				depths = append(depths, d.Depth)
			}
		}
	}
	sort.Ints(depths)

	for _, ph := range []struct {
		name string
		pick func(*CandidateResult) []DepthResult
	}{
		{"tier1 (llama-bench)", func(c *CandidateResult) []DepthResult { return c.T1 }},
		{"tier2 (live server)", func(c *CandidateResult) []DepthResult { return c.T2 }},
	} {
		b.WriteString(fmt.Sprintf("## %s\n\n", ph.name))
		var head []string
		head = append(head, "candidate")
		for _, d := range depths {
			head = append(head, depthLabel(d)+" tg")
		}
		b.WriteString("| " + strings.Join(head, " | ") + " |\n")
		var sep []string
		for range head {
			sep = append(sep, "---")
		}
		b.WriteString("|" + strings.Join(sep, "|") + "|\n")
		for _, c := range m.Candidates {
			var row []string
			row = append(row, c.Name)
			byDepth := map[int]DepthResult{}
			for _, d := range ph.pick(&c) {
				byDepth[d.Depth] = d
			}
			for _, d := range depths {
				if r, ok := byDepth[d]; !ok {
					row = append(row, "-")
				} else {
					cell := fmt.Sprintf("%.1f (%.0f)", r.TG, r.PP)
					if r.acceptRate() >= 0 {
						cell += fmt.Sprintf(", acc %.2f", r.acceptRate())
					}
					row = append(row, cell)
				}
			}
			b.WriteString("| " + strings.Join(row, " | ") + " |\n")
		}
		b.WriteString("\n")
		// skips and errors
		var notes []string
		for _, c := range m.Candidates {
			if c.T1Skip != "" && ph.name == "tier1 (llama-bench)" {
				notes = append(notes, fmt.Sprintf("- %s: %s", c.Name, c.T1Skip))
			}
			if c.T2Skip != "" && ph.name == "tier2 (live server)" {
				notes = append(notes, fmt.Sprintf("- %s: %s", c.Name, c.T2Skip))
			}
			// each error appears only in the section of the phase that made it
			for _, e := range c.Errors {
				switch {
				case strings.HasPrefix(e, "tier1:") && ph.name == "tier1 (llama-bench)":
					notes = append(notes, fmt.Sprintf("- %s: %s", c.Name, e))
				case strings.HasPrefix(e, "tier2:") && ph.name == "tier2 (live server)":
					notes = append(notes, fmt.Sprintf("- %s: %s", c.Name, e))
				case !strings.Contains(e, "tier") && ph.name == "tier2 (live server)":
					notes = append(notes, fmt.Sprintf("- %s: %s", c.Name, e))
				}
			}
		}
		if len(notes) > 0 {
			b.WriteString("notes:\n")
			b.WriteString(strings.Join(notes, "\n") + "\n\n")
		}
	}
	b.WriteString("(each cell: tg t/s (pp t/s), acc = draft acceptance rate)\n")
	return b.String()
}
