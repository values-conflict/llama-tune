package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Workload controls the tier-2 (live server) benchmark requests.
type Workload struct {
	Depths  []int // prompt sizes to benchmark (tokens)
	Gen     int   // tokens to generate per measurement
	Repeats int   // independent prompt variants per depth
	Warmup  bool  // throwaway request before measuring
}

// Tier1 controls the llama-bench (no server) phase.
type Tier1 struct {
	Enabled bool
	Depths  []int
	PP      int // prompt length for pp tests
	Gen     int // tokens for tg tests
	Repeats int
}

type Candidate struct {
	Name          string
	Inherit       string
	NoTier1       bool
	NoTier2       bool
	DepthOverride []int             // optional per-candidate tier2 depths
	FlagKeys      []string          // insertion order, for stable command lines
	Flags         map[string]string // raw key -> value
}

type Config struct {
	Model       string // hf ref or local path
	DefaultKeys []string
	Defaults    map[string]string
	T1          Tier1
	T2          Workload
	Candidates  []Candidate
	ConfigText  string // raw file content, embedded in the run manifest
}

// resolved returns the full flag set for a candidate: defaults, then the
// inherited candidate's flags, then its own.
func (cfg *Config) resolved(c *Candidate, depth int) map[string]string {
	if depth > 8 {
		return nil
	}
	out := make(map[string]string, len(cfg.Defaults)+len(c.Flags))
	for k, v := range cfg.Defaults {
		out[k] = v
	}
	if c.Inherit != "" {
		for _, p := range cfg.Candidates {
			if p.Name == c.Inherit {
				prev := cfg.resolved(&p, depth+1)
				if prev == nil {
					return nil
				}
				for k, v := range prev {
					out[k] = v
				}
				break
			}
		}
	}
	for k, v := range c.Flags {
		out[k] = v
	}
	return out
}

func (c *Candidate) hasSpec(flags map[string]string) bool {
	t, ok := flags["spec-type"]
	return ok && t != "" && t != "none"
}

func parseBool(s string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "true", "on", "yes", "1":
		return true, nil
	case "false", "off", "no", "0":
		return false, nil
	}
	return false, fmt.Errorf("not a bool: %q", s)
}

func parseIntList(s string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("bad int %q in %q", part, s)
		}
		out = append(out, n)
	}
	return out, nil
}

// known server flags that take no value (pair with --no-<name> for false)
var boolFlags = map[string]bool{
	"jinja":             true,
	"kv-unified":        true,
	"warmup":            true,
	"cont-batching":     true,
	"cache-idle-slots":  true,
	"context-shift":     true,
	"cache-prompt":      true,
	"prefill-assistant": true,
	"metrics":           true,
	"log-disable":       true,
	"mmproj-auto":       true,
	"models-autoload":   true,
}

func parseConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseConfigString(string(raw))
}

func parseConfigString(raw string) (*Config, error) {
	cfg := &Config{
		Defaults:    map[string]string{},
		DefaultKeys: []string{},
		ConfigText:  string(raw),
	}
	cfg.T1 = Tier1{Enabled: true, PP: 4096, Gen: 64, Repeats: 2}
	cfg.T2 = Workload{Gen: 64, Repeats: 1, Warmup: true}

	var cur *Candidate
	var candPtrs []*Candidate
	var err error
	section := "" // "", "workload", "tier1", or a candidate name
	for lineno, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if i := indexInlineComment(line); i >= 0 {
			line = strings.TrimSpace(line[:i])
		}
		if line == "" || line[0] == '#' || line[0] == ';' {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(line[1 : len(line)-1])
			if section == "workload" || section == "tier2" || section == "tier1" {
				cur = nil
			} else if section != "" {
				cur = &Candidate{Name: section, Flags: map[string]string{}}
				candPtrs = append(candPtrs, cur)
			} else {
				cur = nil
			}
			continue
		}
		eq := strings.Index(line, "=")
		if eq < 0 {
			return nil, fmt.Errorf("line %d: expected key = value: %s", lineno+1, line)
		}
		key := strings.TrimSpace(line[:eq])
		val := strings.TrimSpace(line[eq+1:])

		switch {
		case section == "":
			if key == "hf" || key == "model" {
				cfg.Model = val
			} else {
				cfg.Defaults[key] = val
				cfg.DefaultKeys = append(cfg.DefaultKeys, key)
			}
		case section == "workload" || section == "tier2":
			switch key {
			case "depths":
				cfg.T2.Depths, err = parseIntList(val)
			case "gen":
				cfg.T2.Gen, err = atoi(key, val)
			case "repeats":
				cfg.T2.Repeats, err = atoi(key, val)
			case "warmup":
				b, e := parseBool(val)
				err = e
				if e == nil {
					cfg.T2.Warmup = b
				}
			default:
				return nil, fmt.Errorf("line %d: unknown workload key %q", lineno+1, key)
			}
		case section == "tier1":
			switch key {
			case "enabled":
				b, e := parseBool(val)
				err = e
				if e == nil {
					cfg.T1.Enabled = b
				}
			case "depths":
				cfg.T1.Depths, err = parseIntList(val)
			case "pp":
				cfg.T1.PP, err = atoi(key, val)
			case "gen":
				cfg.T1.Gen, err = atoi(key, val)
			case "repeats":
				cfg.T1.Repeats, err = atoi(key, val)
			default:
				return nil, fmt.Errorf("line %d: unknown tier1 key %q", lineno+1, key)
			}
		default:
			if cur == nil {
				return nil, fmt.Errorf("line %d: key %q outside any section", lineno+1, key)
			}
			switch key {
			case "inherit":
				cur.Inherit = val
			case "tier1":
				if b, e := parseBool(val); e == nil && !b {
					cur.NoTier1 = true
				}
			case "tier2":
				if b, e := parseBool(val); e == nil && !b {
					cur.NoTier2 = true
				}
			case "depths":
				cur.DepthOverride, err = parseIntList(val)
			default:
				cur.Flags[key] = val
				cur.FlagKeys = append(cur.FlagKeys, key)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineno+1, err)
		}
	}

	for _, p := range candPtrs {
		cfg.Candidates = append(cfg.Candidates, *p)
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("config has no model: set 'hf = ...' or 'model = /path' at the top")
	}
	if len(cfg.Candidates) == 0 {
		return nil, fmt.Errorf("config has no candidate sections")
	}
	return cfg, nil
}

// pinCPUs reads the resolved "pin-cpus" value ("0-3,8" list; empty = all).
func pinCPUs(resolved map[string]string) []int {
	v, ok := resolved["pin-cpus"]
	if !ok || v == "" {
		return nil
	}
	var out []int
	for _, part := range strings.Split(v, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if lo, hi, ok := strings.Cut(part, "-"); ok {
			a, e1 := strconv.Atoi(strings.TrimSpace(lo))
			b, e2 := strconv.Atoi(strings.TrimSpace(hi))
			if e1 != nil || e2 != nil || b < a {
				return nil
			}
			for i := a; i <= b; i++ {
				out = append(out, i)
			}
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil
		}
		out = append(out, n)
	}
	return out
}

// indexInlineComment returns the index of an inline # or ; comment
// (must be preceded by whitespace so values containing the chars survive).
func indexInlineComment(line string) int {
	for i := 1; i < len(line); i++ {
		if (line[i] == '#' || line[i] == ';') && line[i-1] == ' ' {
			return i
		}
	}
	return -1
}

func atoi(key, val string) (int, error) {
	n, err := strconv.Atoi(val)
	if err != nil {
		return 0, fmt.Errorf("bad int for %s: %q", key, val)
	}
	return n, nil
}

// serverArgs turns a resolved flag map into llama-server CLI args.
func serverArgs(flags map[string]string) []string {
	var out []string
	for k, v := range flags {
		switch k {
		case "hf", "model":
			continue // model is passed by the caller
		case "pin-cpus":
			continue // consumed by the harness (process affinity)
		case "flash-attn":
			// serve takes on|off|auto, not true/false - normalize like bench
			switch strings.ToLower(v) {
			case "true", "on", "yes", "1":
				out = append(out, "--flash-attn", "on")
			case "false", "off", "no", "0":
				out = append(out, "--flash-attn", "off")
			default:
				out = append(out, "--flash-attn", v)
			}
		default:
			if boolFlags[k] {
				b, err := parseBool(v)
				if err != nil {
					out = append(out, "--"+k, v)
					continue
				}
				if b {
					out = append(out, "--"+k)
				} else {
					out = append(out, "--no-"+k)
				}
				continue
			}
			out = append(out, "--"+k, v)
		}
	}
	return out
}
