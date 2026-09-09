package main

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
)

func runtimeNumCPU() int { return runtime.NumCPU() }

// Resources is a cgroup/cpuset-aware view of what this container may use.
type Resources struct {
	MemTotalMiB   int64  // MemTotal from /proc/meminfo (host view)
	MemLimitMiB   int64  // cgroup memory limit, 0 = unlimited
	MemAvailMiB   int64  // the envelope we actually plan against
	CGroup        string // "v2", "v1", or "none"
	LogicalCPUs   int    // cpus allowed by cpuset/affinity
	PhysicalCPUs  int
	QuotaCPUs     float64 // from cpu.max / cfs quota, 0 = unlimited
	EffectiveCPUs int     // min(logical, floor(quota))
	FastCPUs      []int   // performance-core cluster (P-cores); empty = homogeneous
	FastPhysical  int     // physical cores inside FastCPUs (0 if none)
}

func (r *Resources) String() string {
	mem := fmt.Sprintf("%d MiB", r.MemAvailMiB)
	if r.MemLimitMiB > 0 && r.MemLimitMiB < r.MemTotalMiB {
		mem = fmt.Sprintf("%d MiB (cgroup limit, host has %d)", r.MemLimitMiB, r.MemTotalMiB)
	}
	cpu := fmt.Sprintf("%d logical / %d physical", r.LogicalCPUs, r.PhysicalCPUs)
	if r.QuotaCPUs > 0 {
		cpu += fmt.Sprintf(" (quota %.1f)", r.QuotaCPUs)
	}
	return fmt.Sprintf("mem %s, %s, cgroup %s", mem, cpu, r.CGroup)
}

// probeResources reads cgroup v2 (then v1) limits plus CPU topology.
func probeResources() (*Resources, error) {
	r := &Resources{}
	r.MemTotalMiB = memTotalMiB()

	if fileExists("/sys/fs/cgroup/cgroup.controllers") {
		r.CGroup = "v2"
		if v, ok := readMemMax("/sys/fs/cgroup/memory.max"); ok {
			r.MemLimitMiB = v
		}
		r.LogicalCPUs = countCPUList(readCPUList("/sys/fs/cgroup/cpuset.cpus.effective"))
		if r.LogicalCPUs == 0 {
			r.LogicalCPUs = runtimeNumCPU()
		}
		if q, ok := readCPUQuota("/sys/fs/cgroup/cpu.max"); ok {
			r.QuotaCPUs = q
		}
	} else if fileExists("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
		r.CGroup = "v1"
		if v, ok := readMemMax("/sys/fs/cgroup/memory/memory.limit_in_bytes"); ok {
			r.MemLimitMiB = v
		}
		r.LogicalCPUs = countCPUList(readCPUList("/sys/fs/cgroup/cpuset/cpuset.cpus"))
		if r.LogicalCPUs == 0 {
			r.LogicalCPUs = runtimeNumCPU()
		}
		if q, ok := readCFSQuota(); ok {
			r.QuotaCPUs = q
		}
	} else {
		r.CGroup = "none"
		r.LogicalCPUs = runtimeNumCPU()
	}

	// physical cores among the allowed cpus (distinct SMT sibling sets)
	r.PhysicalCPUs = distinctPhysical(cpuIDs(r.LogicalCPUs))
	if r.PhysicalCPUs < 1 {
		r.PhysicalCPUs = r.LogicalCPUs
	}

	if r.QuotaCPUs > 0 {
		r.EffectiveCPUs = int(math.Floor(r.QuotaCPUs + 1e-9))
		if r.EffectiveCPUs > r.LogicalCPUs {
			r.EffectiveCPUs = r.LogicalCPUs
		}
		if r.EffectiveCPUs < 1 {
			r.EffectiveCPUs = 1
		}
	} else {
		r.EffectiveCPUs = r.LogicalCPUs
	}

	// plan against the cgroup limit when it is below the host total,
	// leaving room for the kernel/bookkeeping the container also pays
	if r.MemLimitMiB > 0 && r.MemLimitMiB < r.MemTotalMiB {
		r.MemAvailMiB = r.MemLimitMiB
	} else {
		r.MemAvailMiB = r.MemTotalMiB
	}
	r.FastCPUs = detectFastCPUs(r.LogicalCPUs)
	if len(r.FastCPUs) > 0 {
		r.FastPhysical = distinctPhysical(r.FastCPUs)
	}
	return r, nil
}

// distinctPhysical counts physical cores (distinct SMT sibling sets) in cpus.
func distinctPhysical(cpus []int) int {
	sets := map[string]bool{}
	n := 0
	for _, id := range cpus {
		data, err := os.ReadFile(fmt.Sprintf("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", id))
		if err != nil {
			continue
		}
		key := strings.TrimSpace(string(data))
		if !sets[key] {
			sets[key] = true
			n++
		}
	}
	if n == 0 {
		n = len(cpus)
	}
	return n
}

// detectFastCPUs finds the performance-core cluster. Two signals, in order:
//
//  1. topology clusters (Intel hybrid x86): /sys/devices/cpu_core/cpus
//  2. cpufreq max-frequency tiers (ARM big.LITTLE etc): the top tier, within
//     10% of the max (P/E gaps are ~40% on both Apple and Intel hybrid)
//
// Returns nil when the allowed set is homogeneous or the info is missing.
func detectFastCPUs(allowed int) []int {
	if data, err := os.ReadFile("/sys/devices/cpu_core/cpus"); err == nil {
		list := strings.TrimSpace(string(data))
		n := countCPUList(list)
		if n > 0 && n < allowed {
			return expandCPUList(list)
		}
	}
	ids := cpuIDs(allowed)
	freqs := make([]int64, len(ids))
	for i, id := range ids {
		v, err := readMaxFreq(id)
		if err != nil {
			return nil
		}
		freqs[i] = v
	}
	max := int64(0)
	for _, v := range freqs {
		if v > max {
			max = v
		}
	}
	if max == 0 {
		return nil
	}
	var fast []int
	for i, v := range freqs {
		if v*10 >= max*9 {
			fast = append(fast, ids[i])
		}
	}
	if len(fast) == len(ids) {
		return nil // no efficiency tier present
	}
	return fast
}

// expandCPUList expands a "0-15,17" list into cpu ids.
func expandCPUList(list string) []int {
	var out []int
	for _, part := range strings.Split(list, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if lo, hi, ok := strings.Cut(part, "-"); ok {
			a, e1 := strconv.Atoi(lo)
			b, e2 := strconv.Atoi(hi)
			if e1 != nil || e2 != nil || b < a {
				continue
			}
			for i := a; i <= b; i++ {
				out = append(out, i)
			}
			continue
		}
		if n, err := strconv.Atoi(part); err == nil {
			out = append(out, n)
		}
	}
	return out
}

func readMaxFreq(cpu int) (int64, error) {
	for _, p := range []string{
		fmt.Sprintf("/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu),
		fmt.Sprintf("/sys/devices/system/cpu/cpufreq/cpuinfo_max_freq"),
	} {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		v, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
		if err != nil {
			return 0, err
		}
		return v, nil
	}
	return 0, fmt.Errorf("no cpufreq info for cpu %d", cpu)
}

func memTotalMiB() int64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				v, _ := strconv.ParseInt(f[1], 10, 64)
				return v / 1024
			}
		}
	}
	return 0
}

// readMemMax parses a cgroup memory limit; "max" or a near-uint64-max value means unlimited.
func readMemMax(path string) (int64, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	s := strings.TrimSpace(string(data))
	if s == "max" {
		return 0, true
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, false
	}
	// v1 reports unlimited as a huge page-aligned number
	if v > 1<<60 {
		return 0, true
	}
	return v / (1024 * 1024), true
}

func readCPUList(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// countCPUList counts cpus in a "0-7,12" list.
func countCPUList(list string) int {
	n := 0
	for _, part := range strings.Split(list, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if lo, hi, ok := strings.Cut(part, "-"); ok {
			a, err1 := strconv.Atoi(lo)
			b, err2 := strconv.Atoi(hi)
			if err1 == nil && err2 == nil && b >= a {
				n += b - a + 1
				continue
			}
		}
		if _, err := strconv.Atoi(part); err == nil {
			n++
		}
	}
	return n
}

// cpuIDs expands a limit into 0..n-1; we only need the first `count` ids
// for the topology walk, and containers usually start at 0.
func cpuIDs(count int) []int {
	ids := make([]int, 0, count)
	for i := 0; i < count; i++ {
		ids = append(ids, i)
	}
	return ids
}

func readCPUQuota(path string) (float64, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	f := strings.Fields(strings.TrimSpace(string(data)))
	if len(f) != 2 || f[0] == "max" {
		return 0, true
	}
	quota, err1 := strconv.ParseInt(f[0], 10, 64)
	period, err2 := strconv.ParseInt(f[1], 10, 64)
	if err1 != nil || err2 != nil || period <= 0 {
		return 0, false
	}
	return float64(quota) / float64(period), true
}

func readCFSQuota() (float64, bool) {
	q, err1 := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
	p, err2 := os.ReadFile("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
	if err1 != nil || err2 != nil {
		return 0, false
	}
	quota := strings.TrimSpace(string(q))
	if quota == "-1" {
		return 0, true
	}
	qv, e1 := strconv.ParseInt(quota, 10, 64)
	pv, e2 := strconv.ParseInt(strings.TrimSpace(string(p)), 10, 64)
	if e1 != nil || e2 != nil || pv <= 0 {
		return 0, false
	}
	return float64(qv) / float64(pv), true
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
