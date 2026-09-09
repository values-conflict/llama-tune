package main

import (
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

// withAffinity runs fn with the calling goroutine pinned to cpus.
// Child processes spawned inside fn inherit the affinity at execve,
// so bench and server runs are restricted without any taskset wrapper.
var (
	sysSchedSetaffinity uintptr
	sysSchedGetaffinity uintptr
)

func init() {
	// per-arch syscall numbers (get = set + 1 on both):
	switch runtime.GOARCH {
	case "amd64":
		sysSchedSetaffinity, sysSchedGetaffinity = 203, 204
	case "arm64":
		sysSchedSetaffinity, sysSchedGetaffinity = 122, 123
	default:
		// leave both 0: withAffinity will fail clearly if ever used
	}
}

// Uses raw Linux syscalls to stay dependency-free (x/sys/unix would work
// too, but this keeps the module at zero external deps).
func withAffinity(cpus []int, fn func() error) error {
	if len(cpus) == 0 {
		return fn()
	}
	if sysSchedSetaffinity == 0 {
		return fmt.Errorf("pin-cpus: unsupported architecture %s", runtime.GOARCH)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// struct cpumask sized for up to 1024 cpus
	const nwords = 1024 / 64
	var old [nwords]uint64
	var m [nwords]uint64

	// sched_getaffinity(pid_t pid, unsigned int len, unsigned long *mask)
	r1, _, e1 := syscall.Syscall6(sysSchedGetaffinity, 0,
		uintptr(nwords*8), uintptr(unsafe.Pointer(&old[0])), 0, 0, 0)
	if r1 == 0 && e1 == 0 {
		// restore on exit
		defer func() {
			syscall.Syscall6(sysSchedSetaffinity, 0,
				uintptr(nwords*8), uintptr(unsafe.Pointer(&old[0])), 0, 0, 0)
		}()
	} else if e1 != 0 {
		return fmt.Errorf("sched_getaffinity: %w", e1)
	}
	for _, c := range cpus {
		if c < 0 || c >= 1024 {
			return fmt.Errorf("pin-cpus: cpu %d out of range", c)
		}
		m[c/64] |= 1 << (c % 64)
	}
	_, _, e2 := syscall.Syscall6(sysSchedSetaffinity, 0,
		uintptr(nwords*8), uintptr(unsafe.Pointer(&m[0])), 0, 0, 0)
	if e2 != 0 {
		return fmt.Errorf("sched_setaffinity: %w", e2)
	}
	return fn()
}
