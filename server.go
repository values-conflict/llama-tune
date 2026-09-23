package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"
)

// errLoadFailed marks a server that never became healthy (a model/context
// load failure). That outcome is deterministic for a given config and box -
// re-running on resume just burns minutes reaching the same wall - so the
// harness records it and skips the candidate unless -force is given.
var errLoadFailed = errors.New("server failed to load")

// runTier2 starts llama-server with the given flags and benchmarks it at
// each workload depth over the OpenAI-compatible API.
func runTier2(ctx context.Context, cand *Candidate, resolved map[string]string, wl Workload, modelPath, llamaBin, runDir string) ([]DepthResult, error) {
	if resolved == nil {
		return nil, fmt.Errorf("broken inherit chain")
	}

	port, err := freePort()
	if err != nil {
		return nil, err
	}

	var args []string = []string{"serve"}
	args = append(args, "--host", "127.0.0.1", "--port", fmt.Sprintf("%d", port))
	// keep the prompt cache small and deterministic for benchmarking
	if _, ok := resolved["cache-ram"]; !ok {
		args = append(args, "--cache-ram", "512")
	}
	args = append(args, serverArgs(resolved)...)
	if isHFRef(modelPath) {
		args = append(args, "-hf", modelPath)
	} else {
		args = append(args, "-m", modelPath)
	}

	logPath := filepath.Join(runDir, fmt.Sprintf("server-%s.log", cand.Name))
	logF, err := os.Create(logPath)
	if err != nil {
		return nil, err
	}
	defer logF.Close()

	cmd := exec.Command(llamaBin, args...)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = logF
	// the server inherits our affinity at execve; the harness can release it
	aff := pinCPUs(resolved)
	if err := withAffinity(aff, cmd.Start); err != nil {
		return nil, err
	}
	// single Wait: exitCh reports process death so a load failure fails fast
	// instead of waiting out the whole health timeout
	exitCh := make(chan error, 1)
	go func() { exitCh <- cmd.Wait() }()
	defer stopServer(cmd, exitCh)

	url := fmt.Sprintf("http://127.0.0.1:%d", port)
	healthCtx, cancel := context.WithTimeout(ctx, 20*time.Minute)
	defer cancel()
	if err := waitHealthy(healthCtx, url+"/health", exitCh); err != nil {
		// include the tail of the log for quick diagnosis
		tail := tailOf(logPath, 12)
		return nil, fmt.Errorf("%w: %v\nlog tail:\n%s", errLoadFailed, err, tail)
	}

	defer func() {
		// capture a final log chunk that only appears on shutdown (memory breakdown)
		stopServer(cmd, exitCh)
	}()

	// warmup: trigger shader compilation / lazy init on a throwaway request
	if wl.Warmup {
		if _, err := completion(ctx, url, "hello, this is a warmup request.", 8, 30*time.Minute); err != nil {
			return nil, fmt.Errorf("warmup request failed: %w", err)
		}
	}

	var out []DepthResult
	for i, depth := range wl.Depths {
		for r := 0; r < wl.Repeats; r++ {
			prompt := makePrompt(depth, i, r)
			d, err := completion(ctx, url, prompt, wl.Gen, requestTimeout(depth))
			if err != nil {
				// keep the results measured so far; one bad depth should not
				// discard the whole candidate
				return out, fmt.Errorf("depth %d repeat %d: %w", depth, r, err)
			}
			d.Depth = depth
			out = append(out, d)
			acc := ""
			if d.acceptRate() >= 0 {
				acc = fmt.Sprintf(" acc=%.2f", d.acceptRate())
			}
			fmt.Printf("  [%s] depth %6d (prompt %6d): pp %7.1f t/s, tg %7.2f t/s%s\n",
				cand.Name, depth, d.PromptN, d.PP, d.TG, acc)
		}
	}
	return out, nil
}

// requestTimeout bounds one depth: a prefill at a pessimistic 10 t/s plus
// room for the decode and slow hardware. 262144 ctx -> about 7.8 h.
func requestTimeout(depth int) time.Duration {
	return 30*time.Minute + time.Duration(float64(depth)/600.0)*time.Minute
}

type completionResp struct {
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
	} `json:"usage"`
	Timings struct {
		PromptN         int     `json:"prompt_n"`
		PromptPerSecond float64 `json:"prompt_per_second"`
		PredictedN      int     `json:"predicted_n"`
		PredictedPerSec float64 `json:"predicted_per_second"`
		DraftN          int     `json:"draft_n"`
		DraftNAccepted  int     `json:"draft_n_accepted"`
	} `json:"timings"`
}

func completion(ctx context.Context, url, prompt string, maxTokens int, timeout time.Duration) (DepthResult, error) {
	body, _ := json.Marshal(map[string]any{
		"prompt":      prompt,
		"max_tokens":  maxTokens,
		"temperature": 0,
		"stream":      false,
		// the synthetic prompt can end on a spot where the model emits EOS at
		// token 1, which would measure a 0.00 tg depth; force the full decode
		"ignore_eos": true,
	})
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "POST", url+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return DepthResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return DepthResult{}, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return DepthResult{}, err
	}
	if resp.StatusCode != 200 {
		return DepthResult{}, fmt.Errorf("http %d: %.500s", resp.StatusCode, string(data))
	}
	var cr completionResp
	if err := json.Unmarshal(data, &cr); err != nil {
		return DepthResult{}, fmt.Errorf("bad response json: %w", err)
	}
	d := DepthResult{
		PromptN:  cr.Timings.PromptN,
		PP:       cr.Timings.PromptPerSecond,
		TG:       cr.Timings.PredictedPerSec,
		DraftN:   cr.Timings.DraftN,
		DraftAcc: cr.Timings.DraftNAccepted,
		At:       time.Now().UTC().Format(time.RFC3339),
	}
	if d.PromptN == 0 {
		d.PromptN = cr.Usage.PromptTokens
	}
	return d, nil
}

// makePrompt builds a deterministic ~depth-token prompt that is unique per
// (depth, repeat) so the server prompt cache never serves a repeat.
func makePrompt(depth, which, repeat int) string {
	base := "The quick brown fox jumps over the lazy dog near the riverbank while the farmer counts the remaining grain in the barn and the children watch the smoke rise from the evening fire. "
	// roughly 4.6 chars per token
	n := int(float64(depth) * 4.6 / float64(len(base)))
	if n < 1 {
		n = 1
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Benchmark sample %d-%d. ", which+1, repeat+1)
	for i := 0; i < n; i++ {
		b.WriteString(base)
	}
	return b.String()
}

func waitHealthy(ctx context.Context, url string, exitCh <-chan error) error {
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-exitCh:
			// the process is gone: a load failure, not a slow start - fail now
			return fmt.Errorf("server process exited before becoming healthy")
		case <-tick.C:
		}
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func stopServer(cmd *exec.Cmd, exitCh <-chan error) {
	// ProcessState is set once the single Wait (fed into exitCh) has returned,
	// which also guards the second deferred call from blocking on an empty channel
	if cmd.Process == nil || cmd.ProcessState != nil {
		return
	}
	_ = cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-exitCh:
	case <-time.After(60 * time.Second):
		_ = cmd.Process.Kill()
		<-exitCh
	}
}

var buildRe = regexp.MustCompile(`build: (\d+ \([0-9a-f]+\))`)

func extractBuild(logPath string) string {
	data, err := os.ReadFile(logPath)
	if err != nil {
		return ""
	}
	if loc := buildRe.FindSubmatch(data); loc != nil {
		return string(loc[1])
	}
	return ""
}

func tailOf(path string, n int) string {
	f, err := os.Open(path)
	if err != nil {
		return "(no log)"
	}
	defer f.Close()
	var lines []string
	// cheap approach: read whole file, keep last n lines; logs are small
	data, _ := io.ReadAll(f)
	for _, l := range strings.Split(string(data), "\n") {
		lines = append(lines, l)
		if len(lines) > n {
			lines = lines[1:]
		}
	}
	return strings.Join(lines, "\n")
}
