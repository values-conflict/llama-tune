# llama-tune

Benchmarks combinations of llama.cpp server options on a fixed model and
reports the results per context depth, so you can find the fastest
configuration for the hardware you are running on.

Built for a specific setup: an AMD HX 370 (Zen 5, Radeon 890M iGPU, unified
memory) running `llama-server` with Vulkan, but the tool is generic - any
model, any build, any set of flags.

## Commands

    llama-tune scan <model> [model...]
        Inspect a gguf (or an hf ref like org/repo:quant): arch, layers,
        trained context, GQA ratio, MoE, quant, MTP support, and the
        projected KV cache size at 32k/131k/262k for q4_0 kv.
        hf refs are downloaded via `llama download` first (a no-op if the
        file is already in the cache).

    llama-tune run -config sweep.ini -dir run-1 [flags]
        Run the sweep. Writes run-1/run.json, results.tsv, results.md and
        one log per candidate/phase.

    llama-tune plan [flags]
        Generate a sweep config from the model and the local hardware.
        Scans the gguf (MTP, MoE, trained ctx), probes each kv cache
        type with `llama fit-params` (so per-type memory fit is
        authoritative), reads the CPU topology, and emits an .ini on
        stdout ready for `run`. Progress goes to stderr, so
        `llama-tune plan ... > sweep.ini` works.

    llama-tune emit [flags]
        Print the winning candidate of a run as a models.ini section
        (the router's config format) ready to paste: flags resolved
        through the inherit chain, ranked by tg (or pp) at the deepest
        measured depth at or below -depth.

    llama-tune compare <runA> <runB>
        Align two runs cell by cell (same candidate + depth) and print
        the deltas. This is the v3-vs-v4 story: run the same sweep with
        two differently-compiled llama.cpp trees, then compare.

run flags:

    -config FILE        sweep config (required)
    -dir DIR            output directory (default: run-<timestamp>)
    -llama-bin PATH     the unified llama binary (default: llama)
    -only a,b,c         only run these candidates
    -skip a,b,c         exclude these candidates (inverse of -only)
    -phases t1,t2       t1 = llama-bench matrix, t2 = live server (default: both)

Everything goes through the unified `llama` binary: `llama bench` for
tier 1, `llama serve` for tier 2, `llama download` to get models on disk.

## How a sweep works

Two phases, per candidate:

- tier1 (llama-bench): no server, no spec decoding. Runs pp + tg tests at
  each configured context depth. Fast and repeatable, used for the coarse
  knobs (threads, flash attn, kv type, batch sizes). Candidates that use
  speculative decoding are skipped here and only measured in tier2.
- tier2 (live server): starts llama-server with the candidate's full flag
  set (this is where spec decoding, jinja, prompt cache, etc. are tested),
  waits for /health, sends a warmup request (shader compilation), then one
  request per depth: a synthetic prompt of ~depth tokens, generating
  `gen` tokens. Reads back the server's own `timings`
  (prompt_per_second, predicted_per_second, draft_n, draft_n_accepted).
  Prompts are unique per (depth, repeat) so the server prompt cache never
  serves a repeat.

Each run directory carries a manifest: model, llama.cpp build (number +
commit, captured from the tool output/logs), config text, and results.
`compare` relies on candidate names + depths lining up between runs.

## Config format

INI-ish. Top-level keys are defaults for every candidate. `[tier1]` and
`[tier2]` configure the workloads. Everything else is a candidate: a name
plus llama.cpp flags, passed through unchanged (booleans: `true` becomes
`--flag`, `false` becomes `--no-flag`).

    hf = org/model:quant        # or: model = /path/to/file.gguf
    ctx-size = 262144           # defaults for all candidates
    jinja = true
    ...

    [tier1]
    depths = 1024, 16384, 65536 # context depths for llama-bench
    pp = 4096                   # prompt length for pp tests
    gen = 64                    # tokens for tg tests
    repeats = 2

    [tier2]
    depths = 4096, 16384, 65536, 131072, 262144
    gen = 64
    repeats = 1
    warmup = true

    [baseline]
    n-gpu-layers = all
    threads = 12
    flash-attn = true
    cache-type-k = q4_0
    cache-type-v = q4_0
    spec-type = draft-mtp
    spec-draft-n-max = 2

    [threads-24]
    inherit = baseline          # start from another candidate
    threads = 24
    spec-type = none
    depths = 4096, 65536        # optional: override tier2 depths

Candidate-only keys: `inherit = <candidate>`, `tier1 = no`, `tier2 = no`,
`depths = ...` (tier2 override). Anything else is treated as a llama.cpp
flag, so new server flags work without changing the tool.

`spec-type = auto` is special: the tool scans the gguf for the MTP module
(the same check llama.cpp uses - the `blk.<last>.nextn.eh_proj.weight`
tensor) and picks `draft-mtp` when present, otherwise `ngram-mod` with the
same values as llama.cpp's built-in `--spec-default`. This is what makes
the config work unchanged for gemma and friends.

## A typical night

    # on the box, first time with a model:
    llama-tune scan unsloth/Some-27B-GGUF:Q4_K_XL     # what is this model?
    llama-tune plan -hf unsloth/Some-27B-GGUF:Q4_K_XL > sweep.ini  # draft config
    $EDITOR sweep.ini                              # tweak, add escape-hatch candidates
    llama-tune run -config sweep.ini -dir run-v3
    llama-tune emit -run run-v3                    # paste the winner into models.ini

`plan` computes the feasible configuration space, in three tiers:

1. **computed defaults** - ctx, kv type, threads, and the prompt cache
   (`cache-ram`). All memory math is cgroup/cpuset-aware: it reads the
   container's memory limit, cpuset, and cpu.max quota (v2 and v1), not
   the host totals. ggml's own memory views are not cgroup-aware, so
   `plan` takes the *size* from `llama fit-params` (one probe per kv
   type at your target ctx, at -lv 4 to read the "projected to use" line)
   and the *envelope* from the cgroup limits; since size is linear in
   ctx x slots, the whole (ctx, parallel) space falls out of arithmetic
   from those anchors.
2. **active candidates** - a one-knob-at-a-time matrix over the feasible
   values: kv types that fit, flash-attn on/off, thread counts from the
   container's actual CPU view, spec variants from MTP detection, and a
   tier1-only ubatch ladder (batch/ubatch are prompt-processing knobs,
   measured directly by llama-bench's pp test).
3. **commented trade-off sections** - the speed-neutral decisions, with
   exact numbers: each `(ctx, parallel)` tuple that fits at or above
   `-min-ctx` (default 32k) becomes a commented section with the ctx,
   parallel, and the prompt cache headroom it leaves. Uncomment to run.
   These scale to bigger hardware and smaller models automatically
   because they are pure arithmetic from the anchors.

## Resuming and partial results

Results are written after *every* candidate, so an interrupted or killed
run keeps everything it measured. Re-running the same `-dir` resumes:
measured `(candidate, phase, depth)` rows are kept and only the missing
ones are measured (add `-force` to re-measure everything; a dir holding
a different model is refused). This makes patching a sweep trivial -
add a depth to the config, re-run, done. A tier-2 candidate whose server never
becomes healthy (a load failure) is recorded as skipped so resume does not
re-run it; `-force` retries it. The health wait also fails fast when the
server process exits, instead of polling for the full timeout. `llama-tune report -dir <run>`
re-renders results.md/tsv from run.json at any time (e.g. after a crash)
and prints the summary.

## Config flags worth knowing

`spec-type` accepts comma-separated chains (`draft-mtp,ngram-mod`):
drafts are tried in order and the first implementation that produces a
draft wins the step, the rest are skipped. For an MTP model the MTP
draft essentially always succeeds, so ngram behind it is a fallback that
rarely fires - it adds coverage, not speed on top.

Performance cores: `plan` detects a P/E cluster from the topology
(`/sys/devices/cpu_core/cpus` - Intel hybrid x86 like the 14900K)
and falls back to cpufreq max-frequency tiers (ARM big.LITTLE, within
10% of the top). When one exists it emits a `[fast-cpus]` candidate
pinning to the P-cores with one thread per physical P-core (SMT off),
but only when the fit probe shows CPU inference is actually happening
(ngl not all) - pinning is pointless when the gpu does the work. The
plain threads-* candidates cover the HT axis separately (full-width,
physical vs logical). You can also pin any candidate by hand:

    [pinned]
    inherit = baseline-nospec
    pin-cpus = 0-3

The harness sets process affinity around the spawn (raw
`sched_setaffinity`, dependency-free), which children inherit at
execve - no `taskset` needed, and the flag never reaches llama.cpp.

The binary is static Go, no dependencies:

    CGO_ENABLED=0 go build -o llama-tune .

Your image already ships the unified `llama` binary (which contains
`serve`, `bench`, and `download` as subcommands), so the only thing to
add is `llama-tune` itself:

    /install/bin/llama-tune run -config /sweeps/qwen.ini -dir /results/run-v3
    # default -llama-bin is "llama"; pass -llama-bin /install/bin/llama
    # if it is not on PATH

Then rebuild the image with the different compile flags (e.g.
`-march=x86-64-v4`) and run the same sweep into a second directory:

    /install/bin/llama-tune compare /results/run-v3 /results/run-v4

## Notes and caveats

- tier2 numbers are single-shot per depth (Vulkan shader compilation is
  handled by the warmup; run `repeats = 2` in [tier2] for tighter
  numbers at the cost of prefill time).
- The synthetic tier2 prompts are repetitive, which inflates ngram spec
  acceptance (and makes pp look faster than real prompts). draft-mtp
  acceptance is largely unaffected. Read spec results as relative, and
  re-check the winner in real use.
- tier1 (llama-bench) measures without the server layer; expect small
  differences from tier2 for the same flags.
- If a candidate's flag combination is invalid (e.g. q4_0 kv without
  flash attn on some backends), the server fails to start; the harness
  records the error and the log tail and continues with the next
  candidate.
- `sweep-smoke.ini` exercises the whole pipeline with a 0.8B model at
  shallow depths - use it after changing the tool or the llama.cpp
  build, before committing to an overnight run.
