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

    llama-tune compare <runA> <runB>
        Align two runs cell by cell (same candidate + depth) and print
        the deltas. This is the v3-vs-v4 story: run the same sweep with
        two differently-compiled llama.cpp trees, then compare.

run flags:

    -config FILE        sweep config (required)
    -dir DIR            output directory (default: run-<timestamp>)
    -llama-bin PATH     the unified llama binary (default: llama)
    -only a,b,c         only run these candidates
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

## Putting it in the Docker image

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
