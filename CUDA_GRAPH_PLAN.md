# CUDA Graphs for the decode path — implementation plan

Goal: cut per-token **kernel-launch overhead** on the single-token decode path by
capturing the steady-state decode step into a CUDA Graph and replaying it, instead
of issuing ~1500–2400 individual `<<<>>>` launches every token.

This only targets **decode** (autoregressive generation). Prefill is compute-bound
and already the fastest number in the speed table, so it is left on the normal path.

Branch: `cuda-graph` (forked from `main`, independent of the TP work on `tp2`).

---

## Why this should help

Decode on DGX Spark (GB10) is memory-bandwidth bound, but the engine is **not**
saturating bandwidth: each token issues hundreds of tiny kernel launches on the
default stream with gaps between them (CPU launch latency + bubbles where the GPU
is idle waiting for the next launch). A captured graph replays the whole token in
a single driver submission, removing the per-launch CPU cost and tightening
kernel-to-kernel scheduling so the memory bus stays busier.

Upper bound: replay can only push decode **up to** the bandwidth ceiling
(~273 GB/s ÷ bytes-per-token ≈ tens of t/s). It cannot beat physics. Realistic
target is closing the gap between today's effective bandwidth and that ceiling.

**Measure first.** Run the baseline with the per-token profiler (see below). If
`encode` (CPU launch) time is a large fraction of `total`, graphs win big. If
`execute` already dominates, the win is small and we stop here.

---

## Current decode structure (the seam)

Host entry (one token):
- `ds4.c:19461` `metal_graph_eval_token_raw_swa()` — non-streaming decode, the path
  used for q2 weights resident in unified memory (our case).
  ```c
  ds4_gpu_begin_commands();                  // ds4_cuda.cu:2479  (currently no-op)
  metal_graph_encode_token_raw_swa(...);     // ds4.c:16946  records all kernels
  ds4_gpu_end_commands();                    // ds4_cuda.cu:2494 (currently cudaDeviceSynchronize)
  ds4_gpu_tensor_read(g->logits, ...);       // ds4.c:19483  D2H logits — AFTER, outside the step
  ```
- Per-layer kernels: `ds4.c:14842` `metal_graph_encode_decode_layer()`.
- Output head: `ds4.c:16071` `metal_graph_encode_output_head()`.

This is the ideal seam: **`begin_commands` → BeginCapture**, **`end_commands` →
EndCapture (+ instantiate/launch)**. We do not need to touch the ~156 launch
sites — capture wraps the whole encode.

### Properties confirmed from the code
- **Single compute stream**: all `<<<>>>` launches use the **default stream**.
  Named streams (`g_model_prefetch_stream` @2:89, `g_model_upload_stream` @2:90,
  `g_stream_selected_upload_stream` @2:222) are **idle** when q2 weights are
  resident (no SSD streaming).
- **No mid-step host sync** in our case: the only sync (`cudaDeviceSynchronize`)
  and the only blocking D2H (logits) happen *after* the encode. The CPU-router
  and SSD-streaming paths that *do* sync mid-step are gated behind
  `g->ssd_streaming` / PRO-Q4 conditions and are not taken for Flash q2 resident.
- **Stable topology**: grid dims are compile-time/architecture constants
  (`n_head`, `n_expert`, …). Per-token variation is in **by-value scalar args**
  (`pos`, `n_raw`, `raw_start`) and device pointers (`selected_ptr`), not in which
  kernels launch. So the captured graph topology is identical every token.

### The two blockers
1. **Default stream is not capturable.** `cudaStreamBeginCapture` cannot capture
   the legacy default stream (stream 0).
2. **By-value per-token scalars.** A capture-once/replay-many graph would replay
   the `pos`/`n_raw`/`raw_start` values baked in at capture time → wrong output
   for later positions.

---

## Staged implementation

### Stage 1 — capturable stream foundation (build only, inert) ✅ this commit
- Add make target **`cuda-spark-graph`**: same as `cuda-spark` but compiles with
  `--default-stream per-thread` (so every `<<<>>>` goes to the per-thread default
  stream, `cudaStreamPerThread`, which **is** capturable) and defines
  `DS4_CUDA_GRAPH`.
- No capture yet. **Acceptance:** `cuda-spark-graph` build produces numerically
  identical output to `cuda-spark` (same logits / same `ds4-eval` score) and
  t/s within noise. This validates that moving off the legacy default stream did
  not break the existing event-based cross-stream synchronization.
- Risk: per-thread default removes *implicit* stream-0 sync. The code uses
  explicit events, so this should be safe — but it must be confirmed on the Spark
  before Stage 3.

### Stage 2 — device-resident per-token scalars
- Add a small device buffer `g_token_params` holding `{pos, n_raw, raw_start, …}`.
- Replace the by-value scalar kernel args on the decode path with reads from this
  buffer (or pass its pointer). Update it with one tiny H2D (or a device-side
  write) at the start of each token.
- **Acceptance:** still numerically identical with capture *off*. This is the
  refactor that makes a single captured graph valid across positions.

### Stage 3 — capture once, replay many (behind `DS4_CUDA_GRAPH=1` env)
- `begin_commands`: if graph enabled and not yet captured, and this is a
  steady-state decode token (not first token / not prefill / not a structural
  boundary), `cudaStreamBeginCapture(cudaStreamPerThread, ThreadLocal)`.
- `end_commands`: `cudaStreamEndCapture` → `cudaGraphInstantiate` (first time
  only) → cache `cudaGraphExec_t`. Subsequent tokens: skip encode, just update
  `g_token_params` and `cudaGraphLaunch(exec, cudaStreamPerThread)` then sync.
- Keep the logits D2H read outside the graph (already is).
- Re-capture/re-instantiate only on structural change (model/seq config change).
  Verify no decode kernel changes **grid dims** with position (attention grid is
  `(1,n_head,1)`, KV span is a runtime param — confirmed safe).
- Fallbacks: if `g->ssd_streaming`, CPU-router applicable, or capture fails →
  fall back to the normal launch path. Graphs are an optimization, never required.

### Stage 4 — measure & tune
- Re-run the profiler + bench. Compare encode/execute/total and t/s vs baseline.
- If a cheaper-than-reinstantiate update is needed for structural changes, consider
  `cudaGraphExecUpdate` (CUDA-version-sensitive — wire it against the Spark's
  actual toolkit version).

---

## Baseline benchmark (run before any optimization)

```sh
make cuda-spark
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 --gen-tokens 128
```

Per-token launch-vs-execute split (this tells us the graph upside empirically):

```sh
DS4_METAL_GRAPH_TOKEN_PROFILE=1 ./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 4096 --step-incr 4096 --gen-tokens 64
```

The profile prints `encode / execute / read` ms per token (`ds4.c:19488`). Large
`encode` relative to `execute` ⇒ launch-bound ⇒ graphs help. Save the CSV
(`dgx_spark_baseline.csv`) so Stage 4 has something to compare against.

## Correctness gate (every stage)

Numerics must not change with capture off, and must match within sampling
tolerance with capture on. Use the existing regression/eval tooling:

```sh
make cuda-regression
./ds4-eval ...   # same score with cuda-spark vs cuda-spark-graph
```
