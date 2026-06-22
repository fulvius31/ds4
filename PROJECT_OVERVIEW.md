# DwarfStar (`ds4`) — Project Overview & DGX Spark Guide

> A code-grounded walkthrough of how DwarfStar works, plus a deep dive on the
> NVIDIA DGX Spark / GB10 backend. Line references point at the source as of the
> commit this was written against; treat them as starting points, not guarantees.

---

## What it is

**DwarfStar** is a self-contained, from-scratch native inference engine written in
C (plus CUDA/Metal/HIP kernels), purpose-built to run **one model family extremely
well: DeepSeek V4 Flash and DeepSeek V4 PRO**. It is explicitly **not** a general
GGUF runner — it will not load arbitrary models.

The bet is narrow on purpose: pick the best open-weight model for a given memory
class, validate it against the official implementation's logits, and ship the
engine + the GGUF + the test harness + a coding agent as one finished package.

- Repo: `github.com/fulvius31/ds4`
- Status: **beta** quality (the `ds4-agent` is **alpha**).
- Developed "with strong assistance from GPT 5.5"; owes its kernels, quant
  formats, and GGUF ecosystem to **llama.cpp & GGML**.

---

## The big picture

```
                      ┌─────────────────────────────────────────────┐
   5 binaries  ──────▶│  ds4_engine (immutable model, mmap'd GGUF)   │
   ds4 (CLI/REPL)     │  ds4_session (one inference timeline + KV)   │
   ds4-server (HTTP)  └─────────────────────────────────────────────┘
   ds4-agent (coder)                    │
   ds4-bench (speed)                    ▼  ds4_gpu_* (one flat C ABI)
   ds4-eval (quality)     ┌─────────────┴─────────────┐
                          Metal        CUDA/DGX Spark   ROCm/Strix Halo
                       ds4_metal.m    ds4_cuda.cu       ds4_rocm.cu
                       (PRIMARY)      (this is Spark)   (gfx1151)
```

Everything is built around two opaque objects in `ds4.h`:

- **`ds4_engine`** — the immutable, `mmap`-loaded model.
- **`ds4_session`** — one mutable inference timeline that owns the live KV cache
  and the next-token logits.

`ds4.c` is the monster (~28k lines, 1.2 MB): it owns GGUF parsing, the quantized
tensor kernels, the CPU reference forward pass, the tokenizer/chat templating, and
the GPU graph driver. The GPU work goes through a single flat C ABI in `ds4_gpu.h`
(~120 `ds4_gpu_*` functions) that is implemented exactly once per backend and
chosen **at link time**, not at runtime.

---

## The model: DeepSeek V4 and why it's special

The engine hard-codes two shape profiles and **validates every GGUF dimension
against them** (`config_validate_model()`, `ds4.c:3888`) — any mismatch aborts the
load. That is why arbitrary GGUFs do not work.

| | Flash | PRO |
|---|---|---|
| Layers | 43 | 61 |
| `n_embd` | 4096 | 7168 |
| Routed experts | 256 | 384 |
| Indexer top-k | 512 | 1024 |
| Shared by both | `n_head_kv=1`, `head_dim=512`, 6 active experts, 1 shared expert, **4 hyper-connection streams** | |

Three things make this model a good fit for local inference and shape the whole
engine:

1. **Compressed KV cache (the defining feature).** Every layer keeps a raw
   sliding-window ring of the last 128 tokens. From layer 2 on, layers alternate:
   *even* layers compress 4 token positions into 1 KV row **and** run an "indexer"
   (64 heads × 128 dim) that scores rows and selects up to top-k visible ones;
   *odd* layers compress heavily (ratio 128). This time-axis compression is what
   makes million-token context practical without a per-token KV cache in every
   layer. (`ds4_expected_layer_compress_ratio`, `ds4.c:630`.)

2. **Asymmetric, routed-expert-only quantization.** Quantization touches *only*
   the routed MoE experts (the bulk of the bytes). The 2-bit quant = `IQ2_XXS`
   gate/up + `Q2_K` down; everything else — shared experts, projections, router,
   embeddings, output head — stays `Q8_0`/`F16` to protect quality. A
   higher-memory variant uses `Q4_K` for the experts. The README insists "the
   2-bit quants are not a joke" — they call tools reliably under coding agents.

3. **Manifold-Constrained Hyper-Connections (mHC)** wrap each sublayer: 4 residual
   streams get Sinkhorn-mixed down to one, the sublayer (LoRA-style MLA attention
   or the MoE FFN) runs, then results expand back to 4 streams. Attention is
   multi-head-latent (MLA) with tail-only RoPE on the last 64 dims and an FP8
   (e4m3) round-trip on the cached KV. (`layer_forward_raw_swa_one`, `ds4.c:9679`.)

---

## "The KV cache is a first-class disk citizen"

This is the project's headline philosophy. Because DeepSeek V4's KV cache is
already compressed and SSDs are fast, DwarfStar persists session KV state to disk
and treats RAM as a *speed spectrum* rather than a hard "fits or doesn't" cutoff.

- **Disk KV cache** (`ds4_kvstore.c`): each checkpoint is a file named by the SHA1
  of the *rendered byte prefix* of the prompt. A new prompt sharing a leading byte
  prefix reuses the checkpoint; the suffix is re-tokenized. Logits are saved too,
  so a restored checkpoint can sample the next token without re-decoding. Uses
  `read/write` (not mmap) to avoid adding VM mappings on top of an already-huge
  model map. (Aside: the radix tree in `rax.c` is **not** used here — it is only
  used by the server for verbatim tool-call replay.)

- **SSD streaming** (Metal-only, in `ds4_metal.m`): non-routed weights stay
  resident; routed experts live in a fixed in-RAM cache and are `pread` straight
  from the GGUF on a miss. Lets you run models larger than RAM at a graceful speed
  penalty.

---

## The GPU backend abstraction (and a key gotcha)

`ds4_gpu.h` is a **link-time** contract, not a runtime vtable. Whichever single
`.o` is linked (`ds4_metal.o` / `ds4_cuda.o` / `ds4_rocm.o`) provides all the
symbols. Consequences worth knowing:

- **One binary = one backend.** `--cuda`/`--metal`/`--cpu` only *validate* against
  the compiled-in backend; a mismatch aborts. You cannot have a Metal+CUDA
  universal binary.
- **Metal is the primary, reference target.** The "GPU ready" flag is literally
  named `e->metal_ready` even in CUDA/ROCm builds, and the graph functions are
  `metal_graph_*` — a naming fossil showing the other backends were retrofitted
  onto Metal's shape.
- **ROCm isn't even its own enum.** There is no `DS4_BACKEND_ROCM`; under
  `-DDS4_ROCM_BUILD` the value `DS4_BACKEND_CUDA` *means* ROCm. `ds4_cuda.cu` and
  `ds4_rocm.cu` are two separate kernel trees (ROCm bridges to HIP via macros in
  `ds4_rocm.h`); a fix in one is **not** automatically in the other.

---

## The five binaries

| Binary | What it is |
|---|---|
| **`ds4`** | CLI: one-shot `-p`, or an interactive REPL (linenoise, `~/.ds4_history`, slash commands `/think` `/ctx` `/power` `/read` …). Also does `--inspect`, imatrix collection, perplexity. |
| **`ds4-server`** | HTTP API: 7 endpoints — OpenAI `/v1/chat/completions`, OpenAI `/v1/responses` (preferred for Codex CLI), Anthropic `/v1/messages`, legacy `/v1/completions`, plus `/v1/models`. SSE streaming in each protocol's native shape. **Single graph worker, no request batching** — concurrent requests serialize. No web UI, no `/health`. Defaults to `127.0.0.1:8000`. |
| **`ds4-agent`** | Integrated terminal coding agent (**alpha**). Runs the model in-process, drives native DSML tool-calls: `read`/`write`/`edit`/`list`/`search`/`bash`/`google_search`/`visit_page`. A worker thread keeps the model busy while the UI thread stays responsive (Enter queues, ESC interrupts). `ds4_web.c` is its Chrome-DevTools-Protocol browser automation — **not** a server UI. |
| **`ds4-bench`** | Prefill/decode throughput sweeps over context frontiers → CSV. |
| **`ds4-eval`** | 92-question capability suite (GPQA Diamond, SuperGPQA, AIME 2025, COMPSEC) with a TUI and graders. Explicitly a regression suite, not a leaderboard. |

---

## Scale-out: distributed inference

For the full PRO Q4 model (too big for one machine), DwarfStar splits transformer
layers across machines: coordinator owns layers `0:30`, worker owns `31:output`.
It is a **session backend** (`ds4_distributed.c`) — callers still use the normal
`ds4_session` API.

- Transport: custom binary framing (magic `'DS4D'`) over plain TCP with
  `TCP_NODELAY`; messages are `HELLO / WORK / RESULT / SNAPSHOT_*`.
- Correctness: guarded by a rolling 64-bit token-prefix hash before and after each
  span.
- Prefill is pipelined (fast). **Generation is strictly autoregressive — one
  cross-machine hop per token, so it is slower than single-process** (~19% decode
  loss on Thunderbolt 5, much worse over WiFi).
- No auth/encryption — trusted networks only.

---

## Build & models

The `Makefile` branches on `uname` and picks a backend object:

```sh
make                 # macOS Metal — builds all 5 binaries
make cuda-spark      # Linux CUDA, DGX Spark / GB10   ← (see below)
make cuda-generic    # Linux CUDA, other GPUs (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_120   # explicit arch
make strix-halo      # ROCm / gfx1151 (Framework Desktop)
make cpu             # CPU-only diagnostics (reference path; can crash macOS kernel)
make test            # correctness self-tests
```

On Linux, bare `make` just prints help — you must pick a CUDA target explicitly.
Models come from `download_model.sh` (HF repo `antirez/deepseek-v4-gguf`):

| Target | ~Size | Machine class |
|---|---|---|
| `q2-imatrix` | ~81 GB | 96/128 GB RAM |
| `q2-q4-imatrix` | ~98 GB | 128 GB MacBooks (higher quality) |
| `q4-imatrix` | ~153 GB | ≥256 GB |
| `pro-q2-imatrix` | ~430 GB | 512 GB |
| `pro-q4-layers00-30` + `pro-q4-layers31-output` | ~838 GB | two-machine distributed |

Supporting tooling:

- **`gguf-tools/`** — offline pipeline: a plain-C HF-safetensors→GGUF quantizer,
  the routed-MoE **imatrix** collector (Metal-only; accumulates per-expert
  activation importance), and a quality scorer computing per-token NLL against
  official DeepSeek-API continuations.
- **`dir-steering/`** — runtime activation-edit feature (a 43×4096 direction file
  to suppress/amplify a behavior).
- **`QA_BEFORE_RELEASES.md`** — release gate with named hardware hosts per backend.

---

# DGX Spark / GB10 — the details

The DGX Spark is the **NVIDIA GB10 "Grace-Blackwell" unified-memory ARM (aarch64)
desktop box**: a Blackwell-class GPU + Grace ARM CPU on one package sharing
**~128 GB of unified LPDDR memory (UMA), with no separate VRAM pool**. The README
calls it out as a first-class backend: *"NVIDIA CUDA / DGX Spark, CUDA with special
care for the DGX Spark."*

The engine cares because 128 GB of unified memory lets one box hold the large
Flash quants (IQ2 ~81 GiB, mixed q2/q4 ~91 GiB) that no discrete GPU could fit in
VRAM — but UMA breaks the assumptions CUDA code normally makes.

> Note: the GB10 hardware description (ARM superchip, on-package GPU+CPU, LPDDR) is
> industry background. What the repo itself asserts verbatim is "DGX Spark / GB10",
> "128 GB", "UMA / unified-memory machine", "Spark-class systems", and the
> aarch64/sbsa-linux build path.

## Why Spark needs special care

The whole problem is the **Unified Memory Architecture**. On a discrete GPU,
`cudaMemGetInfo()` "free" bytes are dedicated VRAM you own. On Spark that free
number looks huge (~128 GB) but is **shared** with the OS, the model mmap, page
cache, and the KV cache. The code says it directly (`ds4_cuda.cu:592`):

> *"On 96/128 GB UMA Spark-class systems the expanded Q8→F16 cache can pass a simple
> free-memory reserve check but still leave too little room for long-prefill cuBLAS
> execution."*

Two hazards follow:

1. **The UMA free-memory trap** — a naïve "if it fits in free, allocate it" check
   passes yet you have actually starved the shared pool.
2. **cuBLAS long-prefill scratch contention** — long prefills need big transient
   cuBLAS workspaces that must coexist with cached weights.

And a third failure mode: a very large **device-only `cudaMalloc` for KV can make
the whole unified-memory machine unresponsive** (`ds4_cuda.cu:2386`). So the
engine's entire strategy is: *"the free number lies on UMA — budget by model size
and total RAM, not by free bytes."*

## What the build does differently

- **No explicit `nvcc -arch`.** `make cuda-spark` builds with `CUDA_ARCH` *empty* —
  README says omitting `-arch` "is currently the fastest path on GB10"
  (`README.md:1184`). (`cuda-generic` uses `native`; `make cuda` requires explicit
  `sm_N`.)
- **aarch64/SBSA lib path baked in.** `CUDA_LDLIBS` puts
  `-L$(CUDA_HOME)/targets/sbsa-linux/lib` ahead of `lib64` (`Makefile:33`) — the
  ARM SBSA CUDA libraries Spark needs.

## Memory model — what fits and what doesn't

The default device-resident weight-cache budget is **96 GiB**
(`cuda_model_cache_limit_bytes`, `ds4_cuda.cu:1121`). Its comment is the clearest
statement of intent:

> *"One Spark can run the IQ2 model (~81 GiB) and the mixed q2/q4 model (~91 GiB)
> via the startup tensor cache… make the full-Q4 model use distributed layer
> loading unless the operator opts into a larger cache budget explicitly."*

So:

- ✅ **`q2-imatrix` (~81 GB)** — the reliable single-Spark choice.
- ⚠️ **`q2-q4-imatrix` (~98 GB)** — *"Works on DGX Spark but loading may struggle
  compared to q2-imatrix"* (`download_model.sh:44`).
- ❌ **Full Q4 (~153 GB)** — intentionally refused on one Spark; the engine prints
  a message telling you to use distributed layer loading or raise the cap
  (`ds4_cuda.cu:2599`).

Several caches are bounded specifically to protect the shared pool:

- The **Q8→F16 weight cache** is size-tiered on UMA (`ds4_cuda.cu:596`): model
  registered ≥112 GiB → **4 GiB** cap; ≥88 GiB → **16 GiB**; range ≥64 GiB →
  **12 GiB**; else **8 GiB**. It reserves only 512 MiB on ≥112 GiB boxes, else
  `max(4 GiB, 5%)`.
- **KV caches ≥8 GiB** are routed to managed/demand-paged memory so a giant
  device-only alloc cannot freeze the box.
- The model mmap is registered in-place via `cudaHostRegister(Mapped|ReadOnly)` so
  the GPU reads weights out of unified memory rather than copying them — there is
  even a page-aligned tail-registration special case for the 88–96 GiB Flash quant
  class (`ds4_cuda.cu:296`).
- **TF32 tensor-op cuBLAS math** is on by default (good for Blackwell prefill).

## Measured performance

The release table row (`README.md:175`), under standard conditions (single-run
CLI, `--ctx 32768`, `--nothink`, greedy, `-n 256`):

| Machine | Quant | Prompt | Prefill | Generation |
|---|---|---|---|---|
| **DGX Spark GB10, 128 GB** | q2 | 7047 tokens | **343.81 t/s** | **13.75 t/s** |
| MacBook M3 Max, 128 GB | q2 | 11709 tokens | 250.11 | 21.47 |
| MacBook M5 Max, 128 GB | q2 | 11707 tokens | 463.44 | 25.90 |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 | 27.39 |

**The shape to notice:** Spark's *prefill* (343 t/s) is competitive — between M3
Max and the M5/M3 Ultra — because prefill is compute/cuBLAS-bound and benefits from
TF32. But Spark's *generation* (13.75 t/s) is markedly lower than every Mac.

> The codebase does **not** state a single verbatim cause for the low decode
> number, so this is not asserted as fact. Decode is purely autoregressive and
> memory-bandwidth-bound (streaming weights one token at a time), and the README
> notes generally that memory latency/bandwidth pull decode down directly. The most
> plausible read is that GB10's LPDDR has lower effective bandwidth for per-token
> weight streaming than Apple's wide unified memory — plausible, but not stated
> explicitly for Spark in this repo.

## Environment variables (Spark-relevant)

| Var | Effect |
|---|---|
| `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` | Caps the device-resident weight cache (default **96 GiB**). Also the explicit opt-in to load a >96 GiB model on one Spark instead of being pushed to distributed. Never drops the effective limit below 96 GiB. |
| `DS4_CUDA_Q8_F16_CACHE_MB` | Hard cap (MiB) on the Q8→F16 cache; **bypasses the UMA auto-tiering** when set. `0` disables the F16 cache. |
| `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` | Overrides the device-memory reserve kept free for cuBLAS workspaces. |
| `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` | Reserve kept free when sizing the streaming-expert cache (default **16 GiB**). |
| `DS4_CUDA_NO_TF32` | Disables TF32 tensor-op math (also implied by quality mode) — slows prefill. |
| `DS4_CUDA_HOST_REGISTER_PLAIN` | Drops the `ReadOnly` flag on model registration if a driver rejects it. |
| `DS4_CUDA_COPY_MODEL` | Forces a full H2D copy — **undesirable on Spark** (duplicates the model in the shared pool). |
| `DS4_CUDA_NO_Q8_F16_CACHE` / `DS4_CUDA_Q8_F16_ALL` / `DS4_CUDA_STRICT_WEIGHT_CACHE` / `DS4_CUDA_WEIGHT_CACHE_VERBOSE` | Disable / widen / strict-fail / verbose-log the weight cache. |

## Build, run, and the release gate

```sh
make clean && make cuda-spark    # the exact release build step
make cuda-regression             # runs tests/cuda_long_context_smoke (indexer top-k)
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf --ctx 4096 --nothink -p "Reply with exactly: OK"
```

`--cuda` is the default backend in a CUDA build. Per `QA_BEFORE_RELEASES.md` §6,
**you may not claim CUDA is release-ready without running the pass on the actual
GB10 host** (`toor@192.168.0.180`): push the exact commit, `make clean && make
cuda-spark`, `make cuda-regression`, a short Flash-GGUF prompt (record gen t/s),
and a longer prompt exercising routed experts past a few thousand tokens.

## Gotchas to keep in mind on Spark

- **The free-memory trap is real** — do not assume discrete-GPU "if it fits in
  free, allocate it"; that is exactly why the size-tiered caps exist.
- **Raising the caps re-exposes the OOM/freeze risk** — a too-large device-only
  KV/weight alloc can make the box unresponsive.
- **A single OOM silently disables the F16 acceleration cache** for the rest of the
  run (it falls back to q8 kernels, `ds4_cuda.cu:626`).
- **`make cuda-regression` is CUDA-only** — on macOS it just prints a message, so
  it cannot substitute for testing on the real Spark.

---

# Combining two DGX Sparks

> Deeper analysis — tensor/expert parallelism feasibility, published interconnect
> benchmarks, and the Phase-0 latency test — lives in
> [DUAL_SPARK_NOTES.md](DUAL_SPARK_NOTES.md).

**Short version:** yes, ds4 can run across two Sparks — via its **distributed
inference** (layer-split over TCP), which is backend-agnostic and which the CUDA
path explicitly points you to for models too big for one box. It is the *generic*
two-machine mechanism, **not** a Spark-specific NVLink/RDMA cluster feature.

## What it's for

Two Sparks = **256 GB combined unified memory**. The natural target is the **full
Flash Q4 quant** (`q4-imatrix`, ~153 GB) — too big for one Spark's 96 GiB cache
budget, fits as ~76 GB per machine across two. That gives you the **best Flash
quant** (higher quality than the q2 a single Spark runs). The other benefit is
**faster long-prefill** (the pipeline gave 1.4–1.85× in the README's two-machine
table). Note: PRO won't fit — PRO Q2 (~430 GB) and PRO Q4 (~838 GB) need the 512 GB
Mac Studios, not 256 GB of Spark.

The CUDA backend itself directs oversized models here — when a model exceeds the
96 GiB single-GPU budget it refuses and prints (`ds4_cuda.cu:2599`):

> `ds4: CUDA model X GiB exceeds the default single-GPU startup cache budget
> 96 GiB; use distributed layer loading or set DS4_CUDA_WEIGHT_CACHE_LIMIT_GB
> explicitly`

## How it works

- **Layer split.** Each Spark loads only its slice (`--layers`), so the full
  ~153 GB GGUF goes on both but each maps ~half. Communication is worker-to-worker
  (`A -> B -> back to A`); the coordinator keeps normal CLI/API behavior.
- **Backend-agnostic transport.** `ds4_distributed.c` has no Metal-only gating; it
  is plain TCP via `getaddrinfo`/`connect`, so it runs on the CUDA/Spark build.
- **Correctness.** A rolling 64-bit token-prefix hash is checked before and after
  each span; a restarted worker can't silently accept stale work.
- **Decode is slower than one machine.** Generation is strictly autoregressive —
  one cross-machine hop per token (~19% loss even on Thunderbolt 5). The win is
  capacity + prefill, not faster decode.

## Concrete launch (full Flash Q4 across two Sparks)

Both Sparks: `make cuda-spark` from the **same commit**, full `q4-imatrix` GGUF on
each. Flash has 43 layers (0..42); a balanced split is `0:20` / `21:output` (the
worker owns the output head and returns logits directly).

```sh
# Spark A — coordinator: layers 0..20, owns tokenizer/sampling, listens
./ds4 -m gguf/DeepSeek-V4-Flash-Q4KExperts-...-imatrix.gguf \
  --role coordinator --layers 0:20 --listen <A_LINK_IP> 1234

# Spark B — worker: layers 21..output, connects to A
./ds4 -m gguf/DeepSeek-V4-Flash-Q4KExperts-...-imatrix.gguf \
  --role worker --layers 21:output --coordinator <A_LINK_IP> 1234
```

Use the coordinator exactly like a normal `./ds4` (chat, `/read`, server, agent all
go through it). `<A_LINK_IP>` must be the **stacking-link** IP (see below).

## The stacking cable and networking

The bundled **"DGX Spark Stacking Cable QSFP/CX7" (QSFP112 DAC, ~200 Gb/s,
RoCE/RDMA-capable)** wires the two Sparks' ConnectX-7 NICs directly together. It is
a *better* link than the Thunderbolt 5 setup every distributed number in the README
was measured on — so the network is not your concern.

**ds4 uses it as plain TCP, not RDMA.** The repo has **zero** RDMA/RoCE/verbs/
UCX/NCCL/SMC code — it is BSD sockets only, with `TCP_NODELAY` set
(`dist_set_socket_low_latency`, `ds4_distributed.c:1027`). Consequences:

- **Bandwidth: overkill (good).** Per-hop activations are tiny — ~16 KB for one
  decode token, tens of MB per 4096-token prefill chunk. At ~200 Gb/s that's
  microseconds to low-ms. Bandwidth never binds → **keep fp32 activations; do not
  use `--dist-activation-bits`.**
- **Latency: excellent.** A direct DAC + `TCP_NODELAY` gives sub-ms RTT, in the
  class of the README's TB5 row (0.45 ms ping → **25.09 t/s** generation on the
  91 GB Flash quant). Expect distributed decode in that ballpark.
- **RDMA is the only unused capability — and it wouldn't help here.** Per token
  ds4 pays one ~16 KB hop (tens of µs over TCP; single-digit µs with RDMA) against
  a per-token *compute* of milliseconds (README two-machine PRO Q4 telemetry:
  ~84–92 ms/token). The network hop is **<1% of the per-token budget**, so RDMA
  would shave a fraction of a percent off decode. On Spark's **UMA**, GPUDirect
  RDMA's usual win (skip the host bounce to VRAM) is also moot — CPU and GPU share
  the same LPDDR.

## Tuning that actually matters

- **Be on the link.** Give the stacking interface its own IPs (link-local
  `169.254.x.x` or a static `/30`); point `--listen` / `--coordinator` at those —
  **not** the management Ethernet/Wi-Fi IPs. `ping` to confirm <1 ms.
  - Caveat: `DS4_DIST_CONNECT_BIND_IF` interface-pinning is **macOS-only** (relies
    on `IP_BOUND_IF`; returns `ENOTSUP` on Linux, `ds4_distributed.c:1308`). On the
    Sparks, steer traffic by using the link's IPs, not that env var.
- **MTU 9000 (jumbo frames)** on the interface — fewer packets per 67 MB prefill
  chunk → better prefill throughput. (OS setting, not a ds4 flag.)
- **`DS4_DIST_SOCKET_BUFFER_MB`** — default **128**, max **512**; bump it for big
  prefill chunks on a fat link (`ds4_distributed.c:710`).
- **Balance the `--layers` split** and let the last worker own `:output`. Use
  `--debug` on the coordinator for per-hop telemetry to check the split is balanced.

## Optional: forcing RDMA (experimental, low payoff)

- **Transparent shim, no ds4 code:** `smc_run ./ds4 ...` (SMC-R) or
  `LD_PRELOAD=librspreload.so` (rsocket) can carry ds4's TCP over RoCE. Both need
  the RDMA stack configured on the CX7 and are finicky with ds4's threaded +
  non-blocking sockets; expect an **unmeasurable** win given the <1% headroom.
- **Native RDMA transport, real code:** the transport is centralized behind the
  `DS4D` framing, so a UCX/libibverbs backend could plug in without touching the
  engine — but the expected decode gain is the same fraction of a percent. Hard to
  justify on performance grounds.

## Caveats

- **Not separately benchmarked.** All published distributed numbers are Macs;
  `QA_BEFORE_RELEASES.md` lists only **one** CUDA host (`toor@192.168.0.180`) and
  treats CUDA distributed as a conditional test (§8) — so the dual-Spark path is
  supported by the generic mechanism but is a less-exercised road with no published
  two-Spark t/s figure.
- **Same commit on both**, trusted network only — the distributed protocol has **no
  auth/encryption**.
- **Decode is inherently slower** than a single machine; the cable removes the
  network as a concern but cannot remove the autoregressive cross-machine hop.

---

## Key source-file map

| File | Role |
|---|---|
| `ds4.h` | Public engine/session API (`ds4_engine`, `ds4_session`, options, tokenize/chat, disk payload contract). |
| `ds4.c` | Core engine (~28k lines): GGUF parse/mmap, quant kernels, CPU reference forward, MoE FFN, KV cache, GPU graph driver, MTP speculative decode, tokenizer/chat. |
| `ds4_gpu.h` | Backend-agnostic GPU ABI (~120 `ds4_gpu_*` functions). |
| `ds4_metal.m` | Metal backend (primary); also the SSD-streaming expert cache. |
| `ds4_cuda.cu` | CUDA backend + **all DGX Spark / GB10 special handling**. |
| `ds4_rocm.cu` / `ds4_rocm.h` / `rocm/*.cuh` | ROCm backend (HIP port) and the CUDA→HIP shim. |
| `ds4_cli.c` | `ds4` CLI/REPL and runtime backend selection. |
| `ds4_server.c` | HTTP server (OpenAI/Responses/Anthropic/completions), SSE, tool-call replay. |
| `ds4_agent.c` / `ds4_web.c` | Coding agent (alpha) and its Chrome-DevTools browser automation. |
| `ds4_bench.c` / `ds4_eval.c` | Throughput benchmark and capability-eval harness. |
| `ds4_distributed.c/.h` | Distributed inference session backend (TCP `DS4D` protocol, layer split). |
| `ds4_ssd.c/.h` | SSD-streaming budget math (cache machinery itself lives in `ds4_metal.m`). |
| `ds4_kvstore.c/.h` | Disk KV-cache persistence (SHA1 byte-prefix files). |
| `rax.c/.h` | Vendored radix tree — used only by the server for tool-call replay. |
| `Makefile` | Backend selection (`cuda-spark`, `cuda-generic`, `strix-halo`, `cpu`, `test`, …). |
| `download_model.sh` | GGUF downloader + `ds4flash.gguf` symlink. |
| `gguf-tools/` | Offline quantizer, imatrix collector, quality scorer. |
| `dir-steering/` | Runtime directional activation steering. |
| `QA_BEFORE_RELEASES.md` | Per-backend release gate with named test hosts. |
