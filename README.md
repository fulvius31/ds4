# twinspark-GranLanguageModel

**GLM 5.2 at 2-bit, running across two NVIDIA DGX Sparks.** (`twinspark-glm`.)

*Twin Spark*, as in Alfa Romeo's twin-ignition engines — two plugs per cylinder
firing together. Here it is two DGX Sparks firing as one: a two-box pipeline
across a direct 200GbE ConnectX link, splitting a **196.58 GiB** model over a
pair of **121.69 GiB** boxes — 40 layers on the head, 38 on the worker.

And *GLM* is the model's name, but here it reads as **Gran Language Model** — in
the Gran Turismo tradition: not the biggest engine on the road, but the one
built to cross a continent, or a million tokens of context, without stopping.

The pipeline speaks plain TCP over that link. RDMA/RoCE exists in this tree only
as the tensor-parallel gate transport, and TP is parked — see below.

A downstream distribution of [antirez's DwarfStar
(ds4)](https://github.com/antirez/ds4) (MIT). Binaries, flags and engine keep
their upstream `ds4` names; this repo tracks upstream and merges regularly.

## Why this exists

GLM 5.2 at IQ2_XXS is **196.58 GiB** against **121.69 GiB** per Spark: a large
model crushed to ~2-bit routed experts and it **still** does not fit in one
box. That is what SSD streaming, the expert LRU pool and the two-box layer
split exist to solve, and why the full 1,048,576-token window is usable here
at all. The KV cache is packed fp8 (e4m3 + per-row scales), which is what
makes the window affordable.

## Measured performance

Two regimes, and they differ by more than an order of magnitude. **Cold** means
a freshly started server whose expert pool is empty; **warm** is steady state
deep into a long ingest. Short benchmarks land in the first and systematically
understate the engine.

| | prefill | decode |
|---|---:|---:|
| cold, 4,060-token prompt | **41.0 t/s** | **2.69 t/s** |
| warm, ≥100k filled | 77 → 38 t/s (see curve) | 1.18 t/s @98k · 0.59 @1M |

Prefill decays with filled context, roughly **10 t/s per 200k — linear, not
quadratic**, because the DSA sparse-indexer scan sets the pace rather than
attention. Sampled across one 1,006,595-token ingest:

| filled | 100k | 200k | 332k | 537k | 840k |
|---|---:|---:|---:|---:|---:|
| prefill | 77 | 67 | 58 | 48 | 38 t/s |

*(That curve and the warm decode figures predate the I/O work below and are
therefore pessimistic; they have not been re-measured at length.)*

**Validated, not vibes.** Needle-at-depth retrieval is exact at 100k, 190k, 350k
and at a full **1,006,595 tokens** under fp8 — retrieved from 50% depth out of
repeated text where the index had near-identical keys to discriminate between
(5.68 h prefill, answer exact first try). temp-0 output is byte-identical across
cold starts, verified through every optimization here including all of the I/O
work. fp8 costs no speed and no accuracy: rates match f16 to the second decimal
and perplexity is a statistical tie (8.765 vs 8.776 over 2,000 scored tokens).

### What limits it: SSD read bandwidth

Neither prefill nor decode is compute-bound under `--ssd-streaming` — both wait
on the disk. A 4,060-token prefill reads **173 GiB** on the leader alone; 79
decoded tokens read **58 GiB**. So throughput is a straight function of how much
of the NVMe the engine can actually use, and it was using very little:

| access pattern | throughput |
|---|---:|
| sequential, O_DIRECT | 5.0–6.8 GB/s |
| random 9.5 MiB reads (the expert pattern) | 5,545 MiB/s |
| what the engine extracted, before | **1,337 MiB/s** |
| after the work below | **1,751 MiB/s** |

Three problems, found by instrumenting requested bytes and comparing against
`/proc/diskstats`:

1. **Queue depth 1.** The staged path issued one synchronous `pread` per tensor
   per expert, each interleaved with its own H2D submit. `ds4_metal.m` has had a
   threaded pread pool for years; CUDA had four `pread` references to Metal's
   238. → `DS4_CUDA_PREAD_POOL=1`.
2. **Every byte read twice.** `madvise(MADV_WILLNEED)` prefetched every expert
   into a page cache that the O_DIRECT read path never consults, then read it
   again — exactly 2× the physical I/O. Now skipped when the direct fd is live.
3. **No overlap.** Each wave ended in a `cudaStreamSynchronize`, so the drive
   idled through every upload. Now double-buffered: wave *w+1*'s reads are
   submitted before wave *w*'s uploads.

Cumulative, identical 4,060-token cold prefill:

| | prefill | disk read |
|---|---:|---:|
| before | 201.6 s — 20.1 t/s | 345 GiB |
| + pread pool | 154.9 s — 26.2 t/s | 343 GiB |
| + readahead fix | 107.8 s — 37.7 t/s | 173 GiB |
| + double buffering | **99.1 s — 41.0 t/s** | 173 GiB |

**2.03× on half the disk traffic**, with byte-identical greedy output at every
step. Decode gained 1.79 → 2.69 t/s from the first two (its misses fit one wave,
so double-buffering does nothing for it).

Notes: 8 threads is the optimum *after* double-buffering (16 won before it; 24
regresses). `read_ahead_kb` is **not** a lever — `madvise` issues its own
readahead and ignores it (128 vs 4096 KB: 135.1 s vs 134.8 s, same bytes).
`DS4_CUDA_EXPERT_POOL_STATS=1` and `DS4_CUDA_STREAM_LOAD_STATS=1` print live hit
rates and per-layer staging.

Pool size competes directly with the KV plane: at 1M the KV costs 26.12 GiB that
would otherwise hold ~2,200 more expert slots. Avoiding a read beats
accelerating one, so trading window for pool is a real option.

### Pipeline vs tensor parallel — TP loses

Matched allocation (500,000), identical 97.9k prompt:

| | prefill | decode | 98k request |
|---|---:|---:|---:|
| tensor parallel | 52.5 t/s | **1.26 t/s** | 1,897 s |
| **pipeline** | **77.2 t/s** | 1.18 t/s | **1,302 s** |

TP wins decode by 7% and loses prefill by 47%. Prefill is **99.95% of a large
request** (the 1M run: 20,430 s prefill, 10.1 s decode), so that never pays. TP
also caps at half the context because it **replicates** the KV plane
(`kv_layers=78` on *both* ranks) where pipeline **splits** it (40/38): GLM's DSA
attention compresses KV into a per-token latent with no head dimension to shard.

TP is faster only on short prompts. It stays in tree, runtime-gated and inert;
the ds4-server integration is parked on `tp-cuda-server`.

### Maximum context

Pushed until a **real request** stopped surviving — allocation is not capacity.
TP allocates ctx 750,000 and then dies mid-request with 0.26 GiB free.

| | max ctx serving a request | free at floor |
|---|---:|---:|
| TP + streaming + fp8 | 500,000 | 1.9 GiB |
| **pipeline + streaming + fp8** | **1,048,576** (the model's max) | 25.3 GiB |

The memory guard gates on a *planned* KV figure assuming full layers, so it
over-reports pipeline ~2×; the full window needs
`DS4_GLM_MEMORY_GUARD_RESERVE_GB=8` on both ranks, which 25 GiB of measured
headroom justifies.

### Streaming is a prerequisite, not an optimisation

Keeping every weight resident (96 GiB/box) sounds faster — nothing is fetched.
It is not, and the reason is memory, not I/O:

| ctx | largest reliable prompt | prefill | decode |
|---:|---|---:|---:|
| 25,000 | ~4k | 14 t/s | ~3.5 t/s |
| 50,000 | ~4k (marginal) | 14–18 t/s | 3.55 t/s |
| 100,000 | **none — dies on a 2k prompt** | — | — |

Grouped-GEMM prefill dequantises IQ2_XXS to f16 and needs multi-GiB of transient
scratch; resident weights leave about **1 GiB**. So a resident config is stuck
forever on ALU-bound IQ2 dot kernels.

```
streaming → weights 96 → 22 GiB → ~74 GiB freed
          → grouped-GEMM prefill affordable → 77 t/s instead of 14
```

Resident still wins *decode* (3.55 vs 1.18 t/s) — the crossover is
**prompt < ~10.5 × generated tokens**. Short question, long answer is a chat
profile; this project ingests documents and sits on the other side of that line.

## What it is for — and what it is not

**Not for realtime work.** Decode is 1–3 t/s and long prompts take minutes.
Nothing about that improves with tuning; it is what a 2-bit 196 GiB model on two
121 GiB boxes costs.

**This is for work where the answer is worth minutes and nobody is waiting:**
reviewing a diff, auditing a subsystem, reading a design document end to end, a
nightly pass over the day's changes. The cost is paid once per artifact rather
than per keystroke, and the whole 1M window holds the artifact plus everything
it depends on.

Operationally it earns its keep as a **second, independent reviewer** —
different vendor, different architecture, uncorrelated blind spots — and
notably disciplined about *stating* a defect with a reproducer instead of
quietly rewriting the code.

## Quick start

Two DGX Sparks, direct ConnectX link, MTU 9000 — jumbo frames matter because
the pipeline is TCP, not RDMA. After every reboot,
on **both** boxes: `sudo cpupower idle-set -D 100` (deep idle states add ~1 ms
per gate exchange, and the setting does not survive a reboot).

```bash
./download_model.sh glm-antirez-iq2xxs     # 196.6 GiB, on BOTH boxes
./run_glm_1M.sh                            # both ranks + API, health-gated
```

`run_glm_1M.sh` is the supported path: it starts the **worker first** (it dials
the coordinator and retries), then the leader, then verifies the worker actually
joined rather than trusting a `/v1/models` reply. Defaults to pipeline + SSD
streaming + fp8 KV at ctx 1,048,576 with the pread pool on, listening on
`0.0.0.0:8020` — a `127.0.0.1` listener is invisible from inside a container.

```bash
GLM_CTX=200000 ./run_glm_1M.sh      # narrower window, faster decode
POOL=6000 ./run_glm_1M.sh           # bigger expert pool, fewer misses
```

Both ranks must agree on `GLM_CTX` and on `DS4_GLM_FP8_KV_STORE`.

| GLM_CTX | POOL | notes |
|---:|---:|---|
| 8,192 | 8,000 | fastest decode, chat |
| 100,000 | 5,000 | needle-validated at depth |
| 200,000 | 5,000 | good window/decode balance |
| 400,000 | 5,000 | needle-exact at 350k |
| 1,048,576 | 5,000 | the model's max; needs `RESERVE=8` |

### Serving an API

`ds4-server` speaks the OpenAI API (`/v1/models`, `/v1/chat/completions`, SSE),
so the whole envelope is reachable from Open WebUI, LiteLLM or any SDK. Measured
through the server at ctx 1,048,576, cold pool:

| | measured |
|---|---|
| 4,145-token request (Open WebUI, tools on) | 102 s prefill — 40.5 t/s |
| short prompt, end to end | ~9 s |
| 3 concurrent requests | serialise cleanly, no deadlock |

Three ids are advertised — `glm-5.2`, `glm-5.2-chat` (thinking off) and
`glm-5.2-reasoner` (thinking on). Same weights and session; they only preset the
thinking mode, and an explicit `thinking` field overrides them.

**Trim your client's preamble before blaming the engine.** Open WebUI resends a
system prompt plus tool schemas every turn: a bare "ciao" arrived here as
**4,145 tokens**, so ~102 s of the 110 s round trip was spent on schemas for
tools that went unused. Disabling unused tools is the single largest
interactive-latency lever available, and it is a UI setting, not an engine one.

**The KV disk cache stays off.** `--kv-disk-dir` restores a snapshot into the
coordinator's session without telling the worker; the ranks desync on the next
gate. In-memory prefix reuse within one load is fine.

## What differs from upstream

- **SSD-streaming I/O path on CUDA** — the parallel expert pread pool
  (`DS4_CUDA_PREAD_POOL=1`), the O_DIRECT readahead fix, and double-buffered
  banks. **2.03× prefill on half the disk traffic**; see the table above. This
  is the most portable work here: it affects any CUDA user of `--ssd-streaming`,
  not just this two-box setup.
- **Expert LRU pool** for streamed routed experts — upstream's CUDA backend has
  the API but stubs it (`configured_count()` returns `0`); Metal implements it
  fully. Decode-protected against batch-prefill eviction, direct pool reads in
  the decode kernels, batched miss uploads, eager pre-grow under a pressure
  guard.
- **Packed fp8 KV cache** (`DS4_GLM_FP8_KV_STORE=1`): e4m3 rows with per-row
  absmax scales, ~54% of f16 — this is what buys the 1M window. Decode reads go
  through a dequantizing gather, batch prefill through an f16 unpack, so every
  attention kernel runs unmodified. Also an **f16 compact KV cache**
  (`DS4_GLM_COMPACT_CACHE_F32=1` restores f32 for A/B).
- **Two-box CUDA tensor parallelism** — the full Metal TP gate contract on CUDA
  over TCP or RDMA, with attention and shared-expert splits at decode. Present,
  runtime-gated, **not the default and not recommended for GLM 5.2**; the
  ds4-server integration lives on `tp-cuda-server`.
- **KV session save/restore across the pair** (`--kv-save`/`--kv-load`): kill
  both ranks, relaunch, continue with only new tokens prefilled. Format-portable
  between fp8 and f16 runs. `-n 0` is a generation-free pass-through for session
  surgery.
- **Batched GEMM prefill** with the multi-chunk indexer fix that makes long
  prompts byte-correct.
- **Model ids that say what they do** — the aliases all advertised the engine's
  own name, so a model picker showed three identical entries.
- **Server fixes** for GLM thinking mode (unclosed reasoning surfaces as
  `reasoning_content`, never as content), ported from upstream's PR queue with
  credit (#524; #158 in part; #460).
- **A GB10 operations runbook** learned the hard way: unified-memory OOM freezes
  the box rather than killing the process, so this repo ships memory guards,
  planned-budget envelopes and loud diagnostics where silent failures used to
  live.

## Roadmap

- **Find the new bottleneck first.** Prefill is no longer purely storage-bound
  at these sizes. Cutting a 16k prefill's disk reads by **39.6%** (429 → 259
  GiB, by widening spans from 4096 to 8192 tokens so the layer sweep runs 3
  times instead of 5) bought only **6.9%** of wall clock — and the disk ran
  *slower* while doing it (2.10 → 1.36 GiB/s), i.e. the time went somewhere
  else. Reading less is no longer the lever it was before the I/O work; profile
  compute before optimising I/O further.
- Reads are still 1.75 against the ~5.5 GiB/s the device delivers, if that
  turns out to matter: per-tensor reads are 3.09 MiB where 9.5 MiB reads hit
  full speed, and experts are contiguous within `ffn_gate_exps`, so several
  could be coalesced into one `pread`.
- Re-measure the long-context fill curve, which still reflects pre-I/O-work
  numbers and is now pessimistic
- MTP on the pipeline: speculative cycles work end to end behind `--glm-mtp`;
  making it a *win* needs per-draft routed-expert upload and a fused two-token
  verify
- Fix the RC-bulk RDMA memory-region lifecycle so the faster bulk posting works
  under repeated large prefills instead of being bypassed by
  `DS4_TP_NO_RC_BULK=1`
- Make the server's KV prefix cache mirror-aware so it can be re-enabled
- Verify a large prompt end to end *through the server* — 11,170 tokens is the
  largest confirmed; the CLI is needle-exact far beyond that

## Credits

Built on [DwarfStar (ds4)](https://github.com/antirez/ds4) by Salvatore
Sanfilippo (antirez) — the engine, the models, the philosophy. Several fixes
ported from upstream's open PR queue with thanks to their authors. And to Alfa
Romeo, for naming the architecture thirty years early.

MIT, same as upstream.
