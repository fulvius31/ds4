# twinspark-GranLanguageModel

**GLM 5.2 at 2-bit, running interactively across two NVIDIA DGX Sparks.**
(`twinspark-glm` for short.)

*Twin Spark*, as in Alfa Romeo's twin-ignition engines — two spark plugs per
cylinder, firing together for a cleaner, stronger burn. Here it is two DGX
Sparks firing as one: tensor parallelism over a direct RoCE link, serving a
196.6 GiB model on a pair of 121.69 GiB boxes.

And *GLM* is the model's name, but around here it stands for **Gran Language
Model** — in the Gran Turismo tradition: not the largest engine on the road,
but the one built to cross a continent (or 400,000 tokens of context) in
comfort.

This is a downstream distribution of [antirez's DwarfStar
(ds4)](https://github.com/antirez/ds4) (MIT), focused on exactly one thing.
The binaries, flags, and engine keep their upstream `ds4` names; this repo
tracks upstream and merges regularly.

## Why this project exists

On this same hardware, vLLM serves DeepSeek V4 Flash superbly — we measure
~1,035 t/s prefill and 23.5 t/s decode at 906k tokens of a 1M context. If
Flash is the model you want, use vLLM.

This project is for the model that **does not fit**: GLM 5.2 quantized to
IQ2_XXS is 196.6 GiB against 121.69 GiB of unified memory per Spark. ds4's
SSD streaming, an expert LRU cache, and two-box tensor parallelism make it
not just runnable but interactive — with month-scale reliability discipline
(byte-reproducible outputs, memory guards, validated envelopes) built in.

## Measured performance (2× DGX Spark GB10, RoCE v2)

| Mode | Prefill | Decode | Context |
|---|---:|---:|---|
| **TP + SSD streaming** (the flagship) | 58.7 t/s | 3.3 t/s @45k, 2.66 @190k | up to 200,000 (f16 KV) |
| **TP + fp8 KV cache** (opt-in) | 52.8 t/s @190k, 46.5 @350k | 2.66 @190k, 2.39 @350k | **500,000** measured max |
| Pipeline resident (short-context daily driver) | 90 t/s | 5.9 t/s | ≤ 12,288 |
| **Pipeline + SSD streaming + fp8** (widest window) | ~68 t/s | 2.7 t/s | **1,048,576** — the model's own max |

Validated, not vibes: needle-at-depth retrieval is **exact at 100k, 190k and
350k** (the last one under fp8 at ctx 400,000 — 19.2 GiB of KV where f16
would need 35.5 and not fit), temp-0 outputs are byte-identical across cold
starts (verified through every optimization), and every envelope allocates
and runs inside the same ~90 GiB planned budget the memory runbook proves
safe. Decode at 45k went **2.23 → 3.29 t/s (+48%)** across this fork's
optimization cycles, and fp8 costs none of it: 45k and 190k rates match f16
to the second decimal, and the perplexity fixture scores a statistical tie
(8.765 fp8 vs 8.776 f16 over 2,000 scored tokens).

### Maximum context, measured

Both streaming modes were pushed until a **real request** stopped surviving —
allocation is not capacity. TP happily allocates ctx 750,000 (108 GiB planned)
and then dies mid-request with 0.26 GiB free, killed by the memory guard.

| Mode | Max ctx that serves a request | Prefill | Decode | Free at floor |
|---|---:|---:|---:|---:|
| TP + SSD streaming + fp8 | 500,000 | 50.4 t/s | 1.86 t/s | 1.9 GiB |
| Pipeline + SSD streaming + fp8 | **1,048,576** (the model's own max) | 49.4 t/s | 1.13 t/s | 25.3 GiB |

Same 11,082-token prompt on both, so the rates are comparable.

**Pipeline reaches twice the context because KV is split, not replicated.**
Under TP each rank carries the full plane (`kv_layers=78` on both) — 19.2 GiB
per box at 400k, 36 GiB at 750k. Pipeline splits the layers (`kv_layers=40`
leader / `38` worker), so each box holds only its slice: 26.1 GiB per box at
the full 1M window, with 25 GiB still free. TP's replicated KV is what runs it
out of memory first.

Two things to know about the ceiling. At maximum context **TP's prefill
advantage disappears** (50.4 vs 49.4 t/s; at small contexts it is 57 vs 23) —
both become bound elsewhere, and TP keeps only a decode edge. And the guard
gates on a *planned* KV figure that assumes full layers, so for pipeline it
over-reports ~2× (43.2 GiB claimed at 900k, 22.4 GiB actually allocated); the
full 1M window needs `DS4_GLM_MEMORY_GUARD_RESERVE_GB=8` on both ranks to get
past it, which the measured 25 GiB of headroom justifies.

### Prefill throughput vs context fill

Sampled during a single **1,006,595-token** ingest (pipeline + SSD streaming +
fp8, ctx 1,048,576, 4096-token chunks):

| context already filled | prefill |
|---:|---:|
| 100k | 77 t/s |
| 200k | 67 t/s |
| 332k | 58 t/s |
| 537k | 48 t/s |
| 840k | 38 t/s |

Running average **53 t/s across the first 840k tokens**. The decay is close to
**10 t/s per 200k of context — linear in filled context, not quadratic**: what
sets the pace is the DSA sparse-indexer scan, not attention. Dense attention
would be roughly an order of magnitude slower by 840k.

Two practical consequences. A full million-token ingest is a **~5 hour**
operation, so the widest window is a batch tool, not something you sit in front
of. And the first chunk of any request reads far slower than steady state
(25 t/s vs 77 t/s here) because the expert pool is still warming — short
benchmarks systematically understate this engine.

### API server, measured

`ds4-server` on the same pair (TP + streaming + fp8 at ctx 400,000 unless noted):

| what | measured |
|---|---|
| prefill, 5,192-token prompt | 57 t/s |
| prefill, 11,082-token prompt | 50 t/s |
| decode, 400k alloc, <150 tokens filled | 3.5–5.5 t/s |
| decode, 400k alloc, ~4.5k filled | 3.0 t/s |
| decode, 500k alloc, 11k filled | 1.86 t/s |
| decode, 1M alloc, 11k filled | 1.13 t/s |
| first streamed token, short prompt | 1.2 s |
| first streamed token, 5,192-token client payload | ~91 s |
| 3 concurrent requests | serialise cleanly, no deadlock |

**Read the "filled" column carefully.** Decode degrades with *both* how much
context is allocated and how much of it is actually occupied, and every figure
above is at 11k of fill or less — **decode at 400–500k of real fill has not been
measured on TP.** Note also that the 400k and 500k rows differ in allocation
*and* in fill, so they do not isolate either effect. The one near-clean pair is
500k vs 750k at an identical 11k fill (1.86 → 0.99 t/s), and even that is
confounded: the 750k run was 0.26 GiB from death and thrashing.

The last two rows of the table above are the ones that decide whether a chat UI
feels usable.
Clients that resend a system prompt plus tool schemas every turn (Open WebUI)
turn a one-word message into a multi-thousand-token prefill; trim the preamble
before blaming the engine.

## What it is for — and what it is not

**It is not for realtime work.** Decode is 1–3 t/s and a long prompt takes
minutes to ingest: the 11k-token prefill above is ~3.7 minutes, and a client
that resends a fat preamble every turn (Open WebUI) makes even "hello" a
90-second round trip. Nothing about that improves with tuning; it is what a
2-bit 196 GiB model on two 121 GiB boxes costs.

**For chat and anything interactive, use V4 Flash on vLLM** — same hardware,
~1,035 t/s prefill and 23.5 t/s decode, 1M context. That is not a close call.

**This is for work where the answer is worth minutes and nobody is waiting**:
reviewing a diff, auditing a subsystem, reading a design document end to end,
a nightly pass over the day's changes. The cost is paid once per artifact
rather than per keystroke, and the whole 1M window is available to hold the
artifact plus everything it depends on.

On quality, be precise about the claim: across two blind fixtures here GLM 5.2
at 2-bit and V4 Flash **tied** — 8/8 vs 8/8 on authored coding tasks, and on
injected-bug detection GLM found 8 with Flash finding 6 outright plus 2 it
silently fixed, with zero misses either side. So the case for GLM is not that
it is smarter; it is that it is a **second, independent reviewer** — a
different vendor and architecture, with uncorrelated blind spots, and notably
more disciplined about *stating* a defect with a reproducer instead of quietly
rewriting the code. Run it where a genuinely independent read is worth the
wall clock.

## Quick start

Hardware: two DGX Sparks with a direct ConnectX link (RoCE v2), MTU 9000.
After every reboot: `sudo cpupower idle-set -D 100` on both boxes (deep idle
states add ~1 ms to every gate exchange).

```bash
./download_model.sh glm-antirez-iq2xxs     # 196.6 GiB, on BOTH boxes
# TP starts LEADER FIRST — the leader listens, the worker joins.
# (Pipeline mode is the other way round; see below.)
GLM_CTX=200000 POOL=5000 ./run_glm_tp_leader.sh     # head box
GLM_CTX=200000 POOL=5000 ./run_glm_tp_worker.sh     # then the worker box
```

Context/pool envelope (per-box planned memory stays ≤ ~90 GiB):

| GLM_CTX | POOL | Notes |
|---:|---:|---|
| 8,192 | 8,000 | fastest decode, chat |
| 50,000 | 6,000 | soak-validated workhorse |
| 100,000 | 5,000 | needle-validated at depth |
| 200,000 | 5,000 | f16 KV cache (default on CUDA) |
| 400,000 | 5,000 | `DS4_GLM_FP8_KV_STORE=1` on **both** boxes; needle-exact at 350k |

The fp8 cache packs each KV row to e4m3 values plus one f32 scale (~54% of
f16). It is opt-in and experimental in the CLI — the default there stays f16,
and `DS4_GLM_COMPACT_CACHE_F32=1` still restores the f32 era for A/B — while
the API server below turns it on by default, because 400k needs it.

Pipeline mode (`run_glm_worker.sh` then `run_glm_coordinator.sh` — worker
first here, it dials the coordinator) keeps the whole model resident split
across the pair: fastest short-context, ctx capped at 12,288 — its coordinator
runs ~1 GiB from the memory ceiling by design.

### Serving an API

`ds4-server` speaks the OpenAI API (`/v1/models`, `/v1/chat/completions`, SSE
streaming) and now runs tensor-parallel, so the 400k envelope is reachable
from any OpenAI-compatible client — Open WebUI, LiteLLM, an SDK — instead of
only the REPL:

```bash
./run_glm_server.sh                                  # leader + API, start FIRST
DS4_TP_NO_RC_BULK=1 DS4_GLM_FP8_KV_STORE=1 \
  GLM_CTX=400000 POOL=5000 ./run_glm_tp_worker.sh    # then the worker box
```

Defaults to TP + SSD streaming + fp8 at ctx 400,000, listening on
`0.0.0.0:8020` — a `127.0.0.1` listener is invisible from inside a container.
Measured on a 5,192-token prompt: **57 t/s prefill, 2.6–4.9 t/s decode**, and
1.2 s to the first streamed token on a short one. Three ids are advertised:
`glm-5.2`, `glm-5.2-chat` (thinking off) and `glm-5.2-reasoner` (thinking on)
— same weights and session, they only preset the thinking mode, and an
explicit `thinking` field in the request overrides them.

Two constraints, both learned from failures:

- **`DS4_TP_NO_RC_BULK=1` is required**, matching on both ranks. The RC-bulk
  RDMA path faults on the *second* large batch prefill in a process
  (`IBV_WC_LOC_PROT_ERR`) and takes the worker with it. The flag selects the
  older 32 KB-chunked RDMA scheme instead — same wire, same throughput
  (57 vs 53 t/s), no fault. The CLI never trips this because interactive chat
  prefills once and then only decodes.
- **The KV disk cache stays off under TP.** `--kv-disk-dir` restores a
  snapshot into the leader's session that the mirrored worker is never told
  about; the ranks desync on the next gate.

Clients that resend a large preamble every turn (Open WebUI ships a system
prompt plus tool schemas) pay a full prefill per message — a one-word "hello"
arrived here as 5,192 tokens, i.e. ~91 s to first token. Trim the preamble
before blaming the engine. `run_glm_server_pipeline.sh` is the no-TP fallback:
slower, no mirror, same four request patterns verified.

## What differs from upstream

- **Two-box CUDA tensor parallelism**: the full Metal TP gate contract on
  CUDA — dual sequence spaces, slab-flag arrival, release-on-failure — over
  TCP or RDMA (RC queue pairs, RoCE v2 GID pinning); attention and
  shared-expert head/lane splits at decode.
- **Expert LRU pool** for streamed routed experts: decode-protected against
  batch-prefill eviction, **direct pool reads** in the decode kernels (no
  gather copy), batched gate/up/down miss uploads.
- **Batched GEMM prefill under TP** (with the multi-chunk indexer fix that
  makes long prompts byte-correct), QD-aware NVMe readahead.
- **f16 compact KV cache on CUDA** via a runtime switch — halves KV, doubled
  the validated context to 200k. `DS4_GLM_COMPACT_CACHE_F32=1` restores f32
  for A/B without rebuilding.
- **Packed fp8 KV cache** (`DS4_GLM_FP8_KV_STORE=1`): e4m3 rows with per-row
  absmax scales, ~54% of f16 → the 400k envelope. Decode reads go through a
  dequantizing gather into f32 scratch and batch prefill through an f16
  unpack stage, so every attention kernel runs unmodified. Needle-exact at
  190k and 350k with f16-identical speed.
- **Async selected-expert staging** with real CUDA readback events.
- **KV session save/restore across the pair** (`--kv-save` / `--kv-load`,
  `/save` `/load` in the REPL): kill both ranks, relaunch, continue the
  conversation with only new tokens prefilled. Sessions are format-portable:
  an fp8 session restores into an f16 run and vice versa (one universal file
  format; fp8 requantizes on load — quantized codes survive round trips
  exactly, serialized floats wobble ≤2 ulp from scale re-derivation). On a
  tensor-parallel leader, restore streams the payload to the worker over the
  control connection so the mirrored session restores **without re-prefill**
  (sub-second at chat scale; falls back to a mirrored re-prefill if the push
  fails). `-n 0` with `--kv-load`/`--kv-save` is a generation-free
  pass-through for session surgery.
- **Reboot-proof RDMA bring-up**: the launch scripts discover the RoCE v2
  GID index at start (reboots and docker network churn shuffle the table).
- **Tensor parallelism in the API server**: upstream's `ds4-server` had none —
  TP argument parsing, the `adopt → prepare → validate` option ordering a TP
  leader needs (it never passes `--layers`, so the distributed validator has
  to be told those flags belong to the TP transport), worker-role run, leader
  bind, and transport lifecycle tied to server shutdown. The 400k envelope is
  now an HTTP endpoint, not just a REPL.
- **Model ids that say what they do**: the `-chat`/`-reasoner` aliases all
  advertised the engine's own name, so a model picker showed two or three
  identical entries; they now carry distinct display names.
- **Server fixes** for GLM thinking mode (unclosed reasoning surfaces as
  `reasoning_content`, never as content), ported from the upstream PR queue
  with credit (#524; #158 in part; #460 for the miss uploads).
- **A GB10 operations runbook** learned the hard way: unified-memory OOM
  freezes the box rather than killing the process, so this repo ships memory
  guards, planned-budget envelopes, and loud diagnostics where silent
  failures used to live.

Engineering notes for the major changes live outside the tree (they are
session documents, not product docs); ask in issues if you want any of them.

## Roadmap

- Prefill staging/compute overlap + grouped GEMM → 60–100 t/s target
- Packed-row wire format for the TP restore push (ships e4m3 bytes instead
  of f32 planes: ~7× less on the wire for huge fp8 sessions)
- MTP on the pipeline: the probe landed (speculative cycles work end to end
  behind `--glm-mtp`); making it a *win* needs per-draft routed-expert
  upload and a fused two-token verify — until then the flag costs speed
- Fix the RC-bulk memory-region lifecycle
  (`tp_rdma_rc_bulk_post_recv`/`_finish`) so the faster bulk posting works
  under repeated large prefills instead of being bypassed
- Make the server's KV prefix cache mirror-aware, so it can be re-enabled
  under TP (today a restore desyncs the worker; it is also why a worker
  restart mid-session shows up as a false cache hit)
- Ladder the served context — 25k, 50k, 100k. 400,000 allocates and the CLI is
  needle-exact at 350k, but 11,170 tokens is the largest prompt verified
  end to end *through the server*; decode also falls as the window fills

## Credits

Built on [DwarfStar (ds4)](https://github.com/antirez/ds4) by Salvatore
Sanfilippo (antirez) — the engine, the models, the philosophy. Several fixes
ported from upstream's open PR queue with thanks to their authors. And to
Alfa Romeo, for naming the architecture thirty years early.

MIT, same as upstream.
