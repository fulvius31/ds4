# twinspark-glm — the Grand Language Model

**GLM 5.2 at 2-bit, running interactively across two NVIDIA DGX Sparks.**

*Twin Spark*, as in Alfa Romeo's twin-ignition engines — two spark plugs per
cylinder, firing together for a cleaner, stronger burn. Here it is two DGX
Sparks firing as one: tensor parallelism over a direct RoCE link, serving a
196.6 GiB model on a pair of 121.69 GiB boxes.

And *GLM* is the model's name, but around here it stands for **Grand
Language Model** — in the Gran Turismo tradition: not the largest engine on
the road, but the one built to cross a continent (or 200,000 tokens of
context) in comfort.

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
| **TP + SSD streaming** (the flagship) | 58.7 t/s | 3.3 t/s @45k, ~3.0 @100k | **up to 200,000** |
| Pipeline resident (short-context daily driver) | 90 t/s | 5.9 t/s | ≤ 12,288 |
| Pipeline + SSD streaming | ~68 t/s | 2.7 t/s | long-context fallback |

Validated, not vibes: 100k-deep needle retrieval is exact, temp-0 outputs are
byte-identical across cold starts (verified through every optimization), and
the 200k envelope allocates and runs inside the same ~90 GiB planned budget
that the memory runbook proves safe. Decode at 45k went **2.23 → 3.29 t/s
(+48%)** across this fork's optimization cycles.

## Quick start

Hardware: two DGX Sparks with a direct ConnectX link (RoCE v2), MTU 9000.
After every reboot: `sudo cpupower idle-set -D 100` on both boxes (deep idle
states add ~1 ms to every gate exchange).

```bash
./download_model.sh glm-antirez-iq2xxs     # 196.6 GiB, on BOTH boxes
# worker box, then leader box:
GLM_CTX=200000 POOL=5000 ./run_glm_tp_leader.sh     # start FIRST, on the head
GLM_CTX=200000 POOL=5000 ./run_glm_tp_worker.sh     # then on the worker
```

Context/pool envelope (per-box planned memory stays ≤ ~90 GiB):

| GLM_CTX | POOL | Notes |
|---:|---:|---|
| 8,192 | 8,000 | fastest decode, chat |
| 50,000 | 6,000 | soak-validated workhorse |
| 100,000 | 5,000 | needle-validated at depth |
| 200,000 | 5,000 | f16 KV cache (default on CUDA) |

Pipeline mode (`run_glm_worker.sh` then `run_glm_coordinator.sh`) keeps the
whole model resident split across the pair: fastest short-context, ctx capped
at 12,288 — its coordinator runs ~1 GiB from the memory ceiling by design.

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
- **Async selected-expert staging** with real CUDA readback events.
- **KV session save/restore across the pair** (`--kv-save` / `--kv-load`,
  `/save` `/load` in the REPL): kill both ranks, relaunch, continue the
  conversation with only new tokens prefilled.
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

- fp8 compact KV (Metal PR #418 pattern) → ~400k context in the same budget
- Prefill staging/compute overlap + grouped GEMM → 60–100 t/s target
- 190k-depth needle confirmation run at the new 200k envelope
- MTP speculative probe on the resident pipeline config

## Credits

Built on [DwarfStar (ds4)](https://github.com/antirez/ds4) by Salvatore
Sanfilippo (antirez) — the engine, the models, the philosophy. Several fixes
ported from upstream's open PR queue with thanks to their authors. And to
Alfa Romeo, for naming the architecture thirty years early.

MIT, same as upstream.
