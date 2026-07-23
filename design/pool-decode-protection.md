# Expert-pool decode protection — design (v1)

## Problem (measured 2026-07-23, logs 2026-07-2{2,3}_diag_*.log)
Decode at 45k ctx TP = 2.23 t/s, 461 ms/token; largest bucket
shared_gate_up_swiglu 181 ms = expert staging (known misattribution: the
staging block runs inside that stage's timer). Pool stats show the cause:
batch prefill FLOODS the 6000-slot LRU (cumulative hits 2.9-5.5%), so
decode runs with a thrashed pool and pays ~8 experts × 9.28 MiB × 76
layers/token of page-cache→device copies (~74 MiB/layer class → ~180
ms/token — matches the bucket). Pool cannot grow: POOL=7000 @50k ctx OOMs
(memguard kill, avail 697 MB). At 2k/8000-slot configs decode hits reached
88%+ and the same bucket was ~60-80 ms — the mechanism is proven.

## Fix (option A — keep prefill out of the LRU)
Batch prefill staging (n_tokens > 1 in the selected-cache funnel) stops
acquiring/inserting pool slots: it stages straight from the mmap/page
cache into the per-chunk compact scratch (the pre-pool data path). Decode
(n_tokens == 1) keeps the pool exactly as today. Rationale:
- Prefill reuse across chunks is already served by the OS PAGE CACHE and
  the madvise(MADV_WILLNEED) readahead pass; the QD8 lesson showed GB10
  coherent direct copies are fast — the device pool adds little for
  prefill but destroys the decode working set.
- Decode-hot experts then survive any ingest; after a 45k prefill the
  decode pool is exactly as warm as before the prompt arrived.
- Bonus: benefits pipeline+streaming decode too (same funnel).

Escape hatch env: DS4_CUDA_POOL_PREFILL_INSERT=1 restores today's
behavior. No new default envs otherwise.

## Secondary (only if review finds it cheap): async miss-fill
Overlap decode miss copies with compute. NOT required for v1; the hit-rate
restoration is the main win. Keep out of scope unless the reviewer sees a
one-day path.

## Review asks (verify in source, append findings + impl map below)
1. The staging funnel cuda_stream_selected_cache_begin_load
   (ds4_cuda.cu ~23469): confirm it can distinguish batch-prefill calls
   from decode calls (n_tokens or a caller flag) and identify the exact
   pool acquire/insert points to bypass (pass-1 acquire, pass-2 fill,
   pool_slot/pool_hit vectors, g_pool_protect_layer interplay).
2. Confirm the non-pool direct-staging path still exists and is correct
   under TP ownership filtering (slot nulling to -1 must be unaffected),
   or name the smallest change to stage scratch-direct without pool.
3. Prefill-regression risk: does batch prefill currently HIT the pool
   across chunks materially (would bypassing slow prefill)? Estimate from
   code + the measured prefill rates (53.3 t/s with thrashed pool ≈
   near-0% useful hits suggests prefill barely benefits today).
4. Check decode-side effects: does decode's pass-1 pin/protect logic
   assume prefill previously inserted entries (warm-start assumptions),
   and does DS4_CUDA_NO_EXPERT_POOL provide a template for a per-call
   bypass?
5. Also (separate finding, do NOT design): locate why the decode
   'attention' stage grows +28 ms from 2k→45k when indexed attention is
   bounded by 2048 selected rows (ds4.c ~45878-45930 chain; suspect the
   selected-row gather or an O(ctx) kernel loop). Name the responsible
   kernel/loop with line evidence if you can find it quickly.
6. Append implementation map + ordered checklist with correctness
   checkpoints (text-identity runs) below.

## REVIEW FINDINGS (pass 1)

Verdict: **approve option A with two corrections.** The mechanism is real, the
bypass is cheap, and the failure-mode ordering actually improves. But the design
as written would silently drop QD8 readahead for prefill (a regression the doc's
own history warns about), and it references an identifier that does not exist.

### Ask 1 — funnel can distinguish, but not where the doc says

`cuda_stream_selected_cache_begin_load` (ds4_cuda.cu:23469) receives only
`(table, selected_ids, slot_count)` — **it has no n_tokens**. `slot_count`
conflates the cases (decode: `DS4_N_EXPERT_USED`=8; prefill: `n_tokens*8`), and
the funnel does not know `DS4_N_EXPERT_USED`, so inferring batch from
`slot_count` inside the funnel is a layering hack. The correct gate point is one
frame up: `ds4_gpu_stream_expert_cache_prepare_selected_batch`
(ds4_cuda.cu:29014) **already takes `n_tokens` explicitly** and is the only
entry the batch-prefill hosts use:
- ds4.c:41388 in `glm_graph_cuda_stream_prefill_batch_selected_load`
  (ds4.c:41343) — the GLM CUDA prefill staging helper (the path that matters);
- ds4.c:20844 `metal_graph_cuda_stream_prefill_batch_selected_load` (DeepSeek);
- ds4.c:21165 `rocm_graph_batch_selected_async_load_run`.

Decode/MTP go through the other two wrappers, which never see batches:
- `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor`
  (ds4_cuda.cu:27750, funnel call at 27771) — called from decode FFN
  `glm_graph_encode_sparse_ffn_one` (ds4.c:40208), MTP `glm_graph_mtp_step`
  (ds4.c:42540), first-token (ds4.c:46608), test harness (ds4.c:52943, 53247);
- `ds4_gpu_stream_expert_cache_begin_selected_load` (ds4_cuda.cu:28307) —
  DeepSeek decode overrides (ds4.c:20520, 20700, 20763, 20961, 23447, 23475).

Exact bypass points inside the funnel:
- `pool_ok` at ds4_cuda.cu:23570 (`cuda_expert_pool_ensure`) — the single gate;
  everything pool-related is already conditioned on it: pass-1 acquire loop
  23582-23605, late re-acquire 23618-23622, pool fill + D2D gather 23623-23667,
  stats 23688-23699. Setting `pool_ok = 0` for prefill routes every expert to
  the existing direct path at 23669-23686. No `pool_slot`/`pool_hit` vector
  surgery needed — they stay all-`UINT32_MAX`/0.
- **Doc error:** `g_pool_protect_layer` does not exist anywhere in ds4_cuda.cu
  (grep: zero hits). The actual intra-pass protection is the clock reference
  bit: `slot_used[...] = 1` on acquire (ds4_cuda.cu:276, 322) plus the
  second-chance sweep (306-311). There is no layer-protect interplay to worry
  about; strike that sentence from the design.

### Ask 2 — direct path exists, TP-safe; one real gap: readahead

The non-pool direct staging path is ds4_cuda.cu:23669-23686
(`cuda_model_copy_to_device_streamed` mmap→scratch for gate/up/down). TP
ownership filtering is complete before any pool logic runs: peer experts are
nulled to `slot_ids[i] = -1` at 23514-23517, `compact_ids` only ever contains
owned experts (23519-23525), the all-peer placeholder is 23527-23535, and the
slot table upload is 23700-23707. None of that touches the pool. **Bypass
requires zero changes for TP correctness.**

The gap: `fetch_jobs` / `cuda_fetch_readahead` (ds4_cuda.cu:23581-23605,
readahead impl at 341-356) is built **only for pool misses inside
`if (pool_ok)`**. With `pool_ok == 0` the direct path gets **no
MADV_WILLNEED at all** — this is already a latent perf bug for
`DS4_CUDA_NO_EXPERT_POOL` runs, and a naive bypass would hand cold-cache
prefill QD1 reads (4.9 vs 11.2 GB/s per the comment at 328-335). Correction:
hoist the fetch_jobs build so that when the pool is bypassed, all three ranges
of **every** compact expert are hinted before the copy loop.

### Ask 3 — prefill regression risk: NO (bypass is neutral-to-positive)

Code reasoning: per layer per chunk the funnel is called once; `compact_ids`
dedupes, so each unique expert is staged exactly once per call **with or
without the pool** — the pool can only help *across* calls. Cross-call reuse
would need the working set to fit: pool keys are `layer*256+expert`
(ds4_cuda.cu:23587), and a 512-token chunk selects 512×8 = 4096 slots/layer,
i.e. essentially all owned experts (~128 under TP, up to 256 single-rank). One
full pass = 76 layers × ~128 = **~9,728 unique keys vs 6,000 slots** — the
clock (298-320) evicts the entire pool every chunk, so chunk N+1 re-misses
everything. The measured 2.9-5.5% cumulative hit rate is exactly this
prediction; prefill's real cross-chunk reuse is already the OS page cache, as
the doc says.

Cost accounting per prefill layer today (pool on): miss = mmap→pool streamed
copy (23631-23642) **plus** a pool→scratch D2D gather (23654-23662). Bypassed:
mmap→scratch only. So the bypass *removes* ~128 × 9.28 MiB ≈ 1.2 GiB/layer of
D2D traffic on GB10's shared LPDDR5x — order 0.5 s per 512-token chunk, a few
percent of the ~9.6 s chunk time at 53.3 t/s. **Expected prefill delta: 0 to
mildly positive, provided the readahead correction above lands.** Risk call:
low.

### Ask 4 — decode-side effects: no warm-start assumption; safer OOM ordering

Decode pass-1 (23582-23605) treats every key independently: a miss fills via
streamed copies, a hit pins with `slot_used=1`. Nothing assumes prefill
pre-inserted entries. Post-bypass, prefill never calls
`cuda_expert_pool_acquire`, so decode's keys, reference bits, and clock hand
survive the entire prompt untouched — the doc's "exactly as warm as before the
prompt arrived" claim is correct at the code level.

`DS4_CUDA_NO_EXPERT_POOL` (ds4_cuda.cu:215, inside `cuda_expert_pool_ensure`)
confirms the template: a single boolean short-circuit at the `pool_ok` site is
sufficient; no other state needs touching. Use a call parameter, not an env, as
the per-call mechanism; keep `DS4_CUDA_POOL_PREFILL_INSERT` as the restore env
read at the prepare_selected_batch gate.

One behavioral shift worth stating: today the pool grows to full budget during
prefill chunk 1 (9,728 keys force `cuda_expert_pool_grow`, 239-268, to budget
early, while KV is still small). With the bypass, growth moves to the first
decode tokens, when KV is already at 45k. Growth failure is graceful — freeze
budget at current slots (250-255) — so the failure mode *improves*: instead of
a full pool starving KV allocation, a full KV shrinks the pool. On a fresh
process at max context the pool may freeze below 6,000; that is the correct
trade and needs no code, just a note in the perf run.

### Ask 5 — decode 'attention' growth (bounded side-finding)

The premise "bounded by 2048 selected rows" is wrong for two of the stages:
only the attention **core** (`ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor`,
ds4.c:45925) is bounded by `last_indexer_selected_count`. Upstream, two calls
are O(ctx) by construction:
1. `ds4_gpu_glm_indexer_score_one_tensor` (ds4.c:45874, impl
   ds4_cuda.cu:25406 → `glm_indexer_scores_launch` 25348, wmma kernel grid
   `(n_rows+127)/128` at 25371) — scores **all** `visible` rows; ~11.5 MiB of
   f16 indexer-key-cache reads/layer at 45k, ~875 MiB/token over 76 layers
   (~4 ms of bandwidth — real but not 28 ms).
2. **Leading suspect:** `ds4_gpu_indexer_topk_tensor` (ds4.c:45884, impl
   ds4_cuda.cu:12013). At top_k=2048 and `n_comp > 4096` it takes the chunked
   tree path (12040-12108): `ceil(ctx/4096)` chunk blocks (12 at 45k vs 1 at
   2k), then a **serial chain of dependent tree-merge launches**
   (`indexer_topk_tree_merge_pow2_kernel`, 12084-12101: 12→6→3→2 sets, plus
   final merge 12103) with tiny grids (n_tokens=1) that cannot fill the GPU.
   ~5 extra dependent launches × 76 layers of latency-bound bitonic merges is
   the right order of magnitude for +28 ms.
Instrumentation already exists to split them: the substage boundaries
`"indexer_scores"` (ds4.c:45883) and `"indexer_topk"` (ds4.c:45889) under
`DS4_METAL_DECODE_STAGE_PROFILE` (ds4.c:16753). Run 2k vs 45k with that env and
read the two buckets before touching any kernel. Do not fix in this change.

### Ask 6 — implementation map + ordered checklist

Map (all in ds4_cuda.cu unless noted; no ds4.c changes, no extern-C signature
changes, Metal/ROCm backends untouched):
- `cuda_stream_selected_cache_begin_load` (23469): add trailing param
  `int pool_insert`.
- 23570: `const int pool_ok = pool_insert && cuda_expert_pool_ensure(...);`
- 23579-23605: hoist readahead — build `fetch_jobs` for every compact expert
  when `!pool_ok` (all three ranges per expert, same push_back shape as
  23593-23601); keep miss-only jobs when `pool_ok`. Call
  `cuda_fetch_readahead` (23603) unconditionally.
- Callers: 27771 pass `1`; 28311 pass `1`; 29023
  (`ds4_gpu_stream_expert_cache_prepare_selected_batch`) pass
  `(n_tokens <= 1) || getenv("DS4_CUDA_POOL_PREFILL_INSERT") != NULL`.
- Optional: extend the stats print (23688-23699) with a bypassed-expert
  counter so the perf run can confirm prefill traffic left the pool.

Checklist (each step ends runnable):
1. Add the `pool_insert` parameter, pass `1` from all three wrappers
   (no behavior change). Checkpoint: build; short prompt decode; output
   text identical to HEAD; `DS4_CUDA_EXPERT_POOL_STATS=1` shows unchanged
   hit trajectory.
2. Readahead hoist for the `!pool_ok` path. Checkpoint:
   `DS4_CUDA_NO_EXPERT_POOL=1` run — text identical; cold-cache prefill
   (`echo 3 > /proc/sys/vm/drop_caches`) at least as fast as before
   (this path previously had no readahead, expect improvement).
3. Flip 29023 to `pool_insert = (n_tokens <= 1) || env`. Checkpoint:
   chunked-prefill prompt (>1 chunk) — token-for-token identity vs HEAD;
   then `DS4_CUDA_POOL_PREFILL_INSERT=1` — identity again AND pool stats
   match HEAD exactly (proves the escape hatch is bit-faithful).
4. TP two-rank pipeline run (10.0.0.1 + 10.0.0.2) at long context.
   Checkpoint: text identity vs HEAD on both single-rank and TP; confirms
   slot -1 nulling and the all-peer placeholder (23527-23535) unaffected.
5. Perf validation at 45k: `DS4_CUDA_EXPERT_POOL_STATS=1` decode hit rate
   should recover toward the 88%+ seen at small ctx; shared_gate_up_swiglu
   bucket toward 60-80 ms; prefill t/s within noise of 53.3. Also note the
   frozen pool size printed by grow (250-255) for the OOM-ordering note.
6. (Separate, from ask 5) `DS4_METAL_DECODE_STAGE_PROFILE=1` at 2k vs 45k to
   split indexer_scores vs indexer_topk before any kernel work.
