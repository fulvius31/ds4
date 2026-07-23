# TP multi-chunk batch prefill corruption — fix plan

## Symptom
TP + streaming + GEMM batch prefill: any prompt spanning >1 prefill chunk
(chunk = 2048 under TP) generates garbage from the first post-prefill token
(`|a|a|a|a`, temp 0). Single-chunk prompts (1871 tok tested) are exact.
Present since batch prefill landed (2727f2d); baseline b947fa3 reproduces —
the speculative-prefetch work was exonerated by bisect.

## Exonerated (do not re-litigate)
- Transport: TCP, RDMA RC, RC direct bulk — all corrupt identically.
- Chunk size (1024/2048), coverage acceptance (valid_pairs), memsetAsync of
  pair scratch, attention split, shared split (corruption persists with both
  splits off).
- Prefetch stash (stash@{0}) — innocent, committed baseline equally corrupt.
- Prompt content (two different prompts, same `|` first token).

## Repro + reference (fixed methodology)
- Prompt: `/tmp/longprompt.txt` = `head -c 10000 speed-bench/promessi_sposi.txt`
  (~3098 tok = 2 chunks under TP).
- TP run: run_glm_tp_leader.sh / run_glm_tp_worker.sh,
  `DS4_GLM_TP_BATCH_PREFILL_EXPERIMENTAL=1 DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL=1`,
  `--prompt-file /tmp/longprompt.txt -n 8 --temp 0 --nothink`, GLM_CTX=8192 POOL=5000.
- Reference: pipeline+streaming same prompt/flags (text-verified correct config),
  OR single-box streaming (also correct). Save reference continuation text once.
- Every iteration MUST check generated TEXT, not just rate lines (session lesson).
- Logs: ~/logs_ds4_tests/2026-07-2x_tpfix_*.log on BOTH boxes.

## Primary hypothesis (H1): chunk-2 takes a different encode path that
## lacks (some of) the TP handling chunk-1 got
History: the "missing combine" fix (glm_graph_tp_batch_ffn_combine after
dispatch) was added to the indexed batch encoder and validated on a
single-chunk prompt. Chunk routing (ds4.c ~57630-57667) may send follow-up
chunks (pos_base>0) through a different encoder/branch (e.g.
forward_indexed_tokens vs the chunk-1 path), which may still:
- miss the ffn combine (each rank keeps its partial → KV written from
  partial hidden states from chunk 2 on), or
- miss the TP ownership filter in its staging/dispatch (peer-owned slots
  not nulled → double-count or garbage weights), or
- miss the rank0-full/rank1-zero policy for cache-bypass layers.

## Secondary hypotheses
- H2 (staging state carryover): cuda_stream_selected_cache_begin_load reuses
  the compact slot table across chunks; chunk-2 staging may keep chunk-1
  slot_selected entries (incl. -1 ownership nulls) for experts it considers
  already staged, while the GEMM's sel_ids assume fresh mapping.
- H3 (rank KV divergence): if either rank's chunk-1 KV/compact cache rows
  differ (bad combine order vs KV write, or indexer k rows), chunk-2
  attention diverges. Distinguishable by rank0-vs-rank1 dumps (below).
- H4 (gate seq/slab reuse across chunks in the batch space): would usually
  fail loudly (gate-order check), so lowest priority.

## Diagnostic ladder (cheapest first, each step falsifies hypotheses)
1. Confirm repro at HEAD (ac2348a). One TP run, one reference run.
2. Path audit (no runs): read the chunk loop (ds4.c ~57630) and answer:
   which encoder does chunk 1 use vs chunk 2 under
   DISABLE_STREAMING_TOKEN_PREFILL + TP? If they differ, diff the TP blocks
   (combine call, ownership filter, bypass policy) between the two paths.
   H1 is confirmed/denied by inspection alone.
3. Hidden-state divergence hunt (existing hooks DS4_GLM_HIDDEN_DUMP +
   DS4_GLM_HIDDEN_DUMP_LAYER): dump at the LAST prefill position,
   TP-leader vs reference, binary-search the first bad layer
   (dense 0-2 sane? layer 3 = first sparse? mid? MTP-adjacent 77?).
   - Verify first: dump semantics in batch mode (which row it writes).
   - Also dump chunk-1-last position: must MATCH reference (single-chunk
     exactness predicts it) — this validates the harness itself.
4. Rank cross-check at first bad layer: dump same tensor on rank1.
   identical-but-wrong on both ranks → deterministic staging/dispatch bug
   (H1/H2); divergent between ranks → combine/gate/KV (H1-combine/H3).
5. Mechanism pin inside the layer: dump pre-attn / post-attn / post-ffn at
   first bad layer (add temp dump points if needed):
   post-ffn-only bad → combine/ownership/GEMM (H1/H2);
   post-attn bad → KV/indexer from chunk 1 (H3).

## Fix + validation ladder
- Implement the indicated fix (likely: port the TP block — combine +
  ownership filter + bypass policy — to the chunk-2 encoder path, or reset
  staging slot state per chunk).
- V1: 2-chunk text identity vs reference (temp 0, 24 tok), both transports
  ok to test on RDMA only (transport exonerated).
- V2: 3-chunk prompt (head -c 20000, ~6k tok) text identity — rules out
  "fix works only for chunk 2".
- V3: logit rel-L2 vs reference within the documented ~0.4 drift band +
  argmax agreement (lg dump machinery from earlier validations).
- V4: single-chunk regression (1871-tok prompt still exact) + decode-after-
  prefill sanity (no rebuilds, pool hits climbing).
- V5: remove/keep the >64-token safety gate accordingly; commit; then the
  45k timing run the user wants (GLM_CTX=50000 POOL=6000) with token-I/O
  breakdown vs pipeline numbers.

## Stop rule
If after the full diagnostic ladder the first bad layer/mechanism is still
unidentified (2 sessions max), park again with findings appended here and
keep the experimental gate.

## Ops guardrails
- memguard armed on both boxes; POOL=5000 for multi-chunk runs; wait for
  memory drain (>100 GiB avail both) between heavyweight runs; kill and
  launch in SEPARATE shell calls (pkill footgun v2); logs date-prefixed on
  both boxes; worker binary md5-verified before every run.

## REVIEW AMENDMENTS (2026-07-22, reviewer pass 1)
H1 DENIED: one encoder for all chunks (glm_graph_forward_indexed_tokens via
session-sync loop ds4.c:58587-58699, use_batch_ffn hard-true 40760); TP block
complete per layer per chunk (staging filter 42106, dispatch 42131, combine
42146). H2 DENIED: begin_load invalidates+rebuilds slot table per call
(23473), ownership nulling also on pool-hit paths. H3 structurally blocked
(commutative combine) except env-mismatch between ranks — check env parity.
CONFOUND FOUND: multi-chunk ⟺ crossing indexer top_k=2048 in ALL past tests;
chunk 1 uses causal-range select (pos+n<=top_k, 43843), chunk 2+ runs real
indexer top-k (44299+) — a regime NO passing TP test ever executed. Prior:
65% "pos>2048 indexer regime under TP", 35% chunk-boundary mechanics.
NEW STEP 2a (decisive): DS4_GLM_METAL_INDEXED_PREFILL_CHUNK_TOKENS=1024 (env
read at ds4.c:34382, set on BOTH ranks) + 1871-tok prompt = 2 chunks, never
crossing 2048. A/B against same prompt at default chunk (single, known
exact). Corrupt ⇒ chunk mechanics; clean ⇒ top-k regime.
NEW STEP 2b: DS4_GLM_LOGIT_DUMP (34541) last-pos prefill logits TP vs ref on
the 3098 repro — decides "prefill corrupt" vs "first-decode-at-pos>2048
corrupt" (unproven assumption in original plan).
Dump caveats: batch hidden dump writes ALL rows (pin one layer/run; T-suffix
= absolute pos); TP-vs-ref comparisons use rel-L2 drift band NOT byte match;
glm_ffn_batch_* named dumps pass pos=0 → chunk 2 overwrites chunk 1.

## ROOT CAUSE FOUND + FIXED (2026-07-22 late)
Diagnostic path: Run B (1871 tok forced 1024-chunks via new env
DS4_GLM_INDEXED_PREFILL_CHUNK_TOKENS, both chunks causal) = COHERENT →
chunk mechanics innocent. Run C/D logit dumps (DS4_GLM_LOGIT_DUMP): TP
rel-L2 1.06 vs pipeline, argmax 91("|") vs 73022 → prefill itself corrupt.
Run G/H all-layer dump at pos 3097 (new env DS4_GLM_HIDDEN_DUMP_POS):
L99 embedding EXACT 0.000 but L00 already 0.76 → TP touches even dense
layers in batch prefill → found tp_attn_head_split (ds4.c ~43959):
batch-prefill attention head split, world==2, n_tokens>=64 default floor
(DS4_GLM_TP_HEAD_SPLIT_MIN env), INDEPENDENT of DS4_GLM_TP_ATTN_SPLIT
(decode-only env) — why the old "splits off" exoneration missed it.
Run I: DS4_GLM_TP_HEAD_SPLIT_MIN=999999 → COHERENT at 37.6 vs 38.8 t/s
corrupt (split worth only ~3%). Mechanism: indexer score path consumes
all-head state (same constraint that broke decode q_b ranging); split
rank feeds zeroed unowned heads → top-k selection poisoned → chunk 2+
attends garbage positions. Causal-regime chunks never read scores → Run B
clean, single-chunk exact.
FIX: tp_attn_head_split now requires use_causal_range_select (ds4.c).
Validation: V1 = 3098 repro default envs; V2 = 6k 3-chunk
(/tmp/longprompt6k.txt); V4 = 1871 regression (expect byte-same as Run A,
~/logs_ds4_tests/2026-07-22_tpfix_runA.log text "[The text cuts off
mid-sentence at \"dal nostro manoscrit\" - the word appears to be cut
off"); then lift the n_tokens>64 EXPERIMENTAL refusal gate + commit.
Follow-on (optimization, separate): re-enable split in top-k regime by
feeding indexer from full-width q (replicate indexer inputs only).
