# Design: GLM decode-kernel tuning + two-Spark tensor parallelism on CUDA

Goal: raise two-DGX-Spark GLM 5.2 decode from 4.6 t/s toward 15–20 t/s by
(A) tuning the GLM CUDA decode kernels and (B) porting ds4's two-machine
tensor parallelism from Metal to CUDA over the 200GbE RoCE link.
Both target the resident configuration (coordinator+worker, ctx-limited);
they compose with, but do not depend on, the streaming/100k work.

Baseline (measured, this branch):
- Pipeline decode 4.6 t/s @2k ctx = ~217 ms/token: coordinator ~105 ms
  (layers 0:39) + worker ~106 ms (40:output) + ~1 ms network. TCP is
  irrelevant; the pipeline's serial turn-taking is architectural.
- Decode MoE = 1.32 ms/routed-layer/side (gateup 0.87 + down 0.44 via
  per-pair IQ2 dot kernels) ≈ 50 ms of each side's ~105 ms. The kernels
  run 4.5–4.8x above the ~0.28 ms/layer LPDDR bandwidth floor
  (8 experts x 9.28 MiB active weights at ~273 GB/s).
- Unattributed ~55 ms/side: dense Q8 matvecs, DSA attention + indexer
  scan, norms, host orchestration. The distributed decode path does not
  currently emit DS4_METAL_DECODE_STAGE_PROFILE lines (profiling hooks
  live in the non-slice decode functions), so attribution is incomplete.
- Long-context decode is indexer-dominated: measured 4.6 t/s @2k ->
  2.4 t/s @10k (~18 ms per 1k ctx per token). TP halves this (head
  split) but does not remove the linear growth.

## Workstream A — GLM decode kernels (est. 3–5 days, standalone value)

A0. Attribution first: extend the layer-stage decode profiler to the
    distributed slice-eval decode path (or add an env-gated per-stage
    timer to glm_graph_forward_token). Deliverable: a table accounting
    for >90% of the 105 ms. No optimization until this exists.

A1. Fused per-layer decode MoE kernel ("glm_moe_sum8"): one launch per
    routed layer computing gate+up dots, silu*up*router-weight, and the
    down projection accumulated across all 8 selected experts, with the
    IQ2 codebook tables staged in shared memory (the dev_..._lut device
    functions and DeepSeek's moe_down_sum6/decode-LUT-gate kernels are
    the pattern; GLM needs n_expert=8 and IQ2 down). Design constraint:
    take an expert-ownership mask parameter from day one (skip, never
    clamp, masked experts) so Workstream B reuses it unchanged.
    Target: 1.32 -> ~0.45 ms/layer => ~33 ms/side saved.

A2. Shared-memory LUT staging for the standalone gate/up dot kernels
    (used by prefill fallbacks and any n<8 path), same trick, lower
    priority than A1.

A3. From A0's table: fix anything else >10% of decode only if the fix is
    contained (candidate: dense Q8 matvec efficiency; candidate: indexer
    scan — measure its true share at 2k and 10k, since it is also the
    long-context lever).

Validation: these are n_tokens=1 kernels, so the single-token oracle
comparison is exact — bit-diff logits vs the current kernels on the
standard prompts, then a pipeline perplexity spot-check
(logs_ds4_tests/20260721_ppl_pipeline_dot.log avg_nll 1.963734 is the
reference). Expected outcome: pipeline decode 4.6 -> ~7–9 t/s.

## Workstream B — two-machine TP on CUDA (est. 2–3 weeks)

Architecture: mirror the proven Metal design in ds4_tp.c —
50/50 routed-expert ownership plus attention-head halves per machine;
dense/shared/embedding/output weights replicated; both machines walk
every layer of the same token simultaneously and exchange 24 KB partial
sums at per-layer gates. Memory per box ≈ replicated non-routed
(~13 GiB) + expert half (~87 GiB) + KV => fits the ~104 GiB ceiling at
ctx 2048–8192 (uneven expert split is the relief valve if tight).

Key facts grounding the port:
- ds4_tp.c is backend-clean: exactly one Metal reference (the
  validate-options gate). Its transport (TCP + libibverbs RDMA) is
  OS-level; verbs on Linux/mlx5 (ConnectX-7) is native territory —
  better supported than the Mac rdma_ctl path it was written for. The
  operational vLLM stack on these boxes already runs two-node TP over
  this link (NCCL/RoCE), proving the fabric.
- The GPU contract is a shared slab (per-layer out/in vectors, 8-byte
  flags, GPU-written gate-ready words; ds4_tp_slab_bytes). On GB10 the
  slab is plain coherent memory: kernels write partial sums directly,
  no staging copies.
- All 17 ds4_gpu_tp_* entry points exist in ds4_gpu.h; the CUDA side has
  8 stubs + 9 missing — the entire GPU gate layer is new code, but the
  Metal implementations in ds4_metal.m are a line-for-line reference.
- ds4.c's GLM graph already contains the TP branches (tp_world==2 paths,
  tp_attn_head_split consumers, tp_batch bounce/combine, ownership-aware
  MoE dispatch checks) — shared code, not Metal-specific.

B1. Slab + gates on CUDA: implement init/shutdown, gate_encode /
    big_gate_kick (enqueue partial-sum production + flag write via
    cuStreamWriteValue32), gate_wait (cuStreamWaitValue32 on the in-flag
    — stream-ordered, no SM spin), failure propagation. Probe stream
    memops support on GB10 first; fallback is a 32-thread spin-wait
    kernel on the coherent flag (acceptable: gates are ~10 us-scale).
B2. Expert-ownership correctness on CUDA: masked ids (-1) must be
    skipped everywhere. Audit result: the GLM q2K kernels skip; the
    generic IQ2 plain/sorted kernels CLAMP to expert 0 (wrong under TP)
    — change to skip unconditionally (safe: invalid ids never occur
    outside TP). The GEMM prefill path already hard-falls-back on
    masked ids (offsets coverage check); a mask-aware grouped variant is
    a later optimization, not a correctness need (TP prefill can use the
    dot path initially).
B3. Bring-up sequence: TCP transport first; decode-only with
    --tensor-parallel-token-prefill (the exact-arithmetic prefill mode)
    for first light; then batch prefill via the existing tp_batch bounce
    path; RDMA verbs last as a latency refinement.
    Lift the Metal-only gate in ds4_tp_validate_engine_options behind a
    capability check.
B4. Validation: temp-0 token-identity and logit-diff vs the pipeline
    configuration on the standard prompts (the established oracle
    harness), perplexity spot-check, kill-one-rank behavior, and the
    ownership negative test (Q4-routed GLM must be rejected, per README).

Expected outcome: decode ~2x from TP alone (4.6 -> 8–9); combined with
Workstream A, ~12–18 t/s at short ctx; long-ctx decode roughly doubles
at every point on the curve (indexer halves), with the indexer scan
remaining the dominant term beyond ~30k (tracked separately in the
streaming/100k plan).

## Sequencing and risk

Order: A0-A1 first (independent, immediate value, and A1's
ownership-mask parameter is a B prerequisite), then B1->B4, then A2/A3
opportunistically during B's test cycles.

Top risks:
1. sm_121 silent-wrong-results (two prior instances on this arch):
   every new kernel gates through the bit-exact n=1 oracle before use.
2. GB10 stream-memop availability for gate waits (probe day one;
   spin-kernel fallback specced).
3. Memory ceiling under TP (97.5+ GiB/box): start at ctx 2048, measure,
   use uneven expert split if needed; OOM guards stay armed.
4. TP failure modes over TCP (timeouts during multi-second stalls):
   reuse ds4_tp's existing keepalive/pause machinery; test rank-kill.
5. Schedule risk on B: the slab/gate layer is subtle concurrency code;
   the Metal reference de-risks semantics but not CUDA-specific races —
   budget real soak time (long generations, both prompt modes).

## REVIEW AMENDMENTS (2026-07-21, reviewer verdict: A=go, B=conditional)

CORRECTED NUMBERS: replicated non-routed = 19.59 GiB (not 13); expert half
= 88.49 GiB; per-box TP-resident weights = 108.08 GiB vs ~104 ceiling =>
TP-resident DOES NOT FIT. Decode TP splits ONLY the routed MoE today
(attention/dense/indexer replicated; head split exists only in the batch
chain) => TP alone ~1.3x (=> ~6 t/s), TP+A1 ~6.5-7.5 t/s, long-ctx does
NOT halve. 10-12+ t/s requires a new workstream C: GLM decode-side
attention/dense/indexer split (DeepSeek decode TP = in-repo pattern).

AMENDED PLAN:
- A (go now): A0 attribution; A1 fused sum8 with honest 0.9-1.1 ms/layer
  target and epsilon/ppl validation (not bit-diff); expert-ownership mask
  designed in.
- B prerequisites: (1) memory spike PASSES with a mitigation chosen —
  preferred: TP + partially-streamed expert tail (~80 GiB resident/box,
  lift the TP<->streaming exclusion; ~few ms/token at the measured hit
  curve); (2) B2 rewritten as an 8-slot owned-kernel family (zero-filled
  unowned slots, fixed reduction order) modeled on the DeepSeek owned
  kernels — never bare skip; (3) B1 absorbs: dual tagged seq spaces, GEQ
  waits, event-semantics big gates + stale-payload stress test on GB10,
  release-on-failure invariant, sync-free slab accessor, BOTH Apple gates
  lifted (ds4_tp_validate_engine_options AND ds4_engine_tp_bind/callbacks),
  owned-shape-only weight residency (residency_skip is a CUDA stub), DVFS
  check via nvidia-smi -lgc, OOM guards + watchdog pause on both boxes.
- C (new, optional, the real 2x): GLM decode attention/indexer split.
  Scope after B1 lands; without it the honest deliverable is 6.5-7.5 t/s.
