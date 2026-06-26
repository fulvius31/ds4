# Tensor Parallelism (TP) — Implementation Plan

> Goal: **lower single-stream decode latency for Flash on 2 DGX Sparks** by
> sharding within-layer math across both, not just the experts. Branch:
> `tp-tensor-parallel` (off `ep-expert-parallel`, to reuse the NCCL collective
> stack). Gated behind `-DDS4_EP_BUILD` (shared with EP) + a `DS4_TP_*` runtime switch.

## 1. Why TP (vs the EP we already have)
- EP shards **whole experts** → parallelizes only the MoE compute; attention/HC
  run fully on both ranks (replicated). Bounded by Amdahl on the attention part.
- TP shards **matrices** → can parallelize attention (Q heads) **and** the MoE
  FFN. That extra attention parallelism is the only reason to build TP over EP.
- Both reuse one all-reduce/sublayer; TP is perfectly load-balanced at batch=1
  (EP is not).

## 2. What is / isn't shardable on DeepSeek-V4 Flash
| Component | Strategy | Collective |
|---|---|---|
| Routed experts (gate/up/down) | column-parallel gate+up, row-parallel down | 1 all-reduce (MoE) |
| Shared expert | same | (folded into MoE all-reduce) |
| Attention Q heads (64 → 32/rank) | shard heads; replicate compressed KV latent `c_KV`; row-parallel o_proj | 1 all-reduce (attn) |
| Hyper-connections (Sinkhorn) | **replicated** (operates on full n_embd; doesn't shard cleanly) | — |
| Router / `ffn_gate_inp`, embeddings, output head, KV cache | **replicated** | — |

Net: **~2 all-reduces per layer** (attn o_proj + MoE down), each `n_embd` floats.

## 3. Reuse from the EP branch (no new plumbing)
- `ds4_cuda_ep.cu`: `ds4_gpu_collective_init` / `ds4_gpu_all_reduce_f32` /
  `ds4_gpu_collective_shutdown` — used as-is.
- `ds4_ep.c`: TCP `ncclUniqueId` bootstrap, env parse — reuse; add `DS4_TP_*`
  aliases (world size / rank) or share `DS4_EP_*`.
- `ds4_gpu.h`: the collective ABI.

## 4. Build stages (each independently measurable)
1. **FFN/MoE TP** (biggest compute, lowest risk): each rank loads only its
   `n_ff_exp/world` slice of every expert's gate/up (rows) and down (cols);
   MoE kernels run on the half-width intermediate; all-reduce the down output.
   → **Measure Flash TP-2 resident decode vs 1-Spark. Go/no-go here.**
2. **Attention TP**: shard Q heads (replicate `c_KV`, decompress only owned
   heads via `kv_b`), row-parallel o_proj, all-reduce. → measure again.
3. **Fuse / optimize**: combine the two all-reduces where possible; overlap.

## 5. Correctness oracle
Single-Spark Flash (greedy) is the reference. TP-2 rank-0 greedy tokens must
match within FP tolerance (the all-reduces reorder sums → close, not bit-exact).

## 6. Honest risk
- Flash decode is **latency/overhead-bound** (~18% of bandwidth ceiling), so the
  single-stream win may be small (~1.3–1.8×) or a wash if kernel-launch overhead
  dominates. **Stage 1 measurement decides whether stages 2–3 are worth it.**
- Slicing 256 experts × 43 layers' matrices at load is the main new code.

## 7. De-risk BEFORE building (step 0)
Run the **existing EP branch on Flash, resident** (runbook GATE 5 — never run):
validates the NCCL bootstrap + all-reduce on the two Sparks AND signals whether
cross-Spark sharding helps Flash at all. If EP-2 Flash decode ≥ 1-Spark, the
collective overhead is tolerable → build TP. If it's much slower, TP won't save
single-stream either → reconsider.
