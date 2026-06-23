# Expert Parallelism (EP) — Implementation Plan

> Build plan for sharding DeepSeek V4 **PRO** routed experts across two DGX
> Sparks. Rationale, feasibility, and the interconnect research are in
> [DUAL_SPARK_NOTES.md](DUAL_SPARK_NOTES.md); this doc is the *how to build it*.
>
> **Branch:** `ep-expert-parallel`.
> **Compile-time opt-in:** everything is gated behind `-DDS4_EP_BUILD`, so default
> Metal/CUDA/ROCm/CPU builds are untouched.

---

## 1. Scope & non-goals

- **In scope:** Expert Parallelism for **PRO** — shard the routed experts across
  N symmetric ranks (2 Sparks), replicate everything else, one all-reduce of the
  routed FFN output per layer.
- **Out of scope:** dense tensor parallelism of attention/projections (the single
  MLA KV head + hyper-connections force replication — no win); **Flash** (fits one
  Spark → run two independent instances); context/sequence parallelism (a
  separate feature for >fit-on-one KV).

## 2. Decision gate (do NOT skip)

This build is **conditional** on Phase-0 measurements once hardware is in hand
(see DUAL_SPARK_NOTES §5):
1. `ib_write_lat` + `all_reduce_perf -b 16K -e 32K` over the ConnectX-7 link →
   small-message all-reduce ≲ ~30 µs.
2. PRO decode tok/s in the **existing pipeline mode** (the baseline EP must beat).

If (1) fails or (2) is already acceptable, stop. The code below can be *written*
ahead of time, but the go/no-go is a hardware measurement.

## 3. Design

Symmetric N-rank (one process per Spark), **not** the coordinator/worker pipeline:

- **Replicated on every rank:** router (`ffn_gate_inp` + `tid2eid`), shared expert,
  all attention (KV latent, Q heads, out-proj), KV cache (raw + compressed +
  indexer), hyper-connections, indexer mask, output head, sampling.
- **Sharded:** routed experts only — each rank owns a contiguous id-range and
  loads/pins only that slice.
- **Collective:** after each rank computes its owned experts' additive down-proj
  contribution, **one `all_reduce(SUM)` of the n_embd routed vector** per layer;
  then add the (replicated) shared-expert output. 28 KB/token (PRO), ~61
  all-reduces/token, on the critical path.
- **Logits** end up identical on all ranks (output head is replicated), so any
  rank can sample; rank 0 drives the session/API.

## 4. File-level work

| File | Change | Status |
|---|---|---|
| **`ds4_ep.h` / `ds4_ep.c`** (new) | EP context + contiguous expert-range partition + env parse; self-test | **done (this commit, uncompiled here)** |
| **`ds4_gpu.h`** | collective ABI: `ds4_gpu_collective_init`, `ds4_gpu_all_reduce_f32`, `ds4_gpu_collective_shutdown` (guarded `DS4_EP_BUILD`) | **done (this commit)** |
| `ds4_cuda.cu` | NCCL comm lifecycle (persistent `ncclComm_t` + collective stream) in/near `ds4_gpu_init` (`:2255`); implement `ds4_gpu_all_reduce_f32` via `ncclAllReduce`; owned-expert skip in the routed-MoE dispatch + owned-range pinning in the streaming-expert cache (`ds4_gpu.h:75-98`) | pending hardware |
| `ds4.c` | hold a `ds4_ep_context` on the engine; pass owned-range into the MoE dispatch so non-owned experts are skipped and only the owned slice is loaded (per-expert tensors at layer struct `~3046`, per-expert bytes `~3250`); insert the all-reduce of the routed output before the shared+routed sum | pending |
| `ds4_distributed.c` | reuse the existing TCP rendezvous (`getaddrinfo`/`connect`, framing) to broadcast the 128-byte `ncclUniqueId` from rank 0 to others before `ds4_gpu_collective_init` | pending |
| `Makefile` | `cuda-spark-ep` target: `CUDA_ARCH=` (GB10), add `-DDS4_EP_BUILD`, `-lnccl`, and `ds4_ep.o` to `CORE_OBJS`; NCCL lib path | pending |
| `tools/ep_allreduce_smoke.c` (new, optional) | standalone 2-rank `ncclAllReduce` latency probe = the Phase-0 test in ds4's own toolchain | pending hardware |

## 5. Rank/world bootstrap

- Rank/world from env (`DS4_EP_WORLD_SIZE`, `DS4_EP_RANK`) — already parsed by
  `ds4_ep_context_from_env`. A launcher script starts one `ds4` per Spark.
- Rank 0 calls `ncclGetUniqueId`, broadcasts the 128-byte id to other ranks over
  the **existing `ds4_distributed` TCP socket** (no new transport needed for
  bootstrap), then every rank calls `ncclCommInitRank(world, id, rank)`.
- The per-layer all-reduce uses NCCL/RoCE (native IB), **not** TCP. Requires
  `NCCL_NET_PLUGIN=none`, `NCCL_IB_MERGE_NICS=1`, current ConnectX-7 firmware.

## 6. MoE dispatch change (the core of EP)

In the routed-MoE path, today every rank addresses all `n_total_expert`. Under EP:
1. Router runs replicated → identical `selected[6]` + weights on every rank.
2. Each rank computes only experts where `ds4_ep_owns_expert(ep, e)` is true;
   non-owned selected experts contribute zero locally.
3. Each rank's partial routed output (sum over its owned active experts) is
   **all-reduced (SUM)** → the full routed output on every rank.
4. Add the replicated shared-expert output. HC-post continues as normal.

Loading: each rank `cudaHostRegister`s / pins only its owned contiguous expert
byte-range (contiguous id-range → contiguous bytes via the per-expert stride),
halving resident expert memory — the whole point for PRO.

## 7. Validation plan (needs two Sparks)

1. **Correctness on Flash (fits one rank):** run EP with `DS4_EP_WORLD_SIZE=2`
   (disjoint halves — the default `ds4_ep_expert_range` partition) and compare
   decode to a **single-rank** run. They should match within floating-point
   tolerance — *not* bit-identical, because the all-reduce reorders the expert
   sum (FP add is non-associative). This proves masking + collective + bootstrap
   reassemble the full routed output. (Note: "both ranks own all experts" would
   DOUBLE-count through the sum-all-reduce — the disjoint partition is the point.)
2. **EP correctness on PRO:** the real target; verify PRO decode matches the
   pipeline-mode reference (PRO doesn't fit one Spark, so pipeline is the oracle).
3. **EP performance:** PRO decode tok/s **EP-2-Spark vs pipeline-2-Spark**. EP must
   win or the collective overhead ate the bandwidth saving → reconsider.

## 8. Status

- **Implemented and compiled** (`make cuda-spark-ep` builds clean on aarch64/CUDA
  13.2/NCCL): host partition (`ds4_ep.{h,c}` + self-test), TCP bootstrap, the NCCL
  collectives + router-mask kernel (`ds4_cuda_ep.cu`), the ABI (`ds4_gpu.h`), and
  the full `ds4.c` wiring — engine/graph `ep` field, init/bootstrap/shutdown, the
  per-layer router mask (decode + prefill) and routed-output all-reduce (3 decode
  sites + 1 prefill). All behind `-DDS4_EP_BUILD`; default builds untouched.
- **Correctness-complete, NOT yet perf-optimized:** each rank still *computes* all
  selected experts (non-owned weighted to 0) — correct, but the memory/compute
  *saving* (load only the owned slice via the streaming-selected cache, lever 2)
  is the next step. So EP is expected to be correct first, faster second.
- **Pending two Sparks:** all of §7/§10.3 (run-time correctness + EP-vs-pipeline).
- **Run anywhere with a compiler:** `sh tests/run_ep_selftest.sh` (partition math).

## 9. Risks (recap)

- Greenfield NCCL stack; breaks ds4's self-contained no-NCCL ethos (AGENT.md).
- NCCL-on-GB10-aarch64/RoCE viability unverified.
- Decode load imbalance at batch=1 (6 active experts split unevenly per token).
- The win is only over **pipeline mode**, and only for **PRO** — unproven until §7.3.

## 10. Testing

### 10.1 Host partition logic — runs anywhere with a C compiler (no GPU/CUDA/NCCL)

`ds4_ep.c` has a built-in self-test:

```sh
# from the repo root — either of:
sh tests/run_ep_selftest.sh
# or directly:
cc -DDS4_EP_SELFTEST ds4_ep.c -o /tmp/ds4_ep_selftest && /tmp/ds4_ep_selftest
# expected:  ds4_ep selftest: OK
```

Covers: 384→192/192 and 256→128/128 even splits, remainder spread (7→4/3),
contiguous coverage + disjointness for world sizes 1–8, invalid-arg handling, and
the disabled-context "owns everything" fallback.

> Status: authored but **not run here** (the dev box has no C compiler). The test
> is self-contained and expected to pass on first compile.

### 10.2 Link / collective smoke — needs two Sparks, before any ds4 EP build

After current ConnectX-7 firmware + `NCCL_NET_PLUGIN=none NCCL_IB_MERGE_NICS=1`:

```sh
ib_write_lat   -d rocep1s0f0 -i 1 -F        # one-way RDMA latency floor (the unpublished table)
all_reduce_perf -b 16K -e 32K -f 2 -g 1     # small-message all-reduce; read time(us), want <~30
```

### 10.3 EP correctness & performance — needs two Sparks + the full EP build

1. **EP correctness on Flash** (`DS4_EP_WORLD_SIZE=2`, disjoint halves — Flash fits
   one rank) → decode matches a **single-rank** run within FP tolerance (the
   all-reduce reorders the sum, so close but not bit-identical). Proves masking +
   collective + bootstrap are correct.
2. **EP correctness on PRO** → PRO decode must match the **pipeline-mode** reference
   (PRO doesn't fit one Spark, so pipeline is the oracle).
3. **Perf** → PRO decode tok/s **EP-2-Spark vs pipeline-2-Spark**; EP must win or
   the collective overhead ate the bandwidth saving.
