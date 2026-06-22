# Two DGX Sparks — TP/EP Feasibility, Interconnect Research & Phase-0 Plan

> Companion to the **"Combining two DGX Sparks"** section in
> [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md). That section covers *running* two
> Sparks today (pipeline-parallel over the cable). This doc captures the deeper
> findings: whether ds4 could do **tensor parallelism**, what the
> **dual-Spark interconnect actually measures** (from public sources), and the
> **Phase-0 test** that decides if any of it is worth building.
>
> Status: analysis only. **ds4 has no tensor/expert parallelism and no
> NCCL/RDMA code today** — this is a feasibility study, not a feature.
> Code line refs are starting points against the analyzed commit.

---

## 0. TL;DR

- **Possible? Yes. Worth it? Only for PRO, only as Expert Parallelism (EP), only after a hardware check.**
- ds4's distinctive features (single MLA latent KV head, hyper-connections, global indexer) **resist classic tensor parallelism** and force replication, so dense TP's memory/FLOP wins mostly evaporate. The only axis that pays off is **sharding the MoE experts (EP)** — and only for **PRO**, which doesn't fit one Spark.
- **Flash: don't split a single stream.** It fits one Spark; run **two independent instances** for throughput instead.
- The whole EP win is gated on two **unverified** facts that only your hardware settles: (1) the real small-message all-reduce latency (target <~30 µs), and (2) whether EP actually beats the **existing pipeline mode** for PRO.
- Public research confirms EP's *precondition* (decode is memory-bandwidth-bound, the link is ~99.9% idle during decode) but the **latency side is thin**: the RDMA floor (~1–1.5 µs) is **stated, not shown** in any perftest table, and **nobody has published the all-reduce latency smoke test**. The hard latency table is the thing you must generate yourself (§5.1).

---

## 1. The three ways to use two Sparks

| Approach | In ds4 today? | Helps what | When to use |
|---|---|---|---|
| **Two independent instances** + load balancer | ✅ yes (deployment) | aggregate **throughput** (~2×) | serving many requests / agents / batch / Flash |
| **Pipeline parallel** (layer split, TCP) | ✅ yes (`ds4_distributed.c`) | **capacity** + long **prefill** | run a model too big for one box (PRO Q4); decode is *slower* |
| **Tensor / Expert parallel** (weight split, NCCL) | ❌ no — this doc | single-stream **decode** for PRO | only if Phase-0 checks pass; major build |

**Unifying principle:** splitting one stream across two Sparks helps *only when the
working set doesn't fit or compute on one Spark*, and **which resource binds
decides the technique**:

| Regime | What binds one Spark | Right technique | In ds4? |
|---|---|---|---|
| PRO, any context | weights (~405 GiB experts) | **EP** (shard experts) | buildable (this doc) |
| Flash, normal context | nothing — it fits | none → 2 independent instances | n/a |
| Flash, extreme (~1M) context | KV cache | context/sequence parallelism | ds4 lacks it; **EP would not help** (KV is replicated) |

---

## 2. Can ds4 do tensor parallelism?

### 2.1 Why classic TP doesn't fit this architecture

Standard transformer TP splits attention by KV head and the FFN by column/row,
cutting both memory and FLOPs per device. DeepSeek V4 breaks all three
assumptions (per-token layer graph: `layer_forward_raw_swa_one`, `ds4.c:9679`):

1. **Single latent KV head** (`n_head_kv=1`, `head_dim=512`, `ds4.c:184-185`).
   One 512-wide latent KV is shared by all 64/128 query heads (every head reads
   the same rows, `ds4.c:8930`). Nothing to head-split → the KV projection, FP8
   quantize, and the **entire KV cache must be replicated** on both Sparks. No
   KV-memory win; redundant KV compute. *This is the structural killer for
   attention TP.*
2. **Hyper-connections mix the full hidden state.** HC-pre/HC-post run a Sinkhorn
   mix over all 4×n_embd at every sublayer (`ds4.c:6332/6457/6512`). Not
   separable along any feature axis → the **full hidden vector must exist on both
   devices at every sublayer boundary**, forcing an all-*reduce* model (never the
   cheaper reduce-scatter).
3. **The indexer top-k is one global mask** shared by all heads (`ds4.c:9157`) —
   no shard axis, must be replicated.

Net: KV, HC, router, shared expert, indexer, and the output head all end up
**replicated**. The *only* thing that genuinely shards and saves memory is the
routed experts.

### 2.2 The right axis: Expert Parallelism (EP), PRO-only

This part is clean:

- Expert weights are **flat per-expert-contiguous tensors** (`ds4.c:5529`) —
  splitting 384 PRO experts into two id-ranges is a pure pointer-offset split.
- Routing is deterministic per token and the router is tiny (~2 MiB/layer), so
  you **replicate the router bit-for-bit** → **no token-dispatch all-to-all**.
  Each Spark computes whichever of the 6 active experts it owns.
- The down-projection already accumulates additively (`ds4.c:6227`), so the whole
  thing collapses to **one all-reduce of the n_embd routed output per layer**
  (`ds4.c:19207`), folded with the (replicated) shared expert.

### 2.3 Partitioning plan

| Component | Approach | Comms |
|---|---|---|
| Routed MoE experts (dominant per-token read; mandatory for PRO) | **shard** 384/256 experts into two id-ranges via per-expert stride (`ds4.c:5529`); pin owned range in the streaming-expert cache | folded into 1 all-reduce/layer |
| Router (`ffn_gate_inp`) + `tid2eid` hash | **replicate** bit-for-bit | none |
| Shared expert (Q8_0 SwiGLU, every token) | **replicate** (~1 GiB) | none |
| MLA KV latent (`n_head_kv=1`) | **replicate** projection, norm, FP8, full KV cache (2× KV memory, OK on 128 GB) | none |
| Query heads / `q_b` / attn out-proj | **optional** column-parallel by head (respect `n_out_group`, `ds4.c:7120`) | +1 all-reduce/layer — **not recommended** (no memory win) |
| Hyper-connection Sinkhorn | **replicate** (not separable) — dictates full-n_embd all-reduce | none itself |
| Indexer top-k mask | **replicate** (recompute on both) | none |
| Output head (vocab GEMM) | **replicate** (input already mirrored; fires once/token) | none |

### 2.4 Communication design & latency math

- **Volume is trivial; round-trip *count* is the killer.** One all-reduce/layer
  for pure EP. Each is n_embd×4 = **16 KB/token (Flash) / 28 KB/token (Pro)** →
  ~0.7–1.7 MB/token total. Microseconds at line rate.
- **But decode is serial** (token N+1 needs N's logits), so every collective is on
  the critical path with no overlap: **~43–61 serial all-reduces/token** (EP-only);
  ~86–122 if attention is also split.
- Against the **~73–92 ms/token** Spark decode budget:
  - **NCCL/RoCE**, optimistic ~10–30 µs/op → EP-only ≈ **~1.2 ms/token (~1.5%)**. Tolerable.
  - **TCP** (the only transport ds4 has, `ds4_distributed.c:1043`, ~30–100 µs syscall RTT) → EP-only ≈ **~2–6 ms**, and per-layer TCP barriers are a non-starter.
- **Conclusion: the collective MUST be NCCL/RDMA over RoCE.** Reusing the TCP
  pipeline transport is a non-starter for per-layer barriers.

### 2.5 Missing infrastructure (it's greenfield)

Grep-verified: **zero** collectives anywhere; CUDA backend is a single-device
singleton (`cudaSetDevice(0)`, one global cuBLAS at `ds4_cuda.cu:2256`, no
rank/world); `-lnccl` absent from the Makefile (`Makefile:33`). To build EP:

- `ds4_gpu_all_reduce` in the ABI + a NCCL-backed CUDA impl; persistent
  `ncclComm_t` + dedicated stream.
- rank/world model + one-process-per-Spark launcher with `ncclUniqueId` bootstrap
  (can reuse the existing TCP socket for rendezvous).
- RoCE/IB config for ConnectX-7 (`NCCL_NET=IB`, HCA/GID) — no NVLink peer path
  between two hosts.
- EP-aware MoE dispatch (owned-expert id-range, `ds4_gpu.h:842`); owned-range
  pinning in the streaming-expert cache (`ds4_cuda.cu:171`); replicated KV
  allocator; a new **symmetric** parallel mode (not the coordinator/worker
  pipeline) branching at `ds4_session_eval_internal` (`ds4.c:27065`).

### 2.6 Effort & risks

- **Greenfield NCCL stack is the bulk of the work** — weeks before one token
  decodes correctly, and it **breaks ds4's deliberate self-contained, no-NCCL
  ethos** (a permanent heavy build/runtime dep + per-deployment fabric config).
- **NCCL-on-GB10-aarch64/RoCE viability is unverified** — the single biggest
  unknown; de-risk first.
- **KV/HC replication erases two of TP's classic wins** — no KV-memory saving,
  redundant KV compute, all-reduce (not reduce-scatter).
- **Decode load imbalance at batch=1** — the 6 active experts split ~3/3 on
  average but with high variance; the heavier rank stalls the all-reduce each
  layer. Prefill amortizes this; decode does not.

---

## 3. Does EP make sense? (the verdict)

**For PRO: yes, conditionally. For Flash: no.** The interconnect research nudged
this *slightly more favorable* — because the one thing it pinned down is exactly
EP's precondition.

**Favorable (precondition confirmed):**
- Decode is **memory-bandwidth-bound, link ~99.9% idle** (Myers measured ~134 Mbps
  link use during TP decode) → halving per-Spark expert traffic is the right
  lever, and the link has huge headroom for small all-reduces.
- An ***asserted* RDMA floor ~1–1.5 µs** (stated in secondary sources, never shown
  in a perftest output table — the **weakest** of these signals), healthy NCCL
  busbw ~24 GB/s → a 16–28 KB all-reduce *plausibly* lands in low-tens-of-µs.
- PRO **requires** a weight split anyway; EP is the better split than dense TP.

**Unresolved (these decide it, both need hardware):**
1. The **real all-reduce latency** is unmeasured (<30 µs only *inferred*).
2. **EP must beat the existing pipeline mode**, not single-Spark. Pipeline splits
   PRO with **1** hop/token; EP pays **43–61** but parallelizes the dominant
   expert read. EP *should* win, but only if collective overhead stays small.

**Sobering frame:** even if EP works, PRO decode on two Sparks is **single-digit
tok/s** (research analogues: 405B-scale ~1.8 tok/s, large MoE ~11 tok/s on TP=2).
EP makes an already-slow PRO experience *somewhat less slow* — weigh that against
weeks of greenfield NCCL work.

---

## 4. Interconnect research — what's actually published

Searched ~8 months of post-launch material (Oct 2025 → Jun 2026). **No one
published the exact `all_reduce_perf` small-message `time(us)` for two stacked
Sparks.** Everything public is large-message *bandwidth*, not the latency floor.

### 4.1 Measured numbers (with sources)

| Metric | Measured value | Source |
|---|---|---|
| **RDMA write latency** (`ib_write_lat`) | **~1–1.5 µs one-way — *stated, not shown*** (no clean perftest output table is published anywhere; asserted in prose / quoted as guidance) | Thomas (LinkedIn); sparkrun.dev (guidance value) |
| NCCL bandwidth (healthy) | **~24 GB/s busbw** (all_gather, large msg) | NVIDIA Dev Forum 366373 |
| RDMA bandwidth | ~92–109 Gbps/rail, **~185–197 Gbps aggregate** (both rails) | NVIDIA playbook; ServeTheHome |
| iperf3 (TCP) | ~92–111 Gbps single stream post-fix; ~25 untuned; 160–198 w/ jumbo + parallel | ServeTheHome; Forum 370035 |
| Large-msg NCCL latency | ~543 µs @ 17 GB; ~1.2 ms @ 32 MB (bandwidth-bound, **not** a floor) | forums; Macnica (4-node) |
| 2-node inference (PP=2, batch 64) | GPT-OSS-120B ~464–505 tok/s; Llama-3.1-8B ~531–581 tok/s | StorageReview |
| 2-node inference (TP=2, single stream) | 405B ~1.8 tok/s; 70B ~2.9 tok/s; GLM-4.6 MoE ~11 tok/s | Myers (LinkedIn) |
| DeepSeek V4 Flash, TP=2, FP8 | ~41 tok/s decode @ **1M context** (vs 12–15 @ 131K single) — *different engine, long-context capacity effect, not a fixed-context TP speedup* | Flowtivity |
| Link utilization during TP decode | **~134 Mbps** (~0.07% of 200G) — memory-bound | Myers (LinkedIn) |

### 4.2 Two gotchas that gate *any* result

Both confirmed by 3+ independent owners + NVIDIA mods. On a bad stack, any
benchmark (and any EP build) looks like failure:

1. **ConnectX-7 firmware throttle.** Early units negotiate 200G but collapse to
   **~12–16 Gbps** until firmware upgrade (~1.108.20, past 28.45.4028) **+ a full
   physical NIC power-drain** (not just reboot). (Forum 370035)
2. **Silent TCP fallback.** Default NCCL mis-detects PCIe (Gen1 x1) and falls back
   to TCP (~1–3 GB/s) unless `NCCL_NET_PLUGIN=none` + `NCCL_IB_MERGE_NICS=1`.
   (Forum 366266)

**Architecture note:** each QSFP port = **2×100G MACs over two PCIe Gen5 x4
links**, so a single stream/rail caps ~100G — you must load **both rails** to
approach 200G. (ServeTheHome)

---

## 5. Phase 0 — getting a value

### 5.1 The definitive test (run on the hardware, day one)

After confirming firmware is current and `ib_write_bw` shows ~185–197 Gbps aggregate:

```sh
# native IB/RoCE, both rails merged — NOT TCP fallback
export NCCL_NET_PLUGIN=none NCCL_IB_MERGE_NICS=1
# small-message all-reduce latency across the two Sparks (1 GPU per rank)
all_reduce_perf -b 16K -e 32K -f 2 -g 1
# read the time(us) column at 16K–32K
```

- `time(us)` ≲ 20–30 µs → latency budget holds → EP for PRO worth building.
- Higher, or bandwidth columns show TCP → fix the stack first / reconsider.

**Also generate the RDMA latency floor** — the perftest table no public source
actually shows (it's only ever stated). Use the RoCE device name from
`ibv_devices` / the NVIDIA connect-two-sparks guide (e.g. `rocep1s0f0`):

```sh
# Spark A (server)
ib_write_lat -d rocep1s0f0 -i 1 -F
# Spark B (client) → A's RoCE-link IP; -a sweeps all sizes, -s 16384 for the 16 KB point
ib_write_lat -d rocep1s0f0 -i 1 -F <A_link_ip>
# read t_typical / t_avg (usec); ib_send_lat is the analogous send-latency check
```

A ~1–2 µs `t_avg` here confirms the floor the whole inference rests on; the
all-reduce sits a small multiple above it. This is the table to capture, since no
one has published it.

Also run **Phase 0.5 (a day):** PRO on two Sparks in the **existing pipeline
mode**, record decode tok/s. That's the number EP must beat and the absolute
ceiling you're working with.

### 5.2 Finding a proxy value online *now* (before hardware)

Priority: NCCL all-reduce latency → NCCL all-gather small-msg → RDMA verbs
latency (`ib_send_lat`/`ib_write_lat`, the floor) → MPI/OSU all-reduce.

```
# A — the exact thing
DGX Spark all_reduce_perf time us latency two nodes
nccl-tests all_reduce_perf "DGX Spark" 8 bytes latency microseconds
GB10 nccl-tests all_reduce two nodes "time(us)"

# B — close proxy (full nccl-tests dump)
DGX Spark all_gather_perf -b 8 -e 128M two nodes output

# C — RDMA verbs latency (the floor; most likely to exist)
DGX Spark ib_send_lat latency microseconds
DGX Spark ib_write_lat t_avg RoCE
ConnectX-7 GB10 RDMA latency ib_send_lat

# D — MPI / OSU
DGX Spark osu_allreduce latency two nodes

# Source-scoped (paste into Google)
site:forums.developer.nvidia.com "DGX Spark" all_reduce latency
site:forums.developer.nvidia.com "DGX Spark" nccl-tests
site:github.com "DGX Spark" all_reduce_perf
site:github.com "DGX Spark" ib_send_lat OR ib_write_lat
```

Where to look first: **NVIDIA `dgx-spark-playbooks` repo** (connect-two-sparks
Issues/PRs), **NVIDIA Dev Forums** (sort by newest), GitHub cluster repos (`eugr`
launch-cluster, `Sggin1/DGX-SPARK`), ServeTheHome / Level1Techs, LinkedIn (Thomas,
Andrew Myers), reddit r/LocalLLaMA.

### 5.3 How to read what you find

- **Best realistic outcome:** you find NCCL busbw ≈ 24 GB/s tables + a *stated*
  (rarely *shown*) `ib_write_lat` ≈ 1–1.5 µs, and *infer* a 2-rank small-message
  all-reduce ≈ **~5–25 µs** → target plausibly met, **unconfirmed**. Note even the
  verbs-latency figure is usually asserted in prose, not backed by a perftest
  dump — so the hard latency table is something you generate (§5.1), not find.
- **Discard the number** if: old firmware (bw ~12–16 GB/s or NCCL ~3 GB/s), TCP
  not native IB (no `NCCL_NET_PLUGIN=none`/`NCCL_IB_MERGE_NICS=1`), or a
  switch/3–4-node mesh rather than the direct 2-unit cable.
- **Don't be fooled by:** NVLink/intra-node NCCL latency (sub-µs, not comparable
  to two hosts over RoCE), "200 Gb/s" spec claims, or the EXO "2.8×" (1 Spark + 1
  Mac over 10 GbE).

---

## 6. Incremental path

0. **De-risk (days):** firmware fix → `all_reduce_perf -b 16K -e 32K` over the
   cable. If not <~30 µs → **stop**. Also record pipeline-mode PRO decode tok/s.
1. **ABI (1):** add `ds4_gpu_all_reduce` + CUDA/NCCL impl; rank/device in
   `ds4_gpu_init`; `ncclUniqueId` bootstrap over a TCP rendezvous. Validate
   bit-identical decode with replicated weights.
2. **EP for PRO (2):** replicate router/shared-expert/KV/HC/indexer; shard 384
   experts by id-range; owned-range pinning; one all-reduce of routed output
   before the FFN sum (`ds4.c:19207`). Verify vs pipeline mode (PRO doesn't fit
   one Spark, so pipeline is the reference).
3. **Measure honestly (3):** EP-2-Spark decode tok/s **vs** pipeline-2-Spark. If
   EP doesn't beat pipeline, collective overhead ate the win → reconsider.
4. **Optional (4):** query-head column-parallel attention if attention is a
   measured bottleneck. Expect marginal gains.
- **Skip entirely for Flash** — fits one Spark; run two independent instances.

---

## 7. Sources

- NVIDIA `dgx-spark-playbooks` — connect-two-sparks performance benchmarking guide: `github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/performance_benchmarking_guide.md`
- NVIDIA Dev Forum 366266 — NCCL 3 GB/s / PCIe Gen1 x1 (the config-gotcha thread)
- NVIDIA Dev Forum 366373 — NCCL 3→24 GB/s after firmware fix
- NVIDIA Dev Forum 370035 — QSFP 13–16 → 111 Gbps after firmware
- ServeTheHome — "The NVIDIA GB10 ConnectX-7 200GbE Networking is Really Different"
- Thomas — "DGX Spark Network Benchmarks: RDMA Performance over RoCE" (LinkedIn) — *shows the bandwidth output; the ~1.5 µs latency is stated, likely alongside the perftest commands rather than a shown latency table*
- Andrew Myers — "Testing Three Models on Two Sparks" (LinkedIn)
- StorageReview — "NVIDIA DGX Spark Cluster Review"
- Macnica — distributed learning on four DGX Sparks (4-node; switch, not the 2-unit cable)
- Flowtivity — "DeepSeek V4 Flash at 1M Context on Two DGX Sparks"

> Caveats on the sources: "200 Gb/s" is spec, not measured; the Flowtivity 3×
> is a long-context capacity result in a different engine (not a fixed-context TP
> speedup, and EP's replicated KV wouldn't deliver it); the "23,477 tok/s" Qwen3
> figure is prefill at batch=1 (decode ~11.7); "405B on two Sparks" is a capacity
> claim, not a published speed benchmark. And critically: the `ib_write_lat`
> ~1.5 µs figure is **stated in prose, not shown** as a perftest table anywhere
> public — the NVIDIA connect-two-sparks guide gives the latency *commands* but no
> numbers, so the latency table must be self-generated (§5.1).
