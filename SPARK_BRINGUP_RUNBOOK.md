# DGX Spark Bring-Up & EP Validation Runbook

> Step-by-step for two DGX Sparks + the ConnectX-7 stacking cable, from unboxing
> to validating the Expert-Parallel (EP) branch. Each phase has a **GATE** — if it
> fails, fix or stop before continuing. Companion to
> [DUAL_SPARK_NOTES.md](DUAL_SPARK_NOTES.md) and
> [EP_IMPLEMENTATION_PLAN.md](EP_IMPLEMENTATION_PLAN.md).
>
> Placeholders: `<dev>` = RoCE device (find with `ibv_devices` / `ibstat`, e.g.
> `rocep1s0f0` or `mlx5_0`); `<A_ip>`/`<B_ip>` = the stacking-link IPs you assign
> below. "Spark A" = rank 0 (drives the prompt); "Spark B" = rank 1 (compute peer).

---

## Phase 0 — Physical + link bring-up

1. **Cable** the two ConnectX-7 QSFP ports directly with the bundled stacking cable.
2. **Give the link its own subnet** (do NOT route EP over management Ethernet/Wi-Fi):
   ```sh
   # Spark A
   sudo ip addr add 10.0.0.1/30 dev <iface>   # <iface> = the CX7 net interface (ip link)
   sudo ip link set <iface> up
   # Spark B
   sudo ip addr add 10.0.0.2/30 dev <iface>
   sudo ip link set <iface> up
   # confirm
   ping -c3 10.0.0.2     # from A   (expect <1 ms)
   ```
3. **Jumbo frames** (helps prefill): `sudo ip link set <iface> mtu 9000` on both.

**GATE 0:** `ping` works and RTT is sub-millisecond. If not, the interface/subnet
is wrong — fix before anything else.

---

## Phase 1 — Verify the link is HEALTHY (firmware trap)

Many early units ship throttled to ~12–16 Gbps despite a 200G-negotiated link.
Check bandwidth first:

```sh
# Spark A (server)
ib_write_bw -d <dev> -i 1 -p 12000 -F --report_gbits --run_infinitely
# Spark B (client)
ib_write_bw -d <dev> -i 1 -p 12000 -F --report_gbits --run_infinitely <A_ip>
```

- **Healthy:** ~92–109 Gbps per rail (~185–197 Gbps if you load both rails).
- **Throttled (~12–16 Gbps):** update **ConnectX-7 firmware** (to ~1.108.20 / past
  28.45.4028) **and do a full physical NIC power-drain** (not just a reboot), then
  recheck. (NVIDIA Dev Forum threads 366266 / 370035.)

**GATE 1:** bandwidth is in the ~100–190 Gbps range. If stuck at ~13 Gbps after
firmware + power-drain, the EP perf story is dead on arrival — stop and debug HW.

---

## Phase 2 — The go/no-go latency measurement (this decides if EP is worth it)

This is the number nobody published and that the whole EP bet rests on.

```sh
# RDMA latency floor (capture the table)
# Spark A
ib_write_lat -d <dev> -i 1 -F
# Spark B
ib_write_lat -d <dev> -i 1 -F <A_ip>
# read t_typ/t_avg (usec) — hope ~1-2 us
```

Then the real thing — small-message NCCL all-reduce across the two hosts. Build
nccl-tests (github.com/NVIDIA/nccl-tests), then:

```sh
export NCCL_NET_PLUGIN=none NCCL_IB_MERGE_NICS=1 NCCL_IB_HCA=<dev> NCCL_DEBUG=WARN
mpirun -np 2 -H 10.0.0.1,10.0.0.2 \
  -x NCCL_NET_PLUGIN -x NCCL_IB_MERGE_NICS -x NCCL_IB_HCA \
  ./build/all_reduce_perf -b 16K -e 32K -f 2 -g 1
# read the time(us) column at 16K-32K
```

**GATE 2 (the decision):**
- `time(us)` ≲ **20–30 µs** → EP's per-token collective cost is ~1–2 ms (≈1.5% of
  the ~73–92 ms/token decode). **Proceed.**
- Much higher, or NCCL logs show it fell back to TCP (busbw ~1–3 GB/s) → re-check
  `NCCL_NET_PLUGIN=none` + `NCCL_IB_MERGE_NICS=1` + firmware. If it's genuinely
  high after that, EP will likely lose to pipeline mode — **stop and reconsider**;
  use two independent single-Spark servers instead.

---

## Phase 3 — Build ds4 EP on BOTH Sparks (same commit)

```sh
git clone <repo> ds4 && cd ds4 && git checkout ep-expert-parallel   # or git pull
make cuda-spark-ep
sh tests/run_ep_selftest.sh        # -> ds4_ep selftest: OK
```

Do this identically on both Sparks. **GATE 3:** clean build + self-test OK on both.

---

## Phase 4 — Baselines (the numbers EP must beat)

Record these single-machine references first, with `--ctx 32768 --nothink` greedy:

```sh
# (a) single Spark, Flash q2 — the throughput a lone Spark gives (~13.75 t/s expected)
./download_model.sh q2-imatrix
./ds4 -m gguf/<flash-q2>.gguf --cuda --ctx 4096 --nothink -n 64 -p "Count to ten."

# (b) PRO pipeline mode across the two Sparks (the oracle EP must beat for PRO)
#     uses the SPLIT halves + --layers (this is the EXISTING distributed mode)
./download_model.sh pro-q4-layers00-30        # on A
./download_model.sh pro-q4-layers31-output    # on B
# A:  ./ds4 -m gguf/...Layers00-30.gguf  --role coordinator --layers 0:30 --listen <A_ip> 1234
# B:  ./ds4 -m gguf/...Layers-31-output.gguf --role worker --layers 31:output --coordinator <A_ip> 1234
# record decode tok/s
```

**Note the model-file rule:** *pipeline* mode uses the **split** PRO halves; *EP*
uses the **full** model on each rank (symmetric). For the PRO EP test below use a
**single full PRO GGUF** (`pro-q2-imatrix`, ~430 GB) with `--ssd-streaming`, since
the Q4 download is split-only.

---

## Phase 5 — EP CORRECTNESS (Flash — fits one rank, no streaming needed)

Flash fits one Spark, so each rank loads the full model; EP just splits the expert
math. This proves the mask + all-reduce + bootstrap are correct.

```sh
# Spark A (rank 0)
DS4_EP_WORLD_SIZE=2 DS4_EP_RANK=0 DS4_EP_MASTER_ADDR=10.0.0.1 DS4_EP_MASTER_PORT=29500 \
NCCL_NET_PLUGIN=none NCCL_IB_MERGE_NICS=1 NCCL_IB_HCA=<dev> \
./ds4 -m gguf/<flash-q2>.gguf --cuda --ctx 4096 --nothink -n 64 -p "Count to ten."

# Spark B (rank 1) — same prompt; runs in lockstep, greedy => identical tokens
DS4_EP_WORLD_SIZE=2 DS4_EP_RANK=1 DS4_EP_MASTER_ADDR=10.0.0.1 DS4_EP_MASTER_PORT=29500 \
NCCL_NET_PLUGIN=none NCCL_IB_MERGE_NICS=1 NCCL_IB_HCA=<dev> \
./ds4 -m gguf/<flash-q2>.gguf --cuda --ctx 4096 --nothink -n 64 -p "Count to ten."
```

At startup A should print `ds4: EP enabled, rank 0/2 owns experts [0, 128)`.
Compare rank 0's output to the **single-rank** baseline from Phase 4(a) with the
same prompt.

**GATE 5:** EP rank-0 output matches single-rank **within FP tolerance** (greedy
tokens should match; it is not bit-identical because the all-reduce reorders the
expert sum). If the output diverges badly or it hangs at the first layer, the
collective/bootstrap/mask has a runtime bug — capture `NCCL_DEBUG=INFO` and the
`--trace` log. **Do not proceed to PRO until Flash EP is correct.**

---

## Phase 6 — EP PERFORMANCE (PRO — the actual payoff)

Only after GATE 5 passes. PRO doesn't fit one Spark, so use `--ssd-streaming` (EP
load-skip then streams only each rank's owned half):

```sh
./download_model.sh pro-q2-imatrix     # full single PRO GGUF on BOTH Sparks

# Spark A (rank 0)
DS4_EP_WORLD_SIZE=2 DS4_EP_RANK=0 DS4_EP_MASTER_ADDR=10.0.0.1 DS4_EP_MASTER_PORT=29500 \
NCCL_NET_PLUGIN=none NCCL_IB_MERGE_NICS=1 NCCL_IB_HCA=<dev> \
./ds4 -m gguf/<pro-q2>.gguf --cuda --ssd-streaming --ctx 8192 --nothink -n 64 -p "..."
# Spark B (rank 1): same, DS4_EP_RANK=1
```

Compare PRO decode tok/s **EP-2 vs the pipeline-2 baseline** from Phase 4(b).

**GATE 6 (the verdict):**
- **EP > pipeline:** the bandwidth saving beat the collective overhead — EP was
  worth it. Consider productionizing (rank 0 as the API endpoint).
- **EP ≤ pipeline:** the per-token collectives ate the win — report the numbers;
  pipeline mode stays the PRO answer. (This is the honest possible outcome.)

> If CUDA `--ssd-streaming` doesn't bring up PRO at all (it's less exercised than
> Metal streaming), that's a *separate* blocker from EP — verify single-Spark PRO
> streaming works before blaming EP.

---

## Quick reference

**EP env vars**

| Var | Meaning |
|---|---|
| `DS4_EP_WORLD_SIZE` | number of ranks (2) |
| `DS4_EP_RANK` | 0 on Spark A, 1 on Spark B |
| `DS4_EP_MASTER_ADDR` / `DS4_EP_MASTER_PORT` | rank 0's link IP + a port (e.g. 29500) for the ncclUniqueId bootstrap |
| `NCCL_NET_PLUGIN=none` | force native IB/RoCE (avoid the TCP-fallback trap) |
| `NCCL_IB_MERGE_NICS=1` | use both rails for full bandwidth |
| `NCCL_IB_HCA=<dev>` | pin the ConnectX device |
| `NCCL_DEBUG=INFO` | diagnostics if the collective misbehaves |

**The two gotchas** (both real, multi-owner-confirmed): (1) firmware throttle →
~13 Gbps until firmware update **+ NIC power-drain**; (2) silent TCP fallback unless
`NCCL_NET_PLUGIN=none` + `NCCL_IB_MERGE_NICS=1`.

**Model-file rule:** pipeline mode = **split** PRO halves + `--layers`; EP = **full**
model on each rank (Flash full single file; PRO via `pro-q2-imatrix` single + `--ssd-streaming`).

**Decision summary:** GATE 2 (latency <~30 µs) decides whether EP can win at all;
GATE 5 proves it's correct; GATE 6 decides whether it actually beats pipeline. If
GATE 2 fails, run two independent single-Spark servers instead.
