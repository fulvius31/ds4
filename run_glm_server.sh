#!/bin/bash
# DEFAULT GLM 5.2 API server: tensor-parallel + SSD streaming + fp8 KV, 400k ctx.
# Runs on 10.0.0.1 as TP LEADER and serves the OpenAI-compatible API on :8020.
#
# START THIS FIRST, then on 10.0.0.2 (TP order is the reverse of pipeline mode --
# the leader listens, the worker joins):
#
#   DS4_TP_NO_RC_BULK=1 DS4_GLM_FP8_KV_STORE=1 GLM_CTX=400000 POOL=5000 \
#     ./run_glm_tp_worker.sh
#
# DS4_TP_NO_RC_BULK, GLM_CTX, POOL and the fp8 flag must be IDENTICAL on both
# ranks -- the gate schedule is part of the TP identity handshake, so a mismatched
# rank is refused at seq 1 rather than corrupting silently.
#
# WHY DS4_TP_NO_RC_BULK=1 IS NOT OPTIONAL
# ---------------------------------------
# The RC-bulk RDMA path in ds4_tp_big_gate_exchange() (ds4_tp.c:1705-1714)
# faults on the SECOND large batch prefill in a process:
#   ds4-tp: rc bulk completion error 4        (IBV_WC_LOC_PROT_ERR)
#   ds4: TP gate exchange failed (layer 3 gate 1 seq N)
#   -> request returns gen=0, finish=error "tp: gate transport failed", worker dies
#
# Isolated 2026-07-26 over ten controlled runs. The trigger is one large batch
# prefill followed by another; a small request in between clears it (a small
# prompt never uses the big-gate path). NOT caused by: thinking mode, tool
# schemas, --kv-disk-dir, or session rewind -- forcing a fresh session per
# request still fails, which is what proves the stale state lives in the bulk
# transport rather than the session. The CLI never trips it because interactive
# chat only appends, so it does one prefill and then decodes.
#
# The flag does NOT move big gates to TCP -- that was a misreading on my part.
# ds4_tp_big_gate_exchange() has three payload paths tried in order:
#   1. RC-bulk RDMA  (tp_rdma_rc_bulk_*)        <- buggy, skipped by this flag
#   2. plain RDMA big gate (tp_rdma_big_gate_exchange, 32 KB DS4_TP_RDMA_MAX_MSG
#      chunks)                                  <- what we actually run on
#   3. TCP loop over data_fd                    <- only if RDMA is unusable
# So the flag swaps one RDMA posting strategy for an older one. Confirmed on the
# wire: a 4440-token prefill moved 12,265 MiB of RoCE port_xmit_data and 0.0 MiB
# of TCP payload. (data_fd still carries every gate HEADER as the barrier in all
# modes -- see its declaration comment -- so TCP appearing in the path is not
# evidence the payload rides it.)
#
# Verified stable across fresh->large, large->large (twice), and large->small
# with zero errors, and it is still ~1.7x faster than the pipeline config
# (~102 s vs ~174 s for a 4441-token prompt); prefill 56-58 t/s, essentially
# unchanged from the RC-bulk path's 53 t/s, which is itself evidence the payload
# never left RDMA.
#
# Fallback if TP misbehaves: run_glm_server_pipeline.sh (no TP, no mirror,
# also verified against the same four patterns, just slower).
#
# --host 0.0.0.0 so containers (Open WebUI on the docker bridge) can reach it;
# a 127.0.0.1 listener is invisible from inside a bridge-network container.
cd "$(dirname "$0")"
export DS4_TP_NO_RC_BULK="${DS4_TP_NO_RC_BULK:-1}"
export DS4_GLM_FP8_KV_STORE="${DS4_GLM_FP8_KV_STORE:-1}"
export DS4_GLM_TP_ATTN_SPLIT=1
export DS4_GLM_TP_SHARED_SPLIT=1
export DS4_CUDA_WEIGHT_CACHE=1
export DS4_GLM_CUDA_STREAMING=1
export DS4_GLM_MEMORY_GUARD_RESERVE_GB="${DS4_GLM_MEMORY_GUARD_RESERVE_GB:-12}"

# The RoCE v2 IPv4 GID index moves across reboots/docker network changes --
# discover it at launch (falls back to the historical index 3).
GID_DIR=/sys/class/infiniband/rocep1s0f0/ports/1
RDMA_GID=3
for i in $(ls "$GID_DIR/gids" 2>/dev/null | sort -n); do
  case "$(cat "$GID_DIR/gids/$i" 2>/dev/null)" in
    0000:0000:0000:0000:0000:ffff:*)
      case "$(cat "$GID_DIR/gid_attrs/types/$i" 2>/dev/null)" in
        *"RoCE v2"*) RDMA_GID=$i; break;;
      esac;;
  esac
done

# KV disk cache stays OFF: it restores a snapshot into the leader's session that
# the mirrored worker is never told about. Set GLM_KV_DISK=1 only if that is
# ever made mirror-aware.
KV_DISK_ARGS=""
if [ "${GLM_KV_DISK:-0}" = "1" ]; then
  KV_DISK_ARGS="--kv-disk-dir $HOME/.ds4/server-kv --kv-disk-space-mb 8192"
fi

exec ./ds4-server -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
    --cuda --ssd-streaming --ssd-streaming-full-layers 0 \
    --ssd-streaming-cache-experts "${POOL:-5000}" \
    --tensor-parallel --transport "${GLM_TRANSPORT:-rdma}" \
    --rdma-device rocep1s0f0 --rdma-gid-index "$RDMA_GID" \
    --role coordinator --listen 10.0.0.1 9911 \
    --host "${GLM_HOST:-0.0.0.0}" --port "${GLM_PORT:-8020}" \
    $KV_DISK_ARGS \
    -c "${GLM_CTX:-400000}" "$@"
