#!/bin/bash
# TP + SSD streaming leader (run on 10.0.0.1). Start this FIRST, then
# run_glm_tp_worker.sh on 10.0.0.2. Best-known config 2026-07-22:
# RDMA/RoCE v2 gates, attention+shared splits, 8000-expert pool.
#   GLM_CTX=8192 ./run_glm_tp_leader.sh -p "prompt" --tokens 200
# For long context lower the pool: GLM_CTX=32768 POOL=6000 ...
# After a reboot re-run: sudo cpupower idle-set -D 100 (both boxes).
cd "$(dirname "$0")"
export DS4_GLM_TP_ATTN_SPLIT=1
export DS4_GLM_TP_SHARED_SPLIT=1
export DS4_CUDA_WEIGHT_CACHE=1
export DS4_GLM_CUDA_STREAMING=1
export DS4_GLM_MEMORY_GUARD_RESERVE_GB="${DS4_GLM_MEMORY_GUARD_RESERVE_GB:-12}"
# The RoCE v2 IPv4 GID index moves across reboots/docker network changes —
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
exec ./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
    --cuda --ssd-streaming --ssd-streaming-full-layers 0 \
    --ssd-streaming-cache-experts "${POOL:-8000}" \
    --tensor-parallel --transport rdma \
    --rdma-device rocep1s0f0 --rdma-gid-index "$RDMA_GID" \
    --role coordinator --listen 10.0.0.1 9911 \
    -c "${GLM_CTX:-8192}" "$@"
