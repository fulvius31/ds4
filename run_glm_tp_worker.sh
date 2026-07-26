#!/bin/bash
# TP + SSD streaming worker (run on 10.0.0.2, after the leader is up).
# GLM_CTX and POOL must match the leader's values.
cd "$(dirname "$0")"
export DS4_TP_NO_RC_BULK="${DS4_TP_NO_RC_BULK:-1}"
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
    --tensor-parallel --transport "${GLM_TRANSPORT:-rdma}" \
    --rdma-device rocep1s0f0 --rdma-gid-index "$RDMA_GID" \
    --role worker --coordinator 10.0.0.1 9911 \
    -c "${GLM_CTX:-8192}" "$@"
