#!/bin/bash
# TP + SSD streaming worker (run on 10.0.0.2, after the leader is up).
# GLM_CTX and POOL must match the leader's values.
cd "$(dirname "$0")"
export DS4_GLM_TP_ATTN_SPLIT=1
export DS4_GLM_TP_SHARED_SPLIT=1
export DS4_CUDA_WEIGHT_CACHE=1
export DS4_GLM_CUDA_STREAMING=1
export DS4_GLM_MEMORY_GUARD_RESERVE_GB=12
exec ./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
    --cuda --ssd-streaming --ssd-streaming-full-layers 0 \
    --ssd-streaming-cache-experts "${POOL:-8000}" \
    --tensor-parallel --transport rdma \
    --rdma-device rocep1s0f0 --rdma-gid-index 3 \
    --role worker --coordinator 10.0.0.1 9911 \
    -c "${GLM_CTX:-8192}" "$@"
