#!/bin/bash
# Long-context pipeline + SSD streaming, worker half (run on 10.0.0.2 FIRST).
# GLM_CTX up to ~100000; KV is per-box-halved so POOL can stay large.
cd "$(dirname "$0")"
export DS4_CUDA_WEIGHT_CACHE=1
export DS4_GLM_CUDA_STREAMING=1
export DS4_GLM_MEMORY_GUARD_RESERVE_GB=12
exec ./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
    --cuda --ssd-streaming --ssd-streaming-full-layers 0 \
    --ssd-streaming-cache-experts "${POOL:-7000}" \
    --role worker --layers 40:output --coordinator 10.0.0.1 9911 \
    -c "${GLM_CTX:-32768}" "$@"
