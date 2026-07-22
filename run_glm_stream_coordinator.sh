#!/bin/bash
# Long-context pipeline + SSD streaming, coordinator half (this box, SECOND).
# No -p -> interactive chat.  GLM_CTX/POOL must match the worker.
cd "$(dirname "$0")"
export DS4_CUDA_WEIGHT_CACHE=1
export DS4_GLM_CUDA_STREAMING=1
export DS4_GLM_MEMORY_GUARD_RESERVE_GB=12
exec ./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
    --cuda --ssd-streaming --ssd-streaming-full-layers 0 \
    --ssd-streaming-cache-experts "${POOL:-7000}" \
    --role coordinator --layers 0:39 --listen 10.0.0.1 9911 \
    -c "${GLM_CTX:-32768}" "$@"
