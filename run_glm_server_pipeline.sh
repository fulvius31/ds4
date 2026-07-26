#!/bin/bash
# GLM 5.2 API server, PIPELINE variant: pipeline + SSD streaming + fp8 KV.
# No tensor parallelism, so none of the TP mirror/bulk failure modes apply.
#
# START THE WORKER FIRST on 10.0.0.2 (it dials the coordinator and retries;
# this is the opposite of TP, where the leader listens and the worker joins):
#
#   DS4_GLM_FP8_KV_STORE=1 DS4_CUDA_WEIGHT_CACHE=1 DS4_GLM_CUDA_STREAMING=1 \
#   DS4_GLM_MEMORY_GUARD_RESERVE_GB=12 \
#   ./ds4 -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf --cuda \
#     --ssd-streaming --ssd-streaming-full-layers 0 --ssd-streaming-cache-experts 5000 \
#     --role worker --layers 40:output -c <ctx> --coordinator 10.0.0.1 9911
#
# GLM_CTX and the fp8 flag must MATCH on both ranks.
#
# WHY THIS REACHES MORE CONTEXT THAN TP
# -------------------------------------
# Under TP both ranks carry the FULL KV plane (kv_layers=78 on each), so KV is
# replicated: 19.20 GiB per box at ctx 400k, 36 GiB at 750k. Pipeline splits the
# layers (kv_layers=40 leader / 38 worker), so each box holds only its slice:
# 4.98 + 4.67 GiB at ctx 200k -- roughly half the KV per box. That is what buys
# the higher ceiling, at the cost of throughput: the pipeline serialises staging
# and compute across the pair instead of overlapping both ranks.
#
# Caveat: the planner OVER-reports pipeline KV (prints 9.60 GiB at 200k where the
# graph actually allocates 4.98) because it assumes full layers, and the memory
# guard gates on the planned figure -- so it can refuse a context that would in
# fact have fit.
#
# SSD streaming is mandatory here. Pipeline-RESIDENT keeps ~96 GiB of weights per
# box and leaves no room for a large KV plane; this script previously shipped
# without the streaming flags and silently capped out at ctx 12288.
cd "$(dirname "$0")"
export DS4_GLM_FP8_KV_STORE="${DS4_GLM_FP8_KV_STORE:-1}"
export DS4_CUDA_WEIGHT_CACHE=1
export DS4_GLM_CUDA_STREAMING=1
export DS4_GLM_MEMORY_GUARD_RESERVE_GB="${DS4_GLM_MEMORY_GUARD_RESERVE_GB:-12}"

exec ./ds4-server -m gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
    --cuda --ssd-streaming --ssd-streaming-full-layers 0 \
    --ssd-streaming-cache-experts "${POOL:-5000}" \
    --role coordinator --layers 0:39 --listen 10.0.0.1 9911 \
    --host "${GLM_HOST:-0.0.0.0}" --port "${GLM_PORT:-8020}" \
    -c "${GLM_CTX:-200000}" "$@"
