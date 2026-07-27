#!/bin/bash
# One-command bring-up of the WIDEST window: pipeline + SSD streaming + fp8 KV
# at the model's full 1,048,576-token context. Starts both ranks and health-gates.
#
#   ./run_glm_1M.sh            # 1M context (default)
#   GLM_CTX=900000 ./run_glm_1M.sh
#
# WHY PIPELINE AND NOT TP FOR THIS
# --------------------------------
# Measured 2026-07-26 (identical 11,082-token prompt on both):
#   TP + streaming + fp8   max ctx   500,000   prefill 50.4 t/s  decode 1.86 t/s  floor  1.9 GiB
#   pipeline + streaming   max ctx 1,048,576   prefill 49.4 t/s  decode 1.13 t/s  floor 25.3 GiB
# TP REPLICATES the KV plane (kv_layers=78 on BOTH ranks) because GLM's DSA/MLA
# attention compresses KV into a per-token latent with no head dimension to
# shard; pipeline SPLITS it by layer (40 leader / 38 worker). Same ~660 bytes
# per layer per token either way -- TP just stores the whole model's KV twice.
# TP also replicates the non-routed weights (22.55 GiB/box vs 10.14/12.41).
# At max context TP's prefill edge is gone (50.4 vs 49.4); it only keeps decode.
#
# WHY RESERVE 8 AND NOT 12
# ------------------------
# The memory guard gates on a PLANNED KV figure that assumes full layers, so it
# over-reports pipeline by ~2x (43.2 GiB claimed at 900k where the graph really
# allocates 22.4). At 1M the planned figure lands 0.6 GiB over the reserve-12
# budget and the worker is refused, despite ~25 GiB actually being free at load.
# Reserve 8 clears it; the measured floors justify it.
set -u
R=10.0.0.2
CTX="${GLM_CTX:-1048576}"
RESERVE="${DS4_GLM_MEMORY_GUARD_RESERVE_GB:-8}"
PORT="${GLM_PORT:-8020}"
MODEL=gguf/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
cd "$(dirname "$0")"

echo "== stopping any running ranks =="
for i in 1 2 3 4; do pkill -9 -x ds4-server 2>/dev/null; sleep 1; [ "$(pgrep -xc ds4-server)" = "0" ] && break; done
timeout 25 ssh $R 'for i in 1 2 3; do pkill -9 -x ds4 2>/dev/null; sleep 1; [ "$(pgrep -xc ds4)" = "0" ] && break; done' >/dev/null 2>&1

echo "== worker first (10.0.0.2, layers 40:output, ctx $CTX) =="
# Pipeline order: the worker dials the coordinator and retries, so it may start
# before the leader is listening. (TP is the opposite -- leader first.)
timeout 30 ssh $R "cd ds4 && nohup bash -c 'DS4_GLM_MEMORY_GUARD_RESERVE_GB=$RESERVE DS4_GLM_FP8_KV_STORE=1 DS4_CUDA_WEIGHT_CACHE=1 DS4_GLM_CUDA_STREAMING=1 ./ds4 -m $MODEL --cuda --ssd-streaming --ssd-streaming-full-layers 0 --ssd-streaming-cache-experts ${POOL:-5000} --role worker --layers 40:output -c $CTX --coordinator 10.0.0.1 9911 > ~/logs_ds4_tests/glm_1M_worker.log 2>&1' < /dev/null > /dev/null 2>&1 &" >/dev/null 2>&1
sleep 4

echo "== leader + API (:$PORT) =="
DS4_GLM_MEMORY_GUARD_RESERVE_GB=$RESERVE GLM_CTX=$CTX GLM_PORT=$PORT \
  setsid nohup ./run_glm_server.sh > ~/logs_ds4_tests/glm_1M_server.log 2>&1 < /dev/null &
disown

for i in $(seq 1 90); do
  if curl -fsS --max-time 4 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "   API up after ~$((i*5))s on http://0.0.0.0:$PORT"
    grep -aE "allocating compact DSA" ~/logs_ds4_tests/glm_1M_server.log | tail -1
    # A models listing is not health: confirm the worker actually joined.
    timeout 15 ssh $R 'grep -qa "connected to coordinator" ~/logs_ds4_tests/glm_1M_worker.log' \
      && echo "   worker joined" || echo "   !! worker has NOT joined yet -- check glm_1M_worker.log"
    exit 0
  fi
  pgrep -xc ds4-server >/dev/null || { echo "   LEADER DIED"; grep -aE "refused|required model|error" ~/logs_ds4_tests/glm_1M_server.log | tail -3; exit 1; }
  sleep 5
done
echo "   TIMEOUT"; tail -4 ~/logs_ds4_tests/glm_1M_server.log; exit 1
