#!/bin/bash
# Swap the pair from vLLM/Flash to GLM 5.2 on ds4, and bring up the
# OpenAI-compatible API on :8020.  ~4-5 min end to end.
#
# The two models cannot be co-resident (each plans ~100 GiB across both
# Sparks), so this is a hard swap.  Budget ~10 min for a full round trip and
# batch your reviews accordingly -- below ~4 reviews the swap costs more than
# it saves.
set -u
R=10.0.0.2
PORT="${GLM_PORT:-8020}"

echo "== 1/5 pause watchdog (LOCAL box only) =="
# MUST happen before the containers go away: the watchdog treats a missing
# port 8888 as a crash and full-restarts the ~100 GiB stack, which has frozen
# and rebooted this box before.
touch /tmp/dgx-watchdog.pause

echo "== 2/5 remove vLLM containers on BOTH nodes =="
# `docker stop` alone leaves the container in Exited, which the watchdog also
# reads as a crash.  It is only inert when the container is ABSENT -- always rm.
docker stop dsparkablit-vllm-dspark-1 >/dev/null 2>&1
docker rm   dsparkablit-vllm-dspark-1 >/dev/null 2>&1
timeout 60 ssh $R 'docker stop dsparkablit-vllm-dspark-1 >/dev/null 2>&1; docker rm dsparkablit-vllm-dspark-1 >/dev/null 2>&1'
echo "   local:  $(docker ps -a --format '{{.Names}}' | grep -c dsparkablit) container(s) left"
echo "   remote: $(timeout 20 ssh $R 'docker ps -a --format "{{.Names}}" | grep -c dsparkablit')"

echo "== 3/5 check memguard on both boxes =="
for host in local $R; do
  if [ "$host" = local ]; then n=$(pgrep -fc "[m]emguard\.sh" || true)
  else n=$(timeout 20 ssh $R 'pgrep -fc "[m]emguard\.sh" || true'); fi
  [ "${n:-0}" -ge 1 ] && echo "   $host: memguard OK" \
                      || echo "   $host: !! NO MEMGUARD -- start it: setsid bash ~/memguard.sh 400000 &"
done

echo "== 4/5 start TP LEADER + API server (10.0.0.1) =="
# TP order is the reverse of the pipeline config: the leader listens, the
# worker joins. Starting the worker first just burns retries.
cd "$(dirname "$0")"
GLM_CTX="${GLM_CTX:-400000}" POOL="${POOL:-5000}" GLM_PORT="$PORT" \
  setsid nohup ./run_glm_server.sh \
  > ~/logs_ds4_tests/glm_server.log 2>&1 < /dev/null &
disown
sleep 5

echo "== 5/5 start TP worker (10.0.0.2) =="
# ctx, pool and the fp8 flag MUST match the leader or the identity handshake
# rejects the pairing.
timeout 30 ssh $R "cd ds4 && nohup bash -c 'DS4_GLM_FP8_KV_STORE=1 GLM_CTX=${GLM_CTX:-400000} POOL=${POOL:-5000} ./run_glm_tp_worker.sh > ~/logs_ds4_tests/glm_worker.log 2>&1' < /dev/null > /dev/null 2>&1 &"

for i in $(seq 1 90); do
  curl -fsS --max-time 4 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
    echo "   GLM API UP after ~$((i*5))s on http://127.0.0.1:$PORT"; exit 0; }
  pgrep -xc ds4-server >/dev/null || { echo "   SERVER DIED:"; tail -5 ~/logs_ds4_tests/glm_server.log; exit 1; }
  sleep 5
done
echo "   TIMEOUT waiting for API"; tail -5 ~/logs_ds4_tests/glm_server.log; exit 1
