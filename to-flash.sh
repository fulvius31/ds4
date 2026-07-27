#!/bin/bash
# Swap the pair back from GLM/ds4 to the abliterated vLLM Flash stack on :8888,
# then re-arm the watchdog.  ~6 min end to end (the health gate dominates).
set -u
R=10.0.0.2

echo "== 1/4 stop ds4 on both boxes =="
# Exact-name kills only.  A -f pattern matches this script's own command line
# (and the ssh wrapper carrying it), which has self-killed runs before.
pkill -x ds4-server 2>/dev/null
for i in 1 2 3; do pkill -9 -x ds4 2>/dev/null; sleep 1; [ "$(pgrep -xc ds4)" = "0" ] && break; done
timeout 40 ssh $R 'for i in 1 2 3; do pkill -9 -x ds4 2>/dev/null; sleep 1; [ "$(pgrep -xc ds4)" = "0" ] && break; done'
echo "   local ds4=$(pgrep -xc ds4) ds4-server=$(pgrep -xc ds4-server) remote ds4=$(timeout 20 ssh $R 'pgrep -xc ds4')"

echo "== 2/4 start vLLM worker (10.0.0.2) FIRST =="
timeout 180 ssh $R 'cd ~/anemll-test && docker compose -p dsparkablit --env-file worker.env.abliterated -f docker-compose.abliterated.yml up -d' 2>&1 | tail -2

echo "== 3/4 start vLLM head (local) =="
cd ~/anemll-test && timeout 180 docker compose -p dsparkablit --env-file head.env.abliterated -f docker-compose.abliterated.yml up -d 2>&1 | tail -2

echo "== 4/4 health gate (models endpoint, then a REAL generation probe) =="
up=0
for i in $(seq 1 90); do
  curl -fsS --max-time 4 http://127.0.0.1:8888/v1/models >/dev/null 2>&1 && { up=1; echo "   models endpoint up after ~$((i*10))s"; break; }
  sleep 10
done
[ "$up" = 1 ] || { echo "   TIMEOUT -- leaving watchdog PAUSED so it cannot thrash the stack"; exit 1; }

# A models listing alone is not health: vLLM answers it before it can generate.
probe=$(curl -fsS --max-time 60 http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-dspark","messages":[{"role":"user","content":"Reply with exactly: READY"}],"max_tokens":200,"temperature":0}' \
  2>/dev/null | python3 -c 'import sys,json; print((json.load(sys.stdin)["choices"][0]["message"].get("content") or "").strip())' 2>/dev/null)
if [ "$probe" = "READY" ]; then
  # A one-token probe proves the engine answers, NOT that its JIT caches cover
  # real traffic.  Triton/TileLang compile lazily per shape-key; under two-node
  # TP a cold compile mid-step leaves the peer rank spinning in NCCL, and a
  # first-contact concurrent long-context burst can cascade past the 300s
  # sample_tokens budget and kill the engine (docs/jit-cache-incident.md in
  # personal-dgx-spark-solution).  Warm before re-arming the watchdog, so the
  # watchdog never inherits an unwarmed stack.
  WARM=~/personal-dgx-spark-solution/scripts/warmup.sh
  if [ -x "$WARM" ]; then
    "$WARM" || echo "   WARNING: warmup reported a problem -- see output above"
  else
    echo "   WARNING: $WARM not found; stack is UNWARMED and JIT-vulnerable"
  fi
  rm -f /tmp/dgx-watchdog.pause
  echo "   generation probe OK -- watchdog RE-ARMED; Flash live on :8888"
else
  echo "   generation probe FAILED (got: '$probe') -- watchdog left PAUSED on purpose"
  exit 1
fi
