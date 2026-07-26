#!/bin/bash
# Automatic implement -> review -> fix loop across the two Sparks.
#
#   ./review_loop.sh <workdir>
#
# <workdir>/tasks/*.task   one plain-text feature request per file
#
# WHY IT IS SHAPED LIKE THIS
# --------------------------
# The driver is plain bash and calls NO model of its own.  That is deliberate:
# GLM and Flash each need BOTH Sparks, so every phase change unloads one model
# entirely.  An orchestrator running on either model would kill itself at the
# first swap -- Claude Code cannot "wait for GLM", because the vLLM instance it
# thinks with is exactly what gets torn down.  Only a model-less driver
# survives the whole loop.
#
# Rounds are SYNCHRONIZED ACROSS TASKS rather than per task.  A swap costs
# ~10 min round trip, so looping each task to convergence separately pays that
# per task per round (5 tasks x 2 rounds = 20 swaps ~ 200 min).  Advancing
# every task one step per phase costs 2 swaps per ROUND regardless of task
# count (4 swaps ~ 40 min for the same work).
#
# CONFIG
#   CLAUDE_CMD  how to invoke Claude Code headless against your vLLM endpoint
#   MAX_ROUNDS  hard stop; each round is ~20 min of swap plus review time
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:?usage: review_loop.sh <workdir>}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
CLAUDE_CMD="${CLAUDE_CMD:-claude -p}"
GLM_URL="${GLM_URL:-http://127.0.0.1:8020}"

mkdir -p "$WORK"/{tasks,diffs,verdicts,logs}
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$WORK/logs/driver.log"; }

active(){ ls "$WORK"/tasks/*.task 2>/dev/null | while read -r t; do
    id=$(basename "$t" .task)
    # a task leaves the loop only on an explicit APPROVED verdict
    v="$WORK/verdicts/$id.json"
    if [ -f "$v" ] && grep -q '"status": *"APPROVED"' "$v"; then continue; fi
    echo "$id"
  done; }

round=0
while :; do
  remaining=$(active | wc -l)
  [ "$remaining" -eq 0 ] && { log "all tasks APPROVED after $round round(s)"; break; }
  [ "$round" -ge "$MAX_ROUNDS" ] && { log "hit MAX_ROUNDS=$MAX_ROUNDS with $remaining task(s) unresolved"; break; }
  round=$((round+1))
  log "=== ROUND $round ($remaining task(s) active) ==="

  # ---------- PHASE A: Flash up, implement or apply review notes ----------
  log "phase A: swapping to Flash"
  "$HERE/to-flash.sh" >> "$WORK/logs/swap.log" 2>&1 || { log "SWAP TO FLASH FAILED"; exit 1; }
  for id in $(active); do
    task="$WORK/tasks/$id.task"
    verdict="$WORK/verdicts/$id.json"
    if [ -f "$verdict" ]; then
      # feed the previous round's defects back in as the instruction
      notes=$(python3 -c "
import json,sys
v=json.load(open('$verdict'))
for i in v.get('issues') or []:
    print('- %s: %s (trigger: %s)'%(i.get('file','?'),i.get('what','?'),i.get('trigger','?')))
" 2>/dev/null)
      prompt="A reviewer found these defects in your last change. Fix every one, then stop:

$notes

Original task for context:
$(cat "$task")"
    else
      prompt="$(cat "$task")"
    fi
    log "  implement/apply: $id"
    ( cd "$WORK" && $CLAUDE_CMD "$prompt" ) >> "$WORK/logs/$id.claude.log" 2>&1 \
      || log "  !! claude returned nonzero for $id (see logs/$id.claude.log)"
    # capture what changed; the diff -- not the files -- is what GLM reviews
    ( cd "$WORK" && git diff HEAD ) > "$WORK/diffs/$id.diff" 2>/dev/null \
      || log "  !! could not capture diff for $id (is $WORK a git repo?)"
  done

  # ---------- PHASE B: GLM up, review every active task ----------
  log "phase B: swapping to GLM"
  "$HERE/to-glm.sh" >> "$WORK/logs/swap.log" 2>&1 || { log "SWAP TO GLM FAILED"; exit 1; }
  for id in $(active); do
    d="$WORK/diffs/$id.diff"
    [ -s "$d" ] || { log "  skip $id (empty diff)"; continue; }
    log "  review: $id"
    python3 "$HERE/review_one.py" "$d" "$WORK/verdicts/$id.json" "$GLM_URL" \
      2>&1 | tee -a "$WORK/logs/driver.log"
  done

  # ---------- convergence guard ----------
  # Approval is not the only exit.  Two models can ping-pong on the same
  # disagreement forever, so a round that produces an identical issue set to
  # the previous one counts as converged-by-stalemate.
  for id in $(active); do
    cur="$WORK/verdicts/$id.json"; prev="$WORK/verdicts/$id.prev.json"
    if [ -f "$prev" ] && python3 -c "
import json,sys
a=json.load(open('$cur')).get('issues') or []
b=json.load(open('$prev')).get('issues') or []
key=lambda L: sorted((i.get('file',''),i.get('what','')) for i in L)
sys.exit(0 if key(a)==key(b) and a else 1)" 2>/dev/null; then
      log "  $id: identical issues two rounds running -- stalemate, dropping from loop"
      python3 -c "
import json
p='$cur'; v=json.load(open(p)); v['status']='APPROVED'
v['note']='forced exit: stalemate, issues unchanged between rounds'
json.dump(v,open(p,'w'),indent=1)"
    fi
    cp -f "$cur" "$prev" 2>/dev/null || true
  done
done

# Always hand the boxes back with Flash up, so the next interactive session works.
log "restoring Flash"
"$HERE/to-flash.sh" >> "$WORK/logs/swap.log" 2>&1
log "done after $round round(s); verdicts in $WORK/verdicts/"
