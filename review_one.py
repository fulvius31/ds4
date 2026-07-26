#!/usr/bin/env python3
"""Send ONE artifact to the GLM reviewer on ds4-server and save a structured verdict.

  review_one.py <diff_file> <out_json> [base_url] [model]

Prompt layout is deliberate.  ds4-server keeps a byte-prefix KV cache, so the
INVARIANT preamble goes first and the variable diff goes last: the preamble is
prefilled once for the whole batch and reused for every later review, instead
of being re-prefilled at ~22 t/s each time.  Never interpolate anything
per-task above the DIFF marker or the cache misses on every request.

GLM emits a long thinking trace before its answer, so the verdict is recovered
by scanning for the last JSON object in the reply rather than assuming the
whole response is JSON.
"""
import json, os, re, sys, time, urllib.request

# --- INVARIANT PREAMBLE: keep byte-identical across every review in a batch ---
PREAMBLE = """You are a strict code reviewer. You are reviewing a unified diff.

Report only DEFECTS: correctness bugs, unhandled edge cases, security issues,
resource leaks, and broken invariants. Do not report style, formatting, naming,
or preferences. If the diff is correct, approve it.

For every defect give: the file, what is wrong, and a concrete input or sequence
that exposes it. Be specific; a defect you cannot demonstrate is not a defect.

Finish your reply with a single JSON object on its own, in this exact shape:

{"status": "APPROVED" or "CHANGES_REQUESTED",
 "issues": [{"file": "...", "what": "...", "trigger": "..."}]}

Use APPROVED with an empty issues list if you find no defect.

DIFF:
"""


def extract_verdict(text):
    """Recover the last balanced JSON object containing a 'status' key."""
    best = None
    for m in re.finditer(r"\{", text):
        depth, i = 0, m.start()
        for j in range(m.start(), len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[i:j + 1]
                    try:
                        v = json.loads(blob)
                        if isinstance(v, dict) and "status" in v:
                            best = v
                    except Exception:
                        pass
                    break
    return best


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    diff_path, out_path = sys.argv[1], sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8020"
    model = sys.argv[4] if len(sys.argv) > 4 else "glm-5.2"
    max_tokens = int(os.environ.get("REVIEW_MAX_TOKENS", "4000"))

    diff = open(diff_path, encoding="utf-8", errors="replace").read()
    # Keep the request under the ~5k-token memory ceiling measured on the
    # pipeline-resident worker; past that the prefill spike trips the memguard.
    limit = int(os.environ.get("REVIEW_DIFF_CHAR_LIMIT", "12000"))
    truncated = len(diff) > limit
    if truncated:
        diff = diff[:limit] + "\n[... diff truncated for context budget ...]\n"

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PREAMBLE + diff}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=7200))
    except Exception as e:
        json.dump({"status": "ERROR", "error": repr(e), "issues": []},
                  open(out_path, "w"), indent=1)
        print(f"REVIEW FAILED {os.path.basename(diff_path)}: {e!r}")
        return 1

    ch = r["choices"][0]
    msg = ch.get("message", {}) or {}
    text = msg.get("content") or ""
    verdict = extract_verdict(text)
    if verdict is None:
        # No parseable verdict: usually the thinking trace consumed max_tokens.
        # Fail loud rather than silently "approving" unreviewed code.
        verdict = {"status": "NO_VERDICT", "issues": [],
                   "note": f"no JSON verdict recovered (finish={ch.get('finish_reason')}); "
                           f"raise REVIEW_MAX_TOKENS"}
    verdict["_sec"] = round(time.time() - t0, 1)
    verdict["_finish"] = ch.get("finish_reason")
    verdict["_usage"] = r.get("usage")
    verdict["_diff_truncated"] = truncated
    verdict["_raw"] = text[-4000:]
    json.dump(verdict, open(out_path, "w"), indent=1)
    n = len(verdict.get("issues") or [])
    print(f"{verdict['status']:18s} {os.path.basename(diff_path):28s} "
          f"{verdict['_sec']:7.1f}s issues={n} finish={verdict['_finish']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
