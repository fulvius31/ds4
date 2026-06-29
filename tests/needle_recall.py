#!/usr/bin/env python3
"""1M needle-in-a-haystack / fact-recall harness for DeepSeek V4 Flash (ds4).

Builds long prose haystacks with a planted "needle" fact at controlled depths,
runs ./ds4 across a sweep of context lengths, and scores whether the model
recalls the needle. Emits a PASS/FAIL grid + CSV so you can see *where*
long-context recall starts to degrade for a given GGUF -- e.g. the q2 imatrix
quant, whose quality is only validated at <=32k (see CUDA discussion / the
imatrix calibration ctx). This is the empirical answer to "does 1M actually
work?" on your hardware.

stdlib only. Run from the ds4 repo root *after building*:

    make cuda-spark
    ./download_model.sh q2-imatrix          # creates ./ds4flash.gguf
    python3 tests/needle_recall.py -m ds4flash.gguf

Start small, then push the ceiling:

    # quick sanity (seconds-to-minutes):
    python3 tests/needle_recall.py -m ds4flash.gguf --ctx-list 4096,16384,65536

    # the real long-context sweep (SLOW -- 1M prefill is ~tens of minutes):
    python3 tests/needle_recall.py -m ds4flash.gguf \
        --ctx-list 131072,262144,524288,1048576 --depths 0.1,0.5,0.9

WARNING ON TIME: decode is cheap here (we only generate a few dozen tokens), but
*prefill* is O(context). At ~400 tok/s prefill, a 1M-token haystack takes on the
order of 40+ minutes for that single cell. Size your sweep accordingly.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
import sys
import time
from pathlib import Path

DIGIT_WORDS = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]

# Compact harbor-town prose templates, mirroring the style of the existing
# tests/generate_long_context_story_prompt.py fixture so the needle blends into
# realistic text rather than standing out as an obvious anomaly. {lead}/{friend}
# are filled with rotating names; {n} keeps every paragraph unique so the prefix
# cache / dedup cannot trivially collapse the haystack.
SCENE_TEMPLATES = [
    """At first light (entry {n}) the harbor smelled of rope, rain, and cedar smoke.
{lead} crossed the quay with a folded map under one arm while {friend} argued with a
gull over a dropped crust. The bakery cooled its loaves beneath linen and nobody was
in a hurry, because Bellwether moved by tide and habit, not by the council bells.""",
    """By noon (entry {n}) the market filled with pears, lamp oil, brass hooks, and paper
flowers. {lead} bargained for twine while {friend} listened to a sailor describe a storm
that grew taller with every retelling. The town clock had stopped again and every
shopkeeper defended a different hour with total confidence.""",
    """In the afternoon (entry {n}) a rehearsal for the midsummer play blocked the west
road. {lead} carried a crate of lantern glass through the crowd while {friend} read lines
from a damp script. Someone had painted the moon too blue, and three people argued about
whether a theatrical moon was allowed to be wrong.""",
    """Toward evening (entry {n}) the archivist Mara wrote notes in brown ink, never black,
because black ink made old ledgers look like court summonses. She watched {lead} and
{friend} pass the fountain and added a calm line about the fog, the chipped rim of a cup,
and a door that kept opening after it had been firmly shut.""",
    """At dusk (entry {n}) the lamplighter climbed his ladder while {lead} counted herring
crates and {friend} mended a net by feel. The tide came in slow and brown. A child traded
three shells for a button, and the button was declared, by unanimous and unofficial vote,
to be lucky.""",
]

NAMES = ["Bob", "Alice", "Clara", "Diego", "Elena", "Felix", "Greta", "Hugo",
         "Iris", "Jonas", "Kira", "Leo", "Marta", "Nadia", "Owen", "Priya",
         "Rosa", "Sven", "Tomas", "Ulla", "Viktor", "Wren", "Xenia", "Yara"]

OPENING = (
    "You are reading a long, ordinary chronicle of the harbor town of Bellwether. "
    "People walk, trade, argue about weather, and rehearse a play. Somewhere in the "
    "chronicle, the archivist Mara records exactly one private seasonal code. At the "
    "end you will be asked to recall that code. Ignore all other numbers (ages, "
    "prices, room numbers, dates) -- only Mara's recorded seasonal code counts.\n\n"
)


def spelled(digits: list[int]) -> str:
    return "-".join(DIGIT_WORDS[d] for d in digits)


def make_needle(rng: random.Random, ndigits: int) -> tuple[str, str, str]:
    """Return (needle_paragraph, digit_string, spelled_string)."""
    digits = [rng.randint(0, 9) for _ in range(ndigits)]
    digit_str = "".join(str(d) for d in digits)
    spelled_str = spelled(digits)
    para = (
        f"That night, alone with her ledger, Mara wrote a single private line in brown "
        f"ink: the Bellwether lighthouse vault code for this season was {spelled_str}. "
        f"She underlined it twice, blotted the page, and told no one.")
    return para, digit_str, spelled_str


def scene(rng: random.Random, n: int) -> str:
    tmpl = SCENE_TEMPLATES[n % len(SCENE_TEMPLATES)]
    lead, friend = rng.sample(NAMES, 2)
    return tmpl.format(n=n, lead=lead, friend=friend).replace("\n", " ").strip()


def build_haystack(rng: random.Random, target_tokens: int, depth: float,
                   chars_per_token: float, ndigits: int) -> tuple[str, str, str]:
    """Build the user prompt. Needle is placed at `depth` (0..1) of the body."""
    needle, digit_str, spelled_str = make_needle(rng, ndigits)
    question = (
        "\n\nEnd of chronicle.\n\n"
        "Question: Earlier, the archivist Mara recorded exactly one private seasonal "
        "code -- the Bellwether lighthouse vault code for this season. What was that "
        "code? Reply with ONLY the digits, e.g. CODE: 012345")

    # Reserve room for opening + needle + question so the total lands near target.
    reserve_chars = len(OPENING) + len(needle) + len(question) + 256
    body_target_chars = max(0, int(target_tokens * chars_per_token) - reserve_chars)

    scenes: list[str] = []
    total = 0
    n = 0
    while total < body_target_chars:
        s = scene(rng, n)
        scenes.append(s)
        total += len(s) + 2
        n += 1

    insert_at = int(len(scenes) * max(0.0, min(1.0, depth)))
    body_before = "\n\n".join(scenes[:insert_at])
    body_after = "\n\n".join(scenes[insert_at:])
    prompt = OPENING + body_before + "\n\n" + needle + "\n\n" + body_after + question
    return prompt, digit_str, spelled_str


def score(output: str, digit_str: str, spelled_str: str) -> bool:
    low = output.lower()
    digits_only = re.sub(r"[^0-9]", "", output)
    if digit_str in digits_only:
        return True
    alpha_only = re.sub(r"[^a-z]", "", low)
    if "".join(spelled_str.split("-")) in alpha_only:
        return True
    return False


def run_case(args, prompt_path: Path, ctx: int) -> tuple[str, float]:
    cmd = [
        args.bin, "-m", args.model, f"--{args.backend}",
        "--temp", "0", "--seed", str(args.seed),
        "-c", str(ctx + args.gen_tokens + 4096),
        "-n", str(args.gen_tokens),
        "--system", "",
        "--think" if args.think else "--nothink",
        "--prompt-file", str(prompt_path),
    ]
    if args.dry_run:
        print("  DRY-RUN:", " ".join(cmd))
        return "", 0.0
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return "<TIMEOUT>", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return f"<ERROR: {e}>", time.time() - t0
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", default="ds4flash.gguf", help="GGUF path")
    ap.add_argument("--bin", default="./ds4", help="ds4 binary path")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "metal", "cpu"])
    ap.add_argument("--ctx-list", default="4096,16384,65536,131072,262144,524288,1048576",
                    help="comma-separated target context lengths (tokens)")
    ap.add_argument("--depths", default="0.1,0.5,0.9",
                    help="comma-separated needle depths (0..1)")
    ap.add_argument("-n", "--gen-tokens", type=int, default=48)
    ap.add_argument("--ndigits", type=int, default=6, help="needle code length")
    ap.add_argument("--chars-per-token", type=float, default=3.8,
                    help="filler sizing heuristic (bytes/token); tune with --calibrate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think", action="store_true", help="allow thinking (default off)")
    ap.add_argument("--timeout", type=int, default=5400, help="per-case seconds")
    ap.add_argument("--out-dir", default="needle_runs")
    ap.add_argument("--keep-prompts", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="run `ds4 --dump-tokens` on each prompt to report real token counts")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate prompts + print commands, don't run ds4")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ctx_list = [int(x) for x in args.ctx_list.split(",") if x.strip()]
    depths = [float(x) for x in args.depths.split(",") if x.strip()]

    results = {}  # (ctx, depth) -> dict
    rng = random.Random(args.seed)

    for ctx in ctx_list:
        for depth in depths:
            prompt, digit_str, spelled_str = build_haystack(
                rng, ctx, depth, args.chars_per_token, args.ndigits)
            ppath = out_dir / f"prompt_ctx{ctx}_d{depth:.2f}.txt"
            ppath.write_text(prompt, encoding="utf-8")
            approx_tok = int(len(prompt) / args.chars_per_token)

            if args.calibrate and not args.dry_run:
                try:
                    dt = subprocess.run(
                        [args.bin, "-m", args.model, f"--{args.backend}",
                         "--dump-tokens", "--prompt-file", str(ppath)],
                        capture_output=True, text=True, timeout=600)
                    tokln = [l for l in (dt.stdout + dt.stderr).splitlines()
                             if "token" in l.lower()]
                    cal = tokln[-1] if tokln else "(no token line)"
                except Exception as e:  # noqa: BLE001
                    cal = f"(calibrate failed: {e})"
                print(f"[calibrate] ctx~{ctx} d{depth:.2f}: bytes={len(prompt)} "
                      f"approx_tok={approx_tok} :: {cal}")

            print(f"[run] ctx={ctx} depth={depth:.2f} (approx {approx_tok} tok) ...",
                  flush=True)
            out, secs = run_case(args, ppath, ctx)
            # Defend against prompt echo: remove the verbatim haystack from the
            # output so the spelled needle in the haystack can't false-pass.
            # (The digit form is already echo-safe -- digits never appear in the
            # haystack, only the spelled code does.)
            clean = out.replace(prompt, " ") if out and not out.startswith("<") else out
            passed = score(clean, digit_str, spelled_str) if clean and not clean.startswith("<") else False
            tail = clean[-400:].replace("\n", " ⏎ ") if clean else ""
            results[(ctx, depth)] = dict(
                ctx=ctx, depth=depth, expected=digit_str, passed=passed,
                latency_s=round(secs, 1), approx_tok=approx_tok, got_tail=tail)
            status = "PASS" if passed else ("ERR" if out.startswith("<") else "FAIL")
            print(f"      -> {status}  ({secs:.1f}s)  expected={digit_str}")
            if not args.keep_prompts and not args.dry_run:
                ppath.unlink(missing_ok=True)

    # ---- grid ----
    print("\n=== Needle recall grid (rows=depth, cols=ctx) ===")
    header = "depth\\ctx |" + "".join(f"{c:>10}" for c in ctx_list)
    print(header)
    print("-" * len(header))
    for d in depths:
        row = f"{d:>9.2f} |"
        for c in ctx_list:
            r = results[(c, d)]
            cell = "PASS" if r["passed"] else ("ERR" if r["got_tail"].startswith("<") else "FAIL")
            row += f"{cell:>10}"
        print(row)

    # ---- csv ----
    csv_path = out_dir / "needle_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ctx", "depth", "approx_tok", "expected",
                                          "passed", "latency_s", "got_tail"])
        w.writeheader()
        for r in results.values():
            w.writerow(r)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
