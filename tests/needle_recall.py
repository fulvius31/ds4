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
import shlex
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


def parse_token_count(text: str) -> int:
    """Parse `ds4 --dump-tokens` output.

    It prints the token-id array as a single line `[id, id, id, ...]` followed by
    one `   id  <text>` line per token. Count from the bracket line; fall back to
    counting the per-token detail lines.
    """
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            return 0 if s == "[]" else s.count(",") + 1
    return sum(1 for l in text.splitlines() if re.match(r"^\s*\d+\s+\S", l))


def run_case(args, prompt_path: Path, ctx: int) -> tuple[str, float]:
    cmd = [
        args.bin, "-m", args.model, f"--{args.backend}",
        "--temp", "0",
        # -c must exceed the REAL prompt token count. Our token estimate drifts
        # ~1% from the real tokenizer at scale, so a fixed +4096 margin gets eaten
        # at large ctx (e.g. a 524288 target tokenized to 528452 > 528432 and was
        # wrongly rejected as "exceeds context size"). Use a proportional margin.
        "-c", str(ctx + max(8192, ctx // 20) + args.gen_tokens),
        "-n", str(args.gen_tokens),
        "--system", "",
        "--think" if args.think else "--nothink",
    ]
    # ds4 rejects `--seed 0` (0 is its parse-failure sentinel); with --temp 0
    # (greedy) the output is deterministic anyway, so only pass a nonzero seed.
    if args.seed:
        cmd += ["--seed", str(args.seed)]
    # Large contexts (>256k) can exceed a single 128GB Spark's unified memory for
    # weights + KV/context buffers; SSD streaming spills to disk to get there.
    if args.ssd_streaming:
        cmd += ["--ssd-streaming"]
        if args.ssd_cache_experts:
            cmd += ["--ssd-streaming-cache-experts", args.ssd_cache_experts]
    if args.extra:
        cmd += shlex.split(args.extra)
    cmd += ["--prompt-file", str(prompt_path)]
    if args.dry_run:
        print("  DRY-RUN:", " ".join(cmd))
        return "", 0.0
    # Prefill is O(context); at ~150 t/s a 1M prompt needs ~2 hrs. Auto-scale the
    # per-cell timeout with ctx (floored by --timeout) so big cells don't get
    # killed mid-prefill. ctx//50 ≈ assumes >=50 tok/s combined, generous.
    eff_timeout = max(args.timeout, ctx // 50)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=eff_timeout)
    except subprocess.TimeoutExpired:
        return "<TIMEOUT>", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return f"<ERROR: {e}>", time.time() - t0
    if proc.returncode != 0:
        errlines = [l for l in proc.stderr.splitlines() if l.strip()]
        msg = errlines[-1] if errlines else f"exit {proc.returncode}"
        return f"<EXIT {proc.returncode}: {msg}>", time.time() - t0
    return proc.stdout + "\n" + proc.stderr, time.time() - t0


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
    ap.add_argument("--chars-per-token", type=float, default=4.5,
                    help="filler sizing heuristic (bytes/token). Default 4.5 is "
                         "calibrated for DeepSeek V4's tokenizer on this prose "
                         "(~4.49 measured); re-check with --calibrate for other corpora")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed; 0 = omit (ds4 rejects --seed 0, and greedy "
                         "temp=0 is deterministic regardless)")
    ap.add_argument("--think", action="store_true", help="allow thinking (default off)")
    ap.add_argument("--ssd-streaming", action="store_true",
                    help="pass --ssd-streaming to ds4 (needed for very large ctx that "
                         "exceeds unified memory on one Spark)")
    ap.add_argument("--ssd-cache-experts", default="",
                    help="value for ds4 --ssd-streaming-cache-experts, e.g. 16GB")
    ap.add_argument("--extra", default="",
                    help="extra args appended verbatim to the ds4 command line "
                         "(shlex-split), e.g. --extra '--prefill-chunk 2048'")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="per-case seconds (floor); auto-scales up with ctx (ctx//50) "
                         "so large contexts aren't killed mid-prefill")
    ap.add_argument("--out-dir", default="needle_runs")
    ap.add_argument("--keep-prompts", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="run `ds4 --dump-tokens` on each prompt to report real token counts")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate prompts + print commands, don't run ds4")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent needles (different code/position) per cell; "
                         "grid then shows a recall RATE k/n instead of PASS/FAIL. "
                         "Use >1 to characterize probabilistic recall at long ctx")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ctx_list = [int(x) for x in args.ctx_list.split(",") if x.strip()]
    depths = [float(x) for x in args.depths.split(",") if x.strip()]

    results = {}   # (ctx, depth) -> {"passes":k, "total":n, "err":bool}
    all_rows = []  # flat per-trial rows for the CSV
    rng = random.Random(args.seed)

    for ctx in ctx_list:
        for depth in depths:
            passes = 0
            err_count = 0
            for trial in range(args.repeats):
                prompt, digit_str, spelled_str = build_haystack(
                    rng, ctx, depth, args.chars_per_token, args.ndigits)
                ppath = out_dir / f"prompt_ctx{ctx}_d{depth:.2f}_t{trial}.txt"
                ppath.write_text(prompt, encoding="utf-8")
                approx_tok = int(len(prompt) / args.chars_per_token)

                # Calibration only tokenizes (`ds4 --dump-tokens`); independent of
                # --dry-run. Same size every trial, so only do it once per cell.
                if args.calibrate and trial == 0:
                    try:
                        dt = subprocess.run(
                            [args.bin, "-m", args.model, f"--{args.backend}",
                             "--dump-tokens", "--prompt-file", str(ppath)],
                            capture_output=True, text=True, timeout=600)
                        ntok = parse_token_count(dt.stdout + dt.stderr)
                        if ntok:
                            cpt = len(prompt) / ntok
                            cal = (f"real_tokens={ntok}  chars/token={cpt:.2f}  "
                                   f"-> set --chars-per-token {cpt:.2f}")
                        else:
                            cal = "(could not parse --dump-tokens output)"
                    except Exception as e:  # noqa: BLE001
                        cal = f"(calibrate failed: {e})"
                    print(f"[calibrate] target_ctx={ctx} d{depth:.2f}: bytes={len(prompt)} "
                          f"approx_tok={approx_tok} :: {cal}")

                tag = f"ctx={ctx} depth={depth:.2f}"
                if args.repeats > 1:
                    tag += f" trial={trial + 1}/{args.repeats}"
                print(f"[run] {tag} (approx {approx_tok} tok) ...", flush=True)
                out, secs = run_case(args, ppath, ctx)
                # Defend against prompt echo: remove the verbatim haystack so the
                # spelled needle in the haystack can't false-pass. (Digit form is
                # already echo-safe -- digits never appear in the haystack.)
                clean = out.replace(prompt, " ") if out and not out.startswith("<") else out
                passed = score(clean, digit_str, spelled_str) if clean and not clean.startswith("<") else False
                tail = clean[-400:].replace("\n", " ⏎ ") if clean else ""
                status = "PASS" if passed else ("ERR" if out.startswith("<") else "FAIL")
                if status == "ERR":
                    print(f"      -> ERR  ({secs:.1f}s)  {tail.strip()[:240]}")
                    err_count += 1
                else:
                    print(f"      -> {status}  ({secs:.1f}s)  expected={digit_str}")
                passes += int(passed)
                all_rows.append(dict(
                    ctx=ctx, depth=depth, trial=trial, approx_tok=approx_tok,
                    expected=digit_str, passed=passed, status=status,
                    latency_s=round(secs, 1), got_tail=tail))
                if not args.keep_prompts and not args.dry_run:
                    ppath.unlink(missing_ok=True)
            results[(ctx, depth)] = dict(passes=passes, total=args.repeats,
                                         err=(err_count == args.repeats))

    # ---- grid (cells show recall rate k/n, or ERR if every trial errored) ----
    print("\n=== Needle recall grid (rows=depth, cols=ctx; cell = passes/trials) ===")
    header = "depth\\ctx |" + "".join(f"{c:>10}" for c in ctx_list)
    print(header)
    print("-" * len(header))
    for d in depths:
        row = f"{d:>9.2f} |"
        for c in ctx_list:
            r = results[(c, d)]
            cell = "ERR" if r["err"] else f"{r['passes']}/{r['total']}"
            row += f"{cell:>10}"
        print(row)

    # ---- csv (one row per trial) ----
    csv_path = out_dir / "needle_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ctx", "depth", "trial", "approx_tok",
                                          "expected", "passed", "status",
                                          "latency_s", "got_tail"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
