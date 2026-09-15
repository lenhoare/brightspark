#!/usr/bin/env python3
"""Validate a generated prompt bank against prompts/GENERATE.md.

Catches the failure modes that actually happen: markdown fences around the
output, duplicate ids, digits leaking into Set A, unbalanced twist pairs, and
subject counts that drifted. Exits non-zero if anything is wrong, so it can be
handed straight back to the generating model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SUBJECTS = {
    "domestic_life", "dialogue", "software", "medicine", "law",
    "finance", "cooking", "sport", "history", "philosophy",
    "poetry", "travel", "news", "correspondence", "academic",
}

TAGS = {"numerals", "proper_nouns", "structured", "technical_jargon", "dialogue"}

REQUIRED = ["prompt_id", "set", "category", "subject", "register", "tags", "prompt"]

EXPECT = {"A": 8 * len(SUBJECTS), "B": 30, "C": 48}

DIGIT = re.compile(r"\d")
MARKUP = re.compile(r"^\s*(```|#{1,6}\s|[-*]\s|\d+\.\s|\|)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bank")
    ap.add_argument("--strict-digits", action="store_true",
                    help="treat digits in Set A as an error rather than a warning")
    ap.add_argument("--strict-counts", action="store_true",
                    help="require the exact Set A/B/C totals from GENERATE.md")
    args = ap.parse_args()

    path = Path(args.bank)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    raw = path.read_text(encoding="utf-8")
    errs: list[str] = []
    warns: list[str] = []

    if "```" in raw:
        errs.append("file contains markdown fences (```) -- strip them; "
                    "the file must be pure JSONL")

    entries = []
    for n, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"line {n}: not valid JSON ({e.msg})")
            continue
        if not isinstance(d, dict):
            errs.append(f"line {n}: expected an object, got {type(d).__name__}")
            continue
        entries.append((n, d))

    if not entries:
        errs.append("no usable entries found")

    ids = Counter()
    by_set: defaultdict = defaultdict(list)
    subject_counts: Counter = Counter()
    pairs: defaultdict = defaultdict(list)

    for n, d in entries:
        missing = [f for f in REQUIRED if f not in d]
        if missing:
            errs.append(f"line {n}: missing field(s) {', '.join(missing)}")
            continue

        pid = d["prompt_id"]
        ids[pid] += 1

        if not re.fullmatch(r"[a-z0-9_]+", str(pid)):
            errs.append(f"line {n}: prompt_id {pid!r} must be lowercase [a-z0-9_]")

        s = d["set"]
        if s not in ("A", "B", "C"):
            errs.append(f"line {n}: set must be A, B or C (got {s!r})")
            continue
        by_set[s].append((n, d))

        prompt = d.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            errs.append(f"line {n}: empty prompt")
            continue

        words = len(prompt.split())
        if MARKUP.search(prompt):
            errs.append(f"line {n} ({pid}): prompt contains markup/fences")

        if not isinstance(d.get("tags"), list):
            errs.append(f"line {n} ({pid}): tags must be an array")
        else:
            bad = [t for t in d["tags"] if t not in TAGS]
            if bad:
                errs.append(f"line {n} ({pid}): unknown tag(s) {bad}")

        if s == "A":
            subj = d["subject"]
            if subj not in SUBJECTS:
                errs.append(f"line {n} ({pid}): subject {subj!r} not in the list")
            else:
                subject_counts[subj] += 1
            if DIGIT.search(prompt):
                (errs if args.strict_digits else warns).append(
                    f"line {n} ({pid}): Set A prompt contains a digit "
                    f"(tagged and controlled for at analysis time; "
                    f"--strict-digits makes this an error)")
            if d.get("tags"):
                errs.append(f"line {n} ({pid}): Set A prompts must have empty tags")
            if not (15 <= words <= 30):
                warns.append(f"line {n} ({pid}): {words} words, outside 15-30")

        elif s == "B":
            tags = d.get("tags") or []
            if len(tags) != 1:
                errs.append(f"line {n} ({pid}): Set B needs exactly one tag")
            if not (15 <= words <= 30):
                warns.append(f"line {n} ({pid}): {words} words, outside 15-30")

        elif s == "C":
            for f in ("pair_id", "arm"):
                if f not in d:
                    errs.append(f"line {n} ({pid}): Set C requires {f}")
            if d.get("arm") not in ("flat", "turn"):
                errs.append(f"line {n} ({pid}): arm must be 'flat' or 'turn'")
            if "pair_id" in d:
                pairs[d["pair_id"]].append(d)
            if DIGIT.search(prompt):
                warns.append(f"line {n} ({pid}): Set C prompt contains a digit")

    for pid, c in ids.items():
        if c > 1:
            errs.append(f"duplicate prompt_id {pid!r} ({c} times)")

    # twist pairs: two arms, one of each, length-matched
    for pair_id, arms in pairs.items():
        if len(arms) != 2:
            errs.append(f"pair {pair_id}: {len(arms)} arm(s), expected 2")
            continue
        kinds = sorted(a.get("arm") for a in arms)
        if kinds != ["flat", "turn"]:
            errs.append(f"pair {pair_id}: arms are {kinds}, expected flat+turn")
            continue
        a, b = (x["prompt"] for x in sorted(arms, key=lambda x: x["arm"]))
        if abs(len(a.split()) - len(b.split())) > 3:
            errs.append(f"pair {pair_id}: arms differ by "
                        f"{abs(len(a.split()) - len(b.split()))} words (max 3)")
        # the shared setup should genuinely be shared
        pa, pb = a.split(), b.split()
        common = 0
        for wa, wb in zip(pa, pb):
            if wa != wb:
                break
            common += 1
        if common < min(len(pa), len(pb)) * 0.5:
            errs.append(f"pair {pair_id}: arms share only {common} leading words -- "
                        f"the setup must be identical, with only the final cue differing")

    if args.strict_counts:
        for s, want in EXPECT.items():
            got = len(by_set.get(s, []))
            if got != want:
                errs.append(f"set {s}: {got} entries, expected {want}")
        for subj in sorted(SUBJECTS):
            got = subject_counts.get(subj, 0)
            if got != 8:
                errs.append(f"subject {subj}: {got} prompts, expected 8")

    print(f"{path}: {len(entries)} entries  "
          f"(A={len(by_set.get('A', []))} B={len(by_set.get('B', []))} "
          f"C={len(by_set.get('C', []))}, {len(pairs)} pairs)")

    if warns:
        print(f"\n{len(warns)} warning(s):")
        for w in warns[:20]:
            print(f"  ~ {w}")
        if len(warns) > 20:
            print(f"  ... and {len(warns) - 20} more")

    if errs:
        print(f"\n{len(errs)} error(s):")
        for e in errs[:40]:
            print(f"  ! {e}")
        if len(errs) > 40:
            print(f"  ... and {len(errs) - 40} more")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
