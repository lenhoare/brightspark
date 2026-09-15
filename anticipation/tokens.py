"""Classify tokens by what kind of thing they are.

Numerals, punctuation and special tokens dominate raw mismatch counts and say
little about subject matter. Rather than forbidding them when the prompt bank
is written -- which costs the generating model most of its effort -- they are
tagged here so the blind-spot map can be read by kind, and numeral density can
be regressed out of the subject comparison.
"""

from __future__ import annotations

import re

NUMERIC = re.compile(r"\d")
WORDY = re.compile(r"[A-Za-z]")
SPECIAL = re.compile(r"^<\|.*\|>$|^</?[a-z_]+>$|^\[[A-Z_]+\]$")

KINDS = ("numeric", "special", "punct", "whitespace", "capitalised", "word")


def classify(text: str) -> str:
    """One of KINDS. `text` is the decoded token, leading space included.

    "capitalised" is a rough proper-noun proxy: a capitalised word token with a
    leading space, so it is not at the start of a line. It will also catch
    sentence-initial words after some punctuation -- treat it as a hint, not a
    part-of-speech tag.
    """
    if not text:
        return "whitespace"

    if SPECIAL.match(text.strip()):
        return "special"

    if NUMERIC.search(text):
        return "numeric"

    if not text.strip():
        return "whitespace"

    if not WORDY.search(text):
        return "punct"

    body = text.lstrip()
    if text[:1] == " " and body[:1].isupper():
        return "capitalised"

    return "word"


def density(text: str) -> float:
    """Fraction of characters in a span that are digits."""
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def has_digits(text: str) -> bool:
    return bool(NUMERIC.search(text or ""))
