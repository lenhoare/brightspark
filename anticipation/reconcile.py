"""Phase 0 gate: does the trace agree with llama.cpp's own acceptance report?

The server prints a per-task line like

  draft acceptance = 0.93103 (   54 accepted /    58 generated), mean len =  7.00

Summing those over a run must reproduce what the trace logged. If it does not,
the instrumentation is missing or double-counting blocks and nothing downstream
can be trusted.
"""

from __future__ import annotations

import re
from pathlib import Path

from .schema import Block

_LINE = re.compile(
    r"task\s+(\d+)\s*\|\s*draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\)"
)


def parse_server_log(path: str | Path) -> dict[int, tuple[int, int]]:
    """task_id -> (accepted, generated) as reported by llama.cpp."""
    out: dict[int, tuple[int, int]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LINE.search(line)
            if m:
                out[int(m.group(1))] = (int(m.group(3)), int(m.group(4)))
    return out


def check(blocks: list[Block], server_log: str | Path) -> dict:
    reported = parse_server_log(server_log)
    if not reported:
        return {"ok": None, "reason": "no acceptance lines found in server log"}

    rep_acc = sum(a for a, _ in reported.values())
    rep_gen = sum(g for _, g in reported.values())

    log_acc = sum(b.accepted_len for b in blocks)
    log_gen = sum(b.n_draft for b in blocks)

    return {
        "ok": rep_acc == log_acc and rep_gen == log_gen,
        "reported_accepted": rep_acc,
        "logged_accepted": log_acc,
        "reported_generated": rep_gen,
        "logged_generated": log_gen,
        "n_tasks": len(reported),
    }
