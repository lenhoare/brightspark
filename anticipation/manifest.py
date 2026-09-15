"""Attach prompt identity to trace blocks.

llama.cpp logs a task_id, not a prompt_id -- the server has no idea what our
prompt bank is. The harness writes a manifest recording the order it sent
prompts in; because it sends them strictly sequentially against a server run
with --parallel 1, the k-th distinct task_id to appear in the trace is the k-th
prompt sent. label() enforces that assumption rather than assuming it silently.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Block, Header


def write(path: str | Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def read(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def label(header: Header, blocks: list[Block], manifest: list[dict]) -> list[str]:
    """Annotate blocks in place with prompt_id/category. Returns problems found."""
    problems = []

    if header.n_parallel != 1:
        problems.append(
            f"server ran with --parallel {header.n_parallel}; task ordering is only "
            f"reliable at --parallel 1, labels may be wrong"
        )

    order: list[int] = []
    seen = set()
    for b in blocks:
        if b.task_id not in seen:
            seen.add(b.task_id)
            order.append(b.task_id)

    if len(order) != len(manifest):
        problems.append(
            f"{len(order)} distinct task_ids in trace but {len(manifest)} manifest "
            f"entries -- a request may have failed or produced no draft blocks; "
            f"labels are NOT applied"
        )
        return problems

    by_task = {tid: manifest[i] for i, tid in enumerate(order)}

    for b in blocks:
        entry = by_task[b.task_id]
        b.prompt_id = entry.get("prompt_id")
        b.category  = entry.get("category")
        b.subject   = entry.get("subject")
        b.tags      = entry.get("tags") or []
        b.arm       = entry.get("arm")
        b.pair_id   = entry.get("pair_id")

    return problems
