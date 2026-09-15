"""Reader and typed view over a --spec-trace JSONL log.

The log has one header record followed by one record per verified draft block.
Array fields on the draft side are indexed by draft position; array fields on
the verify side cover only the positions the target actually evaluated, so they
are shorter whenever the draft was rejected early. That shortfall is the
censoring, not missing data -- do not pad it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Header:
    spec_type: str = ""
    model_tgt: str = ""
    model_dft: str = ""
    n_max: int = 0
    n_min: int = 0
    p_min: float = 0.0
    n_parallel: int = 1
    n_ctx: int = 0
    trace_n_ctx: int = 0
    build_info: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class Block:
    block_idx: int
    task_id: int
    seq_id: int
    pos_first: int
    n_max: int
    n_draft: int
    n_draft_gen: int
    accepted_len: int
    n_verified: int
    mismatch_pos: int
    is_replay: bool
    draft_stop: str

    draft_tokens: list[int]
    draft_p: list[float]
    draft_entropy: list[float]
    draft_conf: list[float | None]
    draft_n_cand: list[int]

    target_token: list[int]
    target_top1_p: list[float]
    target_entropy: list[float]
    target_p_of_draft: list[float | None]
    draft_rank_in_target: list[int]

    context_tail: str

    # filled by harness.manifest, not by llama.cpp
    prompt_id: str | None = None
    category: str | None = None
    subject: str | None = None
    tags: list[str] = field(default_factory=list)
    arm: str | None = None
    pair_id: str | None = None

    @property
    def diverged(self) -> bool:
        return self.mismatch_pos >= 0

    @property
    def mean_target_entropy(self) -> float:
        """Mean over verified positions only -- see the censoring note above."""
        if not self.target_entropy:
            return float("nan")
        return sum(self.target_entropy) / len(self.target_entropy)

    def mismatch_pair(self) -> tuple[int, int] | None:
        """(draft token, target token) at the break point, or None on full accept."""
        if not self.diverged:
            return None
        i = self.mismatch_pos
        if i >= len(self.draft_tokens) or i >= len(self.target_token):
            return None
        return (self.draft_tokens[i], self.target_token[i])


def _block_from_json(d: dict) -> Block:
    return Block(
        block_idx=d["block_idx"],
        task_id=d["task_id"],
        seq_id=d["seq_id"],
        pos_first=d["pos_first"],
        n_max=d["n_max"],
        n_draft=d["n_draft"],
        n_draft_gen=d.get("n_draft_gen", d["n_draft"]),
        accepted_len=d["accepted_len"],
        n_verified=d["n_verified"],
        mismatch_pos=d["mismatch_pos"],
        is_replay=d.get("is_replay", False),
        draft_stop=d.get("draft_stop", ""),
        draft_tokens=d.get("draft_tokens", []),
        draft_p=d.get("draft_p", []),
        draft_entropy=d.get("draft_entropy", []),
        draft_conf=d.get("draft_conf", []),
        draft_n_cand=d.get("draft_n_cand", []),
        target_token=d.get("target_token", []),
        target_top1_p=d.get("target_top1_p", []),
        target_entropy=d.get("target_entropy", []),
        target_p_of_draft=d.get("target_p_of_draft", []),
        draft_rank_in_target=d.get("draft_rank_in_target", []),
        context_tail=d.get("context_tail", ""),
    )


def load(path: str | Path, drop_replays: bool = True) -> tuple[Header, list[Block]]:
    """Read a trace file. Replayed blocks are duplicates of an earlier block and
    are dropped by default; pass drop_replays=False to inspect them."""
    header = Header()
    blocks: list[Block] = []

    # a token can be a partial multi-byte character, so context_tail may begin
    # mid-sequence and carry invalid UTF-8; substitute rather than refuse the file
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                # a run killed mid-write leaves a torn final line; everything
                # before it is still usable
                raise ValueError(f"{path}:{lineno}: malformed JSON ({e})") from e

            kind = d.get("record")
            if kind == "header":
                header = Header(
                    spec_type=d.get("spec_type", ""),
                    model_tgt=d.get("model_tgt", ""),
                    model_dft=d.get("model_dft", ""),
                    n_max=d.get("n_max", 0),
                    n_min=d.get("n_min", 0),
                    p_min=d.get("p_min", 0.0),
                    n_parallel=d.get("n_parallel", 1),
                    n_ctx=d.get("n_ctx", 0),
                    trace_n_ctx=d.get("trace_n_ctx", 0),
                    build_info=d.get("build_info", ""),
                    raw=d,
                )
            elif kind == "block":
                b = _block_from_json(d)
                if drop_replays and b.is_replay:
                    continue
                blocks.append(b)

    return header, blocks


def check(header: Header, blocks: list[Block]) -> list[str]:
    """Internal-consistency checks. Returns a list of problems; empty is good."""
    problems = []

    if not blocks:
        return ["trace contains no block records"]

    for b in blocks:
        if len(b.target_token) != b.n_verified:
            problems.append(
                f"block {b.block_idx}: n_verified={b.n_verified} but "
                f"{len(b.target_token)} target entries"
            )
        if b.draft_p and len(b.draft_p) != b.n_draft:
            problems.append(
                f"block {b.block_idx}: n_draft={b.n_draft} but "
                f"{len(b.draft_p)} draft entries"
            )
        if b.n_draft_gen and b.n_draft_gen < b.n_draft:
            problems.append(
                f"block {b.block_idx}: n_draft_gen={b.n_draft_gen} < n_draft={b.n_draft}"
            )
        if b.accepted_len > b.n_draft:
            problems.append(
                f"block {b.block_idx}: accepted_len={b.accepted_len} > n_draft={b.n_draft}"
            )
        if b.mismatch_pos >= 0 and b.mismatch_pos != b.accepted_len:
            problems.append(
                f"block {b.block_idx}: mismatch_pos={b.mismatch_pos} != "
                f"accepted_len={b.accepted_len}"
            )

    if header.p_min > 0:
        n_pmin = sum(1 for b in blocks if b.draft_stop == "p_min")
        if n_pmin:
            problems.append(
                f"p_min={header.p_min} truncated {n_pmin}/{len(blocks)} drafts early; "
                f"divergence is confounded with the draft's own early-stop. "
                f"Re-run study passes with --spec-draft-p-min 0."
            )

    return problems


def acceptance_rate(blocks: list[Block]) -> float:
    """Mean accepted tokens per verified block -- the number to reconcile against
    llama.cpp's own end-of-run acceptance report (Phase 0 sanity check)."""
    if not blocks:
        return float("nan")
    return sum(b.accepted_len for b in blocks) / len(blocks)
