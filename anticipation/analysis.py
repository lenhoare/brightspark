"""Analysis primitives for spec-trace logs.

The three things Phase 3 gates on:
  hazard()            -- position-conditioned rejection, the censoring-aware
                         replacement for "mean accepted_len"
  entropy_residual()  -- divergence with target uncertainty regressed out; the
                         residual is the part that is actually about the draft
  mismatch_table()    -- the blind-spot map, graded by how wrong the draft was
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .schema import Block


def hazard(blocks: list[Block], n_max: int | None = None) -> list[dict]:
    """Per-position rejection hazard: P(reject at k | drafted and survived to k).

    A position is *at risk* only if the draft actually proposed a token there
    (n_draft > k) and every earlier position was accepted. This is what makes
    the measure comparable between a block that accepted 7 and one that
    accepted 2 -- a raw mean over the block is not.
    """
    if n_max is None:
        n_max = max((b.n_max for b in blocks), default=0)

    at_risk = [0] * n_max
    rejects = [0] * n_max

    for b in blocks:
        for k in range(min(b.n_draft, n_max)):
            if b.accepted_len < k:
                break  # never reached this position
            at_risk[k] += 1
            if b.accepted_len == k:
                rejects[k] += 1
                break

    out = []
    for k in range(n_max):
        n = at_risk[k]
        out.append({
            "pos": k,
            "at_risk": n,
            "rejects": rejects[k],
            "hazard": rejects[k] / n if n else float("nan"),
        })
    return out


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit y = a + b*x, without pulling in numpy."""
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def entropy_residual(blocks: list[Block]) -> dict:
    """Regress accepted_len on mean target entropy and return the residuals.

    If divergence is fully explained by how uncertain the target itself was,
    the draft head is telling us nothing new and the surprise-proxy line stops
    here. The residual -- diverged *more* than the target's own uncertainty
    predicts -- is the quantity worth building on.
    """
    pairs = [
        (b.mean_target_entropy, float(b.accepted_len), b)
        for b in blocks
        if b.target_entropy
    ]
    pairs = [(x, y, b) for x, y, b in pairs if math.isfinite(x)]

    if len(pairs) < 2:
        return {"n": len(pairs), "r2": float("nan"), "residuals": []}

    xs = [x for x, _, _ in pairs]
    ys = [y for _, y, _ in pairs]

    a, slope = _ols(xs, ys)

    my = sum(ys) / len(ys)
    ss_tot = sum((y - my) ** 2 for y in ys)
    resid = [y - (a + slope * x) for x, y, _ in pairs]
    ss_res = sum(r * r for r in resid)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "n": len(pairs),
        "intercept": a,
        "slope": slope,
        "r2": r2,
        # residual < 0 means the block diverged more than target entropy predicts
        "residuals": [
            {"block_idx": b.block_idx, "residual": r, "block": b}
            for (_, _, b), r in zip(pairs, resid)
        ],
    }


def mismatch_table(blocks: list[Block], detok=None, top_n: int = 30) -> list[dict]:
    """The blind-spot map: most frequent (draft, target) mismatch pairs.

    Graded by the target's own probability of the rejected draft token, so that
    near-miss synonyms can be separated from genuinely different continuations
    rather than drowning the table.
    """
    counts: Counter = Counter()
    p_sums: defaultdict = defaultdict(float)
    ranks: defaultdict = defaultdict(list)

    for b in blocks:
        pair = b.mismatch_pair()
        if pair is None:
            continue
        counts[pair] += 1
        i = b.mismatch_pos
        if i < len(b.target_p_of_draft) and b.target_p_of_draft[i] is not None:
            p_sums[pair] += b.target_p_of_draft[i]
        if i < len(b.draft_rank_in_target):
            ranks[pair].append(b.draft_rank_in_target[i])

    rows = []
    for pair, n in counts.most_common(top_n):
        rank_list = ranks.get(pair, [])
        rows.append({
            "draft_token": pair[0],
            "target_token": pair[1],
            "draft_text": detok(pair[0]) if detok else None,
            "target_text": detok(pair[1]) if detok else None,
            "n": n,
            "mean_p_of_draft": p_sums[pair] / n if n else float("nan"),
            "median_rank": sorted(rank_list)[len(rank_list) // 2] if rank_list else None,
        })
    return rows


def near_miss_split(blocks: list[Block], rank_cut: int = 5) -> dict:
    """Split divergences into near-misses and real failures.

    A rejection where the target had the draft token in its top few is a
    different phenomenon from one where the draft was nowhere near -- lumping
    them together is what makes raw acceptance rate a blunt signal.
    """
    near = real = unknown = 0
    for b in blocks:
        if not b.diverged:
            continue
        i = b.mismatch_pos
        if i >= len(b.draft_rank_in_target):
            unknown += 1
            continue
        r = b.draft_rank_in_target[i]
        if r < 0:
            unknown += 1
        elif r < rank_cut:
            near += 1
        else:
            real += 1
    return {"near_miss": near, "real_divergence": real, "unknown": unknown,
            "rank_cut": rank_cut}


def by_category(blocks: list[Block]) -> dict[str, list[Block]]:
    out: defaultdict = defaultdict(list)
    for b in blocks:
        out[b.category or "unlabelled"].append(b)
    return dict(out)


def by_subject(blocks: list[Block]) -> dict[str, list[Block]]:
    """Bucket by subject, for the blind-spot map's central question.

    Only Set A carries a real subject; everything else lands in "none" and
    should be excluded before comparing rates.
    """
    out: defaultdict = defaultdict(list)
    for b in blocks:
        out[b.subject or "none"].append(b)
    return dict(out)


def subject_rates(blocks: list[Block]) -> list[dict]:
    """Divergence rate per subject against the pooled base rate.

    This is the test the blind-spot map exists to support: does mismatch
    cluster by subject, or is every subject sitting on the same base rate?
    Non-mismatch blocks are the control, which is why context_tail is logged
    on every block and not only on divergences.
    """
    subs = {k: v for k, v in by_subject(blocks).items() if k != "none"}
    if not subs:
        return []

    pooled = [b for bs in subs.values() for b in bs]
    base = sum(1 for b in pooled if b.diverged) / len(pooled)

    rows = []
    for name, bs in subs.items():
        n_div = sum(1 for b in bs if b.diverged)
        rate = n_div / len(bs)
        # binomial SE under the pooled base rate, for a rough z
        se = (base * (1 - base) / len(bs)) ** 0.5
        rows.append({
            "subject": name,
            "n_blocks": len(bs),
            "divergence_rate": rate,
            "delta": rate - base,
            "z": (rate - base) / se if se > 0 else float("nan"),
            "mean_accepted_len": sum(b.accepted_len for b in bs) / len(bs),
        })

    rows.sort(key=lambda r: r["delta"], reverse=True)
    return rows


def summary(blocks: list[Block]) -> dict:
    n = len(blocks)
    if n == 0:
        return {"n_blocks": 0}
    n_div = sum(1 for b in blocks if b.diverged)
    return {
        "n_blocks": n,
        "mean_accepted_len": sum(b.accepted_len for b in blocks) / n,
        "divergence_rate": n_div / n,
        "mean_draft_len": sum(b.n_draft for b in blocks) / n,
        "stops": dict(Counter(b.draft_stop for b in blocks)),
        **near_miss_split(blocks),
    }
