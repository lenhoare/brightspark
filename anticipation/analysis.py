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
import statistics
from collections import Counter, defaultdict

from . import tokens as tokmod
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


def mismatch_table(blocks: list[Block], detok=None, top_n: int = 30,
                   kinds: set[str] | None = None, raw=None) -> list[dict]:
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
        if kinds is not None and raw is not None:
            if tokmod.classify(raw(pair[1])) not in kinds:
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
        d_text = detok(pair[0]) if detok else None
        t_text = detok(pair[1]) if detok else None
        rows.append({
            "draft_token": pair[0],
            "target_token": pair[1],
            "draft_text": d_text,
            "target_text": t_text,
            "kind": tokmod.classify(raw(pair[1])) if raw else None,
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


def kind_breakdown(blocks: list[Block], raw) -> list[dict]:
    """What kind of token is the target producing where the draft breaks?

    This is what replaces forbidding numerals when the bank is written: the
    numeral-driven divergences are still there, they are just labelled, so they
    can be read separately or excluded from the subject comparison.
    """
    counts: Counter = Counter()
    ranks: defaultdict = defaultdict(list)

    for b in blocks:
        pair = b.mismatch_pair()
        if pair is None:
            continue
        k = tokmod.classify(raw(pair[1]))
        counts[k] += 1
        i = b.mismatch_pos
        if i < len(b.draft_rank_in_target):
            ranks[k].append(b.draft_rank_in_target[i])

    total = sum(counts.values()) or 1
    rows = []
    for k in tokmod.KINDS:
        n = counts.get(k, 0)
        if not n:
            continue
        rl = sorted(ranks.get(k, []))
        rows.append({
            "kind": k,
            "n": n,
            "share": n / total,
            "median_rank": rl[len(rl) // 2] if rl else None,
        })
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows


def _ols_multi(X: list[list[float]], y: list[float]) -> list[float] | None:
    """Least squares with an intercept, via the normal equations.

    Small and dependency-free: a handful of covariates is all this needs.
    Returns None if the system is singular (a constant or duplicated column).
    """
    n = len(y)
    if n < len(X[0]) + 2:
        return None

    A = [[1.0] + row for row in X]
    p = len(A[0])

    # normal equations: (A'A) b = A'y
    M = [[sum(A[i][r] * A[i][c] for i in range(n)) for c in range(p)] +
         [sum(A[i][r] * y[i] for i in range(n))] for r in range(p)]

    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(p):
            if r != col and M[r][col]:
                f = M[r][col]
                M[r] = [a - f * b for a, b in zip(M[r], M[col])]

    return [M[r][p] for r in range(p)]


def numeral_density(b: Block) -> float:
    """Digit density of the text immediately preceding this block.

    Uses context_tail, which is logged on every block precisely so that
    non-divergent blocks can serve as the control.
    """
    return tokmod.density(b.context_tail)


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

    # adjust for the two things that drive divergence regardless of subject:
    # how uncertain the target already was, and how many digits are around.
    # this is control by covariate, and is only as good as the corpus allows --
    # if numerals concentrate in one subject, the two cannot be separated.
    usable = [b for b in pooled if b.target_entropy and
              math.isfinite(b.mean_target_entropy)]
    adj: dict[int, float] = {}
    if len(usable) > 20:
        X = [[b.mean_target_entropy, numeral_density(b)] for b in usable]
        y = [1.0 if b.diverged else 0.0 for b in usable]
        coef = _ols_multi(X, y)
        if coef:
            for b, row in zip(usable, X):
                pred = coef[0] + coef[1] * row[0] + coef[2] * row[1]
                adj[id(b)] = (1.0 if b.diverged else 0.0) - pred

    rows = []
    for name, bs in subs.items():
        n_div = sum(1 for b in bs if b.diverged)
        rate = n_div / len(bs)
        se = (base * (1 - base) / len(bs)) ** 0.5
        res = [adj[id(b)] for b in bs if id(b) in adj]
        rows.append({
            "subject": name,
            "n_blocks": len(bs),
            "divergence_rate": rate,
            "delta": rate - base,
            "z": (rate - base) / se if se > 0 else float("nan"),
            "adj_delta": (sum(res) / len(res)) if res else float("nan"),
            "n_adj": len(res),
            "digit_density": sum(numeral_density(b) for b in bs) / len(bs),
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


def _t_crit(df: int) -> float:
    """Two-sided 0.05 critical t. Table lookup beats a dependency here."""
    tbl = {1:12.71, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365,
           8:2.306, 9:2.262, 10:2.228, 12:2.179, 15:2.131, 20:2.086,
           23:2.069, 25:2.060, 30:2.042, 40:2.021, 60:2.000}
    if df <= 0:
        return float("inf")
    for k in sorted(tbl):
        if df <= k:
            return tbl[k]
    return 1.96


def subject_rates_by_prompt(blocks: list[Block]) -> dict:
    """Subject comparison with the prompt as the unit of replication.

    Blocks are nested inside prompts, so a block-level test treats hundreds of
    correlated observations as independent and reports significance that is not
    there. Each prompt contributes one divergence rate; subjects are compared on
    those. Also reports the between/within variance ratio, which says how much
    subject matters at all next to which particular prompt was written.
    """
    per_prompt: defaultdict = defaultdict(list)
    for b in blocks:
        if b.subject and b.subject != "none" and b.prompt_id:
            per_prompt[(b.subject, b.prompt_id)].append(b)

    if not per_prompt:
        return {"rows": [], "n_prompts": 0}

    rates: defaultdict = defaultdict(list)
    for (subj, _pid), blk in per_prompt.items():
        rates[subj].append(sum(1 for x in blk if x.diverged) / len(blk))

    pooled = [r for v in rates.values() for r in v]
    grand = statistics.mean(pooled)

    rows = []
    for subj, v in rates.items():
        m = statistics.mean(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        t = (m - grand) / (sd / len(v) ** 0.5) if sd > 0 else float("nan")
        rows.append({
            "subject": subj,
            "n_prompts": len(v),
            "mean": m,
            "sd": sd,
            "t": t,
            "df": len(v) - 1,
            "sig": (t == t) and abs(t) > _t_crit(len(v) - 1),
        })
    rows.sort(key=lambda r: r["mean"], reverse=True)

    means = [r["mean"] for r in rows]
    between = statistics.variance(means) if len(means) > 1 else 0.0
    within = statistics.mean([r["sd"] ** 2 for r in rows]) if rows else 0.0

    return {
        "rows": rows,
        "n_prompts": len(pooled),
        "grand_mean": grand,
        "between": between,
        "within": within,
        "ratio": (between / within) if within > 0 else float("nan"),
    }


def twist_pairs(blocks: list[Block], window: int | None = None) -> dict:
    """The pre-registered paired test: does the reversal arm diverge more?

    `window` limits each arm to its first N blocks, which is where the cue sits;
    None uses the whole generation. Reports a manipulation check alongside --
    if the target's own entropy did not rise on the turn arm, the pair did not
    create the contrast it was supposed to and a null says nothing.
    """
    arms: defaultdict = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        if b.category == "twist" and b.pair_id and b.arm:
            arms[b.pair_id][b.arm].append(b)

    def summarise(blk):
        blk = sorted(blk, key=lambda x: x.block_idx)
        if window:
            blk = blk[:window]
        if not blk:
            return None
        ents = [x.mean_target_entropy for x in blk if x.target_entropy]
        return {
            "acc": statistics.mean([x.accepted_len for x in blk]),
            "div": sum(1 for x in blk if x.diverged) / len(blk),
            "ent": statistics.mean(ents) if ents else float("nan"),
        }

    d_acc, d_div, d_ent, wins = [], [], [], 0
    for _pid, a in sorted(arms.items()):
        f, t = summarise(a.get("flat", [])), summarise(a.get("turn", []))
        if not f or not t:
            continue
        d_acc.append(t["acc"] - f["acc"])
        d_div.append(t["div"] - f["div"])
        if math.isfinite(f["ent"]) and math.isfinite(t["ent"]):
            d_ent.append(t["ent"] - f["ent"])
        if t["acc"] < f["acc"]:
            wins += 1

    n = len(d_acc)
    if n < 2:
        return {"n": n}

    m = statistics.mean(d_acc)
    sd = statistics.stdev(d_acc)
    t = m / (sd / n ** 0.5) if sd > 0 else float("nan")

    return {
        "n": n,
        "window": window,
        "d_acc": m,
        "sd": sd,
        "t": t,
        "sig": (t == t) and abs(t) > _t_crit(n - 1),
        "d_div": statistics.mean(d_div),
        "d_ent": statistics.mean(d_ent) if d_ent else float("nan"),
        "wins": wins,
    }
