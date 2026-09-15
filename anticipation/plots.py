"""Figures for the Phase 3 divergence report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import analysis  # noqa: E402
from .schema import Block, Header  # noqa: E402


def _timeline(blocks: list[Block], ax_acc, ax_conf) -> None:
    xs = list(range(len(blocks)))
    acc = [b.accepted_len for b in blocks]

    ax_acc.plot(xs, acc, lw=0.8)
    ax_acc.set_ylabel("accepted_len")
    ax_acc.set_title("convergence over generation")

    # shade divergence events rather than marking them, so runs of them read as runs
    for x, b in zip(xs, blocks):
        if b.diverged:
            ax_acc.axvspan(x - 0.5, x + 0.5, color="tab:red", alpha=0.10, lw=0)

    # DSpark's own confidence where available, else the token-head top-1
    conf, src = [], "draft_conf"
    for b in blocks:
        vals = [c for c in b.draft_conf if c is not None]
        if not vals:
            vals, src = b.draft_p, "draft_p (top-10 renormalised)"
        conf.append(sum(vals) / len(vals) if vals else float("nan"))

    ax_conf.plot(xs, conf, lw=0.8, color="tab:purple")
    ax_conf.set_ylabel(src)
    ax_conf.set_xlabel("block index")
    ax_conf.set_title("draft certainty over generation")


def _hazard(header: Header, blocks: list[Block], ax) -> None:
    rows = analysis.hazard(blocks, header.n_max or None)
    ks = [r["pos"] for r in rows]
    hs = [r["hazard"] for r in rows]
    ax.bar(ks, hs, color="tab:blue")
    ax.set_xlabel("position within block (k)")
    ax.set_ylabel("P(reject | survived to k)")
    ax.set_title("position-conditioned rejection hazard")


def _entropy(blocks: list[Block], ax) -> None:
    pts = [(b.mean_target_entropy, b.accepted_len) for b in blocks if b.target_entropy]
    pts = [(x, y) for x, y in pts if x == x]
    if not pts:
        ax.set_visible(False)
        return

    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    ax.scatter(xs, ys, s=4, alpha=0.25)

    er = analysis.entropy_residual(blocks)
    if er.get("slope") == er.get("slope"):
        lo, hi = min(xs), max(xs)
        ax.plot([lo, hi],
                [er["intercept"] + er["slope"] * lo, er["intercept"] + er["slope"] * hi],
                color="tab:red", lw=1.2)
        ax.set_title(f"accepted_len vs target entropy (R²={er['r2']:.3f})")
    ax.set_xlabel("mean target entropy over verified positions (nats)")
    ax.set_ylabel("accepted_len")


def _by_category(blocks: list[Block], ax) -> bool:
    cats = analysis.by_category(blocks)
    if len(cats) < 2:
        ax.set_visible(False)
        return False

    names = sorted(cats)
    data = [[b.accepted_len for b in cats[n]] for n in names]
    ax.boxplot(data, tick_labels=names, showmeans=True)
    ax.set_ylabel("accepted_len")
    ax.set_title("divergence by prompt category")
    ax.tick_params(axis="x", rotation=20)
    return True


def render(header: Header, blocks: list[Block], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    written = []

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    _timeline(blocks, axes[0], axes[1])
    fig.tight_layout()
    p = out_dir / "timeline.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _hazard(header, blocks, axes[0])
    _entropy(blocks, axes[1])
    _by_category(blocks, axes[2])
    fig.tight_layout()
    p = out_dir / "divergence.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    return written
