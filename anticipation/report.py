"""Phase 3 report: python3 -m anticipation.report <run-dir>

Prints the consistency checks and the three analyses the gate depends on, and
writes the divergence figures next to the trace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import analysis, manifest as manifest_mod, reconcile, schema, vocab as vocab_mod


def _fmt(x, nd=3):
    try:
        if x != x:  # NaN
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="directory produced by harness/run_bank.py")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--vocab", default=None,
                    help="GGUF to read the tokenizer from (default: the trace's target model)")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    trace = run / "trace.jsonl" if run.is_dir() else run
    if not trace.exists():
        print(f"error: no trace at {trace}", file=sys.stderr)
        return 1

    header, blocks = schema.load(trace)
    print(f"trace:   {trace}")
    print(f"models:  {Path(header.model_tgt).name} <- {Path(header.model_dft).name}")
    print(f"spec:    {header.spec_type}  n_max={header.n_max}  p_min={header.p_min}")
    print(f"build:   {header.build_info}")
    print(f"blocks:  {len(blocks)}")
    print()

    problems = schema.check(header, blocks)

    mpath = run / "manifest.jsonl" if run.is_dir() else None
    if mpath and mpath.exists():
        problems += manifest_mod.label(header, blocks, manifest_mod.read(mpath))

    if problems:
        print("PROBLEMS")
        for p in problems[:20]:
            print(f"  ! {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        print()

    # --- Phase 0 gate ------------------------------------------------------
    slog = run / "server.log" if run.is_dir() else None
    if slog and slog.exists():
        rc = reconcile.check(blocks, slog)
        print("PHASE 0 RECONCILIATION  trace vs llama.cpp's own acceptance report")
        if rc["ok"] is None:
            print(f"  ? {rc['reason']}")
        elif rc["ok"]:
            print(f"  PASS  {rc['logged_accepted']} accepted / {rc['logged_generated']} "
                  f"generated over {rc['n_tasks']} tasks, exact match")
        else:
            print(f"  FAIL  accepted: logged {rc['logged_accepted']} vs reported "
                  f"{rc['reported_accepted']}")
            print(f"        generated: logged {rc['logged_generated']} vs reported "
                  f"{rc['reported_generated']}")
            print("        instrumentation is missing or double-counting blocks;")
            print("        nothing downstream of this is trustworthy.")
        print()

    # --- summary -----------------------------------------------------------
    s = analysis.summary(blocks)
    print("SUMMARY")
    print(f"  mean accepted_len   {_fmt(s['mean_accepted_len'])}   "
          f"(reconcile against llama.cpp's own acceptance report -- Phase 0 gate)")
    print(f"  mean draft len      {_fmt(s['mean_draft_len'])}")
    print(f"  divergence rate     {_fmt(s['divergence_rate'])}")
    print(f"  draft stop reasons  {s['stops']}")
    print(f"  near-miss / real    {s['near_miss']} / {s['real_divergence']} "
          f"(rank cut {s['rank_cut']}, unknown {s['unknown']})")
    print()

    # --- hazard ------------------------------------------------------------
    print("POSITION HAZARD  P(reject at k | drafted, survived to k)")
    for row in analysis.hazard(blocks, header.n_max or None):
        bar = "#" * int(round(40 * (row["hazard"] if row["hazard"] == row["hazard"] else 0)))
        print(f"  k={row['pos']}  n={row['at_risk']:6d}  "
              f"h={_fmt(row['hazard'])}  {bar}")
    print()

    # --- entropy control ---------------------------------------------------
    er = analysis.entropy_residual(blocks)
    print("ENTROPY CONTROL  accepted_len ~ mean target entropy")
    print(f"  n={er['n']}  slope={_fmt(er.get('slope'))}  R^2={_fmt(er.get('r2'))}")
    r2 = er.get("r2")
    if isinstance(r2, float) and r2 == r2:
        if r2 > 0.8:
            print("  -> divergence is largely a restatement of target uncertainty.")
            print("     The surprise-proxy line does not clear the gate on this run.")
        else:
            print(f"  -> {_fmt(1 - r2)} of variance is NOT explained by target entropy.")
            print("     That residual is the part that is about the draft, not the text.")
    print()

    # --- per-category ------------------------------------------------------
    cats = analysis.by_category(blocks)
    if len(cats) > 1:
        print("BY CATEGORY")
        for name, bs in sorted(cats.items()):
            cs = analysis.summary(bs)
            print(f"  {name:14s} n={len(bs):5d}  acc={_fmt(cs['mean_accepted_len'])}  "
                  f"div={_fmt(cs['divergence_rate'])}  "
                  f"near/real={cs['near_miss']}/{cs['real_divergence']}")
        print()

    # --- subject clustering ------------------------------------------------
    srows = analysis.subject_rates(blocks)
    if srows:
        print("DIVERGENCE BY SUBJECT  (vs pooled base rate; |z|>2 is worth a look)")
        for r in srows:
            flag = " *" if abs(r["z"]) > 2 else ""
            print(f"  {r['subject']:16s} n={r['n_blocks']:5d}  "
                  f"div={_fmt(r['divergence_rate'])}  "
                  f"delta={r['delta']:+.3f}  z={r['z']:+.1f}{flag}")
        print()

    # --- blind-spot map ----------------------------------------------------
    print(f"BLIND-SPOT MAP  top {args.top_n} mismatch pairs")
    v = vocab_mod.load(args.vocab or header.model_tgt)
    detok = v.repr if v else None
    rows = analysis.mismatch_table(blocks, detok=detok, top_n=args.top_n)
    if not rows:
        print("  (no divergences recorded)")
    for r in rows:
        if detok:
            print(f"  {r['n']:5d}x  {r['draft_text']:>18s} -> {r['target_text']:<18s}  "
                  f"p_of_draft={_fmt(r['mean_p_of_draft'])}  rank~{r['median_rank']}")
        else:
            print(f"  {r['n']:5d}x  draft={r['draft_token']:<7d} target={r['target_token']:<7d}  "
                  f"p_of_draft={_fmt(r['mean_p_of_draft'])}  rank~{r['median_rank']}")
    if not detok:
        print()
        print(f"  (could not read a vocabulary from {header.model_tgt} -- "
              f"pass --vocab to decode these ids)")

    if not args.no_plots and run.is_dir():
        try:
            from . import plots
            paths = plots.render(header, blocks, run)
            print()
            print("FIGURES")
            for p in paths:
                print(f"  {p}")
        except ImportError:
            print("\n(matplotlib not installed -- skipping figures)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
