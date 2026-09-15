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
    ap.add_argument("--kinds", default=None,
                    help="restrict the blind-spot map to these token kinds, "
                         "e.g. word,capitalised (default: all)")
    ap.add_argument("--vocab", default=None,
                    help="GGUF to read the tokenizer from (default: the trace's target model)")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    trace = run / "trace.jsonl" if run.is_dir() else run
    if not trace.exists():
        print(f"error: no trace at {trace}", file=sys.stderr)
        return 1

    kinds = set(k.strip() for k in args.kinds.split(",")) if args.kinds else None

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

    # --- the pre-registered surprise test -----------------------------------
    tw_cue   = analysis.twist_pairs(blocks, window=2)
    tw_whole = analysis.twist_pairs(blocks, window=None)
    if tw_cue.get("n", 0) >= 2:
        print("TWIST PAIRS  pre-registered test: does the reversal arm diverge more?")
        for tag, r in (("at the cue (first 2 blocks)", tw_cue), ("whole generation", tw_whole)):
            print(f"  {tag}  n={r['n']} pairs")
            print(f"    accepted_len  turn-flat  {r['d_acc']:+.3f}  "
                  f"(sd {r['sd']:.3f}, t({r['n']-1}) = {r['t']:+.2f})"
                  + ("  SIGNIFICANT" if r["sig"] else ""))
            print(f"    divergence    turn-flat  {r['d_div']:+.3f}")
            print(f"    sign test: turn diverged more in {r['wins']}/{r['n']} pairs")
            print(f"    manipulation check, target entropy turn-flat  {r['d_ent']:+.3f}")
        if not tw_cue["sig"] and not tw_whole["sig"]:
            print("  -> no effect. If the manipulation check is positive, the pairs did")
            print("     create the contrast and the draft simply did not follow it:")
            print("     divergence is not tracking surprise in this design.")
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

    # --- subject clustering, prompt as the unit of replication --------------
    pl = analysis.subject_rates_by_prompt(blocks)
    if pl["rows"]:
        print("DIVERGENCE BY SUBJECT  prompt-level (the unit of replication)")
        print(f"  {pl['n_prompts']} prompts, grand mean divergence {_fmt(pl['grand_mean'])}")
        print(f"  {'subject':16s} {'n':>3s} {'mean':>7s} {'sd':>6s} {'t':>7s}")
        for r in pl["rows"]:
            star = " *" if r["sig"] else ""
            print(f"  {r['subject']:16s} {r['n_prompts']:3d} {r['mean']:7.3f} "
                  f"{r['sd']:6.3f} {r['t']:+7.2f}{star}")
        print(f"  between-subject variance {pl['between']:.5f} vs within-subject "
              f"{pl['within']:.5f}  ->  ratio {_fmt(pl['ratio'], 2)}")
        print("  a ratio well below 1 means which prompt was written matters more")
        print("  than which subject it belongs to. A subject only counts as real if")
        print("  its prompts agree with each other -- look at sd, not just the mean.")
        print()

    # --- the same comparison per block, which overstates significance --------
    srows = analysis.subject_rates(blocks)
    if srows:
        print("DIVERGENCE BY SUBJECT  per block, adjusted for entropy and digit density")
        for r in srows:
            flag = " *" if abs(r["z"]) > 2 else ""
            print(f"  {r['subject']:16s} n={r['n_blocks']:5d}  div={_fmt(r['divergence_rate'])}  "
                  f"delta={r['delta']:+.3f}  z={r['z']:+.1f}{flag}  "
                  f"adj={r['adj_delta']:+.3f}  digits={r['digit_density']*100:.1f}%")
        print("  adj is the mean residual after regressing divergence on target entropy")
        print("  and digit density -- if raw and adj disagree, the raw gap was the")
        print("  covariates talking. A subject whose digit share is far from the rest")
        print("  cannot be fully separated from them by this adjustment.")
        print("  NOTE: these z values treat every block as independent, which they are")
        print("  not -- they are nested in prompts. Trust the prompt-level table above.")
        print()

    # --- blind-spot map ----------------------------------------------------
    v = vocab_mod.load(args.vocab or header.model_tgt)
    detok = v.repr if v else None

    if detok:
        krows = analysis.kind_breakdown(blocks, v.text)
        if krows:
            print("DIVERGENCE BY TOKEN KIND  what the target produced at the break")
            for r in krows:
                bar = "#" * int(round(34 * r["share"]))
                print(f"  {r['kind']:12s} {r['n']:5d}  {r['share']*100:5.1f}%  "
                      f"rank~{str(r['median_rank']):>4s}  {bar}")
            print("  numerals and punctuation are expected to lead here; they are")
            print("  labelled rather than excluded, so subject effects can be read")
            print("  with --kinds word,capitalised.")
            print()

    print(f"BLIND-SPOT MAP  top {args.top_n} mismatch pairs"
          + (f"  [kinds: {','.join(sorted(kinds))}]" if kinds else ""))
    rows = analysis.mismatch_table(blocks, detok=detok, top_n=args.top_n,
                                   kinds=kinds, raw=(v.text if v else None))
    if not rows:
        print("  (no divergences recorded)")
    for r in rows:
        if detok:
            print(f"  {r['n']:5d}x  {r['draft_text']:>18s} -> {r['target_text']:<18s}  "
                  f"{(r['kind'] or ''):<11s} p_of_draft={_fmt(r['mean_p_of_draft'])}  "
                  f"rank~{r['median_rank']}")
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
