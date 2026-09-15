"""Export a run as a self-contained replay payload for the visualiser.

    python3 -m anticipation.export_replay runs/dspark01

Reconstructs the generated token stream from the trace alone, annotating each
token with whether the draft predicted it, what it guessed instead, and how
badly it was wrong. The first tokens of a generation come from prefill rather
than a verified block, so the leading text is recovered from responses.json
and marked as un-drafted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import manifest as manifest_mod, schema, vocab as vocab_mod


def build(run_dir: Path, vocab_path: str | None = None) -> dict:
    header, blocks = schema.load(run_dir / "trace.jsonl")

    mpath = run_dir / "manifest.jsonl"
    entries = manifest_mod.read(mpath) if mpath.exists() else []
    if entries:
        manifest_mod.label(header, blocks, entries)

    v = vocab_mod.load(vocab_path or header.model_tgt)
    if v is None:
        raise SystemExit(f"could not read a tokenizer from {vocab_path or header.model_tgt}")

    rpath = run_dir / "responses.json"
    responses = json.loads(rpath.read_text()) if rpath.exists() else []

    # what the model was actually sent, which is not always what the bank says
    cfg = {}
    cpath = run_dir / "run.json"
    if cpath.exists():
        try:
            cfg = json.loads(cpath.read_text()).get("args") or {}
        except Exception:
            cfg = {}

    by_task: dict[int, list] = {}
    for b in blocks:
        by_task.setdefault(b.task_id, []).append(b)

    bank = {e.get("prompt_id"): e for e in entries}

    takes = []
    for order, (task_id, bs) in enumerate(sorted(by_task.items())):
        bs.sort(key=lambda b: b.block_idx)

        meta = entries[order] if order < len(entries) else {}
        resp = responses[order] if order < len(responses) else {}

        tokens = []
        for b in bs:
            n_acc = b.accepted_len
            # accepted tokens: the draft called these correctly
            for i in range(n_acc):
                tokens.append({
                    "id": b.target_token[i],
                    "hit": True,
                    "k": i,
                    "block": b.block_idx,
                })
            # the target's token at the break point (or the bonus token on a
            # full accept) -- this one the draft did not supply
            if n_acc < len(b.target_token):
                tok = {
                    "id": b.target_token[n_acc],
                    "hit": False,
                    "k": n_acc,
                    "block": b.block_idx,
                }
                if b.diverged and n_acc < len(b.draft_tokens):
                    tok["miss"] = v.text(b.draft_tokens[n_acc])
                    tok["miss_id"] = b.draft_tokens[n_acc]
                    if n_acc < len(b.target_p_of_draft):
                        p = b.target_p_of_draft[n_acc]
                        tok["p"] = round(p, 5) if p is not None else None
                    if n_acc < len(b.draft_rank_in_target):
                        tok["rank"] = b.draft_rank_in_target[n_acc]
                else:
                    tok["bonus"] = True
                tokens.append(tok)

        # byte-level tokens can hold a partial character, so the text has to be
        # decoded across the sequence rather than token by token -- otherwise a
        # curly quote arrives as three replacement marks
        pieces = v.stream_text([t["id"] for t in tokens])
        for tok, piece in zip(tokens, pieces):
            tok["t"] = piece
            del tok["id"]

        # recover the prefill text that precedes the first verified block
        prefix = ""
        content = resp.get("content", "")
        if content and tokens:
            head = "".join(t["t"] for t in tokens[:20])
            idx = content.find(head[:60]) if head else -1
            if idx > 0:
                prefix = content[:idx]

        n_div = sum(1 for b in bs if b.diverged)
        takes.append({
            "prompt_id": meta.get("prompt_id") or f"task_{task_id}",
            "category": meta.get("category"),
            "subject": meta.get("subject"),
            "prompt": meta.get("prompt", ""),
            "finish_reason": resp.get("finish_reason"),
            "prefix": prefix,
            "tokens": tokens,
            "stats": {
                "blocks": len(bs),
                "divergences": n_div,
                "accepted": sum(b.accepted_len for b in bs),
                "drafted": sum(b.n_draft for b in bs),
                "mean_accepted": round(sum(b.accepted_len for b in bs) / len(bs), 3) if bs else 0,
            },
        })

    return {
        "run": run_dir.name,
        "model_tgt": Path(header.model_tgt).name,
        "model_dft": Path(header.model_dft).name or "(self-contained)",
        "spec_type": header.spec_type,
        "n_max": header.n_max,
        "build": header.build_info,
        "system": cfg.get("system", ""),
        "thinking": bool(cfg.get("thinking")),
        "takes": takes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None, help="default: <run_dir>/replay.json")
    ap.add_argument("--vocab", default=None)
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    payload = build(run, args.vocab)

    out = Path(args.out) if args.out else run / "replay.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    n_tok = sum(len(t["tokens"]) for t in payload["takes"])
    n_miss = sum(1 for t in payload["takes"] for k in t["tokens"] if k.get("miss"))
    print(f"{out}  {len(payload['takes'])} takes, {n_tok} tokens, {n_miss} mismatches, "
          f"{out.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
