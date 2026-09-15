# Anticipation

Instrumentation for block speculative decoders (DSpark, DFlash, MTP) that logs,
per draft block, how confident the draft model was and where it diverged from
the target. See `draft_spec.md` for what it is for and what the evidence gate is.

Status: Phases 0 and 1 complete and verified on real DSpark output.

## Layout

```
anticipation/       analysis library (stdlib only, except plots)
  schema.py         trace reader + internal-consistency checker
  reconcile.py      Phase 0 gate: trace vs llama.cpp's own acceptance report
  analysis.py       hazard, entropy control, blind-spot map
  vocab.py          offline token id -> text, read from the target GGUF
  plots.py          figures
  report.py         `python3 -m anticipation.report <run-dir>`
  export_replay.py  builds the visualiser payload from a trace
harness/run_bank.py launches a traced server and drives a prompt bank
harness/validate_bank.py  checks a generated prompt bank
viz/serve.py        local web app (Draft Graveyard)
viz/app.html        its single page
prompts/            prompt banks (bank.example.jsonl is a shape example only)
runs/               one self-describing directory per run
```

## Setup

The instrumented llama.cpp lives on branch `spec-trace` in `~/llama.cpp`:

```
cd ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target llama-server -j4
```

```
python3 -m venv .venv
.venv/bin/pip install matplotlib ~/llama.cpp/gguf-py
```

## Running

```
.venv/bin/python harness/run_bank.py \
  --model       /mnt/d/models/MiniCPM5-2B-Q8_0.gguf \
  --model-draft /mnt/d/models/MiniCPM5-2.6B-DSpark.gguf \
  --spec-type   draft-dspark \
  --bank        prompts/bank.example.jsonl \
  --out         runs/myrun \
  --n-predict   128

.venv/bin/python -m anticipation.report runs/myrun
```

A run directory holds `trace.jsonl`, `server.log`, `bank.jsonl`,
`manifest.jsonl`, `responses.json` and `run.json`, so a run is reproducible
from itself alone.

## Watching a run

```
.venv/bin/python viz/serve.py --open      # http://127.0.0.1:8081
```

**Draft Graveyard** replays a recorded run token by token. Tokens the draft
predicted correctly are underlined in blue; where the draft broke, the target's
real token is filled green and the draft's rejected guess falls down the screen
and piles up at the bottom — that pile is the discarded compute. Falling chips
deepen in rose with how far down the target ranked the guess, so a pale chip is
a near-miss and a saturated one is a genuinely different continuation.

Every run under `runs/` appears in the run dropdown; replays are built on first
request (a few seconds, mostly reading the tokenizer out of the GGUF) and
cached as `replay.json` next to the trace, rebuilt whenever the trace is newer.
Nothing needs regenerating by hand after a new run — just reload.

The server binds to localhost only and serves nothing outside `runs/`.

## Reading a trace

One header record, then one record per verified draft block. Two rules matter:

- **Verify-side arrays are shorter than draft-side arrays.** They cover only
  the positions the target evaluated (`n_verified = accepted_len + 1` on a
  mismatch). The shortfall is the censoring, not missing data. Never pad it —
  that is what `hazard()` exists to handle.
- **`draft_p` is top-10 truncated and renormalised** (the draft sampler runs
  `top_k=10`). It saturates at 1.0. Use `draft_conf`, DSpark's native
  confidence head, as the certainty signal; it has real dynamic range and
  tracks acceptance better.

`n_draft_gen` vs `n_draft`: near the end of a generation the server caps a
draft to the remaining budget after the implementation has already produced a
full block. The trace is aligned to what the target actually saw; `n_draft_gen`
records what was generated.

## Gates

`report.py` prints these in order, and they are meant to be able to fail:

1. **Phase 0 reconciliation** — the trace's totals must exactly match
   llama.cpp's own acceptance report. A FAIL means nothing downstream is
   trustworthy.
2. **Consistency checks** — array lengths, mismatch positions, and a hard error
   if `p_min > 0` truncated any draft in the run being analysed.
3. **Entropy control** — `accepted_len` regressed on target entropy. A high R²
   means divergence is just restating how uncertain the target already was, and
   the surprise-proxy line stops there. The residual is the part that is about
   the draft rather than the text.

## Study defaults

- `--spec-draft-p-min 0` — required; otherwise the draft's own early-stop is
  confounded with target rejection
- temperature 0 — acceptance semantics differ under sampling
- `--parallel 1` — required for the task_id -> prompt_id mapping to hold

## Generating the prompt bank

`prompts/GENERATE.md` is a self-contained brief to hand to a cheaper model. It
specifies three sets — 120 subject prompts with confounds held flat, 30
confound probes, 24 twist matched pairs — and the exact JSONL schema.

Validate whatever comes back before running it:

```
python3 harness/validate_bank.py yourbank.jsonl --strict-counts
```

It catches markdown fences, duplicate ids, digits leaking into Set A,
unbalanced or badly matched twist pairs, and drifted per-subject counts. Hand
the errors straight back to the generating model and re-run until it passes.

## Not done yet

- Phase 2: running the real bank. `prompts/bank.example.jsonl` is a shape
  example; five prompts is a smoke test, not a study.
- Subject bucketing of `context_tail` (deferred by design; the log already
  carries the context on every block, mismatch or not, so it needs no re-run).
