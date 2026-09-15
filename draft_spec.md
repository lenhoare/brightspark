# Anticipation — DSpark Divergence Instrumentation Spec

## Goal

Instrument DSpark speculative decoding to expose, per draft block, how the
draft model's confidence evolves and where it diverges from the target
model's actual output.

Two outputs, both useful, in priority order:

1. **A blind-spot map.** Where does the 324M draft head systematically fail
   to predict the 2B target? If mismatch clusters by subject matter,
   register, or syntactic context, that is a directly useful artefact
   regardless of what it means — it says what the cheap predictor doesn't
   know. Bucketing sentences by subject (likely with a separate model) is
   deferred, but the log must carry enough context to make it possible
   later without a re-run.
2. **A surprise proxy.** Divergence as a computational stand-in for
   certainty / surprise / novelty / boredom. No affect model gets built
   until Phase 3 produces evidence the signal is real and not just
   quantisation/precision noise or a restatement of target entropy.

## Status (2026-09-15)

Phases 0 and 1 are **done and verified on real DSpark output**. The Phase 0
gate passes exactly: llama.cpp's own report and the trace independently agree
at 437 accepted / 1372 generated. Phase 2 (prompt bank) is the next step and
is authoring work, not engineering. See `README.md` for how to run it.

## Models

- **Target:** `openbmb/MiniCPM5-2B` (GGUF, Q8_0 or F16 — small enough to run
  near-full-precision on the 3060, sidestepping the draft/target hidden-state
  precision mismatch that hurts DFlash on quantised targets)
- **Draft:** `MiniCPM5-2.6B-DSpark.gguf` (334 MB on disk)
- **Runtime:** llama.cpp branch `spec-trace` in `~/llama.cpp` (from master
  b10989-38a5b42d9), which adds `--spec-trace`. DSpark is served by the
  **dflash** implementation with an `is_dspark` flag, not by the eagle3 one —
  `common_speculative_impl_draft_dflash` in `common/speculative.cpp`.

## Phase 0 — Baseline run

Confirm the stock path works before touching anything:

```
llama-server \
  -m MiniCPM5-2B-Q8_0.gguf \
  -md MiniCPM5-2B-DSpark.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 7 \
  -ngl 99 -ngld 99 -fa on \
  -c 8192 --jinja --port 8080
```

Note the aggregate acceptance rate llama.cpp reports at the end of a run —
this becomes the sanity check for Phase 1's per-block logging (averaged
logged acceptance should reproduce this number).

**Disable the draft's own early-stop: `--spec-draft-p-min 0`.** This is not
optional. `p_min` truncates a block *before the target ever sees it*, so
`accepted_len` would silently conflate "the target rejected" with "the draft
gave up" — two different signals. The harness pins it to 0, `draft_stop`
records which reason ended each block, and the schema checker raises if it
finds `p_min > 0` truncations in a trace being analysed.

**Pin decoding params for the whole study.** Temperature 0 / greedy, fixed
seed, fixed context length. Acceptance semantics change completely under
sampling (you are then matching against a draw, not an argmax, and
acceptance is stochastic even where the models agree). Record which
verification scheme the implementation actually uses — strict argmax match
vs. proper speculative rejection sampling — since it determines what a
mismatch means.

## Phase 1 — Instrumentation

- Locate the verify loop that drives `draft-dspark` in the llama.cpp
  source (search for `n_accept` / `n_drafted` / the generic speculative
  accept-reject logic — likely shared code path used by all `--spec-type`
  variants, so this instrumentation is reusable for DFlash/EAGLE3 later).
- Add a flag-gated trace (e.g. `--dspark-trace <path>.jsonl`) so normal
  runs are untouched.

### Run header (first line of each JSONL file)

Needed for comparability across runs; cheap and prevents orphaned logs.

```json
{
  "record": "header",
  "target_model": "MiniCPM5-2B-Q8_0.gguf",
  "target_sha": "…",
  "draft_model": "MiniCPM5-2B-DSpark.gguf",
  "draft_sha": "…",
  "spec_type": "draft-dspark",
  "block_size": 7,
  "verify_scheme": "argmax|rejection",
  "temp": 0.0,
  "seed": 1234,
  "llama_cpp_commit": "…"
}
```

### Per-block record

```json
{
  "record": "block",
  "block_idx": 42,
  "prompt_id": "twist_03",
  "tok_abs_start": 512,

  "draft_tokens": [1923, 44, 812, 91, 5, 3301, 78],
  "draft_top1_probs": [0.97, 0.94, 0.88, 0.51, 0.22, 0.83, 0.79],
  "draft_entropy": [0.14, 0.21, 0.39, 1.62, 2.41, 0.55, 0.61],

  "accepted_len": 3,
  "n_verified": 4,

  "target_top1_probs": [0.99, 0.91, 0.85, 0.42],
  "target_entropy": [0.06, 0.28, 0.44, 1.88],
  "target_prob_of_draft": [0.99, 0.91, 0.85, 0.03],
  "draft_rank_in_target": [1, 1, 1, 7],

  "mismatch_pos": 3,
  "draft_token_at_mismatch": 91,
  "target_token_at_mismatch": 604,

  "context_tail": "… decoded text of the preceding N accepted tokens …"
}
```

Field notes:

- **Target arrays are shorter than draft arrays.** They cover only
  positions actually verified: `n_verified = accepted_len + 1` when there
  is a mismatch, `block_size` when the whole block is accepted. Positions
  after the first mismatch are never evaluated by the target. This is the
  censoring described under Phase 3 — do not pad these arrays, the
  shortfall is information.
- `target_prob_of_draft` and `draft_rank_in_target` are the graded version
  of "how wrong was the draft". Both come from target logits already
  computed at verify time, so they cost nothing but a read. A rejected
  token the target had at rank 2 with p=0.3 is a near-miss synonym; one at
  rank 4000 is a different continuation entirely. This distinction is what
  makes the mismatch bucket a measure rather than a curiosity.
- `tok_abs_start` is the absolute token index in the generation, so blocks
  can be aligned back to character offsets in the output text — required
  for pointing at a specific punchline token, and for the later
  subject-bucketing pass.
- `context_tail` (suggest N=32 tokens, flag-tunable) is what makes
  retrospective subject bucketing possible without re-running. Store it on
  every block, not just mismatches — the non-mismatch blocks are the
  control set for any "mismatch clusters by subject" claim.

### Resolved: where the certainty signal comes from

DSpark exposes **its own per-position confidence scalar** — the value that
`--spec-draft-p-min` thresholds on, read from `llama_get_embeddings_nextn`.
This is the model's self-reported certainty and is logged as `draft_conf`.
The instrumentation reads it even when `p_min = 0`, so the signal is captured
on runs that disable the early-stop.

Prefer it over the token head. Measured on the first real run (n=198 blocks):

| signal | corr with `accepted_len` | range |
|---|---|---|
| `draft_conf` (native head) | **+0.798** | 0.71 – 0.999 |
| `draft_p` (token top-1)    | +0.751 | saturates at 1.000 |

The draft sampler runs `top_k = 10`, so `draft_p` and `draft_entropy` are
computed over a **top-10 truncated, renormalised** candidate set, not the full
softmax. That destroys resolution at the high-confidence end, which is exactly
where most positions sit. `draft_conf` has no such problem. Both are logged;
`draft_n_cand` records the truncation so the caveat travels with the data.

## Phase 2 — Test prompt bank

Fixed set, reused across every run for comparability:

- **Boring** — repetitive boilerplate (e.g. a templated log format repeated)
- **Twist** — a joke or narrative with a punchline/reversal
- **Technical** — dense, jargon-heavy factual text
- **Left-turn** — deliberately creative, unpredictable continuation

**"Boring" must mean predictable-to-the-draft, not repetitive-to-a-human.**
The smoke run inverted the spec's prediction: the boilerplate prompt accepted
2.85 and the left-turn prompt 5.05. The reason is visible in the mismatches —
the boilerplate's divergences land on *structure*: field names
(`' timestamp'`→`' log'`, `' status'`→`' latency'`, `'='`→`' worker'`) and
digits (`'02'`→`'20'`). A templated log line is adversarial for a 334 MB draft,
because reproducing it means knowing which field comes next and incrementing a
numeral. Meanwhile the "creative" prompt drew a formulaic continuation the
draft tracked easily.

Two consequences for the bank: build Boring from text that is *easy to
predict* (prose with high redundancy), not text that merely repeats; and keep
numerals and structured formats out of every category unless they are the
thing being tested, since they dominate the mismatch counts wherever they
appear. This is also the first evidence for the blind-spot thesis — the
mismatches cluster by content type, not uniformly.

**Matched-pair subset (Twist).** The between-category comparison is
confounded — the four categories differ in vocabulary rarity, register, and
domain all at once, so a shape difference is close to guaranteed and
carries little information. The Twist category is the only one that
supports a *localized* prediction, so build a paired subset: the same setup
text with (a) a predictable ending and (b) a reversal, matched for length
and vocabulary. The hypothesis becomes "divergence is higher at the
reversal token than at the matched predictable token, in the same context",
which is a paired test on a single token rather than an eyeball comparison
of two curves. Target N≥20 pairs.

## Phase 3 — Analysis & outputs

### A. Divergence/convergence graph

Small standalone Python script (matplotlib) reading the JSONL log:

- X-axis: block index (time through generation)
- Two lines/subplots: `accepted_len` (0–7, convergence proxy) and mean
  `draft_top1_prob` across the block (certainty proxy)
- Mark/shade blocks where `accepted_len < 7` as divergence events
- Overlay by `prompt_id` category to compare shape across Boring / Twist /
  Technical / Left-turn

### B. Position-conditioned hazard

`accepted_len` is right-censored, so a mean over "the block" mixes
evaluated and unevaluated positions and is not comparable between a block
that accepted 7 and one that accepted 2. Compute instead the per-position
hazard: P(reject at position *k* | survived to *k*). DSpark's chained
low-rank bias should produce a rising baseline hazard with *k* — establish
that curve on the Boring prompts first, so content effects are measured as
departures from it rather than confounded with it.

### C. Entropy control

Regress `accepted_len` (and per-position rejection) on target entropy at
the same positions. If divergence is just a 3-bit readout of target
entropy, the draft head adds nothing and Phase 4 has no foundation. **The
residual is the quantity of interest** — blocks where the draft diverged
more than target entropy predicts are where the cheap predictor
specifically failed, as opposed to where the text was simply hard.

### D. Mismatch bucket → blind-spot map

- Running `Counter` of `(draft_token_at_mismatch, target_token_at_mismatch)`
  pairs across all logged divergences; print top-N at end of run.
- Weight or split by `target_prob_of_draft` so near-misses don't drown out
  real failures.
- Keep `context_tail` for each mismatch for qualitative spot-checking
  ("what was it expecting here").
- **Deferred, enabled by this log:** pass `context_tail` for mismatch and
  non-mismatch blocks through a separate model to tag subject / register /
  syntactic role, then test whether mismatch rate varies by tag against the
  non-mismatch base rate. No re-run needed if `context_tail` is logged on
  every block from the start.

## Success criteria (gate before Phase 4)

- **Correctness:** instrumentation doesn't break baseline decoding; logged
  average `accepted_len` matches llama.cpp's own reported acceptance rate
  from Phase 0.
- **Signal, pre-registered:** on the Twist matched pairs, divergence at the
  reversal token exceeds divergence at the matched predictable token —
  paired sign test, N≥20, committed to before looking at the data.
- **Non-redundancy:** the entropy regression (C) leaves meaningful
  residual variance. If `accepted_len` is fully explained by target
  entropy, the surprise-proxy line of work stops here.

Note that failing the surprise gate does not kill the blind-spot map (D),
which stands on its own value — it just means Phase 4's affect framing
doesn't get built on this signal.

## Open questions

- Exact source location of the `draft-dspark` accept/reject loop
- Whether within-block confidence trajectory (position-to-position, given
  DSpark's chained low-rank bias) is a richer signal than the simple
  accepted-length break point — logging both, compared in Phase 3B
- Whether divergence is better read as the target's surprise or as *the
  cheap predictor's* prediction error. These are different claims; the
  second is what is actually measured. Decide which one Phase 4 asserts
  before writing it up.
- How divergence/mismatch-rate ultimately maps onto certainty / surprise /
  novelty / boredom — deliberately deferred until Phase 3 data exists
