# Prompt bank generation brief

You are generating a prompt bank for an experiment that measures where a small
draft model diverges from a larger target model during speculative decoding.

Read all of this before starting. The constraints are not stylistic
preferences — several of them are the difference between a usable result and
an uninterpretable one.

## What the prompts are for

Each prompt is sent to a language model, which generates ~256 tokens of
continuation. While it generates, a small draft model tries to predict each
block of 7 tokens ahead of time, and we log every position where the draft
guessed wrong.

We then ask: **do those wrong guesses cluster by subject matter?**

That question is only answerable if subject is the thing that varies between
prompts and other properties are held still. Your job is mostly to hold
things still.

## Output format

One JSON object per line (JSONL). No markdown fences, no prose, no commentary
before or after — the file is consumed by a script. Every line must parse.

```json
{"prompt_id": "subj_medicine_03", "set": "A", "category": "subject", "subject": "medicine", "register": "plain-expository", "tags": [], "prompt": "Explain why a fever is not itself an illness but a response, and what that means for when it should and should not be treated."}
```

Required fields on every line:

| field | meaning |
|---|---|
| `prompt_id` | unique, lowercase, `[a-z0-9_]` only |
| `set` | `"A"`, `"B"`, or `"C"` — see below |
| `category` | `"subject"`, `"confound"`, or `"twist"` |
| `subject` | one of the subject list below, or `"none"` |
| `register` | `"plain-expository"` unless Set B says otherwise |
| `tags` | array of attribute tags (see Set B); `[]` for Set A |
| `prompt` | the prompt text itself |

Set C adds two more fields: `pair_id` and `arm`.

## Set A — the subject corpus (the main deliverable)

**15 subjects × 8 prompts = 120 lines.**

Subjects, use exactly these strings:

```
domestic_life   dialogue        software        medicine        law
finance         cooking         sport           history         philosophy
poetry          travel          news            correspondence  academic
```

Every Set A prompt must obey all of the following. This is the part that
matters most:

1. **Same register throughout: plain expository prose.** The continuation
   should read as ordinary connected sentences regardless of subject. Do not
   let the subject drag the style with it — a `law` prompt must not invite
   contract formatting, a `cooking` prompt must not invite a numbered recipe,
   a `poetry` prompt must not invite line breaks. Ask for *prose about* the
   subject.
2. **No numerals, and no invitation to produce them.** No dates, quantities,
   prices, measurements, statistics, years, counts, or version numbers.
   Digits swamp the mismatch counts and would masquerade as a subject effect.
   This is the single most common way to ruin this set.
3. **No proper nouns where avoidable**, and never ask for a named person,
   company, place, or product. Write "a coastal town", not a real one.
4. **No lists, headings, bullets, code, tables, or markup of any kind**, and
   nothing that would tempt the model to produce them.
5. **Length: 15–30 words per prompt**, and keep the spread tight across
   subjects. A subject whose prompts are systematically longer is a confound.
6. **Open enough to sustain ~256 tokens** of continuation without the model
   running out and stopping early. Avoid yes/no questions and anything
   answerable in a sentence.
7. **Vary within a subject.** The eight prompts for `medicine` should not be
   eight rephrasings of one question — spread them across different corners of
   the subject, while keeping register and length matched.

Good Set A prompt:

> Explain why a fever is not itself an illness but a response, and what that means for when it should and should not be treated.

Bad Set A prompts, and why:

> List the top 5 causes of fever in adults over 65.
> — numerals, a list, and it will stop early

> Describe how Pfizer developed its vaccine in 2020.
> — proper nouns and a date

> Write a poem about rain.
> — `poetry` as a subject still takes prose: ask *about* poetry, e.g.
>   "Explain what a poem does that a paragraph saying the same thing cannot."

## Set B — the confound probes

**5 attributes × 6 prompts = 30 lines.**

These deliberately violate Set A's constraints, one attribute at a time, so
the cost of each attribute can be measured and subtracted from the subject
effects. Keep the subject mix inside each attribute broad and ordinary — the
attribute is the variable here, not the topic.

| `tags` value | what the prompt should elicit |
|---|---|
| `["numerals"]` | dense quantities, dates, measurements, running counts |
| `["proper_nouns"]` | many named people, places, organisations, products |
| `["structured"]` | output with clear formatting: lists, steps, fields |
| `["technical_jargon"]` | dense domain terminology, plain prose otherwise |
| `["dialogue"]` | back-and-forth speech with attribution |

Set `subject` to `"none"`, `register` to `"plain-expository"` except for
`dialogue` (use `"dialogue"`) and `structured` (use `"structured"`).
Length range is the same as Set A.

## Set C — the twist matched pairs

**24 pairs = 48 lines.** These are the hardest to write well. Take more care
here than anywhere else; if you cannot make a pair genuinely matched, write a
different pair rather than a sloppy one.

Each pair shares an **identical setup**, word for word, and differs only in a
short final clause that leaves the prompt mid-sentence. Both arms must end at
a point where the model will immediately continue the sentence.

- **arm `"flat"`** ends on a cue that makes the next several words nearly
  forced — the sentence is running down a groove.
- **arm `"turn"`** ends on a cue that breaks the pattern, so the next several
  words are not determined by what came before.

Both arms must be **within three words of the same length**, share as much
vocabulary as possible, and differ only in that final cue. The whole point is
that anything other than the expectation differs as little as possible.

```json
{"prompt_id": "twist_07_flat", "set": "C", "category": "twist", "subject": "none", "register": "plain-expository", "tags": [], "pair_id": "twist_07", "arm": "flat", "prompt": "Every morning for a year he took the same seat by the window, opened the same book, and, just as he always did, he"}
{"prompt_id": "twist_07_turn", "set": "C", "category": "twist", "subject": "none", "register": "plain-expository", "tags": [], "pair_id": "twist_07", "arm": "turn", "prompt": "Every morning for a year he took the same seat by the window, opened the same book, and then, for the first time, he"}
```

Same rules as Set A otherwise: no numerals, no proper nouns, no formatting,
prose only. Setup length 20–40 words.

Note on what this measures: the `turn` arm is genuinely less constrained, so
the target model's own uncertainty will also be higher there. That is expected
and handled downstream by regressing target entropy out — what we are looking
for is divergence *beyond* what the target's own uncertainty explains. You do
not need to compensate for it. Just make the pairs as tightly matched as you
can on everything except the expectation.

## Totals

| set | lines |
|---|---|
| A — subjects | 120 |
| B — confounds | 30 |
| C — twist pairs | 48 |
| **total** | **198** |

## Before you finish

Check every one of these:

- [ ] Every line is valid JSON on a single line, with all required fields
- [ ] Every `prompt_id` is unique
- [ ] Set A contains no digits at all — search for them
- [ ] Set A has exactly 8 prompts for each of the 15 subjects
- [ ] Set C pairs share an identical setup and are length-matched within 3 words
- [ ] Every `pair_id` in Set C appears exactly twice, once per arm
- [ ] No prompt asks for a list, table, code, or heading outside Set B
- [ ] Nothing is wrapped in markdown fences

Then validate mechanically:

```
python3 harness/validate_bank.py <yourfile>.jsonl
```

Fix anything it reports and run it again until it passes.
