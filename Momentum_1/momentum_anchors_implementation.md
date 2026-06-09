# How to Use This Document

Read Sections <a href="#sec:arch" data-reference-type="ref" data-reference="sec:arch">2</a>–<a href="#sec:knobs" data-reference-type="ref" data-reference="sec:knobs">6</a> (architecture, the memory decision, schemas, interfaces, knobs) *before writing any code* — they are the holistic design layer and they are coupled, so deciding them piecemeal causes rework. Then follow Section <a href="#sec:buildorder" data-reference-type="ref" data-reference="sec:buildorder">7</a> (build order) literally; it is sequenced so that the cheapest checks run first and the single expensive step (gold labeling) sits behind a gate (Section <a href="#sec:gate" data-reference-type="ref" data-reference="sec:gate">8</a>). Section <a href="#sec:deferred" data-reference-type="ref" data-reference="sec:deferred">9</a> lists the pieces to stub now and implement only if a test triggers them. Section <a href="#sec:checklist" data-reference-type="ref" data-reference="sec:checklist">11</a> is a flat checklist to track against.

# Repository Architecture

Six modules, one directional data flow. Each module’s output is another’s input through a file contract (Section <a href="#sec:schemas" data-reference-type="ref" data-reference="sec:schemas">4</a>), never through shared memory, so any stage can be re-run from stored artifacts without recomputing upstream.

<div class="center">

| **Module**    | **Responsibility**                       | **Consumes / Produces**                  |
|:--------------|:-----------------------------------------|:-----------------------------------------|
| `config.py`   | all knobs, paths, model id, seeds        | — / config object                        |
| `segment.py`  | two-stage boundary-respecting spans      | trace text / span table                  |
| `generate.py` | base traces + resample-and-filter labels | problems / labeled trace store           |
| `model.py`    | teacher-forced logits, streaming         | stored token ids / per-step logit access |
| `momentum.py` | the geometry: $F,R,C$ per span           | logits + spans / score table             |
| `evaluate.py` | partial Spearman, bootstrap, baselines   | scores + labels / results                |
| `tests/`      | synthetic-logit geometry checks          | — / pass/fail                            |

</div>

Data flow: `generate` (uses `segment` internally) $\rightarrow$ labeled trace store $\rightarrow$ `model` re-forwards stored ids $\rightarrow$ `momentum` scores $\rightarrow$ `evaluate` correlates scores against stored labels. `segment` is also called standalone by `generate`; it is factored out because both generation and any re-segmentation must use the *identical* boundary logic.

<div class="notebox">

**Single source of truth for boundaries and the content mask.** The clause/sentence parser (`segment`) and the content-token stop-set (`momentum`) both depend on the same notion of “structural punctuation.” Define that set *once* in `config.py` (as token ids resolved against the actual tokenizer, not characters) and import it into both. If these drift apart, spans and scores silently disagree about what a boundary is.

</div>

# The One Architectural Decision: Forward-Pass Memory

Everything organizes around how logits are obtained and held, so decide this first. The full logit tensor for one trace is $[L, V]$ with $L\sim$ thousands and $V\sim 10^5$; materializing it in fp32 is gigabytes and is the most likely cause of an out-of-memory failure. We adopt **teacher-force-and-stream**:

1.  Run a single forward pass over the stored token-id sequence with caching, but **do not return all logits**. Instead, iterate positions in order and expose one logit vector at a time to the scorer.

2.  The scorer (Section <a href="#sec:momentum-mod" data-reference-type="ref" data-reference="sec:momentum-mod">5.5</a>) holds only: the previous *content*-position scaled-logit vector $\bm{g}_{k-1}$, the EMA state $\bar{\bm{v}}$, and per-position scalar outputs. Peak extra memory is $O(V)$, not $O(LV)$.

3.  This is byte-identical to the online form (during generation logits arrive one at a time), so offline scoring and online inference share one code path — a property to preserve, not an accident.

<div class="notebox">

**Why not chunk-and-cache logits to disk.** Writing $[L,V]$ logits per trace to disk for later scoring is simpler to reason about but multiplies storage by $V$ and breaks the offline/online equivalence. We store *token ids* (tiny) and re-forward when scoring; re-forwarding one trace is one cheap pass and avoids the storage blowup. The only quantity worth caching is the small per-span score table, not the logits.

</div>

#### Practical consequence for `model.py`.

Its public surface is an iterator, not a tensor return. Conceptually: `stream_logits(token_ids) -> yields (position, scaled_logit_vector)`, with temperature scaling applied inside. The scorer consumes the stream; nothing assembles the full matrix.

# Data Contracts (Schemas)

All inter-module artifacts are on disk with explicit schemas. Token ids are the source of truth; text and character offsets are derived. Use a columnar/record format that stores integer arrays natively (e.g. Parquet or JSONL-with-arrays); the field semantics below are what matter, not the container.

## Trace record (produced by `generate.py`)

One record per base trace.

<div class="center">

| **Field**         | **Type** | **Meaning**                                 |
|:------------------|:---------|:--------------------------------------------|
| `trace_id`        | str      | unique key                                  |
| `problem_id`      | str      | source problem (for trace-level grouping)   |
| `token_ids`       | int\[\]  | the generated sequence; **source of truth** |
| `text`            | str      | detokenization of `token_ids` (derived)     |
| `temperature`     | float    | generation $T$ (required for the metric)    |
| `think_start_tok` | int      | index of first token inside `<think>`       |
| `think_end_tok`   | int      | index of first token after `</think>`       |
| `is_correct`      | bool     | did this trace reach the verified answer    |
| `model_id`        | str      | exact model+revision used                   |

</div>

## Span record (produced by `segment.py`, stored with the trace)

One record per span; spans tile the think region.

<div class="center">

| **Field**    | **Type** | **Meaning**                                         |
|:-------------|:---------|:----------------------------------------------------|
| `trace_id`   | str      | foreign key                                         |
| `span_id`    | int      | ordinal within trace (also the position control)    |
| `tok_start`  | int      | half-open start, token-id index                     |
| `tok_end`    | int      | half-open end, token-id index                       |
| `char_start` | int      | derived, for cross-check only                       |
| `char_end`   | int      | derived, for cross-check only                       |
| `unit_id`    | int      | which parsed sentence/clause this window belongs to |
| `n_tokens`   | int      | `tok_end - tok_start`                               |

</div>

## Label record (produced by `generate.py`)

One record per span; the ground truth.

<div class="center">

| **Field**         | **Type** | **Meaning**                                        |
|:------------------|:---------|:---------------------------------------------------|
| `trace_id`        | str      | foreign key                                        |
| `span_id`         | int      | foreign key                                        |
| `label`           | float    | divergence between pools (KL default; TV optional) |
| `label_kind`      | str      | `"kl"` or `"tv"`                                   |
| `n_real`          | int      | usable same-meaning continuations                  |
| `n_diff`          | int      | usable different-meaning continuations             |
| `pool_filled`     | bool     | did both pools reach $R$ (else span dropped)       |
| `median_sim_used` | float    | the dataset-median cosine threshold applied        |

</div>

## Score record (produced by `momentum.py`)

One record per span; the predictor side. Computed both for the content-only default and the mixed-EMA ablation, distinguished by `variant`.

<div class="center">

| **Field**     | **Type** | **Meaning**                                                |
|:--------------|:---------|:-----------------------------------------------------------|
| `trace_id`    | str      | foreign key                                                |
| `span_id`     | int      | foreign key                                                |
| `variant`     | str      | `"content_only"` (default) or `"mixed"`                    |
| `force`       | float    | $\bar F_s$, mean Fisher force over scored steps            |
| `redirection` | float    | $R_s\in[0,1]$, turning energy fraction                     |
| `sign`        | float    | $C_s$, energy-weighted projection ($>0$ accel, $<0$ brake) |
| `surprise`    | float    | pooled $-\log p(x)$ baseline                               |
| `n_scored`    | int      | content steps contributing ($n_s$)                         |
| `beta`        | float    | EMA decay used                                             |
| `in_think`    | bool     | span lies in reasoning region (not answer emission)        |

</div>

<div class="notebox">

**Join discipline.** Every downstream join is on `(trace_id, span_id)` against the *stored* span table — never by re-segmenting text. Before any correlation, assert that span counts per trace match across the span, label, and score tables; a mismatch is a bug, not a result.

</div>

# Module Interfaces

For each module: its job, the functions to write (signatures as intent, not code), and the non-obvious correctness requirements. “Write `f(a) -> b`” means implement a function with that input/output contract.

## `config.py`

Holds every knob (Section <a href="#sec:knobs" data-reference-type="ref" data-reference="sec:knobs">6</a>) and all paths in one place; no logic. Critically, it resolves and exposes the **structural-token id set** (the shared boundary/stop-set definition) against the loaded tokenizer, so `segment` and `momentum` import the same object. Also fixes seeds.

## `segment.py`

Job: turn a trace’s token ids into the span table via the two-stage rule.

- `parse_units(token_ids, think_range) -> list[unit_span]`: stage 1, split into sentences/clauses at structural tokens (from `config`). Operates on token ids; punctuation is identified by id, not by regex over text.

- `window_units(unit_spans, t, t_min) -> list[span]`: stage 2, sub-split any unit longer than $t$ into $\le t$-token windows entirely inside the unit; merge a trailing fragment $< t_{\min}$ into the previous window of the same unit.

- Correctness: windows never cross a unit boundary; spans tile the think region with no gaps or overlaps; every span carries its `unit_id`.

## `generate.py`

Job: produce base traces and the resample-and-filter labels. The expensive module; built only after the Phase 0 gate logic is in place. Sub-steps:

- `generate_base_traces(problems) -> trace records`: sample one CoT per problem at temperature $T$; store token ids, think range, correctness. Verify correctness against the task’s gold answer.

- `resample_slot(trace, span, R_target, oversample_cap) -> continuations`: for the span’s slot, sample continuations *from the token just before `tok_start`* (regenerating the slot and everything after), up to `oversample_cap`, batched across GPUs. Each continuation yields a regenerated slot text and a final answer.

- `embed_and_split(orig_slot, continuations, median_sim) -> (real_pool, diff_pool)`: embed slots with the sentence embedder; same-meaning if cosine to original $>$ `median_sim`, different if $<$. (See the two-phase median in Section <a href="#sec:gate-seq" data-reference-type="ref" data-reference="sec:gate-seq">8.1</a>.)

- `label_from_pools(real_pool, diff_pool, kind) -> label`: aggregate each pool’s final-answer distribution; label is KL (default) or TV between them. Set `pool_filled=False` and drop the span if either pool $< R$.

<div class="notebox">

**Oversampling and pool-fill policy (must be explicit).** The same/different split is post hoc, so some slots yield a lopsided split (a slot the model almost always regenerates near-identically gives few below-median samples). Policy: sample in batches up to `oversample_cap` total; stop early once both pools reach $R$; if `oversample_cap` is hit with either pool $< R$, mark `pool_filled=False` and exclude the span from analysis (recorded, not silently dropped). Report the drop rate — a high rate at a given $t$ is itself information about granularity.

</div>

## `model.py`

Job: deterministic teacher-forced logit access, streaming (Section <a href="#sec:memory" data-reference-type="ref" data-reference="sec:memory">3</a>).

- `load_model(model_id) -> (model, tokenizer)`: fixed revision, eval mode, fixed dtype.

- `stream_logits(token_ids, T) -> iterator of (pos, scaled_logit_vec)`: one forward pass; yields scaled logits ($\bm{g}=\bm{\ell}/T$) one position at a time; never returns the full tensor. Casts to fp32 for the vector handed out (the covariance reduction needs it).

- Correctness: indexing convention is fixed — `logits[i]` predicts token `i+1`; teacher forcing must reproduce the *stored* ids exactly (no re-tokenization of text). Assert the input ids round-trip.

## `momentum.py`

Job: the geometry. Consumes the logit stream and the span table; produces the score table. This is the module to build *first* (against synthetic logits), because it has no model or data dependency beyond the stream interface.

- `fisher_cov(p, u, w) -> scalar`: the exact categorical Fisher inner product $\sum p u w - (\sum p u)(\sum p w)$, fp32.

- `is_content(token_id) -> bool`: true unless the id is in the structural/special stop-set from `config`.

- `score_stream(logit_stream, is_content, beta, variant) -> per-position (F, perp, tot, cproj)`: the streaming loop. `content_only` variant (default): advance the trajectory only across content positions, EMA updates only on content steps, velocity is content-to-content. `mixed` variant (ablation): advance through every token, score only content. Base point is the previous content distribution $p_{k-1}$; force is $\sqrt{\mathrm{cov}(d,d)}$ with $d=v-\bar v$.

- `pool_to_spans(per_position, spans, eps) -> score records`: $\bar F_s$ = mean force over content steps in the span; $R_s$ = $\sum\text{perp}/\sum\text{tot}$; $C_s$ = energy-weighted mean of `cproj`. Emit NaN and drop if $n_s=0$.

- Correctness: EMA warmup means the first span per trace is unreliable — flag it for dropping in `evaluate`, do not special-case it here.

## `evaluate.py`

Job: turn scores + labels into the headline numbers and the test verdicts.

- `partial_spearman(pred, label, covars) -> rho`: rank-transform all; residualize `label` and `pred` on {span index, span length} by OLS; Pearson-correlate residuals.

- `trace_bootstrap(table, statistic_fn, n=1000) -> (point, ci_low, ci_high)`: resample *traces* with replacement (never spans); recompute the statistic each draw; report $2.5/97.5$ percentiles.

- `run_tests(scores, labels) -> verdicts`: compute Test 1–6 quantities (redirection partial-$\rho$, sign/taxonomy enrichment, boundary-clustering of force, transfer, blind-spot fraction, $\beta$ curve) and emit the pre-registered verdict per the design document.

- Correctness: filter to reasoning-region spans (`in_think`), drop first-span-per-trace, drop `pool_filled=False`; always report against both `content_only` and `mixed` so the ablation gap is visible.

# Knobs and Default Values

Every free parameter, with a recommended default to start from. All live in `config.py`.

<div class="center">

| **Knob**                 | **Default**               | **Notes**                                                                                                                             |
|:-------------------------|:--------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| $\beta$ (EMA decay)      | 0.97                      | memory $\approx 1/(1-\beta)$ *content* tokens; swept in Test 6                                                                        |
| $t$ (window budget)      | 15 tokens                 | *budget-dominating*; calibrate at Phase 0 over $\{10,15,25\}$ by label stability / drop rate                                          |
| $t_{\min}$ (merge floor) | 5 tokens                  | trailing fragments below this merge left                                                                                              |
| motion threshold         | small $>0$                | gate $C_s$ where $\langle v,v\rangle$ is negligible; calibrate in Phase 0                                                             |
| $\varepsilon$            | $10^{-8}$                 | denominator floor                                                                                                                     |
| $R$ (per pool)           | 100                       | *budget-dominating*; “gold” level, but confirm sufficiency at Phase 0                                                                 |
| oversample cap           | $\sim$<!-- -->8$\times R$ | ceiling on total rollouts per slot before giving up                                                                                   |
| similarity threshold     | dataset median            | parameter-free; computed in two phases (Sec. <a href="#sec:gate-seq" data-reference-type="ref" data-reference="sec:gate-seq">8.1</a>) |
| embedder                 | Sentence-BERT             | or a current sentence embedder; fixed per study                                                                                       |
| label kind               | KL                        | TV optional; record which                                                                                                             |
| temperature $T$          | match gen. run            | the metric must use the sampling distribution                                                                                         |
| answer-region cutoff     | after `</think>`          | token-id test, not heuristic                                                                                                          |
| bootstrap draws          | 1000                      | trace-level                                                                                                                           |
| dtype (forward)          | bf16/fp16                 | covariances always fp32                                                                                                               |

</div>

<div class="notebox">

**$t$ and $R$ are budget-dominating knobs — calibrate them at Phase 0, do not fix them up front.** Most knobs above are set once and forgotten. Two are different: the window budget $t$ and the per-pool rollout count $R$ together determine the dominant compute cost of the whole study, because total rollouts scale roughly as (number of spans) $\times$ (rollouts per span) — smaller $t$ means more spans, larger $R$ means more rollouts each. The defaults ($t=15$, $R=100$) are starting points chosen to match the sources, not validated optima. Treat them as *Phase-0-calibrated*: the build order (Step 4 and the gate) deliberately measures label stability and pool-fill drop rate across $t\in\{10,15,25\}$ on the tiny set *before* the gold run, so you pick the granularity that localizes anchors without an unacceptable drop rate and confirm $R$ is large enough that labels are stable, before paying for it at full scale. Committing to $t$ and $R$ up front is the most likely way to either overspend or generate a gold dataset at the wrong granularity. This is a judgement call, flagged as such: the numbers are provisional, and Phase 0 exists in part to set them.

</div>

# Build Order

Build in this order. The principle: validate the math with no compute, then the plumbing with one cheap forward pass, then the full loop on a handful of traces, and only then pay for gold labeling. Each step has an explicit exit criterion.

#### Step 0 — `config.py` + skeleton.

Lay down the repo, the schemas as typed records, the config with defaults, and the shared structural-token set resolved against the tokenizer. *Exit:* importing any module and reading config works; the stop-set prints as actual token ids.

#### Step 1 — `momentum.py` against synthetic logits.

Build the geometry and the streaming loop with *no model and no data*. Feed hand-constructed logit streams from `tests/`. *Exit (unit tests pass):*

- `fisher_cov` matches a brute-force covariance on random vectors; the all-ones gauge vector gives $\approx 0$.

- A straight-line (constant-velocity) logit trajectory yields force $\approx 0$ (geodesic $\Rightarrow$ no force).

- A trajectory that reverses direction yields $C_s<0$; a sharp turn yields high $R_s$; a speed-up along a fixed direction yields $C_s>0$, low $R_s$.

- Content masking: inserting “formatting” positions changes nothing in the `content_only` variant.

This step is seconds of compute and catches every math/indexing error before a model is ever loaded.

#### Step 2 — `model.py` + single-trace alignment check.

Load the model; generate or hand-pick *one* trace; store its ids; re-forward and stream. Run `segment` on it and `momentum` over it. *Exit (the highest-value cheap check):*

- Stored token ids round-trip: detokenize$\rightarrow$retokenize equals the stored ids (or the stream is driven by ids directly, sidestepping this).

- The content stop-set actually fires on this tokenizer’s punctuation/newline tokens (inspect a few positions by hand).

- Spans tile the think region; `tok_start/tok_end` align to real clause boundaries on visual inspection.

- Scores are finite, the first span is flagged, the answer region is correctly excluded.

Cost: one forward pass. This is where tokenizer surprises surface; do not proceed until it is clean.

#### Step 3 — `evaluate.py` against mock labels.

Build the stats harness and test it on *synthetic* labels (e.g. a known correlation injected) so the partial-Spearman and trace-bootstrap are verified independently of real data. *Exit:* recovers a planted correlation; trace-bootstrap CIs widen correctly when traces (not spans) are resampled.

#### Step 4 — `generate.py` + Phase 0 mini-run.

Now build the expensive module, but run it small: a handful of traces ($\sim$<!-- -->10–20) on the easiest task family, full resample-and-filter at $R=100$. Run the whole pipeline end to end, and use this set to **calibrate the budget-dominating knobs**: re-segment at each $t\in\{10,15,25\}$ and compare label stability and pool-fill drop rate, and check that labels are stable at $R=100$ (i.e. that a smaller $R$ would not have sufficed, or that $100$ is not excessive). Freeze $t$ and $R$ at the values this sweep justifies before any gold run. *Exit:* the pipeline completes, emits a score+label table with sane pool-fill rates, and $t$/$R$ are chosen from data rather than assumed.

#### Step 5 — THE GATE (Section <a href="#sec:gate" data-reference-type="ref" data-reference="sec:gate">8</a>).

Look at the Phase 0 numbers before scaling. Decision point, not a build step.

#### Step 6 — Gold run.

Only if the gate passes: scale `generate` to the full dataset and curriculum, run the gold labeling, and produce the full results and Test 1–6 verdicts.

# The Phase 0 Gate

<div class="gatebox">

**Do not spend gold-level compute until Phase 0 clears this gate.** After Step 4, on the tiny labeled set, check:

1.  **Plumbing sanity:** joins line up, pool-fill rate is acceptable at the chosen $t$, no NaN floods, answer region excluded. **Also fix $t$ and $R$ here:** compare label stability and drop rate across $t\in\{10,15,25\}$ and confirm $R$ is large enough that the labels are stable; these values are then frozen for the gold run.

2.  **Direction of signal:** is the partial Spearman of $R_s$ (or $C_s$) with the label *positive and plausibly nonzero*, even if not yet significant at this tiny $n$? A near-zero or negative central estimate across the whole small set is a warning.

3.  **Baseline posture:** does the geometry at least track alongside the surprise baseline rather than being dominated by it?

**Pass** $\rightarrow$ Step 6 (gold). **Fail on plumbing** $\rightarrow$ fix and re-run Phase 0 (cheap). **Fail on signal with clean plumbing** $\rightarrow$ this is a real result: invoke the design document’s pre-registered responses (e.g. Test 3 boundary fix if force clusters at span starts; reconsider scale per the model-scale caveat) *before* paying for gold. Do not scale a flat signal hoping $n$ fixes it.

</div>

## Two-phase similarity median (sequencing constraint)

The “different-meaning” threshold is the dataset-median pairwise cosine similarity, so it *cannot be a constant known up front*. Sequence it: (phase A) generate base traces and all resample continuations, embedding and storing similarities but *not* yet splitting; (phase B) compute the global median over all collected similarities, then apply it to split every slot’s pool and compute labels. In the Phase 0 mini-run, compute a provisional median on the small set; recompute the real median on the gold set before gold labeling. Record `median_sim_used` per label so any later re-split is reproducible.

# Deferred-by-Design: Stub Now, Build If Triggered

Write these as named, documented stubs so the interfaces exist, but do not implement until their trigger fires. Implementing them up front is speculative work the Phase 0/test results may tell you to skip.

- **Boundary fix for the small-step assumption** (Test 3 trigger: force clusters at span-initial tokens): EMA reset/down-weight at parsed boundaries, or drop the first $k$ post-boundary content tokens. Stub: a no-op hook in the scoring loop at boundary positions.

- **Arc-length reparameterization** (Test 6 ablation): recompute with Fisher-arc-length time instead of the token clock. Stub: a clock-selection flag in the scorer.

- **Paraphrase perturbation route** (trigger: model too large / rollout budget too high for resample-and-filter): controlled meaning-changed rewrites in place of resampling. Stub: an alternative `make_alternative` behind the same interface `generate` already calls; must re-validate against a resample-and-filter subset before its labels are trusted.

# Cross-Cutting Correctness Requirements

A short list of invariants every module must respect, gathered so they are not buried:

- **Token ids are truth.** Never reconstruct alignment from text; derive text/offsets from ids, never the reverse.

- **Determinism.** Fixed model revision, fixed seeds, eval mode; teacher forcing must reproduce stored ids exactly.

- **fp32 covariances.** Regardless of forward dtype.

- **Trace is the unit.** Every split, bootstrap, and significance statement groups by trace, never by span.

- **Both variants always.** Score `content_only` and `mixed` every run; the gap is a reported diagnostic.

- **Record, don’t silently drop.** Dropped spans (pool unfilled, $n_s=0$, first-span warmup, answer region) are flagged in the tables, not removed upstream.

- **One boundary definition.** Parser and content mask import the same structural-token set.

# Flat Build Checklist

1.  `config.py`: knobs, paths, seeds, shared structural-token id set resolved against tokenizer.

2.  Schemas: trace, span, label, score records as typed structures with the join keys.

3.  `momentum.py`: `fisher_cov`, `is_content`, `score_stream` (both variants), `pool_to_spans`.

4.  `tests/`: synthetic-logit geometry suite; *all pass* before Step 2.

5.  `model.py`: `load_model`, streaming `stream_logits`; id round-trip assertion.

6.  Single-trace alignment check clean (stop-set fires, spans tile, scores finite).

7.  `segment.py`: `parse_units`, `window_units`; tiling/no-cross invariants.

8.  `evaluate.py`: `partial_spearman`, `trace_bootstrap`, `run_tests`; verified on planted correlation.

9.  `generate.py`: base traces, `resample_slot`, `embed_and_split` (two-phase median), `label_from_pools`, oversample/pool-fill policy.

10. Phase 0 mini-run end to end ($\sim$<!-- -->10–20 traces, $R=100$).

11. **GATE:** plumbing sane, signal direction positive, baseline posture acceptable.

12. Gold run + Test 1–6 verdicts (only past the gate).

13. Stubs in place (boundary fix, arc-length, paraphrase) — unbuilt until triggered.
