# CoT Stepwise Analysis

**Can you predict, *while a model is still reasoning*, which steps will actually matter?**

This repo investigates online detection of *thought anchors* — reasoning steps in a chain-of-thought trace that, if replaced with something semantically different, would most shift the model's final answer distribution. All prior anchor-attribution methods are retrodictive (they require the full trace or hundreds of rollouts). The primary contribution claimed here is the first *causal*, generation-time predictor of counterfactual anchor influence.

Target model: **DeepSeek-R1-Distill-Qwen-1.5B**. Ground-truth labels: counterfactual resampling at R=100 (Bogdan et al., arXiv:2506.19143). Validation threshold: Spearman ρ > 0.22 vs. labels (the receiver-head baseline from the original thought-anchors paper).

---

## What's being tested

The central bet is that anchor steps leave a local dynamical signature in the model's forward-pass internals that predicts their future counterfactual importance — even before the model finishes reasoning. Two approaches are developed in parallel:

### 1. Propagation / Attribution (primary)

A single forward pass over the base trace yields mechanistic signals scored at the span level:

| Signal | What it measures | Anchor hypothesis |
|--------|-----------------|-------------------|
| **P_i** (Katz routing) | How much span *i* routes information downstream via attention, propagated multi-hop | *Premise anchors*: steps whose content reaches the answer |
| **G_i** (MLP write magnitude) | How much in-place computation span *i* performs | *Decision anchors*: steps that do the load-bearing operation |
| **D_i** (gradient-scaled outflow) | How much of downstream computation responds to *i*'s specific contribution direction | Intersection: routes *and* drives computation |
| **P'_i** (crude outflow) | D_i's cheaper approximation: routing × total downstream G | Baseline for D_i |

Each signal is a cheap forward-pass quantity. Ground truth A_i is expensive (100 rollouts/span). The hypothesis is that the cheap signals correlate with A_i above a useful threshold.

The two-axis bet: (i) does *any* marginal beat 0.22? (ii) does the *computation axis* (G_i, D_i) beat or add over routing (P_i) alone? A routing-only win is an improved routing result, not evidence that computation carries orthogonal signal.

### 2. Momentum / Physics (V3)

Models the CoT trace as a particle moving through representation space:

- **State**: h_t = hidden state (residual stream) at step t
- **Velocity**: v_t = Δh_t (smoothed via Savitzky–Golay)
- **Metric**: G = Σ⁻¹ (whitened velocity covariance, constant)
- **Potential**: V(h) = -log P(next realized token | h) — surprisal of the generated continuation
- **Energy**: E_t = ½ v⊤Gv + V

The mechanical intuition: a model "coasting" under inertia + the token-probability field should show flat energy (dE/dt ≈ 0). An anchor step is a non-conservative injection — energy changes beyond what the model's own dynamics would predict. Three validity tests determine which physical model (conservative, dissipative, rotational) the dynamics actually support before any detector is built.

---

## Intuition

A reasoning step is important because its content (a) *reaches* the answer-producing computation via attention and (b) the answer actually *depends* on it. Forward-pass signals measure *capacity* for importance; resampling measures *realized* importance. The gap — redundancy, robustness, nonlinearity — caps achievable correlation below 1 and is unknowable from the forward pass alone.

Two anchor species the signals are designed to distinguish:
- **Premise anchors**: establish a fact that propagates to the answer (high P_i, lower G_i)
- **Decision anchors**: perform the load-bearing computation that resolves the question (high G_i, lower P_i)

For the momentum approach, the intuition maps: a coasting step rides existing momentum along the probability field's gradient; an anchor step either injects new energy (reinforcing a direction — Branch A/B) or bends the trajectory without speeding it up (a rotational force, invisible to energy — Branch C).

---

## Methodology

### Ground-truth labels

For each span i in a base trace, the label A_i is:

```
A_i = TV( p̂(answer | real_i),  p̂(answer | diff-resample_i) )
```

— total-variation distance between answer distributions when the span is left intact vs. replaced with a semantically different resample from the model. Large A_i → anchor. Generated at R=100 rollouts/span via the resample-and-filter protocol of Bogdan et al., with a Sentence-BERT cosine-median semantic filter to ensure resamples are genuinely different in meaning.

### Span definition

Two-stage, boundary-respecting segmentation:
1. **Parse clauses**: split at sentence terminators, newlines, comma boundaries, and discourse connectives (so, therefore, thus, then, because, …)
2. **Window within clause**: subdivide each clause into ≤ τ token windows; never cross a clause boundary

τ is swept in Phase 0 over {10, 15, 25} tokens and held fixed thereafter. The resample-and-filter method was designed at sentence level; Phase 0 confirms it transfers to sub-sentence clause windows.

### Attribution signals (propagation approach)

Attention edges are computed as *delivered force*, not raw attention weight:

```
F^ℓ_{s→t} = W_O^{h,ℓ} (α^{ℓ,h}_{t,s} · v^{kv(h)}_s)
```

This folds in the output projection and the value-vector norm, demoting attention sinks (high weight, small value contribution). The target model is pre-norm with GQA — exact additivity holds with no linearization needed, and KV heads are correctly shared across query heads.

Multi-hop routing P_i propagates via Katz centrality on the causal DAG:

```
P_s = Σ_{t>s} E_{s→t} (1 + λ P_t)
```

The gradient signal D_i differentiates the downstream MLP write magnitude w.r.t. the residual, crediting only the component that responds to span i's specific inflow direction. The integrated-gradients variant D_i^{IG} is the faithful fallback when activation saturation makes the first-order approximation unreliable.

### Validity tests (physics approach)

Before any detector is tuned, three tests decide which physical model applies:
- **Test 1** (conservativeness): is dE/dt ≈ 0 on coasting steps? Pass → Branch A (natural Lagrangian). Structured fail → Branch B (dissipative, fit drag C). Unstructured fail → Branch F (no physics survives).
- **Test 2** (curl): is the force field F = -∇V curl-free? Fail → Branch C (rotational, canonical momentum p_can = Gv + A).
- **Test 3** (reparameterization): does the anchor score change materially under arc-length time? Fail → Branch D (Maupertuis/Jacobi, clock-free).

The detector is built from the test outcomes, not assumed up front.

### Dataset curriculum

| Phase | Dataset | Purpose |
|-------|---------|---------|
| 0 | bAbI | Plumbing smoke tests, parameter localization (τ, K, band B); free supporting-fact labels |
| 1 | Synthetic state-tracking / bAbI-derived | Construct validity: labeled supporting facts rank above distractors |
| 2 | BBH Tracking Shuffled Objects | Sequential state updates under swaps |
| 3 | CLUTRR | Compositional kinship reasoning, graded hop count |
| 4 | SCONE | Long-horizon instruction execution |
| Hold-out | GSM8K | OOD generalization test (math) |

---

## Execution

### Tiered, fail-cheap structure

The project runs in escalating tiers. Labeling at R=100 is the dominant compute cost (~10^9.5 tokens over 4–5 days on 8×A100); all cheaper checks gate it:

**Tier 0 — pre-labeling checks** (no rollouts):
1. Reuse check: if Bogdan et al.'s released labels cover the target model, ingest directly (zero labeling cost)
2. bAbI construct check: do signals rank annotated supporting facts above distractors?
3. Small-R pilot (R≈16): noisy early read on ρ; localizes τ, K, and band B

**Tier 1 — minimal marginals** (full R=100 labels, single-signal, no combiner):
- P_i, G_i, P'_i marginals against A_i
- Physics: Test 1 diagnostic selects Branch A or B; Test 3 checks clock artifact
- Baselines reproduced on the same traces: length-corrected kinematic signal, and a learned step-boundary classifier trained directly against A_i

**Tier 2 — first fork** (entered by Tier 1 outcome):
- *If both prongs clear*: add D_i (gradient-scaled), add linear combiner (ridge/PCR); coefficients are the finding
- *If computation axis doesn't clear*: rule out position confound, bad approximation (D_i^{IG}), wrong reach (chained variant as rescue) before declaring the axis inert

**Tier 3 — reach and ceiling**: chained variants P̃'_i, D̃_i with λ sweep; MLP nonlinear ceiling

**Tier 4 — causal validation** (only on a validated detector): noise-fill (Q2 sufficiency/necessity), forced anchoring, mechanistic head analysis

### Implementation scaffold (`Propogation_Analysis/ta/`)

```
ta/
  config/        run configs (yaml)
  core/
    modelio.py   model loading + architecture assertion
    traces.py    base CoT trace generation
    spans.py     clause-respecting segmentation
    signals/     force, Katz, MLP write, gradient attribution, online
    labels.py    resample-and-filter gold labels
    combiners.py ridge / PCR / MLP
    metrics.py   Spearman, bootstrap CI, decision logic
  infra/
    schema.py    typed artifact records
    store.py     atomic keyed read/write
    queue.py     file-based work queue (claim/complete sentinels)
    seeds.py     deterministic seed derivation
    splits.py    grouped-by-problem splits + leak assertions
    manifest.py  run manifest (git sha, config hash, model facts)
  smoke/         correctness tests — block phase launch if any fail
  run/           tmux launch-and-forget cluster runner
  data/          raw datasets (bAbI, BBH, CLUTRR, SCONE, GSM8K)
  artifacts/     all outputs, sharded by phase/dataset/problem_id (gitignored)
```

Key correctness invariants:
- Token ids are the source of truth; text is never re-tokenized for alignment
- GQA head mapping is asserted, not assumed (kv(h) = h // group_size)
- Splits are at problem granularity; within-trace spans are correlated and never split across train/test
- Raw rollouts are retained in full so A_i can be recomputed without re-rolling
- Every dropped span (pool unfilled, n_s=0, answer region) is flagged, never silently removed

### Physics implementation (`Momentum_1/`, `Momentum_2/`, `Propogation_Analysis/`)

Build order:
1. `momentum.py` against synthetic logits (no model needed) — validates geometry math
2. `model.py` + single-trace alignment check — validates tokenizer and streaming
3. `evaluate.py` against mock labels — validates stats harness
4. `generate.py` + Phase 0 mini-run (10–20 traces) — calibrates τ and R before gold run
5. Gate: plumbing sane, signal direction positive, baseline posture acceptable
6. Gold run — only after gate passes

---

## Results

*In progress. Results to be recorded here after Tier 1 runs.*

**Phase 0 / Tier 0 status**: Infrastructure scaffold built; bAbI dataset and DeepSeek-R1-Distill-Qwen-1.5B weights downloaded locally. Smoke-test harness wired. Awaiting Tier-0 pilot run.

**Decision threshold**: Spearman ρ > 0.22 vs. counterfactual-importance labels A_i (receiver-head baseline from Bogdan et al.).

| Signal | ρ vs. A_i | Notes |
|--------|-----------|-------|
| P_i (routing) | — | |
| G_i (computation) | — | |
| D_i (gradient-scaled) | — | |
| P'_i (crude outflow) | — | |
| Physics (Branch ?) | — | Branch determined by Test 1 |
| Ridge combiner | — | |

**Physics validity tests**:
| Test | Result | Branch |
|------|--------|--------|
| Test 1 (energy flatness) | — | — |
| Test 2 (curl) | — | — |
| Test 3 (reparameterization) | — | — |

---

## References

- Bogdan et al., *Thought Anchors: Which LLM Reasoning Steps Matter?* arXiv:2506.19143, 2025
- Zhou et al., *The Geometry of Reasoning: Flowing Logics in Representation Space.* arXiv:2510.09782, ICLR 2026
- Gjølbye et al., *Reasoning Models Don't Just Think Longer, They Move Differently.* arXiv:2605.15454, 2026
- Sun et al., *LLM Reasoning as Trajectories.* arXiv:2604.05655, ACL 2026
- Ferrando & Voita, *Information Flow Routes.* EMNLP 2024
- Marín, *Geometric Analysis of Reasoning Trajectories.* arXiv:2410.04415, 2024
