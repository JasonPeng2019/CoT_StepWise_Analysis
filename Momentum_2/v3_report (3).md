           V3: Reasoning as a Path of Least Action
                         An Inference-Time Thought-Anchor Detector
                           Built on the Momentum–Work Analogy

                   Working specification — conceptual base, validity tests,
                    branch-conditional models, and implementation plan

# Abstract
    The task of this work is online detection of thought anchors: predicting, during chain-of-
    thought generation and with only bounded lag, which reasoning steps will turn out to have high
    counterfactual influence on the final answer in the sense of Bogdan et al. (arXiv:2506.19143).
    All existing anchor-attribution methods are retrodictive — they require the full trace or many
    rollouts — so to our knowledge no causal, generation-time predictor of counterfactual anchor-
    influence exists by any method. That empty cell, not the physics, is the primary contribution
    claimed here. The method, “V3,” is one instantiation: treat the trace as the trajectory of a
    mechanical system and flag steps where momentum is sharply reinforced or redirected. V3 is
    built around an honest commitment — the mechanical analogy is admissible only if the dynam-
    ics support it, so the method is gated by three falsifiability tests and the downstream model
    is selected by the test outcomes rather than assumed. We separate the task from the method
    throughout, place both against the recent trajectory-geometry literature (§3), and organize
    the whole program as a tiered execution plan (§7): a minimal first run carrying only the
    non-negotiable safeguards, then small, outcome-driven increments — fixes if a tier fails, im-
    provements if it succeeds — until either every necessary fix has been tried and failed, or the
    best working version is reached. We state plainly that the load-bearing requirement is valida-
    tion against counterfactual-importance labels (not sentence-type, correctness, or difficulty), and
    give the conceptual base, the exact tests, the branch-conditional model for each outcome, and
    variable-level implementation details. Red boxes throughout give a strict account of what is
    principled, what is a modeling choice, what is numerically fragile, and what no implementation
    can rescue.

 Honesty preamble. Nothing here has been validated on a real model. The synthetic-data
 sanity checks confirm only that the arithmetic does what the equations say, not that the
 equations describe a real reasoning model. Two axes should be kept separate. On the dynamics
 axis, the most likely landing is the dissipative case (§5, Branch B), not the clean conservative
 one. On the outcome axis — does any version beat the baselines — the most likely result is
 still a negative: even the right branch may fail to track anchors above what trivial features
 capture (see §9). The two are compatible: “probably dissipative, and probably still does not
 beat baselines.” That negative is a result, not a failure, and the value of the design is that
 it reaches the verdict cheaply. Read every red box; the boxes are where the real uncertainty
 lives.

# Contents

# 1 Conceptual base                                                                                        3

## 1.1   What the detector is trying to predict . . . . . . . . . . . . . . . . . . . . . . . . . .    3
## 1.2   The mechanical analogy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    3
## 1.3   The defining commitment of V3: the potential . . . . . . . . . . . . . . . . . . . . .        3
## 1.4   The load-bearing reframe: impulse, not work . . . . . . . . . . . . . . . . . . . . . .       4
## 1.5   Task versus method, and the non-negotiable label . . . . . . . . . . . . . . . . . . . .      4
## 1.6   The anchor unit: clause-respecting token windows . . . . . . . . . . . . . . . . . . .        4

# 2 Dataset generation: token-window anchor labels                                                       5
## 2.1 Perturbed resampling labels . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      5
## 2.2 Semantic-difference adjudication: which resamples count . . . . . . . . . . . . . . . .          6
## 2.3 Trace storage with token alignment . . . . . . . . . . . . . . . . . . . . . . . . . . . .       7
## 2.4 The first-content-token probe decision . . . . . . . . . . . . . . . . . . . . . . . . . .       8
## 2.5 Curriculum and the constructive sanity check . . . . . . . . . . . . . . . . . . . . . .         8

# 3 Related work and the precise novelty claim                                                           8

# 4 The three validity tests                                                                             9

# 5 Branch-conditional models                                                                           10
## 5.1 Branch A — natural Lagrangian (all pass) . . . . . . . . . . . . . . . . . . . . . . . .        10
## 5.2 Branch B — dissipative, structured (Test 1 structured fail) . . . . . . . . . . . . . .         10
## 5.3 Branch C — gauge / rotational (Test 2 fail, Test 1 pass) . . . . . . . . . . . . . . . .        11
## 5.4 Branch D — Maupertuis / Jacobi (Test 3 fail, Test 1 pass) . . . . . . . . . . . . . .           11
## 5.5 Branch E — metriplectic / GENERIC (multiple structured fails) . . . . . . . . . . .             11
## 5.6 Branch F — unstructured collapse (Test 1 unstructured fail) . . . . . . . . . . . . .           12
## 5.7 Summary of where each analogy survives . . . . . . . . . . . . . . . . . . . . . . . .          12

# 6 Implementation: variable-level details                                                              12
## 6.1 Shared skeleton (all branches) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    12
## 6.2 The potential V and its gradient . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      13
## 6.3 The active subspace . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     13
## 6.4 Fisher / KL proxy (only if a branch needs a non-trivial metric) . . . . . . . . . . . .         13
## 6.5 Branch-specific objects . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   14
## 6.6 The coasting set — circularity, handled . . . . . . . . . . . . . . . . . . . . . . . . .       14

# 7 Tiered execution plan                                                                               14
## 7.1 Three buckets every feature falls into . . . . . . . . . . . . . . . . . . . . . . . . . . .    14
## 7.2 Tier 1 — the minimal run . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      15
## 7.3 The ladder . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    15
## 7.4 Termination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   16

# 8 What can and cannot be implemented now                                                              16

# 9 What the results would mean                                                                         17

# 1 Conceptual base

## 1.1 What the detector is trying to predict
A thought anchor is defined retrodictively: a reasoning step is an anchor because of its downstream
effect on the final-answer distribution, as measured by counterfactual resampling. At inference
time that downstream effect has not happened yet, so anchor-ness is not directly observable online.
The entire premise of this work is therefore a bet: that anchors carry a local dynamical signature
in the model’s representation trajectory that is predictive of the retrodictive label. We look for
a causal (past-and-present, possibly lagged) functional of the trajectory that correlates with the
future-defined anchor.

  This bet can simply be false. If anchor-ness is genuinely determined by what the model does
  after a step — not encoded in the local trajectory shape — then no causal functional can predict
  it above a low ceiling, and the ceiling is estimable (§9). We design to maximize detection if
  the signal exists; we cannot manufacture it if it does not.

## 1.2 The mechanical analogy
We model the trace as a particle moving through representation space. Let ht ∈ Rd be the model’s
hidden state (residual stream) at step t, and vt = ht − ht−1 the velocity. We endow the space with
a metric G (the mass tensor), a potential V (h), and the Lagrangian

                                                              ∂L
                           L = 21 v ⊤ G v − V (h),      p =      = G v.                           (1)
                                                              ∂v
This is a natural mechanical system in the sense of Arnold: momentum is the covector p = Gv,
force is −∇V , and (for a time-independent L) the energy E = 12 v ⊤ Gv + V is conserved along true
trajectories.

## 1.3 The defining commitment of V3: the potential
What makes this V3 rather than a generic momentum heuristic (“V2”) is the potential V . We fix

                             V (h) = − log Pθ (realized next step | h),                           (2)

the surprisal of the continuation that was actually generated, evaluated as a function of an arbitrary
spliced-in state h. Low potential means the model is rolling toward high-probability continuations;
high potential means it is paying probability cost. Coasting deduction is then motion under inertia
plus this potential; an anchor is a step the potential-plus-inertia dynamics cannot explain — a
non-conservative injection.

  Equation (2) ties V to the realized path, so it is a potential defined relative to the trajectory
  taken. A genuine physical potential is a field that exists independently of the particle. This is
  exactly why the curl test (§4) is non-negotiable: a path-relative “potential” has no guarantee
  of being curl-free, i.e. of being the gradient of any scalar at all. Surprisal is also only one
  defensible potential; a value- or reward-model potential would be different and possibly better,
  and is chosen here only because it is available at inference without a second model.

## 1.4 The load-bearing reframe: impulse, not work
The momentum–work analogy has a robust half and a fragile half, and the distinction organizes
the entire branch tree below.

  Impulse J = F dt = ∆p is a vector: the change in momentum. It is the robust primitive.
                  R

  Work W = F · v dt is a scalar: it is impulse projected onto the direction of motion, W = J · v̂.
                R

   It discards any force perpendicular to v.

Consequently a force that bends the trajectory without speeding it up has full impulse and zero
work. Every branch where the “momentum–work analogy” breaks is a branch where the anchor
force is partly perpendicular to v, and work goes blind to exactly that component. The detector
is therefore built on the impulse residual; work is reported only as a derived, lossy
scalar where it is meaningful.

## 1.5 Task versus method, and the non-negotiable label
Two things must be kept apart for the rest of this document to be read correctly. The task is
online prediction of counterfactual answer-influence: flagging, during generation, which steps would
— if resampled — most change the final-answer distribution. The method (V3, the mechanical
model) is one way to attempt the task; it is not the task, and it can fail while the task remains
well-posed and approachable by other means. Consequently the only valid ground truth is the
resampling-derived counterfactual-importance score of Bogdan et al. Validating against sentence
type (planning/backtracking), correctness, or difficulty would silently substitute an easier, different
target; because those proxies are already known to be decodable from trajectory geometry (§3),
matching them would prove nothing about anchors. Every accuracy claim in this document is
therefore implicitly “versus counterfactual-importance labels on the same traces.”

  This is the easiest place in the whole program to fool oneself. “Planning sentences are anchors”
  is a finding of the original paper, and planning sentences are easy to detect; a detector that
  fires on them will look successful while having learned nothing about counterfactual influence.
  If labels are not the resampling scores, the result is not about anchors.

## 1.6 The anchor unit: clause-respecting token windows
A point of departure from Bogdan et al.: although we adopt their counterfactual notion of impor-
tance, the unit of analysis is not the sentence. An anchor here is a token-window span Si — but
the windows are not cut at an arbitrary token count. The motivation is twofold: small reasoning
models emit short traces, so sentence units give too few labels per trace and coarse localization; yet
a window that ignores syntax can split a single fact, relation, or asserted value across two spans and
dilute its measured importance. We therefore use a two-stage definition that respects linguistic
structure before windowing.
    Stage 1 — parse into clauses. Segment the trace into sentences, then into clauses, splitting at
sentence terminators, newlines, comma-delimited clause boundaries, and discourse connectives (so,
therefore, thus, then, because, . . . ). Each resulting clause is a self-contained linguistic unit.
    Stage 2 — window within clauses. Subdivide each clause into contiguous windows of at most τ
tokens. Windows never cross a clause boundary: a clause shorter than τ is a single span; a longer
clause becomes ⌈len/τ ⌉ spans, with a short trailing fragment merged into its neighbor within the
same clause. The window size τ is swept once and then held fixed.

   This makes a span always a sub-unit of one clause, never a fragment straddling two. Throughout
the implementation (§6), wherever the physics refers to a “step” or a “boundary” it means one
such clause-respecting window and its right edge, not a sentence (note the time/step index t in the
physics is unrelated to the window size τ used here); the physics is unchanged because all per-token
quantities aggregate over arbitrary spans.

  Two knobs now exist where there was one: the clause-parse rule (Stage 1) and the window
  size τ (Stage 2), and both can change the labels. Respecting clause boundaries removes the
  worst failure of arbitrary windowing (a fact split across two spans), but it does not eliminate
  the dependence — a clause longer than τ is still cut internally, and the clause parser itself
  can mis-split (nested clauses, lists, math). Fix both rules once, before any signal is fit, and
  report them; sweeping span geometry alongside the detector would let it launder signal. The
  constructive sanity check (§2.5) is the guard: if a known supporting fact lands cleanly inside
  one span under the chosen τ , the parse is doing its job.

# 2 Dataset generation: token-window anchor labels

The detector needs ground-truth anchor labels to validate against. This section adapts a parallel
protocol’s data-generation pipeline (perturbed parallel resampling at word-group granularity) to
the token-window setting, and adds two elements specific to the present method: storing full traces
with token alignment, and recording a first-content-token probe decision per span. The pipeline is
independent of the physics — it produces the labels any online anchor detector (physics or learned)
would be scored against.

## 2.1 Perturbed resampling labels
For span i in a base trace, importance is the counterfactual dependence of the answer on i’s content.
Generate R continuations from the prefix through i (“real”) and R from a prefix in which i is replaced
by a perturbation i′ , and measure the shift in the final-answer distribution. The perturbation must
be genuinely different in meaning from i, or the comparison is not a counterfactual at all — so
the primary procedure is the Thought-Anchors resample-and-filter: resample i′ from the model,
then keep it only if it passes a semantic-difference test (§2.2). A construction-time alternative is
available on the synthetic curriculum, where the vocabulary is controlled: swap the entity, relation,
or value asserted in i for another from a mutually exclusive set (“the box is in the kitchen” →
“. . . in the garden”), which guarantees a different meaning by construction and needs no filter. We
use the resample-and-filter as primary everywhere (it is the established method and applies to any
task), and the controlled swap as a cheaper, filter-free cross-check on the synthetic tasks where it
is available. For verifiable-answer tasks the answer is discrete, so the label is the total-variation
distance between the real and perturbed answer distributions:
                                                               X
                 Ai = TV p̂ans | reali , p̂ans | perturbedi = 21    p̂(a | i) − p̂(a | i′ ) ,      (3)
                                                                a

with large Ai marking an anchor. Pipeline: (1) generate base traces; (2) parse into clauses, then
window within clauses (§1.6); (3) per span, dispatch 2R rollouts (R real, R perturbed), batched
across GPUs; (4) aggregate answer distributions and compute Ai ; (5) in parallel, hook the forward-
pass quantities the detector needs (hidden states for ht , output distributions for V and the KL
proxy). The independent unit for statistics is the trace, not the span, since within-trace spans are
correlated.

  This is the gold label and it is expensive: 2R rollouts per span × spans per trace × traces. It
  is also the only non-circular ground truth, because it is computed from the text (black-box)
  and is independent of the forward-pass signals being validated. A cheaper hidden-state proxy
  (e.g. a finite-time trajectory-divergence / Lyapunov score) may be calibrated against Ai and
  used to scale labeling only after a high measured correlation — and it carries a circularity
  risk precisely because it reads the same internal states the detector does. Resampling stays
  primary.

## 2.2 Semantic-difference adjudication: which resamples count
A resampled i′ only yields a valid counterfactual if it means something different from i; a para-
phrase contributes a spurious zero-importance reading and corrupts Ai . The procedure that decides
“different vs. not” is therefore load-bearing, and we adopt a tiered set of options trading compute
for fidelity.
    The primary procedure is the Thought-Anchors resample-and-filter; within it, the adjudicator
that decides “different vs. not” is one of the following, ordered by compute. The default adjudicator
is the cosine-median filter; NLI is an upgrade used where the budget is ample, and the rest are
fallbacks for when it is not.

Default adjudicator — Sentence-BERT median-threshold filter (the Thought-Anchors
method). Embed i and each resample i′ with Sentence-BERT; keep i′ as a valid “different” coun-
terfactual only if cos(i, i′ ) falls below the median cosine similarity over all span pairs in the dataset
(the threshold is a single precomputed dataset-wide scalar). This is the established definition of
semantic dissimilarity from the original method, and — importantly for the compute story — it
is cheap: one embedding pass per resample plus a precomputed median. The expensive part of
labeling is the R resampling rollouts, which are paid regardless of how difference is adjudicated;
the filter itself adds negligible cost. Resamples that fail the filter are discarded and re-drawn up
to a fixed attempt budget, so each span accumulates R filtered perturbed rollouts. This is the
adjudicator carried into the primary experiment and the only one that scales unchanged to large
models.

Upgrade — NLI non-entailment (where compute is ample, e.g. the small target model).
“Below-median cosine” is a blunt proxy: it can pass a surface edit that preserves meaning or reject
a genuine change phrased similarly. Where the budget allows — chiefly on the small target model
(DeepSeek-R1-Distill-Qwen-1.5B scale), where a second model is cheap to hold — upgrade the
adjudicator to require that i′ be not entailed by i (and ideally mutually non-entailing) under a
natural-language-inference model: a sharper test of “different meaning” than embedding distance,
at the cost of one NLI forward pass per candidate. This is a strictly higher-fidelity replacement for
the cosine filter, not a different procedure; it is optional, and if used it is fixed for the whole label
set (see the hazard below). The headline experiment uses the cosine-median filter as its
adjudicator — it is the established method, it scales unchanged, and it keeps the label definition
identical across model sizes; NLI is run, if at all, only as a fidelity cross-check on the small target
model, never mixed into the primary label set.

Lower-compute fallbacks (for larger models, when cost gets out of hand). Even with
the cheap cosine filter as adjudicator, two things bite at scale: the re-draw loop (rejected resamples

waste rollouts) and, if the NLI cross-check was being run, a second model held in memory beside
a large target. Fall back, in order:

1. Drop the NLI cross-check, keep the cosine median filter. Removes the second model
   entirely; the embedding pass is negligible. The headline adjudicator is unaffected — this only
   retires the optional fidelity check.
2. Construction-time difference on the synthetic curriculum. Where the vocabulary is
   controlled, draw i′ from a mutually exclusive set so difference is guaranteed and no filter and
   no re-draw are needed — the cheapest path, available precisely on the tasks used for the early
   constructive check (§2.5).
3. Cache and reuse the embedding model across all spans/traces (a small fixed model, e.g.
   a MiniLM-class encoder), and cap the re-draw budget so a span that cannot produce a passing
   resample is dropped rather than retried indefinitely.
4. Cheaper encoder / precomputed threshold. Swap Sentence-BERT for a smaller sentence
   encoder and freeze the median threshold from a calibration subset rather than recomputing over
   all pairs.

  The filter choice changes the labels, so it must be fixed and reported, not tuned. Two specific
  hazards. First, the median threshold is dataset-relative: computed over clause-respecting spans
  (§1.6), not sentences, its value differs from the original paper’s, so the threshold must be
  recomputed on our own span distribution, never borrowed. Second, swapping adjudicators
  between the primary run and a fallback is not free — cosine-median and NLI-non-entailment
  do not select the same resamples, so a label set built under one is not interchangeable with the
  other; if compute forces a switch mid-program, the affected traces must be re-labeled under
  the new rule, not mixed. Pick one adjudicator per label set and state which.

## 2.3 Trace storage with token alignment
Because the unit is a token window and the physics reads per-token hidden states, the stored
artifact must preserve exact token alignment, not just text. For each base trace we store: the token
id sequence (prompt and generation separated); the character/token offsets so spans map back
to text; the per-token clause-boundary and window-boundary flags from the two-stage segmenter;
the per-token surprisal (for V ); the sampled generation-time hidden states at the chosen layer(s)
and a stride; and, per span, the rollout answer distributions and the computed Ai . Storing token
alignment is what lets the same archive serve resampling labels, the physics signals, and the baselines
on a common index (item, trace, run, layer, token), so a span’s label and its dynamical signals are
never misaligned by a re-tokenization.

  Token alignment is a frequent silent bug source: re-tokenizing stored text can yield a different
  token count than generation produced (whitespace, special tokens, BPE merges at span edges),
  which shifts every per-token signal relative to its label by a few positions — enough to destroy
  a real correlation or manufacture a spurious one. Store the generated token ids directly and
  never re-tokenize for alignment. This is mundane and load-bearing.

## 2.4 The first-content-token probe decision
The online constraint forces a decision about when, within a span, the detector commits. We
record a first-content-token decision: the detector scores the span as soon as its first content-
bearing token has been generated (skipping leading whitespace, punctuation, and pure discourse
connectives), rather than waiting for the span to complete. This is the most stringent online setting
— it maximizes lead time — and it is recorded as an explicit per-span decision point so that the
achievable accuracy can be reported as a function of how early the commit is made. Concretely,
for each span we store the token index of its first content token and evaluate the detector’s causal
signal up to that index (plus the fixed Savitzky–Golay lag of §6), giving an early, fixed-lead-time
decision; the span-complete decision is retained as the lower-stringency comparison.

  Two honest points. First, the first-content-token decision is where the causal ceiling bites
  hardest: the less of the span the detector has seen, the less local-trajectory evidence exists,
  so early commit trades accuracy for lead time, and the trade curve — not a single number —
  is the honest result. Second, “content token” is a definitional choice (which tokens count as
  discourse-only) that can shift the decision index; fix the rule with the span definition (§1.6)
  and report it, because moving it changes both the lead time and the score.

## 2.5 Curriculum and the constructive sanity check
Labels are generated over a difficulty ramp of verifiable, state-tracking reasoning, the regime where
anchors are most pronounced and pivotal steps are sequential and identifiable: synthetic state-
tracking with known on-path supporting facts (a constructive ground truth independent of resam-
pling — the load-bearing facts are labeled by construction, so signals and Ai should both rank them
above distractors); then progressively harder state-tracking tasks (object-tracking under swaps,
compositional kinship with graded hop count, long-horizon instruction execution). A different rea-
soning type (e.g. grade- school math) is held out purely as an out-of-distribution generalization
test, not for fitting, so a positive result is not silently confined to state-tracking.

  The constructive check (synthetic tasks with annotated supporting facts) is the cheapest and
  most valuable early signal: it validates the whole pipeline — segmentation, alignment, label
  computation — against a ground truth that needs no resampling. If the resampling label
  Ai does not rank annotated supporting facts above distractors on these tasks, the pipeline is
  broken and no downstream result is trustworthy. Run it first.

# 3 Related work and the precise novelty claim

This work sits next to a fast-moving 2025–2026 literature on hidden-state trajectory geometry. The
claim here is narrow and must be stated against that literature exactly, because the substrate
(vt = ∆ht ) is shared while the target and the causal/online constraint are not.

The anchor target is retrodictive in all prior anchor work. Bogdan et al. (2506.19143)
define anchors and give three attribution methods — counterfactual resampling (∼100 rollouts),
attention aggregation, and attention suppression — all of which require the full trace or many
rollouts. None is causal/online. The task of predicting anchor-influence during generation is not
addressed there or, to our knowledge, anywhere.

Substrate prior art (shared method, different target). A cluster of papers analyzes generation-
or layer-time trajectories but targets something other than counterfactual answer-influence: Zhou et
al. (2510.09782, ICLR 2026) model reasoning as smooth flows and use position/velocity/(Menger)
curvature to study logic-vs-semantics — pure kinematics, no mass/momentum/energy/potential,
and no detector; Damirchi et al. (2603.01326) learn a classifier on layer-wise displacements to de-
tect reasoning validity; “TRACED” (2603.10384) decomposes trajectories into progress (displace-
ment) and stability (curvature) for reasoning-quality assessment; Gjølbye et al. (2605.15454) study
difficulty-vs-geometry coupling and establish that raw geometry is dominated by generation length.
These share the kinematic substrate but not the anchor target; they are prior art on trajectories,
not competitors on online anchor isolation.

Closest near-misses, and why they are not the task. Two works are genuinely online/predictive,
which makes them the right baselines but not task-competitors. Sun et al. (2604.05655, ACL 2026)
predict final-answer correctness mid-reasoning (ROC-AUC up to 0.87) via step-specific subspaces
and add trajectory steering — online and predictive, but the target is correctness, not anchor-
influence, and the mechanism is learned subspace geometry, not physics. Marı́n (2410.04415) is the
one prior work that reaches a genuine mechanical formalism (a reasoning Hamiltonian H = T − V
with p = ∆q), but it is offline, uses a goal-similarity potential that needs the answer (so it can-
not run causally), predates Thought Anchors, and targets valid-vs-invalid reasoning. It also states
plainly that the physics–reasoning connection “remains largely metaphorical” — the same caveat
carried throughout this document.

The novelty claim, stated narrowly. Crossing target × causality × formalism: online predic-
tion of counterfactual anchor-influence is unoccupied by any method; this is the primary claim.
The mechanical/least-action instantiation of it is a secondary, separable claim. The shared kine-
matic substrate (Zhou, Damirchi, Gjølbye), online correctness prediction (Sun), and the offline
mechanical framing (Marı́n) are all claimed elsewhere.

  The walls around the empty cell are close. Two consequences for honesty: (i) the contribution
  is not “trajectories/velocity for reasoning” — that is crowded, with work at ICLR and ACL
  — so the framing must lead with the task, not the geometry; (ii) the real baselines-to-beat
  are concrete and published: a learned step-boundary classifier trained against counterfactual-
  importance labels (the atheoretical version of the task), and a length-corrected kinematic signal
  in the spirit of Gjølbye et al. If the physics merely matches these, it is an elegant reframing,
  not a new capability. The collapse-to-V2 and collapse-to-kinematics ablations (§5, §9) exist
  precisely to detect that outcome. Usefully, the Gjølbye et al. trajectory archive and Sun et al.’s
  released code remove two of the three things one would otherwise have to build (an extraction
  substrate and a strong online baseline).

# 4 The three validity tests

The mechanical model is admissible only if the dynamics support it. Three tests, run before tuning
any detector, decide which model (if any) is valid. They probe independent structure and can fail
in compatible combinations.

Test 1 — conservativeness (coasting energy flatness). Compute Et = 12 vt⊤ Gvt + V (ht ) and
its rate dE/dt over steps labeled as coasting (§6.6). Pass: dE/dt ≈ 0 on coasting; the system

admits a time-independent Lagrangian. Structured fail: dE/dt is nonzero but fittably patterned
(e.g. steady decay)→ dissipative. Unstructured fail: dE/dt large and patternless → no energy
concept survives.

Test 2 — conservativeness of the force (curl). Estimate the force field F (h) = −∇V by
off-path probing (§6.2) and form the subspace Jacobian ∂i Fj . Pass: its antisymmetric part ≈ 0; F
is a gradient, a scalar potential exists. Fail: nonzero stable curl; no scalar potential — the force
has a rotational part.

Test 3 — reparameterization robustness. Recompute the candidate anchor score under a
different time-clock (e.g. arc-length instead of token-count). Pass: the score’s anchor ranking is
stable. Fail: the score changes materially with the clock, i.e. it is an artifact of the (unphysical)
token parameterization, often correlated with span length.

  Test independence is real and useful: a purely rotational force does no work, so Test 1 can pass
  (energy conserved) while Test 2 fails (nonzero curl). The curl estimate in Test 2 is the most
  numerically fragile quantity in the whole program: the antisymmetric part of a finite-difference
  Jacobian is exactly where estimation noise concentrates. “Stable curl” must mean stable under
  varying the probe step ϵ and the subspace; transient curl is noise, not physics.

# 5 Branch-conditional models

Each test outcome routes to a specific, named, textbook framework. The rule is Occam: use the
least general framework consistent with the results, because each generalization buys descriptive
power with extra free functions and thus with renewed unfalsifiability risk. In every live branch
the detector has the same shape — observed impulse minus the impulse the baseline model predicts,
with the baseline fit on coasting — and only the included force and the read-out differ.

## 5.1 Branch A — natural Lagrangian (all pass)
Define: V from (2); E = 21 v ⊤ Gv + V . Shape: essentially nothing; baseline is dE/dt = 0,
threshold from the coasting spread. Momentum: kinetic = canonical = Gv. Anchor score:
the cheap scalar |dE/dt| (needs only V , no gradient probe), or the full impulse residual ∥G∆v +
∇V ∥G−1 . Work analogy: fully intact — anchor work is genuine energy injection. Validate:
coasting |dE/dt| ≈ 0; anchor contrast vs. nulls; the collapse-to-V2 ablation (remove the V term —
if discrimination survives, the potential was dead weight and this is not really V3).

## 5.2 Branch B — dissipative, structured (Test 1 structured fail)
Define: V as above plus a Rayleigh dissipation function F = 21 v ⊤ Cv. Shape: fit the drag C by
regressing coasting dE/dt onto −2F = −v ⊤ Cv (diagonal, PSD-constrained); the baseline becomes
dE/dtexp = −2F(vt ). Momentum: still p = Gv. Anchor score: the residual dE/dt + 2F —
energy change beyond modeled drag. Work analogy: intact with richer bookkeeping; ∆T =
Wcons + Wdiss + Wanchor , and you have modeled the first two. Validate: is the fitted C stable
across traces? If not, linear drag is wrong and you may be heading to Branch E.

  On the dynamics axis this is, in my honest estimate, the most likely landing: autoregressive
  generation has no obvious reason to be conservative, so expect Test 1 to structured-fail into
  here rather than into the clean Branch A. That is a claim about which model fits the dynamics,
  not about whether the detector then beats baselines — a dissipative system whose structure
  still does not predict anchors is a clean negative on the separate outcome axis (§9). Being in
  Branch B is the expected dynamics; beating baselines from it is the open question.

## 5.3 Branch C — gauge / rotational (Test 2 fail, Test 1 pass)
Define: Helmholtz-decompose the probed force in the active subspace, F = −∇Veff + Fsol , and a
vector potential via Fsol = v × B, B = ∇ × A. Crucially redefine momentum as the canonical
momentum
                                     pcan = Gv + A ̸= Gv.                                  (4)
Shape: fit B (antisymmetric part of the subspace Jacobian, obtained from the same probes as
∇V ) on coasting. Anchor score: ∥Fsol ∥, or the canonical-impulse residual; plus |dE/dt| for any
residual energetic anchors. Work analogy: broken for the rotational anchors — they do no
work and are invisible to the work integral; you must use impulse. Fingerprint: bending without
energy change — a momentum shift with no reinforcement. Validate: curl stable under ϵ and
subspace; Fsol concentrates at labeled anchors.

  This branch’s central object, B, is the least trustworthy quantity in the plan. It can be written
  in five lines and still be pure finite-difference noise. If you land here, the first job is to establish
  that the curl is real, before believing any anchor result built on it.

## 5.4 Branch D — Maupertuis / Jacobi (Test 3 fail, Test 1 pass)
Define: fix the conserved energy level E (median of T + V on coasting); the Jacobi metric gE =
2(E − V ) G; recast motion as a geodesic of gE . Shape: estimate E and the metric field along the
path. Anchor score: geodesic deviation in gE , ∥∇guE u∥ with u the arc-length-normalized velocity
— clock-free by construction. Work analogy: dissolved into geometry — work and energy are
absorbed into the conformal factor 2(E − V ); momentum survives as the geodesic tangent. Cost:
gE is position-dependent, so Christoffel symbols return — but now they are the point, and they buy
reparameterization invariance. Validate: does the score now survive the clock change that sent you
here? Dependency: Maupertuis needs a conserved E, so this branch requires Test 1 to have passed;
if Tests 1 and 3 both fail, Jacobi does not apply and the dissipative reparameterization-covariant
generalization is required (much harder).

## 5.5 Branch E — metriplectic / GENERIC (multiple structured fails)
Define: a reversible (Poisson) bracket generated by E plus an irreversible bracket generated by
an entropy S, with the degeneracy conditions {·, S} = 0, [·, E] = 0. Shape: both the conservative
dynamics and the dissipative bracket — many free functions. Anchor score: anomaly in the re-
versible part against the modeled irreversible background. Work analogy: survives but fragments
into reversible/irreversible parts.

  The free-function count here is high enough that a positive result is hard to trust unless the
  simpler branches were cleanly ruled out and the GENERIC structure is strongly constrained

     by data. This is a last resort, not a destination.

## 5.6 Branch F — unstructured collapse (Test 1 unstructured fail)
Define: nothing physical. p = Gv becomes one feature among {p, ∆p, v, surprisal} fed to a plain
supervised anchor predictor. Anchor score: the predictor’s output. Both analogies: dead.
Validate: must beat position + surprisal + backtrack-keyword; if not, drop the approach.

     The honest discipline in Branch F is to say the physics is dead and not relabel a logistic regres-
     sion as mechanics. No metric, energy, or geometry helps once Test 1 fails without structure.

## 5.7 Summary of where each analogy survives

            Branch                            Momentum               Work                 Read-out
            A natural Lagrangian                  Gv                 intact           energy injection
            B dissipative                         Gv                intact∗       drag-corrected injection
            C gauge / rotational               Gv + A                 blind          solenoidal bending
            D Maupertuis / Jacobi          geodesic tangent       dissolved          geodesic deviation
            E GENERIC                             Gv             fragmented       reversible-part anomaly
            F unstructured                   feature only             dead         supervised prediction

∗
    intact after subtracting the modeled dissipative work baseline. Momentum (impulse) survives in five of six branches;
literal momentum–work survives cleanly in only two (A, B).

# 6 Implementation: variable-level details

This section pins each variable to an exact model quantity and marks every parameter as pinned
(read directly), sweep (empirical, no correct a-priori value), or measure (a diagnostic that deter-
mines a ceiling, not a knob to tune for performance).

## 6.1 Shared skeleton (all branches)
State ht (pinned read; sweep layer/position)
  hidden states[layer] from a forward pass with output hidden states=True; the residual
  stream after block layer. Two sub-choices: layer (∼0.7 depth default, sweep) and position
  — the token-window span (§1.6) is represented by its last token by default (or its first-content
  token for the early decision, §2.4); mean pooling over the span is an ablation.

Velocity vt (defined; sweep window)
  Savitzky–Golay first derivative over a symmetric window; the forward half is the lag budget.
  Parameters: half-width w (sweep), polynomial order (2–3). Units caveat: compute per-token
  with dt = 1 and integrate over the span; do not mix a token clock with span-level decisions
  silently.

Metric G (defined; sweep shrinkage)
 Constant whitening metric G = Σ−1 , Σ the velocity covariance pooled over a held-out trace
 corpus (not the scored trace — avoids leakage), with Ledoit–Wolf shrinkage (coefficient sweep)
 before inversion, since Σ is rank-deficient in d ∼ 5000.

Momentum pt = Gvt (pinned)
 Covector.

Coasting set (measure/define; see §6.6)
  Operational label of baseline steps; required to fit every branch baseline.

## 6.2 The potential V and its gradient
V (ht ) (pinned read)                          P
   Summed token surprisal within the step: − tok log Pθ (tok), where logprobs are log softmax(logits)
   gathered at the realized tokens. As a field: splice h at position t, run the upper layers, read the
   realized token’s logprob.

∇V (ht ) (estimated; sweep ϵ)
  No closed form. Central finite difference along each active-subspace basis direction ei : ∂i V ≈
  [V (ht + ϵei ) − V (ht − ϵei )]/2ϵ. Each probe runs only layers L+1 → unembedding, reusing the
  KV-cache for positions < t. Cost: ≈ k (1 − layer frac) forward-pass equivalents per probed step;
  probe at span decision points only (§2.4). ϵ: sweep (too small → float noise; too large → leaves
  the linear regime).

  The off-path splice (overwrite the hidden state at layer L, run upward with the prefix KV-
  cache) is the one piece I can write structurally but cannot guarantee runs unmodified: the API
  for partial-forward-from-a-layer with cache reuse is model- and transformers-version-specific.
  This is engineering, not physics, but it is the load-bearing engineering seam.

## 6.3 The active subspace
Basis {ei }ki=1 (defined; measure k)
  Top-k principal directions of the pooled, held-out velocity series (alternative: top force-field
  directions). k is set by the energy capture diagnostic: the fraction of velocity (or force) energy
  in the top k; choose the knee and report the captured fraction as the recall ceiling. k
  also sets the ∇V probe cost, so cost and accuracy are the same knob.

## 6.4 Fisher / KL proxy (only if a branch needs a non-trivial metric)
KL-proxy step length (pinned read; approx in K)
 Replace v ⊤ GFisher v by the measured 2 KL(Pt−1 ∥Pt ) between consecutive boundary distributions;
 exact as the retained top-K vocab slice → full vocab. K: sweep.

  A constant whitening metric is the right default and keeps the EL residual flat (no Christoffel
  term). Position-dependent Fisher reintroduces the 12 (∂i Gjk )v j v k kinetic-connection term and
  the associated estimation of ∂G; adopt it only if a constant metric demonstrably leaves signal
  on the table. Earlier in the design this Fisher term was dropped by mistake in the residual —
  it is only required because of a position-dependent metric, which is itself optional.

## 6.5 Branch-specific objects
Drag C (B)
  least squares of coasting dE/dt on −v ⊤ Cv, diagonal PSD.

B-field (C)
  antisymmetric part of the subspace Jacobian ∂i Fj — obtained from the same probe set as ∇V
  (one probe set yields the full Jacobian; symmetric part informs ∇V , antisymmetric part is B).

Energy level E (D)
  median of T + V over coasting.

Brackets {, }, [ , ] and entropy S (E)
  fit; high free-function count.

## 6.6 The coasting set — circularity, handled
The baseline fit in every branch needs coasting labels, but a clean label may not exist a priori.
Bootstrap: (i) provisional Euclidean-energy score; (ii) take the calm bottom half as coasting;
(iii) fit the baseline; (iv) recompute and check stability under one iteration. Alternatives: the
constructive on-path/off-path labels from the synthetic curriculum (§2.5; off-path spans are coasting
by construction), or low-surprisal runs.

  This bootstrap is circular by construction. It can be made stable-in-practice but not principled.
  State it as a limitation; do not let a coasting set chosen by a provisional score silently define
  the anchors the final score then “discovers.”

# 7 Tiered execution plan

Everything specified above is something to try eventually; it is not a single experiment. The
program runs in tiers. Tier 1 is the most basic, bells-and-whistles-free run that has a real chance
of working on intuition and what is already known — but it still carries every non-negotiable
safeguard, because omitting one of those is itself a critical failure, not a simplification. Each later
tier adds a small subset of features chosen by the previous tier’s outcome: if a tier failed, add
the specific fix that addresses the failure observed; if it succeeded, skip fixes and add only an
improvement. Tiers accrete until either every necessary fix has been tried and all fail (a clean
negative), or the best working version is reached with every applicable improvement folded in.
Nothing here changes the experiments or design of §§2–9; it only sequences them.

## 7.1 Three buckets every feature falls into
 Always-on safeguards (in every tier; omitting one is a critical failure, not a tier choice):
  counterfactual-importance resampling labels, not proxies (§2.1, §1.5); a frozen semantic-difference
  adjudicator so only genuinely different resamples count toward Ai (§2.2) — the specific adjudi-
  cator may be swapped for a cheaper one at larger scale, but it is fixed per label set, never tuned;
  exact token alignment via stored generated ids (§2.3); a frozen span definition (§1.6); held-out
  covariance for the metric, no leakage (§6); the constructive sanity check run first (§2.5); the
  coasting-fit measurement and the causal-vs-full- information ceiling measured before tuning (§4,
  §9); and the published baselines reproduced on the same traces (§3).

 Fix-features (added only to address a specific observed failure): the branch models keyed
  to which validity test failed — Branch C for nonzero curl, Branch D for a clock artifact,
  Branch E for combined failures (§5); the off-path ∇V probe (§6.2), which a fix needs the moment
  Test 2/Branch C enters; the position-dependent Fisher metric (§6); and the D → DIG →exact-
  ablation ladder for a first-order gradient that under-reads. (Branch B, the dissipative case, is
  not a later fix: it is one of the two Tier-1 detector options, selected by Test 1 — see below.)
 Improvement-features (added only on success). Two kinds, kept distinct because they pull
  in opposite directions on the accuracy axis: capability improvements buy something the detector
  could not do before but are not expected to raise accuracy — chiefly the first-content-token early
  decision and its lead-time/accuracy curve (§2.4), which by the causal-ceiling argument trades
  accuracy for lead time; accuracy improvements aim to raise the score itself — the full impulse
  residual ∥G∆v + ∇V ∥G−1 in place of the cheap scalar (§5, Branch A/B), active-subspace tuning
  toward the energy-capture ceiling, and the KL/Fisher metric upgrade where it demonstrably
  adds signal (§6).

## 7.2 Tier 1 — the minimal run
The cheapest configuration that still tests the core bet. Crucially, Tier 1’s first job is diagnostic:
Test 1 (coasting energy flatness) reads off whether the system is conservative or dissipative, and
that reading — not a prior assumption — selects the Tier 1 detector.

 Detector (chosen by Test 1, not assumed): if coasting dE/dt ≈ 0, Branch A with the
  cheap scalar |dE/dt|; if coasting dE/dt shows a structured (fittable) decay, Branch B with the
  drag-corrected residual dE/dt + 2F. Both use only V and a constant whitening metric — no
  ∇V probe, no Fisher, no Christoffels — and the span-complete decision (not the early commit).
  Branch B adds exactly one fitted parameter (the drag C), and it is fitted only once Test 1 has
  confirmed there is dissipation to fit; fitting drag to a near-conservative system would invent
  structure.
 Diagnostics: Test 1 (energy flatness) and Test 3 (reparameterization robustness) — both
  cheap, no probe. Test 2 (curl) is deferred, because it requires the ∇V probe, a Tier 2+ feature.
 Baselines, reproduced on the same traces and labels: the length-corrected kinematic sig-
  nal and the learned step-boundary classifier trained directly against counterfactual-importance
  labels (§3). The learned classifier is the stronger competitor and the one that decides whether
  the task contribution survives even if the physics does not; it must be in Tier 1, not deferred.
 Everything in the safeguard bucket, on.

This is the no-frills version: fewest moving parts, lowest cost, and a central quantity (dE/dt) that
is essentially free to compute. Its honest expectation is Branch B: the document’s estimate is that
the system is dissipative (§5), so the drag-corrected residual — not the bare |dE/dt| — is the
better-targeted of the two Tier-1 detectors. That it is better-targeted is a claim about fitting the
dynamics; whether it then beats the baselines is the separate open question (§9). Tier 1 is built so
that arriving at the right detector costs one cheap diagnostic, not a wasted run.

## 7.3 The ladder
The previous tier’s outcome selects the next tier’s single increment.

  Tier      Increment added                                         Entered when
# 1         Minimal detector, Branch A or B by Test 1 (drag-        Always: start here.
            corrected residual if dissipative) + all safeguards +
            Tests 1, 3 + both baselines
  2-fix     The fix keyed to the failed test: Branch D if Test 3    Tier 1 detector (A or B) does not beat base-
            (clock) fails; the ∇V probe + Test 2 + Branch C if      lines, or Test 3 fails.
            rotational structure is suspected (§5)
  2-imp     Capability improvement: first-content-token early       Tier 1 detector beats baselines and Test 3
            decision + its lead-time/accuracy curve (§2.4) —        passes.
            adds online lead time, expected to cost accuracy, not
            raise it
  3-fix     Next fix on the stack: Fisher metric if constant        Prior fix tier helped but is insufficient.
            whitening leaves signal; DIG /exact ablation if the
            gradient under-reads; Branch E only if combined fail-
            ures persist
  3-imp     Accuracy improvement: full impulse residual ∥G∆v+       Prior tier worked and more accuracy is
            ∇V ∥G−1 in place of the scalar; subspace/metric tun-    wanted.
            ing toward the energy-capture ceiling

The fix and improvement paths can interleave across tiers: a fix that turns a failure into a success
hands off to the improvement path, and a working detector that later plateaus can take a targeted
fix. The branch selected by the validity tests (§5) is exactly the fix-feature ladder, indexed by which
test failed — the tiering and the test→branch logic are the same logic, framed as execution order.

## 7.4 Termination
Stop at the first of: (i) every necessary fix tried and all fail — a clean negative, reported as “the
momentum framing is metaphor, not mechanism” (§9, Branch F); or (ii) the best working version
— Tests pass, baselines beaten, and every applicable improvement folded in with no remaining
increment that helps.

  “Likely to succeed” is relative, and the document’s own estimate shapes Tier 1: the most
  probable live outcome is the dissipative case (§5, Branch B), so Tier 1’s expected landing is
  the drag-corrected residual, reached via one cheap diagnostic (Test 1) rather than a wasted
  Branch A run. Tier 1 passing on the bare conservative scalar would be the pleasant surprise,
  not the base case. The entire point of tiering is to fail cheap — to learn the core bet is wrong
  (or that the dissipative detector already suffices) before building the probe, the curl machinery,
  the early-decision curve, or any other bell or whistle. A tier that adds more than one feature at
  a time forfeits this: if two features go in together and the result moves, the cause is ambiguous.
  One increment per tier is the discipline that makes each outcome interpretable.

# 8 What can and cannot be implemented now

Fully implementable and testable here (pure NumPy, no model). The entire physics
stage downstream of the extracted arrays: Savitzky–Golay velocity, whitening metric with shrink-
age, momentum, impulse residual, energy and its rate, all branch baselines (drag fit, curl/B from
the probe Jacobian, Jacobi geodesic deviation), the branch router, and all validation diagnostics
(coasting flatness, curl stability, reparameterization robustness, collapse-to-V2 ablation). Synthetic-
anchor tests confirm the arithmetic.

Writable correctly but not verifiable from here (needs the model/GPU). The extraction
stage: teacher-forcing, boundary hidden-state reads, surprisal, and the ∇V probe. Structurally
sound, but the partial-forward-from-layer splice with cache reuse is environment-specific and must
be adapted.

Writable but numerically unvouchable until run on real traces. The curl/B computation
(Branch C) and the gradient step ϵ: the code is correct; whether its output is signal or finite-
difference noise is a property of the data, not the code.

  “I can implement all the physics” is true at the level of operations and true-and-tested for
  the NumPy stage. It is not a claim that the curl will be meaningful, that the probe will
  run unmodified on your model, or — most importantly — that the idea works. The code is
  instrumentation; the first real result is the coasting-fit number.

# 9 What the results would mean

Test 1 passes, contrast beats baselines, ∇V ablation hurts. The strongest positive: rea-
soning is approximately a conservative least-action flow, anchors are energy-injection events, and
the potential carries independent signal. “Beats baselines” here means beats the two concrete,
published competitors named in §3 — a learned step-boundary classifier trained directly against
counterfactual-importance labels (the atheoretical version of the task ), and a length-corrected kine-
matic signal in the spirit of Gjølbye et al. — evaluated against counterfactual-importance labels,
not proxies. If it clears those, it is a solid, publishable result: the first online anchor detector,
with a physics instantiation that earns its keep. It is not field-changing; calibrate expectations
accordingly.

Two distinct collapse tests, two distinct verdicts. The ∇V ablation and the kinematic-
baseline comparison probe different failures. If removing ∇V does not hurt, V3 has collapsed to
V2 : the signal is in the momentum-change term alone, the potential is dead weight, and it should
be reported as an impulse detector, not a least-action one. If the full V3 score does not beat the
length-corrected kinematic baseline, it has collapsed to geometry: the mechanical formalism adds
nothing over directness/curvature, and the honest report is “trajectory geometry predicts anchors,”
crediting the substrate literature. Either collapse still leaves the task contribution intact if the score
beats chance and the proxies are excluded — but it removes the physics contribution.

Structured dissipative / rotational / clock-artifact (B/C/D). The mechanical analogy
holds in a more general form; the corresponding branch is the honest model. A rotational (C) result
is the most scientifically interesting if the curl is real: it would say anchors are re-directional, not
energetic.

Unstructured fail (F). The conservative-system premise is wrong and no fittable dissipative
structure replaces it: dE/dt is large and patternless, so no energy concept survives. This is not
the expected dynamics (the dissipative Branch B is); it is the worst dynamics case, in which the
physics is dead from the start.

Contrast within noise of baselines (the likely outcome). Independently of which branch
the dynamics land in, the most likely outcome is that the detector does not beat the baselines:

anchors may have no local dynamical signature beyond what trivial features capture. This is a
genuine, valuable negative — it tells you the momentum framing is metaphor, not mechanism,
before any expensive scaling. Note it kills the method, not necessarily the task : a learned classifier
might still predict anchor-influence online even where the physics does not.

The ceiling number. Independently, restrict any full-information score to causal information
and measure how much anchor-predictive power survives. The gap is the price of streaming and
caps the inference goal regardless of cleverness. Measure it early.

  Bottom line. The construction is physics-informed in structure (mass=metric, momentum=covector,
  conservative force, energy integral — all textbook) but not physics-derived from the model’s
  actual update rule. Three assumptions — the conservative premise, the token-clock, and
  surprisal-as-potential — are imposed, each with a concrete test that can fail. The most prob-
  able outcome is a clean negative (the detector not beating baselines), even though the most
  probable dynamics is the dissipative Branch B. The design’s worth is that it reaches whichever
  verdict it reaches cheaply, with the failure modes named in advance rather than discovered
  after a large-model run.

References
[1] P. C. Bogdan, U. Macar, N. Nanda, A. Conmy. Thought Anchors: Which LLM Reasoning Steps Matter?
    arXiv:2506.19143, 2025.

[2] J. Marı́n. Geometric Analysis of Reasoning Trajectories: A Phase Space Approach. . . arXiv:2410.04415,
    2024.

[3] Y. Zhou, Y. Wang, X. Yin, S. Zhou, A. R. Zhang. The Geometry of Reasoning: Flowing Logics in
    Representation Space. arXiv:2510.09782, ICLR 2026.

[4] H. Damirchi et al. Truth as a Trajectory: What Internal Representations Reveal. . . arXiv:2603.01326,
    2026.

[5] TRACED: Topological Reasoning Assessment via Curvature Evolution and Displacement Dynamics.
    arXiv:2603.10384, 2026.

[6] A. Gjølbye, L. K. Hansen, S. Koyejo. Reasoning Models Don’t Just Think Longer, They Move Differently.
    arXiv:2605.15454, 2026.

[7] L. Sun, H. Dong, B. Qiao, Q. Lin, D. Zhang, S. Rajmohan. LLM Reasoning as Trajectories: Step-Specific
    Representation Geometry and Correctness Signals. arXiv:2604.05655, ACL 2026.
