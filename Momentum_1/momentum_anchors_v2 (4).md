# Scope and Honesty Statement

This document describes a measurement apparatus and a bet. The bet is that reasoning trajectories move in a physics-like way and that anchors are where a force acts. We separate, explicitly and repeatedly, three categories of claim:

<div class="description">

A statement that is true by a theorem of information geometry or differential geometry, with no approximation beyond finite floating-point arithmetic.

A statement that is an estimator of, or a controlled approximation to, an <span class="sans-serif">\[EXACT\]</span> quantity — principled, but carrying an error we name.

A modeling choice the geometry does *not* force on us. Defensible, but a different choice would yield a different (sibling) law.

Machinery for fair evaluation. Not physics, and not pretending to be.

</div>

<div class="honestbox">

**The single most important honesty point.** “Inertial motion = straight line in logit space” is the geodesic of the *exponential connection* of information geometry, *not* the Levi-Civita (metric) connection that governs Newtonian mechanics on a Riemannian manifold. We therefore test an *information-geometric momentum law*, not Newtonian mechanics on the Fisher manifold. These are coherent siblings, not the same theory. We choose the e-connection deliberately (Section <a href="#sec:connection-fork" data-reference-type="ref" data-reference="sec:connection-fork">4</a>) and we never call the result “Newtonian.”

</div>

# Motivation and Relation to Thought Anchors

Thought Anchors scores each sentence of a chain-of-thought (CoT) by its *counterfactual importance*: it resamples replacement sentences, continues the trace, and measures the effect on the final-answer distribution. This is a *hindsight* quantity — it integrates over everything downstream of the sentence — and it is the ground truth we will try to predict.

<div class="honestbox">

**What can and cannot be done online.** The Thought Anchors importance number is *defined* by resampling the future, so no online quantity can equal it. What *can* be measured online is a present-tense property: whether the trajectory is being deflected right now. Our wager is that “a force is acting here” (local, present-tense, computable with one token of lag) predicts “this step had large counterfactual influence” (hindsight). The wager has a built-in ceiling: anchors whose importance is *purely relational to the future* — a calm, low-force step that happens to gate a decision 200 tokens later — carry no local signature and are undetectable by construction. Measuring the size of that blind spot is itself a result (Section <a href="#sec:tests" data-reference-type="ref" data-reference="sec:tests">11</a>).

</div>

We diverge from the original method in one respect that is an asset: where resampling yields only a magnitude of importance, our decomposition of the force into a *reinforcing* component (along the direction of travel) and a *redirecting* component (perpendicular to it) yields a *sign/structure* — accelerator versus brake — that the counterfactual method cannot produce.

# The Manifold, the Metric, and the Mass

## State space

Let the model have vocabulary of size $V$. At trajectory position $i$ (one token step), the model emits logits $\bm{\ell}_i \in \mathbb{R}^V$ over the next token. With generation temperature $T$, define the scaled logits and the output distribution
$$\bm{g}_i \;=\; \bm{\ell}_i / T, \qquad
  p_i \;=\; \mathrm{softmax}(\bm{g}_i) \in \Delta^{V-1}.$$
The physical state is the distribution $p_i$, a point on the statistical manifold $\Delta^{V-1}$. The hidden state is merely a coordinate that produced it and is *never used*; everything is a function of distributions, which is what makes every quantity invariant to reparameterizations of the representation. <span class="sans-serif">\[EXACT\]</span> (invariance), <span class="sans-serif">\[CONVENTION\]</span> (using $T$ from generation; see Section <a href="#sec:pulled" data-reference-type="ref" data-reference="sec:pulled">8</a>).

## The metric is the mass

In Riemannian (Lagrangian) mechanics the kinetic energy is $T_{\text{kin}}=\tfrac12 g_{ab}(q)\dot q^a \dot q^b$ and the mass *is* the metric tensor; conjugate momentum is $p_a = g_{ab}\dot q^b$ . We take the metric to be the Fisher–Rao information metric, which by Čencov’s theorem is the unique (up to scale) Riemannian metric on a space of probability distributions invariant under sufficient statistics . For a categorical distribution in logit coordinates the Fisher metric has the closed form
$$\bm{G}(p) \;=\; \mathrm{diag}(p) - p\,p^{\top},$$
so that for any two logit-space tangent vectors $\bm{u},\bm{w}$,
$$\boxed{\;\langle \bm{u},\bm{w}\rangle_{p}
  \;=\; \bm{u}^{\top}\bm{G}(p)\,\bm{w}
  \;=\; \mathbb{E}_{p}[u\,w] - \mathbb{E}_{p}[u]\,\mathbb{E}_{p}[w]
  \;=\; \mathrm{Cov}_{p}(u,w).\;}
  \label{eq:fisher-cov}$$
This is the technical keystone. The Fisher inner product is exactly a covariance under $p$, computable in $O(V)$ with no matrix ever formed, no Monte Carlo, and no low-rank sketch. <span class="sans-serif">\[EXACT\]</span>.

<div class="remark">

**Remark 1** (The metric quotients the softmax gauge for free). *Adding a constant $c\bm{1}$ to all logits leaves $p$ unchanged, so the all-ones direction is physically null. Under <a href="#eq:fisher-cov" data-reference-type="eqref" data-reference="eq:fisher-cov">[eq:fisher-cov]</a>, $\langle \bm{1},\bm{w}\rangle_p = \mathbb{E}_p[w]-\mathbb{E}_p[w]=0$ for every $\bm{w}$. The gauge direction is automatically in the kernel of the metric; no centering or log-prob conversion is needed. <span class="sans-serif">\[EXACT\]</span>.*

</div>

## Relation to KL

To second order, $\langle \bm{v},\bm{v}\rangle_p = \bm{v}^\top \bm{G}(p)\bm{v} \approx 2\,\mathrm{KL}(p \,\|\, p_{\bm{v}})$ where $p_{\bm{v}}$ is the distribution displaced by $\bm{v}$. We do *not* use this approximation: <a href="#eq:fisher-cov" data-reference-type="eqref" data-reference="eq:fisher-cov">[eq:fisher-cov]</a> gives the metric exactly. We mention it only to connect to the KL-based intuition of earlier drafts. <span class="sans-serif">\[EXACT\]</span> replaces a former <span class="sans-serif">\[DERIVED\]</span>.

# Inertia: Which Connection?

A “force-free” trajectory is a geodesic, but *geodesic of which connection?* The Fisher manifold of an exponential family carries a one-parameter family of dualistic $\alpha$-connections . Three matter here:

- the **e-connection** ($\alpha=1$): geodesics are straight lines in the natural (logit) parameters;

- the **m-connection** ($\alpha=-1$): geodesics are straight lines in expectation (mean) parameters;

- the **Levi-Civita connection** ($\alpha=0$): the metric connection of Riemannian mechanics.

<div class="honestbox">

**The fork, stated plainly.** Newtonian mechanics on a Riemannian manifold uses the Levi-Civita connection: free motion is the metric geodesic. Our inertial baseline — “the model keeps moving in the same logit-space direction” — is the *e-geodesic*, $\alpha=1$, not $\alpha=0$. So our “force” is the deviation from e-geodesic motion measured in the Fisher metric. This is a self-consistent object in information geometry (it mixes the $\alpha=1$ connection with the $\alpha=0$ metric, which is exactly how natural-gradient and related methods operate), but it is **not** Newtonian mechanics. We choose it knowingly.

</div>

#### Why the e-connection, and why it is the principled choice here, not merely the cheap one.

1.  **It is flat.** For an exponential family the e-connection has zero curvature; the natural parameters are affine coordinates. <span class="sans-serif">\[EXACT\]</span>.

2.  **Flatness makes the discrete bookkeeping exact.** e-parallel transport in logit coordinates is the identity map. Therefore accumulating and differencing velocities by ordinary vector arithmetic in logit space (as our EMA does, Section <a href="#sec:inertial" data-reference-type="ref" data-reference="sec:inertial">5</a>) introduces *no* transport error. Under Levi-Civita this would be false: curvature would force genuine parallel transport and the third-order skewness (Amari–Chentsov) tensor would appear. <span class="sans-serif">\[EXACT\]</span>.

3.  **It is the native geometry of the object we measure.** The model’s outputs *are* a categorical exponential family in the logits; the e-connection is their intrinsic affine structure, not a borrowed mechanical analogy.

The price is solely nomenclature: we must say “information-geometric momentum,” never “Newtonian.” We pay it. <span class="sans-serif">\[CONVENTION\]</span> (the choice of $\alpha=1$ over $\alpha=0$), with <span class="sans-serif">\[EXACT\]</span> consequences once chosen.

# Velocity, Momentum, and Force

## Velocity

We index by *scored content positions*: let $k=1,2,\dots$ enumerate the content tokens in order (formatting positions are skipped, not differenced; Section <a href="#sec:tokens-to-spans" data-reference-type="ref" data-reference="sec:tokens-to-spans">6</a>), and write $\bm{g}_k$, $p_k$ for the scaled logits and distribution at the $k$-th content position. The realized velocity is the e-connection tangent, i.e. the displacement in natural coordinates between consecutive content positions,
$$\bm{v}_k \;=\; \bm{g}_k - \bm{g}_{k-1}.$$
<span class="sans-serif">\[EXACT\]</span> as the e-tangent; <span class="sans-serif">\[DERIVED\]</span> as a finite-difference of a discrete path (leading order in step size — see Assumption <a href="#ass:smallstep" data-reference-type="ref" data-reference="ass:smallstep">1</a>, which the content-only indexing makes slightly more demanding because a content step may skip a short run of formatting tokens).

## Inertial prediction (momentum carried in)

A force-free e-geodesic has constant velocity. We estimate the incoming inertial direction by an exponential moving average of past velocities,
$$\bar{\bm{v}}_{k-1} \;=\; \beta\,\bar{\bm{v}}_{k-2} + (1-\beta)\,\bm{v}_{k-1},
  \qquad \beta \in (0,1),
  \label{eq:ema}$$
computed strictly causally: $\bar{\bm{v}}_{k-1}$ uses information only through content step $k{-}1$.

<div class="honestbox">

**The EMA is not literal momentum; it is an estimator of the inertial direction.** On a true geodesic the velocity is constant, so the EMA of a constant returns that constant and equals the instantaneous velocity. The EMA departs from the instantaneous velocity *only* when the velocity is changing — which is exactly when a force is present. Modeling per-step velocity as “geodesic $+$ observation noise,” the EMA is the natural variance-reduced estimate of the force-free continuation, and $\beta$ is the bias/variance knob with effective memory $\approx 1/(1-\beta)$ *content* tokens. This is precisely a physics-derived quantity tuned for detection accuracy. <span class="sans-serif">\[DERIVED\]</span>.

</div>

<div class="honestbox">

**Pre-registered design fork: which steps build $\bar{\bm{v}}$.** The content-token probe (Section <a href="#sec:tokens-to-spans" data-reference-type="ref" data-reference="sec:tokens-to-spans">6</a>) excludes formatting positions from *scoring* because their velocities are content-free and position-correlated. But the inertial baseline $\bar{\bm{v}}$ is itself an average of velocities, so if it is updated on formatting steps, the contamination the probe removes from $F_k$ re-enters through the reference direction. We therefore make the **content-only EMA the recommended default**: $\bar{\bm{v}}$ is updated only on content$\to$content transitions, and the velocity $\bm{v}_k$ is the displacement from the previous *content* position rather than the previous raw token. The alternative — update $\bar{\bm{v}}$ through every token to keep the raw trajectory unbroken — is retained only as an ablation. We expect content-only to win because it keeps a single, coherent definition of “the direction the reasoning was moving” free of formatting jumps; the gap between the two is reported. <span class="sans-serif">\[DERIVED\]</span>/design choice, defaulted toward the version more likely to carry signal.

</div>

## Force

The force direction is the deviation of the realized velocity from the inertial prediction,
$$\bm{d}_k \;=\; \bm{v}_k - \bar{\bm{v}}_{k-1},$$
and the force magnitude is its Fisher norm, evaluated at the launch point $p_{k-1}$:
$$\boxed{\;F_k \;=\; \sqrt{\langle \bm{d}_k, \bm{d}_k\rangle_{p_{k-1}}}
  \;=\; \sqrt{\mathrm{Cov}_{p_{k-1}}(\bm{d}_k,\bm{d}_k)}.\;}
  \label{eq:force}$$
This is $\lVert m\,\bm{a}\rVert$ with mass $=$ metric and $\bm{a}$ the e-acceleration. <span class="sans-serif">\[DERIVED\]</span> (e-acceleration via the EMA; metric base-point at $p_{k-1}$ rather than transported-averaged across the step — the one irreducible finite-step approximation, leading order).

## Reinforcement versus redirection (the structural signal)

Decompose the realized velocity into the part parallel to the incoming momentum and the part orthogonal to it, *in the Fisher metric*:
$$\begin{aligned}
  c_k &= \frac{\langle \bm{v}_k, \bar{\bm{v}}_{k-1}\rangle_{p_{k-1}}}{\langle \bar{\bm{v}}_{k-1}, \bar{\bm{v}}_{k-1}\rangle_{p_{k-1}} + \varepsilon},
  &\bm{v}_k^{\parallel} &= c_k\,\bar{\bm{v}}_{k-1},
  &\bm{v}_k^{\perp} &= \bm{v}_k - \bm{v}_k^{\parallel}.
  \label{eq:decomp}
\end{aligned}$$
Define the per-step *redirection fraction*
$$\boxed{\;r_k \;=\; \frac{\langle \bm{v}_k^{\perp}, \bm{v}_k^{\perp}\rangle_{p_{k-1}}}{\langle \bm{v}_k, \bm{v}_k\rangle_{p_{k-1}}+\varepsilon}
  \;=\; 1 - \rho_{p_{k-1}}\!\big(\bm{v}_k, \bar{\bm{v}}_{k-1}\big)^2 \;\in[0,1].\;}
  \label{eq:redir}$$

<div class="honestbox">

**This is the kinematic tangential/normal decomposition — textbook-exact.** Tangential acceleration changes speed (reinforcement); normal acceleration changes direction (redirection) and is path curvature. $r_k=\sin^2(\theta_k)$ where $\theta_k$ is the Fisher angle between current velocity and incoming momentum: the fraction of motion that is *turning*. “Anchors as redirection” becomes “anchors as high path curvature.” The decomposition is <span class="sans-serif">\[EXACT\]</span> given the metric and the velocity vectors; the only <span class="sans-serif">\[DERIVED\]</span> input is that the reference direction is the smoothed $\bar{\bm{v}}$ rather than an instantaneous velocity.

</div>

## Sign: accelerator versus brake

The redirection fraction $r_k$ is a magnitude in $[0,1]$ and carries no direction. The *sign* of the along-track motion is read off the same projection coefficient $c_k$ defined above (Eq. <a href="#eq:decomp" data-reference-type="ref" data-reference="eq:decomp">[eq:decomp]</a>):
$$\text{label}_k =
  \begin{cases}
    \textbf{accelerator} & c_k > 0 \\
    \textbf{brake / reversal} & c_k < 0 \\
    \textbf{pure turn} & |c_k|\ \text{small},\ r_k\ \text{large}
  \end{cases}$$
gated by a meaningful-motion threshold on $\langle \bm{v}_k,\bm{v}_k\rangle_{p_{k-1}}$ so that noise near a stationary point is not labeled. The two axes are orthogonal and complementary: $r_k$ says *how much* the trajectory turned; $\mathrm{sign}(c_k)$ says *whether* the along-track motion reinforced the incoming momentum ($c_k>0$, “keep going,” coasting near $c_k=1$ and accelerating for $c_k>1$) or reversed it ($c_k<0$), the geometric signature of backtracking. Pooled to a span, the sign is the energy-weighted mean projection,
$$C_s \;=\; \frac{\sum_{k\in s}\langle \bm{v}_k,\bm{v}_k\rangle_{p_{k-1}}\,c_k}{\sum_{k\in s}\langle \bm{v}_k,\bm{v}_k\rangle_{p_{k-1}}+\varepsilon},$$
so a span is an accelerator ($C_s>0$) or a brake ($C_s<0$). This is the operational definition behind the accelerator/brake claim in the abstract and Test 2; it adds no new machinery beyond reusing $c_k$.

<div class="honestbox">

**Why the sign is a co-primary signal, not a secondary one.** $R_s$ (magnitude of turning) must beat a token-surprise baseline under position controls to count — a hard bar, because surprise-like quantities already capture much of “where the model commits.” $C_s$ (sign: accelerator vs. brake) is different in kind: the resampling ground truth returns only a non-negative importance magnitude, so it *structurally cannot* distinguish a span that drives the reasoning forward from one that reverses it. A directional signal therefore has no baseline to beat — any reliable accelerator/brake separation is information the ground-truth method cannot produce by construction. We accordingly treat $R_s$ and $C_s$ as **co-primary**: $R_s$ for the magnitude correlation (the harder, surprise-contested claim) and $C_s$ for the directional structure (the more likely to be novel and to survive). The document’s headline does not rest on $R_s$ alone.

</div>

<div class="honestbox">

**What is and is not physics here.** The tangential/normal split and the sign of $c_i$ are <span class="sans-serif">\[EXACT\]</span> kinematics (tangential acceleration has a sign; reversal is $c_i<0$). The *interpretation* “$c_i<0 \Rightarrow$ linguistic backtracking” is an empirical hypothesis tested in Test 2, not a geometric fact. We report the sign as a measured quantity and the linguistic mapping as a claim to be verified.

</div>

# From Tokens to Spans

The geometry is exact only for small steps, so it lives at the token level. Ground-truth labels live at the span level, where a span is a boundary-respecting window (Section <a href="#sec:granularity" data-reference-type="ref" data-reference="sec:granularity">7.0.0.4</a>). We compute per-token and pool. We use **half-open** token ranges $[a_s, b_s)$ throughout, matching the reference code and removing the inclusive/exclusive off-by-one.

#### The content-token probe (a correctness decision, not a convenience).

Not every token position carries reasoning signal. Span- and sentence-terminal positions are dominated by formatting: the next-token distribution immediately after a clause is concentrated on punctuation, spaces, and newlines, and is nearly identical across the whole trace. Including those positions injects a large, content-free, position-correlated component into the velocity and contaminates both $F_k$ and $r_k$. We therefore score only *content* tokens: a position $i$ is scored iff the realized token $x_i$ is not pure whitespace/punctuation/structural markup (after detokenization, not in a small stop-set such as `{space, newline, ".", ",", ";", ":", "(", ")"}`) and is not a special/template token. Non-content positions are still consumed — the EMA and $\bm{g}_{\text{prev}}$ advance through them so the trajectory is unbroken — but emit no $F_k$/$r_k$ and are excluded from pooling. This is the single highest-value correctness decision in the pipeline; omitting it was the main gap in the prior draft. <span class="sans-serif">\[DERIVED\]</span>/measurement hygiene; its effect size is itself reported (scored-all vs. content-only, as an ablation).

<div id="ass:smallstep" class="assumption">

**Assumption 1** (Small-step validity). *Token-to-token displacements are small enough that (i) the finite-difference velocity approximates the e-tangent and (ii) the metric is approximately constant across one step, so evaluation at $p_{k-1}$ suffices. This fails across hard discontinuities (e.g. span boundaries with abrupt distribution jumps); see Section <a href="#sec:tests" data-reference-type="ref" data-reference="sec:tests">11</a> for the diagnostic and fix. <span class="sans-serif">\[DERIVED\]</span> assumption made explicit.*

</div>

For span $s$ covering scored content positions $\{k : \text{token}(k)\in[a_s, b_s)\}$, let $n_s$ be the number of such positions:
$$\begin{aligned}
  \text{pooled force} \quad \bar F_s &= \frac{1}{n_s}\sum_{k\in s} F_k,
  &\text{(mean over scored content steps)} \\
  \text{pooled redirection} \quad R_s &= \frac{\sum_{k\in s}\langle \bm{v}_k^{\perp},\bm{v}_k^{\perp}\rangle_{p_{k-1}}}{\sum_{k\in s}\langle \bm{v}_k,\bm{v}_k\rangle_{p_{k-1}}+\varepsilon}.
  &\text{(energy ratio over content steps)}
\end{aligned}$$
The energy-ratio form prevents a single near-stationary step from contributing a noisy high ratio. $R_s$ and the sign $C_s$ (Section <a href="#sec:sign" data-reference-type="ref" data-reference="sec:sign">5.5</a>) are the primary predictors; $\bar F_s$ is secondary. If $n_s=0$ the span emits NaN and is dropped before correlation. <span class="sans-serif">\[STATISTICS\]</span>/engineering for granularity matching (not a physical law).

# Data Generation: Perturbed-Resampling Labels and Stored Traces

We do not rely on a pre-existing released dataset; we generate labels ourselves. The *generation mechanism* — perturbed span re-rollouts producing paired real/perturbed answer distributions — is adapted from a parallel forward-time-detection protocol ; we borrow that procedure only, and make our own choices of model, label divergence, granularity, and evaluation (noted as such below). Generating it ourselves gives full control of granularity, stores the exact generating traces (so the token alignment is exact rather than reconstructed), and yields a clean, black-box ground truth that is independent of every forward-pass signal we compute — which is what makes a correlation with it non-circular.

#### Pipeline.

For a base trace, (1) generate the trace and *store its token ids, the detokenized text, and per-position logits-or-the-trace-for-reforwarding*; (2) segment it into spans (Section <a href="#sec:granularity" data-reference-type="ref" data-reference="sec:granularity">7.0.0.4</a>); (3) for each span $s$, build a “real” pool and a “different-meaning” pool by the resample-and-filter procedure of Section <a href="#sec:perturb" data-reference-type="ref" data-reference="sec:perturb">7.0.0.3</a>; (4) aggregate the final-answer distributions of the two pools and compute the label; (5) store everything keyed by (trace id, span id). The independent statistical unit is the trace/problem, not the span: within-trace spans are correlated, so all bootstrap resampling is over traces.

#### Label (our choice, not borrowed).

The generator yields, per span, two final-answer distributions: $\hat p_{\text{ans}\mid s}$ (real) and $\hat p_{\text{ans}\mid s'}$ (perturbed). The label is a divergence between them. We default to the original Thought Anchors definition, the KL divergence between the two answer distributions ; for verifiable tasks with a discrete answer, total variation $A_s = \tfrac12\sum_a |\hat p(a\mid s)-\hat p(a\mid s')|$ is an equally valid alternative. The choice of divergence is ours and is a property of the ground truth, not of the predictor, which never sees it; we report against whichever was used and, budget permitting, both.

#### How the semantic alternatives are produced (settled): resample-and-filter.

We do *not* craft perturbations by paraphrasing. Following the original Thought Anchors counterfactual-resampling method , the “different-meaning” alternatives are the model’s *own* natural variation, partitioned after the fact:

1.  **Resample from the slot start.** For span $s$ (occupying one window slot), sample many continuations from the prefix ending *just before* $s$, so the model regenerates the slot itself and everything after it. This yields a spread of candidate spans in that slot, all in-distribution (the model actually would have produced them).

2.  **Embed and split by similarity.** Embed each regenerated slot and the original $s$ with a sentence embedder (Sentence-BERT or a current equivalent) and compute cosine similarity. The threshold is **data-driven, not hand-set**: a regenerated slot is “same-meaning” if its similarity to $s$ is above the dataset-wide median pairwise similarity, and “different-meaning” if below . This keeps the cutoff parameter-free and comparable across spans.

3.  **Two pools, one label.** The real pool is the above-median (same-meaning) continuations; the different pool is the below-median ones. The label is the divergence between the two pools’ final-answer distributions (Label paragraph above). Target $R \approx 100$ usable continuations per pool (the “gold” resampling level); because the split is post hoc, oversample enough total rollouts that both pools reach $R$.

Two conventions to match exactly, because they change what is measured: resample *from the start of the perturbed slot* (regenerating $s$), not from after a fixed replacement — the alternative is sampled, never inserted; and define “different” by the *dataset-median* cosine split rather than an absolute cutoff. We adapt only the unit: Thought Anchors split at sentence slots, whereas our slot is a boundary-respecting window (Section <a href="#sec:granularity" data-reference-type="ref" data-reference="sec:granularity">7.0.0.4</a>), so similarity is computed window-to-window.

<div class="honestbox">

**Why resample-and-filter, not paraphrasing.** A sampled alternative is in-distribution — the model could have generated it — so the label measures the model’s own counterfactual, not a reaction to an inserted phrasing it would never have produced. A crafted paraphrase is an out-of-distribution insertion that risks inducing behavior unrelated to importance, and it bakes a second model’s quirks into the ground truth. The cost is rollouts: the split is post hoc, so enough continuations must be sampled that both pools fill, which is the expensive part of the pipeline.

</div>

<div class="honestbox">

**Paraphrase route: a deferred alternative for larger models / tighter budgets.** Resample-and-filter spends its compute on *many* continuations per slot to populate both pools. As the target model grows or the rollout budget tightens, that cost can dominate. In that regime, a **controlled paraphrase** alternative becomes attractive: generate one (or a few) explicitly meaning-changed rewrites of $s$ and resample after each, trading the in-distribution guarantee for far fewer rollouts per slot. We note this as a *future* substitution, not the default, and flag the price: paraphrase insertions are off-distribution and import a generator’s biases, so any run using them must re-validate against a resample-and-filter subset before its labels are trusted. The predictor and geometry are unchanged either way — only the source of $s'$ differs.

</div>

#### Granularity (our choice): boundary-respecting windows.

The generator perturbs whole spans, but is indifferent to how spans are defined; the geometry and pooling (Section <a href="#sec:tokens-to-spans" data-reference-type="ref" data-reference="sec:tokens-to-spans">6</a>) aggregate over *arbitrary* spans. We define spans by a **two-stage** rule that respects linguistic structure rather than cutting on a raw token count:

1.  **Parse first.** Segment the trace into sentences and clauses: split on sentence terminators (`. ? !`, newlines) and on intra-sentence clause boundaries (commas, semicolons, colons, and discourse connectives such as *so, therefore, thus, then, next, because*). Each piece is a coherent syntactic unit.

2.  **Window within, never across.** Any unit longer than the window budget $t$ tokens is sub-split into consecutive $\le t$-token windows that lie *entirely inside* that unit; units of $\le t$ tokens are kept whole. A trailing sub-window shorter than a floor $t_{\min}$ is merged into the previous window of the same unit. Windows never straddle a sentence or clause boundary.

This matters because a window that straddled a boundary would force the generator to perturb half of one clause and half of the next, producing a semantically incoherent replacement and a meaningless label; restricting windows to within a parsed unit keeps every perturbation a swap of a coherent fragment. The window budget $t$ (and floor $t_{\min}$) are knobs we set; the parse boundaries are not negotiable. For long-clause traces $t$ controls localization granularity, while short clauses simply remain whole.

#### Why storing traces matters for us specifically.

Because we re-forward the stored trace under teacher forcing to read logits (Section <a href="#sec:pulled" data-reference-type="ref" data-reference="sec:pulled">8</a>), the token sequence we score must be *bit-identical* to the one that produced the labels. Storing the generated token ids (not just the text) removes the retokenization ambiguity — chat-template insertion, `<think>` tag handling, whitespace normalization — that would otherwise desynchronize the logits from the labeled spans. The stored token ids are the source of truth; the text and offsets are derived from them.

# What Is Pulled From the Model and the Dataset

## From the model

The *only* model output required is per-position logits $\bm{\ell}_i$. No hidden states, no attention, no gradients. We obtain the on-trajectory distributions by *teacher forcing* the **stored token ids** of a generated trace (Section <a href="#sec:datagen" data-reference-type="ref" data-reference="sec:datagen">7</a>): feed the recorded token sequence through the model and read $\bm{\ell}_i$ at each position. Indexing convention, to be obeyed without exception:

<div class="center">

</div>

The target model is our choice; it need only be the *same* model used for data generation, so labels and signals remain commensurable (we fix one model per study). The natural choices are the R1-Distill reasoning models used in the original Thought Anchors work .

<div class="honestbox">

**Model scale is a variable that bears on whether the signal exists at all.** Whether a momentum signal is measurable is not scale-invariant. On a heavily distilled small model (e.g. 1.5B), output distributions are often sharp and the token-to-token trajectory jerky, which strains the small-step assumption (Assumption <a href="#ass:smallstep" data-reference-type="ref" data-reference="ass:smallstep">1</a>) and would surface as the boundary-clustering failure of Test 3. We therefore **recommend first validating on a mid-size model** (where the small-step regime is safest), and only then testing transfer down to a small model — rather than concluding from a small-model null that the signal does not exist. A negative result is informative only at a scale where the measurement assumptions hold. Compute permitting, run at least two scales and report the scale dependence; it is itself a finding about where reasoning momentum is geometrically clean.

</div>

The CoT lives inside `<think>`…`</think>`; only tokens inside that span are scored. <span class="sans-serif">\[EXACT\]</span> (teacher forcing on stored token ids reproduces the on-trajectory distributions).

## From the dataset

Because we generate the data (Section <a href="#sec:datagen" data-reference-type="ref" data-reference="sec:datagen">7</a>), every field is known by construction and stored together: (1) the trace token ids and detokenized text; (2) the generation temperature $T$ (required, since the Fisher metric is evaluated at the distribution actually sampled from); (3) the span boundaries (as token-id index ranges *and* character spans); (4) the per-span label $A_s$ (the divergence between paired answer distributions; TV or KL); (5) the correct/incorrect flag per trace. No schema guessing is required — the prior draft’s open question about released-dataset field names is moot once we own the generator.

## The alignment join

Because we store spans as **token-id index ranges** alongside character spans (Section <a href="#sec:datagen" data-reference-type="ref" data-reference="sec:datagen">7</a>), the join is direct: the span’s scored positions are simply the content tokens within its stored half-open index range $[a_s, b_s)$. The character-offset mapping (`return_offsets_mapping=True`) is retained only as a cross-check that the stored index ranges and the detokenized text agree. The failure mode that dominated when consuming a *released* dataset — re-splitting text with an independent sentence splitter, producing a different span count, and desynchronizing every label after the first mismatch — cannot occur here because we never re-segment: the spans used for labeling are the spans used for scoring, by id. We still assert per-trace span-count and boundary equality before computing anything. <span class="sans-serif">\[STATISTICS\]</span>/data hygiene.

# Reference Implementation (Streaming; Identical Offline and Online)

The loop keeps only the previous scaled-logit vector, the EMA, and per-token scalars; it never materializes the full $[L,V]$ logit tensor. This is simultaneously the memory-optimal offline form and exactly the online inference form (during generation the logits arrive one at a time; the lag is one token, for the span-boundary pooling).

    import torch

    def fisher_cov(p, u, w):                 # exact categorical Fisher inner product, O(V)
        eu = (p * u).sum(); ew = (p * w).sum()
        return (p * u * w).sum() - eu * ew

    def score_trace(logits, token_ids, is_content, T_gen, beta=0.97, eps=1e-8):
        # logits:     [L, V] float32; logits[i] = distribution over token i+1
        # token_ids:  [L]    the stored generated ids (source of truth for alignment)
        # is_content: [L] bool; False for whitespace/punct/structural/special tokens
        # DEFAULT: content-only EMA -- velocity is between consecutive CONTENT positions,
        # and vbar is updated only on content steps (recommended; see paper).
        L = logits.shape[0]
        F     = torch.full((L,), float('nan'))
        perp  = torch.zeros(L); tot = torch.zeros(L)
        cproj = torch.full((L,), float('nan'))  # projection coeff c_i (sign of along-track)
        vbar    = None                           # EMA velocity entering the next content step
        g_prevc = None                           # scaled logits at the previous CONTENT position
        for i in range(L):
            g = logits[i] / T_gen
            if not is_content[i]:
                continue                         # skip formatting: no score, no EMA update
            if g_prevc is not None:
                pb = torch.softmax(g_prevc, -1)          # base point = previous content dist
                v  = g - g_prevc                         # content-to-content velocity
                if vbar is not None:
                    d  = v - vbar                        # force direction (causal)
                    t  = fisher_cov(pb, v, v).clamp_min(0)
                    bb = fisher_cov(pb, vbar, vbar)
                    c  = fisher_cov(pb, v, vbar) / (bb + eps)
                    vp = v - c * vbar
                    perp[i]  = fisher_cov(pb, vp, vp).clamp_min(0)
                    tot[i]   = t
                    F[i]     = fisher_cov(pb, d, d).clamp_min(0).sqrt()
                    cproj[i] = c
                vbar = v if vbar is None else beta * vbar + (1 - beta) * v
            g_prevc = g
        return F, perp, tot, cproj
        # ABLATION (mixed EMA): advance g_prev and vbar through EVERY token, scoring
        # only content positions. Reported as a comparison; expected to underperform.

    def pool_to_spans(F, perp, tot, cproj, spans, eps=1e-8):
        # spans: list of half-open (a_s, b_s) token-id index ranges (stored at gen time)
        out = []
        for (a, b) in spans:
            m = torch.isfinite(F[a:b])           # scored content positions in the span
            if m.sum() == 0:                     # n_s == 0
                out.append(dict(force=float('nan'), R=float('nan'), C=float('nan')))
                continue
            Fbar = F[a:b][m].mean().item()
            e    = tot[a:b][m]                    # energy weights for sign pooling
            R    = (perp[a:b][m].sum() / (e.sum() + eps)).item()
            C    = ((e * cproj[a:b][m]).sum() / (e.sum() + eps)).item()  # energy-wtd sign
            out.append(dict(force=Fbar, R=R, C=C))  # C>0 accelerator, C<0 brake
        return out

#### Numerical and edge-case decisions.

Compute every covariance in float32 even under an fp16/bf16 forward pass: the subtraction $\sum p u w - (\sum p u)(\sum p w)$ loses precision otherwise. Floor norms with $\varepsilon$. During EMA warmup ($\bar{\bm{v}}\approx 0$) the projection collapses and $r_k\to 1$ spuriously; drop the first span of each trace. On frozen deterministic stretches ($\mathrm{tot}_k\approx 0$) the ratio denominator is floored, redirection reads $0$, and the sign $C$ is undefined (reported NaN). <span class="sans-serif">\[DERIVED\]</span>/numerics.

# Validation Design (Pure Statistics)

#### Headline quantity: partial Spearman, explicitly.

The headline is the *partial* rank correlation between the predictor $R_s$ and the label $A_s$, controlling for span index and span length. Procedure, stated so it is implementable without a judgement call:

1.  Rank-transform each of $A_s$, $R_s$, span index, and span length (across all spans in the analysis set).

2.  Regress $\mathrm{rank}(A_s)$ on $\{\mathrm{rank}(\text{index}), \mathrm{rank}(\text{length})\}$ by OLS; keep residuals $\tilde A$.

3.  Regress $\mathrm{rank}(R_s)$ on the same controls; keep residuals $\tilde R$.

4.  The partial Spearman is the Pearson correlation of $\tilde A$ and $\tilde R$.

Equivalent one-liner:

``` python
pingouin.partial_corr(data, x='R', y='A', covar=['idx','len'], method='spearman')
```

Repeat with $\bar F_s$ and, for the directional claim, with the signed $C_s$. <span class="sans-serif">\[STATISTICS\]</span>.

#### Significance by trace-level bootstrap.

Within-trace spans are correlated, so the independent unit is the *trace*, not the span. Confidence intervals come from resampling traces with replacement (not spans), recomputing the partial Spearman each draw, and reporting the $2.5/97.5$ percentiles over $\geq 1000$ draws. Reporting span-level CIs would overstate significance by treating correlated spans as independent. <span class="sans-serif">\[STATISTICS\]</span>.

#### The baseline the physics must beat.

Pooled token surprise $-\log p_i(x_i)$ over the span’s scored content tokens, under the identical controls and bootstrap. For an external reference point, also report the published receiver-head attention predictor on the same traces. If the Fisher-geometry score does not beat token surprise under the same controls, the geometry is not earning its place. <span class="sans-serif">\[STATISTICS\]</span>.

#### The answer region, defined programmatically.

“Final-answer-emission region” is everything from the first token after `</think>` onward; it is excluded from the primary analysis and reported separately. Optionally also guard the final fraction of the think block if confidence-saturation is observed in calibration. This is a token-id test on the stored trace, not a heuristic. <span class="sans-serif">\[STATISTICS\]</span>.

#### Protocol.

Develop on *correct* traces (cleaner signal), test transfer to incorrect traces and to an out-of-distribution held-out reasoning type (e.g. hold out a different task family used only for generalization testing, never for tuning). Sweep $\beta\in[0.9,0.99]$ on a development split and select by held-out partial correlation; report the curve, since the best $\beta$ is itself a finding about the memory length of reasoning. <span class="sans-serif">\[STATISTICS\]</span> / <span class="sans-serif">\[DERIVED\]</span> ($\beta$).

# Tests, Outcomes, and Pre-Registered Responses

We state each test, then — before seeing data — what each outcome would mean and how the implementation would change. Pre-registering the responses is what keeps the exercise honest.

## Test 1: Does redirection predict importance (with controls)?

**Measure:** partial correlation of $R_s$ with importance, controlling for position and length, on correct traces, reasoning region only.

<div class="description">

The core hypothesis survives. *Response:* freeze the pipeline; proceed to Test 2 (structure) and Test 4 (transfer). No code change.

The signal is real but not geometric — it is just “surprising tokens.” *Response:* the physics framing is not justified by the data. Demote the claim to “a Fisher-geometry view motivates a surprise-like feature” and report honestly that the geometry added nothing measurable. Implementation change: keep surprise as the predictor; retain $R_s$ only if it adds incremental signal in a joint model.

The apparent signal was position drift. *Response:* this is the predicted failure of a local detector. Report it as the “relational-criticality dominates” result (Test 5 below). No rescue by adding features that re-import position.

</div>

## Test 2: Does the sign structure recover the taxonomy?

**Measure:** using the signed pooled projection $C_s$ (Section <a href="#sec:sign" data-reference-type="ref" data-reference="sec:sign">5.5</a>), are brake spans ($C_s<0$) enriched for backtracking (“Wait,” “Actually”) and accelerator spans ($C_s>0$, high $\bar F_s$) enriched for plan-continuation, per the reasoning-step taxonomy? Test against span-level category labels (LLM auto-labeled, following the Thought Anchors taxonomy ).

<div class="description">

This is the novel contribution: a present-tense, sign-bearing signal the resampling label cannot produce. *Response:* elevate this to a headline result independent of the magnitude correlation in Test 1.

The decomposition is not capturing the linguistic structure we hypothesized. *Response:* report the magnitude result (if any) without the directional claim; do not over-interpret $\mathrm{sign}(C_s)$ as “brake $=$ backtracking.”

</div>

## Test 3: Is the small-step assumption holding at boundaries?

**Measure:** distribution of large forces by within-span token position; check for clustering at span-initial tokens (where the distribution can jump after a clause or sentence boundary).

<div class="description">

Assumption <a href="#ass:smallstep" data-reference-type="ref" data-reference="ass:smallstep">1</a> is violated by distribution jumps at span starts. *Response (implementation):* reduce the EMA’s sensitivity to a single post-boundary jump by either (a) resetting/down-weighting the EMA state at parsed sentence/clause boundaries so a boundary jump does not masquerade as a force, or (b) excluding the first $k$ content tokens after each boundary from pooling. Both are concrete, pre-planned changes that keep the EMA inertial model intact rather than replacing it. <span class="sans-serif">\[DERIVED\]</span>.

Assumption holds at the operating scale; no change.

</div>

## Test 4: Does it transfer to incorrect traces and online use?

**Measure:** re-fit nothing; apply the frozen pipeline to held-out incorrect traces and to streaming (one-token-lag) computation.

<div class="description">

The law is not an artifact of correct-trace structure or of offline batching. *Response:* report online viability directly (the streaming loop is already the deployment form).

Anchors in failing reasoning have a different geometric signature. *Response:* report the asymmetry as a finding; do not paper over it by pooling correct and incorrect.

</div>

## Test 5: How large is the irreducible blind spot?

**Measure:** fraction of high-importance spans with low local force/redirection (relational anchors with no present-tense signature).

<div class="description">

Importance is mostly locally expressed; the local detector is near-sufficient. Strong positive result.

Much of importance is relational to the future and is unknowable online *by construction*. *Response:* this is not a failure of our method but a measurement of the limit of *any* online method. Report the blind-spot fraction as the headline quantification of “how much of anchor importance is present-tense physics versus hindsight bookkeeping.”

</div>

## Test 6: $\beta$ sensitivity and the connection choice

**Measure:** the partial-correlation-versus-$\beta$ curve; and, as an ablation, recompute with an arc-length (Fisher) time reparameterization instead of the token clock (the time-parameter convention catalogued in Section <a href="#sec:catalog" data-reference-type="ref" data-reference="sec:catalog">13</a>).

<div class="description">

The reasoning trajectory has a characteristic memory length $\approx 1/(1-\beta^\star)$; report it.

Either no inertia matters ($\beta^\star\to 0$, the law reduces to single-step acceleration) or the signal is insensitive to smoothing; report which, as it bears directly on whether “momentum” (as opposed to instantaneous acceleration) is the right description.

The token clock is doing real work; this is informative about the law, not a bug. Report both parameterizations.

</div>

# What the Results Would Mean

- A clean positive in Tests 1, 2, 4 with a small blind spot (Test 5) supports the strong reading: reasoning trajectories obey an information-geometric momentum law and anchors are local force events, detectable online.

- A positive Test 2 with a weak Test 1 supports a narrower but still novel reading: the geometry does not predict magnitude well, but its *sign* structure exposes accelerator/brake roles invisible to resampling.

- A large blind spot (Test 5) with otherwise positive results bounds the enterprise: local geometry captures the locally-expressed anchors and provably cannot capture the relational ones; the bound is the contribution.

- A null after controls (Test 1) with the surprise baseline also null means importance is not locally expressed in the output distribution at all — a clean negative that still maps the territory.

None of these outcomes is an embarrassment, because the apparatus was built to make the conditional (“*if* the trajectory obeys this law”) a fair and powerful test rather than a foregone conclusion.

# Catalog of Provenance

<div class="center">

| **Ingredient**                                                                                                                                               | **Status**                                     | **Note**                                                                      |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------|:------------------------------------------------------------------------------|
| Output distribution as state; invariance to representation                                                                                                   | <span class="sans-serif">\[EXACT\]</span>      | functions of distributions only                                               |
| Fisher metric $=$ mass; $\mathrm{Cov}_p$ closed form <a href="#eq:fisher-cov" data-reference-type="eqref" data-reference="eq:fisher-cov">[eq:fisher-cov]</a> | <span class="sans-serif">\[EXACT\]</span>      | Čencov uniqueness; $O(V)$                                                     |
| Gauge direction null in metric                                                                                                                               | <span class="sans-serif">\[EXACT\]</span>      | no centering needed                                                           |
| e-connection is flat $\Rightarrow$ exact discrete transport                                                                                                  | <span class="sans-serif">\[EXACT\]</span>      | makes EMA bookkeeping exact                                                   |
| Tangential/normal (reinforce/redirect) split <a href="#eq:redir" data-reference-type="eqref" data-reference="eq:redir">[eq:redir]</a>                        | <span class="sans-serif">\[EXACT\]</span>      | curvature decomposition                                                       |
| Sign of along-track motion $\mathrm{sign}(c_i)$ (accel/brake)                                                                                                | <span class="sans-serif">\[EXACT\]</span>      | sign of tangential acceleration                                               |
| “brake $\Rightarrow$ linguistic backtracking”                                                                                                                | *hypothesis*                                   | tested in Test 2                                                              |
| Content-token probe (skip formatting positions)                                                                                                              | <span class="sans-serif">\[DERIVED\]</span>    | measurement hygiene; ablated                                                  |
| Choice of e-connection ($\alpha{=}1$) over Levi-Civita ($\alpha{=}0$)                                                                                        | <span class="sans-serif">\[CONVENTION\]</span> | info-geometric, not Newtonian                                                 |
| Token index as the time parameter                                                                                                                            | <span class="sans-serif">\[CONVENTION\]</span> | arc-length is the alternative (Test 6)                                        |
| Finite-difference velocity; metric at $p_{k-1}$                                                                                                              | <span class="sans-serif">\[DERIVED\]</span>    | leading order; needs small steps                                              |
| EMA as inertial-direction estimator; $\beta$; content-only by default                                                                                        | <span class="sans-serif">\[DERIVED\]</span>    | variance reduction; mixed-update is an ablation                               |
| Force $=$ Fisher norm of deviation <a href="#eq:force" data-reference-type="eqref" data-reference="eq:force">[eq:force]</a>                                  | <span class="sans-serif">\[DERIVED\]</span>    | $\lVert m\bm a\rVert$ with e-acceleration                                     |
| Span pooling (mean force, energy-ratio redirection, energy-wtd sign)                                                                                         | <span class="sans-serif">\[STATISTICS\]</span> | granularity matching                                                          |
| Semantic alternatives by resample-and-filter (median-cosine split)                                                                                           | <span class="sans-serif">\[STATISTICS\]</span> | Thought Anchors method ; paraphrase deferred to large-model/low-budget regime |
| Perturbed-resampling *generation mechanism*; stored traces                                                                                                   | <span class="sans-serif">\[STATISTICS\]</span> | generator borrowed ; label/model/granularity ours                             |
| Partial Spearman, trace-level bootstrap, baselines                                                                                                           | <span class="sans-serif">\[STATISTICS\]</span> | fair-test machinery                                                           |
| “Force predicts anchor importance”                                                                                                                           | *hypothesis*                                   | the empirical bet; not a theorem                                              |

</div>

# Caveats Restated

This momentum is not a conserved quantity: there is no action principle generating the token trajectory, hence no Noether charge and no dynamical law — only kinematics measured along a given path. The geometry defines the quantities exactly; it does not certify that they predict anchors. The e-connection choice means we test an information-geometric law, not Newtonian mechanics. Token-time is a convention. The local detector has an irreducible blind spot for relational anchors. Every one of these is a deliberate, named limitation rather than a hidden one.

<div class="thebibliography">

9
P. C. Bogdan, U. Macar, N. Nanda, and A. Conmy.
*Thought Anchors: Which LLM Reasoning Steps Matter?*
arXiv:2506.19143, 2025.
*Forward-Time Detection of Thought Anchors: Run-Ready Experimental Protocol.*
Companion protocol (data-generation pipeline, perturbed-resampling labels, curriculum), 2026.
S. Amari.
*Information Geometry and Its Applications.*
Springer, 2016.
N. N. Čencov.
*Statistical Decision Rules and Optimal Inference.*
American Mathematical Society, 1982.
V. I. Arnold.
*Mathematical Methods of Classical Mechanics.*
Springer, 2nd ed., 1989.

</div>
