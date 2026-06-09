# The load-bearing claim

Every signal is a cheap forward-pass quantity; the target (resampling importance) is a
counterfactual-on-the-text quantity. A span can be important only if its content (a) *reaches*
the answer-producing computation and (b) the answer actually *depends* on it. Forward-pass
signals measure the *capacity* for importance; resampling measures *realized* importance.
The gap—redundancy, robustness, nonlinearity—caps achievable correlation below 1 and is
unknowable from the forward pass. Whether the ceiling sits above $0.22$ is the empirical question,
measured *first*, before any combiner is built.

# Staged execution: experiment tiers

Everything in this protocol is something we intend to try *eventually*; none of it is meant to run
at once. Execution is *tiered and escalating*: begin with the smallest run that is plausibly
sufficient, and add machinery only as evidence demands. After each tier the path forks—*fix*
features if the tier underperformed, *improvement* features if it succeeded—and we climb until
either every necessary tier has run and none beats the $0.22$ baseline (a clean negative result), or the
working signal has every upgrade it supports. This section changes no concept, definition, or
experiment; it only fixes the order in which they are exercised.

#### Foundation (present in *every* tier—correctness, not features).

The architecture-correct extraction and the label pipeline are prerequisites, never optional, and are
what “critical failure points still carry their fix” refers to: pre-norm/RMSNorm additivity and GQA
head mapping (§<a href="#sec:background" data-reference-type="ref" data-reference="sec:background">3</a>), eager-attention $\alpha$ capture, contribution-based top-$K$
pruning ($\widehat E$), and the per-position local backward for any gradient feature
(§<a href="#sec:gradmath" data-reference-type="ref" data-reference="sec:gradmath">5</a>); clause-respecting spans (§<a href="#sec:granularity" data-reference-type="ref" data-reference="sec:granularity">6.2</a>) and resample-and-filter
labels $A$ (§<a href="#sec:data" data-reference-type="ref" data-reference="sec:data">6</a>; ), scored on Phase 0–1 with the bAbI supporting-fact construct
check (§<a href="#sec:curriculum" data-reference-type="ref" data-reference="sec:curriculum">6.4</a>). These hold from Tier 1 onward regardless of results.

#### Tier 0 — warm-up and cost gates (run before committing the gold-labeling budget).

The Tier-1 *signals* are nearly free, but the $R{=}100$ resample-and-filter labeling they are
scored against is the dominant cost of the project—so it is gated behind cheaper checks, climbed in
order and aborting early if one fails:

1.  **Reuse check (free).** If ’s released resampling dataset covers the
    target model, take their gold labels directly—zero labeling cost and an exact $0.22$ comparison.

2.  **bAbI construct check (free, no resampling).** Extract the Tier-1 signals on bAbI traces and
    test whether they rank the annotated supporting facts above distractors. The cheapest science gate: if
    the signals cannot track on-path facts on the easiest dataset, the labeling budget is likely wasted.
    It is a fail-fast *leading indicator*, not a strict gate—supporting-facts is a coarser label
    than resampling importance, so passing does not guarantee beating $0.22$ and failing is a strong
    warning rather than a proof.

3.  **Small-$R$ pilot.** A low-$R$ ($\approx$<!-- -->10–20) batch on a problem subset gives a noisy
    early read on $\rho$ against (A) and doubles as the $t$/$K$/band-$\mathcal B$ localization Phase 0
    already requires; abort before full labeling if even the noisy pilot shows nothing.

4.  **Full $R{=}100$ gold labeling**—only after the cheaper gates pass.

Only once gold labels exist does the Tier-1 decision below apply.

#### Tier 1 — minimal marginals (the fulcrum; the run most likely to succeed).

The cheapest forward-only single-signal marginals—no backward pass, no chaining, no combiner:
propagated routing $P_i$ (Eq. <a href="#eq:katz" data-reference-type="ref" data-reference="eq:katz">[eq:katz]</a>; Trial 1, the closest analogue to prior white-box work and
the first thing that must clear $0.22$), own computation $G_i$ (Eq. <a href="#eq:G" data-reference-type="ref" data-reference="eq:G">[eq:G]</a>; Trial 2, the orthogonal
thesis), and crude one-hop computation-weighted outflow $P'_i$ (Eq. <a href="#eq:Pprime" data-reference-type="ref" data-reference="eq:Pprime">[eq:Pprime]</a>; Trial 4 without the
gradient). *Decision (two-pronged, matching the central bet):* (i) does any marginal beat $0.22$
against (A)—is there exploitable forward signal at all; and (ii) does the computation axis ($G_i$ or
$P'_i$) beat or add over routing ($P_i$)—the test of the thesis itself, since a win carried by $P_i$
alone is only a cleaner *routing* result, not evidence that computation carries orthogonal signal.

#### Tier 2 — the first fork.

*Thesis supported (both prongs clear):* add the gradient-scaled outflow $D_i$ (Eq. <a href="#eq:grad" data-reference-type="ref" data-reference="eq:grad">[eq:grad]</a>;
Trial 3) for principled credit assignment over crude $P'$—its payoff over $P'$ being the
Trial 3-vs-4 test, not assumed—and the interpretable linear combiner (Trial 5, ridge / PCR), keeping
the computation signal only if it adds over $P_i{+}G_i$; the coefficients are the finding.
*Routing clears $0.22$ but the computation axis does not, or signals are position-dominated (fix):*
a one-hop computation null has three escapes to rule out before declaring the axis inert—a position
confound (add residualization against the position-only null); a bad first-order approximation
(substitute the faithful $D^{\mathrm{IG}}$, Eq. <a href="#eq:ig" data-reference-type="ref" data-reference="eq:ig">[eq:ig]</a>); or *wrong reach*—the axis may carry
signal only *transitively*, so the chained variant (Eq. <a href="#eq:chained" data-reference-type="ref" data-reference="eq:chained">[eq:chained]</a>, otherwise a Tier-3
improvement) is pulled forward here as a rescue, since the design’s own expectation is that the
computation signal may live in the multi-hop form. Only after these does a clean routing-only result
stand (still publishable, but not the bet).

#### Tier 3 — reach and ceiling (improvement) / blow-up control (fix).

Add the chained variants $\widetilde P'_i,\widetilde D_i$ (Eq. <a href="#eq:chained" data-reference-type="ref" data-reference="eq:chained">[eq:chained]</a>) with the $\lambda$
sweep—the reach sub-analyses of Trials 3–4, testing whether computational influence is
transitive—and the nonlinear MLP combiner (Trial 5(ii)) as an accuracy ceiling reported beside the
interpretable model. The chained signals carry the blow-up risk; if they underperform or turn
position-dominated, this tier’s fix branch is the remediation ladder (§<a href="#sec:contingency" data-reference-type="ref" data-reference="sec:contingency">9</a>).

#### Tier 4 — causal validation (only once a detector tracks anchors).

Convert correlation into causation with the interventions (§<a href="#sec:followups" data-reference-type="ref" data-reference="sec:followups">10</a>): Q2 noise-fill
(sufficiency/necessity, the most convincing) first, then Q1 (forced/speculative anchoring), Q3
(math/state heads), and the targeted DecompX probe on confirmed anchors. Q4 stays demoted.

#### Tier 5 — best-possible version (scale-up on a validated detector).

Extend across the full curriculum (Phases 2–4 through SCONE), add the GSM8K out-of-distribution
generalization test, and calibrate the Lyapunov proxy (B) against (A) to scale labeling
(§<a href="#sec:groundtruth" data-reference-type="ref" data-reference="sec:groundtruth">7</a>). This is the upper end: all upgrades to the working form.

#### Termination.

Stop when either every necessary tier has run and none beats $0.22$ (the
central bet is falsified—a clean negative result), or the working signal has received every upgrade it
supports and the next tier’s marginal gain no longer justifies its cost.

# Background and technique provenance

#### Residual stream (additive view) .

Every component writes additively into a shared residual stream; attention heads act independently.
Because attention is linear in the value vectors once weights are fixed, the attention update at
layer $\ell$, head $h$ decomposes by source token.

#### Per-source delivered force .

Attention weight alone misstates information flow: a high-weight token with small transformed-value
norm contributes little. Folding in the output projection, the delivered residual contribution from
source $s$ to target $t$ is
$$\mathbf{F}^{\ell}_{s\to t}=\sum_h \mathbf{W}_O^{h,\ell}\!\left(\alpha^{\ell,h}_{t,s}\,\mathbf{v}^{\ell,h}_s\right),
\qquad
\Delta^{\ell,\mathrm{attn}}_t=\sum_{s\le t}\mathbf{F}^{\ell}_{s\to t}.
\label{eq:F}$$

#### Architecture note: pre-norm RMSNorm and GQA (not the post-LN/MHA setting).

The original residual-accounting work targets *post-norm*
LayerNorm transformers, where one linearizes the post-attention LayerNorm to recover additivity (the
“post-LN delivered vector, additive up to a bias $\beta$”). Our target model is *pre-norm* with
RMSNorm, which changes this in two ways we adopt throughout:

- **No post-attention norm $\Rightarrow$ exact additivity, no bias.** In a pre-norm block,
  $\mathbf{h}'=\mathbf{h}+\mathbf{W}_O(\mathrm{attn}(\mathrm{RMSNorm}(\mathbf{h})))$: the attention output is written into the residual
  *with no normalization after it*. So $\mathbf{F}^{\ell}_{s\to t}$ is *exactly* additive into the
  residual (Eq. <a href="#eq:F" data-reference-type="ref" data-reference="eq:F">[eq:F]</a> holds with no bias term and no linearization needed for the edge
  $E_{s\to t}$). RMSNorm enters only when a *downstream* block reads the residual; for the gradient
  feature (§<a href="#sec:gradmath" data-reference-type="ref" data-reference="sec:gradmath">5</a>) that read-norm is differentiated through automatically, and if the
  chained routing graph ever needs it explicitly, the linearization is $L(u)=\mathrm{diag}(\gamma/\mathrm{rms}(\mathbf{h}'))\,u$
  with the denominator frozen at the operating point—*no bias*, unlike LayerNorm.

- **Grouped-query attention.** The model has $H$ query heads but fewer KV heads; query head
  $h$ shares the value $\mathbf{v}^{kv(h)}_s$ of its group’s KV head. Eq. <a href="#eq:F" data-reference-type="eqref" data-reference="eq:F">[eq:F]</a> must use $\alpha^{h}$ per
  query head with $\mathbf{v}^{kv(h)}$ from the shared KV head, and $\mathbf{W}_O^{h}$ the query-head output block;
  treating the heads as fully independent mis-attributes $\mathbf{F}$.

Information Flow Routes carry the per-source construction to
decoder-only LLMs, folding $\mathbf{W}_{OV}=\mathbf{W}_V\mathbf{W}_O$ into per-(source,head) contributions; we retain the vector
rather than collapsing to a norm prematurely.

#### Scalar edges and span aggregation.

$$E_{s\to t}=\sum_{\ell=1}^{L}\left\lVert \mathbf{F}^{\ell}_{s\to t} \right\rVert_2,
\qquad
E_{i\to j}=\sum_{s\in\mathcal S_i}\sum_{t\in\mathcal S_j}E_{s\to t},
\label{eq:edge}$$
where $\mathcal S_i,\mathcal S_j$ are token spans (§<a href="#sec:granularity" data-reference-type="ref" data-reference="sec:granularity">6.2</a>). To avoid materializing all
$O(T^2)$ edges, prune per target to a top-$K$ in-set $\mathcal N(t)$ using a cheap contribution-norm
upper bound $\widehat E_{s\to t}=\sum_{\ell,h}\alpha^{\ell,h}_{t,s}\left\lVert \mathbf{W}_O^{h,\ell}\mathbf{v}^{\ell,h}_s \right\rVert_2$
(precomputed once per source, $O(TLHd)$); compute exact <a href="#eq:F" data-reference-type="eqref" data-reference="eq:F">[eq:F]</a> norms only on survivors. This
prunes by contribution rather than raw attention, the failure mode flags.

#### Multi-hop propagation: a Katz-style DP .

Direct edges miss indirect influence $s\!\to\!m\!\to\!t$. Propagate with a discounted right-to-left
recursion on the causal DAG:
$$P_s=\sum_{t>s}E_{s\to t}\,(1+\lambda P_t),\qquad \lambda\in[0,1),
\label{eq:katz}$$
and analogously $P_i=\sum_{j>i}E_{i\to j}(1+\lambda P_j)$ at span level. Equation <a href="#eq:katz" data-reference-type="eqref" data-reference="eq:katz">[eq:katz]</a> is
Katz–Bonacich centrality on the delivered-force graph; $\lambda$ tunes multi-hop weight
($\lambda{=}0$ recovers direct out-strength). On the pruned graph the DP costs $O(TK)$.

*The recursion is the full chain.* Unfolding <a href="#eq:katz" data-reference-type="eqref" data-reference="eq:katz">[eq:katz]</a> shows it is the efficient form of a
discounted sum over *every* directed path leaving $s$:
$$P_s=\underbrace{\sum_{t}E_{s\to t}}_{\text{1-hop}}
\;+\;\lambda\underbrace{\sum_{t,u}E_{s\to t}E_{t\to u}}_{\text{2-hop }s\to t\to u}
\;+\;\lambda^2\!\!\underbrace{\sum_{t,u,v}E_{s\to t}E_{t\to u}E_{u\to v}}_{\text{3-hop}}+\cdots
=\sum_{k\ge0}\lambda^{k}\!\!\sum_{\substack{\text{paths }s\to\cdots\\ \text{length }k+1}}\;\prod_{\text{edges}}E_{\cdot\to\cdot}.
\label{eq:pathsum}$$
So node$\to$node$\to$node$\to\cdots$ influence of all lengths is captured, each path geometrically
discounted by $\lambda$ per hop. Because the graph is a strict DAG ($s<t$), the adjacency matrix is
nilpotent: the sum terminates at path length $T$, so $\lambda\in[0,1)$ is safe with no convergence
condition (unlike Katz on a general graph, which requires $\lambda<1/\rho(E)$).

*Pruning can truncate the chain—a caveat with two consequences.* The top-$K$ in-set
$\mathcal N(t)$ is applied *before* propagation, so a weak direct edge $s\to m$ that is pruned
severs the relay $s\to m\to u$ even when $m\to u$ is strong. Two safeguards follow. (1) $K$ (the
“$\mathbf{F}$-budget”) must be set large enough not to cut genuine relay edges; we calibrate $K$ in Phase 0
by checking that increasing it no longer changes the ranking of $P$. (2) Anchor selection is performed
on the *propagated* scores $P_i$, never on direct edges $E_{i\to j}$, because indirect influence
is a global property destroyed by pre-DP pruning. The chain in <a href="#eq:pathsum" data-reference-type="eqref" data-reference="eq:pathsum">[eq:pathsum]</a> is therefore only as
complete as the retained edge set, and reported $\rho$ should be checked for stability under $K$.

#### Why not propagate through the MLP.

Folding the MLP into the graph—linearizing the activation and propagating decomposed vectors—is
DecompX . Trace-wide it costs $O(T^2Ld\,d_{\mathrm{ff}})$ and exceeds the
memory budget at long contexts; even top-$K$-pruned it is a $K\times$ multiplier on the heaviest
block. We therefore treat the MLP position-wise (§<a href="#sec:signals" data-reference-type="ref" data-reference="sec:signals">4</a>) and reserve pruned DecompX as a
*post-hoc* probe on confirmed anchors only (§<a href="#sec:followups" data-reference-type="ref" data-reference="sec:followups">10</a>).

# Signal palette

Signals score a candidate anchor as a *source* span $i$. The unit of analysis is the token span
$\mathcal S_i$ (§<a href="#sec:granularity" data-reference-type="ref" data-reference="sec:granularity">6.2</a>); all per-token quantities aggregate over spans as in
<a href="#eq:edge" data-reference-type="eqref" data-reference="eq:edge">[eq:edge]</a>.

#### Routing.

$E_i$ (direct out-strength, $\lambda{=}0$) and $P_i$ (Eq. <a href="#eq:katz" data-reference-type="ref" data-reference="eq:katz">[eq:katz]</a>, propagated). Comparing them
isolates whether indirect influence matters.

#### Computation (orthogonal axis).

Attention *moves* information between positions; the MLP *transforms* it in place. We read
the position-wise MLP write-magnitude, per-layer normalized (removing depth-driven norm growth) and
aggregated over an empirically localized band $\mathcal B$ (Phase 0):
$$G_t=\sum_{\ell\in\mathcal B}\frac{\left\lVert \mathrm{MLP}_\ell(\tilde\mathbf{h}^{\ell}_t) \right\rVert_2}{\sigma_\ell},
\qquad G_i=\sum_{t\in\mathcal S_i}G_t,
\label{eq:G}$$
with $\sigma_\ell$ the cross-position std at layer $\ell$. Here $\tilde\mathbf{h}^{\ell}_t=\mathrm{RMSNorm}(\mathbf{h}'^{\ell}_t)$
is the post-attention-norm MLP input; in pre-norm $\left\lVert \mathrm{MLP}_\ell \right\rVert$ is the *net* residual write
(the MLP output is added directly, no post-MLP norm), and §<a href="#sec:gradmath" data-reference-type="ref" data-reference="sec:gradmath">5</a> differentiates this same
quantity w.r.t. the residual $\mathbf{h}'$ with the RMSNorm in the autodiff path. $G$ is read off the same
forward pass at $O(T|\mathcal B|)$ cost.

#### Flow-into-computation (two forms $\times$ two reaches).

*Crude (computation-weighted outflow), one-hop:*
$$P'_i=\sum_{j>i}E_{i\to j}\,g(G_j),\qquad g\in\{\mathrm{id},\log,\text{threshold}\},
\label{eq:Pprime}$$
which credits $i$ with all of $G_j$ regardless of whether $i$ drove it.
*Gradient-scaled outflow* $D_i$ (defined and justified in §<a href="#sec:gradmath" data-reference-type="ref" data-reference="sec:gradmath">5</a>) credits $i$ with
the first-order share of $G_j$ that responds to $i$’s specific inflow.

Both <a href="#eq:Pprime" data-reference-type="eqref" data-reference="eq:Pprime">[eq:Pprime]</a> and $D_i$ as written are *one-hop*: they weight only $i$’s direct edges by
the computation they land in, and so do *not* chain “$i$ feeds $j$, whose result feeds $k$’s
computation.” If computational influence is transitive through the reasoning—an anchor sets up a
fact used in a later computation that itself feeds an even later one—the natural multi-hop analogue
folds the computation weight into the Katz recursion:
$$\widetilde P'_i=\sum_{j>i}E_{i\to j}\,g(G_j)\,(1+\lambda\widetilde P'_j),
\qquad
\widetilde D_i=\sum_{j>i}D_{i\to j}\,(1+\lambda\widetilde D_j),
\label{eq:chained}$$
which unfold (cf. Eq. <a href="#eq:pathsum" data-reference-type="ref" data-reference="eq:pathsum">[eq:pathsum]</a>) to $\sum_k\lambda^k\sum_{\text{paths}}\prod E_{\cdot\to\cdot}\,g(G_\cdot)$,
i.e. paths in which *every* node is weighted by its computation. Eqs. <a href="#eq:Pprime" data-reference-type="eqref" data-reference="eq:Pprime">[eq:Pprime]</a>/$D_i$ are
the $\lambda{=}0$ truncation of <a href="#eq:chained" data-reference-type="eqref" data-reference="eq:chained">[eq:chained]</a>. One-hop vs. chained is exactly the $E$-vs-$P$
(direct-vs-propagated) comparison applied to the computation axis, and is tested as such in Trials 3–4.

#### Online and confounds.

$c_i$ (streaming in-degree: increment for every $i$ appearing in some target’s $\mathcal N(t)$ during
generation; an inline approximation to $P_i$); position and span length (confound baselines);
momentum (velocity/curvature) *demoted* to a completeness check.

#### Mechanistic reading.

The palette spans two anchor species: *premise anchors* (state
content that propagates: high routing, low local computation) and *decision anchors* (perform
the load-bearing operation: high computation). $P_i$ targets the first, $G_i$ the second, $D_i/P'_i$
their intersection.

# Gradient-scaled attribution: why, and the math

We want the contribution of source $i$’s outflow to a downstream target’s computation. The MLP is a
*fixed, known, differentiable* function, so we differentiate it rather than re-learn it. Write
the pre-MLP residual at target $t$ as $\mathbf{h}'_t=\mathbf{h}^{\text{pre}}_t+\sum_{s}\mathbf{F}_{s\to t}$ (the inflow adds
into it). The downstream computation magnitude is $G_t=\left\lVert \mathrm{MLP}(\mathbf{h}'_t) \right\rVert$. A first-order (Taylor)
expansion around removing $s$’s inflow gives
$$G_t(\mathbf{h}'_t)\;\approx\;G_t\!\big(\mathbf{h}'_t-\mathbf{F}_{s\to t}\big)\;+\;\underbrace{\nabla_{\mathbf{h}'_t}G_t\cdot\mathbf{F}_{s\to t}}_{\text{marginal contribution of }s},
\label{eq:taylor}$$
so the contribution of $s$’s inflow to $t$’s computation is the directional derivative
$\nabla_{\mathbf{h}'_t}G_t\cdot\mathbf{F}_{s\to t}$, obtained by one vector–Jacobian product:
$$\nabla_{\mathbf{h}'_t}G_t=\frac{\mathrm{MLP}(\mathbf{h}'_t)}{\left\lVert \mathrm{MLP}(\mathbf{h}'_t) \right\rVert}\,\mathbf{J}_{\mathrm{MLP}}(\mathbf{h}'_t),
\qquad
D_{s\to t}=\big(\nabla_{\mathbf{h}'_t}G_t\big)\cdot\mathbf{F}_{s\to t},
\qquad
D_i=\sum_{j>i}\sum_{\substack{s\in\mathcal S_i,\,t\in\mathcal S_j}}D_{s\to t}.
\label{eq:grad}$$

#### Per-layer, local Jacobians.

Eq. <a href="#eq:grad" data-reference-type="eqref" data-reference="eq:grad">[eq:grad]</a> is written at a single level for clarity,
but $G_t$ aggregates MLP writes over the band $\mathcal B$, so the operative form is per-layer,
$D_{s\to t}=\sum_{\ell\in\mathcal B}\big(\nabla_{\mathbf{h}'^{\ell}_t}\left\lVert \mathrm{MLP}_\ell \right\rVert/\sigma_\ell\big)\cdot\mathbf{F}^{\ell}_{s\to t}$,
where each gradient is the *local* Jacobian of layer $\ell$’s MLP w.r.t. *its own* pre-MLP
input—the input is treated as a detached leaf so the gradient does not flow back through the residual
stream into earlier layers (we want layer $\ell$’s in-place sensitivity, not a full-network gradient).
Because the MLP is position-wise ($\partial G_t/\partial \mathbf{h}'_\tau=0$ for $\tau\neq t$), all positions’
gradients are recovered in a single backward pass per layer by back-propagating the summed per-position
norm; the cost is $|\mathcal B|$ local backward passes, each vectorized over positions.

#### Provenance.

Equation <a href="#eq:grad" data-reference-type="eqref" data-reference="eq:grad">[eq:grad]</a> is not new machinery; it is the established
gradient$\times$input / attribution-patching operator, applied to a new metric. The operator
$\nabla F(x)\cdot x$ originates as *gradient$\times$input* , introduced
for input-feature saliency, where multiplying the gradient by the input sharpens attributions and
partly mitigates the vanishing-gradient problem of raw gradients; it is provably equivalent to
$\epsilon$-LRP and DeepLIFT for ReLU networks with a zero baseline . The same
first-order operator was carried into mechanistic interpretability as *(edge) attribution
patching*: Nanda (2023) and use $x_e\,\partial \mathcal L/\partial x_e$ as a
linear approximation of the effect of patching a component, to discover task circuits—and report it
*outperformed* the prior automated method (ACDC) in circuit-recovery AUC while needing only two
forward passes and one backward pass. Our $D_{s\to t}$ is exactly this operator with one substitution:
the metric is not the output loss but the *downstream MLP-write magnitude* $G_t$, and the
perturbation is the delivered force $\mathbf{F}_{s\to t}$. So we inherit a validated, cheap estimator and apply
it to attribute *computation* (the routing-vs-computation axis) rather than loss—the metric
choice is the only new element, and should be verified against the circuit-attribution literature
before being claimed as such.

#### Why gradient-scaled over crude $P'$.

$P'$ (Eq. <a href="#eq:Pprime" data-reference-type="ref" data-reference="eq:Pprime">[eq:Pprime]</a>) credits $i$ with the *total* downstream computation $G_j$, even when
$i$’s specific contribution is irrelevant and some other inflow drives the MLP. The gradient credits
only the component of $G_t$ that *responds* to $i$’s actual contribution direction, and—because
$\nabla G_t$ is evaluated at the true $\mathbf{h}'_t$—it is *context-aware*: it already accounts for
every other inflow present. Under local near-linearity $D_{s\to t}\!\approx\!c\,\left\lVert \mathbf{F}_{s\to t} \right\rVert$ and
$D_i$ coincides with $P'_i$; they diverge exactly where the MLP is nonlinear or the gradient direction
misaligns with $\mathbf{F}_{s\to t}$, so the $D$-vs-$P'$ gap is itself a measure of how much nonlinear credit
assignment matters.

#### First-order caveat and faithful fallback.

Equation <a href="#eq:grad" data-reference-type="eqref" data-reference="eq:grad">[eq:grad]</a> is first-order; under activation saturation the local gradient understates a
finite contribution—the zero-gradient failure that affects gradient$\times$input and attribution
patching alike. The fix is the same one the attribution literature adopted: *integrated
gradients* , which accumulates the gradient along a path from a
baseline to the actual input and is theoretically grounded by a completeness axiom, restoring
faithfulness where the single local derivative fails. Applied to edge attribution this is
*EAP-IG* , which showed recovers more faithful circuits
than vanilla EAP precisely by removing the zero-gradient pathology. Our completeness-respecting variant
is the direct analogue:
$$D^{\mathrm{IG}}_{s\to t}=\Big(\textstyle\int_0^1 \nabla_{\mathbf{h}'}G_t\big(\mathbf{h}'_t-(1-\alpha)\,\mathbf{F}_{s\to t}\big)\,d\alpha\Big)\cdot\mathbf{F}_{s\to t}.
\label{eq:ig}$$
Exact ablation $G_t(\mathbf{h}'_t)-G_t(\mathbf{h}'_t-\mathbf{F}_{s\to t})$ is the faithful-but-costly endpoint, reserved for
confirmed anchors (all three share the baseline $\mathbf{h}'_t-\mathbf{F}_{s\to t}$: full input minus only $s$’s inflow,
other inflows present). The ladder $P'\!\to D\!\to D^{\mathrm{IG}}$ mirrors crude$\,\to$EAP$\,\to$EAP-IG and
is climbed only if the cheaper rung fails to beat baseline. One mitigating point in our favor: EAP is
known to capture the *ordering* of edges better than their absolute effects ;
since we evaluate by Spearman rank correlation, the first-order approximation’s main weakness (poor
absolute values) is largely irrelevant to anchor *ranking*.

# Data generation

## Model and hardware

Target: DeepSeek-R1-Distill-Qwen-1.5B ($L{=}28$, $H{=}12$, $d{=}1536$, $d_h{=}128$), fp16. All phases
use the *same* model so that resampling labels and forward-pass signals remain commensurable;
cross-model transfer is a separate study (see Risks). Hardware: $8\times$A100. With batched generation
($\sim$<!-- -->10$^{9.5\pm0.3}$ generated tokens over 4–5 days) the budget supports gold-level $R{=}100$
resampling across diverse problems; the attribution graph adds $<1\%$ once edges collapse to scalars.

## Granularity: clause-respecting windows

Small CoT models emit short traces, so sentence units yield too few labels per trace and coarse
localization; but a fixed token window cutting across clause boundaries would split a single semantic
unit, making a span an incoherent thing to resample or substitute (§<a href="#sec:data" data-reference-type="ref" data-reference="sec:data">6</a>). We therefore use a
*two-stage, boundary-first* segmentation, so windows are nested *inside* linguistic units and
never straddle a break:

1.  **Parse boundaries first.** Segment the trace into clauses by splitting at sentence
    terminators, newlines, comma-delimited clause boundaries, and reasoning/discourse connectives
    (*so, therefore, thus, then, next, because*). Each resulting unit is one clause.

2.  **Window within a clause.** A clause of $\le t$ tokens is one span. A longer clause is
    subdivided into consecutive $t$-token windows (the final remainder merged with the previous window if
    shorter than $t/2$). Windows are never formed across a clause boundary.

The framework is granularity-agnostic (<a href="#eq:edge" data-reference-type="eqref" data-reference="eq:edge">[eq:edge]</a>,<a href="#eq:G" data-reference-type="eqref" data-reference="eq:G">[eq:G]</a> aggregate over arbitrary spans);
only the span definition changes. We sweep $t\in\{10,15,25\}$ in Phase 0 and pick the granularity at
which span-level resampling labels are most stable. Because the resample-and-filter label
(§<a href="#sec:data" data-reference-type="ref" data-reference="sec:data">6</a>) and its semantic filter were designed at the *sentence* level, this sweep must
also confirm the method *transfers* to sub-sentence windows—that semantically-different
resamples of a short span are obtainable and the filter calibrates sensibly; if fine windows do not
support stable filtered resampling, fall back to sentence granularity, which also aligns with reusing
their sentence-level dataset. Perturbation operates on the whole span, which is
now always a clause or a within-clause window—a coherent unit to perturb.

## Resampling labels (parallel rollouts)

For span $i$ in a base trace, importance is the counterfactual dependence of the answer on $i$’s
content . We adopt the method of as the gold label—
both because it is the published benchmark our $0.22$ target was measured against (changing the label
would make that comparison apples-to-oranges) and because its counterfactuals stay in-distribution.

#### Primary: resample-and-filter .

Resample $R$ replacement spans for
$i$ *from the model itself* (continuing from the prefix up to the start of $i$), filter for the
semantically *different* ones, continue the chain of thought to the answer from each, and compare
the final-answer distribution against continuations of the unperturbed (“real”) prefix. The semantic
filter *is* the similar-vs-different split: replacements that merely paraphrase $i$ are the
implicit control (they should not move the answer), and importance is read off the
semantically-different replacements—so no separate constructed contrast is needed for the gold label.
Because every replacement is something the model would actually emit, this avoids the
out-of-distribution behavior an injected edit can cause. We reuse their released code and—where it
covers the target model—their resampling dataset, and we match their exact importance metric for
comparability. For verifiable-answer tasks the answer is discrete, so the distributional shift is a
total-variation distance (a fine discrete default; match their metric if it differs):
$$A_i \;=\; \mathrm{TV}\!\Big(\hat p_{\text{ans}}\!\mid\!\text{real}_i,\;\;\hat p_{\text{ans}}\!\mid\!\text{diff-resample}_i\Big)
\;=\;\tfrac12\sum_a\big|\hat p(a\mid i)-\hat p(a\mid i')\big|.
\label{eq:label}$$
Large $A_i\Rightarrow$ anchor.

#### Secondary: constructed task/frame swap (construct-validity check; last compute rung).

On the structured tasks we can also *construct* a perturbation deterministically by partitioning a
span’s tokens against the closed task vocabulary: editing the *task-content* tokens (swap an
entity/relation/value for a same-type sibling) gives a guaranteed semantically-*different* edit,
while editing only the *frame* tokens (verbs, connectives) and holding the task tokens fixed gives
a guaranteed-*similar*, meaning-preserving edit (preservation holds by construction, so no
similarity classifier is needed). Its primary value is as a zero-classifier-error
*construct-validity check* on the bAbI/synthetic stage—verifying the gold pipeline behaves
(different moves the answer, similar does not, and a real-vs-real null gives $A\!\approx\!0$). It is
also the last rung of the compute ladder (§<a href="#sec:groundtruth" data-reference-type="ref" data-reference="sec:groundtruth">7</a>): it guarantees a different sample in
one shot, sparing the filter’s oversampling, but at the risk of OOD prefixes and without reducing the
core rollout count. Spans with no task-content tokens (procedural glue) have no clean swap and are
excluded from this check (rate recorded). Not applicable to the free-form GSM8K hold-out, which has no
closed vocabulary.

*Granularity.* We run resample-and-filter at our clause-window granularity
(§<a href="#sec:granularity" data-reference-type="ref" data-reference="sec:granularity">6.2</a>); their released sentence-level labels, if they cover the target model, serve
as a free comparability cross-check at sentence granularity.

#### Pipeline.

\(1\) generate base traces; (2) segment into clause-windows; (3) for each span,
dispatch the resample-and-filter rollouts batched across the 8 GPUs (the dominant cost); (4) aggregate
answer distributions and compute <a href="#eq:label" data-reference-type="eqref" data-reference="eq:label">[eq:label]</a>; (5) in parallel, hook attention
weights/values/keys, $\mathbf{W}_O$ outputs, per-layer MLP writes, and final-layer states for the signals.
Independent unit for power and generalization is the *trace/problem*, not the span (within-trace
spans are correlated).

## Dataset curriculum

A difficulty ramp through *verifiable, state-tracking* reasoning—the regime where anchors are
most pronounced and where pivotal steps are sequential and identifiable:

- **Phase 0 — bAbI :** 20 synthetic QA tasks, short and
  fully specified; pipeline debugging and the $t$/$\mathcal B$ localization sweeps. bAbI ships with
  *supporting-fact annotations*—the facts required to answer each question are labeled—so
  “on-path supporting fact $=$ anchor” is a *free* constructive ground truth, independent of
  resampling, available from Phase 0 onward.

- **Phase 1 — synthetic / bAbI-derived state-tracking:** use bAbI’s supporting-fact labels
  (or a procedural generator with the same on-path/off-path structure) so the load-bearing facts are
  known by construction. This gives a construct-validity check: the white-box signals and the resampling
  label $A_i$ should both rank the annotated supporting facts above the distractors, independent of
  resampling noise. Build a custom generator only if finer hop-count control than bAbI offers is needed.

- **Phase 2 — BBH Tracking Shuffled Objects :** sequential
  state updates under swaps; each swap is a candidate decision anchor.

- **Phase 3 — CLUTRR :** compositional kinship reasoning with graded
  hop count; each relation-composition step is pivotal.

- **Phase 4 — SCONE :** long-horizon context-dependent instruction
  execution (alchemy/scene/tableau); the hardest state-tracking regime.

*Rationale.* The common thread (verifiable answers $+$ sequential
state updates) is exactly where anchors are clean and where the synthetic stage gives a constructive
sanity check the natural datasets cannot. One addition: because the entire ramp is state-tracking, hold
out a *different* reasoning type (e.g. GSM8K math) purely as an out-of-distribution generalization
test—not for training—so a positive result is not silently confined to state-tracking.

# Ground truth

#### (A) Resampling, Eq. <a href="#eq:label" data-reference-type="eqref" data-reference="eq:label">[eq:label]</a> (gold).

The resample-and-filter method of
(§<a href="#sec:data" data-reference-type="ref" data-reference="sec:data">6</a>). Black-box *by design*: independence from the
forward-pass signals is what makes a correlation with it meaningful and non-circular; reusing their
metric (and dataset where it covers the model) keeps the $0.22$ comparison exact.

#### Compute-reduction ladder.

Gold $R{=}100$ resampling is the dominant cost. If the budget
bites, descend this ladder (cheapest correctness loss first), not a single fallback:

1.  **Lower $R$**—simplest knob; the cost is noisier labels, quantified by the bootstrap CI.

2.  **Their masking + token-logits proxy** —their own sentence-level
    measure, validated against resampling at ${\sim}100\times$ less compute; the natural first structural
    fallback because it is already benchmarked against the gold.

3.  **Lyapunov proxy (B), below**—calibrate against a small gold batch, then scale.

4.  **Constructed task/frame swap** (§<a href="#sec:data" data-reference-type="ref" data-reference="sec:data">6</a>)—guarantees a different sample in one
    shot, sparing the filter’s oversampling, but risks OOD and does not cut the core rollout count.

#### (B) Lyapunov proxy.

Finite-time trajectory divergence after perturbation,
$\Lambda_i=\tfrac1\tau\log\!\big(\left\lVert \mathbf{z}^{\text{pert}}_{i+\tau}-\mathbf{z}_{i+\tau} \right\rVert/\left\lVert \mathbf{z}^{\text{pert}}_i-\mathbf{z}_i \right\rVert\big)$,
cheaper but a hidden-state quantity (circularity risk). Used only after calibration against (A), to
scale labeling if $\mathrm{corr}(B,A)$ is high.

# Trials

Primary metric throughout: Spearman $\rho$ against (A), with the receiver-head baseline
*reproduced on the same traces*, and bootstrap CIs over independent
traces. Trials 1–4 are single-signal marginals (the fulcrum); Trial 5 is combination. They are
exercised in the staged order of §<a href="#sec:tiers" data-reference-type="ref" data-reference="sec:tiers">2</a>, not necessarily in numeric sequence: a trial’s cheap
one-hop core enters in an early tier, its chained or nonlinear extensions only in later ones.

<div class="description">

Score spans by $P_i$ (Eq. <a href="#eq:katz" data-reference-type="ref" data-reference="eq:katz">[eq:katz]</a>).
Tests “anchors route.” Sub-analysis: $\lambda$ sweep, $E_i$ vs $P_i$ to isolate multi-hop value.
This is the closest analogue to prior white-box work and the first thing that must clear $0.22$.

Score by $G_i$ (Eq. <a href="#eq:G" data-reference-type="ref" data-reference="eq:G">[eq:G]</a>). Tests “anchors
*are* computation sites” (decision anchors), independent of routing.

Score by $D_i$ (Eq. <a href="#eq:grad" data-reference-type="ref" data-reference="eq:grad">[eq:grad]</a>). Tests
“anchors *drive* downstream computation,” with proper (context-aware, first-order) credit
assignment. We use the gradient form—rather than crude $P'$—for the reason derived in
§<a href="#sec:gradmath" data-reference-type="ref" data-reference="sec:gradmath">5</a>: it credits only the component of downstream computation that responds to $i$’s
own contribution direction. If Trial 3 ties or loses to Trial 4, re-run with $D^{\mathrm{IG}}$
(Eq. <a href="#eq:ig" data-reference-type="ref" data-reference="eq:ig">[eq:ig]</a>) before concluding credit assignment is irrelevant (a first-order null is ambiguous).
*Reach sub-analysis:* one-hop $D_i$ vs. chained $\widetilde D_i$ (Eq. <a href="#eq:chained" data-reference-type="ref" data-reference="eq:chained">[eq:chained]</a>), with a
$\lambda$ sweep—the computation-axis analogue of $E_i$-vs-$P_i$, testing whether computational
influence is transitive.

Score by $P'_i$ (Eq. <a href="#eq:Pprime" data-reference-type="ref" data-reference="eq:Pprime">[eq:Pprime]</a>). The
crude “flow-into-computation” baseline against which Trial 3 is judged; $g(\cdot)$ ablation here. The
Trial 3-vs-4 gap quantifies how much nonlinear credit assignment buys. *Reach sub-analysis:*
one-hop $P'_i$ vs. chained $\widetilde P'_i$ (Eq. <a href="#eq:chained" data-reference-type="ref" data-reference="eq:chained">[eq:chained]</a>), same $\lambda$ sweep. The best
of $\{P'_i,\widetilde P'_i\}$ and $\{D_i,\widetilde D_i\}$ define the outflow signals carried into
Trial 5 (5b and 5a respectively).

Feature set: $\{P_i,\,G_i,\,\text{one outflow signal},\,c_i,\,
\text{position},\,\text{span length}\}$. Two outflow variants:

- **5a:** outflow $=D_i$ (gradient-scaled).

- **5b:** outflow $=P'_i$ (computation-weighted).

Each fit two ways:

- **(i) Interpretable linear:** ridge regression; with principal-component (SVD-based)
  regression as a collinearity-robust variant, since the outflow signal and $P_i$ co-vary. Coefficients
  *are* the result (which mechanism carries the signal, and how much).

- **(ii) Nonlinear:** a 2-layer MLP over the same node features, to capture interactions the
  linear model misses (e.g. “$P_i$ matters only when $G_i$ is high”). Reported *beside* (i) as
  a flexibility ceiling, not a replacement: a gain of (ii) over (i) localizes how much nonlinear
  structure the interpretable model leaves on the table; opacity is the price.

Decision rule: keep the outflow signal in the deployed model only if it adds incremental value over
$P_i+G_i$; prefer 5b ($P'$, no backward pass) if 5a does not beat it; prefer (i) unless (ii)’s gain
exceeds the interpretability cost.

</div>

#### Confound controls (all trials).

*Edge-level (sink): a prediction.* Because $E_{s\to t}$ weights by $\left\lVert \mathbf{W}_O\mathbf{v} \right\rVert$, classic
attention sinks (high $\alpha$, low value-norm) are *demoted*; verify by contrasting the sink
token’s attention-mass vs $\mathbf{F}$-mass. *Propagation-level (position): open.* $P_i,c_i$ over-credit
upstream position topologically; report partial correlation with (A) after residualizing against a
position-only null.

# Contingency: remediating the chained signals

*Invoked only on bad results.* If Phase 1 / Trials 3–4 return poor or position-dominated
correlations for the chained computation signals $\widetilde P'_i,\widetilde D_i$
(Eq. <a href="#eq:chained" data-reference-type="ref" data-reference="eq:chained">[eq:chained]</a>), the likely cause is a pathology specific to chaining *unnormalized*
quantities, with a graded fix. The one-hop signals $P'_i,D_i$ are immune (single sums, no path
products), so this concerns the chained variants only.

#### Why chaining can fail.

Unlike attention (a softmax, bounded ${<}1$), the MLP score $G$ is
unbounded, typically ${\gg}1$, and grows with depth; the edges $E$ are likewise unnormalized. The
per-hop multiplier in Eq. <a href="#eq:chained" data-reference-type="eqref" data-reference="eq:chained">[eq:chained]</a> is $\lambda\,E\,g(G)$, so on the DAG the score—though
finite—can grow numerically and become dominated by chain *length* (upstream position) rather
than importance. Measure the distribution of $G$ in Phase 0 before choosing a control.

#### Remediation ladder (cheapest / most magnitude-preserving first).

1.  **Don’t chain.** The reach sub-analysis already tests one-hop vs. chained; if one-hop wins
    or ties, use it—the pathology is moot and absolute magnitude and graded $G$ are fully retained. This
    is the default unless chaining demonstrably beats one-hop.

2.  **Gain-tuning (magnitude-preserving, soft), in reach-preserving order.** Set $\lambda$ for
    *reach* (the value maximizing correlation in the reach sweep), then bound blow-up with controls
    that do not shrink it:
    (a) *center the multiplier*, $g(G){=}G/\overline{G}$ or $g{=}\log$, so the typical
    $g(G)\!\approx\!1$—free (no new knob), smooth, and fixes the *mean* hop; try first;
    (b) *per-hop cap*: clamp the per-hop term $E\,g(G)$ at a swept ceiling $c$—a hard bound that
    clips the residual *tail* centering leaves, preserves the bulk magnitude and full reach, at the
    cost of one knob; compose it on top of (a);
    (c) *sub-critical $\lambda$*, $\lambda\,\mathbb E[E\,g(G)]<1$—a last resort among the soft
    controls, since it bounds blow-up precisely by sacrificing the multi-hop reach you are chaining for;
    use only if (a)–(b) fail.
    (Threshold $g\in\{0,1\}$ is a side alternative that removes the $G$ amplifier entirely (gain
    $\to\lambda E$) but binarizes computation—use only if graded $G$ carries no signal.) These keep the
    edge magnitude $E$; (a)–(b) leave reach intact, (c) does not.

3.  **Structural normalization (hard bound, magnitude cost).** If (2) does not bound it,
    row-normalize incoming contributions per node so propagation becomes an *average* over paths
    (stochastic-matrix / rollout–style, as in ALTI/GlobEnc): guaranteed boundedness, and out-degree
    inflation removed. This is *not* global rescaling—dividing by the grand total preserves all
    proportions and fixes nothing structural—and it converts delivered force into relative *share*,
    discarding absolute magnitude; use only when (2) is insufficient.

4.  **Position residualization** (already in Phase 1) handles the residual “upstream-because-
    early vs. upstream-because-important” confound that no magnitude control (cap included) disentangles;
    apply regardless of the rung used.

The same discipline applies more mildly to routing $P_i$ (its edges are also unnormalized); the
$K$-stability and position controls usually suffice, but if $P_i$ is itself position-dominated,
rungs (2)–(4) transfer.

# Follow-up: causal interventions (Q1–Q4)

*Conditional on detection succeeding.* Correlation shows a signal tracks anchors; these test that
the *detected set* is causally load-bearing. All are attention-knockout / activation-patching
instances applied at span granularity.

#### Q1 — Skipping between anchors.

\(a\) *Speculative anchoring*: treat detected anchors as targets a draft must hit; test whether
generation can leap between them. (b) *Forced anchoring*: set $\alpha_{t,s}\!\leftarrow\!0$ for
$s\notin\mathcal A$ and renormalize (retain anchor$\to$future edges); measure accuracy vs. unmasked.

#### Q2 — Sufficiency and necessity (noise-fill).

Replace non-anchor span activations with noise / mean-ablate; retained accuracy $\Rightarrow$ anchors
*sufficient*. Filling *anchors* with noise should sharply degrade accuracy
($\Rightarrow$ *necessary*). The most convincing single experiment.

#### Q3 — Math/state heads.

On arithmetic and state-tracking items, isolate heads carrying the largest $\mathbf{F}$-mass into detected
anchors; test for a compact, interpretable subroutine circuit .

#### Q4 — Anchor-gated speculative decoding (demoted).

Draft cheaply, verify expensively, fall back to single-token decoding on detecting an anchor. Noted for
completeness only: heavily overlaps existing adaptive speculative decoding
, the sole novelty being an interpretability-derived trigger;
fitting a large target alongside a draft on commodity hardware is dubious. Not a headline.

#### Targeted mechanistic probe.

Where Q1–Q2 confirm anchors, run pruned DecompX-through-MLP
on *only* the confirmed anchor positions (cheap by sparsity) to
recover which source tokens an anchor’s MLP computes over.

# Risks

- **Core correlation may not beat $0.22$** (Trial 1–4). Primary risk; measured first.
  Better-motivated signals are better-targeted, not more *likely* to correlate.

- **Gradient feature is first-order** (Eq. <a href="#eq:grad" data-reference-type="ref" data-reference="eq:grad">[eq:grad]</a>); $D^{\mathrm{IG}}$ / exact ablation
  is the faithful fallback, reserved for confirmed anchors.

- **Positional confound** in $P_i,c_i$ is topological (distinct from the sink confound the
  edge weight demotes); controlled against a position-only null.

- **Granularity sensitivity**: span definition (window $t$, boundary rule) may change labels; swept
  in Phase 0 and held fixed thereafter.

- **OOD**: results certify anchors only within the state-tracking distribution unless the
  held-out math set transfers; cross-model claims need fresh validation.

- **Combiner opacity**: the 2-layer MLP forfeits the mechanistic account;
  ceiling/candidate, reported alongside—never instead of—the interpretable model.

- **Label-generation dependency**: the gold resample-and-filter label depends on a
  semantic-similarity filter (a classifier with a threshold), and because similar resamples are
  discarded, gross sampling can exceed the net $R{=}100$ budget. Mitigated by the Tier-0 reuse check
  (their released labels, if available) and the compute-reduction ladder (§<a href="#sec:groundtruth" data-reference-type="ref" data-reference="sec:groundtruth">7</a>); the
  constructed-swap check validates the pipeline behaves as intended.

- **Chained-signal blow-up**: $\widetilde P'_i,\widetilde D_i$ propagate unnormalized,
  unbounded $G$ and can be dominated by chain length; if they underperform, apply the remediation ladder
  (§<a href="#sec:contingency" data-reference-type="ref" data-reference="sec:contingency">9</a>). One-hop signals are immune.

<div class="thebibliography">

99
P. C. Bogdan, U. Macar, N. Nanda, A. Conmy.
*Thought Anchors: Which LLM Reasoning Steps Matter?* arXiv:2506.19143, 2025.
N. Elhage et al.
*A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread, 2021.
G. Kobayashi, T. Kuribayashi, S. Yokoi, K. Inui.
*Attention is Not Only a Weight: Analyzing Transformers with Vector Norms.* EMNLP 2020.
G. Kobayashi, T. Kuribayashi, S. Yokoi, K. Inui.
*Incorporating Residual and Normalization Layers into Analysis of Self-Attention.* EMNLP 2021.
J. Ferrando, E. Voita.
*Information Flow Routes: Automatically Interpreting Language Models at Scale.* EMNLP 2024.
A. Modarressi, M. Fayyaz, E. Aghazadeh, Y. Yaghoobzadeh, M. T. Pilehvar.
*DecompX: Explaining Transformers Decisions by Propagating Token Decomposition.* ACL 2023.
Z. Guo et al.
*Attention Score is Not All You Need (Value-Aware Token Pruning).* EMNLP 2024.
L. Katz.
*A New Status Index Derived from Sociometric Analysis.* Psychometrika 18(1), 1953.
M. Sundararajan, A. Taly, Q. Yan.
*Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML 2017.
A. Shrikumar, P. Greenside, A. Kundaje.
*Learning Important Features Through Propagating Activation Differences (DeepLIFT; gradient$\times$input).* ICML 2017.
M. Ancona, E. Ceolini, C. Öztireli, M. Gross.
*Towards Better Understanding of Gradient-Based Attribution Methods for Deep Networks.* ICLR 2018.
A. Syed, C. Rager, A. Conmy.
*Attribution Patching Outperforms Automated Circuit Discovery.* NeurIPS 2023 / BlackboxNLP 2024. arXiv:2310.10348.
M. Hanna, S. Pezzelle, Y. Belinkov.
*Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms (EAP-IG).* arXiv:2403.17806, 2024.
D. Rai, Z. Yao.
*How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning.* arXiv:2402.18312, 2024.
K. Wang, A. Variengien, A. Conmy, B. Shlegeris, J. Steinhardt.
*Interpretability in the Wild: a Circuit for Indirect Object Identification.* ICLR 2023.
K. Meng, D. Bau, A. Andonian, Y. Belinkov.
*Locating and Editing Factual Associations in GPT (ROME / causal tracing).* NeurIPS 2022.
Y. Leviathan, M. Kalman, Y. Matias.
*Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
B. Liao et al.
*Reward-Guided Speculative Decoding for Efficient LLM Reasoning.* arXiv:2501.19324, 2025.
J. Weston, A. Bordes, S. Chopra, A. Rush, B. van Merriënboer, A. Joulin, T. Mikolov.
*Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks (bAbI).* arXiv:1502.05698, 2015.
M. Suzgun et al.
*Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them (BBH).* arXiv:2210.09261, 2022.
K. Sinha, S. Sodhani, J. Dong, J. Pineau, W. L. Hamilton.
*CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text.* EMNLP 2019.
R. Long, P. Pasupat, P. Liang.
*Simpler Context-Dependent Logical Forms via Model Projections (SCONE).* ACL 2016.

</div>
