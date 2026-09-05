# Task 1-b Multi-Demonstration Reaching Experiment

- **Proposed experiment label:** `task_1b_multidemo_v1`
- **Status:** DRAFT for owner protocol lock (gate 1 of Section 8); no
  implementation or evidence yet
- **Relationship to prior work:** New experiment testing hypothesis H3
  (`docs/PLAN.md` Section 2.2). The M3 `task_1a_confirmatory_v2` evidence and
  the complete `task_1a_recovery_v1` negative result remain frozen and
  unchanged.

## 1. Why this is a new experiment

Task 1-a and its recovery extension trained from **one** scripted
demonstration. The recovery experiment concluded negatively under protocol v1:
no state-conditioned augmentation configuration satisfied the predeclared
eligibility rule, while the timing-only redesign (common pre-task hold,
simultaneous activation, explicit generator logging) was feasible and is
carried forward here as validated infrastructure, not as a scientific claim.

Task 1-b changes the **data distribution** instead of synthesizing one: several
demonstrations recorded from distinct initial postures, one common endpoint
target. The scientific question is H3:

> Demonstrations from multiple postures can produce a target-reaching policy
> that succeeds from unseen nearby postures.

This is a different mechanism from `task_1a_recovery_v1`'s synthetic
augmentation. Whether real multi-posture coverage succeeds where synthetic
augmentation of one demonstration did not is exactly the comparison `M4-005`
will make later; this experiment must therefore keep its formulation
augmentation-free so that comparison stays clean.

The robot, endpoint target, trackers, and limits are unchanged from task 1-a.
The task family name is `task_1b` because the task definition itself changes:
reaching from a **region** of initial postures rather than from the single
demonstrated posture. Experiment, study, model, evaluation, and report
artifacts use the `task_1b_multidemo_v1` family name.

## 2. Hypotheses

The primary hypothesis is:

> An ESN trained by teacher forcing on several demonstrations that reach one
> target from distinct initial postures generates references whose tracked
> motion reaches and dwells at that target from unseen initial postures inside
> the demonstrated region, with a larger success region than replaying the
> nearest demonstration.

Supporting hypotheses are:

1. Success degrades gracefully with distance from the training postures rather
   than at a sharp cliff, and interpolation (inside the training hull) succeeds
   more often than extrapolation.
2. The measured-feedback interface of task 1-a (Section 5.1 of `docs/PLAN.md`)
   is sufficient: no task code is needed because the target is fixed.
3. The model-based minimum-jerk baseline succeeds everywhere by construction;
   RC is not required to beat it. The scientific comparison is against
   nearest-demonstration replay, with the model-based baseline bounding what
   ideal planning achieves under the same trackers and limits.

A negative result is valid. In particular, success at the demonstrated
postures does not establish generalization, and the report must retain every
failed evaluation posture.

## 3. Nomenclature

Task 1-a recovery nomenclature (its plan, Section 3) applies unchanged, with
these additions:

| Symbol | Definition | Dimension/unit |
| --- | --- | --- |
| $m$ | Demonstration index, $m \in \{0,\dots,K-1\}$ | scalar |
| $K$ | Number of independent scripted demonstrations | scalar |
| $q_0^{(m)}$ | Task initial posture of demonstration $m$ (first cropped sample) | $n_q$, rad |
| $q^{\mathrm{eval}}_0$ | Initial posture of one evaluation scenario | $n_q$, rad |
| $\mathcal{Q}_{\mathrm{train}}$ | The set $\{q_0^{(m)}\}$ of demonstrated initial postures | — |
| $\mathcal{Q}_{\mathrm{eval}}$ | The disjoint set of evaluation initial postures | — |

There is no $\delta_{i,k}$, $\sigma$, $\phi$, $\gamma$, or $w_k$: this
experiment has **no synthetic augmentation**. Episode $i$ of the recovery plan
is replaced by demonstration $m$.

## 4. Data plan

### 4.1 Demonstrations

Each demonstration is produced by the existing deterministic scripted teacher
(`arm_rc_ctrl.data.teacher`): a 1.0 s pre-roll hold at its initial posture, a
joint-space minimum-jerk reach to the closed-form elbow-up joint solution of
the common target over the configured move interval, and a final dwell,
tracked by the high-gain computed-torque teacher gains in `skelarm`. Every
demonstration is imported through the same validation/preprocessing path as
task 1-a with its own raw and processed artifact records, onset annotation,
crop, and baseline check; the recovery dataset contracts
(`arm_rc_ctrl.data.recovery`) apply per demonstration unchanged.

The demonstrated initial postures $\mathcal{Q}_{\mathrm{train}}$ are placed
deterministically (not sampled per run) around the task 1-a posture
$q_0^{\mathrm{ref}} = (0.2, 1.2)$ rad within the safe envelope, with the task
1-a demonstration itself retained as $m=0$ for continuity. Placement geometry,
count $K$, and radii are open decisions (Section 9, D1/D2).

Following the recovery protocol's precedent (its D6), scripted demonstrations
are the sole independent sources for v1 so the data-distribution change is
isolated; human demonstrations join later as a separately identified
replication dataset, never mixed into v1 training.

### 4.2 Dataset artifact

A versioned `task_1b` dataset record binds the $K$ processed demonstrations
(IDs, digests, crops, onsets) into one training artifact with pooled input
statistics and per-demonstration interval boundaries. All payloads follow the
existing external-storage/Git-pointer policy; Git holds only records.

## 5. Training construction

Teacher forcing follows task 1-a semantics per demonstration: each episode $m$
(1) resets the reservoir to the all-zero state, (2) executes the configured
warm-up on its own repeated initial state $[q_0^{(m)}, 0]$, then (3) teacher
forces $\mathcal{T}([q^{(m)}_k, \dot q^{(m)}_k]) \mapsto q^{(m)}_{k+1}$ on the
cropped episode. No reservoir state passes between episodes. The readout is fit
once by ridge regression over all $K$ episodes' collected states.

The input transform $\mathcal{T}$ is centered on the pooled training set. Its
scale policy is an open decision (Section 9, D3); the M2 evidence that
per-joint standard-deviation scaling destabilized the closed loop while fixed
physical scales worked favors `fixed_scale`, and multi-posture data changes
the statistics question enough that the choice must be explicit, versioned,
and frozen with the recipe.

Absolute next-position prediction is the only readout representation; the
residual formulation was an exploratory negative in `task_1a_recovery_v1` and
is not carried into v1. Warm-up duration $T_w$ is searched over the approved
recovery set $\{0, 0.25, 0.5, 1.0, 2.0\}$ s (the recovery study's feasible
mass sat at $T_w \in \{0, 0.25\}$; both remain included).

## 6. Experimental arms

Frozen PD v2 and computed-torque trackers, symmetric torque limits (10 and
5 N·m), the 0.5% saturation bound, and identical scenario initialization apply
to every arm. For each tracker and identical evaluation scenario, compare:

1. **Replay-nearest:** replay the cropped demonstration whose initial posture
   is nearest to the scenario's $q^{\mathrm{eval}}_0$ under a declared,
   frozen metric (Section 9, D5), with the common pre-task hold at
   $q^{\mathrm{eval}}_0$. This is the primary scientific baseline: it embodies
   "memorize the library, pick the closest tape."
2. **Model-based reaching:** a minimum-jerk joint-space plan from the measured
   $q^{\mathrm{eval}}_0$ to the target's joint solution over the standard move
   duration — the same generator that scripted the teacher, now planned from
   the evaluation posture. It bounds achievable performance under identical
   trackers/limits and is reported, not gated against.
3. **RC multi-demonstration (primary):** the frozen multi-demo ESN with
   measured $[q, \dot q]$ feedback, common pre-task hold, simultaneous
   activation, and `generator_output_q` logging, exactly as validated by the
   recovery timing redesign.

No augmentation arm exists in v1 (Section 1). Budgets and reservoir bounds are
matched to the recovery study's approved ESN search space unless the pilot
motivates a documented change.

## 7. Evaluation

### 7.1 Scenarios and splits

Evaluation postures $\mathcal{Q}_{\mathrm{eval}}$ are seeded, disjoint from
$\mathcal{Q}_{\mathrm{train}}$ and from every prior namespace, and stratified
into declared classes: **near-interpolation**, **far-interpolation**, and
**extrapolation** relative to the training placement geometry (Section 9, D2).
Force and combined classes are deferred to keep v1 focused on the
initial-posture mechanism; the nominal class evaluates every demonstrated
posture itself. A replay-baseline safety pilot fixes one method-independent
posture envelope before any development study, mirroring the recovery pilot.
Development and confirmatory seed namespaces are disjoint.

### 7.2 Metrics

Per `docs/PLAN.md` Section 9.4 (task 1-b): endpoint target-region dwell
success, final error, settling time, path length/efficiency, effort, and
success from unseen initial postures — plus the carried diagnostics
(saturation, smoothness, generated-reference dwell gates). The primary summary
is the **success region**: per-arm success indicators over
$\mathcal{Q}_{\mathrm{eval}}$, reported as per-class success rates and a
posture-space map, paired per scenario against replay-nearest.

### 7.3 Development eligibility and model freeze

Every development run must satisfy the carried safety, saturation, and
actual-motion dwell criteria, and the generated reference must satisfy the
recovery protocol's generated-dwell gates (1 cm / 90% / 0.05 rad/s). The
eligibility rule for freezing — how much of which class RC must succeed in,
paired against replay-nearest, per tracker — is an owner decision (Section 9,
D6) and is locked before any development study reports are read. Selection
among eligible models is lexicographic (worst-class success rate, then
settling time, then applied-torque RMS) unless D6 declares otherwise.
Confirmatory outcomes cannot change these gates.

## 8. Reproducibility and decision gates

The external-payload/Git-pointer policy, clean-worktree provenance rules, and
one-command reproduction requirements carry over unchanged. Implementation
proceeds through the same review gates as the recovery experiment:

1. **Protocol lock:** approve this document's hypotheses, labels, data plan,
   arms, metrics, splits, and open decisions.
2. **Data vertical slice:** record, validate, and preprocess the $K$
   demonstrations; commit records and a dataset artifact with regression
   coverage.
3. **Training/evaluation vertical slice:** train one multi-demo recipe and
   run all three arms on a handful of scenarios with full telemetry.
4. **Safety pilot and locks:** freeze the posture envelope and evaluation
   scenario locks from replay-only evidence.
5. **Development study and model freeze:** search, ablate, and apply the D6
   rule; retain a negative result when no model qualifies.
6. **Confirmatory gate:** one-shot locked suite, clean-checkout reproduction,
   independent audit, complete report.

Software acceptance remains deterministic execution, validation, and complete
evidence — not a favorable scientific result.

## 9. Decisions to approve implementation

Open decisions proposed for owner review; none is implemented before approval:

| ID | Decision | Proposal |
| --- | --- | --- |
| D1 | Demonstration count | $K \in \{4, 8, 12\}$ with $K=8$ as the anchor: the task 1-a demonstration plus 7 placed postures. Larger $K$ is a study dimension only if the data slice shows the cost is negligible. |
| D2 | Posture placement and evaluation classes | Training: $q_0^{(m)} = q_0^{\mathrm{ref}} + \Delta_m$ with $\Delta_m$ on a deterministic ring/grid at radii $\{0.10, 0.20\}$ rad (joint space, per-joint clipped to the safe envelope). Evaluation classes: near-interpolation $\le 0.10$, far-interpolation $\le 0.20$, extrapolation $(0.20, 0.30]$ rad; 20 seeded postures per class plus the $K$ nominal postures. |
| D3 | Input-transform scale policy | `fixed_scale` (0.3 rad, 4 rad/s) as primary, `training_std` as a named study dimension at most — carried from the M2 closed-loop finding. |
| D4 | Search protocol shape | One Optuna study, recovery-v1 ESN bounds, $T_w$ grid as a categorical, $K$/placement fixed (not searched) in v1; budget 500 trials, new disjoint sampler-seed namespace. |
| D5 | Nearest-demonstration rule | Unweighted joint-space Euclidean distance between $q^{\mathrm{eval}}_0$ and $q_0^{(m)}$, ties broken by lower $m$; frozen before the pilot. |
| D6 | Freeze rule | Per tracker independently: RC success rate $\ge$ replay-nearest's in **every** evaluation class, strictly greater in at least one interpolation class, with all carried safety/dwell gates; at least 15 of 20 scenarios per interpolation class succeeding. Numerical revision after evidence creates v2. |
| D7 | Scenario config identity | New `configs/tasks/task_1b.toml` sharing the task 1-a robot, limits, and target by value, with the initial posture supplied per scenario; task 1-a configs stay untouched. |
| D8 | Baseline disposition | Model-based reaching is reported as a bound, never as the paired scientific baseline; H3 is judged against replay-nearest only. |

Approval of D1–D8 locks the protocol for implementation and does not authorize
the confirmatory run, which stays separately gated by gates 2–5 of Section 8.
