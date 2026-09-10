<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Task 1-a Recovery: Repeated-Demonstration Control

- **Experiment label:** `task_1a_repetition_v1`
- **Status:** APPROVED for implementation; D1–D8 approved by the owner on
  2026-09-09. M3REP-001 (frozen panel manifest), M3REP-009 (canonical
  execution environment), M3REP-002 (exact-repetition recipes), and UP-007
  (readout weight accessor, pin 61a29f0) are complete. The Section 6
  numerical validation is complete (M3REP-003, closed 2026-09-10) with 71 of
  72 equivalence comparisons within tolerance and one diagnosed numerical
  exception accepted under C11. No behavioral results yet.
- **Approval date:** 2026-09-09.
- **Approved scope:** A fixed, paired development pilot, its
  numerical controls, and a reproducible report. A larger search is a later
  decision.
- **Relationship to prior work:** A separate extension of
  `task_1a_recovery_v1`, using its demonstration, timing, frozen trackers, and
  development scenarios. The completed recovery evidence and its negative
  model-freeze decision remain unchanged. This is separate from the
  [task 1-b multi-demonstration proposal](../task_1b_multi_demonstration/plan.md).
- **Developer handoff:** Owner approval is registered in the global
  [PLAN.md](../../PLAN.md) and [task ledger](../../TASKS.md) under DOC-006 and
  M3REP-001 through M3REP-008 / M3REP-GATE. Start with M3REP-001; follow the
  dependency and acceptance gates before executing the full pilot.

## 1. Motivation and question

The [recovery experiment](../task_1a_state_conditioned_recovery/overview.html)
compared a single original demonstration with the original plus 16, 32, or
64 synthetic variations. Its timing-only arm produced 134 feasible trials
out of 500, but none met the recovery eligibility rule. Neither absolute-output
augmentation arm produced a feasible model among its 500 sampled trials.

That comparison changed both the trajectory distribution and the number of
training rows. This experiment adds the missing control: replace every
synthetic episode with an exact copy of the original episode. There is still
one independent demonstration, now used 17, 33, or 65 times in total.

The scientific question is:

> At the same ESN configuration, episode count, and ridge parameter, how does
> closed-loop recovery differ when the additional episodes are exact repeats
> versus varied trajectories?

The pilot also tests whether any change from repetition is explained by the
effective regularization strength, and whether a residual (joint-increment)
readout changes the effect of repeating the original demonstration. It does
not test online adaptation,
iterative optimization epochs, new demonstrations, or hardware control.

## 2. Why the ridge parameter must be explicit

The current [trainer](../../../src/arm_rc_ctrl/rc/training.py) resets the
reservoir for every episode, collects its states, and stacks all loss rows
into one ridge fit. Given identical input arrays, warm-up, and reservoir
weights, repeated episodes produce identical states. No reservoir state or
readout update carries from one training episode into another.

Let $X$ contain the loss-row reservoir features, including the readout bias
column, and let $Y$ contain the original demonstration's targets: next
positions for absolute output, or next-step position increments for residual
output. The equivalences below apply separately within each formulation;
they do not imply equivalence between formulations. Let $W$ be the readout
and $\alpha_0$ the base ridge parameter.
The pinned [ridge implementation](../../../third_party/rclib/cpp_core/src/readouts/RidgeReadout.cpp)
solves the summed-loss problem

$$
\min_W \|XW-Y\|_F^2 + \alpha_0\|W\|_F^2.
$$

It adds the ridge parameter directly to the Gram-matrix diagonal, including
the bias entry; it does not divide the loss by the sample count. With $K$
identical episodes and the same solver parameter, the minimizer is therefore

$$
\begin{aligned}
\arg\min_W \left[K\|XW-Y\|_F^2 + \alpha_0\|W\|_F^2\right]
&= \arg\min_W \left[\|XW-Y\|_F^2
  + \frac{\alpha_0}{K}\|W\|_F^2\right].
\end{aligned}
$$

Two mathematical equivalences follow:

1. Repeating $K$ times with solver parameter $\alpha_0$ is equivalent to
   one episode with solver parameter $\alpha_0/K$.
2. Repeating $K$ times with solver parameter $K\alpha_0$ is equivalent to
   one episode with solver parameter $\alpha_0$.

These are exact-arithmetic statements. Different matrix accumulation orders
can produce floating-point differences, and small prediction differences can
grow in closed loop. Section 6 separates numerical validation from the
behavioral comparison.

An improvement from exact repetition alone would thus support an effect of
regularization or numerical computation, not additional state coverage.
For varied episodes there is no identical-row equivalence; matching their
episode count and solver parameter to the repeated arm controls the loss
scaling, but their feature matrices and conditioning can still differ.

## 3. Hypotheses and interpretation

| ID | Question or expectation | Evidence needed |
| --- | --- | --- |
| H1 | Exact repetition at fixed solver parameter changes the solution through the effective parameter $\alpha_0/K$ | Repeated-state identity and agreement with the single-episode $\alpha_0/K$ control |
| H2 | Scaling the repeated arm's solver parameter to $K\alpha_0$ recovers the original single-episode solution | Numerical agreement with the original fit, plus a report of closed-loop agreement or divergence |
| H3 | Varied and repeated trajectories can yield different recovery outcomes at matched count and solver parameter | Within-configuration comparisons of feasibility, early command metrics, target dwell, and failures |
| H4 | Residual output can change the effect of repetition and the hold-to-generated-command handoff | Paired absolute/residual S, R, and R-scaled results, with numerical equivalences checked separately within each formulation |

H3 is a comparison, not a prediction that augmentation will improve recovery.
If the repeated arm is feasible and the matched augmented arm fails, that is
evidence about trajectory variation at those settings. If both fail, the
pilot does not isolate a unique cause. If repetition improves over the
single-episode baseline, the regularization controls must inform the explanation.
H4 does not assume that residual output eliminates the activation jump: its
first predicted increment need not be zero. These configurations were selected
from absolute-output results, not optimized separately for residual output.

The six configurations are chosen using already observed development results.
This is a diagnostic panel, not an unbiased sample of ESNs or a held-out test.
No confidence interval over synthetic copies should treat them as independent
demonstrations. Report descriptive paired results and their scope.

## 4. Shared data, training, and timing

Use the existing processed dataset
[`processed-20260903-ce343c8ce6a5`](../../../data/records/processed/processed-20260903-ce343c8ce6a5.toml)
and its recorded source, preprocessing, crop, input transform, and task
intervals. The episode has 401 task samples and 400 next-step loss rows:
three seconds of movement followed by one second of dwell at 100 Hz.

The following apply to every arm:

- One independent scripted demonstration, one target, two-joint gravity-free
  simulation, and the existing limits except for the approved simulation
  joint-velocity relaxation in Section 7.1.
- Absolute next-joint-position and residual next-step-increment readouts as
  assigned in Section 5.2, both using the original readout bias convention
  and the recipe's `cholesky` solver option (an LDLT factorization of the
  regularized Gram matrix in the pinned `rclib`).
- The frozen input transform from the recovery training path; do not refit
  centering or scaling on repeated or augmented episodes.
- Reset to the same all-zero reservoir state before every episode. Perform
  that configuration's warm-up on the episode's initial state before teacher
  forcing. Every repeated copy has identical warm-up, input, target, and
  loss-mask arrays.
- Fit the readout once on the stacked loss rows; repeated episodes are not
  additional optimization epochs.
- In evaluation, both RC and replay hold the same actual starting posture,
  then activate together at task time zero. Use measured position and
  velocity as ESN input and the existing causal desired-derivative estimator.

Use the existing residual convention (`target = "increment_q"`, controller
`output = "increment"`): train on
$\Delta q_k=q^{\mathrm{demo}}_{k+1}-q^{\mathrm{demo}}_k$, and at inference form
$q^d_{k+1}=q^{\mathrm{measured}}_k+\widehat{\Delta q}_k$. This is a residual
relative to the current measured posture, not the previous generated command
or a replay trajectory. The absolute formulation uses `target = "next_q"`
and directly commands the predicted next position. In both cases the readout
is inactive during warm-up; all position bounds and generated-reference
metrics apply to the resulting absolute command. Record the residual
increment separately from that command.

Store the original dataset once. Repetition is a training recipe property,
not a set of newly invented dataset identities or duplicated external
payloads. Use an explicit `additional_repeats` setting with values 16, 32,
and 64; derive $K=1+\text{additional_repeats}$. Report both counts and the
independent-demonstration count. Loss-row totals are respectively 6,800,
13,200, and 26,000; warm-up rows never enter the loss.

## 5. Approved paired pilot

### 5.1 Six fixed source configurations

Use these trial configurations from
[`recovery-search-1a-no-augmentation-v1`](../task_1a_state_conditioned_recovery/recovery_search_no_augmentation_v1.toml).
Resolve their complete parameters from that pointer's digest-verified report
payload, not from rounded human tables.

| Panel label | Source trial | Why included | Warm-up (s) |
| --- | ---: | --- | ---: |
| feasible-best | 17 | Rank 1 of the 134 feasible trials by early-gap objective | 0.25 |
| feasible-middle | 136 | Rank 67 of the 134 feasible trials | 0 |
| feasible-worst | 53 | Rank 134 of the 134 feasible trials | 1 |
| failure-actual-dwell | 1 | Lowest-numbered trial whose first failure is actual-motion dwell | 0.25 |
| failure-joint-velocity | 0 | Lowest-numbered trial whose first failure is the joint-velocity limit; also the historical anchor | 1 |
| failure-generated-dwell | 28 | Lowest-numbered trial whose first failure is generated-reference dwell | 0 |

Feasible ranks sort by `(objective, trial_number)` ascending, using the
134-candidate
[ablation record](../task_1a_state_conditioned_recovery/development_ablation_v2.json).
Failure categories use the top-level first-failure reason in the study
report. These choices were made before this new experiment; do not replace
a configuration because its repeated or augmented variant performs poorly.

Under M3REP-001, freeze a panel manifest containing these exact identities,
all resolved parameters, the source-pointer/payload hashes, and the selection
rule before training the new arms. A missing or mismatched source fails
manifest construction. No new reservoir seed is introduced in this pilot.
The frozen manifest is [`panel_manifest_v1.json`](panel_manifest_v1.json)
(rendered as [`panel_manifest_v1.md`](panel_manifest_v1.md)), produced by
`scripts/freeze_repetition_panel.py` under M3REP-001 and locked by
`tests/regression/test_repetition_panel_evidence.py`.

### 5.2 Arms and output formulations at every configuration

Let $\alpha_0$ be the source trial's ridge parameter and
$K\in\{17,33,65\}$ the total episode count.

| Arm ID | Training episodes | Solver parameter | Role |
| --- | --- | --- | --- |
| S | Original once | $\alpha_0$ | Single-episode baseline; evaluate once per configuration |
| R | $K$ exact copies of the original | $\alpha_0$ | Requested repetition control |
| R-scaled | $K$ exact copies of the original | $K\alpha_0$ | Regularization-compensated control |
| A-non-decaying | Original + $K-1$ non-decaying synthetic episodes | $\alpha_0$ | Count-matched varied-data comparison |
| A-contractive | Original + $K-1$ contractive synthetic episodes | $\alpha_0$ | Count-matched target-contracting comparison |
| S-effective | Original once | $\alpha_0/K$ | Numerical equivalence reference for R; fit and compare, without a separate behavioral sweep |

Run all six arm IDs above with absolute output. Additionally run S, R,
R-scaled, and S-effective with residual output, using only the original and
its exact copies. Use explicit identities such as `absolute/R` and
`residual/R`; the formulation is part of every recipe and result key.
Residual augmentation is outside this pilot, and the historical augmented
residual study is not a substitute for these new matched fits.

This gives **120 behavioral model configurations**:

- Absolute: $6\times[1+3\times4]=78$.
- Residual: $6\times[1+3\times2]=42$.
- Additional S-effective numerical references:
  $6\times3\times2=36$ fits across the two formulations.

Reusing S as the comparator for several values of $K$ does not create
additional independent baseline observations.

Within each configuration and output formulation, match all reservoir parameters and weights,
warm-up, input transform, output representation, and derivative filters.
Only the training construction and the declared solver-parameter rule differ.
For paired absolute/residual fits, hold those settings and episode inputs
fixed and change only the training target and command reconstruction. Reuse
the six source configurations without residual-specific retuning; the source
feasibility labels describe historical absolute fits only, not a prediction
of residual feasibility.
Verify matching reservoir realizations by deterministic feature probes or
weight digests where supported; an identical integer seed alone is not the
complete verification. Use the project's seed guard and `OMP_NUM_THREADS=1`.

Record `base_alpha`, `solver_alpha`, and `regularization_rule` explicitly.
`R-scaled` may exceed the historical search bound of 1, and `S-effective`
may fall below 0.001. These are prescribed controls, not new search draws;
do not clip them or silently widen the old protocol's search bounds.

### 5.3 Fixed augmentation settings

For both varied-data arms, use the existing seeded generator with
$\sigma=0.05$ rad, $\phi=0.99$, $\gamma=1$, seed bank 1, and the original
shared terminal taper and physical-validation policy. The non-decaying
family uses the generator's existing interpretation of this shared setup.
Retain the finite attempt budget of $4(K-1)$.

Match the two families using the generator's paired accepted episodes.
Check that each smaller episode bank is a prefix of the larger bank at these
fixed settings and retain the accepted attempt indices and array hashes.
Generation failure is a recorded failure, not permission to change seeds,
amplitude, or the attempt budget.

The fixed augmentation anchor keeps this pilot small. It does not repeat
the old search over amplitude, temporal correlation, or contraction exponent.
The varied arms are fresh matched fits at the six panel configurations;
historical study winners or unmatched trial numbers are not substitutes.

## 6. Numerical validation before behavioral evaluation

The first implementation stage must validate the predicted equivalences on
the panel before running the behavioral sweeps.

1. Construct literal repeated episodes through the normal episode-building
   and training interfaces. Require bitwise identity of corresponding
   input, target, loss-mask, warm-up, and harvested-state arrays.
2. For every configuration, output formulation, and $K$, compare R with
   S-effective and R-scaled with S within that formulation. Compare readout
   predictions on an identical probe matrix formed
   from the original and the largest fixed augmented banks' harvested
   task-row states. Retain which probes came from which episode.
   Residual fits use the augmented states only as probes, not training data.
   Compare residual increments directly and reconstructed absolute commands
   using identical measured-posture anchors; never compare an increment
   directly with an absolute position. Require identical harvested states
   for matched absolute/residual recipes with identical inputs and reservoirs.
3. Approved prediction tolerance: elementwise `atol = 1e-8 rad` and
   `rtol = 1e-8`. Record maximum absolute and relative differences as well
   as the pass/fail decision. The absolute threshold is five million times
   smaller than the 0.05 rad reference-settling diagnostic band; it allows
   accumulation roundoff without accepting a meaningful teacher-forced
   command change. It is not a closed-loop stability guarantee.
4. Report readout coefficient differences and conditioning. Also require a
   normalized normal-equation residual at or below `1e-10` for each fit:
   $\|AW-B\|_F/(\|A\|_F\|W\|_F+\|B\|_F)$, with
   $A=X^\mathsf{T}X+\alpha I$ and $B=X^\mathsf{T}Y$ for that fit's actual
   matrix. Define an all-zero numerator and denominator as zero. Use the
   pinned solver's actual bias-column order. This dimensionless diagnostic
   checks the solve separately from agreement of two accumulation paths.
5. Refit each recipe in a fresh process to verify its recorded arrays and
   fit report under the same recipe and environment. This same-path
   reproducibility check is distinct from the tolerance used to compare
   mathematically equivalent but differently accumulated problems.

The tolerances above are approved under D6. If they fail, retain
the failed fits and diagnose the discrepancy before interpreting a behavioral
effect. Do not loosen a tolerance automatically or substitute S-effective
for R and call it literal repetition.

The recorded validation is [`numerical_validation_v1.json`](numerical_validation_v1.json)
(rendered as [`numerical_validation_v1.md`](numerical_validation_v1.md)),
produced by `arm_rc_ctrl.experiments.repetition_numerics` in the canonical
execution environment. One of its 72 comparisons (feasible-middle, absolute
output, $K = 65$, R against S-effective) exceeds the prediction tolerance by
a largest difference of 2.93e-8 rad and is retained as failed; the diagnosis
[`numerical_validation_v1_diagnosis.md`](numerical_validation_v1_diagnosis.md)
supports floating-point sensitivity of the accumulation and solution path
on the panel's worst-conditioned normal matrix (cond2 1.07e9 at
$\alpha_0/65$) with the stacked and single problems agreeing to 4e-16; its
reference solutions are extended-precision solutions of NumPy-reconstructed
normal equations, so the reported solution-path part includes assembly
differences from `rclib`'s Eigen path and is not an isolated measurement of
the LDLT roundoff. The owner accepted this as an explicit numerical exception
(C11): the configuration is retained unchanged, the comparison stays failed,
no tolerance was changed, and the caveat is carried into later results.

After literal repetition is validated, harvesting one identical episode
once and reusing its states can be considered as a later optimization. Such
an optimization must prove agreement with the literal path and record the
execution strategy. The initial pilot uses the literal path as its reference.

## 7. Behavioral evaluation and scientific reporting

### 7.1 Development scenarios and safety

Reuse the scenario definitions from the locked
[recovery development configuration](../../../configs/evaluations/task_1a_recovery_dev_v1.toml)
and both frozen trackers, `pd_v2` and `computed_torque`, with a separately
versioned simulation velocity-limit setting for this pilot:

| Class | Scenarios | Perturbation |
| --- | ---: | --- |
| Nominal | 1 | Cropped demonstration start |
| Small posture | 20 | Joint-offset norm 0.05 rad |
| Large posture | 20 | Joint-offset norm 0.10 rad |
| Force | 4 | 12 N endpoint pulse, four directions, task time 1–1.2 s |
| Combined | 20 | Small offset plus the matched pulse |

Each behavioral model has at most 130 scenario/tracker evaluations, paired
against direct replay at the same warm-up and conditions. Reuse replay
baselines only after verifying their complete condition and provenance
bindings. There are three warm-up values in the panel, hence at most
390 unique replay evaluations if all need to be generated.

**Approved simulation relaxation (D5):** increase the measured joint-speed
abort threshold from **6 to 12 rad/s for each joint**, symmetrically:
abort if any $|\dot q_j|>12$ rad/s. The original threshold is recorded in
[`configs/tasks/task_1a.toml`](../../../configs/tasks/task_1a.toml).
Apply the new threshold uniformly to all learned arms and matched replay,
throughout warm-up, movement, and dwell; do not add phase-specific exceptions
or a grace period at activation. The stricter dwell-speed criterion below
still applies. Twelve rad/s is an approved bounded twofold relaxation, not a
measured optimum or a validated hardware-safe speed. Lock the value before
execution; do not increase it again in response to failures.

This change is simulation-only. Keep a finite hard abort and all other
termination paths; do not modify shared legacy task configs, frozen evidence,
hardware limits, `rtctrl` watchdogs, or hardware approval requirements.
Use a new resolved evaluation/task config and digest. Keep demonstration
preprocessing and synthetic-episode validation at their original limits,
including 6 rad/s, so this relaxation changes evaluation tolerance rather
than the training distribution. Distinguish training-validation limits from
simulation-abort limits in recipes and provenance.

Carry over every other safety, termination, torque, saturation, actual-motion dwell,
and generated-reference dwell check from the
[recovery protocol](../task_1a_state_conditioned_recovery/plan.md), including
10/5 N·m torque limits, at most 0.5% torque saturation, at least 90% of dwell
inside 1 cm, and dwell joint speeds at or below 0.05 rad/s. No unsafe run
continues past its abort condition.

Retain the old 6 rad/s threshold as a non-terminating diagnostic. For each
evaluated scenario/tracker pair and joint, report peak absolute measured
speed, first crossing of each threshold, and observed time above 6 rad/s,
split into warm-up, movement, and dwell. Record crossing timestamps on both
run and task clocks and use the simulator's checked-state cadence, including
the terminal offending sample, rather than only downsampled plot data.
Retain the limit, joint, measured value, bound, and termination timestamp
for every new velocity abort. This enables an explicit motion-phase count
that the historical study summaries do not provide.

Report relaxed-protocol feasibility alongside whether each observed run
crossed the historical limit. A run that passes only after exceeding 6 rad/s
is not a recovery-v1 success. Do not infer a complete historical-protocol
trial verdict from censored or unexecuted pairs. Replay cache identities
must include the new resolved limits; historical replay results must not be
silently reused under the changed config binding.

Use the existing deterministic scenario/tracker order and stop a model's
sweep at its first infeasible pair, as in recovery v1. Preserve that first
failure and explicitly mark all subsequent pairs as not evaluated; they are
not successful or missing-at-random observations. Training or generation
failure similarly remains in the model-level denominator. A failed arm
does not cancel other arms of its matched configuration.

### 7.2 Metrics and comparisons

The primary pilot outputs are the matched model-feasibility outcomes and,
where available, per-scenario command comparisons. Report all six base
configurations and all three counts, with no post-result subset selection.

- Compare A-contractive versus R and A-non-decaying versus R at the same
  $K$ and $\alpha_0$ within absolute output. This is the main
  varied-versus-repeated comparison.
- Compare R versus S to show the practical effect of repetition at fixed
  solver parameter, separately for absolute and residual output. Interpret
  it together with the corresponding S-effective equivalence check.
- Compare R-scaled versus S for numerical and behavioral consistency.
  Perform this comparison separately within each output formulation.
  Report any closed-loop divergence despite agreement on the fixed probe
  matrix. A changed safety or eligibility verdict requires investigation
  and an explicit explanation before attributing a scientific mechanism.
- Compare absolute versus residual S, R, and R-scaled at matched source
  configuration, count, and solver parameter. Report whether repetition's
  effect relative to each formulation's own S baseline differs; do not
  attribute a formulation change to repetition or claim either formulation
  was independently optimized.

Retain the original activation jump and first-0.5-second command-gap integral,
their ratios against matched replay, and the four posture-class-by-tracker
cells. Report actual and generated target dwell, reference settling,
original-trajectory movement RMSE, desired/actual smoothness, and applied
torque and saturation. Distinguish generated-reference settling from
endpoint settling. Force pulses occur after the early window and therefore
cannot be assessed by that early metric alone.

Show signed differences in the original units between matched learned arms
in addition to their ratios against replay. Comparisons involving incomplete
sweeps must state their shared evaluated scenarios; do not fabricate ratios
for failed or unexecuted runs, or interpret a partial 20-scenario cell as a
complete eligibility result.

Apply the historical eligibility structure as a descriptive diagnostic: all
feasibility gates, both median ratios below 1, and improvement of both early
metrics in at least 15 of 20 scenarios in each of the four cells. Keep the
ratio/dwell thresholds and ordering unchanged, but explicitly label the
changed 12 rad/s feasibility limit. A pilot configuration passing them is a
development candidate only; this pilot does not select or freeze a model.

### 7.3 Required report and limitations

Produce machine-readable evidence plus a human-readable `overview.html` in
this experiment directory. The HTML should explain the ridge equivalence,
show the full paired outcome table, display numerical-equivalence errors,
and make failed or unexecuted evaluations visible. Include measured fit time,
evaluation time, and peak memory by count and arm.

Every time-series figure must use **task-relative time**,
$t_{\mathrm{task}}=t_{\mathrm{run}}-T_w$, including plots of posture,
references, velocities, torques, and reservoir diagnostics. Show the full
warm-up interval $[-T_w,0)$ when $T_w>0$, shade and label it as warm-up/hold,
and mark activation at **0 s**. Movement occupies $[0,3)$ s, dwell $[3,4]$ s,
and force pulses $[1,1.2)$ s, independent of warm-up duration. For zero
warm-up, start at zero with no invented negative-time samples. Label axes
`Task time (s)`; retain original run timestamps in raw evidence and record
the transform rather than changing simulation timing or metric windows.
Animation clocks and synchronized cursors must use the same task clock.

Distinguish the externally imposed hold command from the active ESN readout.
Keep ESN output missing/NaN before task time zero; do not draw the hold as an
ESN prediction or connect a generated curve across the inactive interval.
Show the actual tracker reference separately so that the hold-to-generation
jump at zero is visible. For residual output, label the predicted increment
and the reconstructed absolute position separately; compare the latter with
absolute-output and replay references. These conventions apply to this new
report, without regenerating or changing the frozen recovery report.

Prefer paired outcome and command-metric plots over many selected motion
clips. Any new figures or animations must have reproducible generators in
`src/arm_rc_ctrl/experiments/` and thin entry points in `scripts/`. Predeclare
any representative-motion selection rule before choosing examples; do not
choose only successful conditions after seeing the results.

State that the panel is small and chosen from historical development
outcomes, augmentation uses one fixed anchor and seed bank, all training
derives from one demonstration, and first-failure censoring hides later
behavior. The repeated control resolves a particular count/loss-scaling
confound at these settings; it does not establish a universal causal
explanation of the original 2,000-trial result or a general recovery region.
The relaxed velocity threshold also prevents direct comparison of new
feasibility percentages with the historical searches. Within-pilot paired
arms share the same threshold; report historical-limit crossings separately
so increased speed tolerance is not mistaken for improved recovery quality.

## 8. Reproducibility and implementation boundaries

Owner approval and global task registration are complete. Implementation
proceeds through the M3REP queue and its acceptance gates.
Reuse the existing experiment modules while giving every new recipe, study,
run, and report a distinct identity under `task_1a_repetition_v1`.

The developer-facing contracts are:

- Add an explicit exact-repetition training construction. Reject unknown
  keys, invalid counts, and mutually enabled repetition and augmentation
  settings. Do not use `sigma = 0` as an unlabelled augmentation workaround.
- Extend recipe/refit support with deterministic repetition labels and both
  base and solver ridge parameters. Preserve the single source dataset ID.
  Existing recipe validation requires distinct source datasets, so repeated
  source-ID entries are not an acceptable implementation.
- Support both output formulations explicitly using the existing increment
  training and inference paths. Do not inherit the historical residual
  search's requirement for contractive augmentation: this pilot's residual
  recipes use the original alone or exact repetition. Validate target/output
  agreement and retain both predicted increments and absolute commands.
- Preserve legacy recipe loading, serialized content, config digests, and
  reproduction behavior. Use a separately versioned schema where new fields
  would otherwise change an old record's canonical representation; adding
  default fields must not silently change frozen v1 hashes.
- Run the fixed panel through a new pilot protocol/runner. It must support
  prescribed scaled ridge values without modifying the historical search
  space, and the explicit evaluation-only velocity limit without changing
  legacy task configs or augmentation acceptance. No adaptive sampler is
  required for the pilot.
- Cache only values with complete bindings: source hashes, reservoir
  realization, input transform, episode construction, warm-up, and solver
  settings as applicable. Include target/output formulation for fitted
  models, runs, and results; feature-only reuse requires verified identical
  inputs and reservoir realization. Resume by immutable run identity, never by a
  partial key such as source trial number alone.
- Persist failures and provenance through the normal external-store and
  Git-pointer paths. Record project/submodule revisions, resolved configs,
  dataset and episode hashes, seeds, environment/build identities, solver
  rules, and dirty-worktree state. No raw payload duplication in Git.
- Provide one-command reproduction of the panel manifest, episode/state
  identities, fits, recorded evaluation outcomes, metrics, and report assets.
  Declare each exact or tolerance-based comparison with its units and reason.
- Test first: repeat counts and row counts; per-episode reset; literal-array
  identity; both ridge equivalences; invalid settings; recipe round trips
  and legacy compatibility; paired evaluation conditions; first-failure
  retention; deterministic resume and reporting. Cover both output
  formulations, residual target and measured-posture reconstruction, and
  bounds checks on absolute commands. Test plot/animation time conversion
  for $T_w=0,0.25,1$ s, activation and event alignment at task time, and
  missing readout values throughout warm-up. Test the 12 rad/s abort,
  non-terminating 6 rad/s diagnostic, per-phase crossing timestamps,
  unchanged dwell/torque gates and training-validation limits, and cache
  separation across velocity-limit configs. Run the repository quality
  gates before handoff.

Historical evidence must remain intact. The older task 1-a reproduction
test previously showed a `1.608e-10` metric discrepancy against its exact-zero
tolerance during an unrestricted rerun, as documented under DOC-005. The
developer should report its current status separately; this experiment's
numerical comparison tolerances do not authorize changing that historical
test or its evidence.

## 9. Work packages and decision gates

These work packages are registered in `docs/TASKS.md`, the authoritative
status ledger. Protocol registration is complete under DOC-006; implementation
and execution tasks remain TODO. The mapping is: package 2 → M3REP-001;
package 3 → M3REP-002/003; package 4 → M3REP-004/005; package 5 → M3REP-006;
package 6 → M3REP-007/008; package 7 → M3REP-GATE. The upstream `rclib`
weight accessor of Section 12 (C1) is `UP-007`, and the canonical execution
environment of Section 12 (C10) is `M3REP-009`; both must land before
M3REP-003.

| Order | Work package | Acceptance evidence |
| --- | --- | --- |
| 1 | Lock this protocol and register the approved scope | Owner-approved D1–D8, global roadmap update, and implementation task IDs |
| 2 | Resolve and freeze the six-configuration panel | Digest-verified source parameters, deterministic selection, and immutable manifest |
| 3 | Add exact-repetition recipes and numerical controls for both output formulations | Literal-copy/reset tests, both equivalences within each formulation, residual reconstruction, refit checks, and legacy compatibility |
| 4 | Add the paired pilot runner and perform a timing smoke check | One complete configuration across arms/counts, matched replay conditions with the new velocity limit, historical-limit diagnostics and abort timestamps, measured resource usage, resumable records |
| 5 | Execute the full fixed panel | All 120 behavioral configurations accounted for, plus 36 S-effective fits; failures retained |
| 6 | Generate and reproduce the report | Complete paired tables, reproducible task-relative visuals with negative-time warm-up, numerical checks, provenance, and a clean-checkout reproduction note |
| 7 | Review the interpretation and decide follow-up | Owner accepts positive, negative, or inconclusive findings and decides whether a broader search is warranted |

The timing smoke check uses `feasible-best` (trial 17); its resulting records
belong to the fixed panel and are reused when the full run resumes. It is not
an extra opportunity to change the panel, thresholds, or augmentation anchor.
If measured resource needs invalidate the budget estimate, record the new
estimate and discuss a budget change before expanding execution. Slow or
unsuccessful runs do not justify dropping configurations.

Owner approval covers implementing and executing this bounded development
pilot through its report, subject to the acceptance and budget gates. It does not
authorize a 500-trial search, a reservoir-seed robustness panel, model freeze,
or confirmatory execution. A favorable pilot would still require a separately
specified follow-up protocol. The locked recovery confirmatory suite remains
unread and unexecuted by this experiment.

## 10. Cost and possible follow-up

The approval-time engineering estimate was approximately 2–4 working days for
a developer familiar with this codebase, excluding study execution and owner
review. The implementation audit of 2026-09-09 (Section 12) found that the
pilot also needs infrastructure the repository does not have: a deterministic
fixed-panel runner, a persisted replay bank keyed by the evaluation limits,
terminal checked-state retention, fit/evaluation timing and memory
measurement, task-relative plotting separate from the frozen recovery
figures, typed per-comparison reproduction tolerances, parameterized evidence
pointers, a versioned recipe schema, and an upstream `rclib` weight accessor
with a pin advance. The revised planning estimate is approximately 6–9 working
days, to be updated with measured figures in the M3REP-005 report. Episode
repetition itself is small; the existing residual path can be reused, but
needs paired recipe/evaluation coverage and separate increment/position
reporting. A minimal numerical pilot alone could take about half to one day,
but would not fulfill the complete report/reproduction scope above. These are
planning estimates, not measured implementation times.

The maximum behavioral workload is 15,600 RC evaluations
($120\times65\times2$), plus up to 390 unique replay evaluations, which can
be shared across output formulations with identical evaluation conditions.
Compared with the initial draft, this adds 42 behavioral configurations
(5,460 maximum RC evaluations) and 18 numerical reference fits. First-failure
stopping will usually reduce that workload, but the higher velocity threshold
can allow more scenarios to run and increase execution time relative to the
6 rad/s protocol. The diagnostics add no separate behavioral sweep. Literal training uses up to
26,000 loss rows per fit; training work and memory increase with $K$, while
the closed-loop simulation length does not.

For context, a read-only inspection of the historical timing-only study's
Optuna timestamps found approximately 2.79 hours of summed trial duration
for 500 trials, with 20.1 seconds per trial overall and 32.4 seconds for the
134 feasible trials. These historical durations exclude setup/reporting and
are not a benchmark of the new repeated training path. Allow several hours
for the pilot provisionally, then use the timing smoke check for a concrete
estimate; do not multiply the entire old runtime by 65.

Measured (M3REP-005, 2026-09-10, canonical execution environment): the
`feasible-best` configuration across all 20 behavioral arms took 268 s of
wall time for 1,053 runs (130 replay, 923 RC; every fit served from the
cache), 264.5 MiB peak resident set size, and 125.5 MiB of run storage; RC
runs average 0.22 s and replay runs 0.19 s including persistence. Scaled to
the six entries and three warm-ups with no early stop, the full panel is at
most 1.02 h of wall time and about 1.9 GiB of run storage, with about 0.84 h
remaining after the smoke check
([`timing_smoke_check_v1.md`](timing_smoke_check_v1.md)). M3REP-006 still
requires the owner's budget approval on that figure (C8).

At the final review, decide whether a larger matched search is scientifically
useful. Since exact repetition at fixed parameter is equivalent to changing
$\alpha_0/K$, searching count and ridge parameter independently can introduce
redundant dimensions and alter the effective regularization range. A future
search must state whether it tests literal episode-count matching, effective
regularization, or varied-data coverage, and lock its budget and matched
sampling strategy before execution. No 500-trial study is implicit in this
pilot plan.

## 11. Approved decisions

The owner approved D1–D8 on 2026-09-09, including the simulation-only 12 rad/s
hard limit and unchanged 0.05 rad/s dwell requirement clarified immediately
before approval. Material changes require a recorded owner decision before
execution; failures do not authorize retuning these choices.

| ID | Decision | Approved choice |
| --- | --- | --- |
| D1 | Identity and scope | `task_1a_repetition_v1`, development-only fixed pilot; keep previous experiment evidence unchanged |
| D2 | Episode counts, output formulations, and ridge controls | $K=17,33,65$; absolute S, R, R-scaled, both augmentation families, and S-effective; residual S, R, R-scaled, and S-effective using only original demonstrations; no residual augmentation |
| D3 | Configuration panel | Six fixed historical trials: 17, 136, 53, 1, 0, 28; no replacement based on new outcomes |
| D4 | Augmentation matching | Fixed $\sigma=0.05$, $\phi=0.99$, $\gamma=1$, seed bank 1, existing paired generator/taper, attempt factor 4 |
| D5 | Evaluation and velocity relaxation | Existing 65 development scenarios and both frozen trackers; simulation-only 12 rad/s per-joint hard abort, with 6 rad/s crossings recorded by phase; unchanged other gates, training-validation limits, and first-failure stopping; new config/cache identities; no confirmatory data or hardware changes |
| D6 | Numerical acceptance | Bitwise identity for repeated arrays/states; prediction comparison `atol=1e-8 rad`, `rtol=1e-8`; normalized solve residual at most `1e-10`; diagnose failures without automatic relaxation |
| D7 | Initial execution and budget | Literal repetition; 120 behavioral configurations plus 36 reference fits across both output formulations; trial 17 timing smoke check covers both formulations and counts toward that panel; no full search in this approval |
| D8 | Deliverables and follow-up | Versioned recipes, evidence, tests, reproducible HTML report with task-relative time and negative-time warm-up in all time series and animation clocks, and clean-checkout reproduction; owner review decides any larger study |

The global roadmap and task ledger now register this approved scope. The
developer should start M3REP-001, update task status and evidence as work
proceeds, and preserve all historical recovery records. This approval does
not approve the separate task 1-b draft or authorize any broader search.

## 12. Implementation clarifications

The owner settled C1–C9 on 2026-09-09 after a read-only audit of the
implementation against Sections 2–8, and C10 the same day after the
M3REP-001 review. They refine, and do not alter, D1–D8. They are recorded in
the affected `docs/TASKS.md` acceptance criteria before the work they govern
starts.

| ID | Topic | Clarification |
| --- | --- | --- |
| C1 | Readout weights | The pinned `rclib` Python binding exposes no readout weights. Add a read-only weight accessor with binding tests upstream in `rclib`, then advance the pin in a separate commit after verifying that fitting and predictions are unchanged. The Section 6 coefficient and normal-equation diagnostics use that accessor; reconstructing weights from `predict` on basis vectors is not the authoritative diagnostic. Documentation names the `cholesky` solver option as an LDLT factorization. M3REP-001 proceeds meanwhile and distinguishes historical source provenance from the pilot's implementation revision. |
| C2 | Velocity override | The recovery dataset is digest-bound to `configs/tasks/task_1a.toml`, so the 12 rad/s abort is a separately versioned evaluation-config setting (`simulation.velocity_abort = [12.0, 12.0]`) applied to simulation and to every diagnostic and feasibility check that uses the resolved evaluation limit. Training, augmentation validation, and dataset binding keep the legacy 6 rad/s scenario. The new recipe schema binds the training-validation limits and the relevant scenario/config digest; legacy records keep their serialization and hashes. |
| C3 | Reproduction scope | Full evidence verification with subset re-simulation: refit all 156 prescribed fits, verify every stored evidence payload, recompute metrics and report tables from stored arrays and terminal-state records, and re-simulate a deterministic subset whose selection rule is frozen before pilot execution and covers both formulations, both trackers, all warm-up values, repetition and scaling controls, perturbation classes, and available failure examples. Unavailable coverage is reported explicitly. This is not full simulation reproduction. |
| C4 | HTML report | Follow the DOC-005 approach: a tracked hand-written narrative `overview.html`, evidence-binding regression tests, and generated figures, tables, and assets. Reproduction verifies the bindings and generated outputs, not the narrative bytes. Manual browser checks recorded in the ledger suffice. Interactive synchronized cursors are not required; paired animations must still represent matching task times. |
| C5 | Animation clock | Export-side shifting behind a new explicit option (`time − T_w`), legacy export defaults unchanged, run-clock timestamps preserved in evidence. The player's `t = …` label is acceptable when the caption states task time in seconds with warm-up negative and activation at zero. No `skelarm` pin advance solely for that label. |
| C6 | Persistence | One Git pointer per model-configuration evidence manifest and per replay bank; individual run arrays stay external. Each manifest binds every constituent artifact's identity, digest, size, resolved conditions, and completion or failure status. Resume works at individual-run granularity without overwriting completed immutable evidence. |
| C7 | Replay censoring | Recovery-v1 behavior is preserved: a failed replay blocks the paired RC evaluation and stops that model's sweep. Such models are labelled replay-blocked, kept in the overall accounting with subsequent pairs marked unexecuted, and reported separately from models that failed an RC gate. |
| C8 | Budget | No numerical budget is invented from "several hours". M3REP-005 reports measured runtime, memory, storage, the projected full-panel cost, and the revised engineering estimate; owner approval precedes M3REP-006. Peak RSS is reported as process-cumulative; fresh worker processes are used where per-fit comparisons are needed. |
| C9 | Coverage | No new coverage exclusion is pre-approved. Reproduction logic is tested against small deterministic fixture stores, including corruption, missing artifacts, failures, and resume. Any thin orchestration layer that still needs an exclusion is proposed separately with its exact scope; the threshold stays unchanged. |
| C10 | Canonical execution environment | Results on the owner's machine depend on the core type a process starts on: the original task 1-a `--from-evidence` reproduction deviates by exactly 1.608e-10 when pinned to the E-cores and passes when pinned to the P-cores ([probe record](execution_environment_probe_v1.md)). P-core pinning is therefore the canonical execution environment of this machine for evidence generation, numerical comparisons, timing measurements, and canonical reproduction, under five conditions. (1) Apply the affinity before Python starts; child workers inherit and verify the restriction; the CPU numbers are never hard-coded as a portable definition of P-cores. (2) For new pilot runs set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before importing numerical libraries and verify effective thread counts where available; the historical reproduction's environment stays documented separately. (3) Add a versioned execution record holding the requested and effective CPU affinity including workers, the CPU model and logical-CPU/core-type mapping, numerical-library versions, the loaded BLAS implementation and selected architecture/kernel where exposed, effective thread counts, the relevant environment variables, and the launch command; bind its digest into new run evidence and execution-cache identities while legacy provenance serialization, recipe hashes, and frozen records stay unchanged. (4) The experiments establish core-affinity-dependent numerical reproducibility; OpenBLAS dispatch is the leading explanation until backend diagnostics support it; the P-core and E-core commands and outputs are retained, and the discrepancy is never described as universally fixed. (5) Tolerances and historical exact comparisons stay unchanged: affinity defines the canonical environment and does not establish bitwise reproducibility across machines, so mismatched execution environments are disclosed, never treated as equivalent. Implemented and tested as M3REP-009 before M3REP-003; the same policy governs M3REP-005's timing estimate and M3REP-006's execution, and the owner budget decision after the smoke check still applies. |
| C11 | Accepted numerical exception | The Section 6 validation ([`numerical_validation_v1.md`](numerical_validation_v1.md)) passed 71 of 72 equivalence comparisons; the comparison of R against S-effective for feasible-middle, absolute output, $K = 65$ exceeds the approved prediction tolerance (largest difference 2.93e-8 rad) and is retained as failed. Owner decision (2026-09-10): (1) retain the original absolute R/K65 fit of feasible-middle for behavioral evaluation; do not exclude it, substitute S-effective, or change its solver or regularization. (2) The comparison remains failed: the 71/72 result, `all_passed = false`, and the tolerances stay as recorded; task closure is described as "validation completed; one diagnosed numerical exception accepted", never as "all numerical checks passed"; the exception is bound to this exact comparison and evidence identity (validation sha256 `40720b539903`, fits `6912bffcc9a5` and `cc62950ed9a0`, diagnosis sha256 `857a373409e6`) and permits no unrelated failure. (3) The equivalence claim is limited: the exact-arithmetic ridge identity remains valid, but this pair did not demonstrate numerical agreement within the approved tolerance; the caveat is carried into subsequent results, and the small prediction discrepancy does not establish that closed-loop behavior is unaffected. (4) The diagnosis is qualified: the evidence supports sensitivity to floating-point accumulation and solution in an ill-conditioned problem, but the diagnostic reconstructs the normal equations in NumPy while `rclib` assembles them with Eigen, so the reported solution-path part can include assembly differences and is not an isolated measurement of LDLT roundoff, and the extended-precision result is a reference solution, not an exact solution. The fixed panel is preserved and the failure is a reported finding; M3REP-004 may begin after this decision is recorded; M3REP-006 still requires the post-smoke-check budget approval. |

Confirmed readings: the `failure-actual-dwell` panel category matches any
`dwell:*` first-failure head, and prescribed ridge parameters are constructed
directly without changing historical search bounds. Two cautions apply to the
implementation gaps found by the audit: retain the terminal checked state
separately without fabricating unavailable controller telemetry, and report
the DOC-005 discrepancy through an actual original task 1-a reproduction
rerun, not as a recovery-evidence problem.
