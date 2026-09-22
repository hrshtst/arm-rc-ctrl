<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Evaluation reference

[Roadmap](../PLAN.md) · [Work queue](../TASKS.md)

Detailed sections moved from the roadmap on 2026-09-22. Original section
numbers and historical implementation notes are retained so source-code
and configuration references remain interpretable. Current priorities and
gate status are in the roadmap/queue; experiment-specific decisions live
in each linked experiment plan. Read only the sections relevant to the task.

## 9. Metrics and evaluation

### 9.1 Task 1-a metrics

For `N` aligned movement samples and `d` joints, the primary metric is joint
trajectory RMSE:

\[
  \mathrm{RMSE}_q =
  \sqrt{\frac{1}{Nd}\sum_{k=1}^{N}\|\operatorname{wrap}(q_k-q_k^{demo})\|_2^2}.
\]

Report per-joint RMSE as well as the aggregate. Angular differences use the
project's joint-angle convention; wrapping is applied only to continuous joints.

During the final dwell window report:

- endpoint error mean, RMS, maximum, and 95th percentile;
- fraction of samples inside the target tolerance;
- longest continuous in-tolerance duration;
- joint velocity RMS and maximum;
- torque RMS, peak, saturation fraction, and control effort
  `integral(sum(tau**2), t)`;
- success/failure and a structured termination reason.

Trajectory metrics compare the fixed-duration demonstrated motion. No dynamic
time warping is used for the primary result because it can hide timing error. It
may be reported as a labeled diagnostic only.

### 9.2 Task 1-a recovery metrics

The recovery experiment retains task success, safety, dwell, effort, and
saturation metrics, but evaluates its new mechanism from simultaneous task
activation. Primary paired diagnostics are the initial desired-command jump and
the integral of desired-to-actual command gap over the first 0.5 s. Also report
generator deviation from the original reference, restoring alignment, endpoint
settling and dwell, smoothness, torque, and every failure. A qualifying model
must reduce both early metrics consistently relative to replay while its
generated reference and actual motion converge to the common target. The exact
eligibility and lexicographic freeze rules are locked in the experiment plan.

### 9.3 Robustness protocol

Evaluate in this order:

1. exact demonstrated initial posture with no disturbance;
2. small joint-space initial-posture perturbations;
3. larger held-out perturbations;
4. repeatable finite-duration endpoint force pulses during motion;
5. combined posture and force perturbations.

Perturbation grids, force timing, directions, magnitudes, and random seeds are
versioned configuration. A pilot using the frozen direct-replay baseline selects
nontrivial but safe levels. After the confirmatory suite is declared, those
values and seeds are locked and may not be used for tuning. The scenarios are
a pure function of a protocol's levels and seeds (stable IDs; random posture
directions from an independent seeded stream per class), so every method runs
identical scenarios; the suite persists every run, keeps failures in the
per-class aggregation, and takes paired RC-minus-replay effects over the
scenarios where both runs of a pair succeeded, reporting the failed pairs next
to them. Development levels and seeds (`configs/evaluations/*_robustness_dev_*.toml`)
exercise the suite on a frozen recipe before the one-shot confirmatory run.

### 9.4 Later-task metrics

- **Task 1-b:** endpoint target-region dwell success, final error, settling time,
  path length/efficiency, effort, and success from unseen initial postures.
- **Multiple targets:** the same measures per target plus switch settling time,
  peak/RMS acceleration, integrated squared jerk, and discontinuity at switching.
- **Periodic curves:** phase-aligned endpoint RMSE, nearest-curve geometric RMS
  and Hausdorff-like 95th-percentile error, lap-period drift, closure error, and
  recovery time after perturbation.

## 10. Hyperparameter tuning

[Optuna](https://optuna.org/) manages algorithmic ESN tuning. The versioned
search protocol (`configs/studies/esn_search_*.toml`) bounds reservoir size,
spectral radius, sparsity, leak rate, input scaling, reservoir seed, ridge
regularization, and the derivative-filter cutoffs of the causal estimator. In
the completed M3 study, washout was the demonstration's prime phase and the
input transform stayed at its pilot-selected recipe value. The recovery study
instead tunes the approved common pre-task duration including $T_w=0$, while
keeping reset and activation semantics identical across arms and episodes.
Labelled comparison points (the development anchor at the M2
ridge value 1e-2 and at 3e-2, 1e-1, 3e-1) are evaluated before sampling.
Low-level tracker gains are excluded from ESN studies after baseline
qualification.

Task 1-a uses a seeded sampler and pruner. The objective is median movement
joint RMSE across development scenarios. A trial is infeasible and receives a
documented penalty if it diverges, violates configured state/torque limits,
terminates early, fails the configured final-dwell constraint, exceeds the
protocol's saturation bound, or cannot be trained. Scenarios are evaluated in
protocol order and stop at the first infeasible one (the objective is already
decided); the running objective is reported to the pruner after every
feasible scenario. All objective components — per-scenario termination,
movement RMSE, dwell criteria, saturation, boundary jump, and the reason —
are logged separately; the scalar objective is never the only saved result.
Only trials feasible in every development scenario are eligible for selection
and freezing — a study without one selects nothing — and, because Optuna counts
queued comparison points towards the sampler's start-up trials, a protocol
states its start-up count inclusive of them. Before a selection is frozen, a
reservoir-seed sensitivity panel re-evaluates the leading feasible trials with
a predefined list of seeds (everything else unchanged) and records how many
seeds stay feasible and the spread of their objectives; a frozen recipe must
also pass the held-out development robustness suite, otherwise the previous
recipe is retained and the failure documented.

Development/tuning scenarios and seeds are separate from confirmatory scenarios
and seeds. The selected recipe is frozen before confirmatory evaluation. Reusing
confirmatory outcomes to alter hyperparameters creates a new study/version and
invalidates the earlier confirmatory label.

Alternative candidates considered:

- Ray Tune is useful for distributed workloads but unnecessary initially.
- Hydra can compose large configuration trees, but typed TOML keeps the initial
  stack aligned with `skelarm` and `rtctrl`.
- Weights & Biases provides hosted tracking, but local MLflow avoids requiring a
  third-party account and keeps research data local by default.
