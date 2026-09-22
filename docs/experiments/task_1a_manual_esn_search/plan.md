<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Task 1-a: ESN parameter search on manual demonstrations

**Status:** Approved by the owner on 2026-09-22.
Implementation tasks are registered as M3MS-001 through M3MS-GATE in the
[work queue](../../TASKS.md#manual-data-esn-search--m3ms). No implementation or
search has started.
**Created / revised:** 2026-09-22.

This follows the closed
[manual-demonstration experiment](../task_1a_manual_demonstration/report/report.md).
It does not revise that experiment's result or reopen its gate. The owner has
requested an Optuna search of ESN parameters and warm-up on M10, using nominal
performance alone, followed by the five-arm comparison on the frozen evaluation
cases. Tracker and filter settings are inherited and not searched.

## 1. Search question and fixed conditions

Optimize the ESN trained on all ten accepted demonstrations (M10) for nominal
reaching and holding. After selecting and freezing configurations, compare
S, M10, R10, C10 and Replay at those same configurations. Neither singleton
results nor perturbation results enter parameter selection.

Retain the ten accepted recordings, preprocessing, 100 Hz grid, frozen physical
input transform, absolute next-position output, equal total loss weight per
episode and count-scaled ridge rule. Retain the 30 s horizon, final continuous
dwell predicate, generated-reference checks, joint-speed abort and torque
limits. Force cases in final evaluation retain the existing state-triggered
pulse policy. No task or safety criterion is relaxed during tuning.

Keep both existing trackers (`pd_v2` and `computed_torque`) and their gains fixed.
Tracker identity, gains, preprocessing filters and causal estimator cutoffs
are absent from the Optuna search space. Fixing them isolates ESN tuning; it
does not establish that their performance effects are negligible.

The owner selected the historical v4 policy, also used by configuration E.
Use the exact stored values, not their rounded display values:

- velocity cutoff: 29.980411525699598 Hz;
- acceleration cutoff: 10.938122239871603 Hz;
- maximum timestep ratio: 3.0.

Source: [`task_1a_nominal_v4.toml`](../../../configs/evaluations/task_1a_nominal_v4.toml).
This one policy applies to every candidate and both trackers. Warm-up is
searched separately, including zero, rather than inherited as a fixed second.

## 2. Search space

The owner approved these ESN ranges, including the expanded input-scaling upper
bound. They are an initial search domain, not an assertion of optimal bounds.

| Parameter | Approved domain |
| --- | --- |
| Reservoir size | 100 to 400, step 50 |
| Spectral radius | 0.8 to 1.3 |
| Sparsity | 0.5 to 0.98, using the existing library's convention |
| Leak rate | 0.01 to 0.3, logarithmic |
| Input scaling | 0.02 to 1.5, logarithmic |
| Base readout regularization `alpha_0` | 0.001 to 1, logarithmic |
| Warm-up | Categorical: 0, 0.25, 0.5, 1 and 2 s |

Approved warm-up domain: inherit the earlier categorical set
`[0, 0.25, 0.5, 1, 2]` seconds. Zero explicitly means no warm-up phase.
Apply the selected warm-up consistently to training preparation and evaluation
through the existing protocol's mechanisms.

Use the existing Optuna TPE machinery. Preserve bias handling, solver and the
manual recipe's explicit-bias weighted fit. `alpha_0` remains the base
regularizer; each arm uses the inherited episode-count scaling.

Reservoir seed **896 is fixed**, as approved. It is not an Optuna parameter.
Optuna's sampler seed is a separate, recorded reproducibility setting to be
chosen in a new namespace before the search. No multiple-reservoir-seed study
is part of this scope.

## 3. Nominal-only optimization and information separation

Each successfully fitted M10 candidate is evaluated on the nominal starting
posture under both fixed trackers: **two RC runs per candidate**, each with a
30 s active horizon. Every run starts from a fresh reset and uses the existing
complete success predicate. A failed or aborted run counts as unsuccessful;
an infrastructure interruption is retained and resumed rather than converted
into a task failure. A fit failure is retained as a failed trial and consumes
a trial slot.

The objective is the nominal success fraction:

`successful nominal tracker runs / 2`, maximized.

This yields 0, 0.5 or 1, not an estimate over repeated random trials. No
perturbed scenario is simulated for the optimizer. No perturbed score enters
its objective, tie-break, pruning, stopping, promotion or selection. Replay
and the other training arms also remain outside the optimizer's score.

The optimizer and its workers receive a nominal-only evaluation specification.
Their actual scenario lists must equal `("nominal",)`; a parent-side preflight
alone is insufficient. The implementation must test both the parent and real
worker entry points against accidental expansion to all 65 cases.

The nominal objective is intentionally coarse: different trajectories can
have the same score, and many candidates may achieve 1. Do not silently add
settling time, tracking error or perturbed performance to distinguish them.
Retain the **three highest-scoring configurations**, ordered by descending
nominal success fraction and then ascending trial number, as presented in the
owner's configuration-count decision. Use distinct ESN/warm-up parameter points;
if a point is repeated, its earliest trial represents it. Select only fitted
candidates with both nominal runs completed or judged as task failures; an
infrastructure interruption or fit failure is not a scored configuration.
If fewer than three distinct scored candidates exist, report the shortfall
without increasing the 100-trial cap. Selecting an early successful anchor
under this rule is an expected possibility, not evidence of a search failure.

Freeze the selected configuration identities and selection rule before the
final perturbation evaluation begins. Final results must not cause a new
selection, replacement, search-range change or additional adaptive tuning in
this experiment.

The 64 perturbed cases are withheld from this optimizer, but were used in the
closed predecessor study and their results are already known to the research
process. Describe them as withheld from this search, not as a newly drawn,
previously unseen confirmatory set. The owner explicitly excludes new-scenario
validation from this scope.

## 4. Comparison after the freeze

At each selected ESN/warm-up configuration, evaluate the existing 65 cases
(nominal plus 64 perturbed) under both fixed trackers. Keep training and
reporting conventions aligned with the closed experiment:

| Arm | Models or banks per configuration | Meaning |
| --- | ---: | --- |
| S | 10 models | One model for each accepted teacher |
| M10 | 1 model | All ten teachers together |
| R10 | 10 models | Exact copies of each parent, with equivalent regularization |
| C10 | 10 models | Each parent plus nine episodes from its frozen contractive bank |
| Replay | 10 parent banks before sharing | Each parent followed through the same tracker/filter/warm-up policy |

The selected ESN parameters and warm-up are common to all four learned arms.
Replay has no ESN, but shares their evaluation conditions. Retain the existing
contractive construction and seed bank; do not tune augmentation here. Report
paired outcomes per parent, tracker and scenario class, counting M10 once in
per-arm totals. Include all four planned contrasts and the parent-matched
replay comparisons, with unavailable results distinguished from failures.

Every successfully fitted model attempts every case from a fresh reset; an
unsafe run does not suppress later cases. The comparison is conditional on
parameters selected using M10 nominal performance. It is not a comparison of
separately optimized S and M10. The owner explicitly excludes separate
singleton searches, newly drawn scenario validation and multiple-seed
validation. No new human recordings or hardware operation is included.

Replay banks may be verified and reused across configurations only when parent,
warm-up, filters and all other evaluation conditions agree. Since warm-up is
now searched, not all selected configurations necessarily share replay banks.
Nominal M10 search evidence may be reused only if every run condition and its
stored evidence match the final evaluation contract.

## 5. Budgets and revised cost estimates

Approved search cap: **100 total Optuna trials**, including queued anchors and
timing-pilot candidates. Failed candidates count; interrupted work resumes
under its original trial rather than creating a free replacement. Approved
operational limits were **10 hours elapsed execution and 20 GiB of new search
artifacts**. For the expanded final comparison, conservatively apply the same
limits as a shared ceiling across the search and subsequent comparison; do not
add another allowance or enlarge either limit. This tighter combined accounting
is an implementation budget policy, not a claim that the owner approved extra
resources. Keep accumulated evidence and stop scheduling work at a cap;
reserve enough headroom to persist an in-flight result. Count execution time
and artifact usage across resumed invocations rather than resetting the caps.
The fixed trial cap is not increased if the nominal-only search is inexpensive.

The earlier five-hour estimate assumed 65 scenarios per candidate. It no longer
applies to the newly approved nominal-only objective. The
[measured timing report](../task_1a_manual_demonstration/timing_smoke_check_v1_corrected.md)
provides approximate costs of 1.37 s per RC run, 1.42 s per replay run and
0.22–1.10 s per M10 fit in the inherited panel.

| Stage | Maximum runs before reuse | Approximate serial cost | Approximate run telemetry |
| --- | ---: | --- | ---: |
| 100-trial nominal M10 search | 200 RC | 4.6 min simulation; about 5–7 min including measured fit costs | 0.14 GiB |
| Final comparison, per selected configuration | 4,030 RC + 1,300 replay = 5,330 | About 2.1 h plus fit/verification overhead | 3.5 GiB |
| Final comparison, all three configurations | 12,090 RC + at most 3,900 replay = 15,990 | About 6.1 h plus fit/verification overhead | 10.5 GiB |

Search timing excludes startup, data loading, cache verification and reporting;
allow additional time and measure a bounded initial pilot within the 100-trial
cap. The table's storage column excludes fits and other artifacts. These are
estimates, not bounds; candidate parameters and aborted runs affect costs.
No fourfold parallel speedup is assumed: the prior four-worker benchmark
established only 1.03x end-to-end.

The approved three-configuration comparison contains 93 learned models and
at most 30 replay banks. Search plus comparison therefore schedules at most
**16,190 runs** before reuse: 200 nominal search RC runs, 12,090 final RC runs
and 3,900 replay runs. Shared warm-ups can reduce the replay count; verified
nominal reuse may reduce RC work. The projected run telemetry totals about
10.6 GiB, excluding fit and reporting artifacts. The combined time/storage
ceilings above apply even if overhead makes the estimate optimistic.

## 6. Implementation outline

1. Encode the resolved warm-up domain, tie rule and three-configuration
   selection; freeze the nominal-only protocol, budgets and final scope before
   execution.
2. Adapt the existing manual recipe and evaluator to sampled ESN/warm-up
   configurations under a new protocol. Keep closed manifests unchanged.
3. Reuse Optuna scheduling with resumable identities, explicit nominal-only
   worker scopes, source digests and cumulative budget enforcement.
4. Test success scoring, ties, zero warm-up, fixed filter/seed invariance,
   weighted training, interruption handling and exclusion of perturbation
   inputs before a bounded timing pilot inside the candidate budget.
5. Run the approved search, freeze selection, then execute and audit the
   five-arm comparison without feeding perturbation results back to Optuna.
   The reporting assistant interprets the resulting evidence for the owner.

## 7. Decision record

All owner decisions below were received on 2026-09-22.

| Decision | Recorded choice |
| --- | --- |
| Training scope | Search M10 first; comparisons follow the configuration freeze |
| Filters and trackers | Historical v4/E filtering; both existing trackers and gains fixed |
| D1: objective | Maximize nominal-class success rate only; perturbations unknown to optimizer |
| D2: warm-up | Search the inherited categorical set: 0, 0.25, 0.5, 1 and 2 s |
| D3: reservoir randomness | Fix seed 896 |
| D4: ESN bounds | Approved table above, with input scaling expanded to 0.02–1.5 logarithmically |
| D5: search budget | 100 total trials including anchor/pilot/failed candidates; 10 h and 20 GiB caps |
| D6: final comparison | S, M10, R10, C10 and Replay at the same three highest-scoring configurations; no separate singleton tuning, fresh-scenario or multiple-seed validation |
| Selection ties | Earlier trial number, after descending nominal success fraction |

The requested experiment choices are resolved. The protocol implementation
must preserve these decisions and enforce the declared scope and shared
resource ceiling. The [work queue](../../TASKS.md#manual-data-esn-search--m3ms) registers the
implementation tasks with status TODO. No computation has been launched and
no closed M3MAN evidence has changed.
