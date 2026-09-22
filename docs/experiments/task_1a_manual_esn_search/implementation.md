<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# M3MS implementation notes

What each implementation task built, kept beside the experiment so the task
queue can stay a queue. The [plan](plan.md) is the approved protocol and the
[review rounds](review_rounds.md) record the owner's findings; this file records
the shape of the code. Nothing here authorizes execution: no search of the real
bank has run.

## M3MS-001 — the frozen protocol

`configs/studies/manual_esn_search_v1.toml` (digest `029739812525`, sampler seed
20270301) is loaded and validated by `arm_rc_ctrl.experiments.manual_search`. It
binds the six approved ESN ranges, the categorical warm-up set including zero,
reservoir seed 896, both trackers, the inherited v4 filter policy, the
nominal-only objective, the three-configuration selection rule, the approved
caps as a ceiling shared with the comparison, serial trials, and the sequential
five-arm comparison scope.

`scope_mismatches` is the single enforcement point: `ManualSearchProtocol`
refuses to exist while it returns anything, so a protocol that tunes a fixed
condition, widens the optimizer's scenarios, raises a cap, points at a study
outside the repository or restates a filter value that no longer matches its
source is refused where it is loaded. Only the resource caps may be tightened.
`select_configurations` is a pure function over scored trials: descending score,
then earliest trial, one configuration per distinct parameter point, unscored
candidates never selected, evidence validated before it is ranked, and a
shortfall reported rather than filled.

## M3MS-002 — training at a sampled configuration

`arm_rc_ctrl.experiments.manual_sampled` expresses a sampled configuration as a
`StudyConfiguration` — the record the frozen study already uses — so the
existing recipes, conditions and evidence paths take it unchanged and there is
no second training path. A trial's parameters become a checked `SampledPoint`,
and `point_mismatches` is applied both where parameters are read and where a
configuration is built, because that is where a point becomes trainable.

The configuration carries the sampled reservoir with the fixed seed and the
fixed cutoffs; `fixed_policy_mismatches` re-checks that invariance afterwards.
Labels come from a `search-t####` namespace the closed study's six words cannot
occupy. `sampled_esn` builds each arm's ESN through the study's own
`esn_for_arm`, and `contractive_bank` inherits the frozen contractive
construction rather than regrowing one.

## M3MS-003 — the resumable search

`arm_rc_ctrl.experiments.manual_search_run` is both the parent and the worker.

The parent opens or resumes one Optuna study keyed by the protocol digest
through the existing `open_study`, asks for one trial at a time, draws a point,
and spawns a worker **process** per trial (`spawn_trial`, which is also the seam
the tests replace). The worker rebuilds the configuration from the point it was
given, checks the scenarios it actually received, fits the all-ten arm and
evaluates the nominal case under both trackers, then reports verdicts only —
never a trajectory.

One change made that possible without a second path: `ManualFitInputs` resolves
`sampled` configurations beside the frozen ones and refuses a label that would
shadow one, `prepare_runner` accepts them, and the accounting and
expected-ESN checks read the resolved configuration. The search therefore
evaluates through the closed experiment's own runner, and the frozen manifest is
never written to. `study_model` is published for the same reason, with a test
that it still reproduces a frozen entry's fit identity.

Each trial is **reserved** before its worker starts: a record under
`armrc://reports/task_1a_manual_search/trials/trial-NNNN/` names the trial, the
configuration and the exact point. A resumed search finishes every pending
reservation under its original number before it draws anything new, so
interrupted work is never replaced by a fresh trial. The ledger is **derived**
from those retained records — an outcome charges its trial, an interrupted
trial charges the cost that was measured — so an interruption between
finalizing a trial and accounting for it cannot lose the spend.

A worker that reports a fit failure has failed a trial and consumed its slot; a
worker that writes no report has been interrupted, which leaves its reservation
pending for recovery. The remaining elapsed allowance, less the headroom kept
to persist an in-flight result, is the worker's timeout, so the ceiling
constrains a running worker and not only the gap between trials.

A report is not evidence. The parent refuses one that names another trial, and
a scored candidate must point at a model evidence manifest that verifies
against its digest and belongs to this trial's configuration; the successes are
recounted from that manifest's own runs. Stored bytes are measured from the
directories the worker actually wrote — each run's payloads, the evidence
manifest and the fit — including for a candidate that failed.

### What the tests hold

`tests/unit/test_manual_search_run.py` (33 cases): the nominal scope is refused
when widened, replaced, emptied or repeated — in the module, in `evaluate_trial`
and at the real worker entry point, through a subprocess that writes no report;
a worker report that cannot be true is refused; the ledger accumulates, refuses
negative spend and honours a tightened cap; the parent spends its budget once
across resumes and a failed worker consumes its trial; the worker evaluates
exactly two nominal runs over the fixture study; and a sampled fit reproduces
bitwise in a fresh interpreter, which is why workers are processes. The resume
protocol is held to the same standard: a failed fit consumes its trial and is
charged, an interruption — a crash or a reached deadline — leaves the trial
pending and resumes under its own number, the ledger follows the records rather
than a stored total, a spent cap schedules no worker at all, and the worker is
given the remaining allowance as its timeout.
