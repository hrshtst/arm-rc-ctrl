<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# M3MS implementation notes

What each implementation task built, kept beside the experiment so the task
queue can stay a queue. The [plan](plan.md) is the approved protocol and the
[review rounds](review_rounds.md) record the owner's findings; this file records
the shape of the code. Nothing here authorizes execution: the real bank has
seen the search (M3MS-004/005) and nothing perturbed.

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
constrains a running worker and not only the gap between trials. Each attempt
is opened on disk before its worker starts: the parent is the only thing
measuring it, so an attempt whose parent is killed is charged on the next
resume by the wall clock it was in flight rather than given away. A trial drawn
but lost before any parameter reached the study is abandoned explicitly, with a
retained failed outcome, because the cap counts the trials that were spent.

Stored bytes are discovered, never reported. Ownership of a run is recorded
before its payload is published: `write_run` takes a `claim`, called with the
run's identity once the run is accepted and before the staging directory is
renamed into the runs bucket, and the evaluation keeps those claims beside its
progress record. The parent therefore charges the fit cache, the evidence
directory and every run that directory owns — including a payload whose
progress entry never landed — plus the bytes staged before any claim. Staged
bytes are charged to the search itself, once per ledger and for as long as they
are in the store: no outcome counts them, and finalizing the trial that wrote
them does not remove them. A progress or claims record that cannot be read is
refused rather than counted as no runs.

A report is not evidence. The parent refuses one that names another trial, and
a scored candidate must point at a model evidence manifest that verifies
against its digest, carries the reserved identities, and satisfies
`verify_model_evidence` — the one implementation the sweep's resume and the
audit also use — against inputs the parent reconstructs from the protocol and
the study. Recorded identities cannot vouch for the bindings beside them, so
the cached recipe and weights are re-read and the manifest is checked whole;
every run is then loaded through the reader a resume uses, and the successes
are recounted from those runs. A scored candidate cannot be finalized without
the protocol and those inputs at all.

A fault in the store is not a verdict on a candidate.
`EvidenceIntegrityError` lives with the fit cache, the lowest layer that serves
stored evidence, and both the fit readers and the run readers raise it, so
corrupted evidence propagates out of the worker, leaves no report and keeps
the trial recoverable; only a numerical refusal is a failed candidate. Every
read of a cached payload passes through one guard, because a payload that
cannot be decoded is as much a fault in the store as one whose digest
disagrees, and the decoder's own `ValueError` would otherwise be read as a
verdict.

The parent keeps no preamble of its own. Its fit inputs come from
`prepare_runner`, the path a worker uses, which requires the canonical
environment and loads the numerical runtimes before the environment is probed:
the parent's execution identity has to be the identity the study was frozen
under, and a parent that probed before loading rclib's OpenMP runtime is
refused by the very study it is resuming.

### What the tests hold

`tests/unit/test_manual_search_run.py` (48 cases): the nominal scope is refused
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
given the remaining allowance as its timeout. The third round's findings are
held the same way: a parent killed mid-attempt is charged the wall clock its
attempt ran, a run published before its progress record is still charged, an
unreadable inventory is refused rather than read as empty, an abandoned trial
spends the slot it took, a real trial's evidence is refused once its cached
weights no longer match their digest, and a corrupted cache propagates instead
of being scored as a failed candidate. `tests/unit/test_run_record.py` holds
the claim itself: a run is claimed before its payload is published, and a
refused run is never claimed. The fourth round's findings need a fresh
interpreter and the store: a parent that starts with nothing loaded verifies a
real trial's evidence and scores it, staged bytes stay charged once the trial
is finalized, and an undecodable cache — not only one whose digest disagrees —
propagates out of the worker rather than failing the candidate
(`tests/unit/test_manual_numerics.py` holds the same guarantee at each of the
three cached payloads).

## M3MS-004 — the bounded timing pilot

The pilot is the approved search stopped early, not a study of its own.
`search --stop-at-trials N` draws no new trial once N are finalized, but
finishes pending work first. The bound is checked against the cap in force
(`pilot_bound_mismatches`), so it can only stop the search sooner. Pilot trials
therefore consume the 100-trial cap, and an invocation without the bound
continues after them under the next numbers. The pilot takes **10 trials**
(20 nominal runs at most). All ten fall inside the sampler's 20 random start-up
trials, so they are drawn exactly as the search would draw them and TPE has
not begun modelling. No anchor was queued: the frozen protocol declares none.

What the estimates omit is measured where it is spent. The worker's report
carries a `TrialTiming`: preparing the study and environment, the fit (or a
cache hit's verification refit), and the sweep with the simulation and
persistence it encloses, plus the run bytes it wrote. Sums are floats, because
a retried worker served stored evidence measures no sweep, and an integer zero
is a report the strict reader refuses. The parent records each finalizing
attempt's worker wall clock. It also **charges its own verification** to the
trial: that is elapsed execution, and before M3MS-004 the ledger omitted it.
If verification is interrupted, the attempt is still open on disk, so the
existing wall-clock recovery charges it.

Each parent invocation leaves a `SearchInvocation` record under
`armrc://reports/task_1a_manual_search/invocations/`, even when it raises. The
record holds the invocation's wall clock and the ledger's growth over it. The
difference is the parent overhead (start-up, study bookkeeping,
reconciliation) that the per-trial ledger does not charge.

`arm_rc_ctrl.experiments.manual_search_pilot` holds the rest. `preflight`
states what a bounded search will schedule before it runs and never overwrites
a stated plan. `report` then compares that statement with the retained
records, the derived ledger and the Optuna study's trial states. It lists
every disagreement and exits non-zero if there are any. The report divides
each trial's charged time into start-up, preparation, fit, simulation,
persistence and verification. It gives the parent overhead and the bytes the
ledger does not count (the records and the study database), and projects the
remaining search and the comparison that shares the ceiling. The projections
are estimates; the ledger's caps remain what stops the work.

Tests: `tests/unit/test_manual_search_pilot.py` covers the bound (inside the
cap, continued by the search, pending work finished first), verification
charging, invocation records including a failed invocation, the preflight
before and after spend, agreement and each kind of disagreement, and a
portable, reproducible report. `tests/unit/test_manual_search_run.py` checks
the worker's timing over the fixture study, including a retried worker served
stored evidence.

### The pilot's result (2026-09-23)

Ten trials ran at `5a6cde2`, pinned to the P-cores, in one invocation of
114.6 s wall clock. [Preflight](pilot/preflight_v1.json) and
[observation](pilot/timing_pilot_v2.md) agree on every count: 10 finalized and
none pending, 20 nominal runs, `COMPLETE 10` in the study, and the nominal
scenario alone. The ledger holds 10 trials, 111.8 s and 15.1 MiB. The parent
spent 1.4 s outside every charged trial, and 139 KiB of records and study
database are uncharged; both are negligible against the ceiling and recorded
rather than charged.

A trial costs about 10.5 s. Only about 2.8 s of that is the simulation the
plan's estimate counts. Preparing the study and demonstrations takes 5.6–5.9 s,
interpreter start-up about 1.1 s, and the fit 0.4–2.0 s (it grows with
reservoir size). The parent's first verification took 5.7 s because it loads
the study's fit inputs once; later verifications take 0.01 s. A nominal run
costs 1.36 s to simulate and persist, which agrees with the plan's measured
1.37 s, and stores 0.75 MiB. At these rates the 100-trial search needs about
0.31 h and 0.15 GiB. The comparison's runs and fits alone are estimated at 6.1 h
and 11.6 GiB, which leaves about 3.5 h and 8.2 GiB of the shared ceiling for
the comparison's own start-up, verification and reporting. These figures are
estimates, not bounds: nominal runs do not bound the cost of perturbed ones.

Six of the ten trials (1, 2, 4, 7, 8 and 9) scored the maximum of 1.0. These
are distinct points, and under the frozen rule (descending score, then
earliest trial) no later trial can outrank trials 1, 2 and 4. The plan
anticipated this outcome in section 3. It is recorded here because the
remaining 90 trials can no longer change which configurations M3MS-005 would
freeze.

After the owner's review, per-run costs are divided by the runs a worker
actually simulated: runs served from the store are recorded but not measured.
A trial abandoned before any parameter was recorded is reported with its
parameters unavailable. The first report is kept in
[`pilot/superseded/`](pilot/superseded/README.md).

## M3MS-005 — the finished search and the freeze

`search_stopped` is the one statement of when the search has ended: a spent
cap, or no allowance beyond the persistence headroom. `run_search` stops on
it, and `arm_rc_ctrl.experiments.manual_search_freeze` refuses to freeze
until it holds and no reservation is pending.

The freeze also refuses a study that disagrees with the retained records:
different trial numbers, a state or value the records do not give, a trial
still running, or parameters that are not the reserved point. It ranks with
the protocol's own `select_configurations`, so the order is descending nominal
score, then the earliest trial, one per distinct point, and a shortfall is
reported rather than filled. It then verifies each chosen trial's evidence
again against reconstructed inputs, and the recounted verdict must equal the
recorded one.

The record, `ManualSearchFreeze`, refuses to exist when it breaks the rule:
wrong ranks or order, a repeated point, a score that is not its count, a
missing configuration without a shortfall, a label other than "highest
nominal scores", or no stop reason. An edited freeze therefore cannot be
loaded. `freeze_digest` is what the comparison will bind.

Tests: `tests/unit/test_manual_search_freeze.py` and, against the fixture
study's real evidence, `test_a_frozen_trial_is_verified_again_against_its_evidence`
in `tests/unit/test_manual_search_run.py`.

### The search and the freeze (2026-09-23)

The search ran its remaining 90 trials in one invocation from clean
`7eabf48`, pinned, in 16 min 19 s of wall clock. It stopped because the
100-trial cap was spent, having charged 1,084 s and 145.6 MiB. All 100
candidates were fitted and scored: 70 at 1.0, 4 at 0.5 and 26 at 0.0. The
[freeze](freeze/selection_v1.md) was taken from clean `526d79e` (digest
`7e97649720f2`). Its three highest nominal scores are trials 1, 2 and 4, as
the pilot anticipated, with no shortfall, and each was verified again against
its evidence. The three configurations use different warm-ups (1, 0.25 and
0.5 s), so the comparison cannot reuse a replay bank across them.
