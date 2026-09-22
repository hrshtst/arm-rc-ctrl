<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Controller reference

[Roadmap](../PLAN.md) · [Work queue](../TASKS.md)

Detailed sections moved from the roadmap on 2026-09-22. Original section
numbers and historical implementation notes are retained so source-code
and configuration references remain interpretable. Current priorities and
gate status are in the roadmap/queue; experiment-specific decisions live
in each linked experiment plan. Read only the sections relevant to the task.

## 5. Initial ESN control formulation

### 5.1 Signals

At sample `k`, define the robot feedback and optional task condition as

\[
  s_k = [q_k^\mathsf{T},\; \dot q_k^\mathsf{T}]^\mathsf{T},
  \qquad
  u_k = [\bar s_k^\mathsf{T},\; c_k^\mathsf{T}]^\mathsf{T},
\]

where `q` is joint position, `dq` is joint velocity, the bar denotes the model
recipe's input transform, and `c` is a task code. The transform centers every
channel on its training-set mean; its scales follow the policy the recipe
declares: the training-set standard deviations (`training_std`) or one shared
physical scale per channel (`fixed_scale`, e.g. 0.3 rad for `q` and 4 rad/s
for `dq`), which keeps a barely moving joint from amplifying tracking jitter
into the reservoir. The transform is derived from the dataset's stored
statistics (Section 7.3) and recorded in the recipe; the canonical dataset
itself is unchanged by the policy. Task 1-a has no task-code dimensions
because it contains one fixed target. Multi-target experiments append a
one-hot target identifier.

The initial readout target is the next desired joint position:

\[
  y_k = q^{\mathrm{demo}}_{k+1}.
\]

Absolute next-position prediction is the only supported output representation
in task 1-a. Predicting increments or torque is reserved for later ablations.

### 5.2 Reservoir and readout

For a leaky random sparse reservoir,

\[
  x_{k+1} = (1-a)x_k
  + a\tanh(W_{\mathrm{res}}x_k + W_{\mathrm{in}}[1;u_k]),
\]

\[
  \hat q^d_{k+1} = W_{\mathrm{out}}[1;x_{k+1}].
\]

`rclib` constructs the fixed reservoir, and its readout consumes the reservoir
state only, with its own bias term. An input pass-through readout
\(W_{\mathrm{out}}[1;x_{k+1};u_k]\) is a separately named future ablation
(`readout-input-passthrough`), not the primary formulation. Offline learning
fits only the readout using ridge regression:

\[
  W_{\mathrm{out}}
  = \arg\min_W \|Y-XW\|_F^2 + \lambda\|W\|_F^2.
\]

The implementation must follow `rclib`'s bias convention exactly rather than
manually adding a second readout bias.

### 5.3 Completed M3 teacher forcing, priming, and dwell

The completed M3 task 1-a protocol used three contiguous intervals:

1. **Initial hold:** the teacher holds the initial posture. This supplies a
   deterministic reservoir washout/priming interval.
2. **Movement:** the demonstrated reaching motion.
3. **Final dwell:** the teacher holds the endpoint inside the target region so
   the ESN observes the desired equilibrium behavior.

During training, `u_k` is constructed from demonstrated state. Each episode
starts with a reset reservoir; the washout samples update the reservoir but do
not contribute to the ridge loss.

During evaluation, the low-level controller first holds the configured initial
posture while the reset ESN receives the measured state for the same priming
duration. The ESN then runs closed loop: its next input always contains actual
robot feedback, never its previously predicted state.

Episode boundaries may not be concatenated without an explicit reservoir reset.

### 5.4 Approved task 1-a recovery extension

`task_1a_recovery_v1` keeps one independent scripted demonstration but separates
acquisition pre-roll, reservoir warm-up, and task time. Preprocessing uses the
stationary pre-roll as filter and onset-detection context, then crops the derived
episode at the confirmed demonstration motion onset. The first cropped sample
$q_0^{\mathrm{ref}}$ is the task initial posture and the basis of every evaluation
offset; the pre-roll baseline never replaces it.

Each training episode and evaluation run independently resets the reservoir to
zero. A configurable common pre-task hold supplies warm-up only when $T_w>0$;
the approved development set is $T_w\in\{0,0.25,0.5,1.0,2.0\}$ s. At task time
zero, replay starts the cropped reference and RC first evaluates its readout.
Metrics and disturbances use this shared task clock.

Synthetic episodes add seeded, bounded AR(1) Gaussian position perturbations to
the one demonstration. Contractive augmentation decays those perturbations with
endpoint distance and forces them to zero during final dwell; a matched
non-decaying arm isolates that mechanism. Velocity is recomputed from augmented
position. Absolute next-position prediction remains primary, while a residual
readout is a gated exploratory ablation.

Run records distinguish the position-valued `generator_output_q`, the residual
arm's raw `generator_increment_q`, measured motion, and separate warm-up
telemetry. Selection requires common safety and dwell gates plus paired reduction
of the activation jump and early command gap; time-aligned trajectory RMSE is a
diagnostic rather than a success criterion. The experiment-specific document
defines the approved ranges, arms, splits, formulas, and confirmatory gate.

#### Approved repeated-demonstration follow-up

The owner approved D1–D8 of
[`task_1a_repetition_v1`](../experiments/task_1a_repeated_demonstration/plan.md)
on 2026-09-09. This separate development pilot retains the recovery dataset,
six fixed source configurations (trials 17, 136, 53, 1, 0, 28), common timing,
65 development scenarios, and both frozen trackers. It compares one original
episode with 17/33/65 exact copies and ridge-scaling controls; absolute-output
arms also include count-matched contractive and non-decaying augmentation.
Residual arms use only the original and exact copies. Reset the reservoir
per episode and fit once on stacked loss rows; validate the two ridge
equivalences separately within each output formulation before simulation.

The approved panel has 120 behavioral configurations and 36 numerical
reference fits, at most 15,600 RC evaluations plus 390 matched replay runs.
The simulation-only hard speed limit is 12 rad/s per joint, with historical
6 rad/s crossings recorded by phase. All other gates, including 0.05 rad/s
dwell speed, and the original training/augmentation validation limits remain
unchanged. New config identities must not alter legacy records or hardware
limits. New feasibility rates are not directly comparable with recovery v1.

Report paired outcomes, numerical errors, censored failures, resource use,
and provenance in reproducible HTML. All time-series plots and animation
clocks use task time, with warm-up at negative time and activation at zero;
inactive readouts remain missing. M3REP tasks in `TASKS.md` govern test-first
implementation, timing smoke check, full execution, reproduction, and owner
review. No 500-trial search, model freeze, confirmatory suite, or hardware
operation is authorized by this pilot approval.

#### Approved manual-demonstration follow-up

The experiment is closed. The owner accepted the assistant-authored
[human-readable interpretation](../experiments/task_1a_manual_demonstration/report/report.md)
and its reproduction evidence at M3MAN-GATE on 2026-09-22 and recorded a
negative finding for the one-versus-ten question, with copies neutral,
synthetic contractive variation unreliable, and the recordings themselves
trackable; the decision and its follow-up deferrals are in
[plan section 12](../experiments/task_1a_manual_demonstration/plan.md#12-gate-decision-2026-09-22)
and the ledger row. No model is selected, and further recordings, tuning or any
confirmatory study need a separately approved plan. The historical protocol
below records the approved design rather than current task status.

The owner approved
[`task_1a_manual_v1`](../experiments/task_1a_manual_demonstration/plan.md), including
D1–D7 and the final recorder controls, on 2026-09-15. This remains task 1-a:
ten human-guided IK recordings of the simulated 2-DOF arm share the exact
reset posture and target. Full recordings retain the pre-roll from the first logged sample, including natural fluctuations, and natural transient paths/durations. Saving does not require an online
experiment-specific quality check. Validate batches offline and repeat
collection until ten takes pass, with no total attempt cap. The 30 s
per-recording timeout is separate. Discard practice files/history before
study collection; retain all saved study takes, including rejected ones.

Extend the existing `skelarm` recorder upstream where appropriate: Space
starts, S saves, Shift+S saves and prepares the next take, Q closes with an
unsaved-take warning, and F is removed. R discards only unsaved data/trails,
resets posture/velocity, and leaves recording paused; saved files/trails
survive. S then R must produce the same state as Shift+S, including next-take
numbering. Add CLI base filenames, numbered outputs without save dialogs or
plots, and optional current/faint saved tip trails. Integrate reviewed
upstream changes through a separate pin update and dependency rebuild.

At each of six inherited ESN configurations, train all ten singleton models,
one all-ten model, ten singleton-copy controls, and ten models with nine
contractive additions to a singleton: 186 models total. Use absolute next
position, separate episode resets/warm-up, and equal total loss weight per
episode despite unequal lengths. The weighted ridge objective fixes effective
regularization; verify copy equivalence before interpreting results. Synthetic episodes keep each parent's exact first sample and final dwell with versioned envelopes that ramp in from the first sample. Whole-bank copies, fixed-alpha diagnostics, additional held-out
human recordings, new tuning, and a confirmatory study are deferred.

Use both frozen trackers and 65 development scenarios, with a common 30 s
evaluation horizon: at most 24,180 RC runs and 7,800 replay runs, 31,980 in
total. Success requires bounded motion and at least 1 s continuous final
target dwell within 1 cm at joint speeds no greater than 0.05 rad/s. The starting velocity bound
is 6 rad/s per joint; any stricter acquisition-pilot bound is frozen before
study collection and used consistently. Force cases apply a 12 N, 0.2 s pulse
after 0.5 s qualifying target dwell and require recovery afterward. Abort
individual unsafe runs and attempt subsequent scenarios from fresh resets.
Version the new timing, weighting, and evaluation contracts rather than
reusing historical config identities or assuming cropped-dataset semantics.

The developer delivers validated machine-readable evidence, reproducible
plotting/animation assets or tools, and a reproduction audit. The reporting
assistant working with the owner interprets that evidence and authors the
human-facing report; a developer-written narrative does not satisfy this
deliverable. DOC-007, UP-008–009, and M3MAN tasks in `TASKS.md` govern the work.
Existing task 1-a evidence and the separate, unapproved task 1-b proposal remain
unchanged. The present update registers the approved work; implementation and
recording have not begun.

Implementation clarifications I1–I9, recorded in the experiment plan's
Section 9 on 2026-09-15 after the implementability review, qualify that
registration. The 30 s horizon projects to roughly 13 hours of serial
simulation and 26–27 GB of run data at the previous pilot's rates, so bounded
process-based parallel execution with a serial-versus-parallel equivalence
check is explicit scope. The recorder's acquisition clock is defined upstream
and verified in the acquisition pilot before 100 Hz is claimed. The weighted
ridge fit uses an explicit ones column with the library's implicit bias
disabled; a new recipe schema version preserves historical semantics;
completion is judged against the configured horizon; the experiment receives
its own task and evaluation configuration identities; M3MAN-003 delivers a
thin recorder launcher with a tested adapter; the input transform copies the
historical scripted-data centers and scales; and augmentation seeds carry a
stable parent identifier. The inherited zero-phase filter measurably shifts a
held start, so the boundary-preserving preprocessing requirement stays with a reproducing test.

Pilot-1 revisions I10–I14, approved on 2026-09-15 and recorded in the experiment plan's Section 10, follow the first excluded practice
pilot, in which movement began immediately after the first sample and sample
gaps grew with the number of saved trails drawn. The first logged sample stays
exactly at the reset posture while natural pre-roll fluctuations are kept and
smoothed without depending on a stationary hold or introducing a derivative spike at the first sample; the recorder shows only the
current trail and the most recently saved trail; takes are recorded at 50 Hz with
actual timestamps and reconstructed onto the 100 Hz training grid shared by every comparison arm, with
gap and frame checks from the acquisition period and a final dwell of one actual
second; and the contractive envelope ramps in from the first sample. A second
short practice session verifies timing and preprocessing before the settings are
frozen.

### 5.5 Desired derivatives and low-level tracking

The target generator returns desired position at every control sample. A causal,
stateful derivative estimator computes desired velocity and acceleration using
backward differences followed by configurable low-pass filtering. It must:

- reset at episode start;
- emit zero desired velocity and acceleration on its first sample;
- use measured sample intervals and reject non-positive or excessive intervals;
- expose its raw and filtered values in telemetry;
- have one implementation contract shared by training evaluation and C++ parity
  tests.

Two low-level controller combinations are evaluated:

- RC target generator + joint-space PD;
- RC target generator + computed-torque control.

The initial vertical slice uses PD first. Computed torque is added only after the
PD data path and evaluation protocol pass their integration tests.

## 6. Fair baseline protocol

The baselines receive the demonstrated trajectory directly as their reference:

1. joint-space PD trajectory replay;
2. computed-torque trajectory replay.

Comparison is paired by low-level controller:

- RC+PD versus replay+PD;
- RC+computed torque versus replay+computed torque.

Controller gains are tuned on direct replay before ESN tuning, then frozen.
ESN tuning may not change the baseline gains. All paired methods use identical:

- robot model and integration step;
- initial condition and disturbance realization;
- joint, velocity, torque, and endpoint limits;
- demonstration preprocessing;
- run duration and metric definitions;
- development and confirmatory seed sets.

Tuning effort is recorded. Conclusions must distinguish differences caused by
the target generator from differences caused by the tracker.
