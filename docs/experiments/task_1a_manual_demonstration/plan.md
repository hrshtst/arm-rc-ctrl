<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Task 1-a: One Versus Ten Manual Demonstrations

- **Experiment label:** `task_1a_manual_v1`
- **Status:** APPROVED on 2026-09-15, including D1–D7, the final recorder
  controls, and report ownership. Implementation clarifications I1–I9 were
  recorded on 2026-09-15 after the implementability review (Section 9).
  The upstream recorder work, the dataset contract and batch validation, and the recorder launcher are implemented, and the first excluded practice pilot was recorded on 2026-09-15. Pilot-1 revisions I10–I14 (Section 10) were approved on 2026-09-15 with implementation clarifications; no study take has been recorded.
- **Registration:** At the owner's explicit request on 2026-09-15, registered
  in [PLAN.md](../../PLAN.md) and [TASKS.md](../../TASKS.md) under DOC-007,
  UP-008–009, M3MAN-001–012, and M3MAN-GATE. Start with UP-008; the ledger
  defines implementation dependencies and evidence gates.
- **Predecessor:** [Repeated-demonstration experiment](../task_1a_repeated_demonstration/plan.md)
  and its [completed findings](../task_1a_repeated_demonstration/overview.html).

## 1. Question and scope

Does training on ten separately recorded human demonstrations improve
closed-loop reaching compared with training on one demonstration, when all
ten start from the same fixed posture, have naturally varied transient motion,
and settle at the same fixed target for a specified minimum duration?

The revised task emphasizes bounded, realistic movement and sustained target
dwell. The operator chooses when to move, how long the reach takes, and the
intermediate path. Retain the full recording from its first logged sample, including any natural pre-roll fluctuations (I10); neither
motion-onset cropping nor prescribed move/dwell clock windows are required.
The hypothesis is that varied transients leading to a common target equilibrium
can improve robustness. This is an expectation to test, not an assumed result.

The previous task 1-a repetition pilot used one scripted, planned teacher
trajectory. Its accepted conclusion was that benefits from exact repetition
were consistent with reduced effective ridge regularization; copying supplied
no additional motion information. Here, human variation is the experimental
factor. Exact copies control for episode count, and contractive synthetic
episodes provide an alternative source of variation.

The approved study uses manual guidance of the **simulated, horizontal,
gravity-free 2-DOF arm** under D1. No physical recordings are required. The
[task 1-b proposal](../task_1b_multi_demonstration/plan.md) varies training
start postures; this study keeps the nominal training start fixed and remains
task 1-a. Nearby evaluation postures are robustness probes, not additional
training starts. No online adaptation is proposed.

Positive, negative, and inconclusive results are all valid. Ten recordings
from one operator and session cannot establish generalization across people,
sessions, targets, or robots.

## 2. Recording protocol

Use the robot and target from
[`configs/tasks/task_1a.toml`](../../../configs/tasks/task_1a.toml):

| Quantity | Approved setting |
| --- | --- |
| Initial joint posture | `q_start = [0.2, 1.2]` rad, reset before every take |
| Endpoint target | `[0.10, 0.45]` m, same elbow branch on every take |
| Acquisition / training period | 50 Hz acquisition with the actual timestamps retained (I13); every comparison arm uses the frozen 0.01 s (100 Hz) training/control grid, reconstructed from the actual timestamps (support for finer grids does not change this experiment's grid); the second practice pilot verifies the realized acquisition timing (I2, I13) |
| Recording/task start | First logged sample at the exact reset posture, before accepting manual movement |
| Acquisition pre-roll | Retained in full; move when ready. No stationary interval is required: natural fluctuations immediately after the first logged sample are part of the demonstration, subject to the usual motion limits (I10) |
| Movement duration/path | Operator-chosen, subject to validity and velocity limits; no route corridor or prescribed pace |
| Target dwell | Approved offline acceptance rule: at least 1.0 s continuously inside 1 cm with joint speeds at most 0.05 rad/s; checked after collection, never a condition for saving in the recorder; the duration is one actual second regardless of the acquisition rate (I13) |
| Accepted sources | Ten separately performed takes, `D01` through `D10`, in acceptance order |
| Operator/session | One operator, same interface and settings; record session and take order |

This experiment receives its own task and evaluation configuration
identities (proposed spellings `configs/tasks/task_1a_manual_v1.toml` and
`configs/evaluations/task_1a_manual_dev_v1.toml`). They copy the robot
geometry, limits, and target values above and carry the continuous dwell rule
and no fixed prime/move/dwell intervals. The historical `task_1a.toml`, its
90 % occupancy rule, and its fixed intervals stay unchanged because committed
datasets and evidence are digest-bound to it (I6).

The pinned [skelarm teaching interface](../../../third_party/skelarm/docs/guides/teaching_trajectories.md)
supports mouse-guided IK or dynamics recording and exports `.sklog.npz` logs.
Use **IK guidance** for this first dataset, recording the resulting joint
angles as the teacher signal. This is human-authored kinematic motion, not
measured physical dynamics. Validate dynamic trackability separately with
both frozen trackers. The pinned recorder reads its own skeleton and task schema (link limits in
degrees, a task `type`, a target table) and does not accept the arm-rc-ctrl
task TOML; this repository has no recorder launcher yet. M3MAN-003 delivers a
thin launcher (`scripts/record_demo.py`, the name already reserved in the
PLAN.md layout) and a tested adapter in `src/` that builds the recorder's
skeleton and task from the versioned task configuration through the existing
scenario conversion (`build_skeleton`), verifying the posture, units, robot
limits, target, sampling options, and output settings before acquisition (I7).

### 2.1 Recorder improvements

Extend the existing
[`skelarm` trajectory recorder](../../../third_party/skelarm/tools/trajectory_recorder.py)
for consecutive takes in one app session. The following six capabilities are
requested scope for acquisition readiness. Generic UI, recording, and drawing
changes may be implemented upstream in `skelarm`; this repository owns the
experiment configuration, dataset import/validation, and provenance. No new
standalone recorder app is proposed. Save shortcuts are **S** and **Shift+S**
as requested by the owner. New CLI spellings below remain proposals, not
options already implemented.

Before implementing changes in an upstream repository, create a dedicated
feature branch there for the intended PR. Keep each PR's changes and tests
on that branch, and record the branch name and base revision in task evidence
so a PR can be created later. Apply this to UP-008/UP-009 and any additional
generic dependency fixes; dependent PRs may use explicitly documented stacked
branches. The downstream submodule pin update remains a separate change.

The pinned recorder currently starts on the first grab, has **Finish (F)**
which saves and closes the window, accepts a single filename through
`--output`, and shows a post-recording plot only when `--plot` is requested.
It already binds **Q** to close, but its close handler automatically saves;
it has no **R** reset/discard shortcut. Replace Finish/F with the controls
below and replace automatic save-on-close with an unsaved-take warning.

| ID | Required capability | Proposed behavior and acceptance |
| --- | --- | --- |
| REC-1 | Save and next take | Save the current take, keep the app open, and reset to the configured initial posture with zero velocity, cleared drag/force state, a fresh log and reset clock. Show the next take number and wait for recording start. |
| REC-2 | Keyboard recording start | **Space** starts recording immediately while the arm is stationary, logging the reset state at `t = 0` and continuing through the actual pre-roll. A Start button invokes the same action. Retain first-grab start as an optional mode; use explicit keyboard/button start for this study. |
| REC-3 | Keyboard save and save-next | **S** stops and saves the take while keeping it visible; **Shift+S** invokes Save and next take. Neither requires Ctrl. Remove Finish/F, use **Q** to close with an unsaved-take warning, and **R** to discard/reset as specified below. Provide matching buttons and visible shortcut labels. If the take is already saved, save-next advances without writing it again. |
| REC-4 | CLI filename and uninterrupted saving | Keep `--output` for an exact single-take filename; with a proposed `--multi-take` option, treat it as the base for numbered files. For example, `--output reach.sklog.npz --multi-take` produces `reach_001.sklog.npz`, `reach_002.sklog.npz`, etc. Save directly without a filename dialog or a plot/diagram window appearing on each save. |
| REC-5 | Current tip locus | A proposed `--show-tip-trail` option displays the current take's accumulated robot-tip path during recording. Draw the logged/FK tip positions, not the cursor path. R discards an unsaved trail; a saved trail remains in the past-trajectory history after reset. |
| REC-6 | Faint past trajectories | A separate proposed `--show-past-trails` option retains completed, saved takes' tip paths in the same session, drawn with transparent, faint line colors behind the current take. Preserve them across resets to help the operator follow prior paths; distinguish the current path clearly and allow either overlay to be hidden independently. A display-history option limits the faint overlay to the most recently saved take (I12); the study uses the current trail plus the last saved trail, and every saved recording and its provenance are kept. |

The keyboard workflow is **Space → guide and dwell → Shift+S → Space**
for consecutive takes, or **S** to save and inspect the current take.
Starting a take must not require moving the robot or clicking a dialog.
In explicit-start mode, keep the robot at its reset posture until Start;
holding Space or pressing it again during recording must not restart the
clock, add an extra take, or toggle an undocumented pause. Shortcuts should
work with focus on the drawing canvas and must not fire repeatedly through
keyboard auto-repeat or conflict with text-entry controls. Distinguish plain
S from Shift+S explicitly so one keypress cannot invoke both actions.

**Acquisition clock (I2).** The pinned recorder advances a nominal clock by
one 20 ms GUI tick, performs one IK update per tick, and then writes every
elapsed sample boundary with that same posture, so `--sample-rate 100`
produces repeated samples and misleading raw-increment velocities. UP-008
defines the acquisition clock: when poses are updated, how delayed ticks are
recorded, and how display refresh relates to sampling. Changing the GUI timer
to 10 ms alone does not guarantee genuine 100 Hz acquisition. The recorder
must never fabricate intermediate measurements by assigning several
timestamps to one pose update. The acquisition pilot verifies the realized
sample spacing and pose-update rate before the acquisition rate in Section 2 is accepted. Pilot 1 (2026-09-15) recorded at 100 Hz and found sample gaps of 40–63 ms that grew with the number of saved trails drawn, consistent with increasing trail-rendering cost although that cause is not established; acquisition therefore moves to 50 Hz with the last-saved-trail display (I12, I13), and the second practice pilot verifies the timing.

The recorder captures and saves demonstrations; experiment acceptance belongs
to the offline validation pipeline in `arm-rc-ctrl`. Do not hard-code this
experiment's 0.05 rad/s dwell criterion, minimum dwell duration, or acceptance
decision into the upstream recorder. Neither Save nor Save and next take
waits for an experiment-specific condition. No experiment-specific live dwell
indicator is required for REC-1–REC-6. Preserve the recorder's existing joint
limits and other generic safeguards; this separation does not disable them.

Use visible ready/recording/saved states. In multiple-take mode, reaching the
duration cap stops and saves the current take and leaves the app open for
inspection/advancement. Save-next resets only after a successful save; explicit
R discards only an unsaved take and preserves any saved take and trail.
A save failure must retain the unsaved take for retry,
and an empty take must not create a
file or consume a take number. Never overwrite an existing recording: detect
filename collisions and report them before replacing any bytes. Close after
a successful save must not write the same take again. Remove the Finish
button and F binding in both single-take and multiple-take modes.

The recorder controls are:

| Key | Action | Result |
| --- | --- | --- |
| Space | Start | Start a new recording from the ready/reset posture, including its pre-roll samples |
| S | Save | Keep the window open with the saved take visible; no reset |
| Shift+S | Save and next take | Keep the window open, reset the arm, and prepare a new take |
| R | Reset; discard only if unsaved | Discard an unsaved take and its trail, or preserve a saved take and its trail; reset posture and velocity and wait without recording for Space |
| Q | Close | Warn if a take is unsaved; otherwise close without writing another file |

**R** has two cases:

- **Unsaved take:** discard that take's unsaved samples and unsaved trail.
  Do not save it or add it to past-trajectory history.
- **Saved take:** keep both its saved file and its trail. Preserve the trail
  in the faint past-trajectory display after reset, with no removal or duplicate
  history entry. Previously saved takes and their trails are also untouched.

In either case, restore the configured initial posture and zero velocity,
clear active drag/force state, and prepare a fresh empty log with a reset clock.
Recording stays paused/ready until Space starts a new take.

**S followed by R must have the same result as Shift+S:** save exactly once,
retain the same saved file and trajectory history, prepare the same next take
number/output filename, reset the arm, and leave recording paused. Share the
save and reset operations between these paths so their behavior cannot diverge.
Pressing R again while already ready must not remove saved history or consume
an additional take number.

Explicitly discarded unsaved takes are not accepted/rejected dataset records;
the retention policy still applies to every saved study take.

**Q** and the window close button use the same close handler. If any recorded
data are unsaved, display a modal warning with **Save and close**, **Discard
and close**, and **Cancel**. Cancel is the default and leaves the app and take
intact; dismissing the dialog is also Cancel. Save and close uses the same
save path as S and closes only on success. A failed save leaves the window
and unsaved data available for retry. Discard and close requires that explicit
dialog choice. An empty or already-saved take closes without warning or
duplicate saving. Stop sampling while the warning is open; Cancel preserves
the prior recording state without adding dialog-wait time to the trajectory.

S and Shift+S use the same save operation, with no experiment-specific
acceptance gate. F has no recorder action. Normal saving remains free of
dialogs and plots; the warning is specific to closing with unsaved data.

The CLI base determines every output filename without a per-save modal
interaction. Show the saved path and take number in the existing window's
status area and command-line output. Keep post-recording plotting opt-in
through `--plot`; for this collection workflow omit it, and never open a
blocking plot between takes. The live arm view and optional trails remain
visible in the original recorder window. Filenames enumerate **attempts**,
not accepted dataset IDs: rejected take files keep their numbers and are
retained, with accepted `D01`–`D10` assigned later by the quality manifest.

Past-trail overlays use only already saved recordings from the current
session; no future take or generated plan is shown as a past demonstration.
Clear the pilot/practice history before study collection: discard practice
trajectory files and their display history after the acquisition checks.
Practice payloads are disposable and are not part of the experiment's retained
dataset. Use a separate practice namespace/output base so clearing it does not
affect any study take. Keep only the acquisition-readiness check summary and
resolved settings needed for the study; do not require practice-payload hashes
or resolvable practice-artifact pointers in the final bank manifest.
Once study collection starts, retain its saved take trails normally across
Save and next take. The first study take has no practice guide behind it,
while later takes may follow prior study traces. Failed/rejected **study**
takes and experiment runs remain retained as evidence; they are not practice.
Record overlay settings,
opacity/color policy, take order, and the source take IDs visible during each
new take; the first take naturally has no history. Freeze the display policy
before collection, with both overlays enabled under D1. These
are visual guidance aids: they never move the robot, alter logged samples, or
create a route-matching acceptance threshold. Following prior traces may
reduce variation and correlates successive takes; report this acquisition
condition and the observed diversity when interpreting the ten-take bank.

Recorder acceptance includes GUI/keyboard tests for stationary start, reset
and consecutive saves, shortcut repeat/focus behavior, unique filenames,
save failures and collisions, and no per-save dialogs/plots. Verify that
F no longer triggers an action; Q and window-close show the same warning
only for unsaved data; all warning choices, Cancel/default dismissal and
save failures preserve the specified state. Verify R during recording discards
only the unsaved take/trail; after save, R preserves the file and displayed
saved trail. Both cases restore the exact reset state and leave recording
paused until Space. Compare S then R against Shift+S for identical saved
data, history entries, next take number/filename, reset state, and paused
recording; check repeated R causes no extra numbering or history changes.
Verify that the visible current/history paths match their source samples, only saved
takes enter history, unsaved trails clear while saved trails survive reset, and
overlay toggles leave logged data unchanged. Exercise a complete ten-take
pilot session and import every output before study acquisition. Any upstream
change needs its own tests and documentation, followed by a separate submodule
pin update and dependency rebuild here; this plan does not claim those changes
have already been made.

### 2.2 Acquisition and acceptance

Perform a short acquisition pilot, excluded from the ten study takes, to
check start reset, logger timing, target display, and IK branch continuity.
Use the explicit-start workflow above: press Space at the reset posture,
then move when ready; no stationary interval is required after the first logged sample (I10). Do not
manufacture a missing first sample afterward. Move naturally toward the
displayed target, optionally following the faint past traces, and hold still
there for a comfortable interval before pressing S or Shift+S. Holding for
roughly a few seconds provides margin for the approved one-second offline
dwell test; precise timing or monitoring joint velocity is not required of
the operator. Any extra actual hold remains part of the recording. The app
saves the take and advances on request, without deciding whether it qualifies
for this experiment.

Encourage natural variation in transient path, speed, pauses, and arrival
time within the same valid reaching task. No pilot guide trajectory or
joint-history similarity threshold selects the takes. A generous recording
timeout of 30 s, approved under D2, is an operational bound rather than a prescribed
movement duration; check its practicality in the excluded acquisition pilot
and lock it before the ten study takes.

**When acceptance happens:** record an initial batch of ten takes, then import
and validate that batch offline, before any model training. The experiment
pipeline produces per-take measurements and pass/fail reasons against rules
frozen before study collection. Saving a file is not accepting a demonstration.
If fewer than ten qualify, collect an additional batch for the shortfall and
validate it under the same rules. Repeat until ten accepted demonstrations
are available. There is **no total attempt cap**: the earlier proposal to
stop at 20 attempts is removed. The 30 s per-take timeout is independent of
the number of attempts. This avoids an accept/reject interaction after every
take. Every saved study attempt remains in storage whether it passes or fails,
and subsequent model performance cannot change the acquisition verdict.

The offline acceptance rules, including the 0.05 rad/s stationarity
criterion, are approved under D2. That value describes sufficiently
stationary target holding for this experiment, not a generic recorder rule
or the allowed velocity during movement. Freeze the acquisition predicate
after review and the acquisition pilot, before collecting the study bank:

- Every recording begins at the exact simulator reset posture `q_start` with
  zero velocity. Task time zero is recording start, even if movement begins later. Movement may begin immediately after the first sample; natural fluctuations are kept and checked against the usual limits (I10). Verify the first logged state against the reset state using only
  justified numerical serialization tolerance, not a physical start-offset
  allowance. No 0.005 rad cropping tolerance is needed.
- Retain the same IK branch, continuous joint paths, finite timestamps and
  samples, and the task's position/velocity/workspace limits. Reject jumps, missing intervals, or invalid data rather than clipping or filling them. Frame-count and gap limits are expressed in actual time from the acquisition period, not from the training grid (I13).
- Reject unrealistic joint velocities, including IK jumps. D3 approves the
  canonical 6 rad/s per-joint bound for validation and evaluation; a stricter
  common bound may be established from the acquisition pilot before study
  collection, with the final bound recorded explicitly.
  Smoothing must not hide raw jumps or excessive speeds. Check raw increments
  against their actual time differences as well as processed derivatives;
  passing a kinematic speed check alone does not prove dynamic trackability.
- The final uninterrupted target dwell must last at least 1.0 s, with every
  sample inside 1 cm and every joint speed at most 0.05 rad/s. The duration is one actual second regardless of the acquisition rate: it is evaluated on the reconstructed training grid, where one second at 100 Hz is 101 consecutive samples, and the raw recording must cover it without a gap beyond the frozen limit (I13). Compute these
  measurements offline from the saved data. Record endpoint and final-posture
  dispersion; identical target does not mean identical final samples.
- Record joint-displacement overlays, velocity profiles, movement/hold
  durations, and time to target. Require distinct source takes and
  diagnose duplicates by payload/array hashes. Variation should be visible
  above recording resolution; if it is negligible, report that limitation
  rather than manufacturing variation or replacing takes based on RC results.

Select the first ten passing takes in acquisition order after batch validation.
Never choose among passing takes by learned-controller performance. Retain all
rejected study takes and reasons, and repeat collection for any shortfall.
Replay failures after dataset lock remain experimental outcomes and do not
permit replacing a recording. Avoid selecting the ten “best” RC teachers.

## 3. Preprocessing and dataset lock

Preserve raw logs unchanged and retain the entire recorded episode, including the pre-roll from the first logged sample. Task time zero is the first recorded reset
state. Movement onset may be annotated for descriptive plots, but it neither
changes the clock nor determines which samples enter training. No human
motion-onset confirmation is required to define this task.

This intentionally differs from the
[recovery timing semantics](../task_1a_state_conditioned_recovery/plan.md).
Its cropped move/dwell-only dataset contract cannot be reused unchanged.
Provide an explicit full-recording, variable-duration dataset/recipe contract,
while retaining reusable import, validation, and preprocessing components.

Reconstruct each take from its actual timestamps onto the frozen 0.01 s training grid shared by every comparison arm (I13); interpolation supplies intermediate reference values but recovers no motion that was not sampled, and derivatives come from the smoothed reconstructed trajectory.
Apply one smoothing and derivative policy to all takes, fixed from the
acquisition pilot without examining learned-controller outcomes. Derive
velocity from the processed joint positions when absent from the IK log;
never treat cursor coordinates or an IK target as logged joint velocity.
Inspect raw versus processed motion and record filter boundary handling.
The preprocessing policy must preserve the recorded initial reset state;
use an explicitly documented boundary-preserving method if smoothing would
otherwise shift it. Verify that constraint in the acquisition pilot, rather
than silently snapping or translating a filtered trajectory.
The inherited zero-phase Butterworth filter with odd-extension padding does
not preserve the first sample of a hold-then-reach signal: a check on
2026-09-15 with the existing smoothing function measured a start shift of
about 9.5e-5 rad after a 0.1 s hold and 8.3e-10 rad after a 1 s hold. The
requirement stands, and M3MAN-002 includes a reproducing test of that shift
beside the boundary-preserving method (I9). Pilot 1 showed natural movement starting 0.01–0.36 s after the first sample, so the hold-anchored method, which needs a stationary interval longer than its margin, is superseded: the smoothing must keep the first sample exactly at the reset posture, without introducing a velocity or acceleration spike there, while smoothing the trajectory that follows, without depending on a stationary hold, and must keep the recorded pre-roll fluctuations rather than replacing them with a constant posture (I11). The second practice pilot verifies it.
Offline teacher preprocessing may use future context; online feedback and
generated-reference derivative estimation remain causal.

Episode durations and row counts may differ. For episode i, retain its full recording from the first sample, including any pre-roll fluctuations, the transient movement, and the final dwell; let `L_i` be its number
of next-step loss rows. All these rows enter training. Do not time-warp,
average, translate, or pad trajectories to a common length. Do not append
invented stationary training tails. Store the measured final-dwell interval
and its predicate, independently of optional descriptive phase annotations.

Lock a versioned bank manifest with ten raw/processed IDs and digests, source
session/take IDs, quality decisions, full retained intervals, sampling policy,
row counts, dwell annotations and any diagnostic phase masks,
per-array hashes, measured starts, target, acquisition-readiness summary, recorder
version/options, output-attempt mapping, and displayed past-trail sources. Any
needed bank or timing schema extension is future implementation work; do not
silently reinterpret existing single-demonstration record fields.

## 4. Training arms and regularization controls

Run **all ten choices of singleton**, not one hand-picked demonstration.
Let `S_i` train on `D_i` and `M10` train on `D01,…,D10`. The ten singleton
results quantify sensitivity to which demonstration was available; `M10` is
one model per reservoir configuration, reused in those paired comparisons.

The current trainer fits one readout on stacked rows with summed squared
error plus `alpha * ||W||²`. Repeating data K times at fixed solver alpha
therefore reduces effective regularization by K. Unequal recording lengths
also need explicit weighting so a longer take does not dominate the bank.

Propose **equal total weight per episode**, with objective

`(1 / K) * sum_i (||X_i W - Y_i||² / L_i) + lambda * ||W||²`,

where K counts episode occurrences, including copies. Set
`lambda = alpha_0 / 400` to preserve the inherited panel's regularization
scale; 400 is only the historical row-count reference, not a required length
for new takes. Implement the equivalent weighted summed-loss fit using
per-row weights `400 / L_i` and `solver_alpha = K * alpha_0`. Weight the full
readout feature vector, including its bias, and the target consistently.
This weighting capability must be verified or implemented; plain unweighted
stacking of variable-length episodes is not equivalent.

The pinned `rclib` readout accepts no per-sample weights, exposes no weight
setter, and carries its bias as a virtual column that receives the same ridge
penalty as the features. Scaling rows by the square root of the weight while
keeping that implicit bias does not give the objective above. The approved
route (I3) needs no `rclib` change: disable the library's implicit readout
bias, append an explicit ones column to every harvested feature row in this
repository, multiply the entire feature row (including that column) and its
target by the square root of the row weight, and fit once with
`solver_alpha = K * alpha_0`. At inference append an unscaled one to the
feature vector. Every prediction path, including batch prediction and
coefficient extraction, uses the explicit-bias layout; the existing
implicit-bias path stays available for historical recipes.

Weighting changes the fit, not the reservoir dynamics: harvest each complete
episode in its original order before applying loss weights. Keep all phases
in the loss with uniform weight within an episode. Report hold/movement/dwell
row fractions and resulting loss weights: equal episode weighting does not
equalize phase proportions, and long stationary intervals can influence the
learned behavior. It does prevent one long recording from receiving more
total weight solely because it has more rows.

Record solver alpha, lambda, raw row counts, row weights, unique-source count,
copy count, and synthetic count separately. The ten-episode comparisons match
episode count and total loss weight, not necessarily raw sample count.

| Arm | Episodes | Unique manual sources used | Solver alpha | Role |
| --- | --- | ---: | --- | --- |
| `S_i`, i = 1…10 | `D_i` once | 1 | `alpha_0` | One-demonstration baseline |
| `M10` | All ten once | 10 | `10 * alpha_0` | Primary multiple-demonstration model |
| `R10_i`, i = 1…10 | Ten exact copies of `D_i` | 1 | `10 * alpha_0` | Episode-count/weight-matched duplication control |
| `C10_i`, i = 1…10 | `D_i` plus nine contractive episodes from it | 1 | `10 * alpha_0` | Synthetic comparison at matched episode count and total loss weight |
| `M100` | Ten exact copies of the whole ten-take bank | 10 | `100 * alpha_0` | Optional whole-bank duplication control |

**Approved scope (D4):** `S_i`, `M10`, `R10_i`, and `C10_i`; `M100` and the
fixed-alpha diagnostics are deferred unless the owner requests them.
`M100` tests repetition of a fixed dataset, so compare it with `M10`; it does
not provide a count-matched comparison with `C10_i`.

A separately labeled, optional fixed-alpha diagnostic can refit `R10_i` and
`M10` at `alpha_0`. Compare fixed-alpha `R10_i` numerically against `S_i` at
`alpha_0 / 10`. These are prescribed regularization probes, not evidence of
extra information from copies. Freeze their additional execution budget
before including them; they are not part of the approved pilot below.

For every episode, reset the reservoir to zero, warm up on that episode's
initial state using the inherited policy, then teacher-force the full recorded
`[q_k, dq_k]` to **absolute next position** `q_(k+1)`. Fit once; copies are
not optimization epochs, and no next-step pair crosses an episode boundary.
Use absolute output for v1 to keep the data-source comparison focused;
residual-output replication is a later decision.

Model warm-up remains separate from the recorded pre-roll: warm-up has no
loss rows, while the recorded pre-roll does. At evaluation the
model may learn a waiting period or may fail to depart from the start; record
departure latency and non-departure as outcomes. Do not solve a failure by
silently cropping the hold or adding a time/onset input. Varied waiting and
movement durations can make next-step prediction ambiguous for the same
observed posture/velocity; whether reservoir history resolves this sufficiently
is part of the experiment.

Keep reservoir weights, input transform, warm-up, bias treatment, solver,
and derivative estimators identical across arms of each configuration. Use
the predecessor's frozen physical input transform: the actual historical
scripted-data centers and physical scales, copied and digest-bound into every
recipe, never means newly computed from any manual take, so a singleton
receives no statistics from the other nine takes (I8). The current recipe
validation requires the normalization source to belong to the training
sources, so the new recipe contract distinguishes transform provenance from
training data explicitly. Store copies as recipe multiplicities, without
duplicating payloads.

**Recipe contract (I4).** Existing recipes hold either one dataset with
copies or several datasets at multiplicity one, with 400 rows per episode
fixed and the fit identity bound to one dataset, and the existing episode
constructions either treat the initial hold as washout without augmentation
or require cropped move/dwell episodes. A new recipe schema version,
preserving the old semantics for historical recipes, represents variable row
counts, retained hold samples inside the loss, separate warm-up, source
multiplicities, row weights, augmentation parents, transform provenance, and
complete fit identities.

### Contractive construction

Adapt the existing recovery generator separately around each full recorded
manual parent, with proposed inherited settings `sigma = 0.05 rad`,
`phi = 0.99`, `gamma = 1`, the target-distance contraction envelope, and the
terminal taper anchored to that parent's measured final-dwell onset, becoming
zero throughout its final dwell. Preserve the parent's duration and recorded pre-roll; no canonical three-second move boundary is used. Freeze any required
hold-to-movement annotation and taper rules as augmentation metadata only;
they do not crop or retime the parent. Recompute
derivatives and validate complete augmented episodes at the original task
limits. Synthetic targets are the augmented trajectory's next positions;
they are not silently relabeled as the unperturbed parent.

Freeze a new seed namespace, nine accepted episodes per parent, and a finite
36-attempt budget per parent. Generate each bank once and share it across
reservoir configurations; record attempt indices, rejected episodes, seeds,
and array digests. Generation failure remains a reported arm failure.
Each stream is seeded from the namespace, the seed bank, the attempt index,
and a stable parent identifier from the locked bank, so no two parents share
latent draws and the result is independent of worker scheduling and task
ordering; the inherited generator has no parent term (I8).

For this approved fixed-start study, implement a versioned boundary envelope
that is zero at the first sample and ramps in smoothly over a frozen duration from task time zero (a recording may have no stationary initial interval, I14), and tapers it to zero before final dwell. All
synthetic episodes then share the exact parent's fixed first sample and final dwell. This changes the inherited generator's possible initial-posture
perturbation and must be tested and recorded explicitly. Freeze taper durations
from the acquisition pilot; a parent with insufficient transition support
produces a reported augmentation failure rather than altered timing. Report
realized synthetic versus manual variation; the two distributions need not
match merely because episode counts do.

## 5. Fixed pilot and numerical checks

Use the D5-approved six source configurations (trials 17, 136, 53, 1, 0, 28)
from the [frozen repetition panel](../task_1a_repeated_demonstration/panel_manifest_v1.json).
Resolve exact parameters and verify digests; historical feasibility labels
do not predict performance on manual data. This preserves a known diagnostic
panel and avoids selecting hyperparameters after inspecting the new results.
Use the canonical execution launcher, pinned dependency builds, single-thread
settings, and full execution records established by M3REP-009.

Before behavioral evaluation, verify literal repeated inputs, targets, weights,
masks, warm-up, and harvested features are identical. Test weighted fits on
unequal-length episodes against the declared mean-per-episode objective,
including bias regularization and absence of episode-boundary target pairs.
Compare `R10_i` with `S_i`,
and optional `M100` with `M10`, on the same fixed probe states. Each readout's
probes include all ten manual trajectories and any included synthetic bank;
probe-only trajectories never enter its fit or normalization. Propose the
predecessor's prediction tolerance `atol = 1e-8 rad`, `rtol = 1e-8`: far below
the 0.05 rad reference-settling diagnostic band, while allowing accumulation
roundoff. Report coefficient differences, conditioning, and prediction errors.
Validate repeat equivalences separately for the single- and whole-bank cases.
Numerical failures require diagnosis before behavioral interpretation; the
predecessor's accepted numerical exception is not blanket approval here.

| Scope | Models across six configurations | Maximum RC runs at 65 scenarios × 2 trackers |
| --- | ---: | ---: |
| Core: ten `S_i`, ten `R10_i`, one `M10` | 126 | 16,380 |
| Add ten `C10_i` per configuration | +60 | +7,800 |
| Optional `M100` per configuration | +6 | +780 |

The approved core plus contractive comparison is **186 models and at most
24,180 RC runs**, plus replay and numerical checks. Replay is driven through
the same causal derivative policy as the configuration it is paired against,
so a baseline is shared only by models that share a parent, a warm-up and that
policy; the six inherited configurations carry six distinct policies, which
makes at most 6 × 10 × 130 = 7,800 replay runs and 31,980 in total. Variable
training lengths and the longer evaluation horizon change compute cost; old
four-second timing estimates do not apply. A timing smoke
test must estimate memory, storage, and elapsed time before the full pilot.
Run a deterministic nominal subset first as an
implementation check, without dropping panel members based on performance.

**Execution budget projection (I1).** At the per-run rates of the repetition
pilot's [timing smoke check](../task_1a_repeated_demonstration/timing_smoke_check_v1.md)
(one configuration, 4.25 s runs: 0.22 s and 126 KB per RC run, 0.19 s and
94 KB per replay run), the 30 s horizon and the revised 7,800 replay runs
project to roughly 13 hours of serial simulation and 26–27 GB of run data
before training, reproduction, and other artifacts. This is a planning
estimate, not an upper bound: the rates came
from one configuration, while reservoir sizes and recording lengths vary, and
the timing report carries that limitation. The approved horizon and full
telemetry stay. Bounded process-based parallel execution is explicit
implementation scope under M3MAN-008 and is benchmarked under M3MAN-009: one
numerical thread per worker, controlled CPU affinity within the canonical
execution environment, deterministic seeds, safe artifact writes, and a
serial-versus-parallel equivalence check. Parallelism reduces elapsed time,
not storage.

## 6. Evaluation and interpretation

The nominal task starts at the exact configured `q_start` with zero velocity,
matching recording start. Both RC and replay hold the actual evaluation
posture through negative-time model warm-up and activate at task time zero.
Replay starts at recording sample zero and includes the recorded pre-roll;
RC receives measured state feedback and generates its own departure and reach.
There is no imposed movement-onset time or matched transient timing.

Use the D2-approved common evaluation horizon of **30 s from activation**,
for all arms and scenarios. Lock it after the acquisition pilot, before the
study bank and before learned-controller evaluation. This is a finite timeout
for non-convergence, not an instruction to the demonstrator or a three-second
trajectory deadline. Run the complete horizon unless a safety abort occurs,
so reaching briefly and then drifting away cannot count as stable success.
Completion is checked against the configured evaluation horizon, never
against the demonstration's length; the explicit horizon field and the
updated completeness checks land together, because the inherited check
requires the active sample count to equal the reference length (I5). Aborted
runs retain their partial metrics and terminal evidence.

Use the frozen `pd_v2` and `computed_torque` trackers. Replay each of the ten
recordings independently under those trackers and conditions. Compare `S_i`,
`R10_i`, and `C10_i` against replay of their parent. Report `M10` against the
full replay distribution and against each `S_i`; do not choose the best replay
or a different reference per model after seeing its result. A planned-trajectory
baseline may be shown as historical context with protocol differences labeled;
it is not the teacher signal in this study.

After a replay reaches the end of its log, keep commanding its final recorded
position through the common evaluation horizon, using the same causal
derivative estimator. Label this continuation in logs/plots; it is an
evaluation baseline policy, not extra recorded or synthetic training data.

**Primary outcomes:** bounded reaching and sustained target dwell, reported
for nominal operation and as success rates in each robustness class, separately
for each tracker. Success requires completing the horizon within all limits,
at most 0.5% torque saturation, and an uninterrupted target dwell of at least
1.0 s extending to the end of the run. Every sample in that dwell must be
inside 1 cm with all joint speeds at most 0.05 rad/s. Use the same continuous
dwell predicate as acquisition; the old 90%-occupancy rule and fixed movement/
dwell phases are not reused. Any excursion restarts the dwell timer.
Report earliest qualifying dwell, longest dwell, final uninterrupted dwell,
and departures after earlier successful holds.

Apply generated-reference position/workspace/velocity validity checks
throughout, and the same target/stationarity predicate to its final dwell,
using generated joint position and the versioned derivative policy. Keep
actual-motion and generated-reference results separately visible; report
the joint feasibility verdict only when both pass. Check measured-state
limits at the simulator's safety-check cadence, including terminal samples.

**Diagnostic outcomes:** final endpoint error, time to sustained target dwell,
departure latency, peak measured and generated velocity/acceleration, effort,
saturation, activation jump, and tracking error. No transient path shape or
arrival time within the horizon is required to match a particular take.
Show joint/endpoint overlays on their original recording clocks; any optional
phase-aligned visualization is descriptive and never changes training data.
Separate actual-to-generated tracking error from demonstration similarity.

Use the predecessor's 65 development scenarios: one nominal, 20 small
posture offsets (norm 0.05 rad), 20 large offsets (0.10 rad), four 12 N force
pulses, and 20 combined cases. Reuse posture draws and force directions from the
[locked development definitions](../../../configs/evaluations/task_1a_recovery_dev_v1.toml)
as development diagnostics, without describing them as a new held-out test.
Create a new manifest/config for the revised horizon, dwell rule, and force
timing; historical scenario identities and replay caches cannot be reused.

With natural movement durations, a pulse at task time 1 s could hit the pre-roll, movement, or target dwell. Use **target-dwell-triggered
pulses** for the four force and 20 combined cases: apply the inherited 12 N,
0.2 s pulse once, immediately after actual motion first satisfies the target
position/speed predicate continuously for 0.5 s. Apply this same event rule
to every RC/replay run, and record its actual timestamp. Reset the success
dwell timer at pulse end; force-case success requires a complete 1.0 s final
dwell after the pulse. A missing or too-late trigger cannot count as a
successful disturbance-recovery test; report that reason explicitly.
Pulse timing is consequently state-triggered rather than identical wall time
across methods. These cases test target-hold recovery; robustness to forces
during transit would be a separately defined extension. No confirmatory claim
or model freeze is proposed.

**Approved D3 limit:** use the canonical 6 rad/s per joint hard abort for
all evaluation and training validation, with position bounds ±3 rad and torque
bounds 10/5 N·m. The repetition pilot's 12 rad/s simulation relaxation does
not automatically carry over. If continuity at 12 rad/s is desired, approve
a separate sensitivity protocol and include the original-limit diagnostics.
Do not compare raw feasibility rates across differing speed protocols.
Keep all abort paths; physical recording/control would require the separate
M6 checklist, simulation/emulator gates, and accessible power cutoff.

Unlike the predecessor's stop-at-first-infeasible-model-sweep rule, D6 approves
evaluating every scenario independently: abort an individual unsafe run,
retain its terminal state, and continue the next scenario from a fresh reset.
This permits per-class success rates. Include training/generation failures
in attempted-model accounting; mark their runs unavailable, never successful.

Report `M10 - S_i`, `M10 - R10_i`, `C10_i - R10_i`, and `M10 - C10_i` within
each fixed configuration and tracker. Show all ten paired singleton outcomes,
their median and range, and the number improved/worsened/tied. Count `M10`
once even when reused in ten contrasts. Copies, synthetic episodes, repeated
deterministic runs, and overlapping training sets are not independent human
demonstrations. This pilot provides descriptive paired evidence, without
confidence intervals that pretend these comparisons are independent trials.

Do not use random row splits. All ten source takes are used by `M10`, so its
teacher-forced error is training error, not held-out imitation accuracy.
Independent imitation validation would require additional recorded takes
frozen outside this bank, or a separately labeled 9-versus-1 leave-one-take-out
study. Neither is required for the proposed closed-loop 1-versus-10 comparison.

For D7, “additional held-out human recordings” means extra demonstrations,
beyond the ten training takes, reserved solely to test prediction on unseen
human recordings. Deferring them means this pilot does not require that extra
recording work; closed-loop simulation evaluation still proceeds under D6.
The ten-take model's training error cannot then be called held-out imitation
accuracy.

The earlier phrase “confirmatory tuning” was imprecise. A possible later
study would first tune models using development results, then freeze the
chosen model and protocol and evaluate on a separate, untouched confirmatory
set. The confirmatory set must not be used for tuning. D7 approves leaving
that additional study outside this pilot; it does not defer the approved
fixed-panel comparisons or their human-readable report. Extra held-out human
recordings and the separate confirmatory study are deferred follow-up work.

## 7. Deliverables and implementation sequence

1. **Protocol lock — complete (DOC-007):** owner approved this plan and
   explicitly requested roadmap/task registration on 2026-09-15. UP-008–009
   and M3MAN-001–012 / M3MAN-GATE now define the remaining work. Acquisition
   settings delegated to the pilot must still be frozen under M3MAN-003 before
   study collection; this registration does not mark implementation complete.
2. **Acquisition readiness:** implement and verify REC-1–REC-6 in the existing
   recorder, upstream where appropriate; update its pin separately and rebuild
   dependencies. Define the acquisition clock upstream and verify it in the
   pilot (I2); deliver the task launcher and adapter (I7). Test a ten-take session with stationary keyboard start,
   keyboard save/next, Q with an unsaved-take warning, R discard/reset with
   recording paused, removal of F, CLI filenames without save dialogs/plots,
   current/past trails, exact reset, and import. After the second practice pilot, freeze the acquisition rate and training grid, the boundary-preserving filter,
   velocity bound, offline dwell predicate, overlay policy, generous timeouts,
   and acceptance settings. Verify that experiment-specific acceptance never
   blocks saving and advancing in the recorder.
3. **Manual bank:** collect batches, validate them offline, and retain ten
   accepted takes. The developer produces a machine-readable quality report
   with per-take measurements, acceptance/rejection reasons, and reproducible
   raw/processed plotting data/assets; lock the bank before model evaluation.
4. **Training and controls:** test the retained pre-roll and exact start,
   unequal episode lengths and loss weights (including bias), reset and
   boundary handling, derivatives, source isolation, multiplicities, ridge
   scaling, augmentation envelopes, and strict manifest/schema validation.
   Test continuous dwell timers, boundary-duration/off-by-one cases, missing
   and late disturbance triggers, timeout/non-departure outcomes, and replay
   continuation without training-data padding. Pass the
   numerical controls and timing smoke test before the full simulation pilot.
5. **Developer evidence handoff:** retain every fit/run/failure and produce
   validated machine-readable results/reports, paired metrics, and reproducible
   plotting/animation data and tools, as specified in Section 7.1. Include
   negative-time warm-up and separate recorded teacher, RC generated reference,
   and actual measured motion. Select representative cases by a frozen rule.
   The developer is not responsible for writing the human-facing final report.
6. **Reproduction and review:** provide a thin reproducibility entry point in
   `scripts/` with logic in `src/`; verify recorded hashes and metrics, and
   freeze a deterministic re-simulation subset before experiment execution.
   Run `uv run --locked nox` and the documented pre-commit gate for implemented
   work. Hand off the verified results and machine-readable reproduction audit.
7. **Human-readable interpretation:** the planning/reporting assistant (the
   assistant collaborating with the owner in this conversation) reads the
   machine-readable evidence, interprets it, and writes the final report for
   the owner. The owner reviews that report before any tuning or confirmatory
   extension. A developer-generated narrative or raw data dump does not
   fulfill this reporting deliverable.

Store payloads externally; Git holds portable pointers, manifests, plans, and
report assets. Record Git/submodule revisions, dirty state, resolved configs,
DVC hashes where used, payload hashes, seeds, environment/build identities,
preprocessing, raw metrics, and operator annotations. Do not commit recordings
or absolute storage paths. Generic recorder/library fixes belong upstream,
with a separately reviewed pin update and dependency rebuild.

### 7.1 Results handoff and final-report ownership

The **developer** delivers a versioned machine-readable evidence bundle, with
documented field definitions and units, containing:

- Acquisition manifests, raw/processed artifact references and hashes,
  per-take quality measurements and verdicts, and displayed-history provenance.
- Every training recipe, weights/counts, seeds, numerical-control results,
  fit status, and any failure or diagnosed numerical discrepancy.
- Per-model/scenario/tracker outcomes, raw metric values, actual and generated
  trajectory references, disturbance timestamps, abort reasons/terminal states,
  and explicit unavailable/not-executed statuses.
- Paired comparisons and aggregates with their denominators, configuration
  identities, links to underlying runs, and reproduction/quality-check results.
- Reproducible figure and animation inputs/tools or generated assets, with
  labels that distinguish demonstration, generated reference, and actual motion.

This can use JSON/TOML manifests and CSV tables with external array payloads,
following the repository's artifact policy. Machine-readable diagnostics and
schema/usage documentation are developer work; explanatory prose for the owner
is the reporting assistant's responsibility.

The **reporting assistant** verifies the evidence bindings and accounting,
then writes a human-readable report that explains the experiment and answers
whether ten manual demonstrations improved reaching/holding robustness over
one, whether copies or contractive episodes changed the outcome, and how
results depend on configuration/tracker. Lead with those findings, explain
metrics in ordinary language, and use a small number of useful comparisons,
trajectory plots, and animations. Separate measured observations from possible
explanations, show failures and limitations, and link numerical claims to the
underlying machine-readable evidence. Missing evidence must be resolved with
the developer or identified explicitly, never replaced by an invented result.

## 8. Approved decisions

The owner requested retained pre-roll, natural movement timing, velocity
validity, sustained target dwell, and the recorder capabilities REC-1–REC-6.
Those requirements are incorporated here. Save uses **S**, save-next uses
**Shift+S**, close uses **Q** with an unsaved-take warning, discard/reset uses
**R** with recording paused, and recording starts with **Space**. **F** is
removed. New CLI spellings remain implementation choices.
The owner has approved D1–D7, including the 2026-09-15 clarifications to
practice-data handling and collection count, and assigned human-facing report
authorship to the reporting assistant. On 2026-09-15 the owner also approved
the full plan and explicitly requested the global roadmap/task registration,
recorded under DOC-007. No recording, implementation, or execution results are
claimed by that registration.

The recording workflow, retained pre-roll, fixed start/target, natural
transients, and six recorder capabilities are settled requirements. The
decision status is recorded below. Unapproved recommendations are not
approved simply because they appear in this draft.

| ID | Status | Decision and clarification |
| --- | --- | --- |
| D1 | Approved | IK mouse guidance with current and faint past-tip overlays. Pilot-1 revision (approved 2026-09-15, I12): the faint overlay shows only the most recently saved take. Discard practice files and clear pilot display history before study take 1, then accumulate study-take trails normally. Retain the acquisition-readiness summary/settings, not practice payloads. |
| D2 | Approved | Save all takes without experiment-specific acceptance checks in the recorder. Batch validation afterward requires at least 1 s continuously inside 1 cm with joint speeds at most 0.05 rad/s. Repeat collection/validation until ten takes pass, with no total attempt cap. Recording/evaluation timeouts are 30 s; the per-take timeout does not limit the number of takes. Pilot-1 revision (approved 2026-09-15, I13): acquisition at 50 Hz; the dwell stays one actual second. |
| D3 | Approved | Start from 6 rad/s per joint for recording validation and evaluation; use a stricter common bound if the acquisition pilot supports it, fixed and recorded before study collection. |
| D4 | Approved | All ten singleton choices plus all-ten, ten-copy controls, and nine contractive additions per singleton; defer optional whole-bank copies and fixed-alpha diagnostics unless requested. |
| D5 | Approved | Equal total loss weight per episode; six inherited ESN configurations, absolute output, no new search: 186 models, at most 24,180 RC evaluations plus 7,800 replay runs, 31,980 in total. Owner decision 2026-09-17 (configuration-matched cutoffs): replay uses the causal derivative policy of the configuration it is paired against, so a baseline is shared only by models sharing a parent, a warm-up and that policy; the six frozen configurations are unchanged. |
| D6 | Approved | Both frozen trackers and all 65 development cases; force pulses after 0.5 s target dwell; every scenario evaluated independently through the common 30 s horizon, requiring at least 1 s continuous final dwell under D2's predicate. |
| D7 | Approved | Complete the ten-take closed-loop comparison; defer extra held-out human recordings and a later tuning/separate confirmatory study. The current experiment's evaluation and assistant-authored report remain in scope. |

Filtering/boundary handling, overlay opacity/colors, and augmentation taper
details can be resolved through the excluded acquisition pilot and recorded
before the study bank; they need not all be chosen by the owner now. Proposed
contractive settings are nine additions per parent, `sigma = 0.05 rad`,
`phi = 0.99`, and `gamma = 1`, with envelopes that are zero at the first sample, ramp in over a duration frozen before study collection, and taper to zero before the final dwell (I14), as described above. Any change to the agreed comparison scope or numerical limits
must be reflected in this plan before execution. No study data have yet been collected, and this plan makes no claim that manual diversity or synthetic
contraction will improve performance.

## 9. Implementation clarifications (2026-09-15)

Recorded after the implementability review of this plan against the pinned
`skelarm` and `rclib` revisions and the current source. They qualify or
sharpen the sections above without changing D1–D7 or the comparison scope,
and the ledger rows fold them into their acceptance criteria.

| ID | Topic | Clarification | Section | Tasks |
| --- | --- | --- | --- | --- |
| I1 | Execution budget | Roughly 13 h serial and 26–27 GB of run data at the previous pilot's one-configuration rates; a planning estimate, not an upper bound. Horizon and telemetry unchanged; bounded process-based parallel execution with a serial-versus-parallel equivalence check is explicit scope. | 5 | M3MAN-008, M3MAN-009 |
| I2 | Acquisition clock | The pinned recorder updates the pose once per 20 ms tick and repeats it at every elapsed sample boundary. UP-008 defines the acquisition clock and never assigns several timestamps to one update; the acquisition pilot verifies the realized rate before 100 Hz is claimed; I13 revises the acquisition rate to 50 Hz. | 2, 2.1 | UP-008, M3MAN-003 |
| I3 | Weighted ridge | Disable the library's implicit bias, append an explicit ones column, scale the whole row and target by the square root of the weight, append an unscaled one at inference, cover every prediction path, keep the implicit-bias path for historical recipes. | 4 | M3MAN-005 |
| I4 | Recipe contract | A new recipe schema version preserving the old semantics: variable row counts, hold rows in the loss, separate warm-up, source multiplicities, row weights, augmentation parents, transform provenance, complete fit identities. | 4 | M3MAN-005 |
| I5 | Horizon check | Completion is judged against the configured horizon, not the demonstration length; horizon field and completeness checks land together; aborted runs keep partial metrics and terminal evidence. | 6 | M3MAN-008 |
| I6 | Configuration identities | Own task and evaluation configurations copying the robot, limits, and target; `task_1a.toml` and its historical semantics unchanged. | 2 | M3MAN-002, M3MAN-008 |
| I7 | Recorder launcher | Thin `scripts/record_demo.py` plus a tested adapter in `src/` built on the existing scenario conversion, verifying posture, units, limits, target, sampling, and output settings. | 2 | M3MAN-003 |
| I8 | Transform and seeds | The input transform copies the historical scripted-data centers and scales, digest-bound; augmentation streams are seeded with a stable parent identifier, independent of scheduling. | 4 | M3MAN-003, M3MAN-005, M3MAN-006 |
| I9 | Boundary preservation | The inherited zero-phase filter shifts a held start (about 9.5e-5 rad after 0.1 s, 8.3e-10 rad after 1 s); the boundary-preserving requirement stays with a reproducing test. Its hold-anchored method is superseded by I11 (Section 10). | 3 | M3MAN-002 |

## 10. Pilot-1 revisions (approved 2026-09-15)

The first excluded practice pilot (2026-09-15, ten saved takes at 100 Hz with
both trail overlays) was assessed in a throwaway store under the draft rules,
and no take would have been accepted. Movement began 0.01–0.36 s after the
first logged sample, so every take failed the hold-anchored smoothing margin of
I9. Nine takes had sample gaps of 40–63 ms against the 30 ms limit, and late
ticks rose from 3 to about 180 as the number of saved trails drawn grew from
zero to nine; an offscreen repaint benchmark on the recorded tip paths grew by
about 1 ms per saved trail. That pattern is consistent with increasing
trail-rendering cost, but the cause is not established. Everything else met the
draft rules: a median sample interval of 10.0 ms, raw joint speeds of at most
2.3 rad/s, final dwells of 1.2–5.5 s, and takes of 11–17 s. The operator
reported that the tip followed the cursor, the faint trails stayed visible after
ten takes, and the 30 s timeout is worth keeping as margin. The revisions below
fit the experiment's purpose better than requiring unusually precise human
operation. A second short practice session verifies timing and preprocessing
before M3MAN-003 freezes the settings. The owner approved I10–I14 on 2026-09-15; the implementation clarifications about the first-sample derivatives and the training grid are recorded in I11 and I13.

| ID | Topic | Revision | Sections | Tasks |
| --- | --- | --- | --- | --- |
| I10 | Natural pre-roll | The first logged sample stays exactly at the reset posture; natural movement may begin immediately afterward. Pre-roll fluctuations are part of the demonstration and are checked against the usual motion limits; no stationary initial interval is required. | 2, 2.2, 3, 4, 6 | M3MAN-013 |
| I11 | Smoothing without a stationary hold | Preprocessing preserves the first sample exactly while smoothing the subsequent trajectory, without depending on a stationary hold, and keeps the recorded pre-roll fluctuations instead of replacing them with a constant posture, and it introduces no velocity or acceleration spike at the first sample. It supersedes the hold-anchored margin method of I9; the I9 reproduction of the inherited filter's start shift stays. | 3 | M3MAN-013 |
| I12 | Last saved trail | A recorder display option shows only the most recently saved trail behind the current trail. The study uses the current trail plus the last saved trail; every saved recording and its provenance are kept, and each take records which saved take was visible. | 2.1, 8 (D1) | UP-010, M3MAN-014 |
| I13 | Acquisition rate | Takes are recorded at 50 Hz with their actual timestamps, separately from the training/control grid; this experiment keeps the 0.01 s (100 Hz) grid for every comparison arm, and support for finer grids does not change it. Each take is reconstructed from its actual timestamps onto that grid; interpolation supplies intermediate reference values but recovers no unsampled motion, and derivatives come from the smoothed trajectory. Frame-count and gap checks are expressed from the acquisition period; the final dwell remains one actual second regardless of the acquisition rate. | 2, 2.1, 2.2, 3, 8 (D2) | M3MAN-013, M3MAN-014 |
| I14 | Contractive envelope | Without a guaranteed stationary initial interval, the contractive envelope is zero at the first sample and ramps in smoothly over a duration frozen before study collection, instead of staying zero through a recorded hold; the terminal taper before the final dwell is unchanged. | 4 | M3MAN-006 |

## 11. Frozen acquisition settings (2026-09-16)

The owner froze the acquisition and preprocessing settings on 2026-09-16, closing
M3MAN-003 before any study take. The values, the practice evidence behind them,
and the operator note live in
[`acquisition_readiness_v1.md`](acquisition_readiness_v1.md); the settings
themselves are the versioned `task_1a_manual_v2`, `manual_v2`, and recording v2
configurations, except the two that M3MAN-006 implements: the contractive
envelope ramps in over 0.5 s from task time zero (I14) and the augmentation seed
namespace is `task_1a_manual_v1/contractive/v1`.

The second excluded practice session verified what the pilot-1 revisions
changed: a median sample interval of 20.0 ms with the largest gap at 39.9 ms
against the 60 ms limit, acquisition ticks no longer falling behind as takes
accumulate, and a start shift of exactly 0.0 rad in every take. Seven of ten
takes were accepted; the three rejections failed only the one-second dwell
duration. The dwell tolerance stays at a 1 cm radius, since the closed-loop
evaluation uses the same target region. Practice payloads remain disposable and
are not retained.
