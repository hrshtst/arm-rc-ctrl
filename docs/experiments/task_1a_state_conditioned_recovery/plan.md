# Task 1-a Recovery Experiment

- **Proposed experiment label:** `task_1a_recovery_v1`
- **Status:** Draft protocol; no implementation or evidence yet
- **Relationship to prior work:** New experiment derived from task 1-a; the M3
  `task_1a_confirmatory_v2` evidence remains frozen and unchanged.

## 1. Why this is a new experiment

The completed task 1-a experiment established that an ESN trained from one
scripted demonstration can generate a safe reaching reference from measured
feedback. Its initial hold serves two roles at once: it is part of the teacher
trajectory and the reservoir washout interval. Under an initial-posture
perturbation, RC holds the perturbed posture during that interval while replay
tracks the recorded initial posture. The two arms therefore do not begin their
task references under identical timing semantics.

The proposed work changes the training distribution and asks a stronger
scientific question: can a fixed-weight, state-conditioned ESN generate a
reference that initially remains near the perturbed robot state and then
converges toward the demonstrated trajectory and target? This is not online
weight adaptation. A new label prevents this changed hypothesis and protocol
from being mistaken for a retuning of the M3 result.

The task remains `task_1a` because the robot, target, and reaching objective do
not change. Experiment, study, model, evaluation, and report artifacts use the
`task_1a_recovery_v1` family name and advance their own
versions independently.

## 2. Hypotheses

The primary hypothesis is:

> Smooth, contractive augmentation derived from one demonstration teaches an
> ESN to issue state-dependent references that reduce the initial command jump
> after a posture perturbation while still converging to the demonstrated
> trajectory and final target.

Supporting hypotheses are:

1. Separating reservoir warm-up from task time removes the timing asymmetry
   between RC and replay without reducing nominal success.
2. Target-decaying perturbations produce better recovery than equally weighted,
   non-decaying perturbations.
3. Absolute next-position prediction can exhibit the proposed behavior without
   changing the readout to a residual representation.
4. A residual-output formulation may reduce command error further, but is not
   assumed to converge and is therefore an exploratory ablation.

A negative result is valid. In particular, success on one demonstration does
not establish a basin of attraction outside the augmented training tube.

## 3. Nomenclature

All joint vectors have $n_q$ elements and use radians unless noted otherwise.
A subscript $k$ identifies a sample on the task clock; a superscript identifies
the role of a signal, not a different robot.

| Symbol | Definition | Dimension/unit |
| --- | --- | --- |
| $i$ | Training-episode index; $i=0$ denotes the unmodified demonstration | scalar |
| $k$ | Discrete sample index after task activation | scalar |
| $\Delta t_k$ | Time between samples $k$ and $k+1$ | s |
| $T_w$ | Reservoir warm-up duration before task time zero | s |
| $q_k,\dot q_k$ | Measured robot joint position and velocity supplied as feedback during evaluation | $n_q$, rad and rad/s |
| $q_k^{\mathrm{ref}}$ | Cropped demonstrated joint trajectory used by direct replay | $n_q$, rad |
| $q_0^{\mathrm{ref}}$ | Task initial posture: first sample of the cropped, processed reference | $n_q$, rad |
| $q^{\mathrm{pre}}$ | Robust pre-roll baseline used only for validation and onset detection | $n_q$, rad |
| $\Delta q^{\mathrm{eval}}$ | Initial-posture offset specified by an evaluation scenario | $n_q$, rad |
| $\dot q_k^{\mathrm{ref}}$ | Velocity derived from the cropped demonstration by the versioned preprocessing policy | $n_q$, rad/s |
| $q_{i,k}^{\mathrm{aug}}$ | Position in augmented training episode $i$ | $n_q$, rad |
| $\dot q_{i,k}^{\mathrm{aug}}$ | Velocity recomputed from augmented position episode $i$ | $n_q$, rad/s |
| $\delta_{i,k}$ | Smooth joint-space perturbation added to the demonstration | $n_q$, rad |
| $z_{i,k}$ | Correlated latent perturbation process before applying its envelope | $n_q$, rad |
| $\xi_{i,k}$ | Independent standard-Gaussian innovation sampled for episode $i$ | $n_q$, dimensionless |
| $\epsilon_{i,k}$ | Scaled Gaussian innovation that drives the latent process | $n_q$, rad |
| $\sigma_i$ | Per-joint marginal perturbation scale configured for episode $i$ | scalar, rad |
| $\phi$ | Temporal correlation coefficient of the augmentation process; distinct from ESN spectral radius | dimensionless |
| $w_k$ | Target-distance augmentation envelope | scalar in $[0,1]$ |
| $\gamma$ | Exponent controlling how quickly the envelope contracts near the target | positive scalar |
| $p^\star$ | Cartesian endpoint target | 2, m |
| $d_{\mathrm{tip}}(q,p^\star)$ | Euclidean distance from the endpoint at $q$ to the target | m |
| $s_k=[q_k^\mathsf{T},\dot q_k^\mathsf{T}]^\mathsf{T}$ | Measured robot state | $2n_q$ |
| $\mathcal{T}(s_k)$ | Frozen centering/scaling transform applied to the ESN input | $2n_q$, normalized |
| $u_k=\mathcal{T}(s_k)$ | ESN input during task execution; task 1-a has no task-code component | $2n_q$, normalized |
| $x_k$ | Reservoir internal state; reset to the all-zero vector before warm-up | $n_x$, dimensionless |
| $n_x$ | Number of reservoir-state elements | scalar |
| $f_{\mathrm{ESN}}$ | Fixed recurrent reservoir state-transition function | $\mathbb{R}^{n_x}\times\mathbb{R}^{2n_q}\rightarrow\mathbb{R}^{n_x}$ |
| $W_{\mathrm{out}}$ | Trained linear readout, including the `rclib` bias convention | $n_q\times(n_x+1)$ |
| $\hat q^g_{k+1}$ | Absolute-position readout in the primary RC arms | $n_q$, rad |
| $r^g_{k+1}$ | Joint-position increment produced by the residual exploratory readout | $n_q$, rad |
| $q^d_k,\dot q^d_k,\ddot q^d_k$ | Position command and causally estimated derivatives passed to the tracker | $n_q$, rad, rad/s, rad/s² |
| $\tau_k^{\mathrm{req}},\tau_k^{\mathrm{applied}}$ | Tracker-requested torque and torque after the configured limiter | $n_q$, N·m |

The symbol $q^g$ is reserved for a value produced by the absolute-position
readout, while $r^g$ denotes the generated increment of the residual readout.
The common tracker command $q^d$ equals $q^g$ for the primary RC arms, equals
$q^{\mathrm{ref}}$ for replay, and is composed from measured position plus
$r^g$ for the residual arm. This distinction prevents a hold command or a
residual vector from being mislabeled as an ESN position output.

## 4. Timing and data semantics

### 4.1 Recording and preprocessing

A manual or scripted recording should include a stationary pre-roll as
acquisition context. It provides samples for characterizing stationary
measurement noise and the pre-roll baseline $q^{\mathrm{pre}}$, locating motion
onset, initializing filters and derivative estimation without a task-boundary
transient, and accommodating the fact that a human cannot synchronize motion
onset exactly with recording start. It is not an RC washout interval and is not
part of the reaching task to be learned. In particular, $q^{\mathrm{pre}}$ does
not define or replace the task initial posture $q_0^{\mathrm{ref}}$.

A scripted demonstration also includes a configured 1.0 s pre-roll. Although
its motion onset is known, retaining the pre-roll exercises the same acquisition
and preprocessing path and supplies valid filter history. A manual-demonstration
operator is instructed to hold for about 1.0 s or longer, but that nominal
instruction is not treated as measured timing: the raw session records the
complete hold, and motion onset is proposed from the speed profile and confirmed
by a human. The teacher is not required to synchronize motion onset with the
recording start.

Preprocessing uses the pre-roll as left-hand context for filtering and derivative
estimation, then crops it from the derived training episode. The crop boundary is
the confirmed **demonstration motion onset**. Cropping prevents an incidental or
operator-dependent recording delay from becoming learned task content or an
implicit reservoir warm-up. The derived episode therefore contains the reach
plus final dwell and has no embedded reservoir-washout phase. Existing raw
records and M3 processed artifacts are immutable; a new processed-artifact
schema/version records the source interval, crop, derivative policy, and final
dwell. A legacy raw `prime` annotation may be consumed as recording pre-roll,
but its duration does not define model warm-up.

The first sample after this crop is the authoritative task initial posture
$q_0^{\mathrm{ref}}$. The pre-roll baseline is used to propose and validate the
crop; if it differs materially from $q_0^{\mathrm{ref}}$, the recording is
flagged for review or rejected rather than silently substituting one for the
other. The record stores the proposed and confirmed onset sample, detector
configuration, any human adjustment, and the raw-payload digest.

The final dwell remains part of learning because it demonstrates the target
equilibrium.

### 4.2 Warm-up and activation

Reservoir warm-up occurs before task time, over `[-T_w, 0)`, and is configured
in the model/evaluation protocol rather than inferred from the demonstration.
The reservoir starts from its deterministic all-zero state. During warm-up:

- each arm holds its own measured initial posture;
- RC receives the measured `[q, dq]` sequence but does not evaluate its readout;
- replay performs the same hold and has no privileged recovery interval;
- task metrics and task disturbances are disabled.

At **task activation**, `t = 0`, RC and replay activate simultaneously. Replay
begins the cropped demonstration reference at its confirmed motion onset; RC
evaluates its readout from the warmed reservoir state. Thus demonstration motion
onset is aligned with the end of the common pre-task hold. The robot's measured
**actual-motion onset** may occur later because of tracker dynamics and is an
outcome rather than a phase boundary. Force-pulse times and metric windows are
relative to this task clock. Every original or augmented training episode
independently (1) resets the reservoir state to the all-zero vector and then (2)
executes the configured warm-up using that episode's initial state before
teacher forcing begins. The reset is not performed only once for a batch, and no
reservoir state passes between episodes. Arbitrary fake warm-up sequences are
excluded from the primary protocol and may be tested only as a named ablation.

Warm-up duration is selected using development data, frozen with the recipe,
and tested for state convergence and sensitivity. It is not chosen from
confirmatory outcomes.

## 5. Training construction

Let the cropped demonstration be $q_k^{\mathrm{ref}}$. Episode $i=0$ is the
original, unmodified demonstration. Each synthetic episode $i\geq 1$ is

$$
q_{i,k}^{\mathrm{aug}} = q_k^{\mathrm{ref}} + \delta_{i,k}.
$$

The original episode is represented consistently by
$q_{0,k}^{\mathrm{aug}}=q_k^{\mathrm{ref}}$ and $\delta_{0,k}=0$.

The perturbation $\delta$ is smooth, bounded, seeded, and physically valid. The
preferred contractive construction uses a correlated latent process and a
target-distance envelope:

$$
\begin{aligned}
\xi_{i,k} &\sim \mathcal{N}(0,I), \\
\epsilon_{i,k} &= \sigma_i\sqrt{1-\phi^2}\,\xi_{i,k}, \\
z_{i,k+1} &= \phi z_{i,k} + \epsilon_{i,k}, \\
w_k &= \left[\mathrm{clip}\!\left(
  \frac{d_{\mathrm{tip}}(q_k^{\mathrm{ref}},p^\star)}
       {d_{\mathrm{tip}}(q_0^{\mathrm{ref}},p^\star)},
  0,1
\right)\right]^\gamma, \\
\delta_{i,k} &= w_k z_{i,k}.
\end{aligned}
$$

The latent state is initialized independently from its stationary distribution,
$z_{i,0}\sim\mathcal{N}(0,\sigma_i^2 I)$. Thus the innovations are Gaussian,
but the AR(1) filter makes the position perturbation temporally smooth. The
$\sqrt{1-\phi^2}$ factor keeps its marginal scale approximately $\sigma_i$ when
$\phi$ changes, separating the amplitude choice from the correlation-time
choice. Development evaluates $\phi\in\{0.98,0.99,0.995\}$, with 0.99 as the
anchor. At the 0.01 s task period these values correspond to correlation times
of approximately 0.50, 0.99, and 2.00 s. This augmentation coefficient must not
be named or logged as the ESN spectral radius.

The implementation must make the envelope continuously approach zero and force
it to zero throughout the final dwell. Samples that violate joint, velocity,
endpoint, or configured augmentation limits are rejected rather than clipped
silently. Velocity inputs are recomputed from each augmented position sequence
with the versioned preprocessing policy; noise is never added independently to
$\dot q$.

Teacher forcing then learns

$$
\mathcal{T}\!\left(
  \begin{bmatrix}
  q_{i,k}^{\mathrm{aug}} \\
  \dot q_{i,k}^{\mathrm{aug}}
  \end{bmatrix}
\right)
\longmapsto q_{i,k+1}^{\mathrm{aug}}.
$$

Each episode resets the reservoir and then executes its warm-up before teacher
forcing. The augmentation configuration records episode count, seeds, amplitude
distribution, correlation coefficient, envelope exponent, validity rules, and
generated-array digests. There is still exactly one independent demonstration;
reports must distinguish that count from the number of synthetic episodes.

## 6. Experimental arms

Use the same frozen PD v2 and computed-torque trackers unless a safety pilot
requires a separately versioned change.

The scenario's existing symmetric hard torque limits remain unchanged: 10 N·m
for joint 1 and 5 N·m for joint 2. Every arm logs requested and limited applied
torque separately. A development run is ineligible if more than 0.5% of its
samples are torque-saturated, even if it otherwise completes.

For each tracker and identical scenario, compare:

1. **Replay:** cropped teacher trajectory, with the common pre-task hold.
2. **RC/no augmentation:** timing redesign only; absolute-position readout.
3. **RC/non-decaying augmentation:** smooth perturbations whose envelope does
   not contract during movement; used to isolate the contraction mechanism.
4. **RC/contractive augmentation:** primary proposed method.
5. **RC/residual output:** the same contractive data with
   $q^d_{k+1}=q_k+r^g_{k+1}$; exploratory and never substituted for the primary
   arm after confirmatory evaluation.

The current M3 controller is reported as historical evidence, not mixed into
the new confirmatory family. Reservoir capacity and trial budgets should be
matched across RC arms. Any unequal compute or synthetic-episode budget is
reported explicitly.

### 6.1 ESN inputs and outputs by arm

Every RC arm uses the same measured-state interface at evaluation time:

$$
u_k = \mathcal{T}([q_k^\mathsf{T},\dot q_k^\mathsf{T}]^\mathsf{T}),
\qquad
x_{k+1}=f_{\mathrm{ESN}}(x_k,u_k).
$$

Positions and velocities are actual simulator feedback, not the previously
generated reference. Task 1-a has one fixed target, so no one-hot task code is
appended. During teacher forcing, the corresponding demonstrated or augmented
state replaces measured feedback. The arms differ as follows:

| Experimental arm | Teacher-forcing ESN input | Readout training target | Task-time ESN input | Readout and tracker position command |
| --- | --- | --- | --- | --- |
| Replay | None; replay has no ESN | None | None | $q^d_{k+1}=q^{\mathrm{ref}}_{k+1}$ |
| RC/no augmentation | $\mathcal{T}([q_k^{\mathrm{ref}},\dot q_k^{\mathrm{ref}}])$ | $q^{\mathrm{ref}}_{k+1}$ | $\mathcal{T}([q_k,\dot q_k])$ | Absolute readout $\hat q^g_{k+1}$; $q^d_{k+1}=\hat q^g_{k+1}$ |
| RC/non-decaying augmentation | $\mathcal{T}([q_{i,k}^{\mathrm{aug}},\dot q_{i,k}^{\mathrm{aug}}])$ | $q_{i,k+1}^{\mathrm{aug}}$ | $\mathcal{T}([q_k,\dot q_k])$ | Absolute readout $\hat q^g_{k+1}$; $q^d_{k+1}=\hat q^g_{k+1}$ |
| RC/contractive augmentation | $\mathcal{T}([q_{i,k}^{\mathrm{aug}},\dot q_{i,k}^{\mathrm{aug}}])$ | $q_{i,k+1}^{\mathrm{aug}}$, whose perturbation contracts toward zero | $\mathcal{T}([q_k,\dot q_k])$ | Absolute readout $\hat q^g_{k+1}$; $q^d_{k+1}=\hat q^g_{k+1}$ |
| RC/residual output | $\mathcal{T}([q_{i,k}^{\mathrm{aug}},\dot q_{i,k}^{\mathrm{aug}}])$ | $q_{i,k+1}^{\mathrm{aug}}-q_{i,k}^{\mathrm{aug}}$ | $\mathcal{T}([q_k,\dot q_k])$ | Increment readout and command: $r^g_{k+1};\quad q^d_{k+1}=q_k+r^g_{k+1}$ |

For this `skelarm` experiment, $\dot q_k$ is the simulator's integrated robot
velocity. It is not the desired velocity produced by the causal derivative
estimator. A later backend that does not measure velocity must provide a
separately specified feedback-state estimator before $\mathcal{T}$; it must not
reuse the desired derivative as if it were measured feedback.

All RC arms receive the same kind of input during warm-up, but only update
$x_k$: the readout is not evaluated and therefore has no output to log. Training
repeats each episode's initial state $[q_{i,0}^{\mathrm{aug}},0]$ for the frozen
warm-up duration. Evaluation uses measured $[q_k,\dot q_k]$ while the tracker
holds that run's actual initial posture. From task time zero, the absolute RC
arms compute

$$
\hat q^g_{k+1}=W_{\mathrm{out}}[1;x_{k+1}],
$$

while the residual arm interprets the same readout shape as an increment. The
readout consumes the reservoir state and its bias only; raw $u_k$ is not an
additional readout input. The causal derivative estimator is downstream of
these equations: it converts the
selected $q^d$ sequence to $(q^d,\dot q^d,\ddot q^d)$. PD consumes position
and velocity targets; computed torque additionally consumes desired
acceleration. The tracker, not the ESN, outputs $\tau^{\mathrm{req}}$, which the
configured limiter converts to $\tau^{\mathrm{applied}}$. Neither desired
derivatives nor tracker torque feed the ESN in this experiment.

## 7. Evaluation

### 7.1 Scenarios and splits

Retain nominal, small-posture, large-posture, force, and combined classes. New
development and confirmatory seeds are disjoint from each other and from M3.
Initial offsets used to construct augmented episodes are also disjoint from
evaluation directions and seeds. A safety pilot defines common perturbation
levels for every method; method-specific envelopes are not allowed.

Every posture-perturbed evaluation starts from

$$
q^{\mathrm{init}}=q_0^{\mathrm{ref}}+\Delta q^{\mathrm{eval}}.
$$

The nominal scenario has $\Delta q^{\mathrm{eval}}=0$. Small- and large-posture
classes differ only in their locked offset magnitudes/directions; they are never
based on the first raw pre-roll sample or on $q^{\mathrm{pre}}$. All paired arms
in a scenario start from and hold the same $q^{\mathrm{init}}$ during the common
pre-task hold.

Development work may tune ESN parameters, warm-up duration, augmentation
amplitude/correlation/envelope, derivative filters, and ridge regularization.
The protocol, recipe, scenarios, and analysis code are frozen before the
one-shot confirmatory suite. Using its outcomes to revise the method creates
`v2` and a new confirmatory run.

### 7.2 Primary and diagnostic metrics

Task success and all existing safety, dwell, trajectory, effort, and saturation
metrics remain mandatory. The new mechanism is evaluated from task time zero:

- **initial command jump:** `norm(q_desired_rc[0] - q_actual[0])`, paired with
  replay's canonical-reference jump;
- **command gap:** time series and window summaries of
  `norm(q_desired - q_actual)`;
- **reference deviation:** `norm(q_desired_rc - q_ref)`;
- **restoring alignment:** the direction of `q_desired_rc - q_actual` relative
  to the direction from actual state toward the demonstrated reference;
- **contraction:** time for the RC output/reference deviation to enter and remain
  within declared bands, plus its fitted decay diagnostic;
- **target convergence:** endpoint error and dwell success of both generated
  reference and actual motion;
- **smoothness:** desired and actual acceleration/jerk, torque peaks, and
  saturation around activation and disturbances.

No single metric is sufficient: minimizing the command gap alone could make the
generator copy the robot without reaching, while minimizing reference deviation
would reproduce direct replay and reject the proposed state-dependent behavior.
The primary scientific comparison therefore requires both a smaller early
command gap than replay and successful convergence to the common target.

Time-aligned movement RMSE against $q^{\mathrm{ref}}$ remains a diagnostic for
continuity with M3, but it is not a model-freeze or success criterion in this
experiment. It is computed from actual motion against the original, unmodified
demonstration over the movement window. Both nominal and perturbed learned
motions are allowed to take a different safe path or timing to the target.

### 7.3 Development eligibility and model freeze

Every development run must complete without a configured state or safety-limit
violation, remain below the 0.5% torque-saturation bound, and satisfy the
existing actual-motion dwell criteria. The generated reference must separately
place the endpoint inside the 1 cm target region for at least 90% of the dwell
and keep every desired joint below 0.05 rad/s throughout that interval.

For small- and large-posture scenarios, define paired ratios against replay for
the activation command jump and for the command-gap integral over the first
0.5 s. An eligible model demonstrates the proposed mechanism when both median
ratios are below 1 and at least 75% of paired scenarios improve both quantities.
These relative criteria require a consistent reduction without prescribing a
50% reduction before observing development feasibility. Force-only scenarios
remain mandatory safety/target tests, but their disturbance-recovery metrics do
not enter this initial-posture mechanism rule.

Among models satisfying these gates, selection is lexicographic: minimize the
median early command-gap ratio, then endpoint settling time, then applied-torque
RMS. The report retains the complete distributions, original-trajectory RMSE,
smoothness, restoring alignment, and all failures. Confirmatory outcomes cannot
change these gates or their ordering.

Reports show curves in this fixed order: `reference`, `replay_actual`,
`generator_output_q`, `rc_actual`; `generator_output_q` is dashed and labeled
"RC generated reference" for humans. New task-time run records store this
explicit position-valued channel. For an absolute-position arm it is the direct
ESN readout; for the residual arm it is the composed command
$q_k+r^g_{k+1}$, while the raw increment is stored separately as
`generator_increment_q`. Neither channel is populated with a hold command, and
a value must not be called an ESN readout while the readout is inactive. Warm-up
telemetry is stored separately with its interval and activation boundary.

## 8. Reproducibility and decision gates

All generated training episodes, recipes, runs, studies, and reports follow the
existing external-payload/Git-pointer policy. Provenance includes the single raw
demonstration, crop, augmentation config and seeds, warm-up config, code and
submodule revisions, resolved configs, and array digests.

Implementation should proceed through these review gates:

1. **Protocol lock:** approve this hypothesis, labels, timing, arms, metrics,
   and data split before implementation.
2. **Timing vertical slice:** demonstrate simultaneous activation and explicit
   generator logging with no augmentation.
3. **Augmentation validation:** visualize generated episodes; prove smoothness,
   bounds, deterministic regeneration, final-dwell collapse, and source
   provenance.
4. **Development ablation:** compare the three absolute-output arms before
   deciding whether the residual arm merits inclusion.
5. **Model freeze:** apply Section 7.3's safety, target, relative command-gap,
   and reservoir-seed-stability gates; retain a negative result when no model
   qualifies.
6. **Confirmatory gate:** execute the locked suite once, retain all failures,
   reproduce it from a clean checkout, and publish the complete report.

Software acceptance is deterministic execution, validation, and complete
evidence—not a favorable scientific result. The eventual task ledger should
split these gates into reviewable, test-first tasks and must not reopen or edit
M3 artifacts.

## 9. Decisions to approve implementation

### 9.1 Confirmed decisions

These decisions have been reviewed and are no longer implementation choices:

| Decision | Approved protocol |
| --- | --- |
| Experiment identity | Use `task_1a_recovery_v1` as a new experiment family. M3 artifacts and conclusions remain frozen. |
| Recording semantics | Retain a configured 1.0 s pre-roll for scripted recordings and instruct a manual teacher to hold for about 1.0 s or longer. The raw recording retains the complete interval. |
| Crop and activation | Use pre-roll samples as preprocessing context, crop the derived task episode at confirmed demonstration motion onset, and activate replay and RC together at task time zero after the common pre-task hold. |
| Initial posture | Define the task initial posture only as $q_0^{\mathrm{ref}}$. Every perturbed run starts from $q_0^{\mathrm{ref}}+\Delta q^{\mathrm{eval}}$; $q^{\mathrm{pre}}$ is used only for onset detection and validation. |
| Reservoir lifecycle | Reset the reservoir to its all-zero state and execute the configured warm-up independently for every original or synthetic training episode and every evaluation run. |
| Augmentation process | Use seeded Gaussian innovations passed through the AR(1) process, with $\phi\in\{0.98,0.99,0.995\}$ and 0.99 as the anchor. Recompute velocity from augmented position; do not perturb velocity independently. |
| RC interface and logging | All RC arms consume measured $[q,\dot q]$. Absolute position is the primary readout; the residual formulation is an explicitly named ablation. Store `generator_output_q` and, for the residual arm, `generator_increment_q`; do not use `rc_output`. |
| Tracking and safety | Retain the frozen PD v2 and computed-torque trackers and symmetric limits of 10 N·m and 5 N·m. Apply identical limits and scenario initialization to every arm. |
| Evaluation interpretation | Require safety and target dwell, assess the recovery mechanism by paired early command-gap reduction, and retain original-trajectory RMSE only as a diagnostic rather than a success gate. |

### 9.2 Remaining owner approvals

The following choices must be frozen before implementation begins. Recommended
defaults are deliberately bounded development choices, not favorable outcomes:

| ID | Decision | Recommended approval |
| --- | --- | --- |
| D1 | Augmentation budget and search range | Include the original episode and search $N_{\mathrm{aug}}\in\{16,32,64\}$ accepted synthetic episodes, $\sigma\in\{0.01,0.025,0.05,0.10\}$ rad, and $\gamma\in\{0.5,1,2\}$; use $(64,0.05,1)$ as the anchor. Pair non-decaying and contractive arms by using the same episode seeds and amplitudes. Record rejected episodes and fail a configuration after a declared finite attempt budget rather than resampling indefinitely. |
| D2 | Warm-up-duration search | Evaluate $T_w\in\{0.25,0.5,1.0,2.0\}$ s with 1.0 s as the anchor, then freeze the shortest duration that satisfies the declared reservoir-state convergence and output-sensitivity checks across development initial postures. |
| D3 | Evaluation split and perturbation envelope | Retain 13 scenarios per class for continuity with M3, but allocate new mutually disjoint augmentation, development, and confirmatory seed namespaces. Begin the common safety pilot from the M3 levels (0.05/0.10 rad posture offsets and 12 N force); freeze one method-independent envelope from the pilot before confirmatory execution. |
| D4 | Residual-arm scope | Keep residual output development-only initially. Include it in the locked confirmatory suite only if it passes the same safety, target, stability, and seed-panel gates before the protocol freeze; otherwise report it as an exploratory negative or inconclusive result. |
| D5 | Model-freeze rule | Approve Section 7.3 unchanged: both median paired ratios below 1, at least 75% of posture scenarios improving both early metrics, generated-reference and actual-motion dwell gates, safety limits, and lexicographic selection. Any later numerical revision creates a new protocol version. |
| D6 | Demonstration source | Use the existing scripted demonstration as the sole independent source for v1 so the timing/augmentation change is isolated. Add a human demonstration later as a separately identified replication dataset, not as additional v1 training data. |

Approval of D1--D6 locks the protocol for implementation. It does not authorize
the confirmatory run: that remains separately gated by the completed timing
vertical slice, augmentation validation, development ablation, model freeze,
and clean-checkout review in Section 8. Until these choices are approved, this
document remains a proposal and no confirmatory label may be used.
