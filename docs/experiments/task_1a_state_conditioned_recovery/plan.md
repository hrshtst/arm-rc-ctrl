# Task 1-a State-Conditioned Recovery Experiment

- **Proposed experiment label:** `task_1a_state_conditioned_recovery_v1`
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
`task_1a_state_conditioned_recovery_v1` family name and advance their own
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
| $\dot q_k^{\mathrm{ref}}$ | Velocity derived from the cropped demonstration by the versioned preprocessing policy | $n_q$, rad/s |
| $q_{i,k}^{\mathrm{aug}}$ | Position in augmented training episode $i$ | $n_q$, rad |
| $\dot q_{i,k}^{\mathrm{aug}}$ | Velocity recomputed from augmented position episode $i$ | $n_q$, rad/s |
| $\delta_{i,k}$ | Smooth joint-space perturbation added to the demonstration | $n_q$, rad |
| $z_{i,k}$ | Correlated latent perturbation process before applying its envelope | $n_q$, rad |
| $\epsilon_{i,k}$ | Seeded innovation that drives the latent process | $n_q$, rad |
| $\rho$ | Temporal correlation coefficient of the latent perturbation process | dimensionless |
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

A manual or scripted recording should include a short stationary pre-roll for
segmentation, synchronization, filtering, and initial-velocity estimation. The
teacher is not required to start moving at the instant recording begins. The
motion onset is proposed from the speed profile and confirmed by a human.

The derived training episode begins at the confirmed task onset and contains
the reach plus final dwell. It has no embedded reservoir-washout phase. Existing
raw records and M3 processed artifacts are immutable; a new processed-artifact
schema/version records the source interval, crop, derivative policy, and final
dwell. A legacy raw `prime` annotation may be consumed as recording pre-roll,
but its duration does not define model warm-up.

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

At task time `t = 0`, RC and replay activate simultaneously. Replay begins the
cropped demonstration reference; RC evaluates its readout from the warmed
reservoir state. Force-pulse times and metric windows are relative to this task
clock. Training uses the same reset and warm-up rule for every original or
augmented episode, using that episode's initial state. Arbitrary fake warm-up
sequences are excluded from the primary protocol and may be tested only as a
named ablation.

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
z_{i,k+1} &= \rho z_{i,k} + \epsilon_{i,k}, \\
w_k &= \left[\mathrm{clip}\!\left(
  \frac{d_{\mathrm{tip}}(q_k^{\mathrm{ref}},p^\star)}
       {d_{\mathrm{tip}}(q_0^{\mathrm{ref}},p^\star)},
  0,1
\right)\right]^\gamma, \\
\delta_{i,k} &= w_k z_{i,k}.
\end{aligned}
$$

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

Training uses one reservoir reset and configured warm-up per episode. The
augmentation configuration records episode count, seeds, amplitude
distribution, correlation coefficient, envelope exponent, validity rules, and
generated-array digests.
There is still exactly one independent demonstration; reports must distinguish
that count from the number of synthetic episodes.

## 6. Experimental arms

Use the same frozen PD v2 and computed-torque trackers unless a safety pilot
requires a separately versioned change. For each tracker and identical scenario,
compare:

1. **Replay:** cropped teacher trajectory, with the common pre-task warm-up.
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

Reports show curves in this fixed order:
`reference`, `replay_actual`, `rc_output`, `rc_actual`; `rc_output` is dashed.
New task-time run records store an explicit `generator_output_q` channel. It is
never populated with a hold command and must not be called ESN output when the
readout was inactive. Warm-up telemetry is stored separately with its interval
and activation boundary.

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
5. **Model freeze:** require safety, target success, reservoir-seed stability,
   and a declared improvement in the two-part mechanism criterion; freeze even
   if the result is negative.
6. **Confirmatory gate:** execute the locked suite once, retain all failures,
   reproduce it from a clean checkout, and publish the complete report.

Software acceptance is deterministic execution, validation, and complete
evidence—not a favorable scientific result. The eventual task ledger should
split these gates into reviewable, test-first tasks and must not reopen or edit
M3 artifacts.

## 9. Decisions to approve before implementation

The owner should explicitly approve:

- the experiment label and separation from M3;
- the cropped task-onset rule and common pre-task warm-up semantics;
- the primary absolute-output formulation and residual-output status;
- augmentation parameter ranges and number of synthetic episodes;
- development/confirmatory seeds and common perturbation envelope;
- the quantitative model-freeze criterion for command-gap reduction and target
  convergence;
- whether a human demonstration replaces or supplements the scripted source.

Until those choices are frozen, this document is a proposal and no
confirmatory label may be used.
