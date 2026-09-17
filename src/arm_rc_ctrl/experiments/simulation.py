# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Headless closed-loop simulation of a telemetry-logging controller in ``skelarm``.

One loop serves the direct-replay baselines and the RC target generator: it
checks the measured state against the scenario limits before every control
sample, applies the controller's (limited) torque plus an optional endpoint
force pulse, and assembles the run-record arrays from the controller's last
telemetry through a :class:`ChannelMap`. Every exception raised while
computing a command becomes a structured ``invalid_output`` termination
instead of a crash, so a run record always exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton, compute_forward_kinematics, compute_jacobian, integrate_with_limits

from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.run_record import RunArrays
from arm_rc_ctrl.experiments.termination import (
    FAILURE_KINDS,
    FailureKind,
    Termination,
    completed,
    invalid_output,
    invalid_state,
    limit_violation,
)
from arm_rc_ctrl.scenario import build_robot_skeleton

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.scenario import TaskScenario

__all__ = [
    "GENERATOR_CHANNELS",
    "RESIDUAL_CHANNELS",
    "TRACKER_CHANNELS",
    "ChannelMap",
    "CheckedState",
    "DwellCounter",
    "DwellTrigger",
    "TelemetryController",
    "check_state",
    "endpoint",
    "resolve_velocity_abort",
    "simulate",
]

_PLANE: Final = 2

DIVERGENCE_BOUND: Final = 1e3
"""Joint angles or velocities beyond this magnitude are treated as divergence (rad, rad/s)."""


class TelemetryController(Protocol):
    """A ``skelarm`` controller that exposes its last evaluation as named channels."""

    def reset(self, skeleton: Skeleton) -> None:
        """Start an episode at the skeleton's posture."""
        ...

    def control(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        """Return the (limited) joint torque."""
        ...

    @property
    def last(self) -> dict[str, NDArray[np.float64]]:
        """Telemetry of the last control evaluation."""
        ...


@dataclass(frozen=True)
class ChannelMap:
    """Which telemetry channel fills each run array (``None`` derives raw derivatives from the filtered ones)."""

    q_desired: str = "q_ref"
    dq_desired: str = "dq_ref"
    ddq_desired: str = "ddq_ref"
    dq_desired_raw: str | None = None
    ddq_desired_raw: str | None = None
    tracking_error: str = "error"
    tau_requested: str = "tau_requested"
    tau_applied: str = "tau_applied"
    saturation: str = "saturation"
    phase: str | None = None
    esn_state_norm: str | None = None
    generator_output_q: str | None = None
    generator_increment_q: str | None = None
    warmup_state_norm: str | None = None
    warmup_esn_input: str | None = None


TRACKER_CHANNELS: Final = ChannelMap()
"""Direct replay: the reference supplies exact derivatives, so raw and filtered coincide."""

GENERATOR_CHANNELS: Final = ChannelMap(
    q_desired="q_desired",
    dq_desired="dq_desired",
    ddq_desired="ddq_desired",
    dq_desired_raw="dq_desired_raw",
    ddq_desired_raw="ddq_desired_raw",
    phase="phase",
    esn_state_norm="esn_state_norm",
    generator_output_q="generator_output_q",
    warmup_state_norm="warmup_state_norm",
    warmup_esn_input="warmup_esn_input",
)
"""RC target generation: derivatives, hold/generate phase, state norm, and the M3R task-time telemetry.

``generator_output_q`` carries the readout only while active (NaN during the
hold); the warm-up channels carry the priming input and state norm only before
activation. ``generator_increment_q`` stays ``None`` until a residual arm
produces it.
"""

RESIDUAL_CHANNELS: Final = ChannelMap(
    q_desired="q_desired",
    dq_desired="dq_desired",
    ddq_desired="ddq_desired",
    dq_desired_raw="dq_desired_raw",
    ddq_desired_raw="ddq_desired_raw",
    phase="phase",
    esn_state_norm="esn_state_norm",
    generator_output_q="generator_output_q",
    generator_increment_q="generator_increment_q",
    warmup_state_norm="warmup_state_norm",
    warmup_esn_input="warmup_esn_input",
)
"""Residual RC target generation (M3R-014): the generator channels plus the raw increment readout,
which carries the readout only while active (the composed command is ``generator_output_q``).
"""


def endpoint(skeleton: Skeleton) -> NDArray[np.float64]:
    """Endpoint position (m) of the posed skeleton."""
    tip = skeleton.links[-1]
    return np.array([tip.xe, tip.ye], dtype=np.float64)


@dataclass(frozen=True)
class CheckedState:
    """One measured state as the simulator checked it (control cadence), including a terminal offending one."""

    t: float
    step: int
    q: NDArray[np.float64]
    dq: NDArray[np.float64]


def resolve_velocity_abort(scenario: TaskScenario, velocity_abort: Sequence[float] | None) -> tuple[float, ...]:
    """The per-joint speed abort bound in force: the evaluation override when given, else the scenario limit.

    The override is a simulation-only relaxation (repetition plan D5, C2): it never
    changes the scenario file, the training-validation limit, or any other gate.
    """
    if velocity_abort is None:
        return tuple(scenario.limits.velocity)
    bounds = tuple(float(v) for v in velocity_abort)
    if len(bounds) != scenario.dof or any(not (math.isfinite(v) and v > 0) for v in bounds):
        msg = f"velocity_abort must give {scenario.dof} positive finite bounds (rad/s), got {velocity_abort!r}"
        raise ValueError(msg)
    return bounds


def check_state(
    scenario: TaskScenario,
    skeleton: Skeleton,
    t: float,
    step: int,
    *,
    velocity_abort: Sequence[float] | None = None,
) -> Termination | None:
    """The termination the measured state warrants, or ``None`` when it is within every limit.

    ``velocity_abort`` replaces the scenario's per-joint speed bound for the
    abort only (the recorded ``bound`` is the one applied).
    """
    q, dq = skeleton.q, skeleton.dq
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
        return invalid_state(t, step, "measured q or dq is not finite")
    if np.max(np.abs(q)) > DIVERGENCE_BOUND or np.max(np.abs(dq)) > DIVERGENCE_BOUND:
        return invalid_state(t, step, f"state magnitude exceeds {DIVERGENCE_BOUND}")
    for j, (v, bound) in enumerate(zip(dq, resolve_velocity_abort(scenario, velocity_abort), strict=True)):
        if abs(float(v)) > bound:
            return limit_violation(t, step, "joint_velocity", float(v), bound, joint=j)
    radius = float(np.hypot(*endpoint(skeleton)))
    if radius > scenario.limits.endpoint_radius:
        return limit_violation(t, step, "endpoint", radius, scenario.limits.endpoint_radius)
    return None


def _channel(last: dict[str, NDArray[np.float64]], name: str, t: float) -> NDArray[np.float64]:
    try:
        return np.asarray(last[name], dtype=np.float64)
    except KeyError:
        msg = f"controller telemetry lacks channel {name!r} at t = {t} s"
        raise KeyError(msg) from None


class DwellCounter:
    """Counts consecutive qualifying samples and reports the one that first completes the hold.

    The rule is a *continuous* run, as in acquisition: any sample that fails
    the predicate restarts the count, and the counter fires exactly once so a
    scenario can never carry two disturbances.
    """

    def __init__(self, hold_samples: int) -> None:
        """Require at least one qualifying sample."""
        if hold_samples < 1:
            msg = f"hold_samples must be at least 1, got {hold_samples}"
            raise ValueError(msg)
        self._hold = hold_samples
        self._run = 0
        self._fired = False

    @property
    def fired(self) -> bool:
        """Whether the hold has already completed."""
        return self._fired

    def update(self, *, qualifies: bool) -> bool:
        """Advance by one sample; ``True`` on the sample that completes the hold for the first time."""
        if not qualifies:
            self._run = 0
            return False
        self._run += 1
        if self._fired or self._run < self._hold:
            return False
        self._fired = True
        return True


@dataclass(frozen=True)
class DwellTrigger:
    """A force pulse armed by the measured motion rather than by the clock (plan section 6).

    With natural movement durations a fixed task time could land in the
    pre-roll, the movement, or the dwell, so the pulse fires once the arm has
    satisfied the target predicate continuously for ``hold_s``. The realised
    timestamp is recorded, because it differs between methods by design.
    """

    target: tuple[float, float]
    tolerance_m: float
    max_velocity_rad_s: float
    hold_s: float
    duration_s: float
    force: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate the predicate, the hold, and the pulse."""
        if len(self.target) != _PLANE or len(self.force) != _PLANE:
            msg = f"target and force are planar vectors, got {self.target!r} and {self.force!r}"
            raise ValueError(msg)
        for name, value in (
            ("tolerance_m", self.tolerance_m),
            ("max_velocity_rad_s", self.max_velocity_rad_s),
            ("hold_s", self.hold_s),
            ("duration_s", self.duration_s),
        ):
            if not (value > 0 and math.isfinite(value)):
                msg = f"{name} must be positive and finite, got {value!r}"
                raise ValueError(msg)
        if not all(math.isfinite(v) for v in (*self.target, *self.force)):
            msg = "target and force must be finite"
            raise ValueError(msg)

    @classmethod
    def from_polar(
        cls,
        *,
        target: Sequence[float],
        tolerance_m: float,
        max_velocity_rad_s: float,
        hold_s: float,
        duration_s: float,
        magnitude_n: float,
        direction_deg: float,
    ) -> DwellTrigger:
        """Build a trigger whose pulse is described by a magnitude and a direction from the base x axis."""
        if magnitude_n < 0 or not math.isfinite(magnitude_n) or not math.isfinite(direction_deg):
            msg = (
                f"magnitude_n must be finite and >= 0 and direction_deg finite, got {magnitude_n!r}, {direction_deg!r}"
            )
            raise ValueError(msg)
        angle = math.radians(direction_deg)
        goal = tuple(float(v) for v in target)
        if len(goal) != _PLANE:
            msg = f"target must be planar, got {target!r}"
            raise ValueError(msg)
        return cls(
            target=(goal[0], goal[1]),
            tolerance_m=tolerance_m,
            max_velocity_rad_s=max_velocity_rad_s,
            hold_s=hold_s,
            duration_s=duration_s,
            force=(magnitude_n * math.cos(angle), magnitude_n * math.sin(angle)),
        )

    def hold_samples(self, dt: float) -> int:
        """Consecutive samples the hold spans on a ``dt`` grid (the duration plus the sample that starts it)."""
        if not (dt > 0 and math.isfinite(dt)):
            msg = f"dt must be positive and finite, got {dt!r}"
            raise ValueError(msg)
        return round(self.hold_s / dt) + 1

    def qualifies(self, tip: NDArray[np.float64], dq: NDArray[np.float64]) -> bool:
        """Whether this measured sample satisfies the target position and speed predicate (closed bounds)."""
        distance = math.hypot(float(tip[0]) - self.target[0], float(tip[1]) - self.target[1])
        speed = float(np.max(np.abs(dq))) if dq.shape[0] else 0.0
        return distance <= self.tolerance_m and speed <= self.max_velocity_rad_s

    def pulse_at(self, t: float) -> ForcePulse:
        """The pulse this trigger applies once armed at time ``t``."""
        return ForcePulse(start_s=t, duration_s=self.duration_s, force=self.force)


class _Disturbance:
    """The external force acting on a run: a fixed schedule, a state trigger, or nothing at all.

    Keeping the schedule, the trigger, its counter, and the pulse it armed in
    one place lets the loop ask only what force acts at this sample, and makes
    the two rules mutually exclusive by construction.
    """

    def __init__(
        self,
        force: ForcePulse | None,
        trigger: DwellTrigger | None,
        dt: float,
        sink: list[ForcePulse] | None,
    ) -> None:
        """Refuse a run that carries both rules."""
        if force is not None and trigger is not None:
            msg = "force and force_trigger cannot both be given: a run carries one disturbance rule"
            raise ValueError(msg)
        self._force = force
        self._trigger = trigger
        self._counter = None if trigger is None else DwellCounter(trigger.hold_samples(dt))
        self._sink = sink
        self._armed: ForcePulse | None = None

    @property
    def active(self) -> bool:
        """Whether this run carries a disturbance at all (and therefore logs ``ext_force``)."""
        return self._force is not None or self._trigger is not None

    def _maybe_arm(self, skeleton: Skeleton, t: float) -> None:
        """Advance the trigger on the measured state and arm the pulse on the sample that completes the hold."""
        if self._armed is not None or self._trigger is None or self._counter is None:
            return
        # The trigger reads the measured motion, never the commanded one, so every method is
        # disturbed at the state the protocol names rather than at a shared wall time.
        if self._counter.update(qualifies=self._trigger.qualifies(endpoint(skeleton), skeleton.dq)):
            self._armed = self._trigger.pulse_at(t)
            if self._sink is not None:
                self._sink.append(self._armed)

    def step(self, skeleton: Skeleton, t: float) -> NDArray[np.float64] | None:
        """The force applied at this sample, or ``None`` when the run carries no disturbance."""
        if not self.active:
            return None
        self._maybe_arm(skeleton, t)
        applied = self._force if self._force is not None else self._armed
        return np.zeros(_PLANE, dtype=np.float64) if applied is None else applied.at(t)


def simulate(
    scenario: TaskScenario,
    controller: TelemetryController,
    *,
    duration_s: float,
    initial_q: tuple[float, ...] | None = None,
    force: ForcePulse | None = None,
    force_trigger: DwellTrigger | None = None,
    triggered: list[ForcePulse] | None = None,
    channels: ChannelMap = TRACKER_CHANNELS,
    velocity_abort: Sequence[float] | None = None,
    checked_states: list[CheckedState] | None = None,
) -> tuple[RunArrays, Termination]:
    """Run ``controller`` in ``skelarm`` for ``duration_s`` and return the telemetry and termination.

    ``velocity_abort`` overrides the scenario's per-joint speed abort for this
    run only. ``checked_states``, when given, receives every state the
    simulator checked at the control cadence, including a terminal offending
    state that never enters the telemetry (no controller output exists for it).
    """
    dt = scenario.timing.dt
    disturbance = _Disturbance(force, force_trigger, dt, triggered)
    abort_bounds = resolve_velocity_abort(scenario, velocity_abort)
    steps = round(duration_s / dt)
    if steps < 1:
        msg = f"duration {duration_s} s is shorter than one control period {dt} s"
        raise ValueError(msg)
    posture = np.asarray(scenario.task.initial_q if initial_q is None else initial_q, dtype=np.float64)
    skeleton = build_robot_skeleton(scenario.robot, posture)
    controller.reset(skeleton)
    lower = np.array([link.q_min for link in scenario.robot.links])
    upper = np.array([link.q_max for link in scenario.robot.links])
    gravity = np.asarray(scenario.robot.gravity, dtype=np.float64)
    rows: dict[str, list[NDArray[np.float64]]] = {name: [] for name in _row_names(channels, force=disturbance.active)}
    termination: Termination | None = None
    t = 0.0
    for step in range(steps + 1):
        if checked_states is not None:
            checked_states.append(
                CheckedState(
                    t=t, step=step, q=np.array(skeleton.q, dtype=np.float64), dq=np.array(skeleton.dq, dtype=np.float64)
                )
            )
        termination = check_state(scenario, skeleton, t, step, velocity_abort=abort_bounds)
        if termination is not None:
            break
        command = _command(controller, scenario, skeleton, t, step)
        if isinstance(command, Termination):
            termination = command
            break
        tau = command
        _append_sample(rows, channels, controller.last, skeleton, t)
        external = disturbance.step(skeleton, t)
        if external is not None:
            rows["ext_force"].append(external)
            tau = tau + compute_jacobian(skeleton).T @ external
        if step == steps:
            termination = completed(t, step)
            break
        integrate_with_limits(skeleton, tau, dt, lower, upper, gravity)
        compute_forward_kinematics(skeleton)
        t = (step + 1) * dt
    if termination is None:  # pragma: no cover - the loop always terminates
        msg = "simulation loop ended without a termination"
        raise RuntimeError(msg)
    if not rows["t"]:
        msg = f"the initial state already violates the scenario: {termination.detail}"
        raise ValueError(msg)
    return _stack(rows), termination


def _command(
    controller: TelemetryController, scenario: TaskScenario, skeleton: Skeleton, t: float, step: int
) -> NDArray[np.float64] | Termination:
    """The controller's torque for this sample, or the structured termination its failure warrants."""
    try:
        tau = np.asarray(controller.control(t, skeleton), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - any command failure must end the run safely
        return invalid_output(t, step, f"{type(exc).__name__}: {exc}", _failure_of(exc))
    if tau.shape != (scenario.dof,):
        return invalid_output(t, step, f"controller returned a torque of shape {tau.shape}", "shape")
    if not np.all(np.isfinite(tau)):
        return invalid_output(t, step, f"controller returned a non-finite torque {tau.tolist()}", "non_finite")
    return tau


def _failure_of(exc: BaseException) -> FailureKind:
    """The failure category an exception declares (``model_exception`` unless it says otherwise)."""
    category = getattr(exc, "category", None)
    if isinstance(category, str) and category in FAILURE_KINDS:
        return cast("FailureKind", category)
    return "model_exception"


def _row_names(channels: ChannelMap, *, force: bool) -> list[str]:
    names = [
        "t",
        "q",
        "dq",
        "tip",
        "q_desired",
        "dq_desired",
        "ddq_desired",
        "dq_desired_raw",
        "ddq_desired_raw",
        "tracking_error",
        "tau_requested",
        "tau_applied",
        "saturation",
    ]
    if force:
        names.append("ext_force")
    if channels.phase is not None:
        names.append("phase")
    if channels.esn_state_norm is not None:
        names.append("esn_state_norm")
    names.extend(
        name
        for name in ("generator_output_q", "generator_increment_q", "warmup_state_norm", "warmup_esn_input")
        if getattr(channels, name) is not None
    )
    return names


def _append_sample(
    rows: dict[str, list[NDArray[np.float64]]],
    channels: ChannelMap,
    last: dict[str, NDArray[np.float64]],
    skeleton: Skeleton,
    t: float,
) -> None:
    rows["t"].append(np.array([t]))
    rows["q"].append(skeleton.q.copy())
    rows["dq"].append(skeleton.dq.copy())
    rows["tip"].append(endpoint(skeleton))
    rows["q_desired"].append(_channel(last, channels.q_desired, t))
    dq_desired = _channel(last, channels.dq_desired, t)
    ddq_desired = _channel(last, channels.ddq_desired, t)
    rows["dq_desired"].append(dq_desired)
    rows["ddq_desired"].append(ddq_desired)
    raw_dq = dq_desired if channels.dq_desired_raw is None else _channel(last, channels.dq_desired_raw, t)
    raw_ddq = ddq_desired if channels.ddq_desired_raw is None else _channel(last, channels.ddq_desired_raw, t)
    rows["dq_desired_raw"].append(raw_dq)
    rows["ddq_desired_raw"].append(raw_ddq)
    rows["tracking_error"].append(_channel(last, channels.tracking_error, t))
    rows["tau_requested"].append(_channel(last, channels.tau_requested, t))
    rows["tau_applied"].append(_channel(last, channels.tau_applied, t))
    rows["saturation"].append(np.array([float(np.any(_channel(last, channels.saturation, t) > 0))]))
    if channels.phase is not None:
        rows["phase"].append(_channel(last, channels.phase, t).reshape(1))
    if channels.esn_state_norm is not None:
        rows["esn_state_norm"].append(_channel(last, channels.esn_state_norm, t).reshape(1))
    for wide in ("generator_output_q", "generator_increment_q", "warmup_esn_input"):
        source = getattr(channels, wide)
        if source is not None:
            rows[wide].append(_channel(last, source, t))
    if channels.warmup_state_norm is not None:
        rows["warmup_state_norm"].append(_channel(last, channels.warmup_state_norm, t).reshape(1))


def _stack(rows: dict[str, list[NDArray[np.float64]]]) -> RunArrays:
    n = len(rows["t"])
    stacked: dict[str, NDArray[Any]] = {name: np.vstack(values) for name, values in rows.items() if name != "t"}
    stacked["t"] = np.concatenate(rows["t"])
    stacked["saturation"] = stacked["saturation"].ravel().astype(np.int64)
    if "phase" in stacked:
        stacked["phase"] = stacked["phase"].ravel().astype(np.int64)
    if "esn_state_norm" in stacked:
        stacked["esn_state_norm"] = stacked["esn_state_norm"].ravel()
    if "warmup_state_norm" in stacked:
        stacked["warmup_state_norm"] = stacked["warmup_state_norm"].ravel()
    stacked["task_code"] = np.zeros((n, 0), dtype=np.float64)
    return RunArrays(stacked)
