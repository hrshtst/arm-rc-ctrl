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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from skelarm import Skeleton, compute_forward_kinematics, compute_jacobian, integrate_with_limits

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
from arm_rc_ctrl.scenario import ScenarioConfig, build_skeleton

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.disturbances import ForcePulse

__all__ = [
    "GENERATOR_CHANNELS",
    "TRACKER_CHANNELS",
    "ChannelMap",
    "TelemetryController",
    "check_state",
    "endpoint",
    "simulate",
]

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


def endpoint(skeleton: Skeleton) -> NDArray[np.float64]:
    """Endpoint position (m) of the posed skeleton."""
    tip = skeleton.links[-1]
    return np.array([tip.xe, tip.ye], dtype=np.float64)


def check_state(scenario: ScenarioConfig, skeleton: Skeleton, t: float, step: int) -> Termination | None:
    """The termination the measured state warrants, or ``None`` when it is within every limit."""
    q, dq = skeleton.q, skeleton.dq
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
        return invalid_state(t, step, "measured q or dq is not finite")
    if np.max(np.abs(q)) > DIVERGENCE_BOUND or np.max(np.abs(dq)) > DIVERGENCE_BOUND:
        return invalid_state(t, step, f"state magnitude exceeds {DIVERGENCE_BOUND}")
    for j, (v, bound) in enumerate(zip(dq, scenario.limits.velocity, strict=True)):
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


def simulate(
    scenario: ScenarioConfig,
    controller: TelemetryController,
    *,
    duration_s: float,
    initial_q: tuple[float, ...] | None = None,
    force: ForcePulse | None = None,
    channels: ChannelMap = TRACKER_CHANNELS,
) -> tuple[RunArrays, Termination]:
    """Run ``controller`` in ``skelarm`` for ``duration_s`` and return the telemetry and termination."""
    dt = scenario.timing.dt
    steps = round(duration_s / dt)
    if steps < 1:
        msg = f"duration {duration_s} s is shorter than one control period {dt} s"
        raise ValueError(msg)
    posture = np.asarray(scenario.task.initial_q if initial_q is None else initial_q, dtype=np.float64)
    skeleton = build_skeleton(scenario, posture)
    controller.reset(skeleton)
    lower = np.array([link.q_min for link in scenario.robot.links])
    upper = np.array([link.q_max for link in scenario.robot.links])
    gravity = np.asarray(scenario.robot.gravity, dtype=np.float64)
    rows: dict[str, list[NDArray[np.float64]]] = {name: [] for name in _row_names(channels, force=force is not None)}
    termination: Termination | None = None
    t = 0.0
    for step in range(steps + 1):
        termination = check_state(scenario, skeleton, t, step)
        if termination is not None:
            break
        command = _command(controller, scenario, skeleton, t, step)
        if isinstance(command, Termination):
            termination = command
            break
        tau = command
        _append_sample(rows, channels, controller.last, skeleton, t)
        if force is not None:
            external = force.at(t)
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
    controller: TelemetryController, scenario: ScenarioConfig, skeleton: Skeleton, t: float, step: int
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
