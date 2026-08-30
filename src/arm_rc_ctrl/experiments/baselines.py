# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Frozen task 1-a baseline gains and replay snapshots for regression tests (M1-027).

The gains selected by the equal-budget studies (M1-025/M1-026) live in
``configs/controllers/task_1a_*.toml`` and must not be edited. This module
resolves them by tracker method and reduces a replay to a small, TOML-serializable
snapshot (termination, criteria, metrics, final state) that regression tests
compare against committed expectations within declared tolerances. Bitwise
identity of the telemetry is asserted in-process by the tests; the tolerances
cover platform differences in floating-point evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import tomli_w

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Mapping

    from arm_rc_ctrl.experiments.replay import ReplayResult

__all__ = [
    "FROZEN_BASELINES",
    "BaselineExpectations",
    "ReplaySnapshot",
    "Tolerances",
    "build_expectations",
    "compare_snapshots",
    "frozen_baseline_digest",
    "frozen_baseline_file",
    "load_expectations",
    "load_frozen_baseline",
    "snapshot",
    "write_expectations",
]

FROZEN_BASELINES: Final[dict[str, str]] = {
    "pd": "configs/controllers/task_1a_pd.toml",
    "computed_torque": "configs/controllers/task_1a_computed_torque.toml",
}
"""Repository-relative frozen gain files of the task 1-a baselines, by tracker method."""

EXPECTATIONS_SCHEMA_VERSION: Final = 1


def frozen_baseline_file(method: str) -> Path:
    """Path of the frozen gain file for ``method``."""
    try:
        relative = FROZEN_BASELINES[method]
    except KeyError:
        msg = f"no frozen baseline for method {method!r}; known methods: {sorted(FROZEN_BASELINES)}"
        raise KeyError(msg) from None
    return repository_root() / relative


def load_frozen_baseline(method: str) -> TrackerConfig:
    """Load the frozen gains of ``method`` and check that the file declares that tracker type."""
    file = frozen_baseline_file(method)
    config = load_config(file, TrackerConfig)
    if config.type != method:
        msg = f"{file.name} declares tracker type {config.type!r}, expected {method!r}"
        raise ValueError(msg)
    return config


def frozen_baseline_digest(method: str) -> str:
    """SHA-256 of the canonical JSON of the frozen gains, binding expectations to exact values."""
    canonical = json.dumps(to_mapping(load_frozen_baseline(method)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Tolerances:
    """Declared numerical tolerances of a regression comparison."""

    metric_rel: float = 1e-9
    """Relative tolerance for scalar metrics (joint RMSE, dwell, effort)."""
    state_abs: float = 1e-9
    """Absolute tolerance for the final joint state (rad, rad/s) and endpoint (m)."""

    def __post_init__(self) -> None:
        """Tolerances must be positive and finite."""
        for name in ("metric_rel", "state_abs"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                msg = f"{name} must be a positive finite number, got {value!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class ReplaySnapshot:
    """The regression-relevant outcome of one replay run."""

    method: str
    termination: str
    success: bool
    criteria: dict[str, bool]
    n_samples: int
    joint_rmse: float
    """Movement-window aggregate joint RMSE (rad)."""
    joint_rmse_per_joint: tuple[float, ...]
    dwell: dict[str, float]
    """Dwell-window metrics: in-tolerance fraction and longest run, endpoint error, joint velocity."""
    effort: dict[str, float]
    """Applied-torque metrics: RMS, peak, saturation fraction, integrated effort."""
    final_q: tuple[float, ...]
    final_dq: tuple[float, ...]
    final_tip: tuple[float, ...]


@dataclass(frozen=True)
class BaselineExpectations:
    """Committed replay snapshots of the frozen baselines on one dataset."""

    scenario: str
    dataset: str
    gains: dict[str, str]
    """Digest of the frozen gains (see ``frozen_baseline_digest``) per method the snapshots were taken with."""
    tolerances: Tolerances
    runs: dict[str, ReplaySnapshot]
    schema_version: int = field(default=EXPECTATIONS_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Every snapshot needs the digest of the gains it was taken with."""
        if self.schema_version != EXPECTATIONS_SCHEMA_VERSION:
            msg = f"unsupported expectations schema version {self.schema_version}"
            raise ValueError(msg)
        if set(self.gains) != set(self.runs):
            msg = f"gains {sorted(self.gains)} and runs {sorted(self.runs)} must cover the same methods"
            raise ValueError(msg)
        for method, run in self.runs.items():
            if run.method != f"replay+{method}":
                msg = f"run {method!r} holds a snapshot of method {run.method!r}"
                raise ValueError(msg)


def snapshot(result: ReplayResult) -> ReplaySnapshot:
    """Reduce a completed replay to its snapshot; fail for runs that stopped before every metric existed."""
    report = result.report
    if report.joint_rmse is None or report.dwell is None or report.effort is None:
        msg = f"run {report.run_id} terminated with {report.termination_kind!r} before every metric was available"
        raise ValueError(msg)
    arrays = result.run.arrays.arrays
    return ReplaySnapshot(
        method=report.method,
        termination=report.termination_kind,
        success=report.success,
        criteria=dict(result.summary.outcome.criteria),
        n_samples=result.run.arrays.n_samples,
        joint_rmse=report.joint_rmse.aggregate,
        joint_rmse_per_joint=tuple(report.joint_rmse.per_joint),
        dwell={
            "in_tolerance_fraction": report.dwell.in_tolerance_fraction,
            "longest_in_tolerance_s": report.dwell.longest_in_tolerance_s,
            "endpoint_rms": report.dwell.endpoint.rms,
            "endpoint_max": report.dwell.endpoint.max,
            "velocity_max": report.dwell.velocity_max,
        },
        effort={
            "torque_rms": report.effort.torque_rms,
            "torque_peak": report.effort.torque_peak,
            "saturation_fraction": report.effort.saturation_fraction,
            "effort": report.effort.effort,
        },
        final_q=tuple(float(v) for v in arrays["q"][-1]),
        final_dq=tuple(float(v) for v in arrays["dq"][-1]),
        final_tip=tuple(float(v) for v in arrays["tip"][-1]),
    )


def build_expectations(
    scenario: str, dataset: str, results: Mapping[str, ReplayResult], *, tolerances: Tolerances | None = None
) -> BaselineExpectations:
    """Snapshot one replay per method (keyed like ``FROZEN_BASELINES``) with the digests of the gains used."""
    return BaselineExpectations(
        scenario=scenario,
        dataset=dataset,
        gains={method: frozen_baseline_digest(method) for method in results},
        tolerances=Tolerances() if tolerances is None else tolerances,
        runs={method: snapshot(result) for method, result in results.items()},
    )


def _close(name: str, actual: float, expected: float, rel: float) -> str | None:
    if math.isclose(actual, expected, rel_tol=rel, abs_tol=0.0):
        return None
    return f"{name}: {actual!r} differs from expected {expected!r} beyond relative tolerance {rel:g}"


def _compare_tuple(label: str, actual: tuple[float, ...], expected: tuple[float, ...], rel: float) -> list[str]:
    if len(actual) != len(expected):
        return [f"{label}: length {len(actual)} != {len(expected)}"]
    found = (_close(f"{label}[{i}]", a, e, rel) for i, (a, e) in enumerate(zip(actual, expected, strict=True)))
    return [message for message in found if message is not None]


def _compare_table(label: str, actual: Mapping[str, float], expected: Mapping[str, float], rel: float) -> list[str]:
    if set(actual) != set(expected):
        return [f"{label}: keys {sorted(actual)} != {sorted(expected)}"]
    found = (_close(f"{label}.{key}", actual[key], expected[key], rel) for key in sorted(expected))
    return [message for message in found if message is not None]


def _compare_state(label: str, actual: tuple[float, ...], expected: tuple[float, ...], abs_tol: float) -> list[str]:
    if len(actual) != len(expected):
        return [f"{label}: length {len(actual)} != {len(expected)}"]
    return [
        f"{label}[{i}]: {a!r} differs from expected {e!r} beyond absolute tolerance {abs_tol:g}"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True))
        if not math.isclose(a, e, rel_tol=0.0, abs_tol=abs_tol)
    ]


def compare_snapshots(actual: ReplaySnapshot, expected: ReplaySnapshot, tolerances: Tolerances) -> list[str]:
    """Describe every way ``actual`` deviates from ``expected`` beyond the tolerances (empty when it reproduces)."""
    mismatches: list[str] = []
    for name in ("method", "termination", "success", "criteria", "n_samples"):
        a, e = getattr(actual, name), getattr(expected, name)
        if a != e:
            mismatches.append(f"{name}: {a!r} != expected {e!r}")
    rel = tolerances.metric_rel
    mismatches += _compare_tuple("joint_rmse", (actual.joint_rmse,), (expected.joint_rmse,), rel)
    mismatches += _compare_tuple(
        "joint_rmse_per_joint", actual.joint_rmse_per_joint, expected.joint_rmse_per_joint, rel
    )
    mismatches += _compare_table("dwell", actual.dwell, expected.dwell, rel)
    mismatches += _compare_table("effort", actual.effort, expected.effort, rel)
    for name in ("final_q", "final_dq", "final_tip"):
        mismatches += _compare_state(name, getattr(actual, name), getattr(expected, name), tolerances.state_abs)
    return mismatches


def write_expectations(path: Path, expectations: BaselineExpectations) -> None:
    """Write the expectations as TOML with a header explaining how to regenerate them."""
    header = (
        "# Replay snapshots of the frozen task 1-a baselines (M1-027). Regenerate only when the\n"
        "# simulation, metrics, or frozen gains change intentionally: `uv run pytest tests/regression\n"
        "# --update-baselines`, then review the diff. Comparisons use the declared [tolerances].\n"
    )
    path.write_text(header + tomli_w.dumps(to_mapping(expectations)), encoding="utf-8")


def load_expectations(path: Path) -> BaselineExpectations:
    """Load committed expectations."""
    return load_config(path, BaselineExpectations)
