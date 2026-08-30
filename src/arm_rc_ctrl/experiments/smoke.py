# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Headless deterministic smoke experiment (M0-012).

The run simulates a planar 2-DOF ``skelarm`` arm tracking a constant joint
target under joint-space PD, then teacher-forces an ``rclib`` echo state
network on the recorded log to predict ``q[k+1]`` from ``[q[k], dq[k]]``.
Outputs (``arrays.npz`` and ``summary.json`` with provenance) are written
transactionally under ``armrc://runs/<run-id>/``. Two executions with the same
configuration and seed produce identical arrays and metrics.

Determinism is guaranteed per process: run each experiment in a fresh
interpreter. Until upstream task UP-005 is resolved, the pinned ``rclib``
seeds its reservoir weights but starts the spectral-radius power iteration
from ``Eigen::Random()`` (the never re-seeded C ``std::rand``), so a second
reservoir built in the same process is scaled differently.

Run from the command line with
``python -m arm_rc_ctrl.experiments.smoke --run-id <id> [--config PATH] [--exploratory]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from skelarm import JointPD, LinkProp, SampledJointReference, Skeleton, simulate_controlled

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.provenance import (
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    require_clean_for_confirmatory,
    sha256_bytes,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import LinkConfig, RobotConfig
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot, open_storage

__all__ = [
    "DEFAULT_CONFIG",
    "EsnConfig",
    "SimulationConfig",
    "SmokeConfig",
    "SmokeResult",
    "array_digest",
    "main",
    "run_smoke",
    "validate_config",
]

DEFAULT_CONFIG = Path("configs") / "evaluations" / "smoke.toml"
ARRAYS_FILE = "arrays.npz"
SUMMARY_FILE = "summary.json"
SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SimulationConfig:
    """Closed-loop PD tracking of a constant joint target."""

    dt: float
    duration: float
    initial_q: tuple[float, ...]
    target_q: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]


@dataclass(frozen=True)
class EsnConfig:
    """Reservoir and ridge-readout hyperparameters."""

    n_neurons: int
    spectral_radius: float
    sparsity: float
    leak_rate: float
    input_scaling: float
    ridge_alpha: float
    washout: int


@dataclass(frozen=True)
class SmokeConfig:
    """Top-level smoke experiment configuration."""

    seed: int
    robot: RobotConfig
    simulation: SimulationConfig
    esn: EsnConfig


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of one smoke run."""

    run_uri: str
    run_dir: Path
    arrays: dict[str, NDArray[np.float64]]
    summary: dict[str, object]
    provenance: ProvenanceRecord


def _validate_simulation(sim: SimulationConfig, links: tuple[LinkConfig, ...]) -> None:
    dof = len(links)
    for name, values in {"initial_q": sim.initial_q, "target_q": sim.target_q, "kp": sim.kp, "kd": sim.kd}.items():
        if len(values) != dof:
            msg = f"simulation.{name} must have {dof} entries (one per joint), got {len(values)}"
            raise ValueError(msg)
    for name, gains in {"kp": sim.kp, "kd": sim.kd}.items():
        if any(g < 0 for g in gains):
            msg = f"simulation.{name} must be non-negative, got {gains}"
            raise ValueError(msg)
    for name, posture in {"initial_q": sim.initial_q, "target_q": sim.target_q}.items():
        for i, (angle, link) in enumerate(zip(posture, links, strict=True)):
            if not link.q_min <= angle <= link.q_max:
                msg = f"simulation.{name}[{i}]={angle} lies outside joint limits [{link.q_min}, {link.q_max}]"
                raise ValueError(msg)
    if sim.dt <= 0 or sim.duration <= 0:
        msg = "simulation.dt and simulation.duration must be positive"
        raise ValueError(msg)


def _validate_esn(esn: EsnConfig) -> None:
    """Mirror rclib's constructor constraints so failures name the configuration key."""
    checks = (
        (esn.n_neurons > 0, "esn.n_neurons must be positive"),
        (esn.spectral_radius >= 0, "esn.spectral_radius must be non-negative"),
        (0 <= esn.sparsity <= 1, "esn.sparsity must be in [0, 1]"),
        (0 < esn.leak_rate <= 1, "esn.leak_rate must be in (0, 1]"),
        (esn.input_scaling >= 0, "esn.input_scaling must be non-negative"),
        (esn.ridge_alpha >= 0, "esn.ridge_alpha must be non-negative"),
        (esn.washout >= 0, "esn.washout must be non-negative"),
    )
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def validate_config(config: SmokeConfig) -> None:
    """Check cross-field consistency that the type-level loader cannot express."""
    _validate_simulation(config.simulation, config.robot.links)
    _validate_esn(config.esn)
    steps = round(config.simulation.duration / config.simulation.dt)
    if config.esn.washout >= steps:
        msg = f"esn.washout must be in [0, {steps}) for {steps} simulation steps, got {config.esn.washout}"
        raise ValueError(msg)
    if config.seed < 0:
        msg = "seed must be non-negative"
        raise ValueError(msg)


def _build_skeleton(config: SmokeConfig) -> Skeleton:
    props = [
        LinkProp(
            length=link.length,
            m=link.mass,
            i=link.inertia,
            rgx=link.com[0],
            rgy=link.com[1],
            qmin=link.q_min,
            qmax=link.q_max,
        )
        for link in config.robot.links
    ]
    skeleton = Skeleton(props, base_length=config.robot.base_length)
    skeleton.q = np.asarray(config.simulation.initial_q, dtype=np.float64)
    skeleton.dq = np.zeros(len(props), dtype=np.float64)
    return skeleton


def _simulate(config: SmokeConfig) -> dict[str, NDArray[np.float64]]:
    sim = config.simulation
    target = np.asarray(sim.target_q, dtype=np.float64)
    zeros = np.zeros_like(target)
    reference = SampledJointReference(
        times=[0.0, sim.duration], q=[target, target], dq=[zeros, zeros], ddq=[zeros, zeros]
    )
    controller = JointPD(reference, kp=np.asarray(sim.kp), kd=np.asarray(sim.kd))
    log = simulate_controlled(_build_skeleton(config), controller, duration=sim.duration, dt=sim.dt)
    return {
        "t": np.asarray(log.times, dtype=np.float64),
        "q": np.asarray(log.channel("q"), dtype=np.float64),
        "dq": np.asarray(log.channel("dq"), dtype=np.float64),
        "tau": np.asarray(log.channel("tau"), dtype=np.float64),
        "q_ref": np.asarray(log.channel("q_ref"), dtype=np.float64),
    }


def _fit_esn(config: SmokeConfig, sim: dict[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
    """Teacher-force an ESN on ``[q_k, dq_k] -> q_(k+1)`` and predict the same sequence."""
    import rclib

    x = np.hstack([sim["q"][:-1], sim["dq"][:-1]])
    y = sim["q"][1:]
    esn = rclib.ESN()
    esn.add_reservoir(
        rclib.reservoirs.RandomSparse(
            config.esn.n_neurons,
            config.esn.spectral_radius,
            config.esn.sparsity,
            config.esn.leak_rate,
            config.esn.input_scaling,
            include_bias=True,
            seed=config.seed,
        )
    )
    esn.set_readout(rclib.readouts.Ridge(config.esn.ridge_alpha, include_bias=True))
    esn.fit(x, y, washout_len=config.esn.washout)
    # rclib annotates predict() as a bare np.ndarray (unparameterized); pin the dtype here.
    raw: NDArray[np.float64] = esn.predict(x)
    prediction = np.asarray(raw, dtype=np.float64)
    if prediction.shape != y.shape:
        msg = f"ESN prediction shape {prediction.shape} != target shape {y.shape}"
        raise RuntimeError(msg)
    return {"esn_input": x, "esn_target": y, "esn_prediction": prediction}


def _metrics(config: SmokeConfig, arrays: dict[str, NDArray[np.float64]]) -> dict[str, float | int]:
    target = np.asarray(config.simulation.target_q, dtype=np.float64)
    washout = config.esn.washout
    residual = arrays["esn_prediction"][washout:] - arrays["esn_target"][washout:]
    final_error: NDArray[np.float64] = arrays["q"][-1] - target
    return {
        "samples": int(arrays["t"].shape[0]),
        "final_joint_error_rad": float(np.sqrt(np.sum(final_error * final_error))),
        "max_abs_torque": float(np.max(np.abs(arrays["tau"]))),
        "esn_train_rmse": float(np.sqrt(np.mean(residual**2))),
        "esn_max_abs_error": float(np.max(np.abs(residual))),
    }


def _check_finite(arrays: dict[str, NDArray[np.float64]]) -> None:
    for name, array in arrays.items():
        if array.dtype != np.float64:
            msg = f"array {name!r} must be float64, got {array.dtype}"
            raise RuntimeError(msg)
        if not np.all(np.isfinite(array)):
            msg = f"array {name!r} contains NaN or Inf"
            raise RuntimeError(msg)


def _require_single_thread() -> None:
    if os.environ.get("OMP_NUM_THREADS") != "1":
        msg = "set OMP_NUM_THREADS=1 before importing rclib: the smoke experiment requires single-threaded reductions"
        raise RuntimeError(msg)


def run_smoke(
    config: SmokeConfig,
    store: StorageRoot,
    run_id: str,
    *,
    exploratory: bool,
    now: datetime | None = None,
) -> SmokeResult:
    """Execute the smoke experiment and persist it under ``armrc://runs/<run_id>/``.

    Parameters
    ----------
    config : SmokeConfig
        Resolved configuration.
    store : StorageRoot
        Validated external storage root.
    run_id : str
        Immutable run identifier (a valid URI segment); the run directory must not exist yet.
    exploratory : bool
        Tolerate a dirty worktree. Without it, a modified checkout is rejected.
    now : datetime | None, optional
        Provenance timestamp override.

    Raises
    ------
    ValueError
        If the configuration is inconsistent.
    RuntimeError
        If threading is not pinned or outputs are not finite.
    DirtyWorktreeError
        If the checkout is dirty and ``exploratory`` is false.
    FileExistsError
        If the run directory already exists.
    """
    validate_config(config)
    _require_single_thread()
    run_uri = ArtifactUri("runs", (run_id,))
    run_dir = store.path(run_uri, mode="write")
    if run_dir.exists():
        msg = f"{run_uri} already exists; runs are immutable, choose a new run id"
        raise FileExistsError(msg)

    provenance = collect_provenance(config, seeds={"reservoir": config.seed}, exploratory=exploratory, now=now)
    require_clean_for_confirmatory(provenance)

    arrays = _simulate(config)
    arrays.update(_fit_esn(config, arrays))
    _check_finite(arrays)
    metrics = _metrics(config, arrays)
    digests = {name: array_digest(array) for name, array in arrays.items()}
    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "kind": "smoke",
        "run_id": run_id,
        "run_uri": str(run_uri),
        "metrics": metrics,
        "arrays": {
            name: {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": digests[name]}
            for name, array in arrays.items()
        },
        "canonical_digest": sha256_bytes(canonical_json({**digests, "metrics": metrics}).encode()),
        "provenance": provenance.to_mapping(),
    }

    staging = run_dir.with_name(f"{run_id}.partial")
    if staging.exists():
        msg = f"stale staging directory {staging} exists; remove it before retrying"
        raise FileExistsError(msg)
    staging.mkdir()
    # numpy's savez stub cannot express typed **kwargs alongside allow_pickle.
    np.savez(staging / ARRAYS_FILE, **cast("dict[str, Any]", arrays))
    (staging / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    staging.rename(run_dir)
    return SmokeResult(str(run_uri), run_dir, arrays, summary, provenance)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; a thin wrapper around :func:`run_smoke`."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    parser = argparse.ArgumentParser(description="Run the headless deterministic smoke experiment.")
    parser.add_argument("--config", type=Path, default=None, help=f"TOML configuration (default: {DEFAULT_CONFIG})")
    parser.add_argument("--run-id", required=True, help="immutable run identifier under armrc://runs/")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    config_path = repository_root() / DEFAULT_CONFIG if args.config is None else Path(args.config)
    config = load_config(config_path, SmokeConfig)
    result = run_smoke(config, open_storage(), args.run_id, exploratory=args.exploratory)
    print(json.dumps({"run_uri": result.run_uri, "metrics": result.summary["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
