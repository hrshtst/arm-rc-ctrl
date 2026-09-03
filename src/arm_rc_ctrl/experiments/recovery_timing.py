# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproducible recovery timing traces (M3R-009 review evidence for the M3R-010 gate).

Runs four paired slice cases on the committed recovery dataset — nominal
``T_w = 0`` and ``T_w = 1.0`` s, a perturbed start at ``T_w = 1.0`` s, and a
task-relative force pulse at ``T_w = 1.0`` s — and writes one review PNG per
case plus a canonical JSON summary with full provenance. The command **fails
loudly** if any schedule contract is violated: both arms must hold the
(possibly perturbed) initial posture through the pre-task hold, activate
simultaneously at task time zero, and receive no force during the hold with
the pulse starting at ``activation + task start``.

A development recipe is trained in memory from the given model configuration
(the frozen v4 ESN hyperparameters by default); the simulations run against a
private temporary store, so no run artifacts or records are created anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.closed_loop import load_nominal_config
from arm_rc_ctrl.experiments.disturbances import ForcePulse
from arm_rc_ctrl.experiments.recovery_slice import RecoveryPair, run_recovery_pair
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.recipe import DatasetSource, ModelRecipe, TrainingSpec, create_recipe
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.train import ModelConfig, load_model_config
from arm_rc_ctrl.rc.warmup import WarmupConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import StorageRoot, open_storage

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

    from arm_rc_ctrl.controllers.estimator import EstimatorConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "FORCE_PULSE_TASK",
    "PERTURBATION_RAD",
    "TimingCase",
    "TimingTraceError",
    "generate_timing_traces",
    "main",
    "resolved_timing_config",
]

PERTURBATION_RAD: Final[tuple[float, ...]] = (0.05, -0.04)
"""Development perturbation of the initial posture for the perturbed-start case (rad)."""
FORCE_PULSE_TASK: Final = ForcePulse(start_s=0.5, duration_s=0.2, force=(0.0, -6.0))
"""Development endpoint force pulse on the task clock for the force case."""
_HOLD_TOLERANCE_RAD: Final = 1e-9
_GRID_TOLERANCE_S: Final = 1e-9


class TimingTraceError(RuntimeError):
    """A schedule contract was violated by a generated trace."""


@dataclass(frozen=True)
class TimingCase:
    """One paired trace case."""

    name: str
    warmup_s: float
    perturbation_rad: tuple[float, ...] | None
    force: ForcePulse | None
    """Task-clock pulse (shifted by the slice; never inside the hold)."""


def _cases() -> tuple[TimingCase, ...]:
    return (
        TimingCase("nominal_tw0", 0.0, None, None),
        TimingCase("nominal_tw1", 1.0, None, None),
        TimingCase("perturbed_tw1", 1.0, PERTURBATION_RAD, None),
        TimingCase("force_tw1", 1.0, None, FORCE_PULSE_TASK),
    )


def _dev_recipe(
    record: RecoveryDatasetRecord, samples: SampleSet, model_config: ModelConfig, warmup_s: float
) -> ModelRecipe:
    """A development recipe on the recovery dataset with the warmup_hold washout (never written to disk)."""
    if record.normalization is None:  # pragma: no cover - the committed dataset records statistics
        msg = f"dataset {record.artifact.artifact_id} records no normalization statistics"
        raise TimingTraceError(msg)
    transform = InputTransform.derive(
        model_config.input_transform.policy,
        record.normalization,
        fixed_scales=model_config.input_transform.fixed_scales,
    )
    recipe, _ = create_recipe(
        "recovery-timing-dev",
        model_config.esn,
        sources=[
            DatasetSource(
                record.artifact.artifact_id,
                record.artifact.payload.sha256,
                f"data/records/processed/{record.artifact.artifact_id}.toml",
            )
        ],
        samples={record.artifact.artifact_id: samples},
        dof=record.dof,
        task_code_dim=record.task_code_dim,
        preprocessing=record.preprocessing,
        transform=transform,
        training=TrainingSpec(washout="warmup_hold", warmup_s=warmup_s),
    )
    return recipe


def _require(condition: bool, case: str, detail: str) -> None:  # noqa: FBT001
    if not condition:
        msg = f"schedule contract violated in case {case!r}: {detail}"
        raise TimingTraceError(msg)


def _verify_case(case: TimingCase, pair: RecoveryPair, initial_q: tuple[float, ...]) -> list[str]:
    """Assert the schedule contracts of one case; returns the human-readable list of verified facts."""
    verified: list[str] = []
    _require(pair.activation_s == case.warmup_s, case.name, f"activation {pair.activation_s} != T_w {case.warmup_s}")
    for arm, result in (("replay", pair.replay), ("rc", pair.rc)):
        summary = result.summary
        _require(
            summary.activation_s == case.warmup_s, case.name, f"{arm} run records activation {summary.activation_s}"
        )
        _require(summary.termination.kind == "completed", case.name, f"{arm} terminated {summary.termination.kind}")
        arrays = result.run.arrays.arrays
        t = cast("NDArray[np.float64]", arrays["t"])
        hold = t < case.warmup_s - _GRID_TOLERANCE_S
        if case.warmup_s > 0:
            held = cast("NDArray[np.float64]", arrays["q_desired"])[hold]
            _require(
                bool(np.all(np.abs(held - np.asarray(initial_q)) <= _HOLD_TOLERANCE_RAD)),
                case.name,
                f"{arm} desired command leaves the initial posture during the hold",
            )
            _require(
                not bool(cast("NDArray[np.float64]", arrays["dq_desired"])[hold].any()),
                case.name,
                f"{arm} desired velocity is nonzero during the hold",
            )
        if case.force is not None:
            applied = np.abs(cast("NDArray[np.float64]", arrays["ext_force"])).sum(axis=1) > 0
            _require(not bool(applied[hold].any()), case.name, f"{arm} force acts during the hold")
            first = float(t[applied][0])
            expected = case.warmup_s + case.force.start_s
            _require(
                abs(first - expected) <= _GRID_TOLERANCE_S,
                case.name,
                f"{arm} force onset {first} != activation + task start {expected}",
            )
    verified.append(f"both arms activate at task time zero (activation_s = {case.warmup_s})")
    if case.warmup_s > 0:
        verified.append("both arms hold the initial posture with zero desired velocity through the hold")
    if case.perturbation_rad is not None:
        verified.append("the perturbed posture is held by both arms (no pre-activation correction)")
    if case.force is not None:
        verified.append("no force during the hold; onset at activation + task-relative start")
    return verified


def _plot_case(case: TimingCase, pair: RecoveryPair, out_dir: Path) -> Path:
    """One review PNG: joints, phase, state norms, command gap (and external force when pulsed)."""
    rc = pair.rc.run.arrays.arrays
    replay = pair.replay.run.arrays.arrays
    t = cast("NDArray[np.float64]", rc["t"])
    activation = pair.activation_s
    panels = 5 if case.force is not None else 4
    fig, axes_obj = plt.subplots(panels, 1, figsize=(10, 3 * panels), sharex=True, constrained_layout=True)
    axes = axes_obj

    ax = axes[0]
    for j, color in ((0, "tab:blue"), (1, "tab:orange")):
        ax.plot(t, rc["q"][:, j], color=color, linewidth=1.2, label=f"RC measured q[{j}]")
        ax.plot(
            t,
            rc["generator_output_q"][:, j],
            color=color,
            linestyle="--",
            linewidth=1.6,
            label=f"generator_output_q[{j}]",
        )
        ax.plot(
            t,
            replay["q_desired"][:, j],
            color="black",
            linestyle=":",
            linewidth=1.0,
            label="replay desired" if j == 0 else None,
        )
    ax.axvline(activation, color="red", linestyle="--", linewidth=1.2, label="activation (task time 0)")
    ax.set_ylabel("joint angle (rad)")
    ax.set_title(f"{case.name}: T_w = {case.warmup_s} s")
    ax.legend(loc="center right", fontsize="x-small", ncols=2)

    ax = axes[1]
    ax.step(t, rc["phase"], where="post", label="RC phase (0 hold / 1 generate)")
    ax.axvline(activation, color="red", linestyle="--", linewidth=1.2)
    ax.set_ylabel("phase")
    ax.set_yticks([0, 1])
    ax.legend(loc="center right", fontsize="x-small")

    ax = axes[2]
    ax.plot(t, rc["esn_state_norm"], label="esn_state_norm (full run)")
    ax.plot(t, rc["warmup_state_norm"], linewidth=2.0, label="warmup_state_norm (hold only)")
    ax.axvline(activation, color="red", linestyle="--", linewidth=1.2)
    ax.set_ylabel("reservoir state norm")
    ax.legend(loc="center right", fontsize="x-small")

    ax = axes[3]
    rc_gap = np.sqrt(np.sum((rc["q_desired"] - rc["q"]) ** 2, axis=1))
    replay_gap = np.sqrt(np.sum((replay["q_desired"] - replay["q"]) ** 2, axis=1))
    ax.plot(t, rc_gap, label="RC command gap")
    ax.plot(t, replay_gap, label="replay command gap")
    ax.axvspan(activation, activation + 0.5, color="red", alpha=0.08, label="early window")
    ax.axvline(activation, color="red", linestyle="--", linewidth=1.2)
    ax.set_ylabel("command gap (rad)")
    ax.legend(loc="upper right", fontsize="x-small")

    if case.force is not None:
        ax = axes[4]
        for arm, arrays, style in (("replay", replay, "-"), ("rc", rc, "--")):
            force = cast("NDArray[np.float64]", arrays["ext_force"])
            ax.plot(t, np.sqrt(np.sum(force * force, axis=1)), style, label=f"{arm} |ext_force| (N)")
        ax.axvline(activation, color="red", linestyle="--", linewidth=1.2)
        ax.axvline(activation + case.force.start_s, color="purple", linestyle=":", label="activation + task start")
        ax.set_ylabel("|ext_force| (N)")
        ax.legend(loc="upper right", fontsize="x-small")
    axes[panels - 1].set_xlabel("run clock (s)")
    path = out_dir / f"{case.name}.png"
    cast("Any", fig).savefig(path, dpi=150)
    plt.close(fig)
    return path


def resolved_timing_config(
    record: RecoveryDatasetRecord,
    scenario_file: Path,
    model_config: ModelConfig,
    estimator: EstimatorConfig,
    tracker: TrackerConfig,
    command: str,
) -> dict[str, object]:
    """The fully resolved configuration bound into the trace provenance.

    Binds the canonical mappings of the resolved model, derivative estimator,
    and tracker (not just their file names), so changing any one of them
    changes the provenance identity of the evidence.
    """
    return {
        "dataset": record.artifact.artifact_id,
        "scenario": sha256_file(scenario_file),
        "model": to_mapping(model_config),
        "estimator": to_mapping(estimator),
        "tracker": to_mapping(tracker),
        "cases": [case.name for case in _cases()],
        "perturbation_rad": list(PERTURBATION_RAD),
        "force_task": {
            "start_s": FORCE_PULSE_TASK.start_s,
            "duration_s": FORCE_PULSE_TASK.duration_s,
            "force": list(FORCE_PULSE_TASK.force),
        },
        "command": command,
    }


def generate_timing_traces(
    record: RecoveryDatasetRecord,
    samples: SampleSet,
    scenario: ScenarioConfig,
    scenario_file: Path,
    model_config: ModelConfig,
    estimator: EstimatorConfig,
    tracker: TrackerConfig,
    out_dir: Path,
    *,
    store: StorageRoot,
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.recovery_timing",
) -> dict[str, object]:
    """Run every case, verify its schedule contracts, and write the plot set and JSON summary.

    Every simulation runs before anything is written under ``out_dir``: an
    output written mid-run would dirty the worktree and poison the later
    cases' non-exploratory provenance (the M3 robustness-suite lesson).
    """
    payload = record.artifact.payload
    resolved = resolved_timing_config(record, scenario_file, model_config, estimator, tracker, command)
    provenance = collect_provenance(
        resolved,
        seeds={"reservoir": model_config.esn.reservoir.seed},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    recipes: dict[float, ModelRecipe] = {}
    completed_pairs: list[tuple[TimingCase, RecoveryPair]] = []
    summary: dict[str, object] = {
        "dataset": record.artifact.artifact_id,
        "payload_sha256": payload.sha256,
        "scenario": scenario.name,
        "provenance": json.loads(provenance.to_json()),
        "cases": {},
    }
    cases_out = cast("dict[str, object]", summary["cases"])
    for case in _cases():
        recipe = recipes.get(case.warmup_s)
        if recipe is None:
            recipe = _dev_recipe(record, samples, model_config, case.warmup_s)
            recipes[case.warmup_s] = recipe
        initial = (
            tuple(float(q) for q in record.q0_ref)
            if case.perturbation_rad is None
            else tuple(float(q) + d for q, d in zip(record.q0_ref, case.perturbation_rad, strict=True))
        )
        pair = run_recovery_pair(
            scenario,
            scenario_file,
            record,
            samples,
            recipe,
            tracker,
            store=store,
            warmup=WarmupConfig(case.warmup_s),
            exploratory=exploratory,
            estimator=estimator,
            initial_q=initial,
            force=case.force,
            now=now,
            command=command,
        )
        verified = _verify_case(case, pair, initial)
        recovery = pair.recovery
        if recovery is None:  # pragma: no cover - completed runs always yield the report
            msg = f"case {case.name!r} completed without a recovery report"
            raise TimingTraceError(msg)
        completed_pairs.append((case, pair))
        cases_out[case.name] = {
            "plot": f"{case.name}.png",
            "warmup_s": case.warmup_s,
            "initial_q": list(initial),
            "force_task": None
            if case.force is None
            else {"start_s": case.force.start_s, "duration_s": case.force.duration_s},
            "verified": verified,
            "fit_rmse_rad": recipe.fit.rmse,
            "replay_criteria": dict(pair.replay.summary.outcome.criteria),
            "rc_criteria": dict(pair.rc.summary.outcome.criteria),
            "activation_jump_rad": recovery.activation_jump_rad,
            "command_gap_early_integral": recovery.command_gap_early.integral,
            "reference_settling_s": recovery.reference_settling.settling_time_s,
            "generated_dwell_criteria": dict(recovery.generated_dwell_criteria),
        }
    # Every simulation is done; only now touch the output directory.
    out_dir.mkdir(parents=True, exist_ok=False)
    for case, pair in completed_pairs:
        _plot_case(case, pair, out_dir)
    (out_dir / "timing_traces.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; simulates into a private temporary store (no artifacts recorded)."""
    repo = repository_root()
    parser = argparse.ArgumentParser(description="Generate the recovery timing-trace review evidence.")
    parser.add_argument("--dataset", type=Path, required=True, help="recovery dataset record (TOML)")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=repo / "configs" / "models" / "esn_task_1a_v4.toml")
    parser.add_argument("--evaluation", type=Path, default=repo / "configs" / "evaluations" / "task_1a_nominal_v4.toml")
    parser.add_argument("--tracker", type=Path, default=repo / "configs" / "controllers" / "task_1a_pd_v2.toml")
    parser.add_argument("--out", type=Path, required=True, help="output directory (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    ensure_single_thread()
    if Path(args.out).exists():
        msg = f"{args.out} already exists; traces are immutable (choose a new versioned name)"
        raise FileExistsError(msg)
    record = load_record(Path(args.dataset), RecoveryDatasetRecord)
    scenario = load_scenario(Path(args.scenario))
    record.check_scenario(Path(args.scenario))
    samples = load_samples(verify_payload(open_storage(), record.artifact))
    record.check_samples(samples)
    model_config = load_model_config(Path(args.model))
    estimator = load_nominal_config(Path(args.evaluation)).estimator.config(scenario.timing.dt)
    tracker = load_config(Path(args.tracker), TrackerConfig)
    command = command_line("arm_rc_ctrl.experiments.recovery_timing", sys.argv[1:] if argv is None else argv)
    with tempfile.TemporaryDirectory(prefix="armrc-timing-") as scratch:
        store_root = Path(scratch) / "store"
        store_root.mkdir()
        store = StorageRoot(store_root, repositories=(repo,))
        store.path(record.artifact.payload.uri, mode="write").write_bytes(
            verify_payload(open_storage(), record.artifact).read_bytes()
        )
        summary = generate_timing_traces(
            record,
            samples,
            scenario,
            Path(args.scenario),
            model_config,
            estimator,
            tracker,
            Path(args.out),
            store=store,
            exploratory=bool(args.exploratory),
            command=command,
        )
    cases = cast("dict[str, Any]", summary["cases"])
    print(
        json.dumps(
            {name: {"verified": entry["verified"], "plot": entry["plot"]} for name, entry in cases.items()}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
