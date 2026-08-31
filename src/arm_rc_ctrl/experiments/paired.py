# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Paired nominal evaluation: RC target generation versus direct replay with the same tracker (M2-014).

Both runs replay the same reference dataset in the same scenario with the same
frozen gains; their :class:`~arm_rc_ctrl.metrics.report.RunReport` objects come
from one :func:`~arm_rc_ctrl.metrics.report.build_report` with identical windows
and metric definitions, so every difference is attributable to the target
generator. Comparison is paired by low-level controller (docs/PLAN.md section 6).

Usage::

    python -m arm_rc_ctrl.experiments.paired --config configs/evaluations/task_1a_nominal.toml \\
        --scenario configs/tasks/task_1a.toml --dataset data/records/processed/<id>.toml \\
        --recipe data/records/models/<id>.toml --report <json> [--markdown <md>] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.config import from_mapping, load_config, to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.experiments.closed_loop import ClosedLoopResult, load_nominal_config, run_nominal
from arm_rc_ctrl.experiments.replay import ReplayResult, run_replay
from arm_rc_ctrl.metrics.report import RunReport
from arm_rc_ctrl.provenance import command_line
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arm_rc_ctrl.controllers.estimator import EstimatorConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.scenario import ScenarioConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "MetricComparison",
    "PairedReport",
    "PairedResult",
    "compare_reports",
    "load_paired_report",
    "main",
    "paired_to_markdown",
    "run_paired_nominal",
]

PAIRED_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class MetricComparison:
    """One scalar metric of both runs; ``None`` when a run terminated before the metric existed."""

    name: str
    unit: str
    rc: float | None
    replay: float | None
    lower_is_better: bool = True

    @property
    def difference(self) -> float | None:
        """``rc - replay`` when both exist."""
        if self.rc is None or self.replay is None:
            return None
        return self.rc - self.replay

    @property
    def ratio(self) -> float | None:
        """``rc / replay`` when both exist and the replay value is non-zero."""
        if self.rc is None or self.replay is None or self.replay == 0:
            return None
        return self.rc / self.replay


@dataclass(frozen=True)
class PairedReport:
    """RC and replay reports of one reference under one tracker, with their metric comparison."""

    scenario: str
    reference_artifact: str
    tracker: str
    """Low-level tracker method shared by both runs (``pd`` or ``computed_torque``)."""
    recipe: str
    rc: RunReport
    replay: RunReport
    metrics: tuple[MetricComparison, ...] = ()
    schema_version: int = field(default=PAIRED_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Both reports describe the same reference under the same windows, paired by tracker."""
        if self.schema_version != PAIRED_SCHEMA_VERSION:
            msg = f"unsupported paired report schema version {self.schema_version}"
            raise ValueError(msg)
        rc, replay = self.rc, self.replay
        if rc.method != f"rc+{self.tracker}" or replay.method != f"replay+{self.tracker}":
            msg = f"methods {rc.method!r} and {replay.method!r} are not the RC/replay pair of tracker {self.tracker!r}"
            raise ValueError(msg)
        if rc.scenario != self.scenario or replay.scenario != self.scenario:
            msg = f"reports describe scenarios {rc.scenario!r}/{replay.scenario!r}, not {self.scenario!r}"
            raise ValueError(msg)
        if rc.reference_artifact != self.reference_artifact or replay.reference_artifact != self.reference_artifact:
            msg = "both reports must be evaluated against the paired report's reference artifact"
            raise ValueError(msg)
        if rc.windows != replay.windows:
            msg = f"metric windows differ: RC {rc.windows} vs replay {replay.windows}"
            raise ValueError(msg)
        if not self.recipe.strip():
            msg = "recipe must name the model recipe of the RC run"
            raise ValueError(msg)
        if not self.metrics:
            object.__setattr__(self, "metrics", compare_reports(rc, replay))


def _scalar(report: RunReport, path: tuple[str, ...]) -> float | None:
    value: object = report
    for name in path:
        if value is None:
            return None
        value = getattr(value, name)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        msg = f"metric {'.'.join(path)} is not a number: {value!r}"
        raise TypeError(msg)
    return float(value)


_METRICS: Final[tuple[tuple[str, tuple[str, ...], str, bool], ...]] = (
    ("joint_rmse", ("joint_rmse", "aggregate"), "rad", True),
    ("dwell_in_tolerance_fraction", ("dwell", "in_tolerance_fraction"), "", False),
    ("dwell_longest_in_tolerance_s", ("dwell", "longest_in_tolerance_s"), "s", False),
    ("dwell_endpoint_rms", ("dwell", "endpoint", "rms"), "m", True),
    ("dwell_endpoint_max", ("dwell", "endpoint", "max"), "m", True),
    ("dwell_velocity_max", ("dwell", "velocity_max"), "rad/s", True),
    ("effort_torque_rms", ("effort", "torque_rms"), "N*m", True),
    ("effort_torque_peak", ("effort", "torque_peak"), "N*m", True),
    ("effort_saturation_fraction", ("effort", "saturation_fraction"), "", True),
    ("effort", ("effort", "effort"), "N^2*m^2*s", True),
    ("move_coverage", ("move_coverage",), "", False),
    ("dwell_coverage", ("dwell_coverage",), "", False),
)


def compare_reports(rc: RunReport, replay: RunReport) -> tuple[MetricComparison, ...]:
    """Every scalar metric of both reports side by side (same metric definitions, same windows)."""
    return tuple(
        MetricComparison(name, unit, _scalar(rc, path), _scalar(replay, path), lower)
        for name, path, unit, lower in _METRICS
    )


@dataclass(frozen=True)
class PairedResult:
    """Outputs of :func:`run_paired_nominal`."""

    rc: ClosedLoopResult
    replay: ReplayResult
    paired: PairedReport


def run_paired_nominal(
    scenario: ScenarioConfig,
    scenario_file: Path,
    dataset: ProcessedDatasetRecord,
    reference: SampleSet,
    recipe: ModelRecipe,
    recipe_name: str,
    tracker: TrackerConfig,
    *,
    store: StorageRoot,
    estimator: EstimatorConfig,
    training_samples: Mapping[str, SampleSet],
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.paired",
) -> PairedResult:
    """Run the RC closed loop and the direct replay with the same tracker and pair their reports."""
    rc = run_nominal(
        scenario,
        scenario_file,
        dataset,
        reference,
        recipe,
        tracker,
        store=store,
        estimator=estimator,
        training_samples=training_samples,
        exploratory=exploratory,
        now=now,
        command=command,
    )
    replay = run_replay(
        scenario,
        scenario_file,
        dataset,
        reference,
        tracker,
        store=store,
        exploratory=exploratory,
        now=now,
        command=command,
    )
    paired = PairedReport(
        scenario=scenario.name,
        reference_artifact=dataset.artifact.artifact_id,
        tracker=tracker.method,
        recipe=recipe_name,
        rc=rc.report,
        replay=replay.report,
    )
    return PairedResult(rc, replay, paired)


def load_paired_report(path: Path) -> PairedReport:
    """Strictly rebuild a paired report from JSON."""
    return from_mapping(json.loads(path.read_text(encoding="utf-8")), PairedReport)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}" if math.isfinite(value) else str(value)


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _run_row(label: str, report: RunReport) -> str:
    failed = ", ".join(report.failed_criteria) or "-"
    return _row([label, report.method, report.run_id, report.termination_kind, report.success, failed])


def paired_to_markdown(report: PairedReport) -> str:
    """A Markdown table of the paired metrics."""
    windows = report.rc.windows
    lines = [
        f"# Paired nominal evaluation: `rc+{report.tracker}` vs `replay+{report.tracker}`",
        "",
        f"Scenario `{report.scenario}`, reference `{report.reference_artifact}`, recipe `{report.recipe}`.",
        f"Windows: move {list(windows.move)} s, dwell {list(windows.dwell)} s (identical for both runs).",
        "",
        _row(["Run", "Method", "Run ID", "Termination", "Success", "Failed criteria"]),
        "|---|---|---|---|---|---|",
        _run_row("RC", report.rc),
        _run_row("replay", report.replay),
        "",
        _row(["Metric", "Unit", "RC", "Replay", "RC - replay", "RC / replay", "Better"]),
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        _row(
            [
                m.name,
                m.unit or "-",
                _fmt(m.rc),
                _fmt(m.replay),
                _fmt(m.difference),
                _fmt(m.ratio),
                "lower" if m.lower_is_better else "higher",
            ]
        )
        for m in report.metrics
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run the paired RC/replay nominal evaluation of task 1-a.")
    parser.add_argument("--config", type=Path, required=True, help="evaluation config (configs/evaluations/*.toml)")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (the reference)")
    parser.add_argument("--recipe", type=Path, required=True, help="model recipe (TOML)")
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True, help="paired report JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown table to write (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    for target in (args.report, args.markdown):
        if target is not None and Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    config = load_nominal_config(Path(args.config))
    scenario = load_scenario(Path(args.scenario))
    store = open_storage()
    dataset = load_record(Path(args.dataset), ProcessedDatasetRecord)
    reference = load_samples(verify_payload(store, dataset.artifact))
    recipe = load_recipe(Path(args.recipe))
    training = load_training_samples(
        recipe, store, records_root=None if args.records_root is None else Path(args.records_root)
    )
    result = run_paired_nominal(
        scenario,
        Path(args.scenario),
        dataset,
        reference,
        recipe,
        Path(args.recipe).name,
        load_config(config.tracker, TrackerConfig),
        store=store,
        estimator=config.estimator.config(scenario.timing.dt),
        training_samples=training,
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.paired", sys.argv[1:] if argv is None else argv),
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(to_mapping(result.paired), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    if args.markdown is not None:
        Path(args.markdown).write_text(paired_to_markdown(result.paired), encoding="utf-8")
    rmse = next(m for m in result.paired.metrics if m.name == "joint_rmse")
    print(
        json.dumps(
            {
                "rc_run": result.paired.rc.run_id,
                "replay_run": result.paired.replay.run_id,
                "rc_termination": result.paired.rc.termination_kind,
                "rc_success": result.paired.rc.success,
                "replay_success": result.paired.replay.success,
                "joint_rmse_rc": rmse.rc,
                "joint_rmse_replay": rmse.replay,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
