# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Input-scale development pilot for the RC target generator (M2-018).

Usage::

    python -m arm_rc_ctrl.experiments.scale_pilot --protocol configs/studies/input_scale_pilot_1a.toml \\
        --dataset data/records/processed/<id>.toml --report docs/experiments/task_1a/input_scale_pilot.json \\
        --markdown docs/experiments/task_1a/input_scale_pilot.md
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, load_config, to_mapping
from arm_rc_ctrl.controllers.adapter import GeneratorTrackingController
from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator
from arm_rc_ctrl.controllers.tracking import TrackerConfig  # resolved at runtime by the report loader
from arm_rc_ctrl.data.phases import intervals_from_phases
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import SampleSet, load_samples
from arm_rc_ctrl.experiments.baselines import load_frozen_baseline
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.replay import bind_dataset, dwell_outcome
from arm_rc_ctrl.experiments.simulation import GENERATOR_CHANNELS, simulate
from arm_rc_ctrl.metrics.joint import JointAnglePolicy, joint_rmse
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.rc.esn import EsnModel
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.rc.teacher_forcing import InputEncoder, InputTransform, build_episode
from arm_rc_ctrl.rc.train import ModelConfig, load_model_config
from arm_rc_ctrl.rc.training import train_readout
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import require_finite

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Cell",
    "CellOutcome",
    "ScalePilotProtocol",
    "ScalePilotReport",
    "ScaleSelection",
    "Variant",
    "load_protocol",
    "load_scale_pilot_report",
    "main",
    "render_markdown",
    "run_scale_pilot",
    "select_anchor",
    "summarize_cells",
]

REPORT_SCHEMA_VERSION: Final = 1


def _positive(values: tuple[float, ...], name: str) -> None:
    require_finite(values, name)
    if not values or any(v <= 0 for v in values) or len(set(values)) != len(values):
        msg = f"{name} must be a non-empty list of distinct positive values, got {values}"
        raise ValueError(msg)


@dataclass(frozen=True)
class ScaleGrid:
    """Fixed physical input scales to sweep (rad and rad/s)."""

    q_scales: tuple[float, ...]
    dq_scales: tuple[float, ...]

    def __post_init__(self) -> None:
        """Both axes are distinct positive values in increasing order."""
        _positive(self.q_scales, "grid.q_scales")
        _positive(self.dq_scales, "grid.dq_scales")
        if list(self.q_scales) != sorted(self.q_scales) or list(self.dq_scales) != sorted(self.dq_scales):
            msg = "grid axes must be increasing"
            raise ValueError(msg)


@dataclass(frozen=True)
class Variants:
    """Adjacent ESN settings evaluated in every cell."""

    ridge_alphas: tuple[float, ...]
    reservoir_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        """Alphas are positive and seeds distinct and non-negative."""
        _positive(self.ridge_alphas, "variants.ridge_alphas")
        if not self.reservoir_seeds or len(set(self.reservoir_seeds)) != len(self.reservoir_seeds):
            msg = f"variants.reservoir_seeds must be non-empty and distinct, got {self.reservoir_seeds}"
            raise ValueError(msg)
        if any(seed < 0 for seed in self.reservoir_seeds):
            msg = "variants.reservoir_seeds must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class SelectionRules:
    """How cells become feasible and how the anchor is chosen."""

    min_feasible_fraction: float

    def __post_init__(self) -> None:
        """The threshold lies in (0, 1]."""
        if not (0 < self.min_feasible_fraction <= 1):
            msg = f"selection.min_feasible_fraction must lie in (0, 1], got {self.min_feasible_fraction!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScalePilotProtocol:
    """A versioned input-scale pilot."""

    name: str
    scenario: Path
    model: Path
    trackers: tuple[str, ...]
    grid: ScaleGrid
    variants: Variants
    estimator: EstimatorSpec
    selection: SelectionRules

    def __post_init__(self) -> None:
        """Name and trackers are non-empty and distinct."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if not self.trackers or len(set(self.trackers)) != len(self.trackers):
            msg = f"trackers must be a non-empty list of distinct baselines, got {self.trackers}"
            raise ValueError(msg)


def load_protocol(path: Path) -> ScalePilotProtocol:
    """Load and validate a pilot protocol."""
    return load_config(path, ScalePilotProtocol)


@dataclass(frozen=True)
class Cell:
    """One pair of input scales."""

    q_scale: float
    dq_scale: float


@dataclass(frozen=True)
class Variant:
    """One evaluated (cell, ESN setting, tracker) combination."""

    cell: Cell
    ridge_alpha: float
    reservoir_seed: int
    tracker: str
    termination: str
    success: bool
    criteria: dict[str, bool]
    move_joint_rmse: float | None
    fit_rmse: float
    boundary_jump: float | None
    peak_velocity: float
    saturation_fraction: float


@dataclass(frozen=True)
class CellOutcome:
    """Feasibility of one cell over its variants."""

    cell: Cell
    variants: int
    feasible_variants: int
    feasible_fraction: float
    median_move_joint_rmse: float | None
    """Median movement RMSE over the feasible variants (``None`` when there are none)."""
    feasible: bool
    """Whether the fraction reaches the protocol's threshold."""


@dataclass(frozen=True)
class ScaleSelection:
    """The selected anchor and the feasible region around it."""

    q_scale: float
    dq_scale: float
    region: tuple[Cell, ...]
    """Every feasible cell of the grid (the anchor and its neighbours are among them)."""


@dataclass(frozen=True)
class ScalePilotReport:
    """Every variant, every cell, the selection, and the provenance of the run."""

    protocol: str
    scenario_file: str
    dataset: str
    model_config: str
    trackers: dict[str, TrackerConfig]
    rules: SelectionRules
    variants: tuple[Variant, ...]
    cells: tuple[CellOutcome, ...]
    selection: ScaleSelection
    provenance: ProvenanceRecord
    schema_version: int = field(default=REPORT_SCHEMA_VERSION)


def load_scale_pilot_report(path: Path) -> ScalePilotReport:
    """Load a report written by :func:`main`."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ScalePilotReport)


def _repo_relative(path: Path) -> str:
    root = repository_root()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name


def _evaluate_variant(
    scenario: ScenarioConfig,
    samples: SampleSet,
    record: ProcessedDatasetRecord,
    base: ModelConfig,
    cell: Cell,
    alpha: float,
    seed: int,
    trackers: dict[str, TrackerConfig],
    estimator: EstimatorSpec,
) -> list[Variant]:
    """Train one ESN for the cell/variant and run it closed loop with every tracker."""
    if record.normalization is None:
        msg = f"dataset {record.artifact.artifact_id} records no normalization statistics"
        raise ValueError(msg)
    transform = InputTransform.derive(
        "fixed_scale", record.normalization, fixed_scales={"q": cell.q_scale, "dq": cell.dq_scale}
    )
    encoder = InputEncoder(transform, record.dof, record.task_code_dim)
    esn = dataclasses.replace(
        base.esn,
        reservoir=dataclasses.replace(base.esn.reservoir, seed=seed),
        readout=dataclasses.replace(base.esn.readout, alpha=alpha),
    )
    model = EsnModel(esn, input_dim=encoder.input_dim, output_dim=record.dof)
    fit = train_readout(model, [build_episode(samples, encoder, source=record.artifact.artifact_id)])
    lower = np.array([link.q_min for link in scenario.robot.links], dtype=np.float64)
    upper = np.array([link.q_max for link in scenario.robot.links], dtype=np.float64)
    intervals = intervals_from_phases(samples.t, samples.phase)
    move = (samples.t >= intervals.move[0]) & (samples.t < intervals.move[1])
    policy = JointAnglePolicy.limited(scenario.dof)
    results: list[Variant] = []
    for name, gains in trackers.items():
        generator = RcTargetGenerator(
            model,
            encoder,
            CausalDerivativeEstimator(estimator.config(scenario.timing.dt), record.dof),
            position_bounds=(lower, upper),
        )
        controller = GeneratorTrackingController(
            generator, gains, scenario.limits.torque, hold_until_s=scenario.timing.intervals.prime[1]
        )
        arrays, termination = simulate(
            scenario, controller, duration_s=float(samples.t[-1]), channels=GENERATOR_CHANNELS
        )
        q = arrays.arrays["q"]
        n = min(q.shape[0], samples.n_samples)
        rmse = None
        if termination.is_completed and move[:n].any():
            rmse = joint_rmse(q[:n][move[:n]], samples.q[:n][move[:n]], policy).aggregate
        criteria = dwell_outcome(scenario, samples, arrays, termination)
        results.append(
            Variant(
                cell=cell,
                ridge_alpha=alpha,
                reservoir_seed=seed,
                tracker=name,
                termination=termination.kind,
                success=termination.is_completed and all(criteria.values()),
                criteria=criteria,
                move_joint_rmse=rmse,
                fit_rmse=fit.rmse,
                boundary_jump=controller.boundary_jump,
                peak_velocity=float(np.abs(arrays.arrays["dq"]).max()),
                saturation_fraction=float(np.mean(arrays.arrays["saturation"])),
            )
        )
    return results


def summarize_cells(protocol: ScalePilotProtocol, variants: Sequence[Variant]) -> tuple[CellOutcome, ...]:
    """Feasibility fraction and median RMSE of every grid cell."""
    outcomes: list[CellOutcome] = []
    for q_scale, dq_scale in itertools.product(protocol.grid.q_scales, protocol.grid.dq_scales):
        cell = Cell(q_scale, dq_scale)
        members = [v for v in variants if v.cell == cell]
        if not members:
            msg = f"no variants for cell {cell}"
            raise ValueError(msg)
        feasible = [v for v in members if v.success]
        fraction = len(feasible) / len(members)
        rmses = [v.move_joint_rmse for v in feasible if v.move_joint_rmse is not None]
        outcomes.append(
            CellOutcome(
                cell=cell,
                variants=len(members),
                feasible_variants=len(feasible),
                feasible_fraction=fraction,
                median_move_joint_rmse=statistics.median(rmses) if rmses else None,
                feasible=fraction >= protocol.selection.min_feasible_fraction,
            )
        )
    return tuple(outcomes)


def select_anchor(protocol: ScalePilotProtocol, cells: Sequence[CellOutcome]) -> ScaleSelection:
    """The feasible cell with the highest fraction (ties: lower median RMSE) whose grid neighbours are feasible."""
    by_cell = {c.cell: c for c in cells}
    q_axis, dq_axis = protocol.grid.q_scales, protocol.grid.dq_scales

    def neighbours(cell: Cell) -> list[Cell]:
        i, j = q_axis.index(cell.q_scale), dq_axis.index(cell.dq_scale)
        found: list[Cell] = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if 0 <= i + di < len(q_axis) and 0 <= j + dj < len(dq_axis):
                found.append(Cell(q_axis[i + di], dq_axis[j + dj]))
        return found

    candidates = [c for c in cells if c.feasible and all(by_cell[n].feasible for n in neighbours(c.cell))]
    if not candidates:
        msg = "no feasible cell has only feasible grid neighbours; widen the grid or revisit the settings"
        raise ValueError(msg)
    best = max(
        candidates,
        key=lambda c: (
            c.feasible_fraction,
            -(c.median_move_joint_rmse if c.median_move_joint_rmse is not None else 1e9),
        ),
    )
    region = tuple(c.cell for c in cells if c.feasible)
    return ScaleSelection(best.cell.q_scale, best.cell.dq_scale, region)


def run_scale_pilot(
    protocol: ScalePilotProtocol,
    protocol_file: Path,
    dataset: ProcessedDatasetRecord,
    samples: SampleSet,
    *,
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.scale_pilot",
) -> ScalePilotReport:
    """Run the whole pilot and build the report."""
    scenario = load_scenario(protocol.scenario)
    bind_dataset(scenario, protocol.scenario, dataset, samples)
    base = load_model_config(protocol.model)
    trackers = {name: load_frozen_baseline(name) for name in protocol.trackers}
    protocol_mapping = to_mapping(protocol)
    protocol_mapping["scenario"] = _repo_relative(protocol.scenario)
    protocol_mapping["model"] = _repo_relative(protocol.model)
    resolved = {
        "protocol": protocol_mapping,
        "protocol_file": _repo_relative(protocol_file),
        "model": to_mapping(base),
        "trackers": {name: to_mapping(gains) for name, gains in trackers.items()},
        "dataset": dataset.artifact.artifact_id,
        "command": command,
    }
    payload = dataset.artifact.payload
    provenance = collect_provenance(
        resolved,
        seeds={f"reservoir_{i}": seed for i, seed in enumerate(protocol.variants.reservoir_seeds)},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    variants: list[Variant] = []
    for q_scale, dq_scale in itertools.product(protocol.grid.q_scales, protocol.grid.dq_scales):
        for alpha, seed in itertools.product(protocol.variants.ridge_alphas, protocol.variants.reservoir_seeds):
            variants.extend(
                _evaluate_variant(
                    scenario, samples, dataset, base, Cell(q_scale, dq_scale), alpha, seed, trackers, protocol.estimator
                )
            )
    cells = summarize_cells(protocol, variants)
    return ScalePilotReport(
        protocol=protocol.name,
        scenario_file=_repo_relative(protocol.scenario),
        dataset=dataset.artifact.artifact_id,
        model_config=_repo_relative(protocol.model),
        trackers=trackers,
        rules=protocol.selection,
        variants=tuple(variants),
        cells=cells,
        selection=select_anchor(protocol, cells),
        provenance=provenance,
    )


def render_markdown(report: ScalePilotReport) -> str:
    """A feasibility-fraction grid plus the selection."""
    q_axis = sorted({c.cell.q_scale for c in report.cells})
    dq_axis = sorted({c.cell.dq_scale for c in report.cells})
    by_cell = {(c.cell.q_scale, c.cell.dq_scale): c for c in report.cells}
    dirty = " (dirty)" if report.provenance.project_dirty else ""
    trackers = ", ".join(f"`{name}`" for name in sorted(report.trackers))
    lines = [
        f"# Input-scale pilot `{report.protocol}`",
        "",
        (
            f"Dataset `{report.dataset}`, scenario `{report.scenario_file}`, model config `{report.model_config}`, "
            f"commit `{report.provenance.project_commit[:12]}`{dirty}, trackers {trackers}."
        ),
        "",
        (
            "Feasible fraction of variants per cell (rows: q scale in rad; columns: dq scale in rad/s); a cell is "
            f"feasible at >= {report.rules.min_feasible_fraction:g}, marked with *."
        ),
        "",
        "| q \\ dq | " + " | ".join(f"{dq:g}" for dq in dq_axis) + " |",
        "|---|" + "---|" * len(dq_axis),
    ]
    for q in q_axis:
        cells: list[str] = []
        for dq in dq_axis:
            c = by_cell[(q, dq)]
            rmse = "" if c.median_move_joint_rmse is None else f" ({c.median_move_joint_rmse:.2g})"
            cells.append(f"{c.feasible_fraction:.2f}{'*' if c.feasible else ''}{rmse}")
        lines.append(f"| {q:g} | " + " | ".join(cells) + " |")
    s = report.selection
    lines += [
        "",
        "Cell entries: feasible fraction (median movement RMSE in rad over feasible variants).",
        "",
        "## Selection",
        "",
        (
            f"- anchor: q scale {s.q_scale:g} rad, dq scale {s.dq_scale:g} rad/s (highest feasible fraction among "
            "cells whose grid neighbours are all feasible; ties broken by lower median RMSE)"
        ),
        f"- feasible region: {len(s.region)} of {len(report.cells)} cells",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Select the fixed input scales of the RC target generator.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="processed dataset record (TOML)")
    parser.add_argument("--report", type=Path, required=True, help="JSON report to write (must not exist)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown summary (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    for target in (args.report, args.markdown):
        if target is not None and Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    protocol_file = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_file)
    store = open_storage()
    dataset = load_record(Path(args.dataset), ProcessedDatasetRecord)
    samples = load_samples(verify_payload(store, dataset.artifact))
    report = run_scale_pilot(
        protocol,
        protocol_file,
        dataset,
        samples,
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.scale_pilot", sys.argv[1:] if argv is None else argv),
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_mapping(report), indent=2, sort_keys=True, allow_nan=False)
    Path(args.report).write_text(text + "\n", encoding="utf-8")
    if args.markdown is not None:
        Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "variants": len(report.variants),
                "feasible_cells": sum(1 for c in report.cells if c.feasible),
                "selection": to_mapping(report.selection),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
