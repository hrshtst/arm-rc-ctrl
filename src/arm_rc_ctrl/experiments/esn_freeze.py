# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Freezing the point an ESN search selected into versioned configurations (``docs/PLAN.md`` section 10; M3-006).

The selected point becomes a model configuration (``configs/models/*.toml``,
trained into a recipe by ``python -m arm_rc_ctrl.rc.train``) and a nominal
evaluation configuration (``configs/evaluations/*.toml``) carrying the
point's estimator cutoffs with the protocol's frozen tracker. Both files cite
the study report and its protocol digest so the regression locks can bind them.

Command line::

    python -m arm_rc_ctrl.experiments.esn_freeze --report docs/experiments/task_1a/esn_search.json
        --protocol configs/studies/esn_search_1a.toml --name esn-task-1a-v3
        --model configs/models/esn_task_1a_v3.toml --evaluation configs/evaluations/task_1a_nominal_v3.toml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import tomli_w

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec, NominalConfig
from arm_rc_ctrl.experiments.esn_search import load_esn_search, protocol_digest
from arm_rc_ctrl.experiments.esn_study import load_report
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.esn_search import EsnSearchProtocol
    from arm_rc_ctrl.experiments.esn_study import EsnStudyReport
    from arm_rc_ctrl.rc.train import ModelConfig

__all__ = ["frozen_evaluation", "frozen_model", "main", "render_evaluation_toml", "render_model_toml"]


def _check(report: EsnStudyReport, protocol: EsnSearchProtocol) -> None:
    digest = protocol_digest(protocol)
    if report.protocol != protocol.name or report.protocol_sha256 != digest:
        msg = (
            f"report {report.protocol!r} ({report.protocol_sha256[:12]}) was not produced by "
            f"protocol {protocol.name!r} ({digest[:12]})"
        )
        raise ValueError(msg)
    if report.best_point is None or report.summary.best_number is None:
        msg = "the report selects no trial (no feasible completed trial)"
        raise ValueError(msg)
    best = next((t for t in report.summary.trials if t.number == report.summary.best_number), None)
    if best is None or best.flags.get("feasible") is not True:
        number = report.summary.best_number
        msg = f"the selected trial {number} is not flagged feasible; only feasible trials can be frozen"
        raise ValueError(msg)
    if len(report.summary.trials) < report.budget:
        stored = len(report.summary.trials)
        msg = f"the study stored {stored} trials of its budget {report.budget}; finish it before freezing"
        raise ValueError(msg)


def frozen_model(report: EsnStudyReport, protocol: EsnSearchProtocol, *, name: str) -> ModelConfig:
    """The selected point applied to the protocol's base model."""
    _check(report, protocol)
    assert report.best_point is not None  # _check() rejects reports without a selection
    return report.best_point.model_config(protocol.base_model(), name=name)


def frozen_evaluation(
    report: EsnStudyReport, protocol: EsnSearchProtocol, *, name: str, tracker_file: Path
) -> NominalConfig:
    """The nominal evaluation with the selected estimator cutoffs and the protocol's frozen tracker."""
    _check(report, protocol)
    assert report.best_point is not None  # _check() rejects reports without a selection
    point = report.best_point
    estimator = EstimatorSpec(point.velocity_cutoff_hz, point.acceleration_cutoff_hz, protocol.max_dt_ratio)
    return NominalConfig(name=name, tracker=tracker_file, estimator=estimator)


def _header(report: EsnStudyReport, report_file: str, what: str) -> str:
    best = report.summary.best_number
    value = report.summary.best_value
    lines = [
        f"# {what} (docs/TASKS.md M3-006): the point selected by the recorded ESN search",
        f"# {report_file} (protocol {report.protocol_file}, digest {report.protocol_sha256[:12]}; trial {best} of",
        f"# {len(report.summary.trials)} stored / budget {report.budget}, {report.n_feasible} feasible; objective",
        f"# {value!r} rad median movement-window joint RMSE over the development scenarios; dataset {report.dataset}).",
    ]
    return "\n".join(lines) + "\n"


def render_model_toml(report: EsnStudyReport, protocol: EsnSearchProtocol, *, name: str, report_file: str) -> str:
    """The model configuration TOML with a header citing the study."""
    config = frozen_model(report, protocol, name=name)
    header = _header(report, report_file, "Task ESN model configuration")
    note = (
        f"# Input transform and readout solver stay as in {protocol.model.name}; "
        f"tracker {report.tracker} is not tuned.\n"
    )
    return header + note + tomli_w.dumps(to_mapping(config))


def render_evaluation_toml(
    report: EsnStudyReport, protocol: EsnSearchProtocol, *, name: str, tracker_file: Path, report_file: str
) -> str:
    """The nominal evaluation TOML with a header citing the study."""
    config = frozen_evaluation(report, protocol, name=name, tracker_file=tracker_file)
    header = _header(report, report_file, "Nominal RC closed-loop evaluation")
    note = "# The estimator cutoffs are the selected point's; the tracker is the protocol's frozen baseline.\n"
    body = {"name": config.name, "tracker": tracker_file.as_posix(), "estimator": to_mapping(config.estimator)}
    return header + note + tomli_w.dumps(body)


def _relative(path: Path) -> str:
    root = repository_root()
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else path.name


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Freeze the point an ESN search selected into versioned configs.")
    parser.add_argument("--report", type=Path, required=True, help="study report JSON (docs/experiments/...)")
    parser.add_argument("--protocol", type=Path, required=True, help="the search protocol the report came from")
    parser.add_argument("--name", required=True, help="model configuration name (e.g. esn-task-1a-v3)")
    parser.add_argument("--model", type=Path, required=True, help="model configuration TOML to write (must not exist)")
    parser.add_argument("--evaluation", type=Path, required=True, help="evaluation TOML to write (must not exist)")
    parser.add_argument(
        "--tracker",
        type=Path,
        default=None,
        help="tracker TOML the evaluation references (default: configs/controllers/<protocol tracker>.toml)",
    )
    args = parser.parse_args(argv)
    for target in (args.model, args.evaluation):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    report = load_report(Path(args.report))
    protocol = load_esn_search(Path(args.protocol))
    tracker_file = (
        Path("..") / "controllers" / f"task_1a_{protocol.tracker}.toml" if args.tracker is None else Path(args.tracker)
    )
    report_file = _relative(Path(args.report))
    evaluation_name = f"{Path(args.evaluation).stem.replace('_', '-')}"
    Path(args.model).write_text(render_model_toml(report, protocol, name=args.name, report_file=report_file))
    Path(args.evaluation).write_text(
        render_evaluation_toml(
            report, protocol, name=evaluation_name, tracker_file=tracker_file, report_file=report_file
        )
    )
    print(f"wrote {args.model} and {args.evaluation} from trial {report.summary.best_number}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
