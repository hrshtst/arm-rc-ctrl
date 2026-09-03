# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reservoir-seed sensitivity panel of a recovery formulation study (M3R-014/M3R-015; plan section 7.3).

Mirrors the M3 panel (``esn_stability``) for the recovery search: the leading
feasible trials of one study report are re-evaluated with every panel seed —
only the reservoir seed of the ESN point changes; warm-up, augmentation, and
every other parameter stay the trial's own. Each re-evaluation runs the full
paired development grid, so a configuration's panel spread reports how much of
its worst-cell early command-gap ratio is reservoir-seed luck. The panel is a
gate input for the model freeze; it never changes the section 7.3 rules.

Command line::

    python -m arm_rc_ctrl.experiments.recovery_stability --report <study>.json
        --protocol configs/studies/<study>.toml --dataset data/records/processed/<id>.toml
        --top 3 --seeds 101 202 303 --output <docs>/<name>.json [--markdown <docs>/<name>.md]
        [--records-root ROOT] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.recovery_objective import RecoveryTrialContext, evaluate_recovery_point
from arm_rc_ctrl.experiments.recovery_search import (
    RecoveryTrialPoint,  # loaded from JSON (annotation resolution needs the runtime name)
    load_recovery_search,
    point_from_params,
    recovery_protocol_digest,
)
from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport, load_report
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.recovery_search import RecoverySearchProtocol
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "STABILITY_SCHEMA_VERSION",
    "ConfigurationStability",
    "SeedOutcome",
    "StabilityReport",
    "leading_trials",
    "load_stability",
    "main",
    "render_markdown",
    "run_stability",
    "stability_to_json",
]

STABILITY_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class SeedOutcome:
    """One configuration evaluated with one panel reservoir seed."""

    seed: int
    feasible: bool
    objective: float
    reason: str | None
    pairs_evaluated: int


@dataclass(frozen=True)
class ConfigurationStability:
    """A leading trial re-evaluated over the seed panel."""

    trial: int
    label: str | None
    point: RecoveryTrialPoint
    own_objective: float
    outcomes: tuple[SeedOutcome, ...]
    feasible_seeds: int = 0
    objective_median: float | None = None
    objective_min: float | None = None
    objective_max: float | None = None

    def __post_init__(self) -> None:
        """Recompute the summary of the outcomes (feasible ones only) and reject stored mismatches."""
        feasible = [o.objective for o in self.outcomes if o.feasible]
        computed = (
            len(feasible),
            statistics.median(feasible) if feasible else None,
            min(feasible) if feasible else None,
            max(feasible) if feasible else None,
        )
        stored = (self.feasible_seeds, self.objective_median, self.objective_min, self.objective_max)
        if (stored[0] or any(v is not None for v in stored[1:])) and stored != computed:
            msg = f"stored stability summary of trial {self.trial} does not match its outcomes"
            raise ValueError(msg)
        for name, value in zip(
            ("feasible_seeds", "objective_median", "objective_min", "objective_max"), computed, strict=True
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class StabilityReport:
    """The seed panel of a recovery study's leading configurations."""

    protocol: str
    protocol_sha256: str
    formulation: str
    study_report: str
    """Repository-relative path of the study report the configurations come from."""
    dataset: str
    panel_seeds: tuple[int, ...]
    configurations: tuple[ConfigurationStability, ...]
    provenance: ProvenanceRecord
    schema_version: int = field(default=STABILITY_SCHEMA_VERSION)


def leading_trials(report: RecoveryStudyReport, top: int) -> list[int]:
    """Trial numbers of the ``top`` feasible completed trials by objective (ties by number)."""
    feasible = [t for t in report.summary.trials if t.flags.get("feasible") is True and t.value is not None]
    ordered = sorted(feasible, key=lambda t: (cast("float", t.value), t.number))
    return [t.number for t in ordered[:top]]


def run_stability(
    report: RecoveryStudyReport,
    report_file: Path,
    protocol: RecoverySearchProtocol,
    *,
    context: RecoveryTrialContext,
    top: int,
    seeds: Sequence[int],
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.recovery_stability",
) -> StabilityReport:
    """Re-evaluate the leading feasible trials of ``report`` with every panel reservoir seed."""
    digest = recovery_protocol_digest(protocol)
    if report.protocol != protocol.name or report.protocol_sha256 != digest:
        msg = f"report {report.protocol!r} was not produced by protocol {protocol.name!r}"
        raise ValueError(msg)
    if top < 1 or not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        msg = "top must be >= 1 and seeds a non-empty list of distinct non-negative integers"
        raise ValueError(msg)
    trials = leading_trials(report, top)
    if not trials:
        msg = "the study report holds no feasible trial to evaluate"
        raise ValueError(msg)
    payload = context.dataset.artifact.payload
    resolved = {
        "protocol": protocol.name,
        "protocol_sha256": digest,
        "formulation": report.formulation,
        "study_report": _relative(report_file),
        "trials": trials,
        "panel_seeds": list(seeds),
        "command": command,
    }
    provenance = collect_provenance(
        resolved,
        seeds={f"panel.{i}": seed for i, seed in enumerate(seeds)},
        artifacts=[ArtifactReference(payload.uri, payload.sha256, payload.size)],
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    by_number = {t.number: t for t in report.summary.trials}
    configurations: list[ConfigurationStability] = []
    for number in trials:
        trial = by_number[number]
        point = point_from_params(protocol, trial.params)
        outcomes: list[SeedOutcome] = []
        for seed in seeds:
            panel_point = replace(point, esn=replace(point.esn, seed=seed))
            evaluation = evaluate_recovery_point(protocol, context, panel_point)
            outcomes.append(
                SeedOutcome(
                    seed,
                    evaluation.feasible,
                    evaluation.objective,
                    evaluation.reason,
                    len(evaluation.components),
                )
            )
        configurations.append(
            ConfigurationStability(
                trial=number,
                label=trial.labels.get("armrc.comparison"),
                point=point,
                own_objective=cast("float", trial.value),
                outcomes=tuple(outcomes),
            )
        )
    return StabilityReport(
        protocol=protocol.name,
        protocol_sha256=digest,
        formulation=report.formulation,
        study_report=_relative(report_file),
        dataset=context.dataset.artifact.artifact_id,
        panel_seeds=tuple(seeds),
        configurations=tuple(configurations),
        provenance=provenance,
    )


def stability_to_json(report: StabilityReport) -> str:
    """Canonical JSON of the report."""
    return canonical_json(to_mapping(report))


def load_stability(path: Path) -> StabilityReport:
    """Strictly rebuild a stability report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), StabilityReport)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def render_markdown(report: StabilityReport) -> str:
    """Per configuration: its own objective, feasible panel seeds, and the spread of the panel objectives."""
    dirty = " (dirty)" if report.provenance.project_dirty else ""
    seeds = ", ".join(str(s) for s in report.panel_seeds)
    header = [
        "trial",
        "label",
        "own objective (ratio)",
        "own seed",
        "feasible panel seeds",
        "panel median",
        "panel min",
        "panel max",
    ]
    lines = [
        f"# Reservoir-seed sensitivity panel of `{report.protocol}`",
        "",
        f"- Formulation `{report.formulation}`; study report `{report.study_report}`",
        f"  (protocol digest `{report.protocol_sha256[:12]}`), dataset `{report.dataset}`.",
        "- Objective: worst class-by-tracker cell median of the early command-gap ratio.",
        f"- Panel seeds: {seeds}; commit `{report.provenance.project_commit[:12]}`{dirty}.",
        "",
        _row(header),
        _row(["---"] * len(header)),
    ]
    for c in report.configurations:
        cells = [
            c.trial,
            c.label or "-",
            _fmt(c.own_objective),
            c.point.esn.seed,
            f"{c.feasible_seeds}/{len(c.outcomes)}",
            _fmt(c.objective_median),
            _fmt(c.objective_min),
            _fmt(c.objective_max),
        ]
        lines.append(_row(cells))
    for c in report.configurations:
        lines.extend(["", f"## Trial {c.trial}", ""])
        lines.append(_row(["seed", "feasible", "objective (ratio)", "pairs", "reason"]))
        lines.append(_row(["---"] * 5))
        lines.extend(
            _row([o.seed, o.feasible, _fmt(o.objective), o.pairs_evaluated, o.reason or ""]) for o in c.outcomes
        )
    return "\n".join(lines) + "\n"


def _relative(path: Path) -> str:
    root = repository_root()
    resolved = path.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else path.name


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Re-evaluate a recovery study's leading trials over a reservoir-seed panel."
    )
    parser.add_argument("--report", type=Path, required=True, help="recovery study report JSON")
    parser.add_argument("--protocol", type=Path, required=True, help="the search protocol the report came from")
    parser.add_argument("--dataset", type=Path, required=True, help="recovery dataset record (TOML)")
    parser.add_argument("--top", type=int, default=3, help="number of leading feasible trials to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="panel reservoir seeds")
    parser.add_argument("--output", type=Path, required=True, help="stability report JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown to write (must not exist)")
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    for target in (args.output, args.markdown):
        if target is not None and Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    protocol = load_recovery_search(Path(args.protocol))
    report = load_report(Path(args.report))
    store: StorageRoot = open_storage()
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    context = RecoveryTrialContext.load(
        protocol, store=store, dataset_file=Path(args.dataset), records_root=records_root
    )
    result = run_stability(
        report,
        Path(args.report),
        protocol,
        context=context,
        top=args.top,
        seeds=list(args.seeds),
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.recovery_stability", sys.argv[1:] if argv is None else argv),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(stability_to_json(result) + "\n", encoding="utf-8")
    if args.markdown is not None:
        Path(args.markdown).write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol": result.protocol,
                "formulation": result.formulation,
                "configurations": [c.trial for c in result.configurations],
                "feasible_seeds": {str(c.trial): c.feasible_seeds for c in result.configurations},
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
