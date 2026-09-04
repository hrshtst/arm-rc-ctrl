# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Curated representative development pairs of the recovery experiment (M3R-018; negative-result path).

No model is frozen (model_freeze_v2), so the recovery report illustrates the
development evidence with pairs from the best feasible timing-only trial — a
development-representative point, never a selected model. Scenario selection is
deterministic and median-anchored (the M3 rule): per posture class, the
scenario whose stored pd_v2 early command-gap ratio lies closest to that
class's median, ties broken by scenario ID; the nominal scenario and the first
force scenario by ID complete the set. Every selected scenario runs the full
paired schedule under both frozen trackers through :func:`run_recovery_pair`,
runs persist to the external store first, and only after the last simulation
are the Git pointer records and the curated summary installed (repo writes
never race a later run's provenance).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.evidence import load_report_pointer, open_stored_report
from arm_rc_ctrl.experiments.recovery_objective import RecoveryTrialContext, train_recovery_point
from arm_rc_ctrl.experiments.recovery_search import load_recovery_search, point_from_params
from arm_rc_ctrl.experiments.recovery_slice import run_recovery_pair
from arm_rc_ctrl.experiments.run_record import record_run_pointer
from arm_rc_ctrl.metrics.recovery import RecoveryMetricsReport  # rebuilt from JSON
from arm_rc_ctrl.provenance import (
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.rc.warmup import WarmupConfig
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
    from arm_rc_ctrl.experiments.recovery_search import RecoverySearchProtocol, RecoveryTrialPoint
    from arm_rc_ctrl.experiments.run_record import RunPointerRecord
    from arm_rc_ctrl.experiments.studies import TrialRecord
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "REPRESENTATIVE_CLASSES",
    "REPRESENTATIVE_SCHEMA_VERSION",
    "SELECTION_RULE",
    "PairOutcome",
    "RepresentativeRecord",
    "install_representatives",
    "load_representatives",
    "main",
    "representatives_to_json",
    "run_representatives",
    "select_scenarios",
]

REPRESENTATIVE_SCHEMA_VERSION: Final = 1
REPRESENTATIVE_CLASSES: Final = ("nominal", "posture_small", "posture_large", "force")
SELECTION_RULE: Final = (
    "Per posture class: the development scenario whose stored pd_v2 early command-gap ratio of the "
    "source trial lies closest to the class median (ties by scenario ID); the nominal scenario and "
    "the first force scenario by ID complete the set. Deterministic; never picks flattering examples."
)


@dataclass(frozen=True)
class PairOutcome:
    """One persisted representative pair (both arms of one scenario under one tracker)."""

    scenario_id: str
    kind: str
    tracker: str
    replay_run: str
    rc_run: str
    activation_s: float
    replay_completed: bool
    rc_completed: bool
    recovery: RecoveryMetricsReport | None
    """The RC arm's recovery metrics (``None`` when the RC run did not complete)."""

    def __post_init__(self) -> None:
        """Identify the pair and keep completion flags consistent with the metrics."""
        if not self.scenario_id.strip() or not self.tracker.strip():
            msg = "a pair needs its scenario and tracker identities"
            raise ValueError(msg)
        if self.recovery is not None and not self.rc_completed:
            msg = f"pair {self.scenario_id}/{self.tracker}: recovery metrics require a completed RC run"
            raise ValueError(msg)


@dataclass(frozen=True)
class RepresentativeRecord:
    """The curated representative set: source trial, selection, and every persisted pair."""

    study: str
    trial: int
    point_params: dict[str, float]
    warmup_s: float
    dataset: str
    selection_rule: str
    scenarios: dict[str, str]
    """Selected scenario ID per representative class."""
    pairs: tuple[PairOutcome, ...]
    provenance: ProvenanceRecord
    schema_version: int = field(default=REPRESENTATIVE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Classes, scenarios, and pairs must line up."""
        if self.schema_version != REPRESENTATIVE_SCHEMA_VERSION:
            msg = f"unsupported representative schema_version {self.schema_version}"
            raise ValueError(msg)
        if set(self.scenarios) != set(REPRESENTATIVE_CLASSES):
            msg = f"scenarios must cover exactly {REPRESENTATIVE_CLASSES}, got {sorted(self.scenarios)}"
            raise ValueError(msg)
        expected = {(scenario_id, kind) for kind, scenario_id in self.scenarios.items()}
        seen = {(pair.scenario_id, pair.kind) for pair in self.pairs}
        if seen != expected:
            msg = f"pairs cover {sorted(seen)}, expected {sorted(expected)}"
            raise ValueError(msg)
        run_ids = [run for pair in self.pairs for run in (pair.replay_run, pair.rc_run)]
        if len(set(run_ids)) != len(run_ids):
            msg = "every persisted run appears exactly once"
            raise ValueError(msg)


def _ratio_by_scenario(trial: TrialRecord, kind: str, tracker: str) -> dict[str, float]:
    ratios: dict[str, float] = {}
    index = 0
    while f"components.{index}.kind" in trial.labels:
        prefix = f"components.{index}"
        if trial.labels[prefix + ".kind"] == kind and trial.labels[prefix + ".tracker"] == tracker:
            ratio = trial.metrics.get(prefix + ".gap_ratio")
            if ratio is not None:
                ratios[trial.labels[prefix + ".scenario_id"]] = ratio
        index += 1
    return ratios


def select_scenarios(
    trial: TrialRecord, scenarios: Sequence[RobustnessScenario], *, tracker: str = "pd_v2"
) -> dict[str, str]:
    """Apply :data:`SELECTION_RULE` to the source trial's stored components."""
    by_kind: dict[str, list[RobustnessScenario]] = {}
    for scenario in scenarios:
        by_kind.setdefault(str(scenario.kind), []).append(scenario)
    selected: dict[str, str] = {}
    for kind in REPRESENTATIVE_CLASSES:
        candidates = by_kind.get(kind, [])
        if not candidates:
            msg = f"the development scenarios hold no {kind!r} class"
            raise ValueError(msg)
        if kind in ("nominal", "force"):
            selected[kind] = min(c.scenario_id for c in candidates)
            continue
        ratios = _ratio_by_scenario(trial, kind, tracker)
        if set(ratios) != {c.scenario_id for c in candidates}:
            msg = f"trial {trial.number} lacks stored {tracker!r} ratios for every {kind!r} scenario"
            raise ValueError(msg)
        median = statistics.median(ratios.values())
        selected[kind] = min(ratios, key=lambda sid: (abs(ratios[sid] - median), sid))
    return selected


def run_representatives(
    protocol: RecoverySearchProtocol,
    context: RecoveryTrialContext,
    trial: TrialRecord,
    *,
    study: str,
    store: StorageRoot,
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.experiments.recovery_representative",
) -> tuple[RepresentativeRecord, list[RunPointerRecord]]:
    """Run every selected scenario under both frozen trackers; returns the record and the run pointers.

    Runs persist to the external store only; the caller installs Git pointers
    and the curated JSON afterwards (:func:`install_representatives`), so the
    worktree stays clean for every pair's provenance.
    """
    point: RecoveryTrialPoint = point_from_params(protocol, trial.params)
    trained = train_recovery_point(protocol, context, point)
    if isinstance(trained, str):
        msg = f"the source trial no longer trains: {trained}"
        raise TypeError(msg)
    recipe, _model = trained
    estimator = point.esn.estimator(max_dt_ratio=protocol.max_dt_ratio).config(context.scenario.timing.dt)
    selected = select_scenarios(trial, context.scenarios)
    by_id = {scenario.scenario_id: scenario for scenario in context.scenarios}
    pairs: list[PairOutcome] = []
    pointers: list[RunPointerRecord] = []
    for kind in REPRESENTATIVE_CLASSES:
        scenario = by_id[selected[kind]]
        start = scenario.initial_q(context.dataset.q0_ref)
        for tracker_name, tracker in context.trackers.items():
            pair = run_recovery_pair(
                context.scenario,
                context.scenario_file,
                context.dataset,
                context.reference,
                recipe,
                tracker,
                store=store,
                warmup=WarmupConfig(point.warmup_s),
                exploratory=exploratory,
                estimator=estimator,
                initial_q=start,
                force=scenario.pulse,
                now=now,
                command=command,
            )
            pairs.append(
                PairOutcome(
                    scenario_id=scenario.scenario_id,
                    kind=str(scenario.kind),
                    tracker=tracker_name,
                    replay_run=pair.replay.pointer.artifact.artifact_id,
                    rc_run=pair.rc.pointer.artifact.artifact_id,
                    activation_s=pair.activation_s,
                    replay_completed=pair.replay.summary.termination.kind == "completed",
                    rc_completed=pair.rc.summary.termination.kind == "completed",
                    recovery=pair.recovery,
                )
            )
            pointers.extend((pair.replay.pointer, pair.rc.pointer))
    resolved = {
        "study": study,
        "trial": trial.number,
        "point": {k: float(v) for k, v in point.params().items()},
        "selection_rule": SELECTION_RULE,
        "scenarios": dict(selected),
        "runs": [p.artifact.artifact_id for p in pointers],
        "command": command,
    }
    provenance = collect_provenance(
        resolved, seeds={"reservoir": point.esn.seed}, artifacts=[], exploratory=exploratory, now=now
    )
    require_clean_for_confirmatory(provenance)
    record = RepresentativeRecord(
        study=study,
        trial=trial.number,
        point_params={k: float(v) for k, v in point.params().items()},
        warmup_s=point.warmup_s,
        dataset=context.dataset.artifact.artifact_id,
        selection_rule=SELECTION_RULE,
        scenarios=selected,
        pairs=tuple(pairs),
        provenance=provenance,
    )
    return record, pointers


def install_representatives(
    records_root: Path, output: Path, record: RepresentativeRecord, pointers: Sequence[RunPointerRecord]
) -> None:
    """Install the Git side after the last simulation: run pointers, catalog entries, and the summary."""
    for pointer in pointers:
        record_run_pointer(records_root, pointer)
    if output.exists():
        msg = f"refusing to overwrite {output}"
        raise FileExistsError(msg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(representatives_to_json(record) + "\n", encoding="utf-8")


def representatives_to_json(record: RepresentativeRecord) -> str:
    """Canonical JSON of the record."""
    return canonical_json(to_mapping(record))


def load_representatives(path: Path) -> RepresentativeRecord:
    """Strictly rebuild a representative record from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), RepresentativeRecord)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Persist the curated representative recovery pairs.")
    parser.add_argument("--docs", type=Path, required=True, help="experiment docs directory holding the pointers")
    parser.add_argument("--dataset", type=Path, required=True, help="recovery dataset record (TOML)")
    parser.add_argument("--output", type=Path, required=True, help="representative JSON to write (must not exist)")
    parser.add_argument("--records-root", type=Path, default=None)
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    if Path(args.output).exists():
        msg = f"refusing to overwrite {args.output}"
        raise FileExistsError(msg)
    store = open_storage()
    records_root = repository_root() if args.records_root is None else Path(args.records_root)
    pointer = load_report_pointer(Path(args.docs) / "recovery_search_no_augmentation_v1.toml")
    report = open_stored_report(store, pointer)
    if report.summary.best_number is None:
        msg = "the timing-only study selected no feasible trial; there is no representative point"
        raise ValueError(msg)
    trial = next(t for t in report.summary.trials if t.number == report.summary.best_number)
    protocol = load_recovery_search(repository_root() / report.protocol_file)
    context = RecoveryTrialContext.load(
        protocol, store=store, dataset_file=Path(args.dataset), records_root=records_root
    )
    record, pointers = run_representatives(
        protocol,
        context,
        trial,
        study=report.protocol,
        store=store,
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.experiments.recovery_representative", sys.argv[1:] if argv is None else argv),
    )
    install_representatives(records_root, Path(args.output), record, pointers)
    print(
        json.dumps(
            {
                "study": record.study,
                "trial": record.trial,
                "scenarios": record.scenarios,
                "pairs": len(record.pairs),
                "completed_rc": sum(1 for p in record.pairs if p.rc_completed),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
