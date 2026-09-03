# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The absolute-output development ablation report (M3R-013; recovery plan sections 7.3 and 8, gate 4).

Compares the three matched formulation studies — timing-only, non-decaying
augmentation, contractive augmentation — on identical development scenarios,
frozen trackers, limits, data split, and metrics. Every failure is retained
(per-arm reason taxonomies here; the complete per-trial record lives in the
committed study reports and their Optuna databases). Feasible trials are
additionally evaluated against the section 7.3 eligibility rule per
class-by-tracker cell: both paired median ratios (activation command jump and
early command-gap integral) below 1 and at least 15 of 20 scenarios improving
both. Early command-gap ratios are stored per trial; activation-jump ratios
are re-derived deterministically from the trial-independent replay baselines.
Eligibility is identified without changing the protocol; selection itself is
the model-freeze step (M3R-015).

Command line::

    python -m arm_rc_ctrl.experiments.recovery_ablation --docs docs/experiments/<task>
        --dataset data/records/processed/<id>.toml --output <docs>/development_ablation_v1.json
        --markdown <docs>/development_ablation_v1.md [--records-root ROOT] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.recovery_objective import RATIO_CLASSES, RecoveryTrialContext
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS, load_recovery_search
from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport, load_report
from arm_rc_ctrl.provenance import (
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.rc.esn import ensure_single_thread
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arm_rc_ctrl.experiments.studies import TrialRecord

__all__ = [
    "ABLATION_SCHEMA_VERSION",
    "AblationReport",
    "ArmSummary",
    "CandidateCell",
    "CandidateTrial",
    "ablation_to_json",
    "build_ablation",
    "evaluate_candidates",
    "load_ablation",
    "main",
    "min_improving",
    "render_ablation_markdown",
    "replay_jump_table",
    "summarize_arm",
]

ABLATION_SCHEMA_VERSION: Final = 1
IMPROVING_RULE: Final = "ceil(0.75 * n) scenarios improving both paired metrics (15 of 20 per plan section 7.3)"
_TOP_CANDIDATES: Final = 10
"""Eligible trials shown in the Markdown table; the JSON keeps them all."""

type ReplayJumps = Mapping[tuple[str, float, str], float]
"""Replay activation jump (rad) per ``(tracker, warmup_s, scenario_id)`` — trial-independent."""


def min_improving(n: int) -> int:
    """The section 7.3 improvement threshold: at least 15 of 20 scenarios, generalized as ceil(0.75 n)."""
    if n < 1:
        msg = f"a cell needs at least one paired scenario, got {n}"
        raise ValueError(msg)
    return math.ceil(0.75 * n)


@dataclass(frozen=True)
class CandidateCell:
    """Section 7.3 eligibility figures of one class-by-tracker cell of one feasible trial."""

    gap_median: float
    jump_median: float
    improving_both: int
    n: int
    passes: bool

    def __post_init__(self) -> None:
        """The verdict derives from the recorded figures."""
        if self.n < 1 or not 0 <= self.improving_both <= self.n:
            msg = f"cell counts are inconsistent: improving {self.improving_both} of {self.n}"
            raise ValueError(msg)
        expected = self.gap_median < 1.0 and self.jump_median < 1.0 and self.improving_both >= min_improving(self.n)
        if self.passes != expected:
            msg = f"cell verdict {self.passes} contradicts its figures {self}"
            raise ValueError(msg)


@dataclass(frozen=True)
class CandidateTrial:
    """One feasible trial with its per-cell eligibility evaluation."""

    study: str
    number: int
    value: float
    warmup_s: float
    cells: dict[str, CandidateCell]
    eligible: bool

    def __post_init__(self) -> None:
        """A trial is eligible exactly when every cell passes; the four cells must be present."""
        expected_cells = {f"{kind}:{tracker}" for kind in RATIO_CLASSES for tracker in RECOVERY_TRACKERS}
        if set(self.cells) != expected_cells:
            msg = f"candidate cells {sorted(self.cells)} must be exactly {sorted(expected_cells)}"
            raise ValueError(msg)
        if self.eligible != all(cell.passes for cell in self.cells.values()):
            msg = f"eligible={self.eligible} contradicts the cell verdicts of trial {self.number}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ArmSummary:
    """One formulation study as compared by the ablation."""

    study: str
    formulation: str
    file: str
    protocol_sha256: str
    budget: int
    trials_stored: int
    n_feasible: int
    best_number: int | None
    best_value: float | None
    reasons: dict[str, int]
    """Infeasible trials by gate (the first failing pair's reason head)."""
    feasible_by_warmup: dict[str, int]
    anchor_reason: str | None = None
    anchor_value: float | None = None

    def __post_init__(self) -> None:
        """Counts are consistent."""
        if sum(self.reasons.values()) + self.n_feasible != self.trials_stored:
            msg = f"{self.study}: reasons {sum(self.reasons.values())} + feasible {self.n_feasible} != stored"
            raise ValueError(msg)
        if sum(self.feasible_by_warmup.values()) != self.n_feasible:
            msg = f"{self.study}: feasible_by_warmup does not sum to n_feasible"
            raise ValueError(msg)


@dataclass(frozen=True)
class AblationReport:
    """The cross-arm development ablation (JSON evidence; the Markdown renders from it)."""

    dataset: str
    arms: tuple[ArmSummary, ...]
    candidates: tuple[CandidateTrial, ...]
    """Every feasible trial across the arms, each with its section 7.3 evaluation."""
    n_eligible: int
    improving_rule: str
    provenance: ProvenanceRecord
    schema_version: int = field(default=ABLATION_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The eligible count and study names re-derive from the candidates and arms."""
        if self.schema_version != ABLATION_SCHEMA_VERSION:
            msg = f"unsupported ablation schema_version {self.schema_version}"
            raise ValueError(msg)
        if not self.arms:
            msg = "an ablation compares at least one arm"
            raise ValueError(msg)
        if self.n_eligible != sum(1 for c in self.candidates if c.eligible):
            msg = f"n_eligible {self.n_eligible} does not match the candidates"
            raise ValueError(msg)
        studies = {arm.study for arm in self.arms}
        unknown = sorted({c.study for c in self.candidates} - studies)
        if unknown:
            msg = f"candidates reference unknown studies {unknown}"
            raise ValueError(msg)
        by_study: dict[str, int] = {}
        for candidate in self.candidates:
            by_study[candidate.study] = by_study.get(candidate.study, 0) + 1
        for arm in self.arms:
            if by_study.get(arm.study, 0) != arm.n_feasible:
                msg = f"{arm.study}: {by_study.get(arm.study, 0)} candidates != n_feasible {arm.n_feasible}"
                raise ValueError(msg)


def _reason_head(reason: str) -> str:
    tail = reason.partition("]: ")[2] if reason.startswith("scenario ") else reason
    if not tail:
        return "(none)"
    parts = tail.split(":")
    return ":".join(parts[:2]) if parts[0] == "limit_violation" and len(parts) > 1 else parts[0]


def summarize_arm(file: str, report: RecoveryStudyReport) -> ArmSummary:
    """Reduce one study report to the figures the ablation compares."""
    reasons: dict[str, int] = {}
    feasible_by_warmup: dict[str, int] = {}
    anchor_reason: str | None = None
    anchor_value: float | None = None
    for trial in report.summary.trials:
        if trial.labels.get("armrc.comparison") is not None:
            anchor_reason = trial.labels.get("reason") or None
            anchor_value = trial.value
        if trial.flags.get("feasible") is True:
            key = f"{trial.params.get('warmup_s', float('nan')):g}"
            feasible_by_warmup[key] = feasible_by_warmup.get(key, 0) + 1
            continue
        head = _reason_head(trial.labels.get("reason", ""))
        reasons[head] = reasons.get(head, 0) + 1
    return ArmSummary(
        study=report.protocol,
        formulation=report.formulation,
        file=file,
        protocol_sha256=report.protocol_sha256,
        budget=report.budget,
        trials_stored=len(report.summary.trials),
        n_feasible=report.n_feasible,
        best_number=report.summary.best_number,
        best_value=report.summary.best_value,
        reasons=dict(sorted(reasons.items())),
        feasible_by_warmup=dict(sorted(feasible_by_warmup.items())),
        anchor_reason=anchor_reason,
        anchor_value=anchor_value,
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _candidate(study: str, trial: TrialRecord, replay_jumps: ReplayJumps) -> CandidateTrial:
    warmup = float(trial.params["warmup_s"])
    gaps: dict[str, list[float]] = {}
    jumps: dict[str, list[float]] = {}
    improving: dict[str, int] = {}
    index = 0
    while f"components.{index}.kind" in trial.labels:
        prefix = f"components.{index}"
        kind = trial.labels[prefix + ".kind"]
        if kind in RATIO_CLASSES:
            tracker = trial.labels[prefix + ".tracker"]
            scenario_id = trial.labels[prefix + ".scenario_id"]
            cell = f"{kind}:{tracker}"
            gap = trial.metrics.get(prefix + ".gap_ratio")
            rc_jump = trial.metrics.get(prefix + ".activation_jump_rad")
            replay_jump = replay_jumps.get((tracker, warmup, scenario_id))
            if gap is None or rc_jump is None or replay_jump is None or replay_jump <= 0:
                msg = (
                    f"trial {trial.number} of {study!r}: pair ({scenario_id}, {tracker}) lacks its paired "
                    f"figures (gap {gap!r}, jump {rc_jump!r}, replay jump {replay_jump!r})"
                )
                raise ValueError(msg)
            gaps.setdefault(cell, []).append(gap)
            jumps.setdefault(cell, []).append(rc_jump / replay_jump)
            if gap < 1.0 and rc_jump < replay_jump:
                improving[cell] = improving.get(cell, 0) + 1
        index += 1
    cells: dict[str, CandidateCell] = {}
    for cell, values in sorted(gaps.items()):
        gap_median = _median(values)
        jump_median = _median(jumps[cell])
        n = len(values)
        count = improving.get(cell, 0)
        cells[cell] = CandidateCell(
            gap_median=gap_median,
            jump_median=jump_median,
            improving_both=count,
            n=n,
            passes=gap_median < 1.0 and jump_median < 1.0 and count >= min_improving(n),
        )
    value = trial.value
    if value is None:
        msg = f"feasible trial {trial.number} of {study!r} has no objective value"
        raise ValueError(msg)
    return CandidateTrial(
        study=study,
        number=trial.number,
        value=float(value),
        warmup_s=warmup,
        cells=cells,
        eligible=all(cell.passes for cell in cells.values()),
    )


def evaluate_candidates(study: str, trials: Sequence[TrialRecord], replay_jumps: ReplayJumps) -> list[CandidateTrial]:
    """Evaluate every feasible trial of one study against the section 7.3 eligibility rule."""
    return [_candidate(study, t, replay_jumps) for t in trials if t.flags.get("feasible") is True]


def replay_jump_table(
    context: RecoveryTrialContext, pairs: Sequence[tuple[str, float]]
) -> dict[tuple[str, float, str], float]:
    """Recompute the trial-independent replay activation jumps for the requested (tracker, warm-up) pairs."""
    table: dict[tuple[str, float, str], float] = {}
    for tracker, warmup in sorted(set(pairs)):
        for component in context.replay_components(tracker, warmup):
            if component.activation_jump_rad is not None:
                table[(tracker, warmup, component.scenario_id)] = component.activation_jump_rad
    return table


def build_ablation(
    inputs: Sequence[tuple[str, RecoveryStudyReport]],
    replay_jumps: ReplayJumps,
    *,
    dataset: str,
    provenance: ProvenanceRecord,
) -> AblationReport:
    """Compose the cross-arm report from the study reports and the recomputed replay jumps."""
    arms = tuple(summarize_arm(file, report) for file, report in inputs)
    datasets = {report.dataset for _file, report in inputs}
    if datasets != {dataset}:
        msg = f"every study must evaluate dataset {dataset!r}, got {sorted(datasets)}"
        raise ValueError(msg)
    candidates: list[CandidateTrial] = []
    for _file, report in inputs:
        candidates.extend(evaluate_candidates(report.protocol, report.summary.trials, replay_jumps))
    candidates.sort(key=lambda c: (c.value, c.study, c.number))
    return AblationReport(
        dataset=dataset,
        arms=arms,
        candidates=tuple(candidates),
        n_eligible=sum(1 for c in candidates if c.eligible),
        improving_rule=IMPROVING_RULE,
        provenance=provenance,
    )


def ablation_to_json(report: AblationReport) -> str:
    """Canonical JSON of the report."""
    return canonical_json(to_mapping(report))


def load_ablation(path: Path) -> AblationReport:
    """Strictly rebuild a report from JSON (re-deriving its counts)."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), AblationReport)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


_LIMITATIONS: Final = (
    (
        "Synthetic-sample-count confound: the augmented arms train on 1 + N_aug episodes (17-65) against the "
        "timing-only arm's single demonstration, so augmentation family and training-set size change together; "
        "the matched grids bound but do not remove this confound."
    ),
    (
        "First-infeasible censoring: an infeasible trial stops at its first failing (scenario, tracker) pair, "
        "so the reason taxonomy counts first failures, not all failures a full sweep would find."
    ),
    (
        "Single scripted demonstration (approved decision D6): results do not establish a basin of attraction "
        "outside the augmented training tube, and a negative augmentation result here is valid evidence."
    ),
    (
        "Development data only: no confirmatory seed, level, or outcome was read, and the section 7.3 gates "
        "and their ordering are unchanged; candidate identification here does not select a model (M3R-015 "
        "does)."
    ),
)


def render_ablation_markdown(report: AblationReport) -> str:
    """The Markdown development-ablation report."""
    timing = next((arm for arm in report.arms if arm.formulation == "no_augmentation"), None)
    dirty = " (dirty)" if report.provenance.project_dirty else ""
    lines = [
        "# Task 1-a recovery development ablation (v1)",
        "",
        "## Summary",
        "",
        f"- Dataset `{report.dataset}`; commit `{report.provenance.project_commit[:12]}`{dirty}.",
        (
            "- Matched studies (identical budgets, scenarios, trackers, limits, metrics): "
            + ", ".join(f"`{arm.study}`" for arm in report.arms)
            + "."
        ),
        (
            f"- Eligible candidates under the section 7.3 rule: {report.n_eligible} "
            f"of {len(report.candidates)} feasible trials ({report.improving_rule})."
        ),
        "",
        "## Arms",
        "",
        "| study | formulation | budget | stored | feasible | best trial | best worst-cell gap ratio |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {arm.study} | {arm.formulation} | {arm.budget} | {arm.trials_stored} | {arm.n_feasible} "
        f"| {arm.best_number if arm.best_number is not None else 'none'} | {_fmt(arm.best_value)} |"
        for arm in report.arms
    )
    lines.extend(
        [
            "",
            "## Failure taxonomy",
            "",
            "First failing gate of every infeasible trial (complete per-trial records live in the study",
            "reports and their Optuna databases; nothing is discarded):",
            "",
        ]
    )
    for arm in report.arms:
        anchor = f" Anchor `anchor-v4-tw1`: {arm.anchor_reason or 'feasible'}."
        lines.append(f"- `{arm.study}`:{anchor}")
        lines.extend(f"    - {reason}: {count}" for reason, count in sorted(arm.reasons.items(), key=lambda kv: -kv[1]))
    if timing is not None:
        lines.extend(
            [
                "",
                "## Timing-only arm",
                "",
                "Feasible trials by warm-up (D2 asks for the shortest duration passing the common gates):",
                "",
                "| warm-up (s) | feasible trials |",
                "| --- | --- |",
            ]
        )
        lines.extend(f"| {warmup} | {count} |" for warmup, count in timing.feasible_by_warmup.items())
    eligible = [c for c in report.candidates if c.eligible]
    lines.extend(
        [
            "",
            "## Eligible candidates (section 7.3)",
            "",
            (
                "Per class-by-tracker cell: median early command-gap ratio < 1, median activation-jump "
                "ratio < 1 (jump ratios re-derived from the trial-independent replay baselines), and "
                "improvement of both metrics in at least 15 of 20 scenarios."
            ),
            "",
        ]
    )
    if not eligible:
        lines.append("No feasible trial satisfies every cell; this negative result is retained as-is.")
    else:
        lines.extend(
            [
                "| study | trial | worst gap median | worst jump median | min improving | warm-up (s) |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for candidate in eligible[:_TOP_CANDIDATES]:
            worst_gap = max(cell.gap_median for cell in candidate.cells.values())
            worst_jump = max(cell.jump_median for cell in candidate.cells.values())
            least = min(cell.improving_both for cell in candidate.cells.values())
            lines.append(
                f"| {candidate.study} | {candidate.number} | {_fmt(worst_gap)} | {_fmt(worst_jump)} "
                f"| {least} | {candidate.warmup_s:g} |"
            )
        if len(eligible) > _TOP_CANDIDATES:
            lines.append("")
            lines.append(f"({len(eligible) - _TOP_CANDIDATES} further eligible trials are in the JSON report.)")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {text}" for text in _LIMITATIONS)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Render the recovery development ablation from the study reports.")
    parser.add_argument("--docs", type=Path, required=True, help="experiment docs directory holding the studies")
    parser.add_argument("--dataset", type=Path, required=True, help="recovery dataset record (TOML)")
    parser.add_argument("--output", type=Path, required=True, help="ablation JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, required=True, help="ablation Markdown to write (must not exist)")
    parser.add_argument("--records-root", type=Path, default=None, help="root the dataset record is relative to")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    ensure_single_thread()  # before rclib is imported and provenance is collected
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    files = sorted(Path(args.docs).glob("recovery_search_*_v1.json"))
    inputs = [(f.name, load_report(f)) for f in files]
    formulations = {report.formulation for _name, report in inputs}
    if len(inputs) != len(RECOVERY_TRACKERS) + 1 or len(formulations) != len(inputs):
        msg = f"expected the three formulation studies under {args.docs}, found {[n for n, _r in inputs]}"
        raise ValueError(msg)
    root = repository_root() if args.records_root is None else Path(args.records_root)
    protocol_file = repository_root() / next(
        report.protocol_file for _name, report in inputs if report.formulation == "no_augmentation"
    )
    protocol = load_recovery_search(protocol_file)
    store = open_storage()
    context = RecoveryTrialContext.load(protocol, store=store, dataset_file=Path(args.dataset), records_root=root)
    pairs = [
        (tracker, float(trial.params["warmup_s"]))
        for _name, report in inputs
        for trial in report.summary.trials
        if trial.flags.get("feasible") is True
        for tracker in RECOVERY_TRACKERS
    ]
    replay_jumps = replay_jump_table(context, pairs)
    resolved = {
        "reports": {name: sha256_file(Path(args.docs) / name) for name, _report in inputs},
        "dataset": context.dataset.artifact.artifact_id,
        "protocol_file": next(r.protocol_file for _n, r in inputs if r.formulation == "no_augmentation"),
        "command": command_line("arm_rc_ctrl.experiments.recovery_ablation", sys.argv[1:] if argv is None else argv),
    }
    provenance = collect_provenance(
        resolved, seeds={}, artifacts=[], exploratory=bool(args.exploratory), now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    report = build_ablation(inputs, replay_jumps, dataset=context.dataset.artifact.artifact_id, provenance=provenance)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(ablation_to_json(report) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_ablation_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "arms": [arm.study for arm in report.arms],
                "candidates": len(report.candidates),
                "n_eligible": report.n_eligible,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
