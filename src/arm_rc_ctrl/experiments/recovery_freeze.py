# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Applying the approved model-freeze rule to the recovery development evidence (M3R-015; D5, section 7.3).

The freeze record is a pure derivation from committed evidence: the four
formulation study reports and the development ablation (whose eligibility
evaluation already carries the re-derived activation-jump ratios). Eligibility
and ordering follow section 7.3 verbatim; the residual arm enters the record
as an excluded, exploratory input per decision D4. When no candidate is
eligible, the record states the negative outcome with no selection — a valid
result the protocol anticipates — and the confirmatory gate stays closed. The
frozen path of the schema (an eligible selection plus its reservoir-seed
panel) is fully validated so a later protocol version can reuse it, but this
module never freezes automatically: a selection requires an explicit panel
report, and none exists without an eligible candidate.

Command line::

    python -m arm_rc_ctrl.experiments.recovery_freeze --docs docs/experiments/<task>
        --output <docs>/model_freeze_v1.json --markdown <docs>/model_freeze_v1.md [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.evidence import load_report_pointer
from arm_rc_ctrl.experiments.recovery_ablation import AblationReport, load_ablation
from arm_rc_ctrl.provenance import (
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FREEZE_RULE",
    "FREEZE_SCHEMA_VERSION",
    "FreezeRecord",
    "StudyInput",
    "build_freeze",
    "freeze_to_json",
    "load_freeze",
    "main",
    "render_freeze_markdown",
]

FREEZE_SCHEMA_VERSION: Final = 1
FREEZE_RULE: Final = (
    "Per posture class and frozen tracker pair independently: median early command-gap ratio < 1, "
    "median activation-jump ratio < 1, and both paired metrics improving in at least 15 of 20 "
    "scenarios; among eligible models, lexicographic selection on the worst class-by-tracker cell "
    "median of the early command-gap ratio, then the worst cell median of endpoint settling time, "
    "then the worst cell median of applied-torque RMS; reservoir-seed-panel stability precedes any "
    "freeze (recovery plan section 7.3; approved decision D5)."
)

type FreezeOutcome = Literal["frozen", "negative"]


@dataclass(frozen=True)
class StudyInput:
    """One formulation study as an input of the freeze decision."""

    file: str
    study: str
    formulation: str
    protocol_sha256: str
    n_feasible: int
    included: bool
    """Whether the study's candidates enter the freeze rule (the residual arm is excluded per D4)."""
    note: str | None = None

    def __post_init__(self) -> None:
        """Identify every input; an excluded study states why."""
        if not self.file.strip() or not self.study.strip():
            msg = "a study input needs its file and study names"
            raise ValueError(msg)
        if not self.included and not (self.note or "").strip():
            msg = f"excluded study {self.study!r} must state why (e.g. exploratory per D4)"
            raise ValueError(msg)


@dataclass(frozen=True)
class FreezeRecord:
    """The applied freeze rule: inputs, eligibility, and the outcome (a negative is a valid result)."""

    dataset: str
    rule: str
    studies: tuple[StudyInput, ...]
    ablation_file: str
    ablation_sha256: str
    n_candidates: int
    n_eligible: int
    eligible_trials: tuple[str, ...]
    """``study:number`` labels ordered by the primary lexicographic key (worst-cell gap median)."""
    outcome: FreezeOutcome
    selection: str | None
    """The frozen ``study:number`` (``None`` for a negative outcome)."""
    panel: str | None
    """The reservoir-seed panel report backing a freeze (``None`` when inapplicable)."""
    provenance: ProvenanceRecord
    schema_version: int = field(default=FREEZE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The outcome re-derives from the eligibility figures; a freeze needs its panel."""
        if self.schema_version != FREEZE_SCHEMA_VERSION:
            msg = f"unsupported freeze schema_version {self.schema_version}"
            raise ValueError(msg)
        if not self.studies or not any(s.included for s in self.studies):
            msg = "the freeze needs at least one included study"
            raise ValueError(msg)
        if self.n_eligible != len(self.eligible_trials):
            msg = f"n_eligible {self.n_eligible} does not match the {len(self.eligible_trials)} listed trials"
            raise ValueError(msg)
        if self.n_eligible > self.n_candidates:
            msg = "eligible trials cannot exceed the candidates"
            raise ValueError(msg)
        if self.outcome == "negative":
            if self.selection is not None or self.panel is not None or self.n_eligible != 0:
                msg = "a negative outcome records no selection, no panel, and zero eligible trials"
                raise ValueError(msg)
            return
        if self.selection is None or self.selection not in self.eligible_trials:
            msg = "a frozen outcome selects one of the eligible trials"
            raise ValueError(msg)
        if self.panel is None or not self.panel.strip():
            msg = "a frozen outcome cites its reservoir-seed panel report"
            raise ValueError(msg)


def build_freeze(
    ablation: AblationReport,
    *,
    ablation_file: str,
    ablation_sha256: str,
    studies: Sequence[StudyInput],
    provenance: ProvenanceRecord,
) -> FreezeRecord:
    """Apply the rule to the committed evidence; the record freezes nothing without an eligible candidate."""
    arms = {arm.study: arm for arm in ablation.arms}
    for study in studies:
        if study.included:
            arm = arms.get(study.study)
            if arm is None or arm.protocol_sha256 != study.protocol_sha256:
                msg = f"included study {study.study!r} does not match the ablation's arms"
                raise ValueError(msg)
            if arm.n_feasible != study.n_feasible:
                msg = f"study {study.study!r}: feasible count {study.n_feasible} contradicts the ablation"
                raise ValueError(msg)
        elif study.study in arms:
            msg = f"excluded study {study.study!r} must not appear among the ablation's arms"
            raise ValueError(msg)
    included_feasible = sum(s.n_feasible for s in studies if s.included)
    if included_feasible != len(ablation.candidates):
        msg = f"the ablation evaluates {len(ablation.candidates)} candidates, inputs list {included_feasible}"
        raise ValueError(msg)
    eligible = sorted((c for c in ablation.candidates if c.eligible), key=lambda c: (c.value, c.study, c.number))
    labels = tuple(f"{c.study}:{c.number}" for c in eligible)
    if labels:
        msg = (
            f"{len(labels)} candidates are eligible ({', '.join(labels[:3])} ...); freezing requires "
            "the reservoir-seed panel and the full lexicographic keys - run the panel and record the "
            "freeze explicitly rather than through this negative-path builder"
        )
        raise ValueError(msg)
    return FreezeRecord(
        dataset=ablation.dataset,
        rule=FREEZE_RULE,
        studies=tuple(studies),
        ablation_file=ablation_file,
        ablation_sha256=ablation_sha256,
        n_candidates=len(ablation.candidates),
        n_eligible=0,
        eligible_trials=(),
        outcome="negative",
        selection=None,
        panel=None,
        provenance=provenance,
    )


def freeze_to_json(record: FreezeRecord) -> str:
    """Canonical JSON of the record."""
    return canonical_json(to_mapping(record))


def load_freeze(path: Path) -> FreezeRecord:
    """Strictly rebuild a freeze record from JSON (re-deriving its invariants)."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), FreezeRecord)


def render_freeze_markdown(record: FreezeRecord) -> str:
    """The Markdown freeze record."""
    dirty = " (dirty)" if record.provenance.project_dirty else ""
    lines = [
        "# Task 1-a recovery model freeze (v1)",
        "",
        "## Rule",
        "",
        record.rule,
        "",
        "## Inputs",
        "",
        f"- Dataset `{record.dataset}`; commit `{record.provenance.project_commit[:12]}`{dirty}.",
        f"- Development ablation `{record.ablation_file}` (sha256 `{record.ablation_sha256[:12]}`).",
        "",
        "| study | formulation | feasible trials | in the rule | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {s.study} | {s.formulation} | {s.n_feasible} | {'yes' if s.included else 'no'} | {s.note or ''} |"
        for s in record.studies
    )
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            (f"- Candidates evaluated: {record.n_candidates}; eligible under section 7.3: {record.n_eligible}."),
        ]
    )
    if record.outcome == "negative":
        lines.extend(
            [
                (
                    "- **NEGATIVE RESULT — no model is frozen.** No feasible trial satisfies every "
                    "class-by-tracker cell of the eligibility rule, so there is no selection and no "
                    "reservoir-seed panel."
                ),
                (
                    "- The confirmatory gate stays closed: the locked suite must not run without an "
                    "eligible frozen model or an owner-approved protocol revision (a new protocol "
                    "version per D5)."
                ),
            ]
        )
    else:  # pragma: no cover - no frozen record exists under protocol v1
        lines.append(f"- Frozen selection: `{record.selection}` (panel `{record.panel}`).")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Apply the recovery model-freeze rule to the committed evidence.")
    parser.add_argument("--docs", type=Path, required=True, help="experiment docs directory")
    parser.add_argument(
        "--ablation",
        type=str,
        default="development_ablation_v2.json",
        help="ablation JSON (relative to --docs) the rule is applied to",
    )
    parser.add_argument("--output", type=Path, required=True, help="freeze record JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, required=True, help="freeze Markdown to write (must not exist)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    docs = Path(args.docs)
    ablation_file = docs / str(args.ablation)
    ablation = load_ablation(ablation_file)
    studies: list[StudyInput] = []
    for file in sorted(docs.glob("recovery_search_*_v1.toml")):
        pointer = load_report_pointer(file)
        studies.append(
            StudyInput(
                file=file.name,
                study=pointer.study,
                formulation=pointer.formulation,
                protocol_sha256=pointer.protocol_sha256,
                n_feasible=pointer.n_feasible,
                included=True,
            )
        )
    residual_file = docs / "residual_search_1a_v1.toml"
    residual = load_report_pointer(residual_file)
    studies.append(
        StudyInput(
            file=residual_file.name,
            study=residual.study,
            formulation=residual.formulation,
            protocol_sha256=residual.protocol_sha256,
            n_feasible=residual.n_feasible,
            included=False,
            note="exploratory per D4; retained as a negative, never predeclared for confirmatory inclusion",
        )
    )
    resolved = {
        "ablation": {ablation_file.name: sha256_file(ablation_file)},
        "studies": {s.file: sha256_file(docs / s.file) for s in studies},
        "command": command_line("arm_rc_ctrl.experiments.recovery_freeze", sys.argv[1:] if argv is None else argv),
    }
    provenance = collect_provenance(
        resolved, seeds={}, artifacts=[], exploratory=bool(args.exploratory), now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    record = build_freeze(
        ablation,
        ablation_file=ablation_file.name,
        ablation_sha256=sha256_file(ablation_file),
        studies=studies,
        provenance=provenance,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(freeze_to_json(record) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_freeze_markdown(record), encoding="utf-8")
    print(
        json.dumps(
            {
                "outcome": record.outcome,
                "candidates": record.n_candidates,
                "n_eligible": record.n_eligible,
                "selection": record.selection,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
