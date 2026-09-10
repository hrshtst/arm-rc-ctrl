# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Accounting of the executed pilot (M3REP-006; repetition plan section 9, package 5).

After the full fixed panel has run, every one of the 120 behavioral model
configurations and the 36 S-effective numerical fits must be accounted for,
including failures: which configurations have evidence, their status, first
failure, executed and unexecuted pairs, replay-blocked censoring, and whether
they crossed the historical speed limit; which replay banks exist; and that
every manifest binds the canonical execution identity and the C11 caveat. The
accounting lists any missing configuration explicitly rather than inferring
it, so a partial execution is visible as such.

Command line::

    python -m arm_rc_ctrl.experiments.repetition_accounting account
        --manifest <docs>/panel_manifest_v1.json --validation <docs>/numerical_validation_v1.json
        --evidence-dir <docs>/evidence --output <docs>/pilot_execution_v1.json --markdown <docs>/pilot_execution_v1.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.repetition_evaluation import (
    C11_CAVEAT,
    MODEL_STATUSES,
    EvidencePointer,
    ModelEvidence,
    ReplayBank,
    load_model_evidence,
    load_pointer,
    load_replay_bank,
    pointer_name,
)
from arm_rc_ctrl.experiments.repetition_numerics import load_validation
from arm_rc_ctrl.experiments.repetition_panel import EXPERIMENT_LABEL, load_panel
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec, panel_arms
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
    verify_artifact,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import open_storage
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "ACCOUNTING_SCHEMA_VERSION",
    "BankAccount",
    "ModelAccount",
    "PilotAccounting",
    "account_pilot",
    "load_accounting",
    "main",
    "render_accounting_markdown",
]

ACCOUNTING_SCHEMA_VERSION: Final = 1
_SHA256_HEX: Final = 64
_MODULE: Final = "arm_rc_ctrl.experiments.repetition_accounting"


@dataclass(frozen=True)
class ModelAccount:
    """One behavioral configuration's accounting line (from its manifest; ``present`` false when none exists)."""

    panel_label: str
    arm: str
    formulation: str
    count: int
    present: bool
    evaluation_identity: str | None = None
    fit_identity: str | None = None
    status: str | None = None
    first_failure: str | None = None
    n_completed: int = 0
    n_infeasible: int = 0
    n_replay_blocked: int = 0
    n_unexecuted: int = 0
    crossed_historical: bool | None = None
    cells: dict[str, float] = field(default_factory=dict)
    payload: ArtifactReference | None = None
    execution_identity: str | None = None
    carries_c11: bool | None = None

    def __post_init__(self) -> None:
        """A present line carries its manifest facts."""
        if self.present != (
            self.evaluation_identity is not None and self.status is not None and self.payload is not None
        ):
            msg = f"{self.panel_label}/{self.arm}: a present configuration carries its identity, status, and payload"
            raise ValueError(msg)
        if self.status is not None and self.status not in MODEL_STATUSES:
            msg = f"{self.panel_label}/{self.arm}: unknown status {self.status!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class BankAccount:
    """One replay bank's accounting line."""

    warmup_s: float
    identity: str
    n_pairs: int
    n_completed: int
    n_infeasible: int
    payload: ArtifactReference
    execution_identity: str


@dataclass(frozen=True)
class PilotAccounting:
    """The committed accounting of the executed pilot."""

    experiment: str
    panel_manifest_sha256: str
    numerical_validation_sha256: str
    canonical_execution_identity: str
    models: tuple[ModelAccount, ...]
    banks: tuple[BankAccount, ...]
    n_configurations: int
    n_present: int
    n_missing: int
    missing: tuple[str, ...]
    statuses: dict[str, int]
    n_rc_runs: int
    n_replay_runs: int
    n_unexecuted_pairs: int
    n_replay_blocked_pairs: int
    n_crossed_historical: int
    numerical_reference_fits: int
    """The S-effective fits of the numerical validation (never evaluated behaviorally)."""
    all_bind_canonical_execution: bool
    all_carry_c11: bool
    complete: bool
    provenance: ProvenanceRecord
    schema_version: int = field(default=ACCOUNTING_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Counts and decisions re-derive from the lines."""
        if self.schema_version != ACCOUNTING_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = "unsupported accounting schema or experiment"
            raise ValueError(msg)
        for name in ("panel_manifest_sha256", "numerical_validation_sha256", "canonical_execution_identity"):
            if not is_hex(getattr(self, name), _SHA256_HEX):
                msg = f"{name} must be 64 lowercase hex characters"
                raise ValueError(msg)
        present = [m for m in self.models if m.present]
        derived = {
            "n_configurations": len(self.models),
            "n_present": len(present),
            "n_missing": len(self.models) - len(present),
            "missing": tuple(f"{m.panel_label}/{m.arm}" for m in self.models if not m.present),
            "statuses": {s: sum(1 for m in present if m.status == s) for s in MODEL_STATUSES},
            "n_rc_runs": sum(m.n_completed + m.n_infeasible for m in present),
            "n_replay_runs": sum(b.n_pairs for b in self.banks),
            "n_unexecuted_pairs": sum(m.n_unexecuted for m in present),
            "n_replay_blocked_pairs": sum(m.n_replay_blocked for m in present),
            "n_crossed_historical": sum(1 for m in present if m.crossed_historical),
            "all_bind_canonical_execution": all(
                m.execution_identity == self.canonical_execution_identity for m in present
            )
            and all(b.execution_identity == self.canonical_execution_identity for b in self.banks),
            "all_carry_c11": all(bool(m.carries_c11) for m in present),
        }
        for name, value in derived.items():
            if getattr(self, name) != value:
                msg = f"{name} contradicts the accounting lines ({getattr(self, name)!r} != {value!r})"
                raise ValueError(msg)
        expected_complete = (
            self.n_missing == 0 and self.all_bind_canonical_execution and self.all_carry_c11 and bool(self.banks)
        )
        if self.complete != expected_complete:
            msg = "complete contradicts the lines"
            raise ValueError(msg)


def _model_line(
    label: str, arm: ArmSpec, pointer: EvidencePointer | None, evidence: ModelEvidence | None
) -> ModelAccount:
    if pointer is None or evidence is None:
        return ModelAccount(
            panel_label=label, arm=arm.label, formulation=arm.formulation, count=arm.count, present=False
        )
    return ModelAccount(
        panel_label=label,
        arm=arm.label,
        formulation=arm.formulation,
        count=arm.count,
        present=True,
        evaluation_identity=evidence.evaluation_identity,
        fit_identity=None if evidence.fit is None else evidence.fit.identity,
        status=evidence.status,
        first_failure=evidence.first_failure,
        n_completed=evidence.n_completed,
        n_infeasible=evidence.n_infeasible,
        n_replay_blocked=evidence.n_replay_blocked,
        n_unexecuted=evidence.n_unexecuted,
        crossed_historical=evidence.crossed_historical,
        cells=dict(evidence.cells),
        payload=pointer.payload,
        execution_identity=evidence.execution.identity,
        carries_c11=C11_CAVEAT in evidence.caveats,
    )


def account_pilot(
    *,
    store: StorageRoot,
    evidence_dir: Path,
    manifest_file: Path,
    validation_file: Path,
    provenance: ProvenanceRecord,
) -> PilotAccounting:
    """Read every pointer of the panel's behavioral configurations and replay banks and account for them."""
    manifest = load_panel(manifest_file)
    validation = load_validation(validation_file)
    canonical = validation.execution.identity
    arms = [arm for arm in panel_arms() if arm.behavioral]
    lines: list[ModelAccount] = []
    for entry in manifest.entries:
        for arm in arms:
            path = evidence_dir / pointer_name("model", f"{entry.label}/{arm.label}")
            if not path.exists():
                lines.append(_model_line(entry.label, arm, None, None))
                continue
            pointer = load_pointer(path)
            evidence = load_model_evidence(verify_artifact(store, pointer.payload))
            if evidence.evaluation_identity != pointer.identity:
                msg = (
                    f"{path.name}: the manifest {evidence.evaluation_identity[:12]} is not the pointer's "
                    f"{pointer.identity[:12]}"
                )
                raise ValueError(msg)
            lines.append(_model_line(entry.label, arm, pointer, evidence))
    banks: list[BankAccount] = []
    for path in sorted(evidence_dir.glob("replay__*.toml")):
        pointer = load_pointer(path)
        bank: ReplayBank = load_replay_bank(verify_artifact(store, pointer.payload))
        banks.append(
            BankAccount(
                warmup_s=bank.conditions.warmup_s,
                identity=bank.identity,
                n_pairs=len(bank.pairs),
                n_completed=sum(1 for p in bank.pairs if p.status == "completed"),
                n_infeasible=sum(1 for p in bank.pairs if p.status == "infeasible"),
                payload=pointer.payload,
                execution_identity=bank.execution.identity,
            )
        )
    present = [m for m in lines if m.present]
    references = sum(1 for f in validation.fits if f.arm.arm == "S-effective")
    return PilotAccounting(
        experiment=EXPERIMENT_LABEL,
        panel_manifest_sha256=sha256_file(manifest_file),
        numerical_validation_sha256=sha256_file(validation_file),
        canonical_execution_identity=canonical,
        models=tuple(lines),
        banks=tuple(banks),
        n_configurations=len(lines),
        n_present=len(present),
        n_missing=len(lines) - len(present),
        missing=tuple(f"{m.panel_label}/{m.arm}" for m in lines if not m.present),
        statuses={s: sum(1 for m in present if m.status == s) for s in MODEL_STATUSES},
        n_rc_runs=sum(m.n_completed + m.n_infeasible for m in present),
        n_replay_runs=sum(b.n_pairs for b in banks),
        n_unexecuted_pairs=sum(m.n_unexecuted for m in present),
        n_replay_blocked_pairs=sum(m.n_replay_blocked for m in present),
        n_crossed_historical=sum(1 for m in present if m.crossed_historical),
        numerical_reference_fits=references,
        all_bind_canonical_execution=all(m.execution_identity == canonical for m in present)
        and all(b.execution_identity == canonical for b in banks),
        all_carry_c11=all(bool(m.carries_c11) for m in present),
        complete=(len(present) == len(lines))
        and all(m.execution_identity == canonical for m in present)
        and all(b.execution_identity == canonical for b in banks)
        and all(bool(m.carries_c11) for m in present)
        and bool(banks),
        provenance=provenance,
    )


def accounting_to_json(accounting: PilotAccounting) -> str:
    """Canonical JSON."""
    return canonical_json(to_mapping(accounting))


def load_accounting(path: Path) -> PilotAccounting:
    """Strictly rebuild the accounting from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), PilotAccounting)


def render_accounting_markdown(a: PilotAccounting) -> str:
    """The Markdown rendering: totals, replay banks, and one line per configuration in panel and report order."""
    lines = [
        "# Task 1-a repetition pilot execution accounting (v1)",
        "",
        (
            f"Experiment `{a.experiment}`: panel manifest sha256 `{a.panel_manifest_sha256[:12]}`, numerical "
            f"validation sha256 `{a.numerical_validation_sha256[:12]}`, canonical execution identity "
            f"`{a.canonical_execution_identity[:12]}`, project commit `{a.provenance.project_commit[:12]}`"
            f"{' (dirty)' if a.provenance.project_dirty else ''}."
        ),
        "",
        "## Totals",
        "",
        f"- Behavioral configurations: {a.n_present} of {a.n_configurations} with evidence; missing: {a.n_missing}.",
        f"- Statuses: {', '.join(f'{s} {n}' for s, n in a.statuses.items())}.",
        (
            f"- RC runs executed: {a.n_rc_runs}; unexecuted pairs: {a.n_unexecuted_pairs}; replay-blocked pairs: "
            f"{a.n_replay_blocked_pairs}; replay runs: {a.n_replay_runs} in {len(a.banks)} banks."
        ),
        f"- Models that crossed the historical 6 rad/s limit in an executed RC run: {a.n_crossed_historical}.",
        (
            "- S-effective numerical reference fits (validated, never evaluated behaviorally): "
            f"{a.numerical_reference_fits}."
        ),
        (
            f"- Every manifest binds the canonical execution identity: {a.all_bind_canonical_execution}; every "
            f"manifest carries the C11 caveat: {a.all_carry_c11}; accounting complete: **{a.complete}**."
        ),
    ]
    if a.missing:
        lines += ["", "Missing configurations (no evidence pointer): " + ", ".join(f"`{m}`" for m in a.missing) + "."]
    lines += [
        "",
        "## Replay banks",
        "",
        "| warm-up (s) | identity | pairs | completed | infeasible |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {b.warmup_s:g} | `{b.identity[:12]}` | {b.n_pairs} | {b.n_completed} | {b.n_infeasible} |" for b in a.banks
    )
    lines += [
        "",
        "## Configurations",
        "",
        "| entry | arm | status | completed | infeasible | blocked | unexecuted | crossed 6 rad/s | first failure |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for m in a.models:
        if not m.present:
            lines.append(f"| {m.panel_label} | {m.arm} | **missing** |  |  |  |  |  |  |")
            continue
        lines.append(
            f"| {m.panel_label} | {m.arm} | {m.status} | {m.n_completed} | {m.n_infeasible} | {m.n_replay_blocked} "
            f"| {m.n_unexecuted} | {'yes' if m.crossed_historical else 'no'} | {m.first_failure or ''} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        (
            "- This is an execution accounting, not a result: paired metrics, comparisons, and interpretation "
            "belong to the report (M3REP-007) and its review (M3REP-GATE)."
        ),
        (
            "- Feasibility here is under the pilot's 12 rad/s evaluation abort; a run that crossed the historical "
            "6 rad/s limit is not a recovery-v1 success (plan section 7.1)."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Account for the executed repeated-demonstration pilot.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    account = subparsers.add_parser("account", help="read every evidence pointer and account for the panel")
    account.add_argument("--manifest", type=str, required=True, help="frozen panel manifest JSON")
    account.add_argument("--validation", type=str, required=True, help="numerical validation JSON")
    account.add_argument("--evidence-dir", type=str, required=True, help="directory of the Git pointer records")
    account.add_argument("--output", type=str, required=True, help="accounting JSON to write (must not exist)")
    account.add_argument("--markdown", type=str, required=True, help="accounting Markdown to write (must not exist)")
    account.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    root = repository_root()
    store = open_storage()
    manifest_file = Path(cast("str", args.manifest))
    validation_file = Path(cast("str", args.validation))
    evidence_dir = Path(cast("str", args.evidence_dir))
    resolved = {
        "manifest": sha256_file(manifest_file),
        "validation": sha256_file(validation_file),
        "evidence_dir": evidence_dir.relative_to(root).as_posix()
        if evidence_dir.is_relative_to(root)
        else evidence_dir.name,
        "command": command_line(_MODULE, argv),
    }
    provenance = collect_provenance(
        resolved, seeds={}, artifacts=[], exploratory=bool(args.exploratory), now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    accounting = account_pilot(
        store=store,
        evidence_dir=evidence_dir,
        manifest_file=manifest_file,
        validation_file=validation_file,
        provenance=provenance,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(accounting_to_json(accounting) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_accounting_markdown(accounting), encoding="utf-8")
    print(
        json.dumps(
            {
                "present": accounting.n_present,
                "missing": accounting.n_missing,
                "statuses": accounting.statuses,
                "complete": accounting.complete,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
