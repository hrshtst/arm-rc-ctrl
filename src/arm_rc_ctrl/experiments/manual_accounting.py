# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: accounting for every model and replay bank of the manual study (plan section 7.1).

Complete means complete: the study's own entries are walked in manifest order
and a model with no evidence is listed as missing rather than left out, so a
partial execution is visible as partial instead of as a shorter list. Every
present line is read from the manifest its pointer resolves to, verified by
size and digest first, because a pointer that summarises itself is exactly what
this experiment's evidence integrity work established not to trust.

The record re-derives its own totals from those lines, so an accounting that
disagrees with itself is refused rather than reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.experiments.manual_evaluation import (
    load_manual_model_evidence,
    load_manual_pointer,
    load_manual_replay_bank,
    manual_pointer_name,
)
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.provenance import ArtifactReference, ProvenanceRecord, verify_artifact  # loaded back at run time

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_study import StudyManifest
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "ACCOUNTING_SCHEMA_VERSION",
    "BankAccount",
    "ModelAccount",
    "StudyAccounting",
    "account_study",
]

ACCOUNTING_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class ModelAccount:
    """One study model's line: what its evidence says, or that there is none."""

    label: str
    configuration: str
    arm: str
    present: bool
    identity: str | None = None
    fit_identity: str | None = None
    assignment: str | None = None
    status: str | None = None
    n_pairs: int = 0
    n_completed: int = 0
    n_infeasible: int = 0
    n_unexecuted: int = 0
    execution_identity: str | None = None
    payload: ArtifactReference | None = None

    def __post_init__(self) -> None:
        """A present line carries the facts its manifest stated; an absent one carries none of them."""
        stated = self.identity is not None and self.status is not None and self.payload is not None
        if self.present != stated:
            msg = f"{self.label}: a present model carries its identity, status and payload"
            raise ValueError(msg)
        if self.n_completed + self.n_infeasible + self.n_unexecuted != self.n_pairs:
            msg = f"{self.label}: {self.n_completed}+{self.n_infeasible}+{self.n_unexecuted} is not {self.n_pairs}"
            raise ValueError(msg)


@dataclass(frozen=True)
class BankAccount:
    """One replay bank's line, keyed by the parent and the derivative policy it belongs to."""

    identity: str
    assignment: str
    warmup_s: float
    velocity_cutoff_hz: float
    acceleration_cutoff_hz: float
    n_pairs: int
    n_completed: int
    n_infeasible: int
    execution_identity: str
    payload: ArtifactReference

    def __post_init__(self) -> None:
        """The counts agree with each other."""
        if self.n_completed + self.n_infeasible != self.n_pairs:
            msg = (
                f"{self.assignment}: {self.n_completed} completed and {self.n_infeasible} infeasible "
                f"is not {self.n_pairs}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class StudyAccounting:
    """The accounting of what the study has executed so far, and what it has not."""

    experiment: str
    canonical_execution_identity: str
    models: tuple[ModelAccount, ...]
    banks: tuple[BankAccount, ...]
    n_models: int
    n_present: int
    n_missing: int
    missing: tuple[str, ...]
    statuses: dict[str, int]
    n_rc_runs: int
    n_replay_runs: int
    all_bind_canonical_execution: bool
    complete: bool
    provenance: ProvenanceRecord
    schema_version: int = field(default=ACCOUNTING_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Every total re-derives from the lines, or the record contradicts itself."""
        if self.schema_version != ACCOUNTING_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = f"unsupported accounting schema {self.schema_version} or experiment {self.experiment!r}"
            raise ValueError(msg)
        present = [line for line in self.models if line.present]
        derived = (
            len(self.models),
            len(present),
            len(self.models) - len(present),
            tuple(line.label for line in self.models if not line.present),
            sum(line.n_pairs for line in present),
            sum(bank.n_pairs for bank in self.banks),
            not self.models or len(present) == len(self.models),
        )
        recorded = (
            self.n_models,
            self.n_present,
            self.n_missing,
            self.missing,
            self.n_rc_runs,
            self.n_replay_runs,
            self.complete,
        )
        if derived != recorded:
            msg = "the accounting's totals do not re-derive from its own lines"
            raise ValueError(msg)
        keyed = [line.execution_identity for line in present] + [bank.execution_identity for bank in self.banks]
        if self.all_bind_canonical_execution != all(
            identity == self.canonical_execution_identity for identity in keyed
        ):
            msg = "all_bind_canonical_execution contradicts the lines"
            raise ValueError(msg)


def _model_line(entry_label: str, configuration: str, arm: str, evidence_dir: Path, store: StorageRoot) -> ModelAccount:
    """Read one model's line from the manifest its pointer resolves to, or record that it is absent."""
    path = evidence_dir / manual_pointer_name("model", entry_label)
    if not path.exists():
        return ModelAccount(label=entry_label, configuration=configuration, arm=arm, present=False)
    pointer = load_manual_pointer(path)
    evidence = load_manual_model_evidence(verify_artifact(store, pointer.payload))
    if evidence.identity != pointer.identity:
        msg = f"{path.name}: the manifest {evidence.identity[:12]} is not the pointer's {pointer.identity[:12]}"
        raise ValueError(msg)
    return ModelAccount(
        label=entry_label,
        configuration=configuration,
        arm=arm,
        present=True,
        identity=evidence.identity,
        fit_identity=None if evidence.fit is None else evidence.fit.identity,
        assignment=evidence.assignment,
        status=evidence.status,
        n_pairs=evidence.n_pairs,
        n_completed=evidence.n_completed,
        n_infeasible=evidence.n_infeasible,
        n_unexecuted=evidence.n_unexecuted,
        execution_identity=evidence.conditions.execution_identity,
        payload=pointer.payload,
    )


def _bank_lines(evidence_dir: Path, store: StorageRoot) -> tuple[BankAccount, ...]:
    """Every replay bank the evidence directory points at, read from its own manifest."""
    lines: list[BankAccount] = []
    for path in sorted(evidence_dir.glob("replay__*.toml")):
        pointer = load_manual_pointer(path)
        bank = load_manual_replay_bank(verify_artifact(store, pointer.payload))
        completed = sum(1 for pair in bank.pairs if pair.status == "completed")
        lines.append(
            BankAccount(
                identity=bank.identity,
                assignment=bank.assignment,
                warmup_s=bank.conditions.warmup_s,
                velocity_cutoff_hz=bank.conditions.replay_velocity_cutoff_hz,
                acceleration_cutoff_hz=bank.conditions.replay_acceleration_cutoff_hz,
                n_pairs=len(bank.pairs),
                n_completed=completed,
                n_infeasible=len(bank.pairs) - completed,
                execution_identity=bank.conditions.execution_identity,
                payload=pointer.payload,
            )
        )
    return tuple(lines)


def account_study(
    *, store: StorageRoot, evidence_dir: Path, manifest: StudyManifest, provenance: ProvenanceRecord
) -> StudyAccounting:
    """Account for every model of the frozen study and every replay bank pointed at beside them."""
    models = tuple(
        _model_line(entry.label, entry.configuration, entry.arm.label, evidence_dir, store)
        for entry in manifest.entries
    )
    banks = _bank_lines(evidence_dir, store)
    present = [line for line in models if line.present]
    canonical = manifest.execution.identity
    keyed = [line.execution_identity for line in present] + [bank.execution_identity for bank in banks]
    # A present line always carries a status, which __post_init__ enforces, so there is no
    # absent case to guard here: a guard that cannot fire only pretends to be a check.
    statuses: dict[str, int] = {}
    for line in present:
        statuses[str(line.status)] = statuses.get(str(line.status), 0) + 1
    return StudyAccounting(
        experiment=EXPERIMENT_LABEL,
        canonical_execution_identity=canonical,
        models=models,
        banks=banks,
        n_models=len(models),
        n_present=len(present),
        n_missing=len(models) - len(present),
        missing=tuple(line.label for line in models if not line.present),
        statuses=statuses,
        n_rc_runs=sum(line.n_pairs for line in present),
        n_replay_runs=sum(bank.n_pairs for bank in banks),
        all_bind_canonical_execution=all(identity == canonical for identity in keyed),
        complete=not models or len(present) == len(models),
        provenance=provenance,
    )
