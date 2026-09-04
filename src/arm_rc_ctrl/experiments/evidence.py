# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""External study-report payloads with Git pointer records (docs/PLAN.md section 11; M3R-016 finding 1).

Full per-trial study reports are large generated payloads, not curated small
reports, so they live in the external store under ``armrc://reports/`` and Git
commits only a content-addressed pointer: the payload's URI, SHA-256, and
size, plus the compact identity figures Git-only consumers need (study,
formulation, protocol digest, dataset, budget, trial and feasibility counts,
and the selection). Opening a stored report verifies the payload digest and
then cross-checks every duplicated pointer field against the loaded report, so
neither side can drift. Removing a pointer never deletes a payload.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.data.records import load_record, write_record
from arm_rc_ctrl.experiments.recovery_study import RecoveryStudyReport, load_report
from arm_rc_ctrl.provenance import ArtifactReference, verify_artifact

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "REPORT_POINTER_SCHEMA",
    "StoredReport",
    "load_report_pointer",
    "open_stored_report",
    "report_pointer",
    "store_report_payload",
    "write_report_pointer",
]

REPORT_POINTER_SCHEMA: Final = "recovery-study-report"
_REPORTS_PREFIX: Final = "armrc://reports/task_1a_state_conditioned_recovery"


@dataclass(frozen=True)
class StoredReport:
    """Git pointer to one externally stored study report."""

    schema: str
    study: str
    formulation: str
    protocol_sha256: str
    dataset: str
    budget: int
    trials_stored: int
    n_feasible: int
    payload: ArtifactReference
    best_number: int | None = None
    best_value: float | None = None

    def __post_init__(self) -> None:
        """The pointer names its schema and keeps consistent counts."""
        if self.schema != REPORT_POINTER_SCHEMA:
            msg = f"unsupported pointer schema {self.schema!r}; expected {REPORT_POINTER_SCHEMA!r}"
            raise ValueError(msg)
        if not self.study.strip() or not self.formulation.strip() or not self.dataset.strip():
            msg = "a report pointer needs its study, formulation, and dataset names"
            raise ValueError(msg)
        if self.budget < 1 or not 0 <= self.n_feasible <= self.trials_stored:
            msg = f"inconsistent counts in the pointer for {self.study!r}"
            raise ValueError(msg)
        if (self.best_number is None) != (self.best_value is None):
            msg = "best_number and best_value must be recorded together"
            raise ValueError(msg)


def _verified_existing(target: Path, digest: str, uri: str) -> bool:
    """Whether ``target`` already holds exactly the expected bytes (a mismatch is an error, absence False)."""
    if not target.exists():
        return False
    existing = hashlib.sha256(target.read_bytes()).hexdigest()
    if existing != digest:
        msg = f"{uri} exists with digest {existing[:12]}, expected {digest[:12]}; refusing to overwrite"
        raise ValueError(msg)
    return True


def store_report_payload(store: StorageRoot, json_text: str, *, name: str) -> ArtifactReference:
    """Install the report JSON content-addressed; identical payloads are reused and nothing is clobbered.

    Installation stages a uniquely named adjacent file and hard-links it into
    place: ``os.link`` fails atomically when the destination exists, so a file
    that races in between any check and the installation is never replaced —
    it is verified against the expected digest and reused only when identical
    (the digest is part of the name, so a mismatch means corruption, never a
    legitimate neighbour). The staging file is removed on every path.
    """
    data = json_text.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    uri = f"{_REPORTS_PREFIX}/{name}-{digest[:12]}.json"
    target = store.path(uri, mode="write")
    reference = ArtifactReference(uri, digest, len(data))
    if _verified_existing(target, digest, uri):
        return reference
    staged = target.with_name(f"{target.name}.staging-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        staged.write_bytes(data)
        try:
            os.link(staged, target)  # atomic no-clobber: a file that appeared meanwhile is never replaced
        except FileExistsError:
            if not _verified_existing(target, digest, uri):  # pragma: no cover - appeared, then vanished
                msg = f"{uri} appeared concurrently and cannot be verified"
                raise ValueError(msg) from None
    finally:
        staged.unlink(missing_ok=True)
    return reference


def report_pointer(report: RecoveryStudyReport, payload: ArtifactReference) -> StoredReport:
    """The pointer record of one report and its stored payload."""
    return StoredReport(
        schema=REPORT_POINTER_SCHEMA,
        study=report.protocol,
        formulation=report.formulation,
        protocol_sha256=report.protocol_sha256,
        dataset=report.dataset,
        budget=report.budget,
        trials_stored=len(report.summary.trials),
        n_feasible=report.n_feasible,
        best_number=report.summary.best_number,
        best_value=report.summary.best_value,
        payload=payload,
    )


def write_report_pointer(path: Path, pointer: StoredReport) -> None:
    """Commit-side write of the immutable pointer file."""
    write_record(path, pointer)


def load_report_pointer(path: Path) -> StoredReport:
    """Load and validate a pointer record."""
    return load_record(path, StoredReport)


def open_stored_report(store: StorageRoot, pointer: StoredReport) -> RecoveryStudyReport:
    """Verify the payload and load the full report, cross-checking every duplicated pointer field."""
    path = verify_artifact(store, pointer.payload)
    report = load_report(path)
    pairs = (
        ("study", pointer.study, report.protocol),
        ("formulation", pointer.formulation, report.formulation),
        ("protocol_sha256", pointer.protocol_sha256, report.protocol_sha256),
        ("dataset", pointer.dataset, report.dataset),
        ("budget", pointer.budget, report.budget),
        ("trials_stored", pointer.trials_stored, len(report.summary.trials)),
        ("n_feasible", pointer.n_feasible, report.n_feasible),
        ("best_number", pointer.best_number, report.summary.best_number),
        ("best_value", pointer.best_value, report.summary.best_value),
    )
    for name, recorded, actual in pairs:
        if recorded != actual:
            msg = f"pointer field {name} = {recorded!r} contradicts the stored report ({actual!r})"
            raise ValueError(msg)
    return report
