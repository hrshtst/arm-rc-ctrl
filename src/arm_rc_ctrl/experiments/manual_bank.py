# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Batch validation of saved manual takes and the versioned bank manifest (M3MAN-002).

Recording and acceptance are separate: the recorder saves every attempt, and
this tool validates a batch of saved takes offline against the frozen rules of
the manual protocol. Every attempt is imported and retained with its verdict
and reasons, accepted takes derive their full-recording datasets, and the
first ``required`` accepted takes in acquisition order become the bank
(``D01`` .. ``D10``). A shortfall asks for another batch; there is no attempt
cap. Batches are versioned: a batch number is validated once and never edited,
and a batch's Git-tracked records are registered only after all of its takes
were processed, so a confirmatory run never trips over its own outputs.
Everything a batch publishes (reports, records, manifest) is first staged as a
journal in the external store; if publication fails part-way, rerunning the
batch completes that journal without validating again, and no other batch
starts while one is pending. Once
the bank is complete its assignments are locked: later batches may only add
later attempts, which stay retained but unassigned. Practice payloads never
enter a manifest.

Usage (one line)::

    uv run python -m arm_rc_ctrl.experiments.manual_bank --scenario configs/tasks/task_1a_manual_v1.toml
    --config configs/preprocessing/manual_v1.toml --session <session> --batch 1
    --manifest docs/experiments/task_1a_manual_demonstration/bank/bank_v1.json --takes reach_001.sklog.npz ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from arm_rc_ctrl.config import ConfigError, from_mapping, to_mapping
from arm_rc_ctrl.data.manual import (
    DuplicatePayloadError,
    ManualDatasetRecord,
    ManualTakeError,
    ManualTakeRecord,
    TakeAssessment,
    assess_take,
    derive_manual_take,
    import_manual_take,
    load_manual_derive_config,
    load_manual_take,
    register_manual_records,
)
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.data.records import record_path, to_toml
from arm_rc_ctrl.provenance import DirtyWorktreeError, sha256_file
from arm_rc_ctrl.repo import git_output, repository_root
from arm_rc_ctrl.storage import ArtifactUri, StorageError, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.data.records import AccessClass
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "BANK_SCHEMA_VERSION",
    "DEFAULT_REQUIRED",
    "BankManifest",
    "BatchReport",
    "JournalFile",
    "JournalRecord",
    "PublicationJournal",
    "TakeMeasurements",
    "TakeVerdict",
    "attempt_number",
    "load_bank_manifest",
    "main",
    "validate_batch",
    "write_bank_manifest",
]

BANK_SCHEMA_VERSION = 1
JOURNAL_SCHEMA_VERSION = 1
DEFAULT_REQUIRED = 10
_ATTEMPT_RE = re.compile(r"^.+?_(?P<number>\d+)(?:\.[A-Za-z0-9]+)+$")


def attempt_number(path: Path) -> int:
    """The attempt number of a numbered recorder output such as ``reach_007.sklog.npz``.

    Raises
    ------
    ValueError
        If the name is not ``<base>_<number><suffixes>``; attempts are never guessed.
    """
    match = _ATTEMPT_RE.match(path.name)
    if match is None:
        msg = f"{path.name} is not a numbered recorder output (<base>_<number>.sklog.npz)"
        raise ValueError(msg)
    return int(match.group("number"))


@dataclass(frozen=True)
class TakeMeasurements:
    """Summary figures of one assessed take (``None`` where the assessment did not get that far)."""

    n_frames: int | None = None
    duration_s: float | None = None
    median_interval_s: float | None = None
    max_interval_s: float | None = None
    late_frames: int | None = None
    max_raw_speed_rad_s: float | None = None
    start_deviation_rad: float | None = None
    start_shift_rad: float | None = None
    dwell_final_duration_s: float | None = None
    dwell_final_samples: int | None = None
    hold_end_s: float | None = None
    movement_duration_s: float | None = None
    time_to_dwell_s: float | None = None
    path_length_rad: float | None = None
    peak_speed_rad_s: float | None = None
    final_endpoint_error_m: float | None = None


@dataclass(frozen=True)
class TakeVerdict:
    """One attempt's outcome: retained identifiers, the verdict, its reasons, and summary measurements."""

    batch: int
    attempt: int
    source_file: str
    accepted: bool
    reasons: tuple[str, ...]
    raw_artifact_id: str | None
    processed_artifact_id: str | None
    payload_sha256: str | None
    """Digest of the submitted file, used to detect byte-identical copies."""
    q_sha256: str | None
    """Digest of the raw joint trajectory, used to detect duplicate recordings."""
    duplicate_of: str | None
    """Raw artifact of the earlier take this one duplicates, if any; a copy never gets a record of its own."""
    assignment: str | None
    """``D01`` .. ``D<required>`` once the bank is complete, else ``None``."""
    measurements: TakeMeasurements

    def __post_init__(self) -> None:
        """Validate the verdict's internal consistency."""
        if self.batch < 1 or self.attempt < 1:
            msg = f"batch and attempt must be positive, got {self.batch} and {self.attempt}"
            raise ValueError(msg)
        if self.accepted and (self.raw_artifact_id is None or self.processed_artifact_id is None or self.reasons):
            msg = f"attempt {self.attempt} is accepted but lacks its artifacts or carries reasons"
            raise ValueError(msg)
        if not self.accepted and (self.processed_artifact_id is not None or not self.reasons):
            msg = f"attempt {self.attempt} is rejected but has a processed artifact or no reason"
            raise ValueError(msg)
        if self.assignment is not None and not self.accepted:
            msg = f"attempt {self.attempt} is rejected and cannot carry the assignment {self.assignment!r}"
            raise ValueError(msg)
        if self.duplicate_of is not None and self.accepted:
            msg = f"attempt {self.attempt} duplicates {self.duplicate_of} and cannot be accepted"
            raise ValueError(msg)


@dataclass(frozen=True)
class BankManifest:
    """Every validated attempt of a session in (batch, attempt) order and the bank derived from them."""

    bank_schema_version: int
    protocol: str
    session: str
    scenario_path: str
    scenario_sha256: str
    derive_config_path: str
    derive_config_sha256: str
    required: int
    takes: tuple[TakeVerdict, ...]
    updated_at: str

    def __post_init__(self) -> None:
        """Validate the schema version, the ordering of the takes, and the assignments."""
        if self.bank_schema_version != BANK_SCHEMA_VERSION:
            msg = f"unsupported bank_schema_version {self.bank_schema_version}; expected {BANK_SCHEMA_VERSION}"
            raise ValueError(msg)
        if self.required < 1:
            msg = f"required must be positive, got {self.required}"
            raise ValueError(msg)
        attempts = [t.attempt for t in self.takes]
        if attempts != sorted(set(attempts)):
            msg = "takes must be unique and ordered by attempt number (acquisition order)"
            raise ValueError(msg)
        expected = _expected_assignments(self.takes, self.required)
        for take in self.takes:
            if take.assignment != expected.get(take.attempt):
                msg = (
                    f"attempt {take.attempt} carries assignment {take.assignment!r} but the first {self.required} "
                    f"accepted takes in acquisition order give {expected.get(take.attempt)!r}"
                )
                raise ValueError(msg)

    @property
    def accepted_attempts(self) -> tuple[int, ...]:
        """Attempt numbers of the accepted takes in acquisition order."""
        return tuple(t.attempt for t in self.takes if t.accepted)

    @property
    def batches(self) -> tuple[int, ...]:
        """Validated batch numbers."""
        return tuple(sorted({t.batch for t in self.takes}))

    @property
    def complete(self) -> bool:
        """Whether at least ``required`` takes are accepted."""
        return len(self.accepted_attempts) >= self.required

    @property
    def shortfall(self) -> int:
        """How many more accepted takes are needed."""
        return max(0, self.required - len(self.accepted_attempts))

    @property
    def assignments(self) -> dict[str, str]:
        """``D01`` .. ``D<required>`` mapped to raw artifact IDs, once the bank is complete."""
        return {t.assignment: cast("str", t.raw_artifact_id) for t in self.takes if t.assignment is not None}


def _expected_assignments(takes: tuple[TakeVerdict, ...], required: int) -> dict[int, str]:
    """``attempt -> D..`` for the first ``required`` accepted takes in acquisition order, once they exist."""
    accepted = sorted((t for t in takes if t.accepted), key=lambda t: t.attempt)
    if len(accepted) < required:
        return {}
    return {t.attempt: f"D{i + 1:02d}" for i, t in enumerate(accepted[:required])}


def _assign(takes: tuple[TakeVerdict, ...], required: int) -> tuple[TakeVerdict, ...]:
    """Order the takes by attempt and assign ``D01`` .. to the first ``required`` accepted ones."""
    ordered = tuple(sorted(takes, key=lambda t: t.attempt))
    labels = _expected_assignments(ordered, required)
    return tuple(replace(t, assignment=labels.get(t.attempt) if t.accepted else None) for t in ordered)


def _manifest_text(manifest: BankManifest) -> str:
    return json.dumps(to_mapping(manifest), indent=2, sort_keys=True) + "\n"


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_bank_manifest(path: Path, manifest: BankManifest) -> None:
    """Write the manifest atomically as portable JSON (no machine paths)."""
    _write_text_atomic(path, _manifest_text(manifest))


def load_bank_manifest(path: Path) -> BankManifest:
    """Read a manifest strictly: wrong types, unknown keys, and inconsistent assignments are refused.

    Raises
    ------
    ConfigError
        If the document does not satisfy the manifest schema (a ``ValueError``).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(str(path), f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(str(path), "the manifest must be a JSON object")
    return from_mapping(cast("dict[str, Any]", data), BankManifest)


@dataclass(frozen=True)
class BatchReport:
    """Outcome of one batch: the verdicts, the bank's progress, and what to do next."""

    batch: int
    verdicts: tuple[TakeVerdict, ...]
    accepted_total: int
    required: int
    shortfall: int
    complete: bool
    next_action: str
    manifest_file: Path
    report_json: Path
    report_markdown: Path


def _relative(path: Path, records_root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(records_root.resolve()).as_posix()
    except ValueError as exc:
        msg = f"{label} {path} must live inside the repository {records_root}"
        raise ValueError(msg) from exc


def _measurements(assessment: TakeAssessment) -> TakeMeasurements:
    timing = assessment.raw_timing
    motion = assessment.motion
    return TakeMeasurements(
        n_frames=timing.n_frames,
        duration_s=timing.duration_s,
        median_interval_s=timing.median_interval_s,
        max_interval_s=timing.max_interval_s,
        late_frames=timing.late_frames,
        max_raw_speed_rad_s=max(timing.max_raw_speed_rad_s),
        start_deviation_rad=assessment.start.max_deviation_rad,
        start_shift_rad=None if assessment.smoothing is None else assessment.smoothing.start_shift_rad,
        dwell_final_duration_s=None if assessment.dwell is None else assessment.dwell.final_duration_s,
        dwell_final_samples=None if assessment.dwell is None else assessment.dwell.final_samples,
        hold_end_s=None if motion is None else motion.hold_end_s,
        movement_duration_s=None if motion is None else motion.movement_duration_s,
        time_to_dwell_s=None if motion is None else motion.time_to_dwell_s,
        path_length_rad=None if motion is None else motion.path_length_rad,
        peak_speed_rad_s=None if motion is None else motion.peak_speed_rad_s,
        final_endpoint_error_m=None if motion is None else motion.final_endpoint_error_m,
    )


@dataclass(frozen=True)
class _Context:
    scenario_file: Path
    config_file: Path
    store: StorageRoot
    records_root: Path
    session: str
    batch: int
    license_label: str
    access: AccessClass
    exploratory: bool
    now: datetime | None


type _Records = list[ManualTakeRecord | ManualDatasetRecord]


def _rejected(context: _Context, attempt: int, path: Path, reasons: tuple[str, ...], **fields: object) -> TakeVerdict:
    values: dict[str, Any] = {
        "batch": context.batch,
        "attempt": attempt,
        "source_file": path.name,
        "accepted": False,
        "reasons": reasons,
        "raw_artifact_id": None,
        "processed_artifact_id": None,
        "payload_sha256": None,
        "q_sha256": None,
        "duplicate_of": None,
        "assignment": None,
        "measurements": TakeMeasurements(),
    }
    values.update(fields)
    return TakeVerdict(**values)


def _validate_take(
    path: Path,
    attempt: int,
    seen_payloads: dict[str, TakeVerdict],
    seen_trajectories: dict[str, TakeVerdict],
    context: _Context,
) -> tuple[TakeVerdict, _Records]:
    """Import, measure, and (when accepted) derive one attempt without registering records; never raise for a bad take.

    A byte-identical copy of an earlier take is rejected before import and gets
    no record of its own (``duplicate_of`` names the earlier take's artifact);
    an unreadable or structurally invalid file is a ``malformed`` rejection.
    """
    payload_digest = sha256_file(path)
    twin = seen_payloads.get(payload_digest)
    if twin is not None:
        reason = (
            f"duplicate of attempt {twin.attempt} (batch {twin.batch}): byte-identical file, retained as "
            f"{twin.raw_artifact_id or 'that attempt'}"
        )
        return _rejected(
            context, attempt, path, (reason,), payload_sha256=payload_digest, duplicate_of=twin.raw_artifact_id
        ), []
    config = load_manual_scenario(context.scenario_file)
    derive = load_manual_derive_config(context.config_file)
    try:
        imported = import_manual_take(
            path,
            context.scenario_file,
            store=context.store,
            records_root=context.records_root,
            session=context.session,
            take=attempt,
            license_label=context.license_label,
            access=context.access,
            exploratory=context.exploratory,
            now=context.now,
            register=False,
        )
    except DuplicatePayloadError as exc:
        reason = (
            f"duplicate of {exc.existing.artifact.artifact_id} (attempt {exc.existing.take} of session "
            f"{exc.existing.session!r}): byte-identical file"
        )
        return _rejected(
            context,
            attempt,
            path,
            (reason,),
            payload_sha256=payload_digest,
            duplicate_of=exc.existing.artifact.artifact_id,
        ), []
    except ManualTakeError as exc:
        return _rejected(context, attempt, path, (f"malformed: {exc}",), payload_sha256=payload_digest), []
    records: _Records = [imported.record]
    take = load_manual_take(context.store, imported.record)
    q_digest = hashlib.sha256(np.ascontiguousarray(take.q).tobytes()).hexdigest()
    assessment = assess_take(take, config, derive)
    reasons = list(assessment.problems)
    twin = seen_trajectories.get(q_digest)
    duplicate_of: str | None = None
    if twin is not None:
        reasons.append(f"duplicate of attempt {twin.attempt} (batch {twin.batch}): identical joint trajectory")
        duplicate_of = twin.raw_artifact_id
    processed_id: str | None = None
    if not reasons:
        derived = derive_manual_take(
            imported.record,
            context.scenario_file,
            context.config_file,
            store=context.store,
            records_root=context.records_root,
            exploratory=context.exploratory,
            now=context.now,
            register=False,
        )
        processed_id = derived.record.artifact.artifact_id
        records.append(derived.record)
    verdict = TakeVerdict(
        batch=context.batch,
        attempt=attempt,
        source_file=path.name,
        accepted=not reasons,
        reasons=tuple(reasons),
        raw_artifact_id=imported.record.artifact.artifact_id,
        processed_artifact_id=processed_id,
        payload_sha256=payload_digest,
        q_sha256=q_digest,
        duplicate_of=duplicate_of,
        assignment=None,
        measurements=_measurements(assessment),
    )
    return verdict, records


def _load_or_create_manifest(manifest_file: Path, context: _Context, required: int) -> BankManifest:
    config = load_manual_scenario(context.scenario_file)
    fresh = BankManifest(
        bank_schema_version=BANK_SCHEMA_VERSION,
        protocol=config.protocol,
        session=context.session,
        scenario_path=_relative(context.scenario_file, context.records_root, "scenario file"),
        scenario_sha256=sha256_file(context.scenario_file),
        derive_config_path=_relative(context.config_file, context.records_root, "derivation config"),
        derive_config_sha256=sha256_file(context.config_file),
        required=required,
        takes=(),
        updated_at="",
    )
    if not manifest_file.exists():
        return fresh
    existing = load_bank_manifest(manifest_file)
    for field in ("protocol", "session", "scenario_sha256", "derive_config_sha256", "required"):
        if getattr(existing, field) != getattr(fresh, field):
            msg = (
                f"{manifest_file} was created for another {field.replace('_', ' ')} "
                f"({getattr(existing, field)!r}); a bank binds one scenario, derivation config, session, and size"
            )
            raise ValueError(msg)
    if context.batch in existing.batches:
        msg = f"batch {context.batch} is already recorded in {manifest_file}; batches are versioned, never edited"
        raise ValueError(msg)
    return existing


def _next_action(manifest: BankManifest) -> str:
    if manifest.complete:
        return (
            f"bank complete: the first {manifest.required} accepted takes are assigned "
            f"D01..D{manifest.required:02d}; later accepted takes stay retained but unassigned"
        )
    plural = "s" if manifest.shortfall != 1 else ""
    return f"collect at least {manifest.shortfall} more take{plural} in a new batch (no attempt cap)"


def _render_markdown(report: BatchReport, manifest: BankManifest) -> str:
    lines = [
        f"# Manual take batch {report.batch:03d} of session `{manifest.session}`",
        "",
        f"- protocol: `{manifest.protocol}`",
        f"- scenario: `{manifest.scenario_path}` (sha256 `{manifest.scenario_sha256[:12]}`)",
        f"- derivation: `{manifest.derive_config_path}` (sha256 `{manifest.derive_config_sha256[:12]}`)",
        f"- accepted in this batch: {sum(v.accepted for v in report.verdicts)} of {len(report.verdicts)}",
        f"- accepted in total: {report.accepted_total} of the required {report.required}",
        f"- next: {report.next_action}",
        "",
        "| attempt | file | verdict | raw artifact | processed artifact | reasons |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for v in report.verdicts:
        verdict = "accepted" if v.accepted else "rejected"
        raw = v.raw_artifact_id or (f"duplicate of {v.duplicate_of}" if v.duplicate_of else "-")
        lines.append(
            f"| {v.attempt} | `{v.source_file}` | {verdict} | {raw} | "
            f"{v.processed_artifact_id or '-'} | {'; '.join(v.reasons) or '-'} |"
        )
    if manifest.complete:
        lines += ["", "| demonstration | attempt | raw artifact |", "| --- | --- | --- |"]
        by_raw = {t.raw_artifact_id: t for t in manifest.takes}
        lines += [f"| {label} | {by_raw[raw].attempt} | {raw} |" for label, raw in manifest.assignments.items()]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class JournalFile:
    """A report file a batch publishes into the checkout (path relative to the records root)."""

    path: str
    text: str


@dataclass(frozen=True)
class JournalRecord:
    """A Git-tracked record a batch registers, kept as its TOML text."""

    kind: Literal["raw", "processed"]
    toml: str


@dataclass(frozen=True)
class PublicationJournal:
    """Everything one batch publishes into the checkout, staged in the external store before any of it is written.

    A batch computes its verdicts, writes this journal, and then applies it. If
    applying fails part-way, rerunning the batch completes the same journal
    instead of validating again, so the records keep the provenance of the first
    run and the verdicts never change.
    """

    journal_schema_version: int
    session: str
    batch: int
    manifest_path: str
    base_manifest_sha256: str | None
    """Digest of the manifest before the batch (``None`` when the batch creates it)."""
    final_manifest_sha256: str
    attempts: tuple[int, ...]
    payload_sha256s: tuple[str, ...]
    """Digests of the submitted files, in attempt order."""
    files: tuple[JournalFile, ...]
    records: tuple[JournalRecord, ...]
    manifest_text: str

    def __post_init__(self) -> None:
        """Validate the schema version and the submission lists."""
        if self.journal_schema_version != JOURNAL_SCHEMA_VERSION:
            msg = f"unsupported journal_schema_version {self.journal_schema_version}; expected {JOURNAL_SCHEMA_VERSION}"
            raise ValueError(msg)
        if len(self.attempts) != len(self.payload_sha256s) or list(self.attempts) != sorted(set(self.attempts)):
            msg = "journal attempts must be unique, ordered, and paired with their payload digests"
            raise ValueError(msg)


def validate_batch(
    take_files: Sequence[Path],
    *,
    scenario_file: Path,
    config_file: Path,
    store: StorageRoot,
    records_root: Path,
    session: str,
    batch: int,
    manifest_file: Path,
    required: int = DEFAULT_REQUIRED,
    license_label: str,
    access: AccessClass,
    exploratory: bool,
    now: datetime | None = None,
) -> BatchReport:
    """Validate one batch of saved takes, update the bank manifest, and write the batch report.

    A batch whose publication failed part-way is completed from its journal when
    it is rerun with the same takes; without ``exploratory`` the worktree may then
    hold only that journal's own outputs.

    Raises
    ------
    ValueError
        If a file is not a numbered recorder output, an attempt repeats, the
        batch number was recorded before, another batch is pending publication,
        or the manifest binds another scenario, derivation config, session, or
        bank size.
    DirtyWorktreeError
        If a confirmatory run finds changes unrelated to its pending batch.
    """
    if batch < 1 or required < 1:
        msg = f"batch and required must be positive, got {batch} and {required}"
        raise ValueError(msg)
    context = _Context(
        scenario_file, config_file, store, records_root, session, batch, license_label, access, exploratory, now
    )
    manifest_path = _relative(manifest_file, records_root, "bank manifest")
    key = hashlib.sha256(f"{manifest_path}\n{session}".encode()).hexdigest()[:16]
    for journal_file in _pending_journals(store, key):
        journal = _load_journal(journal_file)
        if _file_sha256(manifest_file) == journal.final_manifest_sha256:
            journal_file.unlink()  # published completely earlier; only the journal's removal was missed
            continue
        if journal.batch != batch:
            msg = (
                f"batch {journal.batch} is pending publication (it failed part-way); rerun batch {journal.batch} "
                f"with the same takes before validating batch {batch}"
            )
            raise ValueError(msg)
        _check_resume(journal, take_files, context)
        if not exploratory:
            _require_only_journal_changes(records_root, journal)
        return _apply_journal(journal, journal_file, records_root=records_root, manifest_file=manifest_file)
    manifest = _load_or_create_manifest(manifest_file, context, required)
    numbered = _check_submission(take_files, manifest)
    seen_payloads, seen_trajectories = _seen_digests(manifest.takes)
    verdicts: list[TakeVerdict] = []
    pending: _Records = []
    for attempt, path in numbered:
        verdict, records = _validate_take(path, attempt, seen_payloads, seen_trajectories, context)
        verdicts.append(verdict)
        pending.extend(records)
        _remember(verdict, seen_payloads, seen_trajectories)
    stamp = (now or datetime.now(tz=UTC)).replace(microsecond=0).isoformat()
    base_sha256 = _file_sha256(manifest_file)
    manifest = replace(manifest, takes=_assign((*manifest.takes, *verdicts), required), updated_at=stamp)
    journal = _build_journal(manifest, verdicts, pending, context, manifest_file, base_sha256, stamp)
    journal_file = store.path(
        ArtifactUri("reports", ("manual_bank", "pending", f"{key}-batch-{batch:03d}.json")), mode="write"
    )
    _write_text_atomic(journal_file, json.dumps(to_mapping(journal), indent=2, sort_keys=True) + "\n")
    return _apply_journal(journal, journal_file, records_root=records_root, manifest_file=manifest_file)


def _file_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _pending_journals(store: StorageRoot, key: str) -> list[Path]:
    directory = store.root / "reports" / "manual_bank" / "pending"
    return sorted(directory.glob(f"{key}-batch-*.json")) if directory.is_dir() else []


def _load_journal(path: Path) -> PublicationJournal:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(str(path), f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(str(path), "the journal must be a JSON object")
    return from_mapping(cast("dict[str, Any]", data), PublicationJournal)


def _record_from_journal(item: JournalRecord) -> ManualTakeRecord | ManualDatasetRecord:
    data = tomllib.loads(item.toml)
    if item.kind == "raw":
        return from_mapping(data, ManualTakeRecord)
    return from_mapping(data, ManualDatasetRecord)


def _batch_report(manifest: BankManifest, batch: int, manifest_file: Path) -> BatchReport:
    stem = manifest_file.with_suffix("").name
    return BatchReport(
        batch=batch,
        verdicts=tuple(t for t in manifest.takes if t.batch == batch),
        accepted_total=len(manifest.accepted_attempts),
        required=manifest.required,
        shortfall=manifest.shortfall,
        complete=manifest.complete,
        next_action=_next_action(manifest),
        manifest_file=manifest_file,
        report_json=manifest_file.with_name(f"{stem}_batch_{batch:03d}.json"),
        report_markdown=manifest_file.with_name(f"{stem}_batch_{batch:03d}.md"),
    )


def _build_journal(
    manifest: BankManifest,
    verdicts: list[TakeVerdict],
    pending: _Records,
    context: _Context,
    manifest_file: Path,
    base_sha256: str | None,
    stamp: str,
) -> PublicationJournal:
    report = _batch_report(manifest, context.batch, manifest_file)
    payload = {
        "batch": report.batch,
        "session": context.session,
        "verdicts": [to_mapping(v) for v in report.verdicts],
        "accepted_total": report.accepted_total,
        "required": report.required,
        "shortfall": report.shortfall,
        "complete": report.complete,
        "next_action": report.next_action,
        "updated_at": stamp,
    }
    manifest_text = _manifest_text(manifest)
    root = context.records_root
    return PublicationJournal(
        journal_schema_version=JOURNAL_SCHEMA_VERSION,
        session=context.session,
        batch=context.batch,
        manifest_path=_relative(manifest_file, root, "bank manifest"),
        base_manifest_sha256=base_sha256,
        final_manifest_sha256=hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        attempts=tuple(v.attempt for v in verdicts),
        payload_sha256s=tuple(cast("str", v.payload_sha256) for v in verdicts),
        files=(
            JournalFile(
                _relative(report.report_json, root, "batch report"),
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            ),
            JournalFile(_relative(report.report_markdown, root, "batch report"), _render_markdown(report, manifest)),
        ),
        records=tuple(
            JournalRecord("raw" if isinstance(r, ManualTakeRecord) else "processed", to_toml(r)) for r in pending
        ),
        manifest_text=manifest_text,
    )


def _check_resume(journal: PublicationJournal, take_files: Sequence[Path], context: _Context) -> None:
    """A retry must submit the same files under the same scenario and derivation config as the pending batch."""
    staged = from_mapping(cast("dict[str, Any]", json.loads(journal.manifest_text)), BankManifest)
    if (staged.scenario_sha256, staged.derive_config_sha256) != (
        sha256_file(context.scenario_file),
        sha256_file(context.config_file),
    ):
        msg = f"batch {journal.batch} is pending under another scenario or derivation config; rerun it unchanged"
        raise ValueError(msg)
    missing = [str(path) for path in take_files if not path.is_file()]
    if missing:
        msg = f"take file not found: {', '.join(missing)}"
        raise ValueError(msg)
    submitted = sorted((attempt_number(path), sha256_file(path)) for path in take_files)
    if submitted != list(zip(journal.attempts, journal.payload_sha256s, strict=True)):
        msg = (
            f"batch {journal.batch} is pending publication with attempts {list(journal.attempts)}; "
            "rerun it with the same takes"
        )
        raise ValueError(msg)


def _require_only_journal_changes(records_root: Path, journal: PublicationJournal) -> None:
    """Refuse a confirmatory completion when the worktree holds anything but the pending batch's own outputs."""
    allowed = {item.path for item in journal.files}
    allowed |= {journal.manifest_path, f"{journal.manifest_path}.tmp", "data/catalog.toml"}
    allowed |= {
        record_path(records_root, _record_from_journal(item).artifact).relative_to(records_root).as_posix()
        for item in journal.records
    }
    status = git_output("status", "--porcelain", "--untracked-files=all", cwd=records_root)
    dirty = {line[3:] if len(line) > 2 and line[2] == " " else line[2:] for line in status.splitlines() if line.strip()}  # noqa: PLR2004
    unrelated = sorted(dirty - allowed)
    if unrelated:
        msg = (
            f"the worktree has changes unrelated to pending batch {journal.batch}: {', '.join(unrelated[:5])}; "
            "commit or stash them, or rerun with --exploratory"
        )
        raise DirtyWorktreeError(msg)


def _apply_journal(
    journal: PublicationJournal, journal_file: Path, *, records_root: Path, manifest_file: Path
) -> BatchReport:
    """Write the reports, register the records, write the manifest, and drop the journal; safe to repeat."""
    if _file_sha256(manifest_file) not in (journal.base_manifest_sha256, journal.final_manifest_sha256):
        msg = f"{manifest_file} changed since batch {journal.batch} was staged; restore it before completing the batch"
        raise ValueError(msg)
    for item in journal.files:
        target = records_root / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item.text, encoding="utf-8")
    register_manual_records(records_root, [_record_from_journal(item) for item in journal.records])
    _write_text_atomic(manifest_file, journal.manifest_text)
    journal_file.unlink()
    manifest = from_mapping(cast("dict[str, Any]", json.loads(journal.manifest_text)), BankManifest)
    return _batch_report(manifest, journal.batch, manifest_file)


def _check_submission(take_files: Sequence[Path], manifest: BankManifest) -> list[tuple[int, Path]]:
    """Existing numbered files, unique attempts, and no attempt before the locked assignments of a complete bank."""
    missing = [str(path) for path in take_files if not path.is_file()]
    if missing:
        msg = f"take file not found: {', '.join(missing)}"
        raise ValueError(msg)
    numbered = sorted(((attempt_number(path), path) for path in take_files), key=lambda item: item[0])
    attempts = [attempt for attempt, _ in numbered]
    known = {t.attempt for t in manifest.takes}
    repeated = [a for a in attempts if attempts.count(a) > 1 or a in known]
    if repeated:
        msg = f"attempt numbers must be unique across the session; repeated: {sorted(set(repeated))}"
        raise ValueError(msg)
    if manifest.complete:
        highest = max(t.attempt for t in manifest.takes if t.assignment is not None)
        early = [a for a in attempts if a <= highest]
        if early:
            msg = (
                f"the bank is complete and its assignments are locked: attempts {early} precede the last assigned "
                f"attempt {highest}; later batches may only add later recordings"
            )
            raise ValueError(msg)
    return numbered


def _seen_digests(takes: tuple[TakeVerdict, ...]) -> tuple[dict[str, TakeVerdict], dict[str, TakeVerdict]]:
    """The earliest retained take per payload and per joint trajectory: the original every later copy points at."""
    payloads: dict[str, TakeVerdict] = {}
    trajectories: dict[str, TakeVerdict] = {}
    for earlier in takes:
        _remember(earlier, payloads, trajectories)
    return payloads, trajectories


def _remember(verdict: TakeVerdict, payloads: dict[str, TakeVerdict], trajectories: dict[str, TakeVerdict]) -> None:
    if verdict.raw_artifact_id is None:
        return  # a rejected copy or malformed file is never an original
    if verdict.payload_sha256 is not None:
        payloads.setdefault(verdict.payload_sha256, verdict)
    if verdict.q_sha256 is not None:
        trajectories.setdefault(verdict.q_sha256, verdict)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the batch validation."""
    parser = argparse.ArgumentParser(description="Validate a batch of saved manual takes and update the bank manifest.")
    parser.add_argument("--scenario", type=Path, required=True, help="manual-protocol task TOML")
    parser.add_argument("--config", type=Path, required=True, help="manual derivation TOML")
    parser.add_argument("--session", required=True, help="recording session ID (lowercase, digits, hyphens)")
    parser.add_argument("--batch", type=int, required=True, help="batch number; validated once, never edited")
    parser.add_argument("--manifest", type=Path, required=True, help="bank manifest JSON (created on first use)")
    parser.add_argument("--takes", type=Path, nargs="+", required=True, help="numbered recorder outputs")
    parser.add_argument("--required", type=int, default=DEFAULT_REQUIRED, help="accepted takes the bank needs")
    parser.add_argument("--license", default="proprietary", help="license label recorded on every artifact")
    parser.add_argument("--access", choices=("private", "internal", "public"), default="private")
    parser.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the batch through the configured store; exit 0 when the bank is complete, 2 while it is not."""
    args = build_parser().parse_args(argv)
    try:
        store = open_storage()
        report = validate_batch(
            list(args.takes),
            scenario_file=args.scenario,
            config_file=args.config,
            store=store,
            records_root=repository_root(),
            session=args.session,
            batch=args.batch,
            manifest_file=args.manifest,
            required=args.required,
            license_label=args.license,
            access=cast("AccessClass", args.access),
            exploratory=args.exploratory,
        )
    except (StorageError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    accepted_here = sum(v.accepted for v in report.verdicts)
    print(
        f"batch {report.batch:03d}: accepted {accepted_here} of {len(report.verdicts)} takes; "
        f"{report.accepted_total} of {report.required} accepted in total"
    )
    for verdict in report.verdicts:
        status = "accepted" if verdict.accepted else "rejected: " + "; ".join(verdict.reasons)
        print(f"  attempt {verdict.attempt:03d} {verdict.source_file}: {status}")
    print(f"next: {report.next_action}")
    print(f"manifest: {report.manifest_file}; report: {report.report_markdown}")
    return 0 if report.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
