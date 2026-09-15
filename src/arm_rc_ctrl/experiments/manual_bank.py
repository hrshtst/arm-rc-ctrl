# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Batch validation of saved manual takes and the versioned bank manifest (M3MAN-002).

Recording and acceptance are separate: the recorder saves every attempt, and
this tool validates a batch of saved takes offline against the frozen rules of
the manual protocol. Every attempt is imported and retained with its verdict
and reasons, accepted takes derive their full-recording datasets, and the
first ``required`` accepted takes in acquisition order become the bank
(``D01`` .. ``D10``). A shortfall asks for another batch; there is no attempt
cap. Batches are versioned: a batch number is validated once and never edited.
Practice payloads never enter a manifest.

Usage (one line)::

    uv run python -m arm_rc_ctrl.experiments.manual_bank --scenario configs/tasks/task_1a_manual_v1.toml
    --config configs/preprocessing/manual_v1.toml --session <session> --batch 1
    --manifest docs/experiments/task_1a_manual_demonstration/bank/bank_v1.json --takes reach_001.sklog.npz ...
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from arm_rc_ctrl.data.manual import (
    ManualTakeError,
    TakeAssessment,
    assess_take,
    derive_manual_dataset,
    import_manual_take,
    load_manual_derive_config,
    load_manual_take,
)
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.data.records import AccessClass
    from arm_rc_ctrl.storage import StorageRoot

__all__ = [
    "BANK_SCHEMA_VERSION",
    "DEFAULT_REQUIRED",
    "BankManifest",
    "BatchReport",
    "TakeVerdict",
    "attempt_number",
    "load_bank_manifest",
    "main",
    "validate_batch",
    "write_bank_manifest",
]

BANK_SCHEMA_VERSION = 1
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
class TakeVerdict:
    """One attempt's outcome: retained identifiers, the verdict, its reasons, and summary measurements."""

    batch: int
    attempt: int
    source_file: str
    accepted: bool
    reasons: tuple[str, ...]
    raw_artifact_id: str | None
    processed_artifact_id: str | None
    q_sha256: str | None
    """Digest of the raw joint trajectory, used to detect duplicate takes."""
    assignment: str | None
    """``D01`` .. ``D<required>`` once the bank is complete, else ``None``."""
    measurements: dict[str, float | int | None]


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


def _assign(takes: tuple[TakeVerdict, ...], required: int) -> tuple[TakeVerdict, ...]:
    """Assign ``D01`` .. once ``required`` takes are accepted; later accepted takes stay unassigned."""
    accepted = [t for t in takes if t.accepted]
    labels: dict[int, str] = {}
    if len(accepted) >= required:
        labels = {t.attempt: f"D{i + 1:02d}" for i, t in enumerate(accepted[:required])}
    return tuple(replace(t, assignment=labels.get(t.attempt) if t.accepted else None) for t in takes)


def _manifest_dict(manifest: BankManifest) -> dict[str, Any]:
    return dataclasses.asdict(manifest)


def write_bank_manifest(path: Path, manifest: BankManifest) -> None:
    """Write the manifest atomically as portable JSON (no machine paths)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_manifest_dict(manifest), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_bank_manifest(path: Path) -> BankManifest:
    """Read a manifest written by :func:`write_bank_manifest`.

    Raises
    ------
    ValueError
        If the schema version is not supported.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("bank_schema_version") != BANK_SCHEMA_VERSION:
        msg = f"unsupported bank_schema_version {data.get('bank_schema_version')!r} in {path}"
        raise ValueError(msg)
    takes = tuple(
        TakeVerdict(
            batch=int(t["batch"]),
            attempt=int(t["attempt"]),
            source_file=str(t["source_file"]),
            accepted=bool(t["accepted"]),
            reasons=tuple(str(r) for r in t["reasons"]),
            raw_artifact_id=t["raw_artifact_id"],
            processed_artifact_id=t["processed_artifact_id"],
            q_sha256=t["q_sha256"],
            assignment=t["assignment"],
            measurements=dict(t["measurements"]),
        )
        for t in data["takes"]
    )
    return BankManifest(
        bank_schema_version=BANK_SCHEMA_VERSION,
        protocol=str(data["protocol"]),
        session=str(data["session"]),
        scenario_path=str(data["scenario_path"]),
        scenario_sha256=str(data["scenario_sha256"]),
        derive_config_path=str(data["derive_config_path"]),
        derive_config_sha256=str(data["derive_config_sha256"]),
        required=int(data["required"]),
        takes=takes,
        updated_at=str(data["updated_at"]),
    )


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


def _measurements(assessment: TakeAssessment | None) -> dict[str, float | int | None]:
    if assessment is None:
        return {}
    timing = assessment.raw_timing
    out: dict[str, float | int | None] = {
        "n_frames": timing.n_frames,
        "duration_s": timing.duration_s,
        "median_interval_s": timing.median_interval_s,
        "max_interval_s": timing.max_interval_s,
        "late_frames": timing.late_frames,
        "max_raw_speed_rad_s": max(timing.max_raw_speed_rad_s),
        "start_deviation_rad": assessment.start.max_deviation_rad,
        "start_shift_rad": None if assessment.smoothing is None else assessment.smoothing.start_shift_rad,
        "dwell_final_duration_s": None if assessment.dwell is None else assessment.dwell.final_duration_s,
        "dwell_final_samples": None if assessment.dwell is None else assessment.dwell.final_samples,
    }
    if assessment.motion is not None:
        out.update(
            {
                "hold_end_s": assessment.motion.hold_end_s,
                "movement_duration_s": assessment.motion.movement_duration_s,
                "time_to_dwell_s": assessment.motion.time_to_dwell_s,
                "path_length_rad": assessment.motion.path_length_rad,
                "peak_speed_rad_s": assessment.motion.peak_speed_rad_s,
                "final_endpoint_error_m": assessment.motion.final_endpoint_error_m,
            }
        )
    return out


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


def _validate_take(path: Path, attempt: int, seen: dict[str, TakeVerdict], context: _Context) -> TakeVerdict:
    """Import, measure, and (when accepted) derive one attempt; never raise for a bad take."""
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
        )
    except ManualTakeError as exc:
        return TakeVerdict(
            batch=context.batch,
            attempt=attempt,
            source_file=path.name,
            accepted=False,
            reasons=(f"malformed: {exc}",),
            raw_artifact_id=None,
            processed_artifact_id=None,
            q_sha256=None,
            assignment=None,
            measurements={},
        )
    take = load_manual_take(context.store, imported.record)
    digest = hashlib.sha256(np.ascontiguousarray(take.q).tobytes()).hexdigest()
    assessment = assess_take(take, config, derive)
    reasons = list(assessment.problems)
    twin = seen.get(digest)
    if twin is not None:
        reasons.append(f"duplicate of attempt {twin.attempt} (batch {twin.batch}): identical joint trajectory")
    accepted = not reasons
    processed_id: str | None = None
    if accepted:
        derived = derive_manual_dataset(
            imported.record_file,
            context.scenario_file,
            context.config_file,
            store=context.store,
            records_root=context.records_root,
            exploratory=context.exploratory,
            now=context.now,
        )
        processed_id = derived.record.artifact.artifact_id
    return TakeVerdict(
        batch=context.batch,
        attempt=attempt,
        source_file=path.name,
        accepted=accepted,
        reasons=tuple(reasons),
        raw_artifact_id=imported.record.artifact.artifact_id,
        processed_artifact_id=processed_id,
        q_sha256=digest,
        assignment=None,
        measurements=_measurements(assessment),
    )


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
        lines.append(
            f"| {v.attempt} | `{v.source_file}` | {verdict} | {v.raw_artifact_id or '-'} | "
            f"{v.processed_artifact_id or '-'} | {'; '.join(v.reasons) or '-'} |"
        )
    if manifest.complete:
        lines += ["", "| demonstration | attempt | raw artifact |", "| --- | --- | --- |"]
        by_raw = {t.raw_artifact_id: t for t in manifest.takes}
        lines += [f"| {label} | {by_raw[raw].attempt} | {raw} |" for label, raw in manifest.assignments.items()]
    return "\n".join(lines) + "\n"


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

    Raises
    ------
    ValueError
        If a file is not a numbered recorder output, an attempt repeats, the
        batch number was validated before, or the manifest binds another
        scenario, derivation config, session, or bank size.
    """
    if batch < 1 or required < 1:
        msg = f"batch and required must be positive, got {batch} and {required}"
        raise ValueError(msg)
    context = _Context(
        scenario_file, config_file, store, records_root, session, batch, license_label, access, exploratory, now
    )
    manifest = _load_or_create_manifest(manifest_file, context, required)
    numbered = sorted(((attempt_number(path), path) for path in take_files), key=lambda item: item[0])
    attempts = [attempt for attempt, _ in numbered]
    known = {t.attempt for t in manifest.takes}
    repeated = [a for a in attempts if attempts.count(a) > 1 or a in known]
    if repeated:
        msg = f"attempt numbers must be unique across the session; repeated: {sorted(set(repeated))}"
        raise ValueError(msg)
    seen = {t.q_sha256: t for t in manifest.takes if t.q_sha256 is not None}
    verdicts: list[TakeVerdict] = []
    for attempt, path in numbered:
        verdict = _validate_take(path, attempt, seen, context)
        verdicts.append(verdict)
        if verdict.q_sha256 is not None and verdict.q_sha256 not in seen:
            seen[verdict.q_sha256] = verdict
    stamp = (now or datetime.now(tz=UTC)).replace(microsecond=0).isoformat()
    takes = _assign(tuple(sorted([*manifest.takes, *verdicts], key=lambda t: (t.batch, t.attempt))), required)
    manifest = replace(manifest, takes=takes, updated_at=stamp)
    write_bank_manifest(manifest_file, manifest)
    batch_verdicts = tuple(t for t in takes if t.batch == batch)
    stem = manifest_file.with_suffix("").name
    report = BatchReport(
        batch=batch,
        verdicts=batch_verdicts,
        accepted_total=len(manifest.accepted_attempts),
        required=required,
        shortfall=manifest.shortfall,
        complete=manifest.complete,
        next_action=_next_action(manifest),
        manifest_file=manifest_file,
        report_json=manifest_file.with_name(f"{stem}_batch_{batch:03d}.json"),
        report_markdown=manifest_file.with_name(f"{stem}_batch_{batch:03d}.md"),
    )
    payload = {
        "batch": batch,
        "session": session,
        "verdicts": [dataclasses.asdict(v) for v in batch_verdicts],
        "accepted_total": report.accepted_total,
        "required": required,
        "shortfall": report.shortfall,
        "complete": report.complete,
        "next_action": report.next_action,
        "updated_at": stamp,
    }
    report.report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.report_markdown.write_text(_render_markdown(report, manifest), encoding="utf-8")
    return report


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
