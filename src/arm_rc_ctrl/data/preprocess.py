# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Turn a raw demonstration into a canonical processed dataset, transactionally.

Pipeline (``docs/PLAN.md`` section 4): load and verify the raw log → smooth
joint positions (zero-phase) → resample onto the scenario's control period →
derivatives → endpoint kinematics → phase annotation → :class:`SampleSet` →
validation → normalization statistics. The payload is written to a staging
directory under the ``processed`` bucket, digested, and moved atomically to its
immutable content-addressed location; only then is the Git-tracked record
written and the catalog appended. Nothing is ever overwritten, and the storage
root never falls back to the repository. If an earlier invocation was
interrupted after the payload was finalized, a same-day retry verifies the
identical payload (its digest is its identity), reuses an identical record,
and completes the missing record/catalog steps.

Command line::

    python -m arm_rc_ctrl.data.preprocess --raw data/records/raw/<id>.toml
        --scenario configs/tasks/task_1a.toml [--config configs/preprocessing/default.toml] [--exploratory]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast

import numpy as np

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.phases import annotate_phases
from arm_rc_ctrl.data.raw import RawDemonstration, load_raw_demonstration
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    PROCESSED_PAYLOAD_FORMAT,
    PROCESSED_PAYLOAD_NAME,
    AccessClass,
    ArtifactRecord,
    Catalog,
    Origin,
    Payload,
    Preprocessing,
    ProcessedDatasetRecord,
    RawDemonstrationRecord,
    array_specs,
    catalog_path,
    load_catalog,
    load_record,
    make_artifact_id,
    to_toml,
    write_catalog,
    write_record,
)
from arm_rc_ctrl.data.resampling import ResamplingConfig, resample
from arm_rc_ctrl.data.samples import PHASE_CODES, SAMPLES_SCHEMA_VERSION, SampleSet, save_samples
from arm_rc_ctrl.data.smoothing import SmoothingConfig, smooth
from arm_rc_ctrl.data.validate import ValidationSpec, validate_dataset
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, endpoint_positions, joint_limits, load_scenario
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot, open_storage

__all__ = [
    "DEFAULT_CONFIG",
    "PENDING_RECORD_FILE",
    "PROVENANCE_FILE",
    "ArtifactCarrier",
    "NormalizationSettings",
    "PreprocessConfig",
    "PreprocessError",
    "PreprocessResult",
    "RecordBuilder",
    "ResamplingSettings",
    "finalize_catalog",
    "finalize_payload",
    "finalize_record",
    "main",
    "preprocess_demonstration",
]

DEFAULT_CONFIG = Path("configs") / "preprocessing" / "default.toml"
PROVENANCE_FILE = "provenance.json"
PENDING_RECORD_FILE = "record.toml"
"""Copy of the complete record kept beside the payload from finalization on (see ``finalize_payload``)."""
_TASK_DIM = 2


class PreprocessError(RuntimeError):
    """The demonstration cannot be processed with the given scenario/configuration."""


@dataclass(frozen=True)
class ResamplingSettings:
    """Interpolation method; the period comes from the scenario."""

    interpolation: str = "linear"

    def __post_init__(self) -> None:
        """Validate the method label."""
        if self.interpolation not in ("linear", "cubic"):
            msg = f"resampling.interpolation must be 'linear' or 'cubic', got {self.interpolation!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class NormalizationSettings:
    """Which channels get statistics and the near-zero threshold."""

    channels: tuple[str, ...]
    near_zero: float = 1e-8


@dataclass(frozen=True)
class PreprocessConfig:
    """Complete preprocessing configuration (``configs/preprocessing/*.toml``)."""

    smoothing: SmoothingConfig
    resampling: ResamplingSettings
    derivatives: DerivativeConfig
    normalization: NormalizationSettings


@dataclass(frozen=True)
class PreprocessResult:
    """What one preprocessing run produced."""

    record: ProcessedDatasetRecord
    samples: SampleSet
    record_file: Path
    payload_file: Path
    provenance: ProvenanceRecord
    resumed: bool = False
    """``True`` when an interrupted earlier invocation had already finalized the identical payload."""


def _check_scenario_matches(raw: RawDemonstrationRecord, scenario_path: Path, scenario: ScenarioConfig) -> None:
    digest = sha256_file(scenario_path)
    if raw.scenario.config_sha256 != digest:
        msg = (
            f"raw record {raw.artifact.artifact_id} was recorded under scenario digest "
            f"{raw.scenario.config_sha256[:12]} but {scenario_path} has digest {digest[:12]}"
        )
        raise PreprocessError(msg)
    if raw.scenario.dof != scenario.dof:
        msg = f"raw record dof {raw.scenario.dof} != scenario dof {scenario.dof}"
        raise PreprocessError(msg)


def _build_samples(demo: RawDemonstration, scenario: ScenarioConfig, config: PreprocessConfig) -> SampleSet:
    period = scenario.timing.dt
    raw_rate = 1.0 / demo.record.sampling.period_s
    q_smooth = smooth(demo.q, raw_rate, config.smoothing)
    resampling = ResamplingConfig(period_s=period, interpolation=config.resampling.interpolation)  # type: ignore[arg-type]
    t, q = resample(demo.times, q_smooth, resampling)
    t = t - t[0]
    dq, ddq = differentiate(q, period, config.derivatives)
    tip = endpoint_positions(scenario, q)
    dtip, ddtip = differentiate(tip, period, config.derivatives)
    phase = annotate_phases(t, demo.record.intervals)
    task_code = np.zeros((t.shape[0], 0), dtype=np.float64)
    return SampleSet(t=t, q=q, dq=dq, ddq=ddq, tip=tip, dtip=dtip, ddtip=ddtip, task_code=task_code, phase=phase)


def preprocess_demonstration(
    raw_record_file: Path,
    scenario_file: Path,
    config_file: Path,
    *,
    store: StorageRoot,
    records_root: Path,
    exploratory: bool,
    license_override: str | None = None,
    access_override: AccessClass | None = None,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.data.preprocess",
) -> PreprocessResult:
    """Run the pipeline and persist payload, record, and catalog entry.

    Parameters
    ----------
    raw_record_file, scenario_file, config_file : Path
        Git-tracked raw record, scenario TOML, and preprocessing TOML.
    store : StorageRoot
        Validated external storage root.
    records_root : Path
        Repository root under which ``data/records`` and ``data/catalog.toml`` live.
    exploratory : bool
        Tolerate a dirty worktree (the record marks the origin as dirty either way).
    license_override, access_override : optional
        Replace the license/access inherited from the raw record.
    now : datetime | None, optional
        Creation timestamp override (timezone-aware).
    command : str, optional
        Command recorded in the origin.
    """
    raw = load_record(raw_record_file, RawDemonstrationRecord)
    scenario = load_scenario(scenario_file)
    config = load_config(config_file, PreprocessConfig)
    _check_scenario_matches(raw, scenario_file, scenario)

    demo = load_raw_demonstration(store, raw)
    license_label = license_override or raw.artifact.license
    access = access_override or raw.artifact.access
    resolved = {
        "scenario": to_mapping(scenario),
        "preprocessing": to_mapping(config),
        "raw_artifact": raw.artifact.artifact_id,
        # Immutable record metadata is bound to the provenance so a resumed run rebuilds the record from it.
        "record": {"license": license_label, "access": access, "command": command},
    }
    source_ref = ArtifactReference(raw.artifact.payload.uri, raw.artifact.payload.sha256, raw.artifact.payload.size)
    provenance = collect_provenance(resolved, seeds={}, artifacts=[source_ref], exploratory=exploratory, now=now)
    require_clean_for_confirmatory(provenance)

    samples = _build_samples(demo, scenario, config)
    spec = ValidationSpec(
        dof=scenario.dof,
        task_dim=_TASK_DIM,
        task_code_dim=0,
        period_s=scenario.timing.dt,
        limits=joint_limits(scenario),
    )
    validate_dataset(samples, spec)

    staging = store.root / "processed" / f"staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        payload_file = staging / PROCESSED_PAYLOAD_NAME
        save_samples(payload_file, samples)
        digest = sha256_file(payload_file)
        size = payload_file.stat().st_size

        def rebuild(
            artifact_id: str, origin: ProvenanceRecord, command_line: str, license_label: str, access: AccessClass
        ) -> ProcessedDatasetRecord:
            return _build_record(
                raw,
                scenario,
                config,
                samples,
                artifact_id=artifact_id,
                digest=digest,
                size=size,
                provenance=origin,
                command=command_line,
                license_label=license_label,
                access=access,
            )

        record = rebuild(
            make_artifact_id("processed", provenance.created_at, digest), provenance, command, license_label, access
        )
        (staging / PROVENANCE_FILE).write_text(provenance.to_json() + "\n", encoding="utf-8")
        # The complete pending record travels with the payload, so a retry cannot redefine its metadata.
        (staging / PENDING_RECORD_FILE).write_text(to_toml(record), encoding="utf-8")
        record, provenance, resumed = finalize_payload(
            store,
            staging,
            record,
            provenance,
            rebuild,
            schema=ProcessedDatasetRecord,
            requested=(license_override, access_override),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    artifact_id = record.artifact.artifact_id
    final_dir = store.path(ArtifactUri("processed", (artifact_id,)), mode="write")
    record_file = records_root / "data" / "records" / "processed" / f"{artifact_id}.toml"
    record = finalize_record(record_file, record, schema=ProcessedDatasetRecord, resumed=resumed)
    finalize_catalog(records_root, record.artifact, record_file)
    return PreprocessResult(record, samples, record_file, final_dir / PROCESSED_PAYLOAD_NAME, provenance, resumed)


def _build_record(
    raw: RawDemonstrationRecord,
    scenario: ScenarioConfig,
    config: PreprocessConfig,
    samples: SampleSet,
    *,
    artifact_id: str,
    digest: str,
    size: int,
    provenance: ProvenanceRecord,
    command: str,
    license_label: str,
    access: AccessClass,
) -> ProcessedDatasetRecord:
    """The Git-tracked record of the payload, deterministic given the samples and the provenance."""
    normalization = fit_normalization(
        samples.arrays(),
        config.normalization.channels,
        fitted_on=(artifact_id,),
        training_rows=np.ones(samples.n_samples, dtype=np.bool_),
        near_zero=config.normalization.near_zero,
    )
    return ProcessedDatasetRecord(
        scenario=raw.scenario,
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=provenance.created_at,
            license=license_label,
            access=access,
            payload=Payload(
                uri=f"armrc://processed/{artifact_id}/{PROCESSED_PAYLOAD_NAME}",
                sha256=digest,
                size=size,
                format=PROCESSED_PAYLOAD_FORMAT,
                schema_version=SAMPLES_SCHEMA_VERSION,
            ),
            origin=Origin.from_provenance(provenance, command=command, sources=(raw.artifact.artifact_id,)),
            notes=f"Processed from {raw.artifact.artifact_id} under scenario {scenario.name}.",
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(PHASE_CODES),
        preprocessing=Preprocessing(
            resample_period_s=scenario.timing.dt,
            smoothing=config.smoothing.label,
            smoothing_params=config.smoothing.parameters(),
            derivative_method=config.derivatives.label,
            interpolation=config.resampling.interpolation,
        ),
        arrays=array_specs(samples),
        normalization=normalization,
    )


class ArtifactCarrier(Protocol):
    """A dataset record dataclass carrying the common artifact envelope (e.g. processed and recovery records)."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

    @property
    def artifact(self) -> ArtifactRecord:
        """The common artifact envelope."""
        ...


type RecordBuilder[T: ArtifactCarrier] = Callable[[str, ProvenanceRecord, str, str, AccessClass], T]
"""Rebuilds the record for (artifact ID, provenance, command, license, access) from the freshly processed samples."""


def _finalized_payloads(store: StorageRoot, digest: str) -> tuple[list[Path], list[Path]]:
    """Finalized processed directories holding exactly this payload, on any date, and conflicting ones.

    Directories are pre-selected by the digest prefix in their content-addressed
    ID and then verified by the full SHA-256 of their payload file; a directory
    whose ID carries the prefix but whose payload is missing or different is a
    conflict in the content-addressed namespace and is reported separately.
    """
    bucket = store.root / "processed"
    suffix = make_artifact_id("processed", "2000-01-01T00:00:00+00:00", digest).rsplit("-", 1)[1]
    matches: list[Path] = []
    conflicts: list[Path] = []
    for path in sorted(bucket.glob(f"processed-*-{suffix}")):
        if not path.is_dir():
            continue
        payload = path / PROCESSED_PAYLOAD_NAME
        (matches if payload.is_file() and sha256_file(payload) == digest else conflicts).append(path)
    return matches, conflicts


def finalize_payload[T: ArtifactCarrier](
    store: StorageRoot,
    staging: Path,
    record: T,
    provenance: ProvenanceRecord,
    rebuild: RecordBuilder[T],
    *,
    schema: type[T],
    requested: tuple[str | None, AccessClass | None],
) -> tuple[T, ProvenanceRecord, bool]:
    """Move the staged payload into place, or adopt the identical payload an earlier invocation finalized.

    A finalized payload is located by its digest whatever day it was created on,
    so a retry after a UTC date rollover resumes instead of duplicating it. Adopting
    it requires byte-identical content, a strictly valid stored provenance from the
    same resolved configuration and source artifacts, and a strictly valid pending
    record that this invocation reproduces exactly from that provenance. The stored
    provenance and pending record (license, access, and command included) then
    describe the dataset; a retry asking for other metadata is refused, and anything
    inconsistent is left in place for inspection.
    """
    digest = record.artifact.payload.sha256
    matches, conflicts = _finalized_payloads(store, digest)
    if conflicts:
        names = ", ".join(path.name for path in conflicts)
        msg = f"{names} exists under {store.root / 'processed'} but its payload differs from the freshly processed one"
        raise FileExistsError(msg + "; inspect and remove it manually")
    if not matches:
        final_dir = store.path(ArtifactUri("processed", (record.artifact.artifact_id,)), mode="write")
        staging.rename(final_dir)
        return record, provenance, False
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        msg = f"several finalized payloads carry digest {digest[:12]} ({names}); inspect and remove them manually"
        raise FileExistsError(msg)
    final_dir = matches[0]
    where = f"{final_dir.name} exists under {final_dir.parent} but"
    try:
        stored = ProvenanceRecord.from_json((final_dir / PROVENANCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        msg = f"{where} its {PROVENANCE_FILE} is missing or invalid ({exc}); inspect and remove it manually"
        raise FileExistsError(msg) from exc
    stored_config, fresh_config = stored.config, provenance.config
    if stored.artifacts != provenance.artifacts or any(
        stored_config.get(section) != fresh_config.get(section) for section in _PIPELINE_SECTIONS
    ):
        msg = f"{where} its {PROVENANCE_FILE} was produced from different inputs; inspect and remove it manually"
        raise FileExistsError(msg)
    metadata = _record_metadata(stored_config, where)
    try:
        pending = load_record(final_dir / PENDING_RECORD_FILE, schema)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        msg = f"{where} its {PENDING_RECORD_FILE} is missing or invalid ({exc}); inspect and remove it manually"
        raise FileExistsError(msg) from exc
    artifact_id = make_artifact_id("processed", stored.created_at, digest)
    expected = rebuild(
        artifact_id, stored, metadata["command"], metadata["license"], cast("AccessClass", metadata["access"])
    )
    if artifact_id != final_dir.name or pending != expected:
        msg = f"{where} its {PENDING_RECORD_FILE} does not describe this payload and provenance; inspect and remove it"
        raise FileExistsError(msg)
    for name, wanted in (("license", requested[0]), ("access", requested[1])):
        if wanted is not None and wanted != metadata[name]:
            msg = f"{where} it was recorded with {name} {metadata[name]!r}, and this retry asks for {wanted!r}"
            raise FileExistsError(msg)
    shutil.rmtree(staging, ignore_errors=True)
    return pending, stored, True


_PIPELINE_SECTIONS = ("scenario", "preprocessing", "raw_artifact")
"""Sections of the resolved configuration that define the payload (the ``record`` section defines its metadata)."""


def _record_metadata(config: dict[str, object], where: str) -> dict[str, str]:
    """The immutable record metadata bound into a stored provenance, validated."""
    section = config.get("record")
    values = cast("dict[str, object]", section) if isinstance(section, dict) else {}
    metadata = {name: values.get(name) for name in ("license", "access", "command")}
    if any(not isinstance(value, str) or not value for value in metadata.values()) or metadata["access"] not in (
        "private",
        "internal",
        "public",
    ):
        msg = f"{where} its {PROVENANCE_FILE} lacks valid record metadata (license, access, command); inspect it"
        raise FileExistsError(msg)
    return cast("dict[str, str]", metadata)


def _differing_fields[T: ArtifactCarrier](existing: T, record: T) -> list[str]:
    """Names of the top-level (and artifact-level) fields in which two records differ."""
    names: list[str] = []
    for field in dataclasses.fields(existing):
        a, b = getattr(existing, field.name), getattr(record, field.name)
        if field.name == "artifact":
            names += [
                f"artifact.{f.name}"
                for f in dataclasses.fields(ArtifactRecord)
                if getattr(a, f.name) != getattr(b, f.name)
            ]
        elif a != b:
            names.append(field.name)
    return names


def finalize_record[T: ArtifactCarrier](record_file: Path, record: T, *, schema: type[T], resumed: bool) -> T:
    """Write the record, or on resume accept an existing record equal to the one rebuilt from the stored provenance."""
    if record_file.exists():
        existing = load_record(record_file, schema)
        differing = _differing_fields(existing, record)
        if not resumed or differing:
            detail = f" (differs in {', '.join(differing)})" if differing else ""
            msg = f"{record_file} already exists and does not describe this payload{detail}; records are immutable"
            raise FileExistsError(msg)
        return existing
    record_file.parent.mkdir(parents=True, exist_ok=True)
    write_record(record_file, record)
    return record


def finalize_catalog(records_root: Path, artifact: ArtifactRecord, record_file: Path) -> None:
    """Append the catalog entry unless an identical one is already present."""
    catalog_file = catalog_path(records_root)
    catalog = load_catalog(catalog_file)
    relative = record_file.relative_to(records_root).as_posix()
    entry = catalog.find(artifact.artifact_id)
    appended = Catalog(catalog.schema_version, ()).with_record(artifact, relative).artifacts[0]
    if entry is not None:
        if entry != appended:
            msg = f"catalog entry {artifact.artifact_id} disagrees with the record"
            raise ValueError(msg)
        return
    write_catalog(catalog_file, catalog.with_record(artifact, relative))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; a thin wrapper around :func:`preprocess_demonstration`."""
    parser = argparse.ArgumentParser(description="Preprocess a raw demonstration into a canonical dataset.")
    parser.add_argument("--raw", type=Path, required=True, help="Git-tracked raw demonstration record (TOML)")
    parser.add_argument(
        "--scenario", type=Path, required=True, help="scenario TOML the demonstration was recorded under"
    )
    parser.add_argument("--config", type=Path, default=None, help=f"preprocessing TOML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--license", default=None, help="override the license inherited from the raw record")
    parser.add_argument("--access", default=None, choices=("private", "internal", "public"), help="override access")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    root = repository_root()
    config_file = root / DEFAULT_CONFIG if args.config is None else Path(args.config)
    result = preprocess_demonstration(
        Path(args.raw),
        Path(args.scenario),
        config_file,
        store=open_storage(),
        records_root=root,
        exploratory=args.exploratory,
        license_override=args.license,
        access_override=args.access,
        now=datetime.now(UTC),
        command=command_line("arm_rc_ctrl.data.preprocess", argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "artifact_id": result.record.artifact.artifact_id,
                "uri": result.record.artifact.payload.uri,
                "record": result.record_file.relative_to(root).as_posix(),
                "n_samples": result.record.n_samples,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
