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
root never falls back to the repository.

Command line::

    python -m arm_rc_ctrl.data.preprocess --raw data/records/raw/<id>.toml
        --scenario configs/tasks/task_1a.toml [--config configs/preprocessing/default.toml] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    require_clean_for_confirmatory,
    sha256_file,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, endpoint_positions, joint_limits, load_scenario
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot, open_storage

__all__ = [
    "DEFAULT_CONFIG",
    "NormalizationSettings",
    "PreprocessConfig",
    "PreprocessError",
    "PreprocessResult",
    "ResamplingSettings",
    "main",
    "preprocess_demonstration",
]

DEFAULT_CONFIG = Path("configs") / "preprocessing" / "default.toml"
PROVENANCE_FILE = "provenance.json"
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


def _require_unused(final_dir: Path, record_file: Path, artifact_id: str) -> None:
    """Datasets and records are immutable: an existing payload directory or record file is an error."""
    if final_dir.exists():
        msg = f"{artifact_id} already exists under {final_dir.parent}; datasets are immutable (identical content?)"
        raise FileExistsError(msg)
    if record_file.exists():
        msg = f"{record_file} already exists; records are immutable"
        raise FileExistsError(msg)


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
    resolved = {
        "scenario": to_mapping(scenario),
        "preprocessing": to_mapping(config),
        "raw_artifact": raw.artifact.artifact_id,
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

    created_at = provenance.created_at
    staging = store.root / "processed" / f"staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        payload_file = staging / PROCESSED_PAYLOAD_NAME
        save_samples(payload_file, samples)
        digest = sha256_file(payload_file)
        artifact_id = make_artifact_id("processed", created_at, digest)
        final_dir = store.path(ArtifactUri("processed", (artifact_id,)), mode="write")
        record_file = records_root / "data" / "records" / "processed" / f"{artifact_id}.toml"
        _require_unused(final_dir, record_file, artifact_id)
        normalization = fit_normalization(
            samples.arrays(),
            config.normalization.channels,
            fitted_on=(artifact_id,),
            training_rows=np.ones(samples.n_samples, dtype=np.bool_),
            near_zero=config.normalization.near_zero,
        )
        record = ProcessedDatasetRecord(
            artifact=ArtifactRecord(
                artifact_id=artifact_id,
                kind="processed",
                created_at=created_at,
                license=license_override or raw.artifact.license,
                access=access_override or raw.artifact.access,
                payload=Payload(
                    uri=f"armrc://processed/{artifact_id}/{PROCESSED_PAYLOAD_NAME}",
                    sha256=digest,
                    size=payload_file.stat().st_size,
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
        (staging / PROVENANCE_FILE).write_text(provenance.to_json() + "\n", encoding="utf-8")
        staging.rename(final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    record_file.parent.mkdir(parents=True, exist_ok=True)
    write_record(record_file, record)
    catalog_file = catalog_path(records_root)
    catalog = load_catalog(catalog_file).with_record(record.artifact, record_file.relative_to(records_root).as_posix())
    write_catalog(catalog_file, catalog)
    return PreprocessResult(record, samples, record_file, final_dir / PROCESSED_PAYLOAD_NAME, provenance)


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
        command=" ".join(
            ["python", "-m", "arm_rc_ctrl.data.preprocess", *(argv if argv is not None else sys.argv[1:])]
        ),
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
