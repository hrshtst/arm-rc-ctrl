# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Derive the recovery training episode from a raw demonstration, transactionally (M3R-003).

Pipeline (``docs/PLAN.md`` section 5.4; the recovery plan, section 4.1): load
and verify the raw log → smooth the complete recording, stationary pre-roll
included → resample onto the scenario's control period → derivatives and
endpoint kinematics with full pre-roll context → crop at the confirmed motion
onset (the programmed onset for a scripted demonstration) → re-zero the task
clock and annotate move/dwell phases → recovery validation → normalization on
the cropped episode. The pre-roll baseline ``q_pre`` only validates the crop:
a material deviation from the first cropped sample ``q0_ref`` marks the record
``flagged`` for review and is reported, never silently substituted.

Payload, record, and catalog finalization is shared with
:mod:`arm_rc_ctrl.data.preprocess`: the payload is staged, digested, moved
atomically to its content-addressed location, and never overwritten; the raw
artifact is read-only throughout.

Command line::

    python -m arm_rc_ctrl.data.recover --raw data/records/raw/<id>.toml
        --scenario configs/tasks/task_1a.toml [--config configs/preprocessing/recovery_v1.toml] [--exploratory]
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
from typing import Final, Literal

import numpy as np

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.preprocess import (
    PENDING_RECORD_FILE,
    PROVENANCE_FILE,
    NormalizationSettings,
    ResamplingSettings,
    finalize_catalog,
    finalize_payload,
    finalize_record,
)
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
    RawDemonstrationRecord,
    array_specs,
    load_record,
    make_artifact_id,
    to_toml,
)
from arm_rc_ctrl.data.recovery import (
    TASK_PHASE_CODES,
    BaselineCheck,
    CropWindow,
    OnsetAnnotation,
    RecoveryDatasetRecord,
    TaskIntervals,
    annotate_task_phases,
    validate_recovery_dataset,
)
from arm_rc_ctrl.data.resampling import ResamplingConfig, resample
from arm_rc_ctrl.data.samples import SAMPLES_SCHEMA_VERSION, SampleSet, save_samples
from arm_rc_ctrl.data.smoothing import SmoothingConfig, smooth
from arm_rc_ctrl.data.validate import ValidationSpec
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
    "BaselineSettings",
    "RecoverError",
    "RecoverResult",
    "RecoveryDeriveConfig",
    "main",
    "recover_demonstration",
]

DEFAULT_CONFIG = Path("configs") / "preprocessing" / "recovery_v1.toml"
_TASK_DIM: Final = 2
_GRID_TOLERANCE_S: Final = 1e-9
_MIN_TASK_SAMPLES: Final = 2


class RecoverError(RuntimeError):
    """The demonstration cannot be derived into a recovery episode with the given configuration."""


@dataclass(frozen=True)
class BaselineSettings:
    """Robust pre-roll baseline estimator and the material-difference threshold."""

    estimator: Literal["median"] = "median"
    tolerance_rad: float = 0.05

    def __post_init__(self) -> None:
        """Validate the threshold."""
        if not (self.tolerance_rad > 0 and self.tolerance_rad < float("inf")):
            msg = f"baseline.tolerance_rad must be positive and finite, got {self.tolerance_rad!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecoveryDeriveConfig:
    """Complete recovery derivation configuration (``configs/preprocessing/recovery_*.toml``)."""

    smoothing: SmoothingConfig
    resampling: ResamplingSettings
    derivatives: DerivativeConfig
    normalization: NormalizationSettings
    baseline: BaselineSettings


@dataclass(frozen=True)
class RecoverResult:
    """What one recovery derivation produced."""

    record: RecoveryDatasetRecord
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
        raise RecoverError(msg)
    if raw.scenario.dof != scenario.dof:
        msg = f"raw record dof {raw.scenario.dof} != scenario dof {scenario.dof}"
        raise RecoverError(msg)


def _programmed_onset(record: RawDemonstrationRecord) -> OnsetAnnotation:
    """The scripted demonstration's programmed motion onset (the legacy ``prime``/``move`` boundary)."""
    move_start = record.intervals.move[0]
    period = record.sampling.period_s
    sample = round(move_start / period)
    if sample < 1 or abs(sample * period - move_start) > _GRID_TOLERANCE_S:
        msg = (
            f"the programmed onset {move_start!r} s does not lie on the raw sample grid "
            f"(period {period!r} s) with a non-empty pre-roll"
        )
        raise RecoverError(msg)
    onset_s = sample * period
    return OnsetAnnotation(
        kind="scripted",
        raw_artifact_id=record.artifact.artifact_id,
        raw_payload_sha256=record.artifact.payload.sha256,
        detector="programmed",
        detector_params={},
        sampling_period_s=period,
        proposed_onset_sample=sample,
        proposed_onset_s=onset_s,
        confirmed_onset_sample=sample,
        confirmed_onset_s=onset_s,
        confirmed_by="script",
    )


def _derive(
    demo: RawDemonstration, scenario: ScenarioConfig, config: RecoveryDeriveConfig
) -> tuple[SampleSet, OnsetAnnotation, BaselineCheck, CropWindow]:
    """Process with full pre-roll context, then crop at the confirmed onset."""
    onset = _programmed_onset(demo.record)
    period = scenario.timing.dt
    raw_rate = 1.0 / demo.record.sampling.period_s
    q_smooth = smooth(demo.q, raw_rate, config.smoothing)
    resampling = ResamplingConfig(period_s=period, interpolation=config.resampling.interpolation)  # type: ignore[arg-type]
    t_full, q_full = resample(demo.times, q_smooth, resampling)
    t_full = t_full - t_full[0]
    dq_full, ddq_full = differentiate(q_full, period, config.derivatives)
    tip_full = endpoint_positions(scenario, q_full)
    dtip_full, ddtip_full = differentiate(tip_full, period, config.derivatives)

    crop = int(np.searchsorted(t_full, onset.confirmed_onset_s - _GRID_TOLERANCE_S, side="left"))
    if crop < 1:
        msg = f"the onset at {onset.confirmed_onset_s!r} s leaves no pre-roll samples on the processed grid"
        raise RecoverError(msg)
    if t_full.shape[0] - crop < _MIN_TASK_SAMPLES:
        msg = f"cropping at {onset.confirmed_onset_s!r} s leaves fewer than {_MIN_TASK_SAMPLES} task samples"
        raise RecoverError(msg)
    t_task = t_full[crop:] - t_full[crop]
    boundary = demo.record.intervals.dwell[0] - onset.confirmed_onset_s
    if not 0.0 < boundary < float(t_task[-1]):
        msg = f"the dwell start at task time {boundary!r} s falls outside the cropped episode"
        raise RecoverError(msg)
    task = TaskIntervals(move=(0.0, boundary), dwell=(boundary, float(t_task[-1])))
    phase = annotate_task_phases(t_task, task)

    q0_ref = tuple(float(value) for value in q_full[crop])
    q_pre = tuple(float(value) for value in np.median(q_full[:crop], axis=0))
    deviation = float(np.max(np.abs(np.asarray(q_pre) - np.asarray(q0_ref))))
    tolerance = config.baseline.tolerance_rad
    baseline = BaselineCheck(
        q_pre=q_pre,
        tolerance_rad=tolerance,
        max_deviation_rad=deviation,
        status="passed" if deviation <= tolerance else "flagged",
    )
    crop_window = CropWindow(
        pre_roll=(0.0, onset.confirmed_onset_s), source_duration_s=demo.record.duration_s, task=task
    )
    samples = SampleSet(
        t=t_task,
        q=q_full[crop:],
        dq=dq_full[crop:],
        ddq=ddq_full[crop:],
        tip=tip_full[crop:],
        dtip=dtip_full[crop:],
        ddtip=ddtip_full[crop:],
        task_code=np.zeros((t_task.shape[0], 0), dtype=np.float64),
        phase=phase,
    )
    return samples, onset, baseline, crop_window


def _build_record(
    raw: RawDemonstrationRecord,
    scenario: ScenarioConfig,
    config: RecoveryDeriveConfig,
    samples: SampleSet,
    onset: OnsetAnnotation,
    baseline: BaselineCheck,
    crop: CropWindow,
    *,
    artifact_id: str,
    digest: str,
    size: int,
    provenance: ProvenanceRecord,
    command: str,
    license_label: str,
    access: AccessClass,
) -> RecoveryDatasetRecord:
    """The Git-tracked record of the payload, deterministic given the samples and the provenance."""
    normalization = fit_normalization(
        samples.arrays(),
        config.normalization.channels,
        fitted_on=(artifact_id,),
        training_rows=np.ones(samples.n_samples, dtype=np.bool_),
        near_zero=config.normalization.near_zero,
    )
    return RecoveryDatasetRecord(
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
            notes=(
                f"Recovery episode cropped at the {onset.kind} onset from {raw.artifact.artifact_id} "
                f"under scenario {scenario.name}."
            ),
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(TASK_PHASE_CODES),
        preprocessing=Preprocessing(
            resample_period_s=scenario.timing.dt,
            smoothing=config.smoothing.label,
            smoothing_params=config.smoothing.parameters(),
            derivative_method=config.derivatives.label,
            interpolation=config.resampling.interpolation,
        ),
        onset=onset,
        baseline=baseline,
        crop=crop,
        q0_ref=tuple(float(value) for value in samples.q[0]),
        arrays=array_specs(samples),
        normalization=normalization,
    )


def recover_demonstration(
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
    command: str = "python -m arm_rc_ctrl.data.recover",
) -> RecoverResult:
    """Run the recovery derivation and persist payload, record, and catalog entry.

    Parameters
    ----------
    raw_record_file, scenario_file, config_file : Path
        Git-tracked raw record, scenario TOML, and recovery derivation TOML.
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
    config = load_config(config_file, RecoveryDeriveConfig)
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

    samples, onset, baseline, crop = _derive(demo, scenario, config)
    if baseline.status == "flagged":
        print(
            f"warning: pre-roll baseline deviates from q0_ref by {baseline.max_deviation_rad:.6g} rad "
            f"(> {baseline.tolerance_rad:.6g}); the record is flagged for review",
            file=sys.stderr,
        )
    spec = ValidationSpec(
        dof=scenario.dof,
        task_dim=_TASK_DIM,
        task_code_dim=0,
        period_s=scenario.timing.dt,
        limits=joint_limits(scenario),
        require_all_phases=False,
    )
    validate_recovery_dataset(samples, spec)

    staging = store.root / "processed" / f"staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        payload_file = staging / PROCESSED_PAYLOAD_NAME
        save_samples(payload_file, samples)
        digest = sha256_file(payload_file)
        size = payload_file.stat().st_size

        def rebuild(
            artifact_id: str, origin: ProvenanceRecord, command_line: str, license_label: str, access: AccessClass
        ) -> RecoveryDatasetRecord:
            return _build_record(
                raw,
                scenario,
                config,
                samples,
                onset,
                baseline,
                crop,
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
            schema=RecoveryDatasetRecord,
            requested=(license_override, access_override),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    artifact_id = record.artifact.artifact_id
    final_dir = store.path(ArtifactUri("processed", (artifact_id,)), mode="write")
    record_file = records_root / "data" / "records" / "processed" / f"{artifact_id}.toml"
    record = finalize_record(record_file, record, schema=RecoveryDatasetRecord, resumed=resumed)
    finalize_catalog(records_root, record.artifact, record_file)
    return RecoverResult(record, samples, record_file, final_dir / PROCESSED_PAYLOAD_NAME, provenance, resumed)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; a thin wrapper around :func:`recover_demonstration`."""
    parser = argparse.ArgumentParser(description="Derive the recovery training episode from a raw demonstration.")
    parser.add_argument("--raw", type=Path, required=True, help="Git-tracked raw demonstration record (TOML)")
    parser.add_argument(
        "--scenario", type=Path, required=True, help="scenario TOML the demonstration was recorded under"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help=f"recovery derivation TOML (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument("--license", default=None, help="override the license inherited from the raw record")
    parser.add_argument("--access", default=None, choices=("private", "internal", "public"), help="override access")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    root = repository_root()
    config_file = root / DEFAULT_CONFIG if args.config is None else Path(args.config)
    result = recover_demonstration(
        Path(args.raw),
        Path(args.scenario),
        config_file,
        store=open_storage(),
        records_root=root,
        exploratory=args.exploratory,
        license_override=args.license,
        access_override=args.access,
        now=datetime.now(UTC),
        command=command_line("arm_rc_ctrl.data.recover", argv if argv is not None else sys.argv[1:]),
    )
    print(
        json.dumps(
            {
                "artifact_id": result.record.artifact.artifact_id,
                "uri": result.record.artifact.payload.uri,
                "record": result.record_file.relative_to(root).as_posix(),
                "n_samples": result.record.n_samples,
                "onset_s": result.record.onset.confirmed_onset_s,
                "baseline_status": result.record.baseline.status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
