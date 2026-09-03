# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-005: augmentation families are validated and visualized reproducibly before any training."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from arm_rc_ctrl.config import from_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.records import (
    CANONICAL_UNITS,
    ArtifactRecord,
    Origin,
    Payload,
    Preprocessing,
    Scenario,
    array_specs,
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
)
from arm_rc_ctrl.data.samples import SampleSet, save_samples
from arm_rc_ctrl.experiments.augmentation_validation import (
    ANCHOR,
    AugmentationValidationReport,
    approved_grid,
    main,
    plot_configuration,
    report_to_json,
    report_to_markdown,
    validate_augmentation,
)
from arm_rc_ctrl.provenance import collect_provenance, sha256_file
from arm_rc_ctrl.rc.augment import (
    APPROVED_GAMMA,
    APPROVED_N_SYNTHETIC,
    APPROVED_PHI,
    APPROVED_SIGMA_RAD,
    SEED_NAMESPACE,
    AugmentationConfig,
    generate_augmentation,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
SCENARIO_FILE = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
SCENARIO = load_scenario(SCENARIO_FILE)
DERIVATIVES = DerivativeConfig(method="central")
RAW_SOURCE = "raw-20260830-2a97516c354b"
CREATED = "2026-09-03T09:00:00+00:00"
N = 101
DT = 0.01
TASK = TaskIntervals(move=(0.0, 0.8), dwell=(0.8, 1.0))
GRID = (
    AugmentationConfig(n_synthetic=16, sigma_rad=0.05, phi=0.99, gamma=1.0, seed_bank=1, attempt_budget=64),
    AugmentationConfig(n_synthetic=16, sigma_rad=0.01, phi=0.98, gamma=2.0, seed_bank=1, attempt_budget=64),
)


def _samples() -> SampleSet:
    t = np.arange(N, dtype=np.float64) * DT
    start = np.array([0.3, 0.6])
    goal = np.array([0.8, 0.4])
    s = np.clip(t / TASK.move[1], 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < TASK.dwell[0], 1, 2).astype(np.int64)
    task_code = np.zeros((N, 0), dtype=np.float64)
    return SampleSet(t=t, q=q, dq=dq, ddq=ddq, tip=tip, dtip=dtip, ddtip=ddtip, task_code=task_code, phase=phase)


def _record(samples: SampleSet, payload_sha: str, payload_size: int = 2048) -> RecoveryDatasetRecord:
    artifact_id = make_artifact_id("processed", CREATED, payload_sha)
    onset = OnsetAnnotation(
        kind="scripted",
        raw_artifact_id=RAW_SOURCE,
        raw_payload_sha256="b" * 64,
        detector="programmed",
        detector_params={},
        sampling_period_s=DT,
        proposed_onset_sample=100,
        proposed_onset_s=1.0,
        confirmed_onset_sample=100,
        confirmed_onset_s=1.0,
        confirmed_by="script",
    )
    return RecoveryDatasetRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="processed",
            created_at=CREATED,
            license="LicenseRef-Private",
            access="private",
            payload=Payload(
                f"armrc://processed/{artifact_id}/samples.npz", payload_sha, payload_size, "samples.npz", 1
            ),
            origin=Origin(
                command="synthetic recovery fixture",
                config_sha256="2" * 64,
                project_commit="a" * 40,
                project_dirty=False,
                dependency_commits={},
                sources=(RAW_SOURCE,),
            ),
        ),
        scenario=Scenario(
            config_path="tests/fixtures/configs/planar_2dof_fixture.toml",
            config_sha256=sha256_file(SCENARIO_FILE),
            robot="planar-2dof-fixture",
            task="pd-reach-fixture",
            dof=2,
            initial_q=(0.3, 0.6),
            target=(0.2996, 0.4482),
        ),
        n_samples=samples.n_samples,
        dof=samples.dof,
        task_dim=samples.task_dim,
        task_code_dim=samples.task_code_dim,
        units=dict(CANONICAL_UNITS),
        phases=dict(TASK_PHASE_CODES),
        preprocessing=Preprocessing(
            resample_period_s=DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
        ),
        onset=onset,
        baseline=BaselineCheck(q_pre=(0.3, 0.6), tolerance_rad=0.05, max_deviation_rad=0.0, status="passed"),
        crop=CropWindow(pre_roll=(0.0, 1.0), source_duration_s=2.0, task=TASK),
        q0_ref=tuple(float(v) for v in samples.q[0]),
        arrays=array_specs(samples),
    )


def _fixture(tmp_path: Path) -> tuple[RecoveryDatasetRecord, SampleSet]:
    samples = _samples()
    payload = tmp_path / "samples.npz"
    save_samples(payload, samples)
    return _record(samples, sha256_file(payload)), samples


def _provenance():  # noqa: ANN202
    return collect_provenance({"purpose": "test"}, seeds={}, artifacts=[], exploratory=True)


def _small_grid(**_: object) -> tuple[AugmentationConfig, ...]:
    """A two-configuration stand-in for the approved grid (CLI test speed)."""
    return GRID


def test_grid_covers_the_approved_ranges() -> None:
    """The default grid enumerates every approved D1 combination once and contains the anchor."""
    grid = approved_grid()
    expected = len(APPROVED_N_SYNTHETIC) * len(APPROVED_SIGMA_RAD) * len(APPROVED_PHI) * len(APPROVED_GAMMA)
    assert len(grid) == expected
    combos = {(c.n_synthetic, c.sigma_rad, c.phi, c.gamma) for c in grid}
    assert len(combos) == expected
    assert all(c.attempt_budget == 4 * c.n_synthetic for c in grid)
    assert ANCHOR in grid
    assert (ANCHOR.n_synthetic, ANCHOR.sigma_rad, ANCHOR.phi, ANCHOR.gamma) == (64, 0.05, 0.99, 1.0)


def test_a_valid_dataset_passes_every_check(tmp_path: Path) -> None:
    """Determinism, bounds, smoothness, correlation, envelope, dwell collapse, and separation all pass."""
    record, samples = _fixture(tmp_path)
    report = validate_augmentation(record, samples, SCENARIO, GRID, provenance=_provenance())
    assert report.passed
    assert report.dataset == record.artifact.artifact_id
    assert report.payload_sha256 == record.artifact.payload.sha256
    assert report.seed_namespace == SEED_NAMESPACE
    assert len(report.configurations) == len(GRID)
    names = {check.name for cfg in report.configurations for check in cfg.checks}
    assert {
        "determinism",
        "bounds",
        "smoothness",
        "correlation",
        "envelope",
        "dwell-collapse",
        "episode-separation",
        "rejection-accounting",
    } <= names
    assert {check.name for check in report.checks} >= {"source-binding", "bank-separation"}
    for cfg in report.configurations:
        assert all(check.passed for check in cfg.checks)
        assert {f.family for f in cfg.families} == {"non_decaying", "contractive"}
        for family in cfg.families:
            assert 0.0 < family.rms_median <= family.peak_max


def test_report_round_trips_and_renders_markdown(tmp_path: Path) -> None:
    """The canonical JSON reloads equal; the Markdown names every check and configuration."""
    record, samples = _fixture(tmp_path)
    report = validate_augmentation(record, samples, SCENARIO, GRID, provenance=_provenance())
    text = report_to_json(report)
    loaded = from_mapping(cast("dict[str, object]", json.loads(text)), AugmentationValidationReport)
    assert loaded == report
    markdown = report_to_markdown(report)
    assert "determinism" in markdown
    assert "dwell-collapse" in markdown
    assert record.artifact.artifact_id in markdown
    assert "PASS" in markdown


def test_failures_remain_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An impossible tolerance produces a failing, fully reported check instead of an exception."""
    monkeypatch.setattr("arm_rc_ctrl.experiments.augmentation_validation._CORRELATION_TOLERANCE", 1e-12)
    record, samples = _fixture(tmp_path)
    report = validate_augmentation(record, samples, SCENARIO, GRID, provenance=_provenance())
    assert not report.passed
    failing = [c for cfg in report.configurations for c in cfg.checks if not c.passed]
    assert failing
    assert all(c.name == "correlation" and c.detail for c in failing)
    markdown = report_to_markdown(report)
    assert "FAIL" in markdown


def test_plot_set_is_written(tmp_path: Path) -> None:
    """The anchor visualization writes the delta, position, and statistics panels."""
    samples = _samples()
    result = generate_augmentation(samples.t, samples.q, TASK, SCENARIO, GRID[0], derivatives=DERIVATIVES)
    written = plot_configuration(result, samples.t, tmp_path, stem="augmentation_v1")
    assert [p.name for p in written] == [
        "augmentation_v1_non_decaying.png",
        "augmentation_v1_contractive.png",
        "augmentation_v1_positions.png",
        "augmentation_v1_statistics.png",
    ]
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_command_line_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI verifies the payload, writes report, markdown, and plots, and exits by outcome."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    samples = _samples()
    staged = tmp_path / "samples.npz"
    save_samples(staged, samples)
    record = _record(samples, sha256_file(staged), staged.stat().st_size)
    store = StorageRoot(store_root, repositories=(REPO_ROOT,))
    store.path(record.artifact.payload.uri, mode="write").write_bytes(staged.read_bytes())
    record_file = tmp_path / f"{record.artifact.artifact_id}.toml"
    record_file.write_text(to_toml(record), encoding="utf-8")
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store_root))
    monkeypatch.setattr("arm_rc_ctrl.experiments.augmentation_validation.approved_grid", _small_grid)
    report_file = tmp_path / "validation.json"
    markdown_file = tmp_path / "validation.md"
    plots_dir = tmp_path / "plots"
    code = main(
        [
            "--dataset",
            str(record_file),
            "--scenario",
            str(SCENARIO_FILE),
            "--report",
            str(report_file),
            "--markdown",
            str(markdown_file),
            "--plots",
            str(plots_dir),
            "--exploratory",
        ]
    )
    assert code == 0
    loaded = from_mapping(
        cast("dict[str, object]", json.loads(report_file.read_text(encoding="utf-8"))), AugmentationValidationReport
    )
    assert loaded.passed
    assert loaded.dataset == record.artifact.artifact_id
    assert markdown_file.is_file()
    assert sorted(p.name for p in plots_dir.iterdir())
    with pytest.raises(FileExistsError, match="exists"):
        main(["--dataset", str(record_file), "--scenario", str(SCENARIO_FILE), "--report", str(report_file)])


def test_no_confirmatory_seed_is_consumed() -> None:
    """The augmentation namespace is disjoint from every M3 confirmatory seed and from date-shaped values."""
    confirmatory = {20260901, 20260902, 20260903, 20260904, 20260905}
    assert SEED_NAMESPACE == 415926535
    assert SEED_NAMESPACE not in confirmatory
    assert not 20200101 <= SEED_NAMESPACE <= 20401231  # never a YYYYMMDD-shaped seed
    for config in (dataclasses.replace(GRID[0], seed_bank=7), GRID[1]):
        assert config.seed_bank not in confirmatory
