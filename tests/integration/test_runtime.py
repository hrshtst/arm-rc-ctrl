# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2 review round 1: a recipe's datasets are bound by identity, widths, preprocessing, and transform derivation."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import DatasetSource, ModelRecipe, RclibIdentity, create_recipe
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=20, spectral_radius=0.9, sparsity=0.8, leak_rate=0.5, input_scaling=0.5, seed=2
    ),
    readout=ReadoutConfig(alpha=1e-3),
)


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[StorageRoot, Path, PreprocessResult, ModelRecipe]:
    """A store with the fixture dataset and a recipe trained on it."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD,
        SCENARIO,
        PREPROCESS,
        store=store,
        records_root=records,
        exploratory=True,
        now=datetime(2026, 9, 1, tzinfo=UTC),
    )
    record = processed.record
    assert record.normalization is not None
    source = DatasetSource(
        record.artifact.artifact_id,
        record.artifact.payload.sha256,
        processed.record_file.relative_to(records).as_posix(),
    )
    recipe, _ = create_recipe(
        "binding",
        ESN,
        sources=[source],
        samples={record.artifact.artifact_id: processed.samples},
        dof=record.dof,
        task_code_dim=record.task_code_dim,
        preprocessing=record.preprocessing,
        transform=InputTransform.derive("fixed_scale", record.normalization, fixed_scales={"q": 0.3, "dq": 4.0}),
        rclib=RclibIdentity.current(),
    )
    return store, records, processed, recipe


def test_bound_datasets_load(prepared: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe]) -> None:
    """A consistent recipe resolves its samples and refits."""
    store, records, processed, recipe = prepared
    samples = load_training_samples(recipe, store, records_root=records)
    assert list(samples) == [processed.record.artifact.artifact_id]
    assert np.array_equal(samples[processed.record.artifact.artifact_id].q, processed.samples.q)
    recipe.refit(samples)


def test_mismatched_records_are_refused(prepared: tuple[StorageRoot, Path, PreprocessResult, ModelRecipe]) -> None:
    """Different preprocessing, widths, identity, or a transform from other statistics cannot be loaded silently."""
    store, records, processed, recipe = prepared
    record = processed.record
    assert record.normalization is not None
    other_preprocessing = dataclasses.replace(recipe.preprocessing, interpolation="cubic")
    with pytest.raises(ValueError, match="preprocessed differently"):
        load_training_samples(
            dataclasses.replace(recipe, preprocessing=other_preprocessing), store, records_root=records
        )
    (source,) = recipe.datasets
    wrong_digest = dataclasses.replace(source, payload_sha256="ab" * 32)
    with pytest.raises(ValueError, match="describes"):
        load_training_samples(dataclasses.replace(recipe, datasets=(wrong_digest,)), store, records_root=records)
    shifted = fit_normalization(
        {"q": processed.samples.q + 0.1, "dq": processed.samples.dq},
        ("q", "dq"),
        fitted_on=(record.artifact.artifact_id,),
        training_rows=np.ones(processed.samples.n_samples, dtype=np.bool_),
    )
    other_transform = InputTransform.derive("fixed_scale", shifted, fixed_scales={"q": 0.3, "dq": 4.0})
    with pytest.raises(ValueError, match="does not derive from the recorded normalization"):
        load_training_samples(dataclasses.replace(recipe, transform=other_transform), store, records_root=records)
    other_scales = InputTransform.derive("fixed_scale", record.normalization, fixed_scales={"q": 0.5, "dq": 4.0})
    load_training_samples(
        dataclasses.replace(recipe, transform=other_scales), store, records_root=records
    )  # same source, other policy values: consistent
    with_code = dataclasses.replace(recipe, task_code_dim=1)  # valid recipe, but not this dataset's width
    with pytest.raises(ValueError, match="dof 2 and task_code_dim 0; the recipe expects 2 and 1"):
        with_code.check_dataset_record(source, record)
