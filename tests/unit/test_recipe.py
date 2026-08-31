# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-006: the deterministic model recipe reconstructs and refits the model; no pickle anywhere."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import Preprocessing
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.rc import esn, recipe, teacher_forcing, training
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig
from arm_rc_ctrl.rc.recipe import (
    DatasetSource,
    FitTolerance,
    ModelRecipe,
    RclibIdentity,
    RecipeMismatchError,
    TrainingSpec,
    create_recipe,
    load_recipe,
    write_recipe,
)
from arm_rc_ctrl.rc.training import predict_episode

ESN = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=80, spectral_radius=0.8, sparsity=0.9, leak_rate=0.3, input_scaling=0.3, seed=21
    ),
    readout=ReadoutConfig(alpha=1e-6),
)
SOURCE = DatasetSource("processed-20260830-555555555555", "ab" * 32, "data/records/processed/x.toml")
PREPROCESSING = Preprocessing(0.01, "none", {}, "central-difference")
RCLIB = RclibIdentity.current()
FOREIGN_RCLIB = RclibIdentity("0.0.0", "0" * 40)


def _samples(phase_offset: float = 0.0, n: int = 300) -> SampleSet:
    """A sinusoidal 2-DOF motion with prime/move/dwell phases (30/240/30 samples)."""
    t = np.arange(n, dtype=np.float64) * 0.01
    omega = np.array([2.0, 3.0])
    q = np.sin(omega[None, :] * t[:, None] + phase_offset) * np.array([0.5, 0.3])
    dq = omega[None, :] * np.cos(omega[None, :] * t[:, None] + phase_offset) * np.array([0.5, 0.3])
    phase = np.array([0] * 30 + [1] * 240 + [2] * (n - 270), dtype=np.int64)
    return SampleSet(t, q, dq, np.zeros((n, 2)), q * 0.1, np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 0)), phase)


def _create(tmp_path: Path) -> tuple[ModelRecipe, Path, dict[str, SampleSet]]:
    samples = {SOURCE.artifact_id: _samples()}
    normalization = fit_normalization(
        samples[SOURCE.artifact_id].arrays(),
        ("q", "dq"),
        fitted_on=(SOURCE.artifact_id,),
        training_rows=np.ones(300, dtype=np.bool_),
    )
    made, _model = create_recipe(
        "unit",
        ESN,
        sources=[SOURCE],
        samples=samples,
        dof=2,
        task_code_dim=0,
        preprocessing=PREPROCESSING,
        normalization=normalization,
        rclib=RCLIB,
    )
    file = tmp_path / "recipe.toml"
    write_recipe(file, made)
    return made, file, samples


def test_recipe_round_trips_through_toml_and_refits_exactly(tmp_path: Path) -> None:
    """The written recipe loads back equal, rebuilds the model, and its refit reproduces the fit report."""
    made, file, samples = _create(tmp_path)
    assert file.read_text().startswith("# Deterministic model recipe")
    loaded = load_recipe(file)
    assert loaded == made
    assert (loaded.input_dim, loaded.output_dim) == (4, 2)
    assert loaded.fit.episodes == (SOURCE.artifact_id,)
    assert loaded.fit.rmse < 0.05 * loaded.fit.constant_rmse
    model, report = loaded.refit(samples)
    assert report == made.fit
    assert model.fitted
    episode = loaded.episodes(samples)[0]
    _, again = loaded.refit(samples)
    assert again == report
    first_model, _ = loaded.refit(samples)
    assert np.array_equal(predict_episode(first_model, episode), predict_episode(model, episode))
    with pytest.raises(FileExistsError, match="recipes are immutable"):
        write_recipe(file, made)


def test_tampered_hyperparameters_are_detected_by_the_refit(tmp_path: Path) -> None:
    """A recipe whose seed was altered cannot reproduce its recorded fit report."""
    made, file, samples = _create(tmp_path)
    text = file.read_text()
    assert text.count("seed = 21") == 1
    tampered = tmp_path / "tampered.toml"
    tampered.write_text(text.replace("seed = 21", "seed = 22"))
    loaded = load_recipe(tampered)
    with pytest.raises(RecipeMismatchError, match="does not reproduce its fit report: rmse"):
        loaded.refit(samples)
    loose = dataclasses.replace(loaded, tolerance=FitTolerance(error_abs=1.0))
    loose.refit(samples)  # within a (deliberately absurd) tolerance the refit is accepted
    with pytest.raises(RecipeMismatchError, match="loss_rows 269 != 1"):
        dataclasses.replace(made, fit=dataclasses.replace(made.fit, loss_rows=1)).refit(samples)


def test_missing_or_mismatched_datasets_are_refused(tmp_path: Path) -> None:
    """Refitting needs samples for every referenced dataset."""
    made, _file, _samples = _create(tmp_path)
    with pytest.raises(ValueError, match="samples are missing for datasets"):
        made.refit({})
    with pytest.raises(ValueError, match="every dataset needs samples"):
        create_recipe(
            "x",
            ESN,
            sources=[SOURCE],
            samples={},
            dof=2,
            task_code_dim=0,
            preprocessing=PREPROCESSING,
            normalization=made.normalization,
            rclib=RCLIB,
        )


def test_recipe_validation() -> None:
    """Identity formats, dataset/fit consistency, widths, and the training spec are validated."""
    with pytest.raises(ValueError, match="artifact_id must be a processed artifact ID"):
        DatasetSource("raw-20260830-555555555555", "ab" * 32, "r.toml")
    with pytest.raises(ValueError, match="payload_sha256 must be 64"):
        DatasetSource("processed-20260830-555555555555", "xyz", "r.toml")
    for record in (
        "/abs.toml",
        "../outside.toml",
        "data/../x.toml",
        "./x.toml",
        "data\\records\\x.toml",
        "data//x.toml",
        "x.toml/",
        "C:/x.toml",
    ):
        with pytest.raises(ValueError, match="must be a repository-relative POSIX path"):
            DatasetSource("processed-20260830-555555555555", "ab" * 32, record)
    with pytest.raises(ValueError, match="40-hex commit"):
        RclibIdentity("1.0", "abc")
    with pytest.raises(ValueError, match="unsupported training spec"):
        TrainingSpec(target="delta_q")
    with pytest.raises(ValueError, match="error_abs must be finite and non-negative"):
        FitTolerance(-1.0)
    samples = _samples()
    normalization = fit_normalization(
        samples.arrays(), ("q", "dq"), fitted_on=(SOURCE.artifact_id,), training_rows=np.ones(300, dtype=np.bool_)
    )
    fit = training.FitReport((SOURCE.artifact_id,), 269, 30, (0.1, 0.1), 0.1, 0.5, 0.2)
    base = ModelRecipe("r", ESN, 2, 0, (SOURCE,), PREPROCESSING, normalization, TrainingSpec(), RCLIB, fit)
    with pytest.raises(ValueError, match=r"fit\.episodes .* must equal the dataset order"):
        dataclasses.replace(
            base, datasets=(dataclasses.replace(SOURCE, artifact_id="processed-20260830-666666666666"),)
        )
    with pytest.raises(ValueError, match="rmse_per_joint has 2 joints, expected 3"):
        dataclasses.replace(base, dof=3)
    with pytest.raises(ValueError, match="distinct artifacts"):
        dataclasses.replace(
            base, datasets=(SOURCE, SOURCE), fit=dataclasses.replace(fit, episodes=(SOURCE.artifact_id,) * 2)
        )
    with pytest.raises(ValueError, match="unsupported recipe schema version 2"):
        dataclasses.replace(base, schema_version=2)
    with pytest.raises(ValueError, match="name must not be empty"):
        dataclasses.replace(base, name=" ")


def test_current_rclib_identity_matches_the_pin() -> None:
    """The recipe records the pinned rclib submodule commit and its declared version."""
    identity = RclibIdentity.current()
    assert len(identity.commit) == 40
    assert identity.version


def test_no_pickle_in_the_rc_package() -> None:
    """Model artifacts are recipes; the rc package never imports pickle or related serializers."""
    for module in (esn, recipe, teacher_forcing, training):
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        for forbidden in ("import pickle", "from pickle", "import joblib", "from joblib", "import dill", "cloudpickle"):
            assert forbidden not in source, (module.__name__, forbidden)


def test_refit_requires_the_recipes_rclib_pin(tmp_path: Path) -> None:
    """A recipe made with another rclib version or commit is refused before the model is rebuilt."""
    made, _file, samples = _create(tmp_path)
    made.refit(samples)  # the checkout's pin matches
    foreign = dataclasses.replace(made, rclib=FOREIGN_RCLIB)
    with pytest.raises(RecipeMismatchError, match=r"was made with rclib 0\.0\.0 \(000000000000\) but"):
        foreign.refit(samples)
    with pytest.raises(RecipeMismatchError, match="is installed"):
        made.refit(samples, installed=FOREIGN_RCLIB)
    made.require_rclib(RclibIdentity.current())
