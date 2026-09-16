# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-004: typed ESN construction on rclib with reproducible reservoirs and a state-only readout."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError, load_config, to_mapping
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel, ReadoutConfig, ReservoirConfig, require_single_thread

CONFIG = EsnConfig(
    reservoir=ReservoirConfig(
        n_neurons=40, spectral_radius=0.9, sparsity=0.9, leak_rate=0.5, input_scaling=0.5, seed=11
    ),
    readout=ReadoutConfig(alpha=1e-3),
)
RNG = np.random.default_rng(2026)
X = RNG.standard_normal((120, 3))
Y = RNG.standard_normal((120, 2))


def _harvest(model: EsnModel, inputs: np.ndarray) -> np.ndarray:
    model.reset()
    return np.vstack([model.advance(row) for row in inputs])


def test_factory_builds_the_configured_dimensions() -> None:
    """Widths and reservoir size follow the config; the state starts at zero and has one entry per neuron."""
    model = EsnModel(CONFIG, input_dim=3, output_dim=2)
    assert (model.input_dim, model.output_dim, model.n_neurons) == (3, 2, 40)
    assert model.config == CONFIG
    assert model.fitted is False
    assert np.array_equal(model.state(), np.zeros(40))
    state = model.advance(X[0])
    assert state.shape == (40,)
    assert np.all(np.isfinite(state))
    assert np.array_equal(model.state(), state)
    model.reset()
    assert not model.state().any()


def test_same_config_is_bitwise_reproducible_in_process() -> None:
    """Two models built in one interpreter from one config evolve identically (UP-005 guard)."""
    first = _harvest(EsnModel(CONFIG, input_dim=3, output_dim=2), X)
    second = _harvest(EsnModel(CONFIG, input_dim=3, output_dim=2), X)
    assert np.array_equal(first, second)
    other = dataclasses.replace(CONFIG, reservoir=dataclasses.replace(CONFIG.reservoir, seed=12))
    assert not np.array_equal(first, _harvest(EsnModel(other, input_dim=3, output_dim=2), X))


def test_readout_on_harvested_states_matches_rclib_sequence_fit() -> None:
    """Fitting the readout on states harvested sample by sample equals rclib's single-sequence fit exactly."""
    washout = 15
    reference = EsnModel(CONFIG, input_dim=3, output_dim=2)
    reference.fit_sequence(X, Y, washout_len=washout)
    expected = reference.predict_sequence(X)

    model = EsnModel(CONFIG, input_dim=3, output_dim=2)
    states = _harvest(model, X)
    model.fit_readout(states[washout:], Y[washout:])
    assert model.fitted is True
    assert np.array_equal(model.predict_sequence(X), expected)
    # the readout sees reservoir states only (rclib convention); a fitted model reproduces its training targets roughly
    residual = model.predict_sequence(X)[washout:] - Y[washout:]
    assert np.sqrt(np.mean(residual**2)) < np.std(Y)


def test_online_step_matches_batch_prediction() -> None:
    """Advance + readout per sample equals the batch teacher-forced prediction within numerical tolerance."""
    model = EsnModel(CONFIG, input_dim=3, output_dim=2)
    model.fit_sequence(X, Y, washout_len=10)
    batch = model.predict_sequence(X)
    model.reset()
    online = np.vstack([model.step(row) for row in X])
    assert online.shape == (120, 2)
    assert np.allclose(online, batch, atol=1e-12, rtol=0)


def test_unfitted_and_malformed_use_is_rejected() -> None:
    """Reading out before fitting, wrong widths, and non-finite inputs are errors."""
    model = EsnModel(CONFIG, input_dim=3, output_dim=2)
    with pytest.raises(RuntimeError, match="readout has not been fitted"):
        model.step(X[0])
    with pytest.raises(RuntimeError, match="readout has not been fitted"):
        model.predict_sequence(X)
    with pytest.raises(ValueError, match=r"input must have shape \(3,\), got \(2,\)"):
        model.advance(X[0, :2])
    with pytest.raises(ValueError, match="input must be finite"):
        model.advance(np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(ValueError, match=r"states must have shape \(M, 40\)"):
        model.fit_readout(np.zeros((5, 3)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match=r"targets must have shape \(5, 2\)"):
        model.fit_readout(np.zeros((5, 40)), np.zeros((5, 1)))
    with pytest.raises(ValueError, match="non-empty and finite"):
        model.fit_readout(np.full((5, 40), np.inf), np.zeros((5, 2)))
    with pytest.raises(ValueError, match=r"washout_len must be in \[0, 120\)"):
        model.fit_sequence(X, Y, washout_len=120)
    with pytest.raises(ValueError, match=r"inputs must be \(M, 3\)"):
        model.fit_sequence(X[:, :2], Y, washout_len=1)
    model.fit_sequence(X, Y, washout_len=1)
    with pytest.raises(ValueError, match=r"state must have shape \(40,\)"):
        model.readout(np.zeros(3))
    with pytest.raises(ValueError, match=r"inputs must have shape \(M, 3\)"):
        model.predict_sequence(X[:, :2])
    with pytest.raises(ValueError, match="input_dim and output_dim must be >= 1"):
        EsnModel(CONFIG, input_dim=0, output_dim=2)


@pytest.mark.parametrize(
    ("section", "overrides", "message"),
    [
        ("reservoir", {"n_neurons": 0}, "n_neurons must be positive"),
        ("reservoir", {"spectral_radius": -0.1}, "spectral_radius must be non-negative"),
        ("reservoir", {"sparsity": 1.5}, r"sparsity must be in \[0, 1\]"),
        ("reservoir", {"leak_rate": 0.0}, r"leak_rate must be in \(0, 1\]"),
        ("reservoir", {"input_scaling": -1.0}, "input_scaling must be non-negative"),
        ("reservoir", {"seed": -1}, "seed must be non-negative"),
        ("reservoir", {"leak_rate": float("nan")}, "reservoir"),
        ("readout", {"alpha": -1e-3}, "alpha must be non-negative"),
        ("readout", {"solver": "svd"}, "solver must be one of"),
        ("readout", {"tolerance": 0.0}, "tolerance must be positive"),
    ],
)
def test_config_constraints(section: str, overrides: dict[str, object], message: str) -> None:
    """Hyperparameter constraints mirror rclib and name the offending key."""
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(getattr(CONFIG, section), **overrides)


def test_config_round_trips_through_toml(tmp_path: Path) -> None:
    """The typed config loads from TOML with rclib's defaults filled in and unknown keys rejected."""
    file = tmp_path / "esn.toml"
    file.write_text(
        "[reservoir]\nn_neurons = 40\nspectral_radius = 0.9\nsparsity = 0.9\nleak_rate = 0.5\n"
        "input_scaling = 0.5\nseed = 11\n[readout]\nalpha = 1e-3\n"
    )
    loaded = load_config(file, EsnConfig)
    assert loaded == CONFIG
    assert loaded.readout.solver == "cholesky"
    assert to_mapping(loaded)["readout"] == {
        "alpha": 1e-3,
        "solver": "cholesky",
        "include_bias": True,
        "tolerance": 1e-10,
        "explicit_bias": None,  # absent means the historical implicit bias; stored records strip it
    }
    file.write_text(file.read_text() + "washout = 3\n")
    with pytest.raises(ConfigError):
        load_config(file, EsnConfig)


def test_single_thread_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a model requires OMP_NUM_THREADS=1 for bitwise reproducible reductions."""
    require_single_thread()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS=1"):
        EsnModel(CONFIG, input_dim=3, output_dim=2)


def test_readout_weights_are_the_solver_layout_and_a_copy() -> None:
    """The weights reproduce the readout with the bias row last, are refused before fit, and are a copy (UP-007)."""
    config = EsnConfig(
        reservoir=ReservoirConfig(
            n_neurons=20, spectral_radius=0.9, sparsity=0.8, leak_rate=0.5, input_scaling=0.5, seed=11
        ),
        readout=ReadoutConfig(alpha=1e-3),
    )
    model = EsnModel(config, input_dim=3, output_dim=2)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.readout_weights()
    rng = np.random.default_rng(3)
    states = rng.standard_normal((50, 20))
    targets = rng.standard_normal((50, 2))
    model.fit_readout(states, targets)
    weights = model.readout_weights()
    assert weights.shape == (21, 2)
    state = rng.standard_normal(20)
    assert np.allclose(model.readout(state), state @ weights[:-1] + weights[-1], atol=1e-12, rtol=0.0)
    weights[:] = 0.0
    assert np.allclose(model.readout(state), model.readout_weights()[:-1].T @ state + model.readout_weights()[-1])


EXPLICIT_BIAS = EsnConfig(
    reservoir=CONFIG.reservoir, readout=ReadoutConfig(alpha=1e-3, include_bias=False, explicit_bias=True)
)


def test_explicit_bias_requires_the_library_bias_to_be_disabled() -> None:
    """A configuration that would fit two bias terms is refused, naming both keys (M3MAN-005, I3)."""
    with pytest.raises(ValueError, match=r"readout\.explicit_bias requires readout\.include_bias"):
        ReadoutConfig(alpha=1e-3, include_bias=True, explicit_bias=True)
    assert ReadoutConfig(alpha=1e-3).explicit_bias is None  # the historical implicit-bias layout is the default


def test_explicit_bias_prediction_paths_agree() -> None:
    """Batch prediction equals stepping bitwise, and the extracted weights reproduce the readout with the bias last."""
    model = EsnModel(EXPLICIT_BIAS, input_dim=3, output_dim=2)
    states = _harvest(model, X)
    model.fit_readout(states[15:], Y[15:])
    weights = model.readout_weights()
    assert weights.shape == (41, 2)  # n_neurons + 1 with include_bias False and explicit_bias True
    batch = model.predict_sequence(X)
    model.reset()
    online = np.vstack([model.step(row) for row in X])
    assert np.array_equal(online, batch)  # the explicit-bias batch path is reset + advance + readout
    model.reset()
    state = model.advance(X[0])
    assert np.allclose(model.readout(state), state @ weights[:-1] + weights[-1], atol=1e-12, rtol=0.0)
    residual = batch[15:] - Y[15:]
    assert np.sqrt(np.mean(residual**2)) < np.std(Y)


def test_explicit_bias_widths_and_weight_use_are_validated() -> None:
    """The state width is checked against the raw harvest, and row weights are refused on the implicit layout."""
    model = EsnModel(EXPLICIT_BIAS, input_dim=3, output_dim=2)
    with pytest.raises(ValueError, match=r"states must have shape \(M, 40\)"):
        model.fit_readout(np.zeros((5, 41)), np.zeros((5, 2)))
    implicit = EsnModel(CONFIG, input_dim=3, output_dim=2)
    with pytest.raises(ValueError, match=r"per-sample weights require readout\.explicit_bias"):
        implicit.fit_readout(np.zeros((5, 40)), np.zeros((5, 2)), weights=np.ones(5))
    implicit.fit_readout(np.zeros((5, 40)), np.zeros((5, 2)))
    assert implicit.readout_weights().shape == (41, 2)  # the historical implicit bias row is still last
