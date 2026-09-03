# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-012: one search protocol per generator formulation; inapplicable parameters are absent."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import optuna
import pytest

from arm_rc_ctrl.experiments.recovery_search import (
    RECOVERY_TRACKERS,
    check_matched_protocols,
    load_recovery_search,
    point_from_params,
    recovery_protocol_digest,
    suggest_recovery_point,
    training_spec_for,
)
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()


def _fixed(params: dict[str, float | int]) -> optuna.Trial:
    """A FixedTrial widened to the Trial interface (Optuna's documented testing pattern)."""
    return cast("optuna.Trial", optuna.trial.FixedTrial(params))


_BASE = """
name = "{name}"
scenario = "{scenario}"
model = "{model}"
formulation = "{formulation}"
budget = 12
seed_bank = 1
attempt_factor = 4
development = "{development}"

[sampler]
kind = "tpe"
seed = {sampler_seed}
n_startup_trials = 4

[pruner]
kind = "none"

[objective]
kind = "worst_cell_median_gap_ratio"
infeasible_penalty = 10.0

[esn]
n_neurons = {{ low = 50, high = 200, step = 50 }}
spectral_radius = {{ low = 0.8, high = 1.3 }}
sparsity = {{ low = 0.5, high = 0.98 }}
leak_rate = {{ low = 0.01, high = 0.3, log = true }}
input_scaling = {{ low = 0.02, high = 0.5, log = true }}
seed = {{ low = 1, high = 1000 }}
alpha = {{ low = 1e-3, high = 1.0, log = true }}
velocity_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}
acceleration_cutoff_hz = {{ low = 5.0, high = 30.0, log = true }}

[space]
warmups_s = [0.0, 0.25, 1.0]
{augmentation}
"""

_AUGMENTATION = """n_synthetic = [16, 32]
sigma_rad = [0.025, 0.05]
phi = [0.98, 0.99]
gamma = [0.5, 1.0]
"""


def _write(
    directory: Path,
    *,
    name: str = "recovery-search-test",
    formulation: str = "contractive",
    augmented: bool = True,
    sampler_seed: int = 77,
) -> Path:
    scenario = (REPO_ROOT / "configs" / "tasks" / "task_1a.toml").as_posix()
    model = (REPO_ROOT / "configs" / "models" / "esn_task_1a_v4.toml").as_posix()
    development = (REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml").as_posix()
    file = directory / f"{name}.toml"
    file.write_text(
        _BASE.format(
            name=name,
            scenario=scenario,
            model=model,
            formulation=formulation,
            development=development,
            sampler_seed=sampler_seed,
            augmentation=_AUGMENTATION if augmented else "",
        ),
        encoding="utf-8",
    )
    return file


def test_recovery_trackers_are_fixed_not_searched() -> None:
    """Both frozen trackers are evaluated per trial; the tracker is never an Optuna parameter."""
    assert RECOVERY_TRACKERS == ("pd_v2", "computed_torque")


def test_augmented_formulations_require_the_approved_grids(tmp_path: Path) -> None:
    """Off-grid augmentation values and missing sections are rejected."""
    protocol = load_recovery_search(_write(tmp_path))
    assert protocol.formulation == "contractive"
    assert protocol.space.augmentation is not None
    bad = _write(tmp_path, name="bad-grid")
    bad.write_text(bad.read_text(encoding="utf-8").replace("sigma_rad = [0.025, 0.05]", "sigma_rad = [0.2]"))
    with pytest.raises(ValueError, match="approved"):
        load_recovery_search(bad)
    missing = _write(tmp_path, name="missing-aug", augmented=False)
    with pytest.raises(ValueError, match="augmentation"):
        load_recovery_search(missing)


def test_no_augmentation_forbids_the_augmentation_section(tmp_path: Path) -> None:
    """Inapplicable parameters are absent, never dummy-filled."""
    protocol = load_recovery_search(_write(tmp_path, name="noaug", formulation="no_augmentation", augmented=False))
    assert protocol.space.augmentation is None
    with pytest.raises(ValueError, match="no_augmentation"):
        load_recovery_search(_write(tmp_path, name="noaug-bad", formulation="no_augmentation", augmented=True))


def test_suggested_points_carry_only_applicable_parameters(tmp_path: Path) -> None:
    """The no-augmentation study samples ESN + warm-up only; augmented studies add the D1 grid."""
    augmented = load_recovery_search(_write(tmp_path))
    plain = load_recovery_search(_write(tmp_path, name="noaug", formulation="no_augmentation", augmented=False))
    fixed_common = {
        "n_neurons": 100,
        "spectral_radius": 0.9,
        "sparsity": 0.9,
        "leak_rate": 0.1,
        "input_scaling": 0.1,
        "seed": 5,
        "alpha": 1e-2,
        "velocity_cutoff_hz": 20.0,
        "acceleration_cutoff_hz": 20.0,
        "warmup_s": 1.0,
    }
    fixed_augmented = {**fixed_common, "n_synthetic": 16, "sigma_rad": 0.05, "phi": 0.99, "gamma": 1.0}
    point = suggest_recovery_point(augmented, _fixed(fixed_augmented))
    assert point.warmup_s == 1.0
    assert point.augmentation is not None
    assert point.augmentation.n_synthetic == 16
    params = point.params()
    assert {"n_synthetic", "sigma_rad", "phi", "gamma"} <= set(params)
    plain_point = suggest_recovery_point(plain, _fixed(dict(fixed_common)))
    assert plain_point.augmentation is None
    assert not {"n_synthetic", "sigma_rad", "phi", "gamma"} & set(plain_point.params())
    assert "warmup_s" in plain_point.params()


def test_point_round_trips_and_rejects_off_space_values(tmp_path: Path) -> None:
    """Stored parameters rebuild the point exactly; values off the protocol space are refused."""
    protocol = load_recovery_search(_write(tmp_path))
    fixed = {
        "n_neurons": 150,
        "spectral_radius": 1.0,
        "sparsity": 0.9,
        "leak_rate": 0.05,
        "input_scaling": 0.1,
        "seed": 42,
        "alpha": 1e-2,
        "velocity_cutoff_hz": 15.0,
        "acceleration_cutoff_hz": 12.0,
        "warmup_s": 0.25,
        "n_synthetic": 32,
        "sigma_rad": 0.025,
        "phi": 0.98,
        "gamma": 0.5,
    }
    point = suggest_recovery_point(protocol, _fixed(fixed))
    rebuilt = point_from_params(protocol, {k: float(v) for k, v in point.params().items()})
    assert rebuilt == point
    off = {k: float(v) for k, v in point.params().items()}
    off["warmup_s"] = 2.0  # approved globally but outside this protocol's searched subset
    with pytest.raises(ValueError, match="warmup"):
        point_from_params(protocol, off)


def test_training_spec_binds_formulation_family_and_bank(tmp_path: Path) -> None:
    """The trial's training spec carries the warm-up, the formulation's family, and the shared bank."""
    protocol = load_recovery_search(_write(tmp_path))
    fixed = {
        "n_neurons": 100,
        "spectral_radius": 0.9,
        "sparsity": 0.9,
        "leak_rate": 0.1,
        "input_scaling": 0.1,
        "seed": 5,
        "alpha": 1e-2,
        "velocity_cutoff_hz": 20.0,
        "acceleration_cutoff_hz": 20.0,
        "warmup_s": 1.0,
        "n_synthetic": 16,
        "sigma_rad": 0.05,
        "phi": 0.99,
        "gamma": 1.0,
    }
    point = suggest_recovery_point(protocol, _fixed(fixed))
    spec = training_spec_for(protocol, point)
    assert spec.washout == "warmup_hold"
    assert spec.warmup_s == 1.0
    assert spec.augmentation is not None
    assert spec.augmentation.family == "contractive"
    assert spec.augmentation.seed_bank == protocol.seed_bank
    assert spec.augmentation.attempt_budget == protocol.attempt_factor * 16
    plain = load_recovery_search(_write(tmp_path, name="noaug", formulation="no_augmentation", augmented=False))
    plain_fixed = {k: v for k, v in fixed.items() if k not in ("n_synthetic", "sigma_rad", "phi", "gamma")}
    plain_spec = training_spec_for(plain, suggest_recovery_point(plain, _fixed(plain_fixed)))
    assert plain_spec.augmentation is None


def test_matched_protocols_share_counts_and_banks(tmp_path: Path) -> None:
    """The three formulation studies keep identical trial counts; the augmented pair shares its bank."""
    contractive = load_recovery_search(_write(tmp_path, name="a", formulation="contractive"))
    non_decaying = load_recovery_search(_write(tmp_path, name="b", formulation="non_decaying", sampler_seed=78))
    check_matched_protocols(contractive, non_decaying)
    mismatched_text = _write(tmp_path, name="c", formulation="non_decaying").read_text(encoding="utf-8")
    mismatched = tmp_path / "c.toml"
    mismatched.write_text(mismatched_text.replace("budget = 12", "budget = 13"), encoding="utf-8")
    with pytest.raises(ValueError, match="budget"):
        check_matched_protocols(contractive, load_recovery_search(mismatched))
    assert recovery_protocol_digest(contractive) != recovery_protocol_digest(non_decaying)
