# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3R-012: the three committed recovery formulation studies are matched, disjoint, and development-only."""

from __future__ import annotations

import tomllib
from itertools import combinations
from typing import cast

import pytest

from arm_rc_ctrl.experiments.esn_search import TrialPoint, load_esn_search
from arm_rc_ctrl.experiments.perturbations import load_development_robustness
from arm_rc_ctrl.experiments.recovery_search import (
    AugmentationPoint,
    RecoverySearchProtocol,
    check_matched_protocols,
    load_recovery_search,
    recovery_protocol_digest,
)
from arm_rc_ctrl.experiments.tuning import load_protocol as load_tuning_protocol
from arm_rc_ctrl.rc.augment import (
    APPROVED_GAMMA,
    APPROVED_N_SYNTHETIC,
    APPROVED_PHI,
    APPROVED_SIGMA_RAD,
    SEED_NAMESPACE,
)
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.rc.warmup import APPROVED_WARMUPS_S
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
STUDIES = REPO_ROOT / "configs" / "studies"
FILES = {
    "no_augmentation": STUDIES / "recovery_search_1a_no_augmentation_v1.toml",
    "non_decaying": STUDIES / "recovery_search_1a_non_decaying_v1.toml",
    "contractive": STUDIES / "recovery_search_1a_contractive_v1.toml",
}
DEVELOPMENT = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
MODEL_V4 = REPO_ROOT / "configs" / "models" / "esn_task_1a_v4.toml"
NOMINAL_V4 = REPO_ROOT / "configs" / "evaluations" / "task_1a_nominal_v4.toml"
M3_CONFIRMATORY_SEEDS = frozenset({20260901, 20260902, 20260903, 20260904, 20260905})
M3_DEVELOPMENT_FILES = (
    REPO_ROOT / "configs" / "evaluations" / "task_1a_robustness_dev_v1.toml",
    REPO_ROOT / "configs" / "evaluations" / "task_1a_robustness_dev_v2.toml",
)
M3_STUDY_FILES = (
    STUDIES / "baseline_gains_1a.toml",
    STUDIES / "baseline_gains_1a_v2.toml",
)
M3_ESN_FILES = (
    STUDIES / "esn_search_1a.toml",
    STUDIES / "esn_search_1a_v2.toml",
)


def _protocols() -> dict[str, RecoverySearchProtocol]:
    return {formulation: load_recovery_search(file) for formulation, file in FILES.items()}


def _find_key(mapping: object, key: str) -> float:
    """The single value of ``key`` anywhere in a nested TOML mapping."""
    hits: list[float] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for name, value in cast("dict[str, object]", node).items():
                if name == key and isinstance(value, (int, float)):
                    hits.append(float(value))
                else:
                    walk(value)

    walk(mapping)
    assert len(hits) == 1, f"{key}: {hits}"
    return hits[0]


def test_the_three_studies_are_matched() -> None:
    """Identical budgets, spaces, objectives, feasibility, pruners, and start-up counts; one shared bank."""
    protocols = _protocols()
    for first, second in combinations(protocols.values(), 2):
        check_matched_protocols(first, second)
        assert first.pruner == second.pruner
        assert first.sampler.kind == second.sampler.kind
        assert first.sampler.n_startup_trials == second.sampler.n_startup_trials
    assert len({p.budget for p in protocols.values()}) == 1
    assert len({p.seed_bank for p in protocols.values()}) == 1
    assert len({p.attempt_factor for p in protocols.values()}) == 1


def test_formulations_names_and_digests_identify_each_arm() -> None:
    """Each file declares its formulation, carries it in the study name, and hashes distinctly."""
    protocols = _protocols()
    digests: set[str] = set()
    for formulation, protocol in protocols.items():
        assert protocol.formulation == formulation
        assert formulation.replace("_", "-") in protocol.name
        digests.add(recovery_protocol_digest(protocol))
    assert len(digests) == 3


def test_search_spaces_cover_the_approved_ranges() -> None:
    """Warm-ups cover the full approved D2 set; augmented grids cover the full approved D1 sets."""
    protocols = _protocols()
    for protocol in protocols.values():
        assert set(protocol.space.warmups_s) == APPROVED_WARMUPS_S
    assert protocols["no_augmentation"].space.augmentation is None
    for formulation in ("non_decaying", "contractive"):
        grid = protocols[formulation].space.augmentation
        assert grid is not None
        assert set(grid.n_synthetic) == APPROVED_N_SYNTHETIC
        assert set(grid.sigma_rad) == APPROVED_SIGMA_RAD
        assert set(grid.phi) == APPROVED_PHI
        assert set(grid.gamma) == APPROVED_GAMMA


def test_esn_bounds_equal_the_m3_search_v2_space() -> None:
    """The recovery studies search the same reservoir space the frozen v4 point came from."""
    v2 = load_esn_search(STUDIES / "esn_search_1a_v2.toml")
    for protocol in _protocols().values():
        assert protocol.esn == v2.search


def test_sampler_seeds_are_a_new_disjoint_namespace() -> None:
    """The three sampler seeds are distinct and disjoint from every recorded M3 and recovery seed."""
    seeds = {p.sampler.seed for p in _protocols().values()}
    assert len(seeds) == 3
    used: set[int] = set(M3_CONFIRMATORY_SEEDS)
    used.add(SEED_NAMESPACE)
    for file in M3_DEVELOPMENT_FILES:
        used |= set(load_development_robustness(file).seeds)
    for file in M3_STUDY_FILES:
        used.add(load_tuning_protocol(file).sampler_seed)
    for file in M3_ESN_FILES:
        used.add(load_esn_search(file).sampler.seed)
    used |= set(load_development_robustness(DEVELOPMENT).seeds)
    with (REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_confirmatory_v1.toml").open("rb") as handle:
        used |= {int(seed) for seed in tomllib.load(handle)["seeds"]}
    assert not seeds & used


def test_development_levels_are_the_locked_split_never_the_confirmatory_lock() -> None:
    """Every study points at the locked development levels; the confirmatory file is out of reach."""
    locked = load_development_robustness(DEVELOPMENT)
    for protocol in _protocols().values():
        assert protocol.development.name == "task_1a_recovery_dev_v1.toml"
        assert "confirmatory" not in protocol.development.name
        assert load_development_robustness(protocol.development) == locked


def test_anchors_are_the_paired_v4_point() -> None:
    """One anchor per study: the frozen v4 reservoir/readout with its selected cutoffs at T_w = 1 s.

    The augmented anchors add the approved D1 anchor (64, 0.05 rad, gamma 1; phi 0.99 as the
    mid-grid choice for the dimension D1 leaves open) and are identical in both augmented studies.
    """
    model = load_model_config(MODEL_V4)
    with NOMINAL_V4.open("rb") as handle:
        nominal = tomllib.load(handle)
    expected_esn = TrialPoint(
        n_neurons=model.esn.reservoir.n_neurons,
        spectral_radius=model.esn.reservoir.spectral_radius,
        sparsity=model.esn.reservoir.sparsity,
        leak_rate=model.esn.reservoir.leak_rate,
        input_scaling=model.esn.reservoir.input_scaling,
        seed=model.esn.reservoir.seed,
        alpha=model.esn.readout.alpha,
        velocity_cutoff_hz=_find_key(nominal, "velocity_cutoff_hz"),
        acceleration_cutoff_hz=_find_key(nominal, "acceleration_cutoff_hz"),
    )
    protocols = _protocols()
    for formulation, protocol in protocols.items():
        (anchor,) = protocol.comparison
        assert anchor.label == "anchor-v4-tw1"
        assert anchor.point.esn == expected_esn
        assert anchor.point.warmup_s == 1.0
        if formulation == "no_augmentation":
            assert anchor.point.augmentation is None
        else:
            assert anchor.point.augmentation == AugmentationPoint(n_synthetic=64, sigma_rad=0.05, phi=0.99, gamma=1.0)
    augmented = [protocols["non_decaying"].comparison, protocols["contractive"].comparison]
    assert augmented[0] == augmented[1]
