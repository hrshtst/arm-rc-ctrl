# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The committed ESN search protocol: planned parameters, bounds, seeds, budget, and separation (M3-003)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from arm_rc_ctrl.experiments.baselines import baseline_method
from arm_rc_ctrl.experiments.confirmatory import load_confirmatory
from arm_rc_ctrl.experiments.esn_search import PLANNED_PARAMETERS, EsnSearchSpace, load_esn_search
from arm_rc_ctrl.experiments.tuning import load_protocol as load_tuning_protocol
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml"
CONFIRMATORY = REPO_ROOT / "configs" / "evaluations" / "task_1a_confirmatory_v2.toml"
GAIN_STUDIES = (
    REPO_ROOT / "configs" / "studies" / "baseline_gains_1a.toml",
    REPO_ROOT / "configs" / "studies" / "baseline_gains_1a_v2.toml",
)
ALPHA_REQUIRED = (3e-2, 3e-1)
"""Ridge regularization the search must cover at least (owner instruction at the M2 review)."""
ALPHA_REFERENCE = 1e-2
"""The M2 ridge value that stays a comparison point."""


def test_protocol_defines_every_planned_parameter_with_bounds() -> None:
    """The search space holds exactly the planned parameters, each with explicit bounds."""
    protocol = load_esn_search(PROTOCOL)
    assert tuple(f.name for f in dataclasses.fields(EsnSearchSpace)) == PLANNED_PARAMETERS
    space = protocol.search
    assert space.alpha.low <= ALPHA_REQUIRED[0]
    assert space.alpha.high >= ALPHA_REQUIRED[1]
    assert space.alpha.low <= ALPHA_REFERENCE
    assert space.alpha.log
    assert space.n_neurons.low < 200 < space.n_neurons.high  # the development anchor is interior
    assert space.spectral_radius.low < 0.9 < space.spectral_radius.high
    assert space.leak_rate.low < 0.3 < space.leak_rate.high
    assert space.input_scaling.low < 0.5 < space.input_scaling.high
    assert space.velocity_cutoff_hz.low < 20.0 < space.velocity_cutoff_hz.high


def test_comparison_points_keep_the_anchor_and_the_reference_alpha() -> None:
    """The development anchor is evaluated at 1e-2 (reference) and across the required alpha range."""
    protocol = load_esn_search(PROTOCOL)
    base = load_model_config(protocol.model)
    assert protocol.model.name == "esn_task_1a_v2.toml"
    alphas = sorted(c.point.alpha for c in protocol.comparison)
    assert ALPHA_REFERENCE in alphas
    assert alphas[0] <= ALPHA_REQUIRED[0]
    assert alphas[-1] >= ALPHA_REQUIRED[1]
    for comparison in protocol.comparison:
        point = comparison.point
        reservoir = base.esn.reservoir
        assert (point.n_neurons, point.spectral_radius, point.sparsity) == (
            reservoir.n_neurons,
            reservoir.spectral_radius,
            reservoir.sparsity,
        )
        assert (point.leak_rate, point.input_scaling, point.seed) == (
            reservoir.leak_rate,
            reservoir.input_scaling,
            reservoir.seed,
        )
        assert (point.velocity_cutoff_hz, point.acceleration_cutoff_hz) == (20.0, 20.0)
    assert any(math.isclose(c.point.alpha, base.esn.readout.alpha) for c in protocol.comparison)


def test_budget_seed_sampler_and_tracker_are_fixed() -> None:
    """Budget covers the comparison points and the random start-up; the tracker is a frozen registry name."""
    protocol = load_esn_search(PROTOCOL)
    assert protocol.budget >= len(protocol.comparison) + protocol.sampler.n_startup_trials + 1
    assert protocol.sampler.kind == "tpe"
    assert protocol.pruner.kind == "median"
    assert baseline_method(protocol.tracker) == "pd"
    assert protocol.tracker == "pd_v2"
    assert protocol.feasibility.max_saturation_fraction < 1.0
    assert protocol.objective.kind == "median_move_joint_rmse"


def test_no_confirmatory_seed_level_timing_or_direction_is_used() -> None:
    """Development-only: seeds and scenarios are disjoint from the locked confirmatory protocol."""
    protocol = load_esn_search(PROTOCOL)
    confirmatory = load_confirmatory(CONFIRMATORY)
    confirmatory.forbid_seeds([protocol.sampler.seed], "the ESN search")
    gain_seeds = {load_tuning_protocol(path).sampler_seed for path in GAIN_STUDIES}
    assert protocol.sampler.seed not in gain_seeds
    assert protocol.search.seed.high < min(confirmatory.seeds)  # reservoir seeds cannot collide either
    norms = [float(np.linalg.norm(offset)) for offset in protocol.development.initial_posture_offsets]
    for level in (confirmatory.posture.small_magnitude_rad, confirmatory.posture.large_magnitude_rad):
        assert not any(math.isclose(norm, level, rel_tol=1e-9) for norm in norms)
    assert protocol.development.nominal_first
    for pulse in protocol.development.force_pulses:
        assert pulse.start_s != confirmatory.force.start_s
        assert pulse.direction_deg not in confirmatory.force.directions_deg
        assert pulse.magnitude_n < confirmatory.force.magnitude_n
