# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: projecting the study's cost from what a smoke check measured.

The projection multiplies the study's own counts by measured means. Those
counts are derived from the frozen study rather than written down here: six
configurations of 31 arms are 186 models, and after the configuration-matched
cutoffs decision each of the six configurations has its own bank per parent, so
the replay side is 60 banks rather than the three warm-up banks the repeated
demonstration pilot had. Getting that divisor wrong would understate the replay
budget fivefold, so it is asserted against the frozen constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.manual_evaluation import ManualModelTiming, ManualRunTiming
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_study import ARM_COUNT, CONFIGURATION_COUNT, MODEL_COUNT
from arm_rc_ctrl.experiments.manual_timing import (
    ManualStudyProjection,
    project_study,
    summarize_timings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

PAIRS_PER_MODEL = 130
"""65 locked development cases under both frozen trackers."""


def _run(arm: str, *, simulate: float, persist: float, run_bytes: int, index: int = 0) -> ManualRunTiming:
    return ManualRunTiming(
        arm=arm,
        scenario_id=f"case-{index}",
        tracker="pd_v2",
        rows=3001,
        simulate_seconds=simulate,
        persist_seconds=persist,
        run_bytes=run_bytes,
    )


def _measured() -> tuple[ManualRunTiming, ...]:
    """Two runs per arm, with means that are easy to check by hand."""
    return (
        _run("rc", simulate=1.0, persist=0.2, run_bytes=1000, index=0),
        _run("rc", simulate=2.0, persist=0.4, run_bytes=2000, index=1),
        _run("replay", simulate=0.5, persist=0.1, run_bytes=400, index=2),
        _run("replay", simulate=1.5, persist=0.3, run_bytes=600, index=3),
    )


def _models(count: int = 2, *, fit_seconds: float = 10.0) -> tuple[ManualModelTiming, ...]:
    return tuple(
        ManualModelTiming(
            label=f"feasible-best/S/D{index + 1:02d}",
            fit_seconds=fit_seconds,
            fit_cache_hit=False,
            sweep_seconds=60.0,
            runs=PAIRS_PER_MODEL,
        )
        for index in range(count)
    )


def _projection(runs: Sequence[ManualRunTiming] | None = None, *, completed: int = 2) -> ManualStudyProjection:
    return project_study(
        _models(),
        _measured() if runs is None else runs,
        pairs_per_model=PAIRS_PER_MODEL,
        completed_models=completed,
    )


# --- the counts come from the frozen study -----------------------------------------------------


def test_the_projection_follows_the_frozen_study() -> None:
    """186 models and 60 banks over 130 pairs: 24,180 RC runs beside 7,800 replay runs, 31,980 in total."""
    p = _projection()
    assert (p.configurations, p.arms, p.models) == (CONFIGURATION_COUNT, ARM_COUNT, MODEL_COUNT)
    assert p.parents == len(ASSIGNMENTS)
    assert p.replay_banks == CONFIGURATION_COUNT * len(ASSIGNMENTS) == 60
    assert p.rc_runs == MODEL_COUNT * PAIRS_PER_MODEL == 24_180
    assert p.replay_runs == p.replay_banks * PAIRS_PER_MODEL == 7_800
    assert p.total_runs == p.rc_runs + p.replay_runs == 31_980


def test_the_projection_scales_the_measured_means() -> None:
    """Each count is multiplied by the mean cost measured for its arm, plus one fit per model."""
    p = _projection()
    assert p.rc_run_seconds == pytest.approx((1.2 + 2.4) / 2)
    assert p.replay_run_seconds == pytest.approx((0.6 + 1.8) / 2)
    assert p.fit_seconds == pytest.approx(MODEL_COUNT * 10.0)
    expected = p.rc_runs * p.rc_run_seconds + p.replay_runs * p.replay_run_seconds + p.fit_seconds
    assert p.total_seconds == pytest.approx(expected)
    assert p.storage_bytes == int(p.rc_runs * 1500.0 + p.replay_runs * 500.0)


def test_what_this_check_already_completed_is_not_projected_again() -> None:
    """The remaining estimate is what is left to do, so a resumed full run is not quoted the whole cost."""
    p = _projection(completed=2)
    assert 0 < p.remaining_seconds < p.total_seconds


def test_an_arm_that_was_not_measured_contributes_nothing() -> None:
    """A smoke check that built no replay bank still projects the RC side rather than dividing by zero."""
    rc_only = tuple(r for r in _measured() if r.arm == "rc")
    p = _projection(rc_only)
    assert p.replay_run_seconds == 0.0
    assert p.rc_run_seconds > 0.0
    assert p.total_seconds > 0.0


# --- per-arm statistics ------------------------------------------------------------------------


def test_summarize_reports_each_measured_arm() -> None:
    """Mean, median and worst case per arm, so a projection built on means can be judged."""
    stats = {s.arm: s for s in summarize_timings(_measured())}
    assert set(stats) == {"rc", "replay"}
    assert stats["rc"].runs == 2
    assert stats["rc"].mean_simulate_s == pytest.approx(1.5)
    assert stats["rc"].max_simulate_s == pytest.approx(2.0)
    assert stats["rc"].mean_persist_s == pytest.approx(0.3)
    assert stats["rc"].mean_bytes == pytest.approx(1500.0)
    assert summarize_timings(()) == ()


# --- the guards on the projection itself -------------------------------------------------------


def _valid_projection() -> dict[str, object]:
    """Keyword arguments of a consistent projection, for mutating one field at a time."""
    return {
        "configurations": CONFIGURATION_COUNT,
        "arms": ARM_COUNT,
        "models": MODEL_COUNT,
        "pairs_per_model": PAIRS_PER_MODEL,
        "parents": len(ASSIGNMENTS),
        "replay_banks": CONFIGURATION_COUNT * len(ASSIGNMENTS),
        "rc_runs": MODEL_COUNT * PAIRS_PER_MODEL,
        "replay_runs": CONFIGURATION_COUNT * len(ASSIGNMENTS) * PAIRS_PER_MODEL,
        "total_runs": MODEL_COUNT * PAIRS_PER_MODEL + CONFIGURATION_COUNT * len(ASSIGNMENTS) * PAIRS_PER_MODEL,
        "rc_run_seconds": 1.0,
        "replay_run_seconds": 0.5,
        "fit_seconds": 10.0,
        "total_seconds": 100.0,
        "storage_bytes": 1000,
        "completed_models": 0,
        "remaining_seconds": 50.0,
    }


def test_a_consistent_projection_is_accepted() -> None:
    """The fixture these guard cases mutate is itself valid, or they would prove nothing."""
    assert ManualStudyProjection(**_valid_projection()).total_runs == 31_980  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("models", 185, "models is not"),
        ("replay_banks", 59, "banks is not"),
        ("total_runs", 31_979, "runs is not"),
        ("remaining_seconds", -1.0, "no negative figures"),
    ],
)
def test_a_projection_refuses_figures_that_contradict_its_counts(field: str, value: object, message: str) -> None:
    """The counts are a claim about the study, so a record that cannot be true is refused rather than reported."""
    arguments = _valid_projection()
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        ManualStudyProjection(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(("pairs", "completed"), [(-1, 0), (PAIRS_PER_MODEL, -1)])
def test_projecting_refuses_negative_inputs(pairs: int, completed: int) -> None:
    """A negative count would silently produce a smaller estimate than the truth."""
    with pytest.raises(ValueError, match="non-negative"):
        project_study(_models(), _measured(), pairs_per_model=pairs, completed_models=completed)


def test_a_study_with_no_measured_fit_still_projects_its_runs() -> None:
    """A check that served every fit from the cache reports no fit cost, not a division by zero."""
    projection = project_study((), _measured(), pairs_per_model=PAIRS_PER_MODEL, completed_models=0)
    assert projection.fit_seconds == 0.0
    assert projection.total_seconds > 0.0
