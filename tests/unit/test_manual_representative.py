# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the frozen rule that picks which cases the final report illustrates.

Selecting the study's models or evaluations on outcome would bias the
experiment; selecting which of its results to *show* does not, and a report that
never showed a failure would be worse than one that did. So this rule may read
verdicts, but it is frozen before execution and applied afterwards, and what it
yields are illustrations: aggregate conclusions come from the whole study.

Absent categories are reported rather than quietly dropped, so a reader can see
that a contrast the rule looked for did not occur.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.manual_representative import (
    ILLUSTRATION_ARMS,
    representative_cases,
)

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.manual_representative import ArmVerdict

TRACKER = "pd_v2"
ORDER = ("nominal", "posture-small-00", "posture-large-00", "force-000deg", "combined-00")


def _verdicts(**by_scenario: dict[str, bool]) -> tuple[ArmVerdict, ...]:
    """Verdicts for one tracker, given ``scenario -> {arm: succeeded}``."""
    from arm_rc_ctrl.experiments.manual_representative import ArmVerdict

    return tuple(
        ArmVerdict(scenario_id=scenario, tracker=TRACKER, arm=arm, succeeded=ok)
        for scenario, arms in by_scenario.items()
        for arm, ok in arms.items()
    )


def _all(*, succeeded: bool) -> dict[str, bool]:
    return dict.fromkeys(ILLUSTRATION_ARMS, succeeded)


def test_the_nominal_case_is_always_illustrated() -> None:
    """Every configuration and tracker shows its nominal comparison, whatever the contrasts find."""
    selection = representative_cases(_verdicts(nominal=_all(succeeded=True)), order=ORDER)
    assert "nominal" in [case.scenario_id for case in selection.cases]
    assert selection.categories["nominal"] == ("nominal",)


def test_the_first_case_where_the_all_ten_arm_wins_is_selected() -> None:
    """The first scenario in frozen order where M10 succeeds and the singleton fails."""
    verdicts = _verdicts(
        nominal=_all(succeeded=True),
        **{
            "posture-small-00": {"S": True, "M10": True, "R10": True, "C10": True},
            "posture-large-00": {"S": False, "M10": True, "R10": True, "C10": True},
            "force-000deg": {"S": False, "M10": True, "R10": True, "C10": True},
        },
    )
    selection = representative_cases(verdicts, order=ORDER)
    assert selection.categories["all_ten_wins"] == ("posture-large-00",), "the first in frozen order, not the last"


def test_the_first_case_where_the_singleton_wins_is_selected() -> None:
    """And the reverse contrast, so the report is not one-sided."""
    verdicts = _verdicts(
        nominal=_all(succeeded=True),
        **{
            "posture-small-00": {"S": True, "M10": False, "R10": True, "C10": True},
            "posture-large-00": {"S": True, "M10": False, "R10": True, "C10": True},
        },
    )
    selection = representative_cases(verdicts, order=ORDER)
    assert selection.categories["singleton_wins"] == ("posture-small-00",)


def test_the_first_failure_of_any_comparison_arm_is_selected() -> None:
    """A report that never showed a failure would misrepresent the study."""
    verdicts = _verdicts(
        nominal=_all(succeeded=True),
        **{
            "posture-small-00": _all(succeeded=True),
            "posture-large-00": {"S": True, "M10": True, "R10": False, "C10": True},
        },
    )
    selection = representative_cases(verdicts, order=ORDER)
    assert selection.categories["first_failure"] == ("posture-large-00",)


def test_a_case_selected_twice_appears_once_in_frozen_order() -> None:
    """Deduplicated, and each case records every category that chose it."""
    verdicts = _verdicts(
        nominal=_all(succeeded=True),
        **{"posture-small-00": {"S": False, "M10": True, "R10": True, "C10": True}},
    )
    selection = representative_cases(verdicts, order=ORDER)
    ids = [case.scenario_id for case in selection.cases]
    assert ids == sorted(ids, key=ORDER.index), "frozen order is kept"
    assert len(ids) == len(set(ids)), "no case is illustrated twice"
    chosen = next(case for case in selection.cases if case.scenario_id == "posture-small-00")
    assert set(chosen.categories) == {"all_ten_wins", "first_failure"}


def test_a_category_that_never_occurs_is_reported_as_absent() -> None:
    """The rule looked and found nothing, which is a result the reader must see stated."""
    selection = representative_cases(_verdicts(nominal=_all(succeeded=True)), order=ORDER)
    assert selection.categories["all_ten_wins"] == ()
    assert selection.categories["singleton_wins"] == ()
    assert selection.categories["first_failure"] == ()
    assert "all_ten_wins" in selection.absent
    assert "nominal" not in selection.absent


def test_every_selected_case_carries_all_four_arms() -> None:
    """A comparison needs all four arms present, or it is not a comparison."""
    verdicts = _verdicts(
        nominal=_all(succeeded=True),
        **{"posture-small-00": {"S": False, "M10": True, "R10": True, "C10": True}},
    )
    selection = representative_cases(verdicts, order=ORDER)
    for case in selection.cases:
        assert set(case.arms) == set(ILLUSTRATION_ARMS)


def test_a_scenario_missing_an_arm_is_refused() -> None:
    """An incomplete comparison would be illustrated as if a missing arm had not been run."""
    verdicts = _verdicts(nominal={"S": True, "M10": True, "R10": True})
    with pytest.raises(ValueError, match="all four arms"):
        representative_cases(verdicts, order=ORDER)


def test_a_scenario_outside_the_frozen_order_is_refused() -> None:
    """The order is the study's; a case from outside it cannot be ranked, so it is not silently ignored."""
    verdicts = _verdicts(**{"invented-case": _all(succeeded=True)})
    with pytest.raises(ValueError, match="outside the frozen order"):
        representative_cases(verdicts, order=ORDER)
