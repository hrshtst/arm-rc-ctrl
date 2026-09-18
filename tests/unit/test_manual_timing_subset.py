# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: which models the timing smoke check measures, and why that rule is fixed in advance.

Plan section 5 inherits the six configurations precisely to avoid choosing
after seeing results, and requires the deterministic nominal subset to run
"without dropping panel members based on performance". So the subset is taken
by position from the frozen manifest and never consults an outcome, a status,
or what a configuration's label happens to mean. The repeated-demonstration
pilot measured its ``feasible-best`` entry by name; inheriting that default
here would reintroduce exactly the selection this study is designed to avoid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.experiments.manual_timing import BUDGET_PARENT, budget_entries, smoke_entries

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel


def _labels(entries: tuple[StudyModel, ...]) -> list[str]:
    return [entry.label for entry in entries]


def test_the_subset_is_a_prefix_of_the_frozen_manifest(manual_fixture: ManualFixture) -> None:
    """Taken by position: the first models the study itself lists, in the order it lists them."""
    manifest = manual_fixture.manifest
    chosen = smoke_entries(manifest, count=4)
    assert _labels(chosen) == _labels(tuple(manifest.entries[:4]))
    assert len(chosen) == 4


def test_the_subset_grows_as_a_prefix(manual_fixture: ManualFixture) -> None:
    """Each larger subset extends the last: the rule can neither skip an entry nor reorder them.

    A positional prefix is the whole of the rule, which is what keeps the check
    honest. The manifest refuses to exist with configurations renamed, so
    "ignores the labels" cannot be demonstrated by relabelling one; what can be
    demonstrated is that the selection is a slice of the manifest's own order
    and so has nothing else to consult.
    """
    manifest = manual_fixture.manifest
    previous: list[str] = []
    for count in range(1, 6):
        chosen = _labels(smoke_entries(manifest, count=count))
        assert chosen == _labels(tuple(manifest.entries[:count]))
        assert chosen[: len(previous)] == previous, "a larger subset extends the smaller one"
        previous = chosen
    assert smoke_entries(manifest, count=3) == smoke_entries(manifest, count=3), "and it is deterministic"


def test_the_subset_refuses_a_count_it_cannot_honour(manual_fixture: ManualFixture) -> None:
    """Asking for more models than the study holds is a mistake, not a silently shorter check."""
    manifest = manual_fixture.manifest
    with pytest.raises(ValueError, match="models"):
        smoke_entries(manifest, count=len(manifest.entries) + 1)
    with pytest.raises(ValueError, match="at least one"):
        smoke_entries(manifest, count=0)


def test_explicit_labels_override_the_rule(manual_fixture: ManualFixture) -> None:
    """A named subset is still available for a targeted check, and is reported as chosen by hand."""
    manifest = manual_fixture.manifest
    wanted = (manifest.entries[5].label, manifest.entries[2].label)
    chosen = smoke_entries(manifest, count=2, labels=wanted)
    assert set(_labels(chosen)) == set(wanted)
    assert _labels(chosen) == [manifest.entries[2].label, manifest.entries[5].label], "manifest order is kept"


def test_an_unknown_label_is_refused_rather_than_quietly_dropped(manual_fixture: ManualFixture) -> None:
    """A misspelt label would otherwise measure fewer models than asked, and divide the projection by them."""
    manifest = manual_fixture.manifest
    with pytest.raises(ValueError, match="unknown model labels"):
        smoke_entries(manifest, count=2, labels=(manifest.entries[0].label, "feasible-best/S/D99"))


# --- the subset the measured budget rests on ---------------------------------------------------


def test_the_budget_subset_covers_every_configuration_and_arm_kind(manual_fixture: ManualFixture) -> None:
    """Coverage rather than a prefix: all six configurations and all four arm kinds, 24 models.

    A growing prefix is deterministic but can sit inside one configuration and
    miss the expensive training arms, so the projection would scale costs it
    never measured. This subset spans both dimensions that drive cost.
    """
    chosen = budget_entries(manual_fixture.manifest)
    assert len(chosen) == 24
    by_configuration: dict[str, set[str]] = {}
    for entry in chosen:
        by_configuration.setdefault(entry.configuration, set()).add(entry.arm.arm)
    assert len(by_configuration) == 6, "every inherited configuration is represented"
    assert all(kinds == {"S", "M10", "R10", "C10"} for kinds in by_configuration.values())


def test_the_budget_subset_fixes_one_parent_for_parent_specific_arms(manual_fixture: ManualFixture) -> None:
    """One fixed parent keeps the subset comparable across configurations; the all-ten arm has none."""
    chosen = budget_entries(manual_fixture.manifest)
    for entry in chosen:
        if entry.arm.arm == "M10":
            assert entry.arm.assignment is None, "the all-ten arm trains on the whole bank"
        else:
            assert entry.arm.assignment == BUDGET_PARENT


def test_the_budget_subset_needs_exactly_six_replay_banks(manual_fixture: ManualFixture) -> None:
    """One bank per configuration: the parent-specific arms of a configuration share its policy and parent."""
    chosen = budget_entries(manual_fixture.manifest)
    banks = {(entry.configuration, entry.arm.assignment) for entry in chosen if entry.arm.assignment is not None}
    assert len(banks) == 6


def test_the_budget_subset_is_deterministic_and_in_manifest_order(manual_fixture: ManualFixture) -> None:
    """Chosen by rule and never by result: the same models, in the study's own order, every time."""
    manifest = manual_fixture.manifest
    first = budget_entries(manifest)
    assert [entry.label for entry in first] == [entry.label for entry in budget_entries(manifest)]
    positions = [manifest.entries.index(entry) for entry in first]
    assert positions == sorted(positions), "manifest order is kept"


def test_the_budget_subset_refuses_a_parent_the_study_does_not_have(manual_fixture: ManualFixture) -> None:
    """A missing parent would quietly shrink the subset and the denominator the projection divides by."""
    with pytest.raises(ValueError, match="parent"):
        budget_entries(manual_fixture.manifest, parent="D99")
