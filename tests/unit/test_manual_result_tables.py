# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: the CSV tables of the derived evidence read back exactly as they were written.

A table is only evidence if a reader recovers the same values the derivation
held: floats to the last bit, an absent value distinct from any present one, a
tuple's missing element distinct from a missing tuple, and a header that is the
record's own fields. Anything the format cannot represent unambiguously is
refused when it is written, not guessed at when it is read.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Any, cast

import pytest

from arm_rc_ctrl.experiments.manual_contrasts import ManualArmSummary, ManualContrastSummary
from arm_rc_ctrl.experiments.manual_results import ManualRunRow, table_columns, table_from_csv, table_to_csv


def _summary(**changes: object) -> ManualContrastSummary:
    base: dict[str, Any] = {
        "configuration": "feasible-best",
        "tracker": "pd_v2",
        "scenario_class": "all",
        "contrast": "M10-S",
        "arm_a": "M10",
        "arm_b": "S",
        "n_scenarios": 65,
        "n_parents": 9,
        "differences": (3, -1, 0, None, 2, 2, 1, 0, 0, 5),
        "median": 1.0,
        "minimum": -1,
        "maximum": 5,
        "parents_improved": 5,
        "parents_worsened": 1,
        "parents_tied": 3,
    }
    return ManualContrastSummary(**cast("dict[str, Any]", base | changes))


def _run_row(**changes: object) -> ManualRunRow:
    empty: dict[str, Any] = {f.name: None for f in dc.fields(ManualRunRow)}
    fixed: dict[str, Any] = {
        "source": "rc",
        "configuration": "feasible-best",
        "arm": "S/D01",
        "arm_kind": "S",
        "parent": "D01",
        "model_label": "feasible-best/S/D01",
        "tracker": "pd_v2",
        "scenario_id": "nominal",
        "scenario_class": "nominal",
        "scenario_index": 0,
        "status": "infeasible",
        "success": False,
        "reason": 'dwell failed, "final" dwell 0.3 s; see run\nsummary',
        "initial_q": (0.1 + 0.2, -1e-300),
        "run_uri": "armrc://runs/run-20260918-000000000000/run.json",
        "n_active_samples": 3001,
        "final_endpoint_error_m": 0.1 + 0.2,
        "sources": ("processed-20260916-a8c94bb35358",),
    }
    return ManualRunRow(**cast("dict[str, Any]", empty | fixed | changes))


def test_a_summary_round_trips_with_its_absent_differences() -> None:
    """An absent parent's difference is an element that is null, not a missing column or a zero."""
    rows = (_summary(), _summary(tracker="computed_torque"))
    assert table_from_csv(table_to_csv(rows, ManualContrastSummary), ManualContrastSummary) == rows


def test_a_one_element_tuple_holding_an_absent_value_stays_distinct_from_an_empty_one() -> None:
    """The all-ten arm's single per-model count may be absent; that is not the same as having no models."""
    row = ManualArmSummary(
        configuration="feasible-best",
        tracker="pd_v2",
        scenario_class="force",
        arm_kind="M10",
        n_scenarios=4,
        n_models=0,
        n_runs=0,
        excluded_runs=3,
        unavailable_runs=1,
        successes=0,
        per_model=(None,),
        median=None,
        minimum=None,
        maximum=None,
    )
    text = table_to_csv((row,), ManualArmSummary)
    assert table_from_csv(text, ManualArmSummary) == (row,)
    assert "[null]" in text


def test_floats_and_awkward_strings_survive_exactly() -> None:
    """Floats come back bit for bit; commas, quotes and newlines in a reason are quoted, not split."""
    row = _run_row()
    back = table_from_csv(table_to_csv((row,), ManualRunRow), ManualRunRow)
    assert back == (row,)
    assert back[0].final_endpoint_error_m == 0.1 + 0.2
    assert back[0].initial_q == (0.1 + 0.2, -1e-300)


def test_the_header_is_the_records_fields_in_order() -> None:
    """A reader can take the columns straight from the schema's field list."""
    header = table_to_csv((), ManualRunRow).splitlines()[0]
    assert tuple(header.split(",")) == table_columns(ManualRunRow)


def test_a_table_with_another_header_is_refused() -> None:
    """A table of one record is never read as another, even when the cells would happen to parse."""
    text = table_to_csv((_summary(),), ManualContrastSummary)
    with pytest.raises(ValueError, match="header"):
        table_from_csv(text, ManualArmSummary)


def test_an_empty_string_is_refused_rather_than_written_as_absent() -> None:
    """An empty cell means absent; a present but empty string would read back as absent, so it is not written."""
    with pytest.raises(ValueError, match="indistinguishable"):
        table_to_csv((_run_row(reason=""),), ManualRunRow)


def test_a_cell_of_the_wrong_type_is_refused_when_read() -> None:
    """A difference that is not an integer is not coerced into one."""
    text = table_to_csv((_summary(),), ManualContrastSummary).replace("[3,-1,0,null", "[3.5,-1,0,null")
    with pytest.raises(ValueError, match="is not of type int"):
        table_from_csv(text, ManualContrastSummary)


def test_a_row_of_another_record_is_refused_when_written() -> None:
    """One table holds one record."""
    with pytest.raises(TypeError, match="cannot hold"):
        table_to_csv((_summary(),), ManualArmSummary)
