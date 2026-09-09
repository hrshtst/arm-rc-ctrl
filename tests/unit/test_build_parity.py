# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""UP-007: two probes of a dependency build compare bitwise, and any difference is reported, never tolerated."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.experiments.build_parity import (
    FIXTURE_CASES,
    SUMMARY_FILE,
    BuildParityReport,
    CaseParity,
    ProbeCase,
    ProbeSummary,
    compare_probes,
    fixture_cases,
    load_parity,
    load_probe,
    main,
    parity_to_json,
    probe_fixture_recipes,
    render_parity_markdown,
    write_probe,
)
from arm_rc_ctrl.provenance import collect_provenance
from arm_rc_ctrl.rc.recipe import RclibIdentity

if TYPE_CHECKING:
    from pathlib import Path


def _provenance() -> object:
    return collect_provenance({"kind": "test"}, seeds={}, exploratory=True)


def test_fixture_probes_are_deterministic_and_compare_bitwise(tmp_path: Path) -> None:
    """Two probes of the installed build agree bitwise; a tampered prediction or case set is reported."""
    old, new = tmp_path / "old", tmp_path / "new"
    first = probe_fixture_recipes(old)
    second = probe_fixture_recipes(new)
    assert set(first.cases) == set(second.cases) == set(FIXTURE_CASES)
    assert first.rclib_installed == RclibIdentity.current()
    assert all(case.exact is None for case in first.cases.values())
    provenance = _provenance()
    report = compare_probes(old, new, provenance=provenance)  # type: ignore[arg-type]
    assert report.all_bitwise_equal
    assert [c.case for c in report.cases] == sorted(FIXTURE_CASES)
    assert all(c.max_abs_prediction_diff == 0.0 and c.fit_report_equal for c in report.cases)
    assert report.rclib_old == report.rclib_new == RclibIdentity.current()
    # Roundtrip and rendering.
    file = tmp_path / "parity.json"
    file.write_text(parity_to_json(report) + "\n", encoding="utf-8")
    assert load_parity(file) == report
    markdown = render_parity_markdown(report)
    assert "All predictions bitwise equal and all fit reports equal: **True**" in markdown
    assert all(case in markdown for case in FIXTURE_CASES)
    # A one-ulp change in one prediction is a difference, not a tolerance question.
    summary, predictions = load_probe(new)
    tampered = tmp_path / "tampered"
    label = FIXTURE_CASES[0]
    predictions[label] = np.nextafter(predictions[label], np.inf)
    write_probe(tampered, summary, predictions)
    differing = compare_probes(old, tampered, provenance=provenance)  # type: ignore[arg-type]
    assert not differing.all_bitwise_equal
    case = next(c for c in differing.cases if c.case == label)
    assert not case.predictions_bitwise_equal
    assert 0.0 < case.max_abs_prediction_diff < 1e-12
    assert case.fit_report_equal  # the fit report was not touched
    assert "**False**" in render_parity_markdown(differing)
    # Mismatched case sets and overwrites are refused; the invariants re-derive on load.
    partial = tmp_path / "partial"
    write_probe(
        partial, ProbeSummary(summary.rclib_installed, {label: summary.cases[label]}), {label: predictions[label]}
    )
    with pytest.raises(ValueError, match="case sets differ"):
        compare_probes(old, partial, provenance=provenance)  # type: ignore[arg-type]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_probe(new, summary, predictions)
    with pytest.raises(ValueError, match="same cases"):
        write_probe(tmp_path / "x", summary, {})
    good = report.cases[0]
    with pytest.raises(ValueError, match="contradicts the prediction digests"):
        replace(good, predictions_bitwise_equal=False)
    with pytest.raises(ValueError, match="cannot differ"):
        replace(good, max_abs_prediction_diff=1.0)
    with pytest.raises(ValueError, match="all_bitwise_equal contradicts"):
        replace(report, all_bitwise_equal=False)
    with pytest.raises(ValueError, match="at least one case"):
        replace(report, cases=())
    with pytest.raises(ValueError, match="unsupported parity schema_version"):
        replace(report, schema_version=2)
    with pytest.raises(ValueError, match="at least one case"):
        ProbeSummary(summary.rclib_installed, {})
    assert isinstance(good, CaseParity)
    assert isinstance(report, BuildParityReport)


def test_summary_layout_holds_the_installed_rclib_and_one_entry_per_case(tmp_path: Path) -> None:
    """The summary JSON keeps the flat probe layout (``rclib_installed`` plus cases) with an optional ``exact``."""
    _prediction, report = fixture_cases()[FIXTURE_CASES[0]]
    summary = ProbeSummary(
        RclibIdentity.current(), {"frozen": ProbeCase(report, exact=True), "fixture": ProbeCase(report)}
    )
    text = summary.to_json()
    mapping = json.loads(text)
    assert set(mapping) == {"rclib_installed", "frozen", "fixture"}
    assert mapping["frozen"]["exact"] is True
    assert "exact" not in mapping["fixture"]
    assert ProbeSummary.from_json(text) == summary
    del tmp_path


def test_main_probes_and_compares(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command line probes the fixture recipes and writes the parity evidence; outputs are never overwritten."""
    old, new = tmp_path / "old", tmp_path / "new"
    assert main(["probe", "--output", str(old), "--fixture-only"]) == 0
    assert main(["probe", "--output", str(new), "--fixture-only"]) == 0
    printed = capsys.readouterr().out
    assert all(case in printed for case in FIXTURE_CASES)
    output, markdown = tmp_path / "parity.json", tmp_path / "parity.md"
    argv = ["compare", "--old", str(old), "--new", str(new), "--output", str(output), "--markdown", str(markdown)]
    assert main([*argv, "--exploratory"]) == 0
    report = load_parity(output)
    assert report.all_bitwise_equal
    assert report.provenance.exploratory
    assert markdown.read_text(encoding="utf-8") == render_parity_markdown(report)
    assert (old / SUMMARY_FILE).is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main([*argv, "--exploratory"])
