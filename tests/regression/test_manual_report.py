# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-012: presentation uses audited counts without treating M10 as ten independent models."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from arm_rc_ctrl.experiments.manual_report import CASES, CLASSES, TRACKERS, load_report_data, main, render_summaries
from arm_rc_ctrl.experiments.manual_results import load_results
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression
DOCS = repository_root() / "docs/experiments/task_1a_manual_demonstration"


def test_report_keeps_the_single_all_ten_model_and_every_parent() -> None:
    """The plotted M10 point is one model; each singleton contributes its own point."""
    data = load_report_data(DOCS)
    chosen = [r for r in data.arms if r.configuration == "feasible-middle" and r.tracker == "pd_v2"]
    all_ten = next(r for r in chosen if r.arm_kind == "M10" and r.scenario_class == "all")
    singletons = next(r for r in chosen if r.arm_kind == "S" and r.scenario_class == "all")
    assert all_ten.per_model == (14,)
    assert singletons.per_model == (59, 65, 65, 65, 65, 65, 65, 65, 65, 65)
    assert (all_ten.n_runs, singletons.n_runs) == (65, 650)


@pytest.mark.parametrize("name", ["arm_summary_v1.csv", "figure_inputs_v1.json"])
def test_report_refuses_a_summary_that_no_longer_matches_the_audit(tmp_path: Path, name: str) -> None:
    """A cosmetic or numerical edit to a source file requires a newly verified source."""
    docs = tmp_path / "experiment"
    shutil.copytree(DOCS, docs)
    path = docs / "results" / name
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_report_data(docs)


def test_both_summary_figures_render_from_the_audited_tables(tmp_path: Path) -> None:
    """Render the real 12 configuration/tracker cells without external payloads."""
    render_summaries(load_report_data(DOCS), tmp_path)
    for name in ("success_counts.png", "class_comparisons.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_the_presentation_command_records_inputs_and_refuses_to_overwrite(tmp_path: Path) -> None:
    """A reproducible render keeps its input bindings and preserves earlier exports."""
    out = tmp_path / "assets"
    assert main(["--docs", str(DOCS), "--output", str(out)]) == 0
    assert (out / "render_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        main(["--docs", str(DOCS), "--output", str(out)])


def _assert_asset_bindings(docs: Path) -> None:
    """Check complete input/output inventories as well as every recorded fingerprint."""
    assets = docs / "report/assets"
    manifest = json.loads((assets / "render_manifest.json").read_text())
    root = repository_root()
    results = load_results(docs / "results/results_v1.json")
    relative = DOCS.relative_to(root)
    expected_inputs = {
        str(relative / name)
        for name in ("audit/reproduction_audit_v4.json", "study_manifest_v1.json", "results/results_v1.json")
    }
    expected_inputs.update(str(relative / "results" / item.name) for item in results.documents)
    assert set(manifest["inputs"]) == expected_inputs
    for name, digest in manifest["inputs"].items():
        assert sha256_file(root / name) == digest, name
    assert manifest["renderer_sha256"] == sha256_file(root / "src/arm_rc_ctrl/experiments/manual_report.py")
    expected_outputs = {
        "success_counts.png",
        "class_comparisons.png",
        "demonstrations.png",
        "ten_helps.png",
        "ten_hurts.png",
        "ten_hurts.gif",
    }
    assert set(manifest["outputs"]) == expected_outputs
    assert {p.name for p in assets.iterdir()} == expected_outputs | {"render_manifest.json"}
    assert manifest["case_ids"] == list(CASES)
    assert manifest["trajectories"] is True
    for name, bound in manifest["outputs"].items():
        assert sha256_file(assets / name) == bound["sha256"], name
        assert (assets / name).stat().st_size == bound["size"], name


def test_committed_presentation_binds_all_inputs_source_and_assets() -> None:
    """No external store is needed to detect drift in a committed presentation."""
    _assert_asset_bindings(DOCS)


def test_asset_lock_catches_the_reviewers_single_byte_alteration(tmp_path: Path) -> None:
    """Even a PNG edit that leaves its size intact invalidates the presentation lock."""
    docs = tmp_path / "experiment"
    shutil.copytree(DOCS, docs)
    path = docs / "report/assets/success_counts.png"
    content = path.read_bytes()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(AssertionError, match=r"success_counts\.png"):
        _assert_asset_bindings(docs)


def test_committed_summary_figures_reproduce_byte_for_byte(tmp_path: Path) -> None:
    """The committed images must illustrate the current audited tables and renderer."""
    render_summaries(load_report_data(DOCS), tmp_path)
    for name in ("success_counts.png", "class_comparisons.png"):
        assert sha256_file(tmp_path / name) == sha256_file(DOCS / "report/assets" / name), name


def _numbers(text: str) -> list[float]:
    """Parse visible Markdown numbers, including Unicode minus signs and thousands separators."""
    return [float(v) for v in re.findall(r"[+-]?\d+(?:\.\d+)?", text.replace("\N{MINUS SIGN}", "-").replace(",", ""))]


def _assert_prose(text: str) -> None:
    """Bind all main-table cells and headline accounting to the audited CSVs."""
    data = load_report_data(DOCS)
    arms = {(r.configuration, r.tracker, r.scenario_class, r.arm_kind): r for r in data.arms}
    contrasts = {
        (r.configuration, r.tracker): r for r in data.contrasts if r.scenario_class == "all" and r.contrast == "M10-S"
    }
    rows = [r for r in text.splitlines() if re.match(r"\| [A-Z] / ", r)]
    assert len(rows) == len(data.study.configurations) * len(TRACKERS)
    for index, config in enumerate(data.study.configurations):
        for ti, (tracker, label) in enumerate(zip(TRACKERS, ("PD", "computed torque"), strict=True)):
            row = rows[index * len(TRACKERS) + ti]
            cells = [c.strip() for c in row.strip("|").split("|")]
            assert cells[0] == f"{chr(65 + index)} / {label}"
            single = arms[config.label, tracker, "all", "S"]
            ten = arms[config.label, tracker, "all", "M10"]
            difference = contrasts[config.label, tracker]
            assert _numbers(cells[1]) == [single.median, single.minimum, single.maximum]
            assert _numbers(cells[2]) == [ten.successes]
            assert _numbers(cells[3]) == [difference.median, difference.minimum, difference.maximum]
            assert _numbers(cells[4]) == [
                difference.parents_improved,
                difference.parents_worsened,
                difference.parents_tied,
            ]
    words = ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
    medians = [r.median for r in contrasts.values()]
    assert all(v is not None for v in medians)
    negative = sum(v is not None and v < 0 for v in medians)
    tied = medians.count(0)
    positive = sum(v is not None and v > 0 for v in medians)
    prose = " ".join(text.split())
    assert (
        f"negative in {words[negative]} combinations and tied in {words[tied]}; it was positive in {words[positive]}."
    ) in prose
    totals = [r for r in data.arms if r.scenario_class == "all"]
    learned = [r for r in totals if r.arm_kind != "replay"]
    replay = [r for r in totals if r.arm_kind == "replay"]
    assert (
        f"Replay succeeded in {sum(r.successes for r in replay):,} of {sum(r.n_runs for r in replay):,} runs." in prose
    )
    copy_pairs = sum(r.n_runs for r in learned if r.arm_kind == "S")
    assert f"All {copy_pairs:,} matched singleton-versus-copy comparisons agreed." in prose
    model_count = sum(r.n_models for r in learned if r.tracker == TRACKERS[0])
    bank_count = sum(r.n_models for r in replay if r.tracker == TRACKERS[0])
    assert f"**{model_count} learned models and {bank_count} replay banks**" in prose
    assert (
        f"{sum(r.n_runs for r in learned):,} learned-motion runs and "
        f"{sum(r.n_runs for r in replay):,} replay runs with none missing."
    ) in prose
    assert (
        f"success counts, **{sum(r.successes for r in learned):,}** and **{sum(r.successes for r in replay):,}**"
    ) in prose
    assert all(r.excluded_runs == r.unavailable_runs == 0 for r in totals)


def test_report_table_and_headlines_follow_the_audited_summaries() -> None:
    """Handwritten interpretation retains the exact paired numbers it interprets."""
    _assert_prose((DOCS / "report/report.md").read_text())


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("56 [0\N{EN DASH}59]", "57 [0\N{EN DASH}59]"),
        ("negative in six", "negative in five"),
        ("**14,184**", "**14,185**"),
    ],
)
def test_prose_lock_refuses_changed_table_headline_or_total(old: str, new: str) -> None:
    """Regression locks detect visible numerical drift without external payloads."""
    text = (DOCS / "report/report.md").read_text()
    assert old in text
    with pytest.raises(AssertionError):
        _assert_prose(text.replace(old, new, 1))


def test_audit_binding_requires_the_entire_location(tmp_path: Path) -> None:
    """A correct digest under an unrelated path cannot attest to this report input."""
    docs = tmp_path / "experiment"
    shutil.copytree(DOCS, docs)
    path = docs / "audit/reproduction_audit_v4.json"
    audit = json.loads(path.read_text())
    item = next(item for item in audit["bundle"] if item["location"].endswith("/arm_summary_v1.csv"))
    item["location"] = "unrelated/arm_summary_v1.csv"
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="audit binding"):
        load_report_data(docs)


def test_plot_labels_and_colors_follow_the_recorded_case_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed panel denominator changes labels and color normalization together."""
    data = load_report_data(DOCS)
    data = replace(
        data,
        arms=tuple(replace(r, n_scenarios=r.n_scenarios * 2, n_runs=r.n_runs * 2) for r in data.arms),
        contrasts=tuple(replace(r, n_scenarios=r.n_scenarios * 2) for r in data.contrasts),
    )
    figures: list[Figure] = []

    def capture(fig: Figure, *_args: object, **_kwargs: object) -> None:
        figures.append(fig)

    monkeypatch.setattr(Figure, "savefig", capture)
    render_summaries(data, tmp_path)
    assert "out of 130" in figures[0].axes[0].get_xlabel()
    assert [t.get_text() for t in figures[1].axes[0].get_xticklabels()] == [
        "Nominal\n2 cases",
        "Small offset\n40 cases",
        "Large offset\n40 cases",
        "Force\n8 cases",
        "Combined\n40 cases",
    ]
    records = {(r.configuration, r.tracker, r.scenario_class): r for r in data.contrasts if r.contrast == "M10-S"}
    expected = [
        [float(records[c.label, tr, cl].median or 0) / records[c.label, tr, cl].n_scenarios * 100 for cl in CLASSES]
        for c in data.study.configurations
        for tr in TRACKERS
    ]
    np.testing.assert_array_equal(figures[1].axes[0].images[0].get_array(), expected)
