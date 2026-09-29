# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only
"""CI-001: the M3MS expert report's builders and refusals, exercised without the evidence store.

The committed presentation is verified elsewhere; here the store-reading seams
(the run table artifact and each run's verified payload) are replaced by small
synthetic runs, so the binding, rendering and refusal logic runs on every
checkout, as it must on CI.
"""

from __future__ import annotations

import dataclasses as dc
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from arm_rc_ctrl.experiments import manual_search_report as report
from arm_rc_ctrl.experiments import manual_search_report_plots as plots
from arm_rc_ctrl.experiments.manual_evaluation import ManualRunArtifact
from arm_rc_ctrl.experiments.manual_figures import ManualFigureCase, ManualFigureInputs
from arm_rc_ctrl.experiments.manual_results import ManualRunRow, table_to_csv
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

DOCS = report.DOCS
LENGTHS = np.asarray([0.30, 0.25], dtype=np.float64)
ACTIVATION_S = 0.25


def _row(index: int, **changes: object) -> ManualRunRow:
    """One simulated run row; ``index`` makes its payload digests unique."""
    empty: dict[str, Any] = {f.name: None for f in dc.fields(ManualRunRow)}
    fixed: dict[str, Any] = {
        "source": "rc",
        "configuration": "search-t0001",
        "arm": "S/D01",
        "arm_kind": "S",
        "parent": "D01",
        "model_label": "search-t0001/S/D01",
        "tracker": "pd_v2",
        "scenario_id": "nominal",
        "scenario_class": "nominal",
        "scenario_index": 0,
        "status": "completed",
        "success": True,
        "reason": None,
        "activation_s": ACTIVATION_S,
        "n_active_samples": 106,
        "final_endpoint_error_m": 0.004,
        "time_to_final_dwell_s": 0.5,
        "dwell_final_s": 0.8,
        "run_artifact_id": f"run-{index:04d}",
        "run_uri": f"armrc://runs/run-{index:04d}/run.json",
        "run_sha256": f"{index:064x}",
        "run_size": 100 + index,
        "arrays_sha256": f"{index + 10_000:064x}",
        "sources": ("processed-0000",),
    }
    return ManualRunRow(**cast("dict[str, Any]", empty | fixed | changes))


def _case_rows() -> tuple[ManualRunRow, ...]:
    """Exactly one row per role of every illustrated case: M10 fails its dwell, the others pass."""
    rows: dict[tuple[str, str, str, str], ManualRunRow] = {}
    for spec in report.CASE_SPECS:
        for role in plots.ROLES:
            arm = "M10" if role == "M10" else f"{role}/{spec.parent}"
            key = (spec.configuration, spec.tracker, arm, spec.scenario)
            if key in rows:
                continue
            force = spec.scenario.startswith("force")
            fails = role == "M10"
            rows[key] = _row(
                len(rows),
                source="replay" if role == "replay" else "rc",
                configuration=spec.configuration,
                arm=arm,
                arm_kind=role,
                parent=None if role == "M10" else spec.parent,
                model_label=f"{spec.configuration}/{arm}",
                tracker=spec.tracker,
                scenario_id=spec.scenario,
                scenario_class=spec.scenario.split("-")[0],
                status="infeasible" if fails else "completed",
                success=not fails,
                reason="dwell: final hold 0.2 s < 0.5 s" if fails else None,
                final_endpoint_error_m=None if fails else 0.004,
                time_to_final_dwell_s=None if fails else 0.5,
                pulse_start_s=0.75 if force else None,
                pulse_end_s=0.95 if force else None,
            )
    return tuple(rows.values())


def _tip(q: NDArray[np.float64]) -> NDArray[np.float64]:
    angles = np.cumsum(q, axis=1)
    return np.stack(((np.cos(angles) * LENGTHS).sum(axis=1), (np.sin(angles) * LENGTHS).sum(axis=1)), axis=1)


def _arrays(*, with_applied: bool) -> dict[str, NDArray[Any]]:
    """A short stored run on the recording clock, with warm-up before activation."""
    t = np.round(np.arange(0.0, 1.3, 0.01), 10)
    q = np.stack((0.4 + 0.3 * t, 1.2 - 0.2 * t), axis=1)
    arrays: dict[str, NDArray[Any]] = {
        "t": t,
        "q": q,
        "q_desired": q + 0.01,
        "tip": _tip(q),
        "dq": np.tile([0.3, -0.2], (len(t), 1)),
        "dq_desired": np.tile([0.31, -0.21], (len(t), 1)),
        "tau_requested": np.tile([1.0, -0.5], (len(t), 1)),
    }
    if with_applied:
        arrays["tau_applied"] = np.tile([0.9, -0.4], (len(t), 1))
    return arrays


@pytest.fixture
def fake_payloads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the store-reading seams of ``render_case``; returns the run URIs it was asked for."""
    requested: list[str] = []

    def load(_store: object, run: ManualRunArtifact, *, where: str) -> tuple[object, dict[str, NDArray[Any]]]:
        assert where
        requested.append(run.uri)
        summary = SimpleNamespace(termination=SimpleNamespace(kind="horizon"))
        return summary, _arrays(with_applied=len(requested) % 2 == 0)

    def plot_case(inputs: ManualFigureInputs, case_id: str, out: Path, *, store: object, root: Path) -> Path:
        assert case_id in {c.case_id for c in inputs.cases}
        assert store is not None
        assert root == repository_root()
        Image.new("RGB", (40, 30), "#ffffff").save(out)
        return out

    monkeypatch.setattr(plots, "load_verified_payload", load)
    monkeypatch.setattr(plots, "plot_case", plot_case)
    return requested


def _render_one(rows: tuple[ManualRunRow, ...], out: Path, slug: str) -> dict[str, object]:
    figures = report.case_inputs(rows, repository_root())
    index = next(i for i, s in enumerate(report.CASE_SPECS) if s.slug == slug)
    return plots.render_case(figures.cases[index], report.CASE_SPECS[index], figures, out, cast("Any", object()), rows)


def test_load_data_refuses_an_index_or_freeze_the_audit_did_not_issue(tmp_path: Path) -> None:
    """The results index and the freeze are accepted only with the passing audit's digests."""
    root = repository_root()
    for name, match in (("results/results_v1.json", "passing audit"), ("freeze/selection_v1.json", "freeze")):
        copy = tmp_path / name.split("/")[0]
        shutil.copytree(root / DOCS, copy / DOCS)
        path = copy / DOCS / name
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(ValueError, match=match):
            report.load_data(copy)


def test_summaries_refuse_inconsistent_scenario_denominators(tmp_path: Path) -> None:
    """A class count that differs between rows is refused before any plot is drawn."""
    data = report.load_data(repository_root())
    last = data.contrasts[-1]
    changed = dc.replace(data, contrasts=(*data.contrasts[:-1], dc.replace(last, n_scenarios=last.n_scenarios + 1)))
    with pytest.raises(ValueError, match="denominators"):
        report.render_summaries(changed, tmp_path)
    assert not list(tmp_path.iterdir())


def test_signed_keeps_a_missing_value_distinct_from_zero() -> None:
    """Missing, zero, positive and negative differences each render differently."""
    assert [report.signed(v) for v in (None, 0.0, 2.0, -1.5)] == ["—", "0", "+2", "\N{MINUS SIGN}1.5"]


def test_diagnostic_table_counts_every_m10_deciding_reason() -> None:
    """Each configuration and tracker lists its M10 passes and failures by deciding reason, nothing else."""
    rows = (
        _row(0, arm="M10", arm_kind="M10", parent=None),
        _row(1, arm="M10", arm_kind="M10", parent=None, status="infeasible", success=False, reason="dwell: short"),
        _row(2, arm="M10", arm_kind="M10", parent=None, status="infeasible", success=False, reason="dwell: short"),
        _row(3, tracker="computed_torque", arm="M10", arm_kind="M10", parent=None, status="infeasible", success=False,
             reason="saturation: 0.4"),
        _row(4, status="infeasible", success=False, reason="unmapped: an S failure is not counted"),
    )  # fmt: skip
    table = report.diagnostic_table(rows)
    lines = table.splitlines()
    assert lines == [
        "<tr><th>search-t0001 / pd_v2</th><td>1</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>",
        "<tr><th>search-t0001 / computed_torque</th><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr>",
    ]
    unknown = _row(5, arm="M10", arm_kind="M10", parent=None, status="infeasible", success=False, reason="novel: x")
    with pytest.raises(ValueError, match="unmapped M10 deciding reason"):
        report.diagnostic_table((*rows, unknown))


def test_case_inputs_bind_every_illustration_to_its_audited_rows() -> None:
    """Each post-hoc case names its five runs in role order, with the parent's recorded teacher."""
    rows = _case_rows()
    figures = report.case_inputs(rows, repository_root())
    assert [c.case_id for c in figures.cases] == [s.case_id for s in report.CASE_SPECS]
    assert figures.scenario_sha256 == sha256_file(repository_root() / report.TASK)
    by_uri = {r.run_uri: r for r in rows}
    for spec, case in zip(report.CASE_SPECS, figures.cases, strict=True):
        assert case.teacher_assignment == spec.parent
        assert case.categories == ("post-hoc explanatory selection",)
        assert case.activation_s == ACTIVATION_S
        assert [r.role for r in case.runs] == list(plots.ROLES)
        for run in case.runs:
            row = by_uri[run.run.uri]
            assert (row.arm_kind, row.scenario_id, row.tracker) == (run.role, spec.scenario, spec.tracker)
            assert (run.run.sha256, run.run.arrays_sha256, run.success) == (
                row.run_sha256,
                row.arrays_sha256,
                row.success,
            )
        if spec.scenario.startswith("force"):
            assert all(r.pulse_start_s == 0.75 and r.pulse_end_s == 0.95 for r in case.runs)


def test_case_inputs_refuse_a_task_configuration_the_teachers_did_not_use(tmp_path: Path) -> None:
    """The illustrated kinematics must be the teacher study's task, byte for byte."""
    root = repository_root()
    for relative in (report.TEACHERS, report.TASK):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, tmp_path / relative)
    task = tmp_path / report.TASK
    task.write_text(task.read_text() + "\n")
    with pytest.raises(ValueError, match="task configuration fingerprint"):
        report.case_inputs(_case_rows(), tmp_path)


def test_render_case_exports_bound_plots_and_a_sampled_playback(tmp_path: Path, fake_payloads: list[str]) -> None:
    """A force case draws all five runs, keeps pulse events in the playback, and reports stored diagnostics."""
    rows = _case_rows()
    info = _render_one(rows, tmp_path, "t1-force")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "t1-force-joints.png",
        "t1-force-space.png",
        "t1-force-time.png",
        "t1-force.js",
    ]
    assert len(fake_payloads) == len(plots.ROLES)
    for suffix in ("space", "time", "joints"):
        with Image.open(tmp_path / f"t1-force-{suffix}.png") as image:
            assert image.mode == "P"
    script = (tmp_path / "t1-force.js").read_text()
    prefix, suffix = "window.manualCases.push(", ");\n"
    assert script.startswith(prefix)
    assert script.endswith(suffix)
    payload = json.loads(script[len(prefix) : -len(suffix)])
    assert payload["lengths"] == LENGTHS.tolist()
    assert payload["target"] == [0.10, 0.45]
    assert payload["radius"] == 0.01
    assert [run["role"] for run in payload["runs"]] == list(plots.ROLES)
    for run in payload["runs"]:
        times = np.asarray(run["t"])
        assert times[0] == pytest.approx(-ACTIVATION_S)
        assert run["end"] == pytest.approx(1.29 - ACTIVATION_S)
        assert times[-1] == pytest.approx(run["end"])
        assert run["pulse"] == pytest.approx([0.5, 0.7])
        for event in (0.0, 0.5, 0.7):
            assert np.min(np.abs(times - event)) < 1e-9
        assert len(run["q"]) == len(times)
        assert run["samples"] == 130
        assert run["termination"] == "horizon"
    assert info["case_id"] == report.CASE_SPECS[2].case_id
    runs = cast("list[dict[str, Any]]", info["runs"])
    assert {"t", "q", "ref", "color"}.isdisjoint(runs[0])
    m10 = runs[plots.ROLES.index("M10")]
    assert (m10["success"], m10["final_error_mm"], m10["reason"]) == (False, None, "dwell: final hold 0.2 s < 0.5 s")
    assert runs[0]["final_error_mm"] == pytest.approx(4.0)


def test_case_html_marks_verdicts_and_escapes_stored_text(tmp_path: Path, fake_payloads: list[str]) -> None:
    """Missing diagnostics show a dash and stored text never becomes markup."""
    info = _render_one(_case_rows(), tmp_path, "t1-nominal")
    assert fake_payloads
    runs = cast("list[dict[str, Any]]", info["runs"])
    runs[0] = runs[0] | {"success": False, "reason": "limit_violation: <joint 1>"}
    spec = report.CASE_SPECS[0]
    page = plots.case_html(spec, info)
    assert f'id="case-{spec.slug}"' in page
    assert "<th>S</th><td>Fail</td><td>4.00</td><td>0.50</td><td>limit_violation: &lt;joint 1&gt;</td>" in page
    assert "<th>M10</th><td>Fail</td><td>—</td><td>—</td>" in page
    assert "<th>replay</th><td>Pass</td><td>4.00</td><td>0.50</td><td>All criteria met</td>" in page
    assert "post-hoc explanatory selection" in page
    assert f"assets/{spec.slug}-space.png" in page


def test_render_case_refuses_unbound_or_mismatched_inputs(tmp_path: Path, fake_payloads: list[str]) -> None:
    """A changed task digest, a missing table row, or a disagreeing binding stops the case."""
    rows = _case_rows()
    figures = report.case_inputs(rows, repository_root())
    spec, case = report.CASE_SPECS[0], figures.cases[0]
    store = cast("Any", object())
    wrong_task = dc.replace(figures, scenario_sha256="0" * 64)
    with pytest.raises(ValueError, match="bound digest"):
        plots.render_case(case, spec, wrong_task, tmp_path, store, rows)
    s_uri = case.runs[0].run.uri
    without_s = tuple(r for r in rows if r.run_uri != s_uri)
    with pytest.raises(ValueError, match="exactly one audited run row"):
        plots.render_case(case, spec, figures, tmp_path, store, without_s)
    duplicated = (*rows, next(r for r in rows if r.run_uri == s_uri))
    with pytest.raises(ValueError, match="exactly one audited run row"):
        plots.render_case(case, spec, figures, tmp_path, store, duplicated)
    retabled = tuple(dc.replace(r, arrays_sha256="f" * 64) if r.run_uri == s_uri else r for r in rows)
    with pytest.raises(ValueError, match="table and figure bindings disagree"):
        plots.render_case(case, spec, figures, tmp_path, store, retabled)
    assert fake_payloads == [s_uri]


def _copied_report(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    copied = tmp_path / "report"
    shutil.copytree(repository_root() / DOCS / "report", copied)
    return copied, json.loads((copied / "manifest.json").read_text())


def _write_manifest(copied: Path, manifest: dict[str, Any]) -> None:
    (copied / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_verify_report_refuses_changed_case_and_table_bindings(tmp_path: Path) -> None:
    """The case list and the run-table binding are part of what verification compares."""
    root = repository_root()
    copied, manifest = _copied_report(tmp_path)
    _write_manifest(copied, manifest | {"cases": manifest["cases"][:-1]})
    with pytest.raises(ValueError, match="case inventory"):
        report.verify_report(copied, root=root)
    table = manifest["run_table"] | {"sha256": "0" * 64}
    _write_manifest(copied, manifest | {"run_table": table})
    with pytest.raises(ValueError, match="table binding"):
        report.verify_report(copied, root=root)


def test_verify_report_refuses_a_page_whose_tables_were_rewritten(tmp_path: Path) -> None:
    """Refingerprinting an edited page does not make its tables audited evidence."""
    root = repository_root()
    copied, manifest = _copied_report(tmp_path)
    page = copied / "index.html"
    table = report.summary_table(report.load_data(root))
    text = page.read_text()
    assert table in text
    page.write_text(text.replace(table, table.replace("</td>", " </td>", 1)))
    manifest["outputs"]["index.html"] = {"sha256": sha256_file(page), "size": page.stat().st_size}
    _write_manifest(copied, manifest)
    with pytest.raises(ValueError, match="table differs from the audited evidence"):
        report.verify_report(copied, root=root)


def test_presentation_verification_refuses_a_changed_source_digest(tmp_path: Path) -> None:
    """Every declared source and input byte is compared, not only the produced files."""
    copied, manifest = _copied_report(tmp_path)
    name = next(iter(manifest["inputs"]))
    manifest["inputs"][name] = "0" * 64
    _write_manifest(copied, manifest)
    with pytest.raises(ValueError, match="presentation input or source fingerprint differs"):
        plots.verify_report(copied, root=repository_root())


def test_build_report_renders_a_directory_that_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the store replaced by a synthetic run table, the full build writes a presentation that verifies."""
    rows = _case_rows()
    table_path = tmp_path / "runs.csv"
    table_path.write_text(table_to_csv(rows, ManualRunRow))
    data = report.load_data(repository_root())
    expected = next(t for t in data.index.tables if t.record == "ManualRunRow").payload
    rendered: list[str] = []

    def verify_artifact(store: object, payload: object) -> Path:
        assert store == "store"
        assert payload == expected
        return table_path

    def render_case(
        case: ManualFigureCase, spec: plots.Illustration, _inputs: object, out: Path, store: object, seen: object
    ) -> dict[str, object]:
        assert (store, seen, case.case_id) == ("store", rows, spec.case_id)
        rendered.append(spec.slug)
        (out / f"{spec.slug}.js").write_text("window.manualCases.push({});\n")
        runs = [
            {"role": r.role, "success": r.success, "final_error_mm": None, "dwell_task_s": None, "reason": None}
            for r in case.runs
        ]
        return {"slug": spec.slug, "case_id": case.case_id, "categories": list(case.categories), "runs": runs}

    monkeypatch.setattr(report, "open_storage", lambda: "store")
    monkeypatch.setattr(report, "verify_artifact", verify_artifact)
    monkeypatch.setattr(report, "render_case", render_case)
    output = tmp_path / "out"
    report.build_report(output)
    assert rendered == [s.slug for s in report.CASE_SPECS]
    page = (output / "index.html").read_text()
    assert "{{" not in page
    assert report.diagnostic_table(rows) in page
    assert f"{len(report.CASE_SPECS)}" in page
    for spec in report.CASE_SPECS:
        assert f'<script src="assets/{spec.slug}.js"></script>' in page
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["task"] == "M3MS-008"
    assert manifest["cases"] == [s.case_id for s in report.CASE_SPECS]
    assert manifest["run_table"] == {"uri": expected.uri, "sha256": expected.sha256}
    assert {"index.html", "cases.json", "README.md", "assets/player.js", "assets/report.css"} <= set(
        manifest["outputs"]
    )
    report.verify_report(output)
    with pytest.raises(FileExistsError):
        report.build_report(output)


def test_main_verifies_or_builds_as_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """``--verify`` checks the committed presentation without the store; otherwise the build is invoked."""
    committed = repository_root() / DOCS / "report"
    monkeypatch.setattr(sys, "argv", ["render", "--output", str(committed), "--verify"])
    report.main()
    assert "verified" in capsys.readouterr().out
    built: list[Path] = []
    monkeypatch.setattr(report, "build_report", built.append)
    monkeypatch.setattr(sys, "argv", ["render", "--output", str(tmp_path / "new")])
    report.main()
    assert built == [tmp_path / "new"]
