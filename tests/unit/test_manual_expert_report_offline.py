# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""CI-001: the expert report's rendering and binding contracts, exercised without the evidence store.

The store-backed tests render the committed report from real runs. These tests
feed the same renderer synthetic stored runs through its store-reading seams, so
the binding checks, the player payload and the report assembly are verified on
every checkout, including CI, which has no store.
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

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments import manual_expert_report as report
from arm_rc_ctrl.experiments.manual_evaluation import ManualRunArtifact
from arm_rc_ctrl.experiments.manual_figures import ManualFigureCase, ManualFigureInputs, load_figure_inputs
from arm_rc_ctrl.experiments.manual_report import DOCUMENT_LOCATION
from arm_rc_ctrl.experiments.manual_results import ManualRunRow, table_to_csv
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

DOCS = repository_root() / DOCUMENT_LOCATION
FIGURE_INPUTS = DOCS / "results/figure_inputs_v1.json"
FORCE_CASE = "feasible-worst__computed_torque__force-12N-000deg"
STORE = cast("StorageRoot", object())


def _inputs() -> ManualFigureInputs:
    return load_figure_inputs(FIGURE_INPUTS)


def _case(case_id: str) -> ManualFigureCase:
    return next(c for c in _inputs().cases if c.case_id == case_id)


def _spec(case_id: str) -> report.Illustration:
    return next(s for s in report.CASES if s.case_id == case_id)


def _rows(case: ManualFigureCase) -> tuple[ManualRunRow, ...]:
    """One audited row per role, bound to the figure run it describes."""
    empty: dict[str, Any] = {f.name: None for f in dc.fields(ManualRunRow)}
    rows: list[ManualRunRow] = []
    for i, run in enumerate(case.runs):
        arm = "M10" if run.role == "M10" else f"{run.role}/{case.teacher_assignment}"
        fixed: dict[str, Any] = {
            "source": "replay" if run.role == "replay" else "rc",
            "configuration": case.configuration,
            "arm": arm,
            "arm_kind": run.role,
            "parent": None if run.role == "M10" else case.teacher_assignment,
            "model_label": run.label,
            "tracker": case.tracker,
            "scenario_id": case.scenario_id,
            "scenario_class": case.scenario_class,
            "scenario_index": 0,
            "status": "completed" if run.success else "infeasible",
            "success": run.success,
            "reason": None if run.success else "final dwell too short",
            "run_uri": run.run.uri,
            "run_sha256": run.run.sha256,
            "arrays_sha256": run.run.arrays_sha256,
            "n_active_samples": 200,
            "final_endpoint_error_m": None if i == 0 else 0.001 * i,
            "time_to_final_dwell_s": None if i == 0 else 1.5 * i,
        }
        rows.append(ManualRunRow(**(empty | fixed)))
    return tuple(rows)


def _arrays(*, tau_applied: bool, tip_offset: float = 0.0) -> dict[str, NDArray[Any]]:
    """A short stored run whose tip is the forward kinematics of its joint angles."""
    scenario = load_manual_scenario(repository_root() / _inputs().scenario_file)
    lengths = np.asarray([link.length for link in scenario.robot.links], dtype=np.float64)
    t = np.arange(0.0, 10.5, 0.05, dtype=np.float64)
    q = np.stack((0.3 + 0.1 * np.sin(t), -0.8 + 0.05 * t), axis=1)
    q_desired = q + 0.01
    angles = np.cumsum(q, axis=1)
    tip = np.stack(
        (
            lengths[0] * np.cos(angles[:, 0]) + lengths[1] * np.cos(angles[:, 1]),
            lengths[0] * np.sin(angles[:, 0]) + lengths[1] * np.sin(angles[:, 1]),
        ),
        axis=1,
    )
    arrays: dict[str, NDArray[Any]] = {
        "t": t,
        "q": q,
        "q_desired": q_desired,
        "dq": np.gradient(q, t, axis=0),
        "dq_desired": np.gradient(q_desired, t, axis=0),
        "tip": tip + tip_offset,
        "tau_requested": np.full_like(q, 0.5),
    }
    if tau_applied:
        arrays["tau_applied"] = np.full_like(q, 0.4)
    return arrays


def _fake_plot_case(inputs: object, case_id: str, out: Path, *, store: object, root: Path) -> None:
    del inputs, case_id, store, root
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(out)


@pytest.fixture
def offline_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve synthetic stored runs to the renderer and record which run URIs it read."""
    read: list[str] = []

    def payload(store: object, run: ManualRunArtifact, *, where: str) -> tuple[object, dict[str, NDArray[Any]]]:
        del store, where
        read.append(run.uri)
        summary = SimpleNamespace(termination=SimpleNamespace(kind="horizon"))
        return summary, _arrays(tau_applied=len(read) % 2 == 0)

    monkeypatch.setattr(report, "load_verified_payload", payload)
    monkeypatch.setattr(report, "plot_case", _fake_plot_case)
    return read


def test_render_case_writes_bound_plots_and_a_player_payload(tmp_path: Path, offline_runs: list[str]) -> None:
    """Every role's stored run is read once and presented from its audited row, pulses included."""
    case = _case(FORCE_CASE)
    rows = _rows(case)
    info = report.render_case(case, _spec(FORCE_CASE), DOCS, tmp_path, STORE, rows)

    assert offline_runs == [run.run.uri for run in case.runs]
    slug = _spec(FORCE_CASE).slug
    for suffix in ("space", "time", "joints"):
        with Image.open(tmp_path / f"{slug}-{suffix}.png") as image:
            assert image.mode == "P"
    script = (tmp_path / f"{slug}.js").read_text()
    assert script.startswith("window.manualCases.push(")
    payload = json.loads(script.removeprefix("window.manualCases.push(").removesuffix(");\n"))
    assert payload["activation"] == case.activation_s
    assert len(payload["lengths"]) == report.JOINT_COUNT
    assert [r["role"] for r in payload["runs"]] == list(report.ROLES)
    t = _arrays(tau_applied=False)["t"] - case.activation_s
    for run, figure_run, row in zip(payload["runs"], case.runs, rows, strict=True):
        assert run["end"] == pytest.approx(float(t[-1]))
        assert run["samples"] == len(t)
        assert run["t"][-1] == pytest.approx(float(t[-1]))
        assert run["pulse"] == pytest.approx(
            [figure_run.pulse_start_s - case.activation_s, figure_run.pulse_end_s - case.activation_s]  # type: ignore[operator]
        )
        # Each displayed pulse edge keeps its nearest stored sample.
        for edge in run["pulse"]:
            assert float(t[int(np.argmin(np.abs(t - edge)))]) == pytest.approx(
                min(run["t"], key=lambda v: abs(v - edge))
            )
        expected_error = None if row.final_endpoint_error_m is None else row.final_endpoint_error_m * 1000
        assert run["final_error_mm"] == expected_error
        assert run["sha256"] == figure_run.run.sha256
        assert run["termination"] == "horizon"
    runs = cast("list[dict[str, object]]", info["runs"])
    assert info["case_id"] == FORCE_CASE
    assert all(not {"t", "q", "ref", "color"} & set(run) for run in runs)


def test_render_case_refuses_a_row_bound_to_different_run_bytes(tmp_path: Path, offline_runs: list[str]) -> None:
    """A table row whose digest names another run cannot be presented beside this run's arrays."""
    del offline_runs
    case = _case(FORCE_CASE)
    rows = list(_rows(case))
    rows[2] = dc.replace(rows[2], run_sha256="0" * 64)
    with pytest.raises(ValueError, match="table and figure bindings disagree"):
        report.render_case(case, _spec(FORCE_CASE), DOCS, tmp_path, STORE, tuple(rows))


@pytest.mark.parametrize("rows_for_role", [0, 2])
def test_render_case_requires_exactly_one_row_per_role(
    tmp_path: Path, offline_runs: list[str], rows_for_role: int
) -> None:
    """A missing or duplicated audited row is refused before any run is read."""
    case = _case(FORCE_CASE)
    rows = [r for r in _rows(case) if r.arm_kind != "R10"]
    rows += [r for r in _rows(case) if r.arm_kind == "R10"] * rows_for_role
    with pytest.raises(ValueError, match=r"/R10: expected exactly one audited run row"):
        report.render_case(case, _spec(FORCE_CASE), DOCS, tmp_path, STORE, tuple(rows))
    assert offline_runs == []


def test_render_case_refuses_a_tip_that_is_not_the_stored_joint_kinematics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The presented path is checked against the stored joints, not trusted."""

    def payload(store: object, run: object, *, where: str) -> tuple[object, dict[str, NDArray[Any]]]:
        del store, run, where
        return SimpleNamespace(termination=SimpleNamespace(kind="horizon")), _arrays(tau_applied=True, tip_offset=1e-6)

    monkeypatch.setattr(report, "load_verified_payload", payload)
    case = _case(FORCE_CASE)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        report.render_case(case, _spec(FORCE_CASE), DOCS, tmp_path, STORE, _rows(case))


def test_render_case_refuses_a_changed_task_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Figures are drawn only for the task file whose digest the figure inputs bind."""
    changed = dc.replace(_inputs(), scenario_sha256="0" * 64)

    def load(path: Path) -> ManualFigureInputs:
        del path
        return changed

    monkeypatch.setattr(report, "load_figure_inputs", load)
    case = _case(FORCE_CASE)
    with pytest.raises(ValueError, match="differs from its bound digest"):
        report.render_case(case, _spec(FORCE_CASE), DOCS, tmp_path, STORE, _rows(case))


def test_render_case_refuses_a_robot_other_than_the_bound_two_link_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planar drawing assumes two links and no base offset; anything else is refused."""
    scenario = load_manual_scenario(repository_root() / _inputs().scenario_file)
    shifted = SimpleNamespace(robot=SimpleNamespace(links=scenario.robot.links, base_length=0.1))

    def load(path: Path) -> object:
        del path
        return shifted

    monkeypatch.setattr(report, "load_manual_scenario", load)
    case = _case(FORCE_CASE)
    with pytest.raises(ValueError, match="two-link, zero-base"):
        report.render_case(case, _spec(FORCE_CASE), DOCS, tmp_path, STORE, _rows(case))


def _case_info(spec: report.Illustration, case: ManualFigureCase) -> dict[str, object]:
    """What render_case returns, with one aborted run that has no error or hold time."""
    runs: list[dict[str, object]] = [
        {
            "role": run.role,
            "success": run.success,
            "reason": None if run.success else "joint <speed> limit",
            "final_error_mm": None if i == 0 else 1.234 * i,
            "dwell_task_s": None if i == 0 else 2.5,
        }
        for i, run in enumerate(case.runs)
    ]
    return {"slug": spec.slug, "case_id": case.case_id, "categories": list(case.categories), "runs": runs}


@pytest.fixture
def offline_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Replace only the store and the per-case renderer; everything else is the committed evidence."""
    rendered: list[str] = []
    table = tmp_path / "runs.csv"
    table.write_text(table_to_csv((), ManualRunRow), encoding="utf-8")

    def render(
        case: ManualFigureCase,
        spec: report.Illustration,
        docs: Path,
        out: Path,
        store: object,
        rows: tuple[ManualRunRow, ...],
    ) -> dict[str, object]:
        del docs, store
        assert rows == ()
        rendered.append(case.case_id)
        for suffix in ("space", "time", "joints"):
            (out / f"{spec.slug}-{suffix}.png").write_bytes(spec.slug.encode())
        (out / f"{spec.slug}.js").write_text("window.manualCases.push({});\n")
        return _case_info(spec, case)

    monkeypatch.setattr(report, "open_storage", lambda: STORE)

    def verified(store: object, payload: object) -> Path:
        del store, payload
        return table

    def summaries(data: object, out: Path) -> None:
        del data
        (out / "summary.png").write_bytes(b"png")

    monkeypatch.setattr(report, "verify_artifact", verified)
    monkeypatch.setattr(report, "render_case", render)
    monkeypatch.setattr(report, "render_summaries", summaries)
    return rendered


def test_build_report_assembles_a_verifiable_portable_report(tmp_path: Path, offline_build: list[str]) -> None:
    """Every frozen illustration is rendered once, expanded into the page, and fingerprinted."""
    output = tmp_path / "report"
    report.build_report(output)

    assert offline_build == [s.case_id for s in report.CASES]
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "{{" not in page
    for spec in report.CASES:
        assert f'id="case-{spec.slug}"' in page
        assert f'<script src="assets/{spec.slug}.js"></script>' in page
    assert page.count('data-result="') == 12
    assert "joint &lt;speed&gt; limit" in page
    assert "<td>All criteria met</td>" in page
    assert "<td>—</td>" in page
    assert "<td>2.47</td>" in page
    cases = json.loads((output / "cases.json").read_text())
    assert [c["case_id"] for c in cases] == [s.case_id for s in report.CASES]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["task"] == "M3MAN-015"
    assert manifest["display"]["stride"] == 10
    assert "assets/demonstrations.png" in manifest["outputs"]
    assert "src/arm_rc_ctrl/experiments/manual_expert_report.py" in manifest["sources"]
    report.verify_report(output)

    (output / "assets/rescue.js").write_text("window.manualCases.push({'x':1});\n")
    with pytest.raises(ValueError, match=r"assets/rescue\.js: presentation fingerprint differs"):
        report.verify_report(output)


def test_build_report_never_overwrites_an_existing_report(tmp_path: Path, offline_build: list[str]) -> None:
    """A new report is always a new directory; an existing one is left untouched."""
    output = tmp_path / "report"
    output.mkdir()
    (output / "keep.txt").write_text("kept")
    with pytest.raises(FileExistsError):
        report.build_report(output)
    assert offline_build == []
    assert [p.name for p in output.iterdir()] == ["keep.txt"]


def test_build_report_refuses_an_unexpanded_template_field(
    tmp_path: Path, offline_build: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A template field the renderer does not know is an error, not literal text in the report."""
    del offline_build
    source = tmp_path / "source"
    shutil.copytree(repository_root() / report.SOURCE, source)
    with (source / "report.html").open("a", encoding="utf-8") as template:
        template.write("{{UNKNOWN_FIELD}}\n")
    monkeypatch.setattr(report, "SOURCE", source)
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="unexpanded report template field"):
        report.build_report(output)
    assert not (output / "index.html").exists()


def test_main_dispatches_verification_and_building(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--verify checks an existing report without the store; otherwise a new report is built."""
    calls: list[tuple[str, Path]] = []

    def verify(path: Path) -> None:
        calls.append(("verify", path))

    def build(path: Path) -> None:
        calls.append(("build", path))

    monkeypatch.setattr(report, "verify_report", verify)
    monkeypatch.setattr(report, "build_report", build)
    monkeypatch.setattr(sys, "argv", ["render", "--output", str(tmp_path), "--verify"])
    report.main()
    monkeypatch.setattr(sys, "argv", ["render", "--output", str(tmp_path)])
    report.main()
    assert calls == [("verify", tmp_path), ("build", tmp_path)]


def test_render_case_without_a_force_pulse_marks_no_pulse(tmp_path: Path, offline_runs: list[str]) -> None:
    """A nominal case keeps activation as its only event and records no pulse for any role."""
    case_id = "feasible-best__pd_v2__nominal"
    case = _case(case_id)
    report.render_case(case, _spec(case_id), DOCS, tmp_path, STORE, _rows(case))
    assert len(offline_runs) == len(report.ROLES)
    script = (tmp_path / f"{_spec(case_id).slug}.js").read_text()
    payload = json.loads(script.removeprefix("window.manualCases.push(").removesuffix(");\n"))
    assert [run["pulse"] for run in payload["runs"]] == [[]] * len(report.ROLES)


def test_build_report_refuses_a_teacher_figure_that_differs_from_its_binding(
    tmp_path: Path, offline_build: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copied teacher figure must be the bytes the earlier presentation manifest bound."""
    real = report.sha256_file

    def digest(path: Path) -> str:
        return "0" * 64 if path.name == "demonstrations.png" else real(path)

    monkeypatch.setattr(report, "sha256_file", digest)
    with pytest.raises(ValueError, match="teacher figure differs"):
        report.build_report(tmp_path / "report")
    assert offline_build == []
