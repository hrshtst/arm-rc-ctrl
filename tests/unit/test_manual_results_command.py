# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: the derivation of the manual study's machine-readable evidence, end to end.

The real evaluation command runs the four illustrated arms of one configuration
against their parent's replay over two scenarios, and the derivation then reads
that evidence the way it reads the full study: every run verified by digest
before it is measured, every model the study names accounted for even when it
has no evidence, the large tables kept in the store behind pointers, and every
output written once. The figure tools render one case from the derived inputs.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments import manual_evaluation, manual_figures, manual_results
from arm_rc_ctrl.experiments.manual_contrasts import ManualArmSummary, ManualContrastRow, ManualContrastSummary
from arm_rc_ctrl.experiments.manual_evaluation import (
    load_manual_model_evidence,
    load_manual_pointer,
    load_verified_run,
    manual_pointer_name,
)
from arm_rc_ctrl.experiments.manual_figures import animate_case, load_figure_inputs, plot_case
from arm_rc_ctrl.experiments.manual_handoff import (
    handoff_inputs,
    handoff_to_json,
    representative_rule,
    run_ordering,
)
from arm_rc_ctrl.experiments.manual_results import (
    ManualRunRow,
    load_results,
    table_from_csv,
)
from arm_rc_ctrl.experiments.manual_schema import result_schema, schema_to_json
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import sha256_file, verify_artifact
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_evaluation import ManualPairRecord
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyManifest, StudyModel

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration"
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
CONFIGURATION = "feasible-best"
ARMS = ("S/D01", "M10", "R10/D01", "C10/D01")
"""The frozen rule's four illustrated arms, so the selection and figure inputs have something to choose."""
HOLD_S, PULSE_S, HORIZON_S = 0.05, 0.02, 1.0
NARROWED = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
)


def _evaluation_file(f: ManualFixture) -> Path:
    evaluations = f.root / "configs" / "evaluations"
    evaluations.mkdir(parents=True, exist_ok=True)
    development = evaluations / DEVELOPMENT_SOURCE.name
    shutil.copyfile(DEVELOPMENT_SOURCE, development)
    scenario = f.scenario_file
    limits = ", ".join(f"{v}" for v in load_manual_scenario(scenario).limits.velocity)
    target = evaluations / "task_1a_manual_dev_fixture.toml"
    target.write_text(
        f'name = "task-1a-manual-dev-fixture"\n'
        f'development = "{development.as_posix()}"\n'
        f'scenario = "{scenario.as_posix()}"\n'
        f"horizon_s = {HORIZON_S}\n\n"
        f"[trigger]\nhold_s = {HOLD_S}\nduration_s = {PULSE_S}\nmagnitude_n = 3.0\n\n"
        f"[simulation]\nvelocity_abort = [{limits}]\n",
        encoding="utf-8",
    )
    return target


def _entries(manifest: StudyManifest) -> tuple[StudyModel, ...]:
    return tuple(e for e in manifest.entries if e.configuration == CONFIGURATION and e.arm.label in ARMS)


def _four_arms(manifest: StudyManifest, labels: tuple[str, ...] | None = None) -> tuple[StudyModel, ...]:
    """Stand-in for the study's entry list: the rule's four arms of one configuration."""
    del labels
    return _entries(manifest)


def _narrowed(*_args: object, **_kwargs: object) -> tuple[RobustnessScenario, ...]:
    """Stand-in for the locked sixty-five: the derivation's own logic is what is under test."""
    return NARROWED


class _Study:
    """The fixture's evidence and the frozen inputs a derivation binds, built once for this module."""

    def __init__(self, f: ManualFixture, base: Path) -> None:
        self.f = f
        self.evaluation = _evaluation_file(f)
        self.evidence = base / "evidence"
        self.schema_v3 = base / "result_schema_v3.json"
        self.schema_v3.write_text(schema_to_json(result_schema(3)) + "\n", encoding="utf-8")
        inputs = handoff_inputs(f.manifest_file, self.evaluation, DOCS / "result_schema_v2.json")
        ordering = run_ordering(
            f.manifest, scenarios=[s.scenario_id for s in NARROWED], trackers=RECOVERY_TRACKERS, inputs=inputs
        )
        self.ordering = base / "run_ordering_v1.json"
        self.ordering.write_text(handoff_to_json(ordering) + "\n", encoding="utf-8")
        rule = representative_rule(
            f.manifest, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=sha256_file(self.ordering)
        )
        self.rule = base / "representative_rule_v1.json"
        self.rule.write_text(handoff_to_json(rule) + "\n", encoding="utf-8")

    def derive_argv(self, output: Path) -> list[str]:
        return [
            "derive",
            "--study",
            str(self.f.manifest_file),
            "--evaluation",
            str(self.evaluation),
            "--evidence-dir",
            str(self.evidence),
            "--run-ordering",
            str(self.ordering),
            "--representative-rule",
            str(self.rule),
            "--result-schema",
            str(self.schema_v3),
            "--output",
            str(output),
            "--exploratory",
        ]


@pytest.fixture(scope="module")
def study(manual_fixture: ManualFixture, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Study]:
    """Run the real evaluation command over the four illustrated arms, once for every test here."""
    base = tmp_path_factory.mktemp("results")
    with pytest.MonkeyPatch.context() as patch:
        for name, value in manual_fixture.env.items():
            patch.setenv(name, value)
        built = _Study(manual_fixture, base)
        patch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
        patch.setattr(manual_evaluation, "evaluation_entries", _four_arms)
        patch.setattr(manual_evaluation, "evaluation_scenarios", _narrowed)
        argv = [
            "run",
            "--study",
            str(manual_fixture.manifest_file),
            "--evaluation",
            str(built.evaluation),
            "--evidence-dir",
            str(built.evidence),
            "--exploratory",
        ]
        assert manual_evaluation.main(argv) == 0
        yield built


@pytest.fixture
def derived(study: _Study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Path:
    """Derive the evidence into a fresh directory and return it."""
    for name, value in study.f.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_results, "repository_root", lambda: study.f.root)
    monkeypatch.setattr(manual_results, "evaluation_scenarios", _narrowed)
    monkeypatch.setattr(manual_figures, "repository_root", lambda: study.f.root)
    output = tmp_path / "results"
    assert manual_results.main(study.derive_argv(output)) == 0
    capsys.readouterr()
    return output


# --- the verified loader ---------------------------------------------------------------------------


def _pair(study: _Study, label: str) -> ManualPairRecord:
    pointer = load_manual_pointer(study.evidence / manual_pointer_name("model", label))
    evidence = load_manual_model_evidence(verify_artifact(study.f.store, pointer.payload))
    return evidence.pairs[0]


def _flip_last_byte(path: Path) -> None:
    """Change the file for certain: a zip archive's last byte is often already zero."""
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))


def _arrays_file(study: _Study, pair: ManualPairRecord) -> Path:
    assert pair.run is not None
    return study.f.store.path(pair.run.uri, mode="read").parent / "arrays.npz"


def test_a_verified_run_loads_its_summary_and_arrays(study: _Study) -> None:
    """The loader returns the run the record describes, after the digest and verdict checks a resume applies."""
    pair = _pair(study, f"{CONFIGURATION}/M10")
    summary, arrays = load_verified_run(study.f.store, pair)
    assert summary.activation_s is not None
    assert set(arrays) == set(summary.arrays)


def test_a_rewritten_arrays_file_is_refused(study: _Study, tmp_path: Path) -> None:
    """A payload that no longer matches either reference it carries is never measured."""
    pair = _pair(study, f"{CONFIGURATION}/S/D01")
    arrays = _arrays_file(study, pair)
    backup = tmp_path / "arrays.npz"
    shutil.copyfile(arrays, backup)
    try:
        _flip_last_byte(arrays)
        with pytest.raises(ValueError, match="no longer matches"):
            load_verified_run(study.f.store, pair)
    finally:
        shutil.copyfile(backup, arrays)


# --- the derivation --------------------------------------------------------------------------------


def test_the_derivation_indexes_every_output_by_digest(derived: Path, study: _Study) -> None:
    """The index binds its inputs, and every stored table and committed document matches its recorded digest."""
    results = load_results(derived / "results_v1.json")
    assert results.inputs.evaluation_sha256 == sha256_file(study.evaluation)
    assert results.inputs.run_ordering_sha256 == sha256_file(study.ordering)
    for table in results.tables:
        verify_artifact(study.f.store, table.payload)
    for document in results.documents:
        assert sha256_file(derived / document.name) == document.sha256
    assert (derived / "results_v1.md").read_text(encoding="utf-8").startswith("# Task 1-a")


def test_every_run_the_protocol_names_has_a_row_and_only_real_runs_carry_verdicts(derived: Path, study: _Study) -> None:
    """Models with no evidence appear as unavailable rows rather than disappearing from the table."""
    results = load_results(derived / "results_v1.json")
    runs = next(t for t in results.tables if t.record == "ManualRunRow")
    rows = table_from_csv(verify_artifact(study.f.store, runs.payload).read_text(encoding="utf-8"), ManualRunRow)
    pairs = len(NARROWED) * len(RECOVERY_TRACKERS)
    assert len(rows) == runs.n_rows == len(study.f.manifest.entries) * pairs + pairs  # one replay bank
    simulated = [row for row in rows if row.status in ("completed", "infeasible")]
    assert len(simulated) == (len(ARMS) + 1) * pairs
    assert all(row.success is not None and row.n_active_samples for row in simulated)
    assert all(row.success is None for row in rows if row.status == "unavailable")
    assert results.n_rc_runs == len(ARMS) * pairs
    assert results.n_replay_runs == pairs


def test_the_comparisons_that_exist_are_complete_and_the_rest_are_unavailable(derived: Path, study: _Study) -> None:
    """Parent D01's four arms are all present, so its comparisons are complete; every other parent's is not."""
    results = load_results(derived / "results_v1.json")
    table = next(t for t in results.tables if t.record == "ManualContrastRow")
    rows = table_from_csv(verify_artifact(study.f.store, table.payload).read_text(encoding="utf-8"), ManualContrastRow)
    ours = [r for r in rows if r.configuration == CONFIGURATION and r.parent == "D01"]
    assert {r.status for r in ours} == {"complete"}
    assert {r.status for r in rows if r.parent == "D02"} == {"unavailable"}


def test_the_committed_summaries_parse_and_count_the_all_ten_arm_once(derived: Path) -> None:
    """The class summaries carry one all-ten model, and the contrast summaries one difference per parent."""
    arms = table_from_csv((derived / "arm_summary_v1.csv").read_text(encoding="utf-8"), ManualArmSummary)
    m10 = next(a for a in arms if (a.configuration, a.arm_kind, a.scenario_class) == (CONFIGURATION, "M10", "all"))
    assert (m10.n_models, m10.n_runs) == (1, len(NARROWED))
    summaries = table_from_csv((derived / "contrast_summary_v1.csv").read_text(encoding="utf-8"), ManualContrastSummary)
    summary = next(
        s for s in summaries if (s.configuration, s.contrast, s.scenario_class) == (CONFIGURATION, "M10-S", "all")
    )
    assert summary.n_parents == 1
    assert summary.differences[0] is not None
    assert all(d is None for d in summary.differences[1:])


def test_the_accounting_names_every_model_without_evidence(derived: Path, study: _Study) -> None:
    """The study is partial here, and the accounting says so model by model."""
    accounting = json.loads((derived / "accounting_v1.json").read_text(encoding="utf-8"))
    assert accounting["n_present"] == len(ARMS)
    assert accounting["n_missing"] == len(study.f.manifest.entries) - len(ARMS)
    assert accounting["complete"] is False


def test_the_rule_selects_the_nominal_case_and_the_figure_inputs_bind_its_runs(derived: Path) -> None:
    """Every application shows its nominal comparison; its figure draws four arms, the replay, and the teacher."""
    selections = json.loads((derived / "selections_v1.json").read_text(encoding="utf-8"))
    ours = [a for a in selections["applications"] if a["configuration"] == CONFIGURATION]
    assert len(ours) == len(RECOVERY_TRACKERS)
    assert all(a["selection"]["categories"]["nominal"] == ["nominal"] for a in ours)
    inputs = load_figure_inputs(derived / "figure_inputs_v1.json")
    case = next(c for c in inputs.cases if c.case_id == f"{CONFIGURATION}__pd_v2__nominal")
    assert [r.role for r in case.runs] == ["S", "M10", "R10", "C10", "replay"]
    assert case.teacher_assignment == "D01"
    assert case.missing == ()


def test_deriving_again_over_existing_outputs_is_refused(derived: Path, study: _Study) -> None:
    """Derived evidence is versioned: a second derivation into the same place is refused, not merged."""
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        manual_results.main(study.derive_argv(derived))


def test_a_tampered_run_stops_the_derivation(study: _Study, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run whose payload changed after it was recorded is never measured into the tables."""
    for name, value in study.f.env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(manual_results, "repository_root", lambda: study.f.root)
    monkeypatch.setattr(manual_results, "evaluation_scenarios", _narrowed)
    pair = _pair(study, f"{CONFIGURATION}/C10/D01")
    arrays = _arrays_file(study, pair)
    backup = tmp_path / "arrays.npz"
    shutil.copyfile(arrays, backup)
    try:
        _flip_last_byte(arrays)
        with pytest.raises(ValueError, match="no longer matches"):
            manual_results.main(study.derive_argv(tmp_path / "results"))
    finally:
        shutil.copyfile(backup, arrays)


# --- the figure tools ------------------------------------------------------------------------------


def test_one_case_renders_as_a_plot_and_one_run_as_an_animation(derived: Path, study: _Study, tmp_path: Path) -> None:
    """The inputs are sufficient to draw the demonstration, the commanded reference and the actual motion."""
    inputs = load_figure_inputs(derived / "figure_inputs_v1.json")
    case_id = f"{CONFIGURATION}__pd_v2__nominal"
    png = plot_case(inputs, case_id, tmp_path / "case.png", store=study.f.store, root=study.f.root)
    gif = animate_case(
        inputs, case_id, "M10", tmp_path / "case.gif", store=study.f.store, root=study.f.root, fps=5, stride=20
    )
    assert png.read_bytes()[:4] == b"\x89PNG"
    assert gif.read_bytes()[:6] in (b"GIF87a", b"GIF89a")
    with pytest.raises(FileExistsError):
        plot_case(inputs, case_id, png, store=study.f.store, root=study.f.root)
    with pytest.raises(KeyError, match="no case"):
        plot_case(inputs, "invented", tmp_path / "other.png", store=study.f.store, root=study.f.root)
