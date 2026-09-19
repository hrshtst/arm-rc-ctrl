# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-010: the committed derived evidence is bound to its inputs and agrees with itself.

The derivation reads the external store, which continuous integration does not
have, so this lock checks everything that the committed files alone can answer:
every document matches the digest the results index recorded for it, the index
binds the frozen inputs as they are committed, the totals equal the frozen run
ordering's expectation and the accounting, and the tables agree with each other
-- each paired difference is the two arms' own per-model counts subtracted, and
the arm totals, with the all-ten arm counted once, reproduce the sweep's success
counts. A derivation that dropped, double-counted or mislabelled a run could not
pass all of these at once.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from arm_rc_ctrl.config import from_mapping
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_accounting import StudyAccounting
from arm_rc_ctrl.experiments.manual_contrasts import (
    ALL_CLASSES,
    CONTRASTS,
    PARENTS,
    ManualArmSummary,
    ManualContrastSummary,
)
from arm_rc_ctrl.experiments.manual_evaluation import evaluation_scenarios, load_manual_evaluation_config
from arm_rc_ctrl.experiments.manual_figures import load_figure_inputs
from arm_rc_ctrl.experiments.manual_handoff import RepresentativeRule, RunOrdering, load_handoff
from arm_rc_ctrl.experiments.manual_results import (
    RESULT_DOCUMENTS,
    ManualSelections,
    evidence_digest,
    load_results,
    render_results_markdown,
    table_from_csv,
)
from arm_rc_ctrl.experiments.manual_study import load_study
from arm_rc_ctrl.experiments.perturbations import load_development_robustness
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration"
RESULTS = DOCS / "results"
INDEX = RESULTS / "results_v1.json"
EVALUATION = REPO_ROOT / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"


def _document(role: str) -> Path:
    return RESULTS / RESULT_DOCUMENTS[role]


def _arms() -> tuple[ManualArmSummary, ...]:
    return table_from_csv(_document("arm_summary").read_text(encoding="utf-8"), ManualArmSummary)


def _contrasts() -> tuple[ManualContrastSummary, ...]:
    return table_from_csv(_document("contrast_summary").read_text(encoding="utf-8"), ManualContrastSummary)


def _classes() -> dict[str, int]:
    """Scenarios per class, from the evaluation configuration itself rather than from the outputs."""
    config = load_manual_evaluation_config(EVALUATION)
    cases = evaluation_scenarios(load_development_robustness(config.development), load_manual_scenario(config.scenario))
    sizes: dict[str, int] = {ALL_CLASSES: len(cases)}
    for case in cases:
        sizes[str(case.kind)] = sizes.get(str(case.kind), 0) + 1
    return sizes


# --- bindings --------------------------------------------------------------------------------------


def test_every_committed_document_matches_the_digest_the_index_recorded() -> None:
    """A document edited after derivation would no longer be the one the index vouches for."""
    results = load_results(INDEX)
    assert {d.name for d in results.documents} == set(RESULT_DOCUMENTS.values())
    for document in results.documents:
        path = RESULTS / document.name
        assert (sha256_file(path), path.stat().st_size) == (document.sha256, document.size), document.name


def test_the_index_binds_the_frozen_inputs_as_committed() -> None:
    """The outputs were derived from these exact files and these exact evidence pointers."""
    results = load_results(INDEX)
    inputs = results.inputs
    assert inputs.study_manifest_sha256 == sha256_file(DOCS / "study_manifest_v1.json")
    assert inputs.evaluation_sha256 == sha256_file(EVALUATION)
    assert inputs.run_ordering_sha256 == sha256_file(DOCS / "run_ordering_v1.json")
    assert inputs.representative_rule_sha256 == sha256_file(DOCS / "representative_rule_v1.json")
    assert inputs.result_schema_sha256 == sha256_file(DOCS / "result_schema_v3.json")
    assert (inputs.evidence_sha256, inputs.n_pointers) == evidence_digest(DOCS / "evidence")
    assert results.provenance.project_dirty is False
    assert results.provenance.exploratory is False


def test_the_markdown_index_is_generated_from_the_committed_index() -> None:
    """The rendering is regenerated, never edited beside the record."""
    rendered = render_results_markdown(load_results(INDEX))
    assert rendered == (RESULTS / "results_v1.md").read_text(encoding="utf-8")


def test_the_stored_tables_are_cited_by_digest_and_size_with_their_schema() -> None:
    """The two large tables live in the store; the index says exactly which bytes and which record."""
    results = load_results(INDEX)
    records = {t.record: t for t in results.tables}
    assert set(records) == {"ManualRunRow", "ManualContrastRow"}
    assert records["ManualRunRow"].n_rows == results.n_rc_runs + results.n_replay_runs + results.n_unavailable_runs
    ordering = load_handoff(DOCS / "run_ordering_v1.json", RunOrdering)
    classes = len(_classes())
    configurations = len({label.split("/")[0] for label in ordering.models})
    expected = configurations * len(ordering.trackers) * classes * len(CONTRASTS) * len(PARENTS)
    assert records["ManualContrastRow"].n_rows == expected


# --- totals --------------------------------------------------------------------------------------


def test_the_totals_equal_the_frozen_expectation() -> None:
    """Every run the protocol names was executed, and none was left unavailable."""
    results = load_results(INDEX)
    expected = load_handoff(DOCS / "run_ordering_v1.json", RunOrdering).expected
    assert (results.n_models, results.n_replay_banks) == (expected.n_models, expected.n_replay_banks)
    assert (results.n_rc_runs, results.n_replay_runs) == (expected.n_rc_runs, expected.n_replay_runs)
    assert results.n_unavailable_runs == 0


def test_the_accounting_is_complete_and_its_successes_are_the_indexes() -> None:
    """The accounting and the tables were derived from the same evidence, so they count the same successes."""
    mapping = cast("dict[str, object]", json.loads(_document("accounting").read_text(encoding="utf-8")))
    accounting = from_mapping(mapping, StudyAccounting)
    results = load_results(INDEX)
    assert accounting.complete
    assert accounting.all_bind_canonical_execution
    assert accounting.canonical_execution_identity == load_study(DOCS / "study_manifest_v1.json").execution.identity
    assert sum(line.n_completed for line in accounting.models) == results.n_rc_successes
    assert sum(bank.n_completed for bank in accounting.banks) == results.n_replay_successes


# --- the tables agree with each other ---------------------------------------------------------------


def test_the_arm_totals_count_the_all_ten_arm_once_and_reproduce_the_success_counts() -> None:
    """Summed over configurations and trackers, the class `all` totals are the sweep's own success counts."""
    arms = [a for a in _arms() if a.scenario_class == ALL_CLASSES]
    results = load_results(INDEX)
    assert all(a.n_models == 1 for a in arms if a.arm_kind == "M10")
    assert all(a.n_models == len(PARENTS) for a in arms if a.arm_kind != "M10")
    assert all(a.excluded_runs == 0 and a.unavailable_runs == 0 for a in arms)
    assert sum(a.successes for a in arms if a.arm_kind != "replay") == results.n_rc_successes
    assert sum(a.successes for a in arms if a.arm_kind == "replay") == results.n_replay_successes
    assert sum(a.n_runs for a in arms if a.arm_kind != "replay") == results.n_rc_runs


def test_every_class_is_summarised_over_its_own_denominator() -> None:
    """Each class's size comes from the evaluation configuration, and the classes add up to all of them."""
    sizes = _classes()
    arms = _arms()
    assert {(a.scenario_class, a.n_scenarios) for a in arms} == set(sizes.items())
    for key in {(a.configuration, a.tracker, a.arm_kind) for a in arms}:
        by_class = {a.scenario_class: a for a in arms if (a.configuration, a.tracker, a.arm_kind) == key}
        assert sum(by_class[c].successes for c in by_class if c != ALL_CLASSES) == by_class[ALL_CLASSES].successes


def test_each_parent_difference_is_the_two_arms_counts_subtracted() -> None:
    """The contrast summaries and the arm summaries were derived separately; they must agree parent by parent."""
    arms = {(a.configuration, a.tracker, a.scenario_class, a.arm_kind): a for a in _arms()}
    kinds = {name: (a, b) for name, a, b in CONTRASTS}
    summaries = _contrasts()
    assert summaries
    for s in summaries:
        kind_a, kind_b = kinds[s.contrast]
        a = arms[s.configuration, s.tracker, s.scenario_class, kind_a]
        b = arms[s.configuration, s.tracker, s.scenario_class, kind_b]
        for index, difference in enumerate(s.differences):
            count_a = a.per_model[0] if kind_a == "M10" else a.per_model[index]
            count_b = b.per_model[index]
            assert difference is not None
            assert count_a is not None
            assert count_b is not None
            assert difference == count_a - count_b, (s.configuration, s.tracker, s.scenario_class, s.contrast, index)
        assert s.n_parents == len(PARENTS)


# --- selections and figure inputs ---------------------------------------------------------------------


def test_the_frozen_rule_was_applied_to_every_configuration_and_tracker_in_its_order() -> None:
    """One application each, bound to the rule and ordering as committed, each showing its nominal case."""
    mapping = cast("dict[str, object]", json.loads(_document("selections").read_text(encoding="utf-8")))
    selections = from_mapping(mapping, ManualSelections)
    rule = load_handoff(DOCS / "representative_rule_v1.json", RepresentativeRule)
    assert selections.rule_sha256 == sha256_file(DOCS / "representative_rule_v1.json")
    assert selections.ordering_sha256 == sha256_file(DOCS / "run_ordering_v1.json")
    assert [(a.configuration, a.tracker) for a in selections.applications] == [
        (c, t) for c in rule.configurations for t in rule.trackers
    ]
    for application in selections.applications:
        assert application.arms == rule.arm_labels
        assert application.omitted == ()
        assert application.selection.categories["nominal"] == ("nominal",)


def test_every_selected_case_has_complete_figure_inputs() -> None:
    """Each case the rule chose draws its four arms and the parent's replay beside the parent's demonstration."""
    mapping = cast("dict[str, object]", json.loads(_document("selections").read_text(encoding="utf-8")))
    selections = from_mapping(mapping, ManualSelections)
    inputs = load_figure_inputs(_document("figure_inputs"))
    chosen = [
        f"{a.configuration}__{a.tracker}__{case.scenario_id}"
        for a in selections.applications
        for case in a.selection.cases
    ]
    assert [case.case_id for case in inputs.cases] == chosen
    manifest = load_study(DOCS / "study_manifest_v1.json")
    teacher = next(d for d in manifest.demonstrations if d.assignment == "D01")
    assert inputs.scenario_sha256 == sha256_file(REPO_ROOT / inputs.scenario_file)
    for case in inputs.cases:
        assert [run.role for run in case.runs] == ["S", "M10", "R10", "C10", "replay"]
        assert case.missing == ()
        assert (case.teacher_assignment, case.teacher) == ("D01", teacher.dataset)


def test_the_validation_renderings_are_of_a_case_the_inputs_hold() -> None:
    """One figure and one animation were rendered from the committed inputs to show the tools work."""
    figures = RESULTS / "figures"
    names = sorted(path.name for path in figures.iterdir())
    pngs = [n for n in names if n.endswith(".png")]
    gifs = [n for n in names if n.endswith(".gif")]
    assert (len(pngs), len(gifs), len(names)) == (1, 1, 2)
    case_ids = {case.case_id for case in load_figure_inputs(_document("figure_inputs")).cases}
    assert pngs[0].removesuffix(".png") in case_ids
    case_id, _, role = gifs[0].removesuffix(".gif").rpartition("__")
    assert case_id in case_ids
    assert role in {"S", "M10", "R10", "C10", "replay"}
