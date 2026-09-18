# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the committed handoff freezes, pinned.

Three artifacts freeze the protocol before execution: the run ordering, the
representative-case rule, and the re-simulation subset. Each is re-derived here
from the code and its bound inputs and compared whole with what is committed. A
difference is not something to refresh: a frozen artifact is never rewritten, so
a change to what it records is a version decision.
"""

from __future__ import annotations

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import evaluation_scenarios, load_manual_evaluation_config
from arm_rc_ctrl.experiments.manual_handoff import (
    HANDOFF_VERSIONS,
    RepresentativeRule,
    ResimulationFreeze,
    RunOrdering,
    handoff_inputs,
    load_handoff,
    render_handoff_markdown,
    representative_rule,
    resimulation_freeze,
    run_ordering,
)
from arm_rc_ctrl.experiments.manual_schema import load_schema
from arm_rc_ctrl.experiments.manual_study import load_study
from arm_rc_ctrl.experiments.perturbations import load_development_robustness
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO = repository_root()
DOCS = REPO / "docs" / "experiments" / "task_1a_manual_demonstration"
STUDY = DOCS / "study_manifest_v1.json"
EVALUATION = REPO / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"
SCHEMA_V2 = DOCS / "result_schema_v2.json"
ORDERING = DOCS / "run_ordering_v1.json"
RULE = DOCS / "representative_rule_v1.json"
SUBSET = DOCS / "resimulation_subset_v1.json"

type Handoff = RunOrdering | RepresentativeRule | ResimulationFreeze


def _load(kind: str) -> Handoff:
    """One committed artifact, by kind, under its own version's name."""
    path = DOCS / f"{kind}_v{HANDOFF_VERSIONS[kind]}.json"
    if kind == "run_ordering":
        return load_handoff(path, RunOrdering)
    if kind == "representative_rule":
        return load_handoff(path, RepresentativeRule)
    return load_handoff(path, ResimulationFreeze)


def _cases() -> tuple[object, ...]:
    config = load_manual_evaluation_config(EVALUATION)
    return evaluation_scenarios(load_development_robustness(config.development), load_manual_scenario(config.scenario))


@pytest.mark.parametrize("kind", tuple(HANDOFF_VERSIONS))
def test_each_artifact_is_named_by_its_own_version(kind: str) -> None:
    """Each is versioned on its own; a change to one does not version the others."""
    assert _load(kind).schema_version == HANDOFF_VERSIONS[kind]


@pytest.mark.parametrize("kind", tuple(HANDOFF_VERSIONS))
def test_each_artifact_binds_the_committed_inputs(kind: str) -> None:
    """The study manifest, evaluation configuration and result schema v2 it was derived from."""
    assert _load(kind).inputs == handoff_inputs(STUDY, EVALUATION, SCHEMA_V2)


@pytest.mark.parametrize("kind", tuple(HANDOFF_VERSIONS))
def test_each_artifact_names_records_the_bound_schema_defines(kind: str) -> None:
    """Definitions are referenced in the committed v2 schema, not restated in the freeze."""
    defined = {record.name for record in load_schema(SCHEMA_V2).records}
    assert set(_load(kind).describes) <= defined


@pytest.mark.parametrize("kind", tuple(HANDOFF_VERSIONS))
def test_every_rendering_matches_its_artifact(kind: str) -> None:
    """The committed Markdown is generated from the record, never edited beside it."""
    markdown = DOCS / f"{kind}_v{HANDOFF_VERSIONS[kind]}.md"
    assert render_handoff_markdown(_load(kind)) == markdown.read_text(encoding="utf-8")


def test_the_ordering_is_what_the_code_derives_now() -> None:
    """186 models, 65 scenarios, two trackers and 60 banks, compared whole with the committed ordering."""
    derived = run_ordering(
        load_study(STUDY),
        scenarios=[case.scenario_id for case in _cases()],  # type: ignore[attr-defined]
        trackers=RECOVERY_TRACKERS,
        inputs=handoff_inputs(STUDY, EVALUATION, SCHEMA_V2),
    )
    assert derived == load_handoff(ORDERING, RunOrdering)


def test_the_rule_is_what_the_code_derives_now() -> None:
    """Arms, categories, scope and the statements of tie-breaking, deduplication and absence."""
    derived = representative_rule(
        load_study(STUDY),
        trackers=RECOVERY_TRACKERS,
        inputs=handoff_inputs(STUDY, EVALUATION, SCHEMA_V2),
        ordering_sha256=sha256_file(ORDERING),
    )
    assert derived == load_handoff(RULE, RepresentativeRule)


def test_the_subset_is_what_the_code_derives_now() -> None:
    """The 300 explicit identities, re-enumerated by the subset rule and compared in order."""
    derived = resimulation_freeze(
        load_study(STUDY),
        cases=_cases(),  # type: ignore[arg-type]
        trackers=RECOVERY_TRACKERS,
        inputs=handoff_inputs(STUDY, EVALUATION, SCHEMA_V2),
        ordering_sha256=sha256_file(ORDERING),
    )
    assert derived == load_handoff(SUBSET, ResimulationFreeze)


def test_the_rule_and_subset_bind_the_committed_ordering() -> None:
    """Both take their scenario order from the ordering, so they name the exact one they read."""
    digest = sha256_file(ORDERING)
    assert load_handoff(RULE, RepresentativeRule).ordering_sha256 == digest
    assert load_handoff(SUBSET, ResimulationFreeze).ordering_sha256 == digest


def test_the_frozen_counts() -> None:
    """The totals the complete study must meet, and the size of the audit sample."""
    expected = load_handoff(ORDERING, RunOrdering).expected
    assert (
        expected.n_models,
        expected.n_replay_banks,
        expected.pairs_per_model,
        expected.n_rc_runs,
        expected.n_replay_runs,
        expected.n_runs,
    ) == (186, 60, 130, 24_180, 7_800, 31_980)
    subset = load_handoff(SUBSET, ResimulationFreeze)
    assert (subset.n_rc_runs, subset.n_replay_runs, subset.n_runs) == (240, 60, 300)
