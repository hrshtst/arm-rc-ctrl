# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the handoff freezes -- run ordering, the illustration rule, and the re-simulation subset.

Each freezes part of the protocol before execution: what will run and in what
order, how the report will choose its illustrations, and which runs a
clean-checkout audit redoes. None of them holds a result. Populated accounts and
actual selections are derived from the evidence afterwards; what is frozen here
is only what they will be derived by, and against which inputs.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import evaluation_scenarios, load_manual_evaluation_config
from arm_rc_ctrl.experiments.manual_handoff import (
    RepresentativeRule,
    ResimulationFreeze,
    RunOrdering,
    handoff_inputs,
    handoff_to_json,
    load_handoff,
    main,
    representative_rule,
    resimulation_freeze,
    run_ordering,
)
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_representative import ILLUSTRATION_ARMS, ILLUSTRATION_CATEGORIES
from arm_rc_ctrl.experiments.manual_resimulation import RESIMULATION_CLASSES, resimulation_subset
from arm_rc_ctrl.experiments.manual_schema import RECORDS_BY_VERSION
from arm_rc_ctrl.experiments.manual_study import load_study
from arm_rc_ctrl.experiments.perturbations import load_development_robustness
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_handoff import HandoffInputs
    from arm_rc_ctrl.experiments.manual_study import StudyManifest
    from arm_rc_ctrl.experiments.perturbations import RobustnessScenario

REPO = repository_root()
DOCS = REPO / "docs" / "experiments" / "task_1a_manual_demonstration"
STUDY = DOCS / "study_manifest_v1.json"
EVALUATION = REPO / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"
SCHEMA_V2 = DOCS / "result_schema_v2.json"
PARENT = "D01"
OTHER = "0" * 64


@pytest.fixture(scope="module")
def manifest() -> StudyManifest:
    """The committed study manifest the freezes are derived from."""
    return load_study(STUDY)


@pytest.fixture(scope="module")
def cases() -> tuple[RobustnessScenario, ...]:
    """The 65 locked development cases, from the real evaluation configuration."""
    config = load_manual_evaluation_config(EVALUATION)
    return evaluation_scenarios(load_development_robustness(config.development), load_manual_scenario(config.scenario))


@pytest.fixture(scope="module")
def inputs() -> HandoffInputs:
    """The three input files, bound by digest."""
    return handoff_inputs(STUDY, EVALUATION, SCHEMA_V2)


@pytest.fixture(scope="module")
def ordering(manifest: StudyManifest, cases: tuple[RobustnessScenario, ...], inputs: HandoffInputs) -> RunOrdering:
    """The run ordering exactly as the command would freeze it."""
    return run_ordering(
        manifest, scenarios=[case.scenario_id for case in cases], trackers=RECOVERY_TRACKERS, inputs=inputs
    )


# --- the inputs every freeze binds ----------------------------------------------------------------


def test_the_inputs_are_bound_by_digest(inputs: HandoffInputs) -> None:
    """A freeze names the files it was derived from, so it cannot be read against different ones."""
    assert inputs.study_manifest_sha256 == sha256_file(STUDY)
    assert inputs.evaluation_sha256 == sha256_file(EVALUATION)
    assert inputs.result_schema_sha256 == sha256_file(SCHEMA_V2)
    assert inputs.result_schema_version == 2


def test_a_schema_that_is_not_version_two_is_refused(tmp_path: Path) -> None:
    """The freezes reference the v2 record definitions; binding v1 would reference records it lacks."""
    with pytest.raises(ValueError, match="version"):
        handoff_inputs(STUDY, EVALUATION, DOCS / "result_schema_v1.json")
    del tmp_path


# --- run ordering ---------------------------------------------------------------------------------


def test_the_ordering_names_every_model_in_manifest_order(manifest: StudyManifest, ordering: RunOrdering) -> None:
    """All 186 models, in the manifest's own order."""
    assert ordering.models == tuple(entry.label for entry in manifest.entries)
    assert ordering.expected.n_models == len(manifest.entries) == 186


def test_the_ordering_carries_the_locked_scenarios_and_trackers(
    cases: tuple[RobustnessScenario, ...], ordering: RunOrdering
) -> None:
    """The 65 locked cases and both trackers, in order."""
    assert ordering.scenarios == tuple(case.scenario_id for case in cases)
    assert len(ordering.scenarios) == 65
    assert ordering.trackers == tuple(RECOVERY_TRACKERS)


def test_replay_banks_are_keyed_by_protocol_not_environment(manifest: StudyManifest, ordering: RunOrdering) -> None:
    """A bank's identity hash includes the execution identity; its protocol key does not, so it is what is frozen."""
    expected = [(c.label, a) for c in manifest.configurations for a in ASSIGNMENTS]
    assert [(bank.configuration, bank.assignment) for bank in ordering.replay_banks] == expected
    for bank in ordering.replay_banks:
        configuration = manifest.configuration(bank.configuration)
        assert bank.warmup_s == configuration.warmup_s
        assert bank.velocity_cutoff_hz == configuration.velocity_cutoff_hz
        assert bank.acceleration_cutoff_hz == configuration.acceleration_cutoff_hz


def test_the_ordering_expands_to_the_whole_study(ordering: RunOrdering) -> None:
    """The compact form expands deterministically to every run, in a stated nesting."""
    rc = list(ordering.rc_runs())
    replay = list(ordering.replay_runs())
    assert len(rc) == ordering.expected.n_rc_runs == 24_180
    assert len(replay) == ordering.expected.n_replay_runs == 7_800
    assert ordering.expected.n_runs == 31_980
    assert rc[0] == (ordering.models[0], ordering.scenarios[0], ordering.trackers[0])
    assert rc[-1] == (ordering.models[-1], ordering.scenarios[-1], ordering.trackers[-1])
    assert len(set(rc)) == len(rc), "every RC run is named once"
    assert len(set(replay)) == len(replay), "every replay run is named once"


def test_totals_that_do_not_re_derive_are_refused(ordering: RunOrdering) -> None:
    """Expected totals are stated for the accounting to meet, so they must follow from the lists."""
    wrong = dataclasses.replace(ordering.expected, n_rc_runs=ordering.expected.n_rc_runs - 1)
    with pytest.raises(ValueError, match="re-derive"):
        dataclasses.replace(ordering, expected=wrong)


# --- the illustration rule ----------------------------------------------------------------------


def test_the_rule_states_what_the_code_applies(manifest: StudyManifest, inputs: HandoffInputs) -> None:
    """Arms, categories, parent and scope are the ones the selection code enforces."""
    rule = representative_rule(manifest, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=OTHER)
    assert rule.arms == ILLUSTRATION_ARMS
    assert rule.parent == PARENT
    assert tuple(category.name for category in rule.categories) == ILLUSTRATION_CATEGORIES
    assert rule.configurations == tuple(c.label for c in manifest.configurations)
    assert rule.trackers == tuple(RECOVERY_TRACKERS)
    assert rule.ordering_sha256 == OTHER, "the scenario order it breaks ties by is the frozen ordering's"
    assert all(category.selects.strip() for category in rule.categories)


# --- the re-simulation subset ---------------------------------------------------------------------


def test_the_subset_names_240_rc_and_60_replay_runs(
    manifest: StudyManifest, cases: tuple[RobustnessScenario, ...], inputs: HandoffInputs
) -> None:
    """Every audited run is listed, with counts stated explicitly."""
    subset = resimulation_freeze(
        manifest, cases=cases, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=OTHER
    )
    assert (subset.n_rc_runs, subset.n_replay_runs, subset.n_runs) == (240, 60, 300)
    assert len(subset.rc_runs) == 240
    assert len(subset.replay_runs) == 60


def test_the_subset_agrees_with_the_enumeration_it_freezes(
    manifest: StudyManifest, cases: tuple[RobustnessScenario, ...], inputs: HandoffInputs
) -> None:
    """The explicit identities are the subset rule's own output, in its frozen order."""
    subset = resimulation_freeze(
        manifest, cases=cases, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=OTHER
    )
    enumerated = resimulation_subset(manifest, scenarios=cases, trackers=RECOVERY_TRACKERS).run_identities()
    assert tuple((run.model, run.scenario_id, run.tracker) for run in subset.rc_runs) == enumerated
    assert subset.classes == RESIMULATION_CLASSES
    kinds = {case.scenario_id: case.kind for case in cases}
    assert tuple(kinds[scenario] for scenario in subset.scenarios) == RESIMULATION_CLASSES
    assert {(run.configuration, run.assignment) for run in subset.replay_runs} == {
        (c.label, PARENT) for c in manifest.configurations
    }


# --- what every freeze shares ---------------------------------------------------------------------


def test_each_freeze_names_result_schema_records_that_exist(
    manifest: StudyManifest, cases: tuple[RobustnessScenario, ...], inputs: HandoffInputs, ordering: RunOrdering
) -> None:
    """A freeze references its definitions in the v2 schema by name and digest instead of restating them."""
    rule = representative_rule(manifest, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=OTHER)
    subset = resimulation_freeze(
        manifest, cases=cases, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=OTHER
    )
    for freeze in (ordering, rule, subset):
        assert freeze.describes, "each freeze names the records it relates to"
        assert set(freeze.describes) <= set(RECORDS_BY_VERSION[2])


def test_every_freeze_round_trips_through_its_json(
    manifest: StudyManifest,
    cases: tuple[RobustnessScenario, ...],
    inputs: HandoffInputs,
    ordering: RunOrdering,
    tmp_path: Path,
) -> None:
    """Each freeze rebuilds exactly from what it writes."""
    rule = representative_rule(manifest, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=OTHER)
    subset = resimulation_freeze(
        manifest, cases=cases, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=OTHER
    )
    for record, cls in ((ordering, RunOrdering), (rule, RepresentativeRule), (subset, ResimulationFreeze)):
        path = tmp_path / f"{cls.__name__}.json"
        path.write_text(handoff_to_json(record) + "\n", encoding="utf-8")
        assert load_handoff(path, cls) == record


# --- the command ----------------------------------------------------------------------------------


def _freeze(artifact: str, docs: Path) -> int:
    return main(
        [
            "freeze",
            "--artifact",
            artifact,
            "--study",
            str(STUDY),
            "--evaluation",
            str(EVALUATION),
            "--result-schema",
            str(SCHEMA_V2),
            "--docs",
            str(docs),
        ]
    )


def test_the_command_freezes_all_three_under_their_own_versions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three artifacts, each named by its own version, the later two binding the ordering's digest."""
    for artifact in ("run_ordering", "representative_rule", "resimulation_subset"):
        assert _freeze(artifact, tmp_path) == 0
    capsys.readouterr()
    for name in ("run_ordering_v1", "representative_rule_v1", "resimulation_subset_v1"):
        assert (tmp_path / f"{name}.json").exists()
        assert (tmp_path / f"{name}.md").exists()
    ordering_digest = sha256_file(tmp_path / "run_ordering_v1.json")
    assert load_handoff(tmp_path / "representative_rule_v1.json", RepresentativeRule).ordering_sha256 == ordering_digest
    assert load_handoff(tmp_path / "resimulation_subset_v1.json", ResimulationFreeze).ordering_sha256 == ordering_digest


def test_a_frozen_version_is_never_overwritten(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A change to a frozen artifact is a new version, never a rewrite."""
    assert _freeze("run_ordering", tmp_path) == 0
    capsys.readouterr()
    with pytest.raises(FileExistsError, match="version"):
        _freeze("run_ordering", tmp_path)


def test_the_rule_and_subset_are_not_frozen_before_the_ordering(tmp_path: Path) -> None:
    """Both break ties and order runs by the frozen ordering, so it has to exist first."""
    for artifact in ("representative_rule", "resimulation_subset"):
        with pytest.raises(FileNotFoundError, match="run_ordering"):
            _freeze(artifact, tmp_path)


def test_an_ordering_frozen_against_other_inputs_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The ordering a later freeze binds must be of the same study, evaluation and schema."""
    assert _freeze("run_ordering", tmp_path) == 0
    capsys.readouterr()
    path = tmp_path / "run_ordering_v1.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["inputs"]["evaluation_sha256"] = OTHER
    path.write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs"):
        _freeze("representative_rule", tmp_path)
