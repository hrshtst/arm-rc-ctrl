# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-004: the paired pilot evaluation on the planar fixture: conditions, first failure, resume, evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.records import ArtifactRecord, Origin, Payload
from arm_rc_ctrl.experiments import repetition_evaluation
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.repetition_evaluation import (
    C11_CAVEAT,
    PAIR_STATUSES,
    BudgetExceededError,
    ExecutionBudget,
    ModelEvidence,
    PairRecord,
    RepetitionEvaluationConfig,
    SimulationLimits,
    augmentation_bank_record,
    check_bank_prefix,
    load_evaluation_config,
    load_model_evidence,
    load_pointer,
    load_replay_bank,
    main,
    pointer_name,
)
from arm_rc_ctrl.experiments.repetition_fits import FitStore
from arm_rc_ctrl.experiments.repetition_fixture import (
    DOCS,
    PLANAR_DIGESTS,
    PLANAR_SCENARIOS,
    PLANAR_TRACKER,
    WARMUP_ROWS,
    CraftedSimulator,
    N,
    PlanarFixture,
    build_pilot_runner,
    committed_numerical_binding,
    write_pilot_evaluation_config,
)
from arm_rc_ctrl.experiments.repetition_numerics import PanelContext
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec
from arm_rc_ctrl.experiments.run_record import RunPointerRecord, RunSummary, load_run
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from arm_rc_ctrl.execution import ExecutionRecord

REPO_ROOT = repository_root()
EVALUATION_FILE = REPO_ROOT / "configs" / "evaluations" / "task_1a_repetition_dev_v1.toml"


def _pointer(store: StorageRoot, artifact_id: str) -> RunPointerRecord:
    """Rebuild the run pointer from the stored run summary (the pilot keeps no per-run Git pointers)."""
    directory = store.path(f"armrc://runs/{artifact_id}/run.json", mode="read").parent
    summary_file = directory / "run.json"
    summary = RunSummary.from_json(summary_file.read_text(encoding="utf-8"))
    payload = Payload(
        f"armrc://runs/{artifact_id}/run.json", sha256_file(summary_file), summary_file.stat().st_size, "run.json", 1
    )
    provenance = summary.provenance
    return RunPointerRecord(
        artifact=ArtifactRecord(
            artifact_id=artifact_id,
            kind="run",
            created_at=provenance.created_at,
            license="LicenseRef-Private",
            access="private",
            payload=payload,
            origin=Origin(
                command="x",
                config_sha256=provenance.config_sha256,
                project_commit=provenance.project_commit,
                project_dirty=provenance.project_dirty,
                dependency_commits={},
                sources=(),
            ),
        ),
        method=summary.method,
        scenario=summary.scenario,
        termination_kind=summary.termination.kind,
        success=summary.outcome.success,
        duration_s=summary.duration_s,
        n_samples=int(summary.arrays["t"].shape[0]),
        arrays_sha256=summary.arrays_sha256,
    )


# --- configuration and identities -------------------------------------------------------


def test_committed_evaluation_config_relaxes_the_abort_only() -> None:
    """The versioned pilot config names the locked development levels and the 12 rad/s symmetric abort."""
    config = load_evaluation_config(EVALUATION_FILE)
    assert config.name == "task-1a-repetition-dev-v1"
    assert config.development.name == "task_1a_recovery_dev_v1.toml"
    assert config.development.is_file()
    assert config.simulation.velocity_abort == (12.0, 12.0)
    with pytest.raises(ValueError, match="positive finite"):
        SimulationLimits(velocity_abort=(12.0, 0.0))
    with pytest.raises(ValueError, match="development levels only"):
        RepetitionEvaluationConfig(
            name="x", development=Path("task_1a_recovery_confirmatory_v1.toml"), simulation=config.simulation
        )
    with pytest.raises(ValueError, match="name must not be empty"):
        RepetitionEvaluationConfig(name=" ", development=config.development, simulation=config.simulation)


def test_numerical_binding_carries_the_accepted_exception() -> None:
    """The M3REP-003 evidence binds with its one exception and the C11 caveat text."""
    binding = committed_numerical_binding()
    assert not binding.all_passed
    assert binding.comparisons == 72
    assert binding.comparisons_passed == 71
    assert binding.exception_candidate_identity is not None
    assert binding.caveat == C11_CAVEAT
    with pytest.raises(ValueError, match="contradicts"):
        replace(binding, comparisons_passed=72)
    with pytest.raises(ValueError, match="binds its exception"):
        replace(binding, exception_candidate_identity=None, exception_reference_identity=None)


def test_conditions_identity_binds_limits_warmup_trackers_and_environment(fixture: PlanarFixture) -> None:
    """Changing the velocity abort, the warm-up, a tracker digest, the estimator, or the environment changes the key."""
    f = fixture
    base = build_pilot_runner(f).conditions(f.entry)
    replay = base.replay
    assert base.velocity_abort == replay.velocity_abort == (40.0, 40.0)
    assert replay.historical_velocity_limit == tuple(f.inputs.scenario.limits.velocity)
    assert replay.scenario_ids == tuple(s.scenario_id for s in PLANAR_SCENARIOS)
    assert len(base.pairs) == 10
    assert base.pairs[0] == ("nominal", "pd_v2")
    assert base.warmup_s == 0.25
    relaxed = build_pilot_runner(f, velocity_abort=(60.0, 60.0)).conditions(f.entry)
    assert relaxed.identity != base.identity
    assert relaxed.replay.identity != replay.identity
    assert replace(replay, warmup_s=1.0).identity != replay.identity
    assert replace(replay, trackers={"pd_v2": "c" * 64, "computed_torque": "b" * 64}).identity != replay.identity
    assert replace(replay, execution_identity="e" * 64).identity != replay.identity
    # The estimator is model-side: it changes the evaluation key but not the replay bank key.
    with_other_estimator = replace(base, estimator=replace(base.estimator, max_dt_ratio=2.0))
    assert with_other_estimator.identity != base.identity
    assert with_other_estimator.replay.identity == replay.identity
    assert base.identity == build_pilot_runner(f).conditions(f.entry).identity
    with pytest.raises(ValueError, match="same joints"):
        replace(replay, velocity_abort=(40.0,))
    with pytest.raises(ValueError, match="distinct scenario ids"):
        replace(replay, scenario_ids=("a", "a"))
    with pytest.raises(ValueError, match="tracker_order"):
        replace(replay, tracker_order=("pd_v2",))
    assert replay.tracker_order == ("pd_v2", "computed_torque")
    with pytest.raises(ValueError, match="execution identity"):
        build_pilot_runner(replace(f, inputs=replace(f.inputs, execution_identity="e" * 64)))


# --- the real simulator: replay bank and the velocity abort -----------------------------


def test_real_replay_bank_is_feasible_and_diagnosed(fixture: PlanarFixture) -> None:
    """The fixture's direct replay under the gentle tracker completes every pair without saturation."""
    f = fixture
    log: list[str] = []
    bank = build_pilot_runner(f, log=log).replay_bank(f.entry)
    assert bank.complete
    assert bank.identity == build_pilot_runner(f).conditions(f.entry).replay.identity
    assert [p.status for p in bank.pairs] == ["completed"] * 10
    assert [(p.scenario_id, p.tracker) for p in bank.pairs] == list(bank.conditions.pairs)
    assert len(log) == 10
    for pair in bank.pairs:
        assert pair.arm == "replay"
        assert pair.replay is not None
        assert pair.replay.feasible
        assert pair.run is not None
        assert pair.velocity is not None
        assert pair.velocity.abort is None
        assert pair.velocity.terminal_state is None
        assert pair.velocity.abort_limit == (40.0, 40.0)
        assert not pair.crossed_historical
        loaded = load_run(f.store, _pointer(f.store, pair.run.artifact_id))
        assert loaded.summary.method == f"replay+{pair.tracker}"
        assert loaded.summary.activation_s == f.entry.warmup_s
        assert loaded.summary.arrays_sha256 == pair.run.arrays_sha256
        assert loaded.summary.provenance.seeds == {}
    # A second runner serves the bank from the store without simulating.
    log2: list[str] = []
    again = build_pilot_runner(f, log=log2).replay_bank(f.entry)
    assert again == bank
    assert log2 == []
    bank_dir = f.store.root / "reports" / "task_1a_repetition_v1" / "replay" / bank.identity
    (manifest,) = sorted(bank_dir.glob("manifest-*.json"))
    assert load_replay_bank(manifest) == bank


def test_real_rc_gate_failure_stops_the_sweep_and_marks_the_rest_unexecuted(fixture: PlanarFixture) -> None:
    """A tight abort makes the first RC run infeasible; later pairs are unexecuted, not missing or successful."""
    f = fixture
    evidence = build_pilot_runner(f, velocity_abort=(0.05, 0.05)).evaluate(f.entry, ArmSpec("absolute", "R", 16))
    assert evidence.status == "rc_gate_failure"
    assert evidence.first_failure is not None
    assert evidence.first_failure.startswith("scenario 0 [pd_v2]: limit_violation:joint_velocity")
    assert [p.status for p in evidence.pairs] == ["infeasible", *(["unexecuted"] * 9)]
    assert evidence.n_infeasible == 1
    assert evidence.n_unexecuted == 9
    assert evidence.cells == {}
    first = evidence.pairs[0]
    assert first.rc is not None
    assert first.rc.limit == "joint_velocity"
    assert first.run is not None
    assert first.velocity is not None
    assert first.velocity.abort is not None
    assert first.velocity.abort.bound == 0.05
    assert first.velocity.terminal_state is not None
    assert first.velocity.terminal_state.step == first.velocity.abort.step
    assert first.velocity.samples == first.velocity.abort.step + 1
    loaded = load_run(f.store, _pointer(f.store, first.run.artifact_id))
    assert loaded.summary.termination.kind == "limit_violation"
    assert loaded.arrays.n_samples == first.velocity.abort.step  # the offending state never enters the telemetry
    # The replay bank of the tight abort is a different bank (the resolved limits are part of the key).
    assert evidence.conditions.velocity_abort == (0.05, 0.05)
    assert evidence.replay_bank != build_pilot_runner(f).conditions(f.entry).replay.identity
    for pair in evidence.pairs[1:]:
        assert pair.run is None
        assert pair.rc is None
        assert pair.velocity is None
        assert pair.crossed_historical is None


# --- the sweep on crafted runs -----------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated(fixture: PlanarFixture) -> tuple[CraftedSimulator, ModelEvidence]:
    """The absolute S arm of the fixture entry over the five scenarios and both trackers, on crafted runs."""
    fake = CraftedSimulator(fixture.samples)
    evidence = build_pilot_runner(fixture, velocity_abort=(41.0, 41.0), simulate_fn=fake).evaluate(
        fixture.entry, ArmSpec("absolute", "S")
    )
    return fake, evidence


def test_feasible_sweep_records_every_pair_with_runs_and_diagnostics(
    fixture: PlanarFixture, evaluated: tuple[CraftedSimulator, ModelEvidence]
) -> None:
    """Ten RC pairs completed against ten replay pairs; every run is stored, digest-bound, and diagnosed."""
    f = fixture
    fake, evidence = evaluated
    assert (fake.replay_calls, fake.rc_calls) == (10, 10)
    assert set(fake.aborts) == {(41.0, 41.0)}
    assert evidence.status == "feasible"
    assert evidence.first_failure is None
    assert evidence.n_pairs == evidence.n_completed == 10
    assert evidence.n_unexecuted == evidence.n_infeasible == evidence.n_replay_blocked == 0
    assert set(evidence.cells) == {
        "posture_small:pd_v2",
        "posture_small:computed_torque",
        "posture_large:pd_v2",
        "posture_large:computed_torque",
    }
    # Crafted command gaps are a quarter of the crafted replay gaps (same ramp), so every cell ratio is 0.25.
    assert all(v == pytest.approx(0.25, rel=1e-6) for v in evidence.cells.values())
    assert evidence.fit is not None
    assert evidence.fit.arm == ArmSpec("absolute", "S")
    assert evidence.fit.formulation == "absolute"
    assert evidence.augmentation is None
    assert evidence.augmentation_prefix_verified is None
    assert C11_CAVEAT in evidence.caveats
    assert evidence.numerical == committed_numerical_binding()
    assert evidence.execution == f.execution
    assert [p.status for p in evidence.pairs] == ["completed"] * 10
    assert [(p.scenario_id, p.tracker) for p in evidence.pairs] == list(evidence.conditions.pairs)
    assert not evidence.crossed_historical
    for pair in evidence.pairs:
        assert pair.arm == "rc"
        assert pair.run is not None
        assert pair.rc is not None
        assert pair.velocity is not None
        assert pair.rc.feasible
        assert pair.rc.generated_criteria == {"generated_dwell_in_tolerance": True, "generated_dwell_stationary": True}
        assert pair.velocity.abort is None
        assert pair.velocity.samples == WARMUP_ROWS + N
        loaded = load_run(f.store, _pointer(f.store, pair.run.artifact_id))
        assert loaded.summary.method == f"rc+{pair.tracker}"
        assert loaded.summary.provenance.seeds == {"reservoir": f.inputs.base.esn.reservoir.seed}
    manifest_dir = f.store.root / "reports" / "task_1a_repetition_v1" / "models" / evidence.evaluation_identity
    (manifest,) = sorted(manifest_dir.glob("manifest-*.json"))
    assert load_model_evidence(manifest) == evidence


def test_resume_serves_completed_runs_and_refuses_corruption(
    fixture: PlanarFixture, evaluated: tuple[CraftedSimulator, ModelEvidence]
) -> None:
    """A second evaluation re-simulates nothing and reproduces the manifest; a tampered run stops the resume."""
    f = fixture
    _fake, evidence = evaluated
    counting = CraftedSimulator(f.samples)
    resumed = build_pilot_runner(f, velocity_abort=(41.0, 41.0), simulate_fn=counting).evaluate(
        f.entry, ArmSpec("absolute", "S")
    )
    assert (counting.replay_calls, counting.rc_calls) == (0, 0)
    assert resumed == evidence
    pair = evidence.pairs[3]
    assert pair.run is not None
    run_json = f.store.path(pair.run.uri, mode="read")
    original = run_json.read_bytes()
    run_json.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="no longer matches its record"):
        build_pilot_runner(f, velocity_abort=(41.0, 41.0), simulate_fn=counting).evaluate(
            f.entry, ArmSpec("absolute", "S")
        )
    run_json.write_bytes(original)
    again = build_pilot_runner(f, velocity_abort=(41.0, 41.0), simulate_fn=counting).evaluate(
        f.entry, ArmSpec("absolute", "S")
    )
    assert again == evidence
    assert (counting.replay_calls, counting.rc_calls) == (0, 0)


def test_interrupted_sweep_resumes_at_run_granularity(fixture: PlanarFixture) -> None:
    """A sweep interrupted after three RC runs continues from the fourth without re-running the first three."""
    f = fixture
    arm = ArmSpec("residual", "S")
    with pytest.raises(KeyboardInterrupt):
        build_pilot_runner(
            f, velocity_abort=(41.0, 41.0), simulate_fn=CraftedSimulator(f.samples, interrupt_after_rc=3)
        ).evaluate(f.entry, arm)
    counting = CraftedSimulator(f.samples)
    evidence = build_pilot_runner(f, velocity_abort=(41.0, 41.0), simulate_fn=counting).evaluate(f.entry, arm)
    assert (counting.replay_calls, counting.rc_calls) == (0, 7)
    assert evidence.status == "feasible"
    assert evidence.n_completed == 10
    assert evidence.fit is not None
    assert evidence.fit.formulation == "residual"
    for pair in evidence.pairs:
        assert pair.run is not None
        loaded = load_run(f.store, _pointer(f.store, pair.run.artifact_id))
        assert "generator_increment_q" in loaded.arrays.arrays


def test_replay_blocked_models_are_labelled_and_reported_apart(fixture: PlanarFixture) -> None:
    """A posture-class replay failure blocks the paired RC run (C7): replay-blocked model, the rest unexecuted."""
    f = fixture
    fake = CraftedSimulator(f.samples, blocked_replays=("small-2",))
    evidence = build_pilot_runner(f, velocity_abort=(42.0, 42.0), simulate_fn=fake).evaluate(
        f.entry, ArmSpec("absolute", "R-scaled", 16)
    )
    assert evidence.status == "replay_blocked"
    assert evidence.first_failure is not None
    assert evidence.first_failure.startswith("scenario 2 [pd_v2]: replay_infeasible:saturation")
    assert [p.status for p in evidence.pairs] == ["completed"] * 4 + ["replay_blocked"] + ["unexecuted"] * 5
    blocked = evidence.pairs[4]
    assert blocked.rc is not None
    assert blocked.rc.termination == "not_simulated"
    assert blocked.run is None
    assert blocked.velocity is None
    assert evidence.n_replay_blocked == 1
    assert evidence.n_unexecuted == 5
    assert fake.rc_calls == 4


def test_training_failure_is_recorded_without_pairs(fixture: PlanarFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fit that cannot be produced yields a training-failure manifest with every pair unexecuted."""
    f = fixture

    def failing(*args: object, **kwargs: object) -> object:
        del args, kwargs
        msg = "singular ridge problem"
        raise ValueError(msg)

    monkeypatch.setattr(FitStore, "fit_or_load", failing)
    evidence = build_pilot_runner(f, velocity_abort=(41.0, 41.0), simulate_fn=CraftedSimulator(f.samples)).evaluate(
        f.entry, ArmSpec("absolute", "R", 32)
    )
    assert evidence.status == "training_failure"
    assert evidence.fit is None
    assert evidence.training_failure == "ValueError: singular ridge problem"
    assert evidence.first_failure == "training_failure:ValueError: singular ridge problem"
    assert evidence.n_unexecuted == 10
    assert evidence.cells == {}


def test_augmented_arms_record_their_bank_and_prefix(fixture: PlanarFixture) -> None:
    """An A- arm records the accepted attempts and digests; smaller banks are prefixes of larger ones."""
    f = fixture
    small = augmentation_bank_record(
        f.samples, f.inputs.scenario, "central-difference", family="non_decaying", n_synthetic=16
    )
    large = augmentation_bank_record(
        f.samples, f.inputs.scenario, "central-difference", family="non_decaying", n_synthetic=32
    )
    assert small.n_synthetic == 16
    assert small.attempt_budget == 64
    assert len(small.episode_sha256) == 16
    assert check_bank_prefix(small, large)
    assert not check_bank_prefix(large, small)
    assert not check_bank_prefix(replace(small, family="contractive"), large)
    with pytest.raises(ValueError, match="accepted episodes"):
        replace(small, accepted_attempts=(1, 2))
    evidence = build_pilot_runner(f, velocity_abort=(41.0, 41.0), simulate_fn=CraftedSimulator(f.samples)).evaluate(
        f.entry, ArmSpec("absolute", "A-contractive", 16)
    )
    assert evidence.augmentation is not None
    assert evidence.augmentation.family == "contractive"
    assert evidence.augmentation.n_synthetic == 16
    assert evidence.augmentation_prefix_verified is True
    assert evidence.fit is not None
    assert evidence.fit.loss_rows == 17 * (N - 1)


def test_numerical_reference_arms_are_refused(fixture: PlanarFixture) -> None:
    """S-effective is a numerical reference, never evaluated behaviorally."""
    with pytest.raises(ValueError, match="numerical reference"):
        build_pilot_runner(fixture).evaluate(fixture.entry, ArmSpec("absolute", "S-effective", 16))


def test_pointers_are_written_once_and_never_overwritten(
    fixture: PlanarFixture, evaluated: tuple[CraftedSimulator, ModelEvidence], tmp_path: Path
) -> None:
    """Pointers of completed manifests are written idempotently; a differing pointer is refused."""
    f = fixture
    _fake, evidence = evaluated
    runner = build_pilot_runner(f, velocity_abort=(41.0, 41.0), simulate_fn=CraftedSimulator(f.samples))
    assert runner.evaluate(f.entry, ArmSpec("absolute", "S")) == evidence
    written = runner.write_pointers(tmp_path / "evidence")
    assert sorted(p.name for p in written) == sorted(
        [pointer_name("replay", "warmup-0.25s"), pointer_name("model", "feasible-best/absolute/S")]
    )
    assert pointer_name("model", "feasible-best/absolute/R/K17") == "model__feasible-best__absolute__R__K17.toml"
    path = tmp_path / "evidence" / pointer_name("model", "feasible-best/absolute/S")
    pointer = load_pointer(path)
    assert pointer.identity == evidence.evaluation_identity
    assert pointer.status == "feasible"
    assert pointer.n_pairs == 10
    manifest = f.store.path(pointer.payload.uri, mode="read")
    assert sha256_file(manifest) == pointer.payload.sha256
    assert load_model_evidence(manifest) == evidence
    assert runner.write_pointers(tmp_path / "evidence") == []  # identical pointers are idempotent
    path.write_text(path.read_text(encoding="utf-8").replace('status = "feasible"', 'status = "rc_gate_failure"'))
    assert load_pointer(path).status == "rc_gate_failure"
    with pytest.raises(FileExistsError, match="never overwritten"):
        runner.write_pointers(tmp_path / "evidence")


def test_records_rederive_their_invariants(
    fixture: PlanarFixture, evaluated: tuple[CraftedSimulator, ModelEvidence]
) -> None:
    """Counts, statuses, and the caveat cannot be edited independently of the pairs."""
    _fake, evidence = evaluated
    pair = evidence.pairs[0]
    with pytest.raises(ValueError, match="pair status"):
        replace(pair, status="other")
    with pytest.raises(ValueError, match="carries its run"):
        replace(pair, run=None)
    with pytest.raises(ValueError, match="contradicts the component"):
        replace(pair, status="infeasible")
    with pytest.raises(ValueError, match="crossed_historical"):
        replace(pair, crossed_historical=True)
    unexecuted = PairRecord(
        index=0, scenario_id="nominal", kind="nominal", tracker="pd_v2", arm="rc", status="unexecuted"
    )
    with pytest.raises(ValueError, match="carries no component"):
        replace(unexecuted, rc=pair.rc)
    with pytest.raises(ValueError, match="recorded pair counts"):
        replace(evidence, n_completed=9)
    with pytest.raises(ValueError, match="contradict"):
        replace(evidence, status="rc_gate_failure", first_failure="x", cells={})
    with pytest.raises(ValueError, match="C11"):
        replace(evidence, caveats=("other",))
    with pytest.raises(ValueError, match="cells are recorded"):
        replace(evidence, cells={})
    with pytest.raises(ValueError, match="unsupported"):
        replace(evidence, schema_version=2)
    assert set(PAIR_STATUSES) == {"completed", "infeasible", "replay_blocked", "unexecuted"}
    path = fixture.root / "roundtrip.json"
    path.write_text(json.dumps(to_mapping(evidence)), encoding="utf-8")
    assert load_model_evidence(path) == evidence


@pytest.mark.usefixtures("pinned_environment")
def test_run_command_evaluates_selected_arms_and_writes_pointers(
    fixture: PlanarFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command loads the panel, checks the bound development file and trackers, evaluates, and writes pointers."""
    f = fixture
    evaluation, file = write_pilot_evaluation_config(f.root, velocity_abort=(43.0, 43.0))
    manifest = tmp_path / "panel.json"
    manifest.write_text("{}", encoding="utf-8")

    class FakeConfigs:
        development_sha256 = sha256_file(evaluation.development)
        trackers = PLANAR_DIGESTS

    class FakeRule:
        labels = ("feasible-best",)

    class FakeManifest:
        configs = FakeConfigs()
        rule = FakeRule()
        entries = (f.entry,)

        def entry(self, label: str) -> object:
            assert label == "feasible-best"
            return f.entry

    def fake_load(manifest_file: Path, *, store: StorageRoot, root: Path, execution: ExecutionRecord) -> PanelContext:
        del store, root
        return PanelContext(
            manifest=cast("Any", FakeManifest()),
            manifest_sha256=sha256_file(manifest_file),
            inputs=replace(f.inputs, execution_identity=execution.identity),
            dataset=f.record,
            payload=f.payload,
        )

    def fixture_scenarios(*args: object, **kwargs: object) -> tuple[RobustnessScenario, ...]:
        del args, kwargs
        return PLANAR_SCENARIOS

    def fixture_tracker(name: str) -> TrackerConfig:
        del name
        return PLANAR_TRACKER

    monkeypatch.setattr(PanelContext, "load", fake_load)
    monkeypatch.setattr(repetition_evaluation, "repository_root", lambda: f.root)
    monkeypatch.setattr(repetition_evaluation, "robustness_scenarios", fixture_scenarios)
    monkeypatch.setattr(repetition_evaluation, "load_frozen_baseline", fixture_tracker)
    monkeypatch.setattr(repetition_evaluation, "frozen_baseline_digest", PLANAR_DIGESTS.__getitem__)
    monkeypatch.setattr(repetition_evaluation, "simulate", CraftedSimulator(f.samples))
    argv = [
        "run",
        "--manifest",
        str(manifest),
        "--evaluation",
        str(file),
        "--validation",
        str(DOCS / "numerical_validation_v1.json"),
        "--evidence-dir",
        str(tmp_path / "evidence"),
        "--entries",
        "feasible-best",
        "--arms",
        "absolute/S",
        "residual/S",
        "--exploratory",
    ]
    assert main(argv) == 0
    pointers = sorted(p.name for p in (tmp_path / "evidence").glob("*.toml"))
    assert pointers == sorted(
        [
            pointer_name("replay", "warmup-0.25s"),
            pointer_name("model", "feasible-best/absolute/S"),
            pointer_name("model", "feasible-best/residual/S"),
        ]
    )
    with pytest.raises(ValueError, match="unknown behavioral arms"):
        main([*argv[:-1], "--arms", "absolute/S-effective/K17", "--exploratory"])
    # A reached storage allowance checkpoints: the command exits 3 after writing the pointers it completed.
    budgeted = [*argv, "--storage-budget-bytes", "1", "--arms", "absolute/R/K17"]
    assert main(budgeted) == 3
    assert not (tmp_path / "evidence" / pointer_name("model", "feasible-best/absolute/R/K17")).exists()


def test_budget_guard_checkpoints_and_refuses_the_next_run(fixture: PlanarFixture) -> None:
    """A reached allowance stops before the next run; completed runs stay persisted and the resume continues."""
    f = fixture
    fake = CraftedSimulator(f.samples)
    runner = build_pilot_runner(f, velocity_abort=(46.0, 46.0), simulate_fn=fake)
    runner.budget = ExecutionBudget(time_seconds=None, storage_bytes=1, storage_baseline_bytes=0)
    with pytest.raises(BudgetExceededError, match="storage allowance"):
        runner.evaluate(f.entry, ArmSpec("absolute", "S"))
    assert fake.replay_calls == 1  # the first replay run was persisted, the second refused
    assert runner.used_bytes > 0
    resumed = build_pilot_runner(f, velocity_abort=(46.0, 46.0), simulate_fn=CraftedSimulator(f.samples))
    evidence = resumed.evaluate(f.entry, ArmSpec("absolute", "S"))
    assert evidence.status == "feasible"
    with pytest.raises(ValueError, match="non-negative"):
        ExecutionBudget(time_seconds=-1.0, storage_bytes=None)
    with pytest.raises(BudgetExceededError, match="execution allowance"):
        ExecutionBudget(time_seconds=10.0, storage_bytes=None).check(elapsed_seconds=10.0, used_bytes=0)
    ExecutionBudget(time_seconds=10.0, storage_bytes=100, storage_baseline_bytes=50).check(
        elapsed_seconds=1.0, used_bytes=49
    )
    with pytest.raises(BudgetExceededError, match="storage allowance"):
        ExecutionBudget(time_seconds=None, storage_bytes=100, storage_baseline_bytes=50).check(
            elapsed_seconds=1.0, used_bytes=50
        )
