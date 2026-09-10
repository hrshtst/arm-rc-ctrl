# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-008: the pilot reproduction's selection rule, verifiers, recomputation, re-simulation, and audit (C9)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from arm_rc_ctrl.execution import load_execution
from arm_rc_ctrl.experiments.repetition_accounting import account_pilot
from arm_rc_ctrl.experiments.repetition_evaluation import ModelEvidence, PairRecord, PilotRunner
from arm_rc_ctrl.experiments.repetition_fits import FitStore
from arm_rc_ctrl.experiments.repetition_fixture import (
    DOCS,
    PLANAR_DIGESTS,
    PLANAR_SCENARIOS,
    PLANAR_TRACKER,
    CraftedSimulator,
    PlanarFixture,
    build_pilot_runner,
    committed_numerical_binding,
    exploratory_provenance,
    silent,
    write_pilot_evaluation_config,
)
from arm_rc_ctrl.experiments.repetition_numerics import FreshRefit, PanelContext, load_validation
from arm_rc_ctrl.experiments.repetition_panel import load_panel
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec, panel_arms
from arm_rc_ctrl.experiments.repetition_report import ReportInputs
from arm_rc_ctrl.experiments.repetition_timing import load_timing
from arm_rc_ctrl.experiments.reproduce_1a import Check, ReproductionError
from arm_rc_ctrl.experiments.reproduce_repetition import (
    DECLARED_COMPARISONS,
    RESIMULATION_RULE,
    STEPS,
    Doc005Rerun,
    GateResult,
    RepetitionReproduction,
    Reproducer,
    SubsetCoverage,
    _forwarded,  # pyright: ignore[reportPrivateUsage]
    _select_model,  # pyright: ignore[reportPrivateUsage]
    audit_markdown,
    boundary_jump_from_arrays,
    checked_states_from_run,
    main,
    prescribed_fits,
    reproduce,
    resimulation_subset,
    run_doc005,
    run_from_checkout,
    run_gates,
    subset_coverage,
    verify_fit_cache,
    verify_run_payloads,
)
from arm_rc_ctrl.experiments.run_record import RUN_ARRAYS_FILE, RunArrays, load_run, pointer_from_summary
from arm_rc_ctrl.provenance import ArtifactMismatchError, sha256_file
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from arm_rc_ctrl.experiments.repetition_evaluation import SimulateFn
    from arm_rc_ctrl.experiments.run_record import LoadedRun
    from arm_rc_ctrl.experiments.termination import Termination

MANIFEST = DOCS / "panel_manifest_v1.json"
VALIDATION = DOCS / "numerical_validation_v1.json"
CRAFTED_ABORT = (49.0, 49.0)
TIGHT_ABORT = (0.05, 0.05)


@pytest.fixture(scope="module")
def crafted(fixture: PlanarFixture, tmp_path_factory: pytest.TempPathFactory) -> ReportInputs:
    """Crafted sweeps of every behavioral arm of the fixture entry, as the report would read them."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=CRAFTED_ABORT, simulate_fn=CraftedSimulator(f.samples))
    models = {f"{f.entry.label}/{arm.label}": runner.evaluate(f.entry, arm) for arm in panel_arms() if arm.behavioral}
    evidence_dir = tmp_path_factory.mktemp("crafted") / "evidence"
    runner.write_pointers(evidence_dir)
    accounting = account_pilot(
        store=f.store,
        evidence_dir=evidence_dir,
        manifest_file=MANIFEST,
        validation_file=VALIDATION,
        provenance=exploratory_provenance(),
    )
    return ReportInputs(
        docs=DOCS,
        store=f.store,
        root=f.root,
        manifest=load_panel(MANIFEST),
        accounting=accounting,
        validation=load_validation(VALIDATION),
        timing=load_timing(DOCS / "timing_smoke_check_v1.json"),
        execution=load_execution(DOCS / "execution_environment_v1.json"),
        models=models,
        banks={f.entry.warmup_s: runner.replay_bank(f.entry)},
        fit_seconds={},
        sources={},
    )


@pytest.fixture(scope="module")
def simulated(fixture: PlanarFixture, crafted: ReportInputs) -> ReportInputs:
    """One real sweep under a tight abort: every replay and the first RC pair abort with terminal states."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=TIGHT_ABORT)
    bank = runner.replay_bank(f.entry)
    evidence = runner.evaluate(f.entry, ArmSpec("absolute", "S"))
    assert evidence.status == "rc_gate_failure"
    return replace(crafted, models={f"{f.entry.label}/absolute/S": evidence}, banks={f.entry.warmup_s: bank})


def _reproducer(f: PlanarFixture, inputs: ReportInputs, scratch: Path) -> Reproducer:
    reproducer = Reproducer(
        scratch,
        False,  # noqa: FBT003 - positional dataclass field
        f.store,
        DOCS,
        None,
        root=f.root,
        scenarios=PLANAR_SCENARIOS,
        trackers={"pd_v2": PLANAR_TRACKER, "computed_torque": PLANAR_TRACKER},
        tracker_digests=dict(PLANAR_DIGESTS),
        entries={f.entry.label: f.entry},
    )
    reproducer.store = f.store
    (scratch / "store").mkdir(parents=True)
    reproducer.scratch_store = StorageRoot(scratch / "store", repositories=(f.root,))
    reproducer.current_execution = f.execution
    reproducer.context = PanelContext(
        manifest=load_panel(MANIFEST),
        manifest_sha256=sha256_file(MANIFEST),
        inputs=f.inputs,
        dataset=f.record,
        payload=f.payload,
    )
    reproducer.report_inputs = inputs
    return reproducer


def _scratch_runner(
    f: PlanarFixture, store_dir: Path, *, velocity_abort: tuple[float, ...], simulate_fn: SimulateFn | None = None
) -> PilotRunner:
    evaluation, file = write_pilot_evaluation_config(f.root, velocity_abort=velocity_abort)
    store_dir.mkdir(parents=True)
    return PilotRunner(
        store=StorageRoot(store_dir, repositories=(f.root,)),
        inputs=f.inputs,
        dataset=f.record,
        evaluation=evaluation,
        evaluation_file=file,
        root=f.root,
        execution=f.execution,
        provenance=exploratory_provenance(),
        numerical=committed_numerical_binding(),
        scenarios=PLANAR_SCENARIOS,
        trackers={"pd_v2": PLANAR_TRACKER, "computed_torque": PLANAR_TRACKER},
        tracker_digests=PLANAR_DIGESTS,
        development_sha256=sha256_file(evaluation.development),
        simulate_fn=simulate_fn,
        log=silent,
    )


# --- records and orchestration -------------------------------------------------------------


def test_reproduction_ok_requires_every_step_gate_and_rerun() -> None:
    """A truncated run, a failed step, a failed gate, or a crashed DOC-005 rerun is never ok."""
    passing = tuple(Check(name, ok=True, detail="", elapsed_s=0.0) for name in STEPS)

    def result(
        checks: tuple[Check, ...], gates: tuple[GateResult, ...] = (), reruns: tuple[Doc005Rerun, ...] = ()
    ) -> RepetitionReproduction:
        return RepetitionReproduction(
            started_at="t",
            checks=checks,
            inputs={},
            environment={},
            max_deviation=0.0,
            elapsed_s=1.0,
            comparisons=DECLARED_COMPARISONS,
            resimulation_rule=RESIMULATION_RULE,
            coverage=None,
            doc005=reruns,
            gates=gates,
        )

    rerun = Doc005Rerun(
        label="x",
        cpus=(0,),
        command="cmd",
        returncode=0,
        ok=True,
        max_deviation=0.0,
        tolerance=0.0,
        elapsed_s=1.0,
        detail="ok",
    )
    assert result(passing).ok
    assert result(passing, gates=(GateResult("nox", 0, 1.0),), reruns=(rerun,)).ok
    assert not result(passing[:-1]).ok
    assert not result((*passing[:-1], Check(STEPS[-1], ok=False, detail="boom", elapsed_s=0.0))).ok
    assert not result(passing, gates=(GateResult("nox", 1, 1.0),)).ok
    assert result(passing, reruns=(replace(rerun, ok=False, returncode=1, max_deviation=1.6e-10),)).ok
    assert not result(passing, reruns=(replace(rerun, ok=None, returncode=2),)).ok


def test_step_captures_failures_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising step becomes a named, failed check instead of an exception."""
    reproducer = Reproducer(tmp_path, False, None, tmp_path, None)  # noqa: FBT003 - positional dataclass field

    def boom() -> str:
        msg = "missing input"
        raise ReproductionError(msg)

    monkeypatch.setattr(reproducer, "records", boom)
    check = reproducer.step("records")
    assert not check.ok
    assert check.name == "records"
    assert "missing input" in check.detail
    with pytest.raises(ReproductionError, match="requires an earlier step"):
        reproducer.payloads()


def test_reproduce_refuses_a_used_scratch_and_stops_at_the_first_failure(tmp_path: Path) -> None:
    """The scratch must be fresh, and without --keep-going a failed environment step ends the run."""
    used = tmp_path / "used"
    used.mkdir()
    (used / "x").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        reproduce(scratch=used, docs=tmp_path)
    result = reproduce(scratch=tmp_path / "fresh", docs=tmp_path / "no-docs")
    assert not result.ok
    assert [c.name for c in result.checks] == ["environment"]
    assert not result.checks[0].ok
    assert result.coverage is None
    assert result.doc005 == ()
    assert result.gates == ()
    assert result.comparisons == DECLARED_COMPARISONS


# --- the re-simulation rule ---------------------------------------------------------------


def test_resimulation_rule_selects_by_position_and_reports_coverage(crafted: ReportInputs) -> None:
    """Every configuration's first pair, the first pair per class and tracker of complete sweeps, and each bank's."""
    selections = resimulation_subset(crafted)
    rc = [s for s in selections if s.kind == "rc"]
    replay = [s for s in selections if s.kind == "replay"]
    assert len(rc) == 20 * 8  # five scenarios over four classes, two trackers, the nominal first pair shared
    assert len(replay) == 8
    assert {s.reason for s in rc} == {"first pair", "first pair of class and tracker"}
    assert sum(1 for s in rc if s.reason == "first pair") == 20
    assert all(s.arm is None and s.entry == "feasible-best" for s in replay)
    assert len({(s.kind, s.entry, s.arm, s.scenario_id, s.tracker) for s in selections}) == len(selections)
    coverage = subset_coverage(selections, crafted)
    assert coverage == SubsetCoverage(
        runs=168,
        rc_runs=160,
        replay_runs=8,
        formulations=("absolute", "residual"),
        arms=tuple(sorted(a.label for a in panel_arms() if a.behavioral)),
        warmups_s=(0.0, 0.25, 1.0),
        trackers=("computed_torque", "pd_v2"),
        classes=("force", "nominal", "posture_large", "posture_small"),
        first_pair_failures=0,
        later_failures=0,
        unavailable=(
            "replay-blocked pairs: none exist in the evidence",
            "training failures: none exist in the evidence",
        ),
    )


def test_rule_selects_a_later_first_failure_and_skips_configurations_without_runs() -> None:
    """Clause 2 selects the first infeasible pair when it is not the first pair; no run means no selection."""

    def pair(scenario_id: str, tracker: str, status: str, kind: str = "nominal") -> SimpleNamespace:
        return SimpleNamespace(
            scenario_id=scenario_id,
            tracker=tracker,
            status=status,
            kind=kind,
            run=None if status == "unexecuted" else object(),
        )

    later = SimpleNamespace(
        status="rc_gate_failure",
        n_completed=2,
        n_pairs=4,
        pairs=(
            pair("s0", "pd_v2", "completed"),
            pair("s0", "ct", "completed"),
            pair("s1", "pd_v2", "infeasible"),
            pair("s1", "ct", "unexecuted"),
        ),
    )
    selected: list[Any] = []
    _select_model("entry/absolute/R/K17", cast("ModelEvidence", later), selected.append)
    assert [(s.scenario_id, s.tracker, s.reason) for s in selected] == [
        ("s0", "pd_v2", "first pair"),
        ("s1", "pd_v2", "first infeasible pair"),
    ]
    none = SimpleNamespace(status="training_failure", n_completed=0, n_pairs=0, pairs=())
    selected.clear()
    _select_model("entry/absolute/S", cast("ModelEvidence", none), selected.append)
    assert selected == []


# --- payload and cache verification -------------------------------------------------------


def test_verify_run_payloads_detects_corruption_and_missing_files(
    fixture: PlanarFixture, crafted: ReportInputs
) -> None:
    """Every run payload is digest-checked; a tampered or missing array file fails clearly and nothing is deleted."""
    f = fixture
    runs, arrays_bytes = verify_run_payloads(f.store, crafted)
    assert runs == 20 * 10 + 10
    assert arrays_bytes > 0
    pair = next(iter(crafted.models.values())).pairs[0]
    assert pair.run is not None
    summary_file = f.store.path(pair.run.uri, mode="read")
    arrays_file = summary_file.parent / RUN_ARRAYS_FILE
    original = arrays_file.read_bytes()
    mode = arrays_file.stat().st_mode
    try:
        arrays_file.chmod(0o644)
        arrays_file.write_bytes(original + b"\0")
        with pytest.raises(ReproductionError, match="arrays digest"):
            verify_run_payloads(f.store, crafted)
        arrays_file.rename(arrays_file.with_suffix(".moved"))
        with pytest.raises(ReproductionError, match="has no arrays file"):
            verify_run_payloads(f.store, crafted)
        arrays_file.with_suffix(".moved").rename(arrays_file)
        arrays_file.write_bytes(original)
        summary_text = summary_file.read_bytes()
        summary_mode = summary_file.stat().st_mode
        summary_file.chmod(0o644)
        summary_file.write_bytes(summary_text + b"\n")
        with pytest.raises(ArtifactMismatchError):
            verify_run_payloads(f.store, crafted)
        summary_file.write_bytes(summary_text)
        summary_file.chmod(summary_mode)
    finally:
        arrays_file.write_bytes(original)
        arrays_file.chmod(mode)
    assert verify_run_payloads(f.store, crafted) == (runs, arrays_bytes)


def test_prescribed_fits_and_cache_verification(fixture: PlanarFixture, crafted: ReportInputs) -> None:
    """The prescription is the validation's fits plus the evidence-bound ones; every entry must be cached."""
    f = fixture
    validation = load_validation(VALIDATION)
    prescribed = prescribed_fits(validation, crafted)
    assert len(prescribed) == len(validation.fits) + 20
    assert sum(1 for p in prescribed.values() if p.source == "evidence") == 20
    with pytest.raises(ReproductionError, match="is not cached"):
        verify_fit_cache(FitStore(f.store), prescribed, crafted)
    bound = {k: v for k, v in prescribed.items() if v.source == "evidence"}
    assert verify_fit_cache(FitStore(f.store), bound, crafted) == 20
    label, evidence = next(iter(crafted.models.items()))
    assert evidence.fit is not None
    tampered = replace(crafted, models={label: replace(evidence, fit=replace(evidence.fit, weights_sha256="0" * 64))})
    bound_tampered = {k: v for k, v in prescribed_fits(validation, tampered).items() if v.source == "evidence"}
    with pytest.raises(ReproductionError, match="recorded digest"):
        verify_fit_cache(FitStore(f.store), bound_tampered, tampered)


# --- recomputation from stored runs --------------------------------------------------------


def test_checked_states_rebuild_from_rows_plus_the_recorded_terminal_state(
    fixture: PlanarFixture, simulated: ReportInputs, crafted: ReportInputs
) -> None:
    """Rows are checked states; a non-completed run appends the recorded terminal state, a completed one none."""
    f = fixture
    aborted = next(iter(simulated.models.values())).pairs[0]
    assert aborted.run is not None
    assert aborted.velocity is not None
    assert aborted.velocity.terminal_state is not None
    loaded = load_run(f.store, pointer_from_summary(f.store, aborted.run.artifact_id))
    states = checked_states_from_run(loaded, aborted)
    assert len(states) == loaded.arrays.n_samples + 1
    assert states[-1].step == aborted.velocity.terminal_state.step
    assert tuple(states[-1].q) == aborted.velocity.terminal_state.q
    assert [s.step for s in states[:-1]] == list(range(loaded.arrays.n_samples))
    undiagnosed = SimpleNamespace(scenario_id=aborted.scenario_id, tracker=aborted.tracker, velocity=None)
    with pytest.raises(ReproductionError, match="terminal state"):
        checked_states_from_run(loaded, cast("PairRecord", undiagnosed))
    completed = next(iter(crafted.models.values())).pairs[0]
    assert completed.run is not None
    loaded_completed = load_run(f.store, pointer_from_summary(f.store, completed.run.artifact_id))
    assert len(checked_states_from_run(loaded_completed, completed)) == loaded_completed.arrays.n_samples


def test_boundary_jump_from_arrays_is_the_first_generated_target_against_the_hold() -> None:
    """The formula the controller records: max over joints of the first generated target minus the held posture."""
    rows = 4
    q = np.tile(np.array([0.1, -0.2]), (rows, 1))
    base: dict[str, Any] = {
        "t": np.arange(rows, dtype=np.float64) * 0.01,
        "q": q,
        "dq": np.zeros((rows, 2)),
        "tip": np.zeros((rows, 2)),
        "q_desired": np.array([[0.1, -0.2], [0.1, -0.2], [0.4, -0.25], [0.5, -0.3]]),
        "dq_desired_raw": np.zeros((rows, 2)),
        "dq_desired": np.zeros((rows, 2)),
        "ddq_desired_raw": np.zeros((rows, 2)),
        "ddq_desired": np.zeros((rows, 2)),
        "tracking_error": np.zeros((rows, 2)),
        "tau_requested": np.zeros((rows, 2)),
        "task_code": np.zeros((rows, 0)),
        "saturation": np.zeros(rows, dtype=np.int64),
    }
    generated = SimpleNamespace(arrays=RunArrays({**base, "phase": np.array([0, 0, 1, 1], dtype=np.int64)}))
    assert boundary_jump_from_arrays(cast("LoadedRun", generated)) == pytest.approx(0.3)
    held = SimpleNamespace(arrays=RunArrays({**base, "phase": np.zeros(rows, dtype=np.int64)}))
    assert boundary_jump_from_arrays(cast("LoadedRun", held)) is None
    assert boundary_jump_from_arrays(cast("LoadedRun", SimpleNamespace(arrays=RunArrays(base)))) is None


def test_boundary_jump_from_arrays_matches_a_real_controller(fixture: PlanarFixture) -> None:
    """On a real run that reaches activation, the telemetry gives exactly the controller's recorded jump."""
    f = fixture
    evidence = build_pilot_runner(f, velocity_abort=(48.0, 48.0)).evaluate(f.entry, ArmSpec("absolute", "S"))
    first = evidence.pairs[0]
    assert first.run is not None
    assert first.rc is not None
    assert first.rc.boundary_jump is not None
    loaded = load_run(f.store, pointer_from_summary(f.store, first.run.artifact_id))
    assert boundary_jump_from_arrays(loaded) == first.rc.boundary_jump


def _fake_jump(_run: LoadedRun) -> float:
    return 0.123


def test_metrics_recompute_exactly_from_stored_runs(
    fixture: PlanarFixture, simulated: ReportInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Components, diagnostics, statuses, and cells recompute from the arrays; a changed input is detected."""
    f = fixture
    reproducer = _reproducer(f, simulated, tmp_path / "a")
    detail = reproducer.metrics()
    assert detail.startswith("10 replay runs and the runs of 1 executed model pairs")
    assert reproducer.max_deviation == 0.0
    tampered = _reproducer(f, simulated, tmp_path / "b")
    monkeypatch.setattr("arm_rc_ctrl.experiments.reproduce_repetition.boundary_jump_from_arrays", _fake_jump)
    with pytest.raises(ReproductionError, match="component"):
        tampered.metrics()


def test_metrics_refuse_a_bank_filed_under_the_wrong_warmup(
    fixture: PlanarFixture, simulated: ReportInputs, tmp_path: Path
) -> None:
    """A replay bank keyed by a warm-up it does not record is a categorical error."""
    f = fixture
    wrong = replace(simulated, banks={f.entry.warmup_s + 1.0: next(iter(simulated.banks.values()))})
    with pytest.raises(ReproductionError, match="records warm-up"):
        _reproducer(f, wrong, tmp_path).metrics()


# --- re-simulation ------------------------------------------------------------------------


def test_resimulate_with_reproduces_crafted_runs_and_detects_drift(
    fixture: PlanarFixture, crafted: ReportInputs, tmp_path: Path
) -> None:
    """The subset re-simulates bitwise into a scratch store; perturbed arrays or other conditions fail clearly."""
    f = fixture
    reproducer = _reproducer(f, crafted, tmp_path / "ok")
    runner = _scratch_runner(
        f, tmp_path / "ok-store", velocity_abort=CRAFTED_ABORT, simulate_fn=CraftedSimulator(f.samples)
    )
    detail = reproducer.resimulate_with(runner)
    assert detail.startswith("168 runs re-simulated bitwise")
    assert reproducer.coverage is not None
    assert reproducer.coverage.rc_runs == 160
    assert any((tmp_path / "ok-store").rglob("run.json"))

    fake = CraftedSimulator(f.samples)

    def perturbed(scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        arrays, termination = fake(scenario, controller, **kwargs)
        data = dict(arrays.arrays)
        data["q"] = data["q"] + 1e-9
        return RunArrays(data), termination

    drifted = _reproducer(f, crafted, tmp_path / "drift")
    with pytest.raises(ReproductionError, match="arrays differ"):
        drifted.resimulate_with(
            _scratch_runner(f, tmp_path / "drift-store", velocity_abort=CRAFTED_ABORT, simulate_fn=perturbed)
        )
    other = _reproducer(f, crafted, tmp_path / "other")
    with pytest.raises(ReproductionError, match="not the committed bank"):
        other.resimulate_with(
            _scratch_runner(f, tmp_path / "other-store", velocity_abort=(40.0, 40.0), simulate_fn=fake)
        )


def test_resimulate_with_reproduces_real_aborted_runs(
    fixture: PlanarFixture, simulated: ReportInputs, tmp_path: Path
) -> None:
    """Real aborted replay and RC runs re-simulate bitwise, terminal states included."""
    f = fixture
    reproducer = _reproducer(f, simulated, tmp_path / "real")
    detail = reproducer.resimulate_with(_scratch_runner(f, tmp_path / "real-store", velocity_abort=TIGHT_ABORT))
    assert detail.startswith("9 runs re-simulated bitwise")
    assert reproducer.coverage is not None
    assert reproducer.coverage.first_pair_failures == 1
    assert reproducer.coverage.unavailable[0].startswith("replay-blocked")


# --- the audit, the DOC-005 rerun, and the gates ------------------------------------------


def test_audit_markdown_renders_every_section() -> None:
    """The audit lists the steps, the declared comparisons, the subset coverage, DOC-005, gates, and inputs."""
    coverage = SubsetCoverage(3, 2, 1, ("absolute",), ("absolute/S",), (0.25,), ("pd_v2",), ("nominal",), 1, 0, ("x",))
    rerun = Doc005Rerun(
        label="canonical",
        cpus=(0, 1),
        command="python -m arm_rc_ctrl.experiments.reproduce_1a --from-evidence",
        returncode=0,
        ok=True,
        max_deviation=0.0,
        tolerance=0.0,
        elapsed_s=12.0,
        detail="ok True",
    )
    result = RepetitionReproduction(
        started_at="2026-09-10T00:00:00+00:00",
        checks=(Check("environment", ok=True, detail="ok", elapsed_s=0.1),),
        inputs={"dataset": "processed-test"},
        environment={"python": "3.12", "affinity": "0,1"},
        max_deviation=0.0,
        elapsed_s=1.0,
        comparisons=DECLARED_COMPARISONS,
        resimulation_rule=RESIMULATION_RULE,
        coverage=coverage,
        doc005=(rerun,),
        gates=(GateResult("uv run --locked nox", 0, 700.0),),
    )
    text = audit_markdown(result, command="python scripts/reproduce_repetition.py", auditor="someone")
    assert text.startswith("# Task 1-a repeated-demonstration pilot reproduction")
    assert "| environment | True | 0.1 | ok |" in text
    assert "## Declared comparisons" in text
    assert "| historical task 1-a reproduction (DOC-005) | tolerance |" in text
    assert "Selected runs: 3 (2 RC, 1 replay)" in text
    assert "Unavailable coverage: x." in text
    assert "| canonical | 0,1 | 0 | True | 0.0 | 0.0 | 12.0 | ok True |" in text
    assert "| `uv run --locked nox` | 0 | 700.0 |" in text
    assert "- dataset: `processed-test`" in text
    assert "Auditor: someone." in text
    bare = replace(result, coverage=None, doc005=(), gates=())
    text = audit_markdown(bare, command="c")
    assert "did not run" in text
    assert "pass --doc005" in text
    assert "pass --gates" in text


def test_run_doc005_reports_the_summary_or_its_absence(tmp_path: Path) -> None:
    """The rerun's summary is parsed when written; a crash without a summary is reported with its exit status."""

    def writing(command: list[str], **_kwargs: object) -> SimpleNamespace:
        summary = command[command.index("--summary") + 1]
        with open(summary, "w", encoding="utf-8") as handle:
            json.dump({"ok": True, "max_deviation": 0.0, "checks": [{"name": "environment", "ok": True}]}, handle)
        return SimpleNamespace(returncode=0, stderr="")

    rerun = run_doc005(tmp_path / "a", label="canonical", runner=writing)
    assert rerun.ok is True
    assert rerun.max_deviation == 0.0
    assert rerun.returncode == 0
    assert rerun.tolerance == 0.0
    assert "--from-evidence" in rerun.command
    assert "reproduce_1a.json" in rerun.command
    assert "/" not in rerun.command.split("--scratch")[1].split()[0]
    assert rerun.cpus

    def crashing(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=3, stderr="boom")

    crashed = run_doc005(tmp_path / "b", label="cpus", cpus=(16, 17), runner=crashing)
    assert crashed.ok is None
    assert crashed.returncode == 3
    assert crashed.cpus == (16, 17)
    assert crashed.command.startswith("taskset -c 16,17 python -m arm_rc_ctrl.experiments.reproduce_1a")
    assert "no summary written" in crashed.detail
    with pytest.raises(FileExistsError):
        run_doc005(tmp_path / "b", label="again", runner=crashing)


def test_run_gates_records_exit_statuses() -> None:
    """Each gate command is run in order and its exit status recorded."""
    calls: list[list[str]] = []

    def fake(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=len(calls) - 1)

    gates = run_gates(runner=fake, commands=(("nox",), ("nox", "-s", "pre_commit")))
    assert [g.command for g in gates] == ["nox", "nox -s pre_commit"]
    assert [g.returncode for g in gates] == [0, 1]
    assert calls == [["nox"], ["nox", "-s", "pre_commit"]]


def test_forwarded_arguments_keep_flags_and_absolute_output_paths(tmp_path: Path) -> None:
    """--from-checkout forwards the flags and makes the summary and audit paths absolute."""
    args = SimpleNamespace(
        keep_going=True,
        exploratory=False,
        skip_resimulation=True,
        skip_fits=False,
        doc005=True,
        gates=False,
        doc005_cpus="16-31",
        summary=tmp_path / "s.json",
        audit=None,
    )
    forwarded = _forwarded(cast("Any", args))
    assert forwarded == [
        "--keep-going",
        "--skip-resimulation",
        "--doc005",
        "--doc005-cpus",
        "16-31",
        "--summary",
        str((tmp_path / "s.json").resolve()),
    ]


def test_fits_step_refits_every_prescribed_fit_through_the_injected_refit(
    fixture: PlanarFixture, crafted: ReportInputs, tmp_path: Path
) -> None:
    """The fits step refits each prescription in a fresh process and fails on the first non-bitwise refit."""
    f = fixture
    validation = load_validation(VALIDATION)
    prescribed = {k: v for k, v in prescribed_fits(validation, crafted).items() if v.source == "evidence"}
    calls: list[str] = []

    def passing(identity: str, *, root: Path, parent_identity: str, output: Path) -> FreshRefit:
        calls.append(identity)
        assert root == f.root
        assert parent_identity == f.execution.identity
        assert output.parent.is_dir()
        return FreshRefit(
            identity=identity,
            worker_execution_identity=parent_identity,
            environment_match=True,
            weights_bitwise_equal=True,
            max_abs_weight_diff=0.0,
            states_bitwise_equal=True,
            fit_report_equal=True,
            rmse_abs_diff=0.0,
            seconds=0.1,
            passed=True,
        )

    reproducer = _reproducer(f, crafted, tmp_path / "ok")
    reproducer.prescribed = prescribed
    reproducer.refit = passing
    assert reproducer.fits().startswith("20 fits refitted bitwise")
    assert sorted(calls) == sorted(prescribed)
    assert reproducer.max_deviation == 0.0

    def failing(identity: str, *, root: Path, parent_identity: str, output: Path) -> FreshRefit:
        del root, output
        return FreshRefit(
            identity=identity,
            worker_execution_identity=parent_identity,
            environment_match=True,
            weights_bitwise_equal=False,
            max_abs_weight_diff=1e-9,
            states_bitwise_equal=True,
            fit_report_equal=True,
            rmse_abs_diff=0.0,
            seconds=0.1,
            passed=False,
        )

    broken = _reproducer(f, crafted, tmp_path / "broken")
    broken.prescribed = prescribed
    broken.refit = failing
    with pytest.raises(ReproductionError, match="did not refit bitwise"):
        broken.fits()
    skipped = _reproducer(f, crafted, tmp_path / "skipped")
    skipped.skip_fits = True
    assert skipped.fits() == "skipped on request (--skip-fits)"


def test_resimulation_step_honours_skip_and_refuses_exploratory(
    fixture: PlanarFixture, crafted: ReportInputs, tmp_path: Path
) -> None:
    """The step is skippable on request and refuses to rerun from a dirty checkout."""
    f = fixture
    skipped = _reproducer(f, crafted, tmp_path / "skipped")
    skipped.skip_resimulation = True
    assert skipped.resimulation() == "skipped on request (--skip-resimulation)"
    dirty = _reproducer(f, crafted, tmp_path / "dirty")
    dirty.exploratory = True
    with pytest.raises(ReproductionError, match="clean checkout"):
        dirty.resimulation()


def test_checkout_reproduction_refuses_an_unknown_commit(tmp_path: Path) -> None:
    """A commit git cannot resolve fails before any worktree is created; the scratch stays fresh."""
    with pytest.raises(ReproductionError, match="rev-parse"):
        run_from_checkout(tmp_path / "scratch", "no-such-commit-0123", [])
    assert not (tmp_path / "scratch").exists()
    with pytest.raises(ReproductionError, match="rev-parse"):
        main(["--from-checkout", "no-such-commit-0123", "--scratch", str(tmp_path / "main-scratch")])


def test_checkout_reproduction_resolves_a_relative_scratch_for_the_inner_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative scratch is resolved once; the checkout, the sync, and the inner command all use that directory."""
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[list[str], object]] = []

    def fake_git(*args: str) -> str:
        if args[0] == "rev-parse":
            return "0123456789abcdef0123456789abcdef01234567\n"
        if args[0] == "worktree":
            Path(args[3]).mkdir(parents=True)
        return ""

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((list(command), kwargs.get("cwd")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("arm_rc_ctrl.experiments.reproduce_repetition._git", fake_git)
    monkeypatch.setattr("arm_rc_ctrl.experiments.reproduce_repetition.subprocess.run", fake_run)
    assert run_from_checkout(Path("relative-scratch"), "HEAD", ["--skip-fits"]) == 0
    resolved = (tmp_path / "relative-scratch").resolve()
    assert (resolved / "checkout").is_dir()
    assert [cwd for _, cwd in calls] == [resolved / "checkout"] * 3
    inner = calls[-1][0]
    assert inner[inner.index("--scratch") + 1] == str(resolved / "inner")
    assert inner[-1] == "--skip-fits"
