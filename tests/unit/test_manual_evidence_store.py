# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-008: the sweep's evidence is written once, served again, and rebuilt strictly.

A completed bank or model manifest is immutable evidence: a second sweep under
the same protocol must serve what is stored rather than simulate again, and a
manifest must rebuild from its own JSON without losing a field. Both matter
before the full execution, where re-simulating would be both expensive and a
different experiment.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.experiments.manual_evaluation import (
    PROGRESS_FILE,
    ManualEvaluationConfig,
    ManualEvaluationRunner,
    ManualModelEvidence,
    ManualReplayBank,
    load_manual_evaluation_config,
    load_manual_model_evidence,
    load_manual_replay_bank,
    manual_bank_to_json,
    manual_evidence_to_json,
    model_uri,
    replay_bank_uri,
)
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.run_record import RUN_ARRAYS_FILE, RunArrays
from arm_rc_ctrl.experiments.termination import completed
from arm_rc_ctrl.provenance import sha256_bytes, sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import joint_target

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.manual_scenario import ManualScenarioConfig
    from arm_rc_ctrl.experiments.manual_evaluation import SimulateFn
    from arm_rc_ctrl.experiments.manual_fixture import ManualFixture
    from arm_rc_ctrl.experiments.manual_study import StudyModel
    from arm_rc_ctrl.experiments.termination import Termination

REPO_ROOT = repository_root()
DEVELOPMENT_SOURCE = REPO_ROOT / "configs" / "evaluations" / "task_1a_recovery_dev_v1.toml"
TRACKER = TrackerConfig(type="pd", kp=(10.0, 5.0), kd=(1.5, 0.8))
REPLAY_CUTOFFS = (20.0, 20.0)
"""The causal derivative policy replay is driven through; a bank belongs to one policy."""
HOLD_S, PULSE_S, HORIZON_S, WARMUP_S = 0.05, 0.02, 1.0, 0.25
CONFIGURATION, ARM_LABEL = "feasible-best", "S/D01"
ARM_CORRUPTED = "S/D02"
ARM_ALL_TEN = "M10"
WARMUP_TRUSTED = 0.3
"""A warm-up of its own for the combined alteration, so it starts from an evidence directory of its own."""
ARM_RENAMED = "S/D04"
"""A model of its own for the corruption case, so it never serves another test's stored evidence."""

SCENARIOS = (
    RobustnessScenario("nominal", "nominal", (0.0, 0.0)),
    RobustnessScenario("small-1", "posture_small", (0.02, -0.01), seed=1, draw=0, magnitude_rad=0.05),
)


def _evaluation(f: ManualFixture) -> tuple[ManualEvaluationConfig, Path]:
    """The fixture's own evaluation configuration (written identically on every call)."""
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
    return load_manual_evaluation_config(target), target


class _CountingSimulator:
    """Crafts feasible runs and counts how many times it was asked to simulate."""

    def __init__(self, scenario_config: ManualScenarioConfig, *, reaches: bool = True) -> None:
        self.scenario = scenario_config
        self.reaches = reaches
        self.calls = 0

    def __call__(self, scenario: object, controller: object, **kwargs: object) -> tuple[RunArrays, Termination]:
        del scenario, controller
        self.calls += 1
        rows = round(cast("float", kwargs["duration_s"]) / self.scenario.timing.dt) + 1
        start = tuple(float(v) for v in cast("tuple[float, ...]", kwargs["initial_q"]))
        arrays = _crafted(self.scenario, rows, start, rc=kwargs.get("channels") is not None, reaches=self.reaches)
        return arrays, completed(float(arrays.arrays["t"][-1]), rows - 1)


def _crafted(
    scenario_config: ManualScenarioConfig, rows: int, start: tuple[float, ...], *, rc: bool, reaches: bool = True
) -> RunArrays:
    """A run that reaches the target early and holds it, or never leaves the start and so fails its dwell."""
    dt = scenario_config.timing.dt
    on_target = np.asarray(joint_target(scenario_config), dtype=np.float64)
    t: NDArray[np.float64] = np.arange(rows, dtype=np.float64) * dt
    q = np.tile(np.asarray(start, dtype=np.float64), (rows, 1))
    if reaches:
        q[max(1, rows // 4) :] = on_target
    dq = np.zeros((rows, 2), dtype=np.float64)
    zeros = np.zeros((rows, 2), dtype=np.float64)
    data: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "t": t,
        "q": q,
        "dq": dq,
        "tip": manual_endpoint_positions(scenario_config, q),
        "q_desired": q.copy(),
        "dq_desired": dq.copy(),
        "dq_desired_raw": dq.copy(),
        "ddq_desired": zeros.copy(),
        "ddq_desired_raw": zeros.copy(),
        "tracking_error": zeros.copy(),
        "tau_requested": zeros.copy(),
        "task_code": np.zeros((rows, 0), dtype=np.float64),
        "saturation": np.zeros(rows, dtype=np.int64),
    }
    if rc:
        active = t >= WARMUP_S - 1e-9
        readout = np.full((rows, 2), np.nan, dtype=np.float64)
        readout[active] = q[active]
        data["generator_output_q"] = readout
        data["phase"] = active.astype(np.int64)
    return RunArrays(data)


def _entry(f: ManualFixture, arm_label: str = ARM_LABEL) -> StudyModel:
    return next(e for e in f.manifest.entries if (e.configuration, e.arm.label) == (CONFIGURATION, arm_label))


def _runner(f: ManualFixture, simulate_fn: SimulateFn) -> ManualEvaluationRunner:
    config, file = _evaluation(f)
    return ManualEvaluationRunner(
        store=f.store,
        inputs=f.inputs,
        config=config,
        evaluation_file=file,
        scenarios=SCENARIOS,
        trackers={"pd_v2": TRACKER, "computed_torque": TRACKER},
        root=f.root,
        execution=f.execution,
        provenance=f.provenance,
        simulate_fn=simulate_fn,
    )


def _rebuilt_bank(tmp_path: Path, bank: ManualReplayBank) -> ManualReplayBank:
    """Write the bank's manifest and read it back through the loader the sweep itself uses."""
    path = tmp_path / "bank.json"
    path.write_text(manual_bank_to_json(bank), encoding="utf-8")
    return load_manual_replay_bank(path)


def _rebuilt_evidence(tmp_path: Path, evidence: ManualModelEvidence) -> ManualModelEvidence:
    """Write the model manifest and read it back through the shipped loader."""
    path = tmp_path / "model.json"
    path.write_text(manual_evidence_to_json(evidence), encoding="utf-8")
    return load_manual_model_evidence(path)


# --- written once, served again ---------------------------------------------------------------


def test_a_stored_bank_is_served_without_simulating_again(manual_fixture: ManualFixture) -> None:
    """A completed bank is immutable evidence: a fresh runner over the same store reuses it."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    first_sim = _CountingSimulator(scenario_config)
    first = _runner(manual_fixture, first_sim).replay_bank("D03", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS)
    assert first_sim.calls == len(SCENARIOS) * 2
    second_sim = _CountingSimulator(scenario_config)
    second = _runner(manual_fixture, second_sim).replay_bank("D03", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS)
    assert second_sim.calls == 0
    assert second.identity == first.identity
    assert [(p.scenario_id, p.tracker) for p in second.pairs] == [(p.scenario_id, p.tracker) for p in first.pairs]


def test_stored_model_evidence_is_served_without_simulating_again(manual_fixture: ManualFixture) -> None:
    """The same model under the same protocol is one experiment, run once."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    first_sim = _CountingSimulator(scenario_config)
    first = _runner(manual_fixture, first_sim).evaluate(entry, warmup_s=WARMUP_S)
    ran = first_sim.calls
    assert ran > 0
    second_sim = _CountingSimulator(scenario_config)
    second = _runner(manual_fixture, second_sim).evaluate(entry, warmup_s=WARMUP_S)
    assert second_sim.calls == 0
    assert second.identity == first.identity
    assert second.n_completed == first.n_completed


# --- rebuilt strictly from their own JSON ------------------------------------------------------


def test_a_bank_rebuilds_from_its_manifest(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """Every field survives the round trip, including the ones that are unset on a replay run."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    bank = _runner(manual_fixture, _CountingSimulator(scenario_config)).replay_bank(
        "D04", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS
    )
    rebuilt = _rebuilt_bank(tmp_path, bank)
    assert to_mapping(rebuilt) == to_mapping(bank)
    assert rebuilt.identity == bank.identity


def test_model_evidence_rebuilds_from_its_manifest(manual_fixture: ManualFixture, tmp_path: Path) -> None:
    """The nested verdicts survive too: the dwell reports, the generated reference, and the trigger."""
    scenario_config = load_manual_scenario(manual_fixture.scenario_file)
    entry = _entry(manual_fixture)
    evidence = _runner(manual_fixture, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_S)
    rebuilt = _rebuilt_evidence(tmp_path, evidence)
    assert to_mapping(rebuilt) == to_mapping(evidence)
    assert rebuilt.identity == evidence.identity


# --- served only while the runs behind it are still intact -------------------------------------


def _bank_directory(f: ManualFixture, runner: ManualEvaluationRunner, assignment: str, *, warmup_s: float) -> Path:
    """Where one bank keeps its manifest and progress, under the warm-up it was actually built at.

    The warm-up is part of the conditions a bank is keyed by, so it is asked for
    explicitly: answering for a different one would silently name a directory
    that never existed.
    """
    uri = replay_bank_uri(runner.conditions(warmup_s, REPLAY_CUTOFFS), assignment)
    return f.store.path(f"{uri}/{PROGRESS_FILE}", mode="write").parent


def _run_directory(f: ManualFixture, bank: ManualReplayBank) -> Path:
    """The stored run of the bank's first pair."""
    run = bank.pairs[0].run
    assert run is not None
    return f.store.path(run.uri, mode="read").parent


def test_a_stored_bank_is_refused_when_a_run_payload_is_missing(manual_fixture: ManualFixture) -> None:
    """Completed evidence is only evidence while the runs it cites exist: a lost payload is not servable."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    bank = _runner(f, _CountingSimulator(scenario_config)).replay_bank(
        "D05", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS
    )
    (_run_directory(f, bank) / RUN_ARRAYS_FILE).unlink()
    with pytest.raises(ValueError, match="missing"):
        _runner(f, _CountingSimulator(scenario_config)).replay_bank(
            "D05", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS
        )


def test_stored_model_evidence_is_refused_when_a_run_summary_is_corrupted(manual_fixture: ManualFixture) -> None:
    """A summary that no longer hashes to what the manifest recorded is not the run that was evaluated."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f, ARM_CORRUPTED)
    evidence = _runner(f, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_S)
    run = evidence.pairs[0].run
    assert run is not None
    summary = f.store.path(run.uri, mode="read")
    summary.write_text(summary.read_text(encoding="utf-8").replace('"notes"', '"notes_"', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer matches"):
        _runner(f, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_S)


def test_an_interrupted_sweep_is_refused_when_its_arrays_are_missing(manual_fixture: ManualFixture) -> None:
    """The recovery path verifies too: resuming must not build a manifest on runs that are already gone."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    runner = _runner(f, _CountingSimulator(scenario_config))
    bank = runner.replay_bank("D06", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS)
    for manifest in _bank_directory(f, runner, "D06", warmup_s=WARMUP_S).glob("manifest-*.json"):
        manifest.unlink()  # an interrupted sweep: progress recorded, no completed manifest
    (_run_directory(f, bank) / RUN_ARRAYS_FILE).unlink()
    with pytest.raises(ValueError, match="missing"):
        _runner(f, _CountingSimulator(scenario_config)).replay_bank(
            "D06", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS
        )


# --- the manifest itself, not only the runs it cites ---------------------------------------------


def _manifest_of(f: ManualFixture, uri: str) -> Path:
    """The single stored manifest of one evidence directory."""
    directory = f.store.path(f"{uri}/{PROGRESS_FILE}", mode="write").parent
    manifests = sorted(directory.glob("manifest-*.json"))
    assert len(manifests) == 1, manifests
    return manifests[0]


def _without_source_bindings(path: Path) -> str:
    """The manifest's JSON with every run's source bindings removed, as the review's alteration did."""
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    pairs = cast("list[dict[str, object]]", document["pairs"])
    runs = [cast("dict[str, object]", pair["run"]) for pair in pairs if pair.get("run") is not None]
    assert runs, "the manifest must carry runs for this alteration to mean anything"
    assert all(run["sources"] for run in runs), "and those runs must record source bindings to begin with"
    for run in runs:
        run["sources"] = []
    return json.dumps(document, indent=1, sort_keys=True)


def _rename_to_own_digest(path: Path, text: str) -> Path:
    """Store altered content under the name its own content demands, defeating a name check alone."""
    path.unlink()
    renamed = path.parent / f"manifest-{sha256_bytes(text.encode('utf-8'))[:12]}.json"
    renamed.write_text(text, encoding="utf-8")
    return renamed


def test_a_manifest_altered_after_it_was_written_is_refused(manual_fixture: ManualFixture) -> None:
    """A manifest is named by what it contains, so content that no longer hashes to its name is not evidence."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f, ARM_ALL_TEN)
    evidence = _runner(f, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_S)
    path = _manifest_of(f, model_uri(evidence.identity))
    path.write_text(_without_source_bindings(path), encoding="utf-8")
    fresh = _runner(f, _CountingSimulator(scenario_config))
    with pytest.raises(ValueError, match="does not match its content"):
        fresh.evaluate(entry, warmup_s=WARMUP_S)
    assert fresh.pointers == (), "altered evidence must not be pointed at either"


def test_a_renamed_alteration_is_caught_by_the_source_bindings(manual_fixture: ManualFixture) -> None:
    """Renaming to the altered content's own digest passes the name check, so the bindings are checked too."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f, ARM_RENAMED)
    evidence = _runner(f, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_S)
    path = _manifest_of(f, model_uri(evidence.identity))
    _rename_to_own_digest(path, _without_source_bindings(path))
    fresh = _runner(f, _CountingSimulator(scenario_config))
    with pytest.raises(ValueError, match="source"):
        fresh.evaluate(entry, warmup_s=WARMUP_S)
    assert fresh.pointers == ()


def test_a_renamed_alteration_of_a_bank_is_caught_too(manual_fixture: ManualFixture) -> None:
    """The same guard on the baseline side: a bank states the demonstration its runs replayed."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    bank = _runner(f, _CountingSimulator(scenario_config)).replay_bank(
        "D07", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS
    )
    path = _manifest_of(f, replay_bank_uri(bank.conditions, "D07"))
    _rename_to_own_digest(path, _without_source_bindings(path))
    with pytest.raises(ValueError, match="source"):
        _runner(f, _CountingSimulator(scenario_config)).replay_bank(
            "D07", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS
        )


def test_an_interrupted_sweep_is_refused_when_its_recorded_sources_were_altered(manual_fixture: ManualFixture) -> None:
    """The recovery path must check the bindings too, before it turns recorded progress into a manifest.

    A stored run summary does not say what its model trained on -- the bindings
    live only in this experiment's own records -- so verifying run payloads
    cannot detect this alteration, and a resumed sweep would otherwise launder
    it into a fresh manifest carrying a correct name.
    """
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    runner = _runner(f, _CountingSimulator(scenario_config))
    runner.replay_bank("D08", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS)
    directory = _bank_directory(f, runner, "D08", warmup_s=WARMUP_S)
    for manifest in directory.glob("manifest-*.json"):
        manifest.unlink()  # an interrupted sweep: progress recorded, no completed manifest
    progress = directory / PROGRESS_FILE
    progress.write_text(_without_source_bindings(progress), encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        _runner(f, _CountingSimulator(scenario_config)).replay_bank(
            "D08", warmup_s=WARMUP_S, replay_cutoffs=REPLAY_CUTOFFS
        )
    assert not list(directory.glob("manifest-*.json")), "nothing may be installed from altered records"


def _retarget_to_one_parent(path: Path, artifact_id: str) -> str:
    """The review's combined alteration: claim a single parent, and rewrite every binding to agree with it."""
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    assert document["assignment"] is None, "this alteration only means something on the all-ten arm"
    document["assignment"] = "D01"
    pairs = cast("list[dict[str, object]]", document["pairs"])
    runs = [cast("dict[str, object]", pair["run"]) for pair in pairs if pair.get("run") is not None]
    assert runs, "the manifest must carry runs for this alteration to mean anything"
    for run in runs:
        run["sources"] = [artifact_id]
    return json.dumps(document, indent=1, sort_keys=True)


def test_evidence_claiming_another_parent_is_refused(manual_fixture: ManualFixture) -> None:
    """What evidence must contain is decided by the requested study entry, never by the manifest being checked.

    Deriving the expectation from the manifest's own ``assignment`` lets an
    alteration move the goalposts with it: claim one parent, rewrite every
    binding to that parent, rename to the new digest, and the evidence checks
    out against itself. The all-ten arm has no single parent, so a manifest
    claiming one is not this entry's evidence whatever it says.
    """
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f, ARM_ALL_TEN)
    assert entry.arm.assignment is None, "M10 trains on the whole bank"
    evidence = _runner(f, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_TRUSTED)
    path = _manifest_of(f, model_uri(evidence.identity))
    _rename_to_own_digest(path, _retarget_to_one_parent(path, f.manifest.sources["D01"].artifact_id))
    fresh = _runner(f, _CountingSimulator(scenario_config))
    with pytest.raises(ValueError, match="assignment"):
        fresh.evaluate(entry, warmup_s=WARMUP_TRUSTED)
    assert fresh.pointers == (), "altered evidence must not be pointed at either"


ALTERED_FIT_FIELDS = (
    ("solver_alpha", 999.0, 0.36),
    ("recipe_sha256", "a" * 64, 0.37),
    ("weights_sha256", "b" * 64, 0.38),
    ("arm", "S/D02", 0.39),
    ("configuration", "feasible-worst", 0.40),
)
"""Each recorded fit field, with a well-formed replacement and an evidence directory of its own."""


def _alter_fit_field(path: Path, field: str, value: object) -> str:
    """Change one recorded fit field, leaving the identity it stores intact."""
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    fit = cast("dict[str, object]", document["fit"])
    assert field in fit, f"{field} is not a recorded fit field: {sorted(fit)}"
    assert fit[field] != value, "the alteration has to change something to mean anything"
    fit[field] = value
    return json.dumps(document, indent=1, sort_keys=True)


@pytest.mark.parametrize(("field", "value", "warmup_s"), ALTERED_FIT_FIELDS)
def test_an_altered_fit_binding_is_refused(
    manual_fixture: ManualFixture, field: str, value: object, warmup_s: float
) -> None:
    """The whole recorded fit is compared, not the identity it happens to store.

    That identity is itself a recorded field, so checking only it leaves the
    regularization and the fitted model free to claim anything: served evidence
    could misreport the ridge parameter it was solved at, or the weights it was
    produced from, and still be accepted.
    """
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f)
    evidence = _runner(f, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=warmup_s)
    path = _manifest_of(f, model_uri(evidence.identity))
    _rename_to_own_digest(path, _alter_fit_field(path, field, value))
    fresh = _runner(f, _CountingSimulator(scenario_config))
    with pytest.raises(ValueError, match="fit"):
        fresh.evaluate(entry, warmup_s=warmup_s)
    assert fresh.pointers == (), "altered evidence must not be pointed at either"


WARMUP_CLAIMED = 0.41
WARMUP_RESUMED = 0.42
"""A warm-up of its own per verdict case, so each starts from a directory of its own."""


def _claim_success(path: Path) -> str:
    """Claim one failed pair completed, and re-derive the counts so the record stays self-consistent."""
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    pairs = cast("list[dict[str, object]]", document["pairs"])
    target = next((pair for pair in pairs if pair["status"] == "infeasible"), None)
    assert target is not None, "the evidence must contain a failed pair for this alteration to mean anything"
    target["status"] = "completed"
    completed_n = sum(1 for pair in pairs if pair["status"] == "completed")
    document["n_completed"] = completed_n
    document["n_infeasible"] = sum(1 for pair in pairs if pair["status"] == "infeasible")
    document["status"] = "feasible" if completed_n == len(pairs) else "infeasible"
    return json.dumps(document, indent=1, sort_keys=True)


def _flip_recorded_verdict(path: Path) -> str:
    """Contradict one recorded pair's own outcome in an interrupted sweep's progress."""
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    pairs = cast("list[dict[str, object]]", document["pairs"])
    assert pairs, "the interrupted sweep must have recorded pairs"
    assert pairs[0]["status"] == "completed", "expected a completed pair to contradict"
    pairs[0]["status"] = "infeasible"
    return json.dumps(document, indent=1, sort_keys=True)


def test_a_manifest_claiming_success_its_runs_deny_is_refused(manual_fixture: ManualFixture) -> None:
    """A pair's verdict is checked against the run's own stored verdict, not only against the other pairs.

    Re-deriving the status and counts only makes a record agree with itself. The
    run summary carries the verdict the sweep reached, so a manifest that claims
    a completed pair where its own run recorded failure is caught even when the
    counts were adjusted to match and the file renamed to its new digest.
    """
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f)
    evidence = _runner(f, _CountingSimulator(scenario_config, reaches=False)).evaluate(entry, warmup_s=WARMUP_CLAIMED)
    assert evidence.n_infeasible > 0, "the crafted runs must fail before one can be claimed successful"
    path = _manifest_of(f, model_uri(evidence.identity))
    _rename_to_own_digest(path, _claim_success(path))
    fresh = _runner(f, _CountingSimulator(scenario_config, reaches=False))
    with pytest.raises(ValueError, match="verdict"):
        fresh.evaluate(entry, warmup_s=WARMUP_CLAIMED)
    assert fresh.pointers == (), "altered evidence must not be pointed at either"


def test_an_interrupted_sweep_with_an_altered_verdict_is_refused(manual_fixture: ManualFixture) -> None:
    """The recovery path checks verdicts too, before recorded progress becomes a manifest."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    runner = _runner(f, _CountingSimulator(scenario_config))
    runner.replay_bank("D09", warmup_s=WARMUP_RESUMED, replay_cutoffs=REPLAY_CUTOFFS)
    directory = _bank_directory(f, runner, "D09", warmup_s=WARMUP_RESUMED)
    for manifest in directory.glob("manifest-*.json"):
        manifest.unlink()  # an interrupted sweep: progress recorded, no completed manifest
    progress = directory / PROGRESS_FILE
    progress.write_text(_flip_recorded_verdict(progress), encoding="utf-8")
    with pytest.raises(ValueError, match="verdict"):
        _runner(f, _CountingSimulator(scenario_config)).replay_bank(
            "D09", warmup_s=WARMUP_RESUMED, replay_cutoffs=REPLAY_CUTOFFS
        )
    assert not list(directory.glob("manifest-*.json")), "nothing may be installed from altered records"


WARMUP_CLAIMED_SUCCESS = 0.43
WARMUP_RESUMED_SUCCESS = 0.44
WARMUP_ARRAYS = 0.45
WARMUP_ARRAYS_RESUMED = 0.46
"""A warm-up of its own per case, so each starts from a directory of its own."""


def _arrays_of(f: ManualFixture, uri: str) -> Path:
    """The stored arrays beside one run's summary."""
    return f.store.path(uri, mode="read").parent / RUN_ARRAYS_FILE


def _claim_outcome_success(path: Path) -> str:
    """Claim a failed pair's own outcome succeeded, leaving the criteria that failed untouched.

    ``success`` is a stored field tied only to ``reason``, so this stays
    internally valid: the criteria still record the failure, and they still
    agree with the run summary.
    """
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    pairs = cast("list[dict[str, object]]", document["pairs"])
    target = next((pair for pair in pairs if pair["status"] == "infeasible"), None)
    assert target is not None, "the record must contain a failed pair for this alteration to mean anything"
    outcome = cast("dict[str, object]", target["outcome"])
    assert outcome["success"] is False, "the pair must actually have failed"
    outcome["success"] = True
    outcome["reason"] = None
    target["status"] = "completed"
    if "n_completed" in document:  # a model manifest re-derives its counts; a progress file has none
        completed_n = sum(1 for pair in pairs if pair["status"] == "completed")
        document["n_completed"] = completed_n
        document["n_infeasible"] = sum(1 for pair in pairs if pair["status"] == "infeasible")
        document["status"] = "feasible" if completed_n == len(pairs) else "infeasible"
    return json.dumps(document, indent=1, sort_keys=True)


def _alter_arrays_and_record(path: Path, arrays: Path, uri: str) -> str:
    """Change the stored arrays and update only the record's own copy of their digest."""
    arrays.write_bytes(arrays.read_bytes() + b"\x00")
    digest = sha256_file(arrays)
    document = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    pairs = cast("list[dict[str, object]]", document["pairs"])
    runs = [cast("dict[str, object]", p["run"]) for p in pairs if p.get("run") is not None]
    target = next((run for run in runs if run["uri"] == uri), None)
    assert target is not None, f"no recorded run for {uri}"
    assert target["arrays_sha256"] != digest, "the alteration has to change the digest"
    target["arrays_sha256"] = digest
    return json.dumps(document, indent=1, sort_keys=True)


def test_evidence_claiming_its_own_outcome_succeeded_is_refused(manual_fixture: ManualFixture) -> None:
    """A stored success is checked against the verdict its run computed, not only against the status beside it."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f)
    evidence = _runner(f, _CountingSimulator(scenario_config, reaches=False)).evaluate(
        entry, warmup_s=WARMUP_CLAIMED_SUCCESS
    )
    assert evidence.n_infeasible > 0, "the crafted runs must fail before one can claim success"
    path = _manifest_of(f, model_uri(evidence.identity))
    _rename_to_own_digest(path, _claim_outcome_success(path))
    fresh = _runner(f, _CountingSimulator(scenario_config, reaches=False))
    with pytest.raises(ValueError, match="verdict"):
        fresh.evaluate(entry, warmup_s=WARMUP_CLAIMED_SUCCESS)
    assert fresh.pointers == (), "altered evidence must not be pointed at either"


def test_an_interrupted_sweep_claiming_success_is_refused(manual_fixture: ManualFixture) -> None:
    """The recovery path checks the stored success against the run's own verdict too."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    runner = _runner(f, _CountingSimulator(scenario_config, reaches=False))
    runner.replay_bank("D10", warmup_s=WARMUP_RESUMED_SUCCESS, replay_cutoffs=REPLAY_CUTOFFS)
    directory = _bank_directory(f, runner, "D10", warmup_s=WARMUP_RESUMED_SUCCESS)
    for manifest in directory.glob("manifest-*.json"):
        manifest.unlink()
    progress = directory / PROGRESS_FILE
    progress.write_text(_claim_outcome_success(progress), encoding="utf-8")
    with pytest.raises(ValueError, match="verdict"):
        _runner(f, _CountingSimulator(scenario_config, reaches=False)).replay_bank(
            "D10", warmup_s=WARMUP_RESUMED_SUCCESS, replay_cutoffs=REPLAY_CUTOFFS
        )


def test_arrays_contradicting_their_run_summary_are_refused(manual_fixture: ManualFixture) -> None:
    """The run summary holds its own reference to the arrays, so the manifest's copy cannot answer alone."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    entry = _entry(f)
    evidence = _runner(f, _CountingSimulator(scenario_config)).evaluate(entry, warmup_s=WARMUP_ARRAYS)
    run = evidence.pairs[0].run
    assert run is not None
    path = _manifest_of(f, model_uri(evidence.identity))
    _rename_to_own_digest(path, _alter_arrays_and_record(path, _arrays_of(f, run.uri), run.uri))
    fresh = _runner(f, _CountingSimulator(scenario_config))
    with pytest.raises(ValueError, match="run summary"):
        fresh.evaluate(entry, warmup_s=WARMUP_ARRAYS)
    assert fresh.pointers == ()


def test_an_interrupted_sweep_with_arrays_contradicting_its_summary_is_refused(manual_fixture: ManualFixture) -> None:
    """The same independent reference is required when recorded progress is resumed."""
    f = manual_fixture
    scenario_config = load_manual_scenario(f.scenario_file)
    runner = _runner(f, _CountingSimulator(scenario_config))
    bank = runner.replay_bank("D10", warmup_s=WARMUP_ARRAYS_RESUMED, replay_cutoffs=REPLAY_CUTOFFS)
    run = bank.pairs[0].run
    assert run is not None
    directory = _bank_directory(f, runner, "D10", warmup_s=WARMUP_ARRAYS_RESUMED)
    for manifest in directory.glob("manifest-*.json"):
        manifest.unlink()
    progress = directory / PROGRESS_FILE
    progress.write_text(_alter_arrays_and_record(progress, _arrays_of(f, run.uri), run.uri), encoding="utf-8")
    with pytest.raises(ValueError, match="run summary"):
        _runner(f, _CountingSimulator(scenario_config)).replay_bank(
            "D10", warmup_s=WARMUP_ARRAYS_RESUMED, replay_cutoffs=REPLAY_CUTOFFS
        )
