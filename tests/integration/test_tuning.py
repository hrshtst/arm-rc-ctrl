# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-024: the tuning protocol fixes search spaces, objective, seed, budget, penalty, and dev scenarios."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from arm_rc_ctrl.config import ConfigError
from arm_rc_ctrl.controllers.tracking import TrackerConfig
from arm_rc_ctrl.data.preprocess import preprocess_demonstration
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, RawDemonstrationRecord, load_record
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.tuning import (
    DevelopmentPulse,
    Feasibility,
    GainRange,
    Objective,
    TuningProtocol,
    evaluate_gains,
    load_protocol,
    main,
    run_study,
    sample_gains,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "baseline_gains_1a.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 8, 30, 11, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord]:
    """Processed fixture dataset, its scenario, and its record."""
    base = tmp_path_factory.mktemp("tuning")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    result = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    return result.samples, load_scenario(SCENARIO), result.record


@pytest.fixture(scope="module")
def protocol() -> TuningProtocol:
    """The committed protocol."""
    return load_protocol(PROTOCOL)


def test_committed_protocol_fixes_everything(protocol: TuningProtocol) -> None:
    """Search spaces, objective, penalty, seed, budget, scenario, and development scenarios are versioned."""
    assert protocol.name == "baseline-gains-1a"
    assert protocol.scenario == REPO_ROOT / "configs" / "tasks" / "task_1a.toml"
    assert load_scenario(protocol.scenario).name == "task-1a-reach"
    assert protocol.budget == 64
    assert protocol.sampler_seed == 20260830
    assert protocol.objective == Objective("median_move_joint_rmse", 10.0)
    assert protocol.search.pd.kp == GainRange(1.0, 300.0, log=True)
    assert protocol.search.computed_torque.kd == GainRange(2.0, 120.0, log=True)
    assert protocol.development.initial_posture_offsets[0] == (0.0, 0.0)
    assert len(protocol.development.initial_posture_offsets) == 4


def test_sampling_is_seeded_log_uniform_and_within_bounds(protocol: TuningProtocol) -> None:
    """The same seed reproduces the same gains; every draw respects the per-tracker bounds."""
    first = [sample_gains(protocol, "pd", 2, np.random.default_rng(1)) for _ in range(20)]
    second = [sample_gains(protocol, "pd", 2, np.random.default_rng(1)) for _ in range(20)]
    assert first == second
    for gains in first:
        assert gains.type == "pd"
        assert all(1.0 <= k <= 300.0 for k in gains.kp)
        assert all(0.05 <= k <= 60.0 for k in gains.kd)
    ct = sample_gains(protocol, "computed_torque", 2, np.random.default_rng(2))
    assert ct.type == "computed_torque"
    assert all(10.0 <= k <= 900.0 for k in ct.kp)
    # Log-uniform: the log of many draws is roughly uniform, so its mean sits near the log-midpoint.
    rng = np.random.default_rng(3)
    logs = np.log([k for _ in range(400) for k in sample_gains(protocol, "pd", 2, rng).kp])
    assert abs(float(np.mean(logs)) - 0.5 * (math.log(1.0) + math.log(300.0))) < 0.3


def test_evaluate_gains_reports_every_component(
    dataset: tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord], protocol: TuningProtocol
) -> None:
    """A feasible trial's objective is the median movement RMSE over the development scenarios."""
    samples, scenario, _record = dataset
    gains = TrackerConfig("computed_torque", (100.0, 100.0), (20.0, 20.0))
    objective, feasible, components = evaluate_gains(protocol, scenario, samples, gains)
    assert len(components) == 4
    assert [c.index for c in components] == [0, 1, 2, 3]
    assert components[0].initial_q == scenario.task.initial_q
    assert all(c.termination == "completed" for c in components)
    rmses = [c.move_joint_rmse for c in components]
    assert all(r is not None and math.isfinite(r) for r in rmses)
    assert all(set(c.criteria) == {"completed", "dwell_in_tolerance", "dwell_stationary"} for c in components)
    if feasible:
        assert objective == pytest.approx(float(np.median([r for r in rmses if r is not None])))
    else:
        assert objective == protocol.objective.infeasible_penalty
        assert any(not c.feasible for c in components)


def test_infeasible_trials_receive_the_documented_penalty(
    dataset: tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord], protocol: TuningProtocol
) -> None:
    """A limit violation in any development scenario yields the penalty, with the cause recorded."""
    samples, scenario, _record = dataset
    strict = dataclasses.replace(scenario, limits=dataclasses.replace(scenario.limits, velocity=(0.3, 0.3)))
    objective, feasible, components = evaluate_gains(
        protocol, strict, samples, TrackerConfig("pd", (20.0, 10.0), (2.0, 1.0))
    )
    assert objective == 10.0
    assert feasible is False
    assert all(c.termination == "limit_violation" for c in components)
    assert all(c.move_joint_rmse is None for c in components)
    assert all(
        c.criteria == {"completed": False, "dwell_in_tolerance": False, "dwell_stationary": False} for c in components
    )


def test_study_is_deterministic_and_selects_the_minimum(
    dataset: tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord], protocol: TuningProtocol
) -> None:
    """Two runs with the same seed give identical trials; the best trial has the lowest objective."""
    samples, scenario, record = dataset
    small = dataclasses.replace(protocol, budget=4)
    a = run_study(small, record, samples, "pd", scenario_file=SCENARIO, scenario=scenario)
    b = run_study(small, record, samples, "pd", scenario_file=SCENARIO, scenario=scenario)
    assert a == b
    assert a.budget == 4
    assert a.sampler_seed == protocol.sampler_seed
    assert [t.number for t in a.trials] == [0, 1, 2, 3]
    assert a.best.objective == min(t.objective for t in a.trials)
    assert a.feasible_trials == sum(1 for t in a.trials if t.feasible)
    ct = run_study(small, record, samples, "computed_torque", scenario_file=SCENARIO, scenario=scenario)
    assert ct.sampler_seed == a.sampler_seed
    assert ct.budget == a.budget  # equal budget for both trackers


def test_study_refuses_a_dataset_from_another_scenario(
    dataset: tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord], protocol: TuningProtocol
) -> None:
    """The protocol's scenario file must be the one the dataset was derived under."""
    samples, _scenario, record = dataset
    with pytest.raises(ValueError, match="was derived under scenario digest"):
        run_study(dataclasses.replace(protocol, budget=1), record, samples, "pd")  # protocol scenario = task_1a


def test_study_result_consistency_is_enforced(
    dataset: tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord], protocol: TuningProtocol
) -> None:
    """Budget must match the trial count and best must be the minimum."""
    samples, scenario, record = dataset
    result = run_study(
        dataclasses.replace(protocol, budget=2), record, samples, "pd", scenario_file=SCENARIO, scenario=scenario
    )
    with pytest.raises(ValueError, match="trials but budget"):
        dataclasses.replace(result, budget=3)
    worst = max(result.trials, key=lambda t: t.objective)
    if worst != result.best:
        with pytest.raises(ValueError, match="lowest objective"):
            dataclasses.replace(result, best=worst)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("budget = 64", "budget = 0", "budget must be >= 1"),
        ("sampler_seed = 20260830", "sampler_seed = -1", "sampler_seed must be non-negative"),
        ("infeasible_penalty = 10.0", "infeasible_penalty = 0.0", "infeasible_penalty must be positive"),
        (
            "kp = { low = 1.0, high = 300.0, log = true }",
            "kp = { low = 300.0, high = 1.0, log = true }",
            "0 < low < high",
        ),
        (
            "initial_posture_offsets = [[0.0, 0.0], [0.05, -0.05], [-0.05, 0.05], [0.08, 0.0]]",
            "initial_posture_offsets = []",
            "must not be empty",
        ),
    ],
    ids=["budget", "seed", "penalty", "range", "offsets"],
)
def test_protocol_invariants(tmp_path: Path, old: str, new: str, expected: str) -> None:
    """Invalid protocol files fail to load."""
    text = PROTOCOL.read_text().replace(old, new, 1)
    path = tmp_path / "studies" / "bad.toml"
    path.parent.mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "task_1a.toml").write_text((REPO_ROOT / "configs" / "tasks" / "task_1a.toml").read_text())
    path.write_text(text)
    with pytest.raises(ConfigError, match=expected):
        load_protocol(path)


def test_dwell_criteria_decide_feasibility(
    dataset: tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord], protocol: TuningProtocol
) -> None:
    """A trial that completes but fails a dwell criterion is infeasible and penalised."""
    samples, scenario, _record = dataset
    gains = TrackerConfig("computed_torque", (100.0, 100.0), (20.0, 20.0))
    lenient = dataclasses.replace(
        scenario, task=dataclasses.replace(scenario.task, dwell_min_fraction=0.0, dwell_max_velocity=10.0)
    )
    demanding = dataclasses.replace(
        scenario, task=dataclasses.replace(scenario.task, dwell_min_fraction=1.0, dwell_max_velocity=1e-6)
    )
    objective_ok, feasible_ok, _ = evaluate_gains(protocol, lenient, samples, gains)
    objective_bad, feasible_bad, components = evaluate_gains(protocol, demanding, samples, gains)
    assert feasible_ok is True
    assert objective_ok < protocol.objective.infeasible_penalty
    assert feasible_bad is False
    assert objective_bad == protocol.objective.infeasible_penalty
    assert all(c.termination == "completed" and c.move_joint_rmse is not None for c in components)
    assert all(c.criteria["completed"] and not c.criteria["dwell_stationary"] for c in components)


def test_command_line_runs_a_study_writes_the_report_and_freezes_gains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI resolves the dataset through the store, runs the seeded study, and writes report + frozen TOML."""
    import json

    from arm_rc_ctrl.config import load_config, to_mapping
    from arm_rc_ctrl.experiments.tuning import StudyReport, main

    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    # A small protocol bound to the fixture scenario, mirroring the committed one otherwise.
    small = tmp_path / "studies" / "small.toml"
    small.parent.mkdir()
    text = (
        PROTOCOL.read_text()
        .replace('scenario = "../tasks/task_1a.toml"', f'scenario = "{SCENARIO}"')
        .replace("budget = 64", "budget = 2")
    )
    small.write_text(text)
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    report = tmp_path / "report.json"
    frozen = tmp_path / "pd_frozen.toml"
    args = [
        "--protocol",
        str(small),
        "--dataset",
        str(processed.record_file),
        "--tracker",
        "pd",
        "--report",
        str(report),
        "--freeze",
        str(frozen),
        "--exploratory",
    ]
    assert main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["budget"] == 2
    assert printed["best_trial"] in (0, 1)
    loaded = json.loads(report.read_text())
    assert loaded["result"]["budget"] == 2
    assert len(loaded["result"]["trials"]) == 2
    assert loaded["provenance"]["seeds"] == {"sampler": 20260830}
    assert loaded["provenance"]["artifacts"][0]["uri"] == processed.record.artifact.payload.uri
    assert str(tmp_path) not in report.read_text()  # no machine paths in the curated report
    assert loaded["scenario_file"] == "tests/fixtures/configs/planar_2dof_fixture.toml"
    assert loaded["provenance"]["config_json"].count(str(tmp_path)) == 0
    gains = load_config(frozen, TrackerConfig)
    assert gains.type == "pd"
    assert to_mapping(gains) == loaded["result"]["best"]["gains"]
    assert frozen.read_text().startswith("# Frozen by study")
    with pytest.raises(FileExistsError, match="immutable"):
        main(args)
    del StudyReport


# --- M3-015: robustness-constrained protocol version 2 ---------------------------------------------

PROTOCOL_V2 = REPO_ROOT / "configs" / "studies" / "baseline_gains_1a_v2.toml"


def test_committed_v2_protocol_adds_robustness_constraints() -> None:
    """v2 keeps v1's search spaces and adds directions, force pulses, a saturation bound, and the nominal objective."""
    v1 = load_protocol(PROTOCOL)
    v2 = load_protocol(PROTOCOL_V2)
    assert v2.name == "baseline-gains-1a-v2"
    assert v2.scenario == v1.scenario
    assert v2.search == v1.search
    assert v2.sampler_seed != v1.sampler_seed
    assert v2.objective.kind == "nominal_move_joint_rmse"
    assert v2.feasibility.max_saturation_fraction == 0.05
    offsets = v2.development.initial_posture_offsets
    assert offsets[0] == (0.0, 0.0)
    assert len(offsets) == 17
    norms = sorted({round(math.hypot(*o), 6) for o in offsets[1:]})
    assert norms == [0.04, 0.08]  # distinct from every confirmatory level (0.05 and 0.06 rad)
    pulses = v2.development.force_pulses
    assert len(pulses) == 8
    assert {p.direction_deg for p in pulses} == {45.0, 135.0, 225.0, 315.0}  # not the confirmatory 0/90/180/270
    assert {p.start_s for p in pulses} == {2.5}  # not the confirmatory 2.0 s
    assert {p.magnitude_n for p in pulses} == {6.0, 9.0}
    assert v1.feasibility.max_saturation_fraction == 1.0  # v1 unchanged: no headroom requirement
    assert v1.development.force_pulses == ()


def test_v2_validation() -> None:
    """The nominal objective needs the unperturbed posture first; pulses are validated; the bound lies in [0, 1]."""
    v2 = load_protocol(PROTOCOL_V2)
    shifted = dataclasses.replace(v2.development, initial_posture_offsets=v2.development.initial_posture_offsets[1:])
    with pytest.raises(ValueError, match="needs the unperturbed posture as the first development offset"):
        dataclasses.replace(v2, development=shifted)
    with pytest.raises(ValueError, match="max_saturation_fraction must lie in"):
        Feasibility(1.5)
    with pytest.raises(ValueError, match="duration_s must be > 0"):
        DevelopmentPulse(2.5, 0.0, 6.0, 45.0)
    assert DevelopmentPulse(2.5, 0.2, 6.0, 90.0).pulse().force == pytest.approx((0.0, 6.0), abs=1e-12)


def test_force_scenarios_and_saturation_bound_decide_feasibility(
    dataset: tuple[SampleSet, ScenarioConfig, ProcessedDatasetRecord],
) -> None:
    """Force scenarios are evaluated at the nominal posture, saturation is reported, and the bound is enforced."""
    samples, scenario, _ = dataset
    v2 = load_protocol(PROTOCOL_V2)
    development = dataclasses.replace(
        v2.development,
        initial_posture_offsets=((0.0, 0.0), (0.01, 0.0)),
        force_pulses=(DevelopmentPulse(0.12, 0.05, 2.0, 45.0),),
    )
    relaxed = dataclasses.replace(v2, scenario=SCENARIO, development=development, feasibility=Feasibility(1.0))
    strong = TrackerConfig("pd", (300.0, 300.0), (60.0, 60.0))
    objective, feasible, components = evaluate_gains(relaxed, scenario, samples, strong)
    assert [c.kind for c in components] == ["posture", "posture", "force"]
    assert components[2].pulse == development.force_pulses[0]
    assert components[2].initial_q == tuple(scenario.task.initial_q)
    assert all(c.saturation_fraction is not None and 0.0 <= c.saturation_fraction <= 1.0 for c in components)
    worst = max(c.saturation_fraction or 0.0 for c in components)
    assert worst > 0.0  # kp = 300 saturates the fixture's 10/5 N*m actuators
    if feasible:
        assert objective == components[0].move_joint_rmse  # the nominal scenario alone ranks feasible trials
    tight = dataclasses.replace(relaxed, feasibility=Feasibility(max_saturation_fraction=worst / 2))
    objective, feasible, components = evaluate_gains(tight, scenario, samples, strong)
    assert feasible is False
    assert objective == tight.objective.infeasible_penalty
    assert any(
        c.saturation_fraction is not None and c.saturation_fraction > worst / 2 and not c.feasible for c in components
    )


def test_command_line_refuses_to_freeze_without_a_feasible_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A study in which every trial fails the robustness constraints writes its report but freezes nothing."""
    root = tmp_path / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = tmp_path / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    protocol = tmp_path / "hopeless.toml"
    protocol.write_text(
        PROTOCOL_V2.read_text()
        .replace('scenario = "../tasks/task_1a.toml"', f'scenario = "{SCENARIO.as_posix()}"')
        .replace("budget = 128", "budget = 2")
        .replace("max_saturation_fraction = 0.05", "max_saturation_fraction = 0.0")
        .replace("kp = { low = 1.0, high = 300.0, log = true }", "kp = { low = 250.0, high = 300.0, log = true }")
    )
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(root))
    report = tmp_path / "report.json"
    frozen = tmp_path / "frozen.toml"
    argv = [
        "--protocol",
        str(protocol),
        "--dataset",
        str(processed.record_file),
        "--tracker",
        "pd",
        "--report",
        str(report),
        "--freeze",
        str(frozen),
        "--exploratory",
    ]
    with pytest.raises(RuntimeError, match="satisfied every development scenario; report written, nothing frozen"):
        main(argv)
    assert report.is_file()
    assert not frozen.exists()
    assert json.loads(report.read_text())["result"]["feasible_trials"] == 0
    capsys.readouterr()
