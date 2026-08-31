# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reservoir-seed sensitivity panel: leading trials, per-seed outcomes, summaries, report, and CLI (M3-016)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arm_rc_ctrl.data.preprocess import PreprocessResult, preprocess_demonstration
from arm_rc_ctrl.data.records import RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments import esn_stability
from arm_rc_ctrl.experiments.esn_objective import TrialContext, TrialEvaluation
from arm_rc_ctrl.experiments.esn_search import EsnSearchProtocol, TrialPoint, load_esn_search, protocol_digest
from arm_rc_ctrl.experiments.esn_stability import (
    ConfigurationStability,
    SeedOutcome,
    leading_trials,
    load_stability,
    main,
    render_markdown,
    run_stability,
    stability_to_json,
)
from arm_rc_ctrl.experiments.esn_study import EsnStudyReport, report_to_json
from arm_rc_ctrl.experiments.studies import StudySummary, TrialRecord
from arm_rc_ctrl.provenance import collect_provenance
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

REPO_ROOT = repository_root()
PROTOCOL = REPO_ROOT / "configs" / "studies" / "esn_search_1a.toml"
MODEL = REPO_ROOT / "tests" / "fixtures" / "configs" / "esn_fixture.toml"
RAW_RECORD = REPO_ROOT / "tests" / "fixtures" / "records" / "raw-20260830-287036d83d46.toml"
RAW_LOG = REPO_ROOT / "tests" / "fixtures" / "raw" / "demo.sklog.npz"
SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "planar_2dof_fixture.toml"
PREPROCESS = REPO_ROOT / "configs" / "preprocessing" / "default.toml"
FIXED_TIME = datetime(2026, 8, 31, 15, 0, 0, tzinfo=UTC)
POINT_A = TrialPoint(100, 0.9, 0.9, 0.3, 0.5, 31, 1e-2, 20.0, 20.0)
POINT_B = TrialPoint(150, 1.0, 0.8, 0.2, 0.3, 77, 3e-2, 15.0, 15.0)


def trial(number: int, value: float, point: TrialPoint, *, feasible: bool, label: str | None = None) -> TrialRecord:
    """A stored trial."""
    labels = {"reason": "" if feasible else "scenario 0: divergence"}
    if label is not None:
        labels["armrc.comparison"] = label
    return TrialRecord(
        number,
        "COMPLETE",
        value,
        {k: float(v) for k, v in point.params().items()},
        flags={"feasible": feasible},
        labels=labels,
    )


def make_report(protocol: EsnSearchProtocol, protocol_file: str) -> EsnStudyReport:
    """A finished study: trial 1 best (feasible), trial 0 feasible but worse, trial 2 infeasible."""
    trials = (
        trial(0, 0.2, POINT_A, feasible=True, label="anchor"),
        trial(1, 0.05, POINT_B, feasible=True),
        trial(2, 10.0, replace(POINT_B, seed=78), feasible=False),
    )
    summary = StudySummary(
        protocol.name, f"armrc://optuna/{protocol.name}.db", "minimize", {}, trials, 3, 0, 1, 0.05, "feasible"
    )
    provenance = collect_provenance(
        {"protocol": protocol.name}, seeds={"sampler": 5}, artifacts=[], exploratory=True, now=FIXED_TIME
    )
    return EsnStudyReport(
        protocol=protocol.name,
        protocol_file=protocol_file,
        protocol_sha256=protocol_digest(protocol),
        dataset="processed-20260830-feaf73e6663c",
        tracker=protocol.tracker,
        tracker_sha256="0" * 64,
        budget=3,
        trials_run=3,
        summary=summary,
        best_point=POINT_B,
        n_feasible=2,
        provenance=provenance,
    )


@pytest.fixture(scope="module")
def prepared(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[StorageRoot, Path, PreprocessResult, EsnSearchProtocol, Path]:
    """The fixture dataset in a store and the committed protocol re-targeted at it (three-trial budget)."""
    base = tmp_path_factory.mktemp("stability")
    root = base / "store"
    root.mkdir()
    store = StorageRoot(root, repositories=(REPO_ROOT,))
    raw = load_record(RAW_RECORD, RawDemonstrationRecord)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(RAW_LOG.read_bytes())
    records = base / "repo"
    (records / "data" / "records" / "processed").mkdir(parents=True)
    processed = preprocess_demonstration(
        RAW_RECORD, SCENARIO, PREPROCESS, store=store, records_root=records, exploratory=True, now=FIXED_TIME
    )
    text = PROTOCOL.read_text(encoding="utf-8")
    text = re.sub(r"^name = .*$", 'name = "esn-search-stability"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^scenario = .*$", f'scenario = "{SCENARIO}"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^model = .*$", f'model = "{MODEL}"', text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^budget = .*$", "budget = 3", text, count=1, flags=re.MULTILINE)
    text = text[: text.index("[[comparison]]")] + text[text.index("[development]") :]
    text = text[: text.index("[development]")] + "[development]\ninitial_posture_offsets = [[0.0, 0.0]]\n"
    protocol_file = base / "esn_search_stability.toml"
    protocol_file.write_text(text, encoding="utf-8")
    return store, records, processed, load_esn_search(protocol_file), protocol_file


def test_leading_trials_are_the_best_feasible_ones(
    prepared: tuple[StorageRoot, Path, PreprocessResult, EsnSearchProtocol, Path],
) -> None:
    """Feasible trials ordered by objective; infeasible ones are never leading."""
    _, _, _, protocol, protocol_file = prepared
    report = make_report(protocol, protocol_file.name)
    assert leading_trials(report, 1) == [1]
    assert leading_trials(report, 5) == [1, 0]


def test_panel_records_every_seed_and_summarizes_feasible_outcomes(
    prepared: tuple[StorageRoot, Path, PreprocessResult, EsnSearchProtocol, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Each leading trial is re-evaluated per seed; summaries are recomputed on load; guards hold."""
    store, records, processed, protocol, protocol_file = prepared
    report = make_report(protocol, protocol_file.name)
    context = TrialContext.load(protocol, store=store, dataset_file=processed.record_file, records_root=records)
    calls: list[tuple[int, int]] = []

    def fake_evaluate(
        _protocol: EsnSearchProtocol, _context: TrialContext, point: TrialPoint, **_: object
    ) -> TrialEvaluation:
        calls.append((point.n_neurons, point.seed))
        feasible = point.seed % 2 == 1
        objective = 0.01 * point.seed if feasible else 10.0
        return TrialEvaluation(
            point,
            objective,
            feasible,
            not feasible,
            None if feasible else "scenario 0: divergence",
            1e-4,
            1,
            (),
            (objective,),
        )

    monkeypatch.setattr(esn_stability, "evaluate_point", fake_evaluate)
    result = run_stability(
        report,
        tmp_path / "study.json",
        protocol,
        context=context,
        top=2,
        seeds=[1, 2, 3],
        exploratory=True,
        now=FIXED_TIME,
    )
    assert [c.trial for c in result.configurations] == [1, 0]
    assert calls == [(150, 1), (150, 2), (150, 3), (100, 1), (100, 2), (100, 3)]
    first = result.configurations[0]
    assert first.label is None
    assert result.configurations[1].label == "anchor"
    assert first.point == POINT_B
    assert [o.feasible for o in first.outcomes] == [True, False, True]
    assert (first.feasible_seeds, first.objective_min, first.objective_max) == (2, 0.01, 0.03)
    assert first.objective_median == pytest.approx(0.02)
    assert result.panel_seeds == (1, 2, 3)
    assert result.provenance.seeds == {"panel.0": 1, "panel.1": 2, "panel.2": 3}
    out = tmp_path / "stability.json"
    out.write_text(stability_to_json(result) + "\n", encoding="utf-8")
    assert load_stability(out) == result
    text = render_markdown(result)
    assert "| 1 | - | 0.05 | 77 | 2/3 | 0.02 | 0.01 | 0.03 |" in text
    assert "| 2 | False | 10 | 0 | scenario 0: divergence |" in text
    tampered = json.loads(out.read_text(encoding="utf-8"))
    tampered["configurations"][0]["feasible_seeds"] = 3
    out.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="stored stability summary"):
        load_stability(out)
    with pytest.raises(ValueError, match="was not produced by protocol"):
        run_stability(
            report,
            tmp_path / "s.json",
            replace(protocol, budget=4),
            context=context,
            top=1,
            seeds=[1],
            exploratory=True,
        )
    with pytest.raises(ValueError, match="distinct non-negative"):
        run_stability(report, tmp_path / "s.json", protocol, context=context, top=1, seeds=[1, 1], exploratory=True)
    with pytest.raises(ValueError, match="no feasible trial"):
        run_stability(
            replace(
                report,
                n_feasible=0,
                best_point=None,
                summary=replace(
                    report.summary, trials=report.summary.trials[2:], n_complete=1, best_number=None, best_value=None
                ),
            ),
            tmp_path / "s.json",
            protocol,
            context=context,
            top=1,
            seeds=[1],
            exploratory=True,
        )
    with pytest.raises(ValueError, match="stored stability summary"):
        ConfigurationStability(
            0,
            None,
            POINT_A,
            0.1,
            (SeedOutcome(1, feasible=True, objective=0.1, reason=None, scenarios_evaluated=1),),
            feasible_seeds=2,
        )


def test_command_evaluates_the_leading_trial_for_real(
    prepared: tuple[StorageRoot, Path, PreprocessResult, EsnSearchProtocol, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The command trains and simulates the leading trial with each panel seed and writes both files."""
    store, records, processed, protocol, protocol_file = prepared
    monkeypatch.setenv("ARM_RC_CTRL_STORAGE_ROOT", str(store.root))
    report_file = tmp_path / "esn_search.json"
    report_file.write_text(report_to_json(make_report(protocol, protocol_file.name)) + "\n", encoding="utf-8")
    output = tmp_path / "stability.json"
    argv = [
        "--report", str(report_file), "--protocol", str(protocol_file), "--dataset", str(processed.record_file),
        "--top", "1", "--seeds", "5", "6", "--output", str(output), "--markdown", str(tmp_path / "stability.md"),
        "--records-root", str(records), "--exploratory",
    ]  # fmt: skip
    assert main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["configurations"][0]["trial"] == 1
    result = load_stability(output)
    assert [o.seed for o in result.configurations[0].outcomes] == [5, 6]
    assert all(o.scenarios_evaluated == 1 for o in result.configurations[0].outcomes)
    assert (tmp_path / "stability.md").read_text(encoding="utf-8").startswith("# Reservoir-seed sensitivity panel")
    with pytest.raises(FileExistsError, match="refusing"):
        main(argv)
