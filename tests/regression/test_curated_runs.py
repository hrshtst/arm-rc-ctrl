# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Every run cited by a committed paired report or robustness suite is tracked, resolvable, and linked (M3-017)."""

from __future__ import annotations

from pathlib import Path

import pytest

from arm_rc_ctrl.data.records import load_catalog, load_record, verify_payload
from arm_rc_ctrl.experiments.baselines import baseline_method
from arm_rc_ctrl.experiments.paired import load_paired_report
from arm_rc_ctrl.experiments.paired_suite import load_paired_suite
from arm_rc_ctrl.experiments.robustness import load_suite, suite_to_markdown
from arm_rc_ctrl.experiments.run_record import RunPointerRecord, load_run
from arm_rc_ctrl.experiments.tracking import RUN_ID_TAG, MlflowTracker
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, open_storage

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a"
PAIRED_REPORTS = sorted(DOCS.glob("paired_nominal_*.json"))
SUITES = sorted(DOCS.glob("paired_suite_*.json"))
ROBUSTNESS = sorted(DOCS.glob("robustness_*.json"))


def _cited_runs() -> dict[str, tuple[str, str]]:
    """Run ID -> (method, scenario) for every run cited by a committed paired report."""
    cited: dict[str, tuple[str, str]] = {}
    for file in PAIRED_REPORTS:
        report = load_paired_report(file)
        for run in (report.rc, report.replay):
            cited[run.run_id] = (run.method, run.scenario)
    for file in ROBUSTNESS:
        suite = load_suite(file)
        arms = {arm.name: arm for arm in suite.arms}
        for run in suite.runs:
            arm = arms[run.arm]
            cited[run.run_id] = (f"{arm.generator}+{baseline_method(arm.tracker)}", suite.scenario)
    return cited


def test_reports_exist() -> None:
    """The nominal evidence of M2 is committed."""
    assert PAIRED_REPORTS, "no committed paired reports under docs/experiments/task_1a"
    assert SUITES
    assert ROBUSTNESS, "no committed robustness suites under docs/experiments/task_1a"


@pytest.mark.parametrize("run_id", sorted(_cited_runs()))
def test_cited_runs_have_pointer_records_and_catalog_entries(run_id: str) -> None:
    """Each cited run has a Git pointer record whose method/scenario match the report and a catalog entry."""
    method, scenario = _cited_runs()[run_id]
    pointer_file = REPO_ROOT / "data" / "records" / "runs" / f"{run_id}.toml"
    assert pointer_file.is_file(), f"missing pointer record for {run_id}"
    pointer = load_record(pointer_file, RunPointerRecord)
    assert pointer.artifact.artifact_id == run_id
    assert (pointer.method, pointer.scenario) == (method, scenario)
    assert pointer.artifact.origin.project_dirty is False
    entry = load_catalog(REPO_ROOT / "data" / "catalog.toml").find(run_id)
    assert entry is not None
    assert entry.kind == "run"
    assert entry.record == f"data/records/runs/{run_id}.toml"
    assert entry.sha256 == pointer.artifact.payload.sha256


def test_suites_cite_the_same_runs() -> None:
    """The two-tracker suites reference runs of the committed paired reports only."""
    cited = _cited_runs()
    for file in SUITES:
        suite = load_paired_suite(file)
        for report in (suite.pd, suite.computed_torque):
            assert report.rc.run_id in cited
            assert report.replay.run_id in cited


@pytest.mark.parametrize("run_id", sorted(_cited_runs()))
def test_cited_run_payloads_verify_through_the_store(run_id: str) -> None:
    """With the configured store, each cited run's payload resolves and verifies (skipped without the store)."""
    pointer = load_record(REPO_ROOT / "data" / "records" / "runs" / f"{run_id}.toml", RunPointerRecord)
    try:
        store = open_storage()
        verify_payload(store, pointer.artifact)
    except (StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"configured external store with {run_id} not available: {exc}")
    run = load_run(store, pointer)
    assert run.summary.termination.kind == pointer.termination_kind
    assert run.summary.provenance.project_dirty is False


@pytest.mark.parametrize("file", ROBUSTNESS, ids=[f.stem for f in ROBUSTNESS])
def test_robustness_suites_track_every_run_and_render_their_markdown(file: Path) -> None:
    """Every suite run names its pointer record and MLflow run; the committed Markdown is the rendered suite."""
    suite = load_suite(file)
    assert suite.provenance.project_dirty is False
    assert len(suite.runs) == len(suite.arms) * len(suite.scenarios)
    for run in suite.runs:
        pointer = f"data/records/runs/{run.run_id}.toml"
        assert run.pointer == pointer
        assert (REPO_ROOT / pointer).is_file()
        assert run.mlflow_run_id, f"{run.run_id} has no MLflow run"
    assert suite_to_markdown(suite) == file.with_suffix(".md").read_text(encoding="utf-8")


@pytest.mark.parametrize("file", ROBUSTNESS, ids=[f.stem for f in ROBUSTNESS])
def test_robustness_suite_mlflow_links_resolve_through_the_store(file: Path) -> None:
    """With the configured store, each suite run's MLflow run carries the run ID tag (skipped without the store)."""
    suite = load_suite(file)
    try:
        tracker = MlflowTracker(open_storage())
        tags = {run.run_id: tracker.tags(run.mlflow_run_id or "") for run in suite.runs}
    except (StorageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        pytest.skip(f"configured external store with the MLflow runs not available: {exc}")
    for run_id, run_tags in tags.items():
        assert run_tags[RUN_ID_TAG] == run_id
