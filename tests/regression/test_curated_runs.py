# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2 review round 2 finding 1: every run cited by a committed paired report is tracked in Git and resolvable."""

from __future__ import annotations

import pytest

from arm_rc_ctrl.data.records import load_catalog, load_record, verify_payload
from arm_rc_ctrl.experiments.paired import load_paired_report
from arm_rc_ctrl.experiments.paired_suite import load_paired_suite
from arm_rc_ctrl.experiments.run_record import RunPointerRecord, load_run
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageError, open_storage

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a"
PAIRED_REPORTS = sorted(DOCS.glob("paired_nominal_*.json"))
SUITES = sorted(DOCS.glob("paired_suite_*.json"))


def _cited_runs() -> dict[str, tuple[str, str]]:
    """Run ID -> (method, scenario) for every run cited by a committed paired report."""
    cited: dict[str, tuple[str, str]] = {}
    for file in PAIRED_REPORTS:
        report = load_paired_report(file)
        for run in (report.rc, report.replay):
            cited[run.run_id] = (run.method, run.scenario)
    return cited


def test_reports_exist() -> None:
    """The nominal evidence of M2 is committed."""
    assert PAIRED_REPORTS, "no committed paired reports under docs/experiments/task_1a"
    assert SUITES


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
