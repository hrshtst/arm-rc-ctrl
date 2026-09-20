# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-011: the committed reproduction audit says what it checked, and its record agrees with the repository.

Continuous integration has no external store, so this lock checks what the
committed audit record alone can answer: that it passed every step it declares,
that it covers the properties the task asks for, that its re-simulation sample
is the frozen 300 runs and reproduced them bitwise, that the bundle it hands off
cites repository files that still have those digests, and that its rendering is
generated from the record rather than written beside it.
"""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.manual_audit import AUDIT_SCHEMA_VERSION, load_audit, render_audit_markdown
from arm_rc_ctrl.experiments.manual_results import load_results
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration"
RECORD = DOCS / "audit" / "reproduction_audit_v1.json"
MARKDOWN = DOCS / "audit" / "reproduction_audit_v1.md"
EXPECTED_STEPS = (
    "checkout",
    "sources",
    "datasets",
    "manifests",
    "fits",
    "payloads_and_metrics",
    "aggregates",
    "completeness",
    "figures",
    "resimulation",
    "gates",
)


def test_the_audit_passed_every_step_it_declares() -> None:
    """A retained failure would be listed here; the verdict re-derives from the steps when the record loads."""
    audit = load_audit(RECORD)
    assert [step.name for step in audit.steps] == list(EXPECTED_STEPS)
    assert [step.name for step in audit.steps if not step.ok] == []
    assert (audit.ok, audit.n_failures) == (True, 0)
    assert audit.schema_version == AUDIT_SCHEMA_VERSION


def test_the_audit_covers_every_run_every_manifest_and_every_fit() -> None:
    """Verification and recomputation cover the whole study; only re-simulation is sampled."""
    checked = {step.name: step.checked for step in load_audit(RECORD).steps}
    assert checked["payloads_and_metrics"] == 31980
    assert checked["manifests"] == 246
    assert checked["fits"] == 186
    assert checked["resimulation"] == 300
    assert checked["gates"] == 2, "the documented nox and pre-commit gates ran inside the audit"


def test_the_frozen_subset_reproduced_bitwise_under_the_declared_tolerances() -> None:
    """The sample was frozen before execution, and every run of it reproduced the stored arrays exactly."""
    audit = load_audit(RECORD)
    assert len(audit.resimulated) == 300
    assert all(run.bitwise for run in audit.resimulated)
    assert all(run.max_abs_deviation is None for run in audit.resimulated)
    assert all(run.committed_status == run.rebuilt_status for run in audit.resimulated)
    assert [run.arm for run in audit.resimulated].count("replay") == 60
    assert audit.tolerances.arrays_bitwise is True
    assert (audit.tolerances.metric_abs_tol, audit.tolerances.metric_rel_tol) == (0.0, 0.0)
    assert audit.tolerances.aggregates_exact is True


def test_the_audit_ran_from_a_clean_checkout_of_committed_code() -> None:
    """An audit of a modified checkout would audit code that is not the committed evidence's."""
    audit = load_audit(RECORD)
    assert audit.checkout_dirty is False
    assert audit.provenance.project_dirty is False
    assert audit.provenance.exploratory is False
    assert audit.results_commit == load_results(DOCS / "results" / "results_v1.json").provenance.project_commit
    assert audit.validated_commits, "the record names the commits whose checks it applies"


def test_the_handed_off_bundle_still_matches_the_repository() -> None:
    """Every repository file the bundle cites is still the one the audit digested."""
    bundle = load_audit(RECORD).bundle
    local = [item for item in bundle if not item.location.startswith("armrc://")]
    assert len(local) >= 15
    for item in local:
        path = REPO_ROOT / item.location
        assert path.is_file(), item.location
        assert (sha256_file(path), path.stat().st_size) == (item.sha256, item.size), item.location
    assert [item for item in bundle if item.location.startswith("armrc://")], "the stored tables are handed off too"


def test_the_rendering_is_generated_from_the_record() -> None:
    """The Markdown is regenerated, never edited beside the record it reports."""
    assert render_audit_markdown(load_audit(RECORD)) == MARKDOWN.read_text(encoding="utf-8")
