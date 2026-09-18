# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the committed timing evidence of the manual-demonstration study, pinned.

Three reports are committed and each answers a different question. The measured
serial check is the measurement of record: the authorized nominal subset, run
once, serially. Its projection is wrong -- it scaled the scenarios the
invocation had been restricted to rather than the locked protocol it was
estimating -- so the corrected derivative recomputes that one figure from the
same measurement and names the report it came from by digest. The four-worker
benchmark is the authorized parallel measurement, whose projection was already
right because it ran after that defect was fixed.

What is pinned here is that the derivative changed nothing except what was
computed, and that both measurements are the sanctioned shape: 24 models over
six replay banks, one nominal scenario, two trackers, 48 RC and 12 replay runs.

The benchmark's pointers live in ``benchmark_evidence/``, not ``evidence/``.
A pointer's filename is built from the model label alone, so a later full-study
sweep would write the same twenty-four names with different protocol identities
and be refused by the overwrite guard. That refusal is correct and stays; what
was wrong was benchmark evidence occupying the directory the full study needs.

The committed pointers are the serial check's. Both stores hold evidence of the
same protocol -- the model evidence identities are equal, and every one of the
60 pairs agrees on its arrays digest -- but their run records are NOT identical
files: each invocation writes its own run under its own id, because a run's
recorded provenance carries the command that produced it (``manual_timing
smoke`` in the parent, ``manual_evaluation evaluate-model`` in a worker), its
timestamp, and its project commit. Numerical agreement is therefore asserted on
``arrays_sha256``, never on a run record's own digest, which is expected to
differ between two invocations of the same protocol.
"""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.manual_evaluation import load_manual_pointer, manual_pointer_name
from arm_rc_ctrl.experiments.manual_timing import load_timing, render_timing_markdown
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration"
MEASURED = DOCS / "timing_smoke_check_v1.json"
MEASURED_MD = DOCS / "timing_smoke_check_v1.md"
CORRECTED = DOCS / "timing_smoke_check_v1_corrected.json"
CORRECTED_MD = DOCS / "timing_smoke_check_v1_corrected.md"
PARALLEL = DOCS / "timing_smoke_check_workers4_v1.json"
PARALLEL_MD = DOCS / "timing_smoke_check_workers4_v1.md"
POINTERS = DOCS / "benchmark_evidence"
MANIFEST = DOCS / "study_manifest_v1.json"
EVALUATION = REPO_ROOT / "configs" / "evaluations" / "task_1a_manual_dev_v1.toml"

AUTHORIZED_RC = 48
"""24 models x one nominal scenario x two trackers."""
AUTHORIZED_REPLAY = 12
"""Six replay banks x one nominal scenario x two trackers."""


def _arms(report_path: object) -> tuple[int, int]:
    report = load_timing(report_path)  # type: ignore[arg-type]
    return (
        sum(1 for run in report.runs if run.arm == "rc"),
        sum(1 for run in report.runs if run.arm == "replay"),
    )


@pytest.mark.parametrize("path", [MEASURED, PARALLEL])
def test_both_measurements_are_the_authorized_shape(path: object) -> None:
    """Serial and parallel measured the same sanctioned subset, in the canonical environment."""
    report = load_timing(path)  # type: ignore[arg-type]
    assert report.study_manifest_sha256 == sha256_file(MANIFEST)
    assert report.evaluation_sha256 == sha256_file(EVALUATION)
    assert len(report.models) == 24
    assert report.replay_banks_built == 6
    assert len(report.runs) == report.runs_this_invocation == 60
    assert _arms(path) == (AUTHORIZED_RC, AUTHORIZED_REPLAY)
    assert all(model.runs > 0 for model in report.models), "a model with no runs would be an unmeasured sweep"
    assert report.projection.unmeasured_arms == (), "nothing may be projected at no measured cost"
    assert not report.provenance.project_dirty
    assert not report.provenance.exploratory
    assert report.execution.policy == "p-cores"
    assert report.wall_seconds > 0
    assert report.peak_rss_bytes > 0
    assert report.storage_bytes > 0


def test_the_serial_check_ran_serially_and_the_benchmark_under_four_workers() -> None:
    """The pair is what makes the elapsed-time comparison meaningful, so the counts are pinned."""
    assert load_timing(MEASURED).workers == 1
    assert load_timing(PARALLEL).workers == 4


def test_the_derivative_recomputes_the_projection_and_nothing_else() -> None:
    """Every measured figure is carried across unchanged; only what was calculated from them differs."""
    measured, corrected = load_timing(MEASURED), load_timing(CORRECTED)
    assert measured.schema_version == 1, "the original stays readable under the schema it was written in"
    assert corrected.schema_version == 2
    derivation = corrected.derivation
    assert derivation is not None
    assert derivation.derived_from_sha256 == sha256_file(MEASURED)
    assert not derivation.derivation_dirty, "committed evidence is derived from a clean worktree"
    assert corrected.runs == measured.runs
    assert corrected.models == measured.models
    assert corrected.wall_seconds == measured.wall_seconds
    assert corrected.storage_bytes == measured.storage_bytes
    assert corrected.peak_rss_bytes == measured.peak_rss_bytes
    assert corrected.execution == measured.execution
    assert corrected.provenance == measured.provenance


def test_the_corrected_projection_is_of_the_locked_study() -> None:
    """65 locked scenarios under two trackers, over every model and bank of the frozen study."""
    corrected = load_timing(CORRECTED)
    assert corrected.projection.pairs_per_model == 130
    assert corrected.projection.models == 186
    assert corrected.projection.replay_banks == 60
    assert corrected.projection.rc_runs == 24_180
    assert corrected.projection.replay_runs == 7_800
    assert corrected.projection.total_runs == 31_980
    assert load_timing(MEASURED).projection.total_runs == 492, "the defect the derivative corrects"


@pytest.mark.parametrize(
    ("path", "markdown"), [(MEASURED, MEASURED_MD), (CORRECTED, CORRECTED_MD), (PARALLEL, PARALLEL_MD)]
)
def test_every_rendering_matches_the_report_it_renders(path: object, markdown: object) -> None:
    """The committed Markdown is generated, never edited by hand."""
    assert render_timing_markdown(load_timing(path)) == markdown.read_text(encoding="utf-8")  # type: ignore[attr-defined,arg-type]


def test_a_pointer_is_committed_for_every_model_and_replay_bank() -> None:
    """The measurement is checkable: each manifest the serial check produced has its Git pointer here."""
    report = load_timing(MEASURED)
    for label in report.entries:
        pointer = load_manual_pointer(POINTERS / manual_pointer_name("model", label))
        assert pointer.kind == "model"
        assert pointer.n_pairs == 2, "one nominal scenario under two trackers"
        assert pointer.n_completed + pointer.n_infeasible == pointer.n_pairs
    banks = sorted(POINTERS.glob("replay__*.toml"))
    assert len(banks) == 6
    for path in banks:
        bank = load_manual_pointer(path)
        assert bank.kind == "replay"
        assert bank.status == "feasible"
        assert bank.n_pairs == 2
