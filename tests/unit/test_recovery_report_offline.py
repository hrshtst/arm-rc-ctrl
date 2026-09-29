# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""CI-001: the recovery report's input builder, figures, and CLI without the external store.

The committed evidence (study pointers, ablation, freeze, representative pairs,
dataset and run pointers, development protocol) is real; only the two store
reads are replaced: the reference samples by a synthetic move/dwell task and
each representative run by a synthetic trajectory on the run clock.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from arm_rc_ctrl.data.synthetic import synthetic_task_samples
from arm_rc_ctrl.experiments import recovery_report
from arm_rc_ctrl.experiments.perturbations import RobustnessScenario
from arm_rc_ctrl.experiments.recovery_report import (
    PLOT_FILES,
    ReportInputs,
    build_report_inputs,
    main,
    render_recovery_report,
    write_recovery_plots,
)
from arm_rc_ctrl.experiments.recovery_representative import REPRESENTATIVE_CLASSES
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arm_rc_ctrl.data.records import ArtifactRecord
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.run_record import LoadedRun, RunPointerRecord
    from arm_rc_ctrl.storage import StorageRoot

ROOT = repository_root()
DOCS = ROOT / "docs" / "experiments" / "task_1a_state_conditioned_recovery"
ACTIVATION_S = 0.25
HOLD_SAMPLES = 25
"""Samples before activation at the synthetic 10 ms period (0.25 s hold)."""


def _store() -> StorageRoot:
    return cast("StorageRoot", SimpleNamespace())


def _run(pointer: RunPointerRecord, reference: SampleSet) -> LoadedRun:
    """A run holding its start for 0.25 s, then following the reference with a small offset."""
    rc = pointer.method.startswith("rc")
    n = HOLD_SAMPLES + reference.n_samples
    t = np.arange(n, dtype=np.float64) * 0.01
    q = np.vstack((np.repeat(reference.q[:1], HOLD_SAMPLES, axis=0), reference.q)) + (0.02 if rc else 0.01)
    generated = q.copy()
    generated[:HOLD_SAMPLES] = np.nan
    arrays: dict[str, object] = {
        "t": t,
        "q": q,
        "generator_output_q": generated,
        "saturation": np.r_[np.zeros(n - 10, dtype=np.int64), np.ones(10, dtype=np.int64)],
        # RC runs record the applied torque; replay runs only the request, exercising both sources.
        ("tau_applied" if rc else "tau_requested"): np.full((n, reference.dof), 2.0 if rc else 1.0),
    }
    return cast("LoadedRun", SimpleNamespace(pointer=pointer, arrays=SimpleNamespace(arrays=arrays)))


def _offline(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the store reads; returns the run IDs in the order they are loaded."""
    reference = synthetic_task_samples()
    loaded: list[str] = []

    def verify_payload(_store: StorageRoot, artifact: ArtifactRecord) -> Path:
        return Path(artifact.artifact_id)

    def load_samples(_path: Path) -> SampleSet:
        return reference

    def load_run(_store: StorageRoot, pointer: RunPointerRecord) -> LoadedRun:
        loaded.append(pointer.artifact.artifact_id)
        return _run(pointer, reference)

    monkeypatch.setattr(recovery_report, "verify_payload", verify_payload)
    monkeypatch.setattr(recovery_report, "load_samples", load_samples)
    monkeypatch.setattr(recovery_report, "load_run", load_run)
    return loaded


@pytest.fixture
def inputs(monkeypatch: pytest.MonkeyPatch) -> ReportInputs:
    """Report inputs built from the committed evidence with synthetic store reads."""
    _offline(monkeypatch)
    return build_report_inputs(DOCS, store=_store(), records_root=ROOT)


def test_inputs_bind_the_committed_evidence_and_derive_run_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every representative run is loaded through its tracked pointer and its effort/RMSE derived."""
    loaded = _offline(monkeypatch)
    built = build_report_inputs(DOCS, store=_store(), records_root=ROOT)
    assert sorted(built.pointers) == [
        "recovery_search_contractive_v1.toml",
        "recovery_search_no_augmentation_v1.toml",
        "recovery_search_non_decaying_v1.toml",
        "residual_search_1a_v1.toml",
    ]
    pairs = built.representative.pairs
    assert loaded == [run for pair in pairs for run in (pair.replay_run, pair.rc_run)]
    assert set(built.runs) == set(loaded)
    for pair in pairs:
        # Constant torque: RMS equals peak equals the recorded level of the source channel.
        assert built.effort[pair.rc_run] == pytest.approx(2.0)
        assert built.torque_peak[pair.rc_run] == pytest.approx(2.0)
        assert built.effort[pair.replay_run] == pytest.approx(1.0)
        assert built.saturation[pair.rc_run] == pytest.approx(10 / (HOLD_SAMPLES + 101))
        # The active segment is the reference plus the constant offset in both joints.
        assert built.move_rmse[pair.rc_run] == pytest.approx(0.02)
        assert built.move_rmse[pair.replay_run] == pytest.approx(0.01)
    assert set(built.representative.scenarios.values()) <= set(built.scenarios)


def test_inputs_refuse_representative_scenarios_absent_from_the_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """A representative scenario the development protocol does not define is refused."""
    _offline(monkeypatch)

    def no_scenarios(*_args: object, **_kwargs: object) -> tuple[RobustnessScenario, ...]:
        return ()

    monkeypatch.setattr(recovery_report, "robustness_scenarios", no_scenarios)
    with pytest.raises(ValueError, match=r"representative scenarios are absent from .*task_1a_recovery_dev_v1\.toml"):
        build_report_inputs(DOCS, store=_store(), records_root=ROOT)


def test_plots_follow_the_declared_files_and_refuse_overwriting_pairs(inputs: ReportInputs, tmp_path: Path) -> None:
    """All seven figures are written in PLOT_FILES order; pair figures are write-once."""
    written = write_recovery_plots(inputs, tmp_path)
    assert written == list(PLOT_FILES)
    for name in PLOT_FILES:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob("*.tmp.png"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_recovery_plots(inputs, tmp_path)


def test_report_renders_recovery_metrics_and_every_animation_layout(inputs: ReportInputs) -> None:
    """Metric tables, scenario descriptions, paired/single clips and extra clips are all rendered."""
    by_class = {pair.kind: pair for pair in inputs.representative.pairs if pair.tracker == "pd_v2"}
    nominal = by_class["nominal"]
    assert nominal.recovery is not None
    markdown = render_recovery_report(
        inputs,
        plots=("cell_gap_medians.png",),
        animations=(
            "nominal_rc_pd.gif",
            "nominal_replay_pd.gif",
            "posture_small_rc_pd.gif",
            "posture_large_replay_pd.gif",
            "force_rc_pd.gif",
            "force_replay_pd.gif",
            "extra_view.gif",
        ),
    )
    assert f"| {inputs.effort[nominal.rc_run]:.4g} |" in markdown
    assert (
        f"| nominal | pd_v2 | {inputs.move_rmse[nominal.rc_run]:.4g} | {inputs.move_rmse[nominal.replay_run]:.4g} "
        in (markdown)
    )
    assert f"| {nominal.recovery.activation_jump_rad:.4g} |" in markdown
    assert "no posture offset or external force is applied" in markdown
    assert "Both arms hold this perturbed posture" in markdown
    assert "end-effector pulse acts toward +x" in markdown
    assert "Both simulations completed. RC/replay actual-motion RMSE" in markdown
    small, large = by_class["posture_small"], by_class["posture_large"]
    assert f"`{small.rc_run}` — actual motion commanded by the feedback-conditioned ESN:" in markdown
    assert f"`{large.replay_run}` — actual motion commanded by the original teacher trajectory:" in markdown
    assert "Additional animation `extra_view.gif`:" in markdown
    assert "Additional animation `nominal_rc_pd.gif`" not in markdown
    assert "(exploratory, D4): 0 of" in markdown
    assert markdown.endswith("![extra_view](animations/extra_view.gif)\n")


def test_report_describes_unknown_and_other_scenario_classes(inputs: ReportInputs) -> None:
    """A scenario absent from the protocol falls back to its ID; a class without a setup text is named."""
    by_class = {pair.kind: pair for pair in inputs.representative.pairs if pair.tracker == "pd_v2"}
    force, large = by_class["force"], by_class["posture_large"]
    scenarios = {k: v for k, v in inputs.scenarios.items() if k != force.scenario_id}
    scenarios[large.scenario_id] = RobustnessScenario(large.scenario_id, "combined", (0.0, 0.0))
    markdown = render_recovery_report(
        dataclasses.replace(inputs, scenarios=scenarios),
        animations=("force_rc_pd.gif", "posture_large_rc_pd.gif"),
    )
    assert f"Scenario `{force.scenario_id}`; consult the representative table" in markdown
    assert f"Scenario `{large.scenario_id}` belongs to class `combined`." in markdown
    assert "Small initial-posture perturbation" not in markdown
    assert "Additional animation" not in markdown


def test_report_without_residual_plots_or_animations_ends_at_the_limitations(inputs: ReportInputs) -> None:
    """Optional sections are omitted, not rendered empty."""
    pointers = {k: v for k, v in inputs.pointers.items() if k != "residual_search_1a_v1.toml"}
    markdown = render_recovery_report(dataclasses.replace(inputs, pointers=pointers))
    assert "(exploratory, D4)" not in markdown
    assert "## Plots" not in markdown
    assert "## Animations" not in markdown
    assert markdown.rstrip("\n").splitlines()[-2].startswith("- ")


def test_command_line_writes_report_and_plots_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI renders the report, writes the figures, prints them, and never overwrites a report."""
    _offline(monkeypatch)
    monkeypatch.setattr(recovery_report, "open_storage", _store)
    output = tmp_path / "report.md"
    plots_dir = tmp_path / "plots"
    argv = ["--docs", str(DOCS), "--output", str(output), "--plots-dir", str(plots_dir)]
    assert main([*argv, "--records-root", str(ROOT), "--animations", "nominal_rc_pd.gif"]) == 0
    printed = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert printed == {"output": str(output), "plots": list(PLOT_FILES)}
    markdown = output.read_text(encoding="utf-8")
    assert markdown.startswith("# Task 1-a state-conditioned recovery: development results (v1)\n")
    for name in PLOT_FILES:
        assert f"plots/recovery_report_v1/{name}" in markdown
    assert "animations/nominal_rc_pd.gif" in markdown
    assert sorted(p.name for p in plots_dir.iterdir()) == sorted(PLOT_FILES)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)


def test_every_representative_class_has_a_primary_pair(inputs: ReportInputs) -> None:
    """The figure loop needs a PD v2 pair per class; the committed record supplies all of them."""
    assert {p.kind for p in inputs.representative.pairs if p.tracker == "pd_v2"} == set(REPRESENTATIVE_CLASSES)


@pytest.mark.parametrize("t", [np.zeros(1), np.array([0.0, np.nan]), np.zeros((2, 1))])
def test_pair_plot_refuses_an_invalid_run_clock(t: NDArray[np.float64], tmp_path: Path) -> None:
    """A clock that is not a finite 1-D array of at least two samples is refused before plotting."""
    q = np.zeros((2, 2))
    with pytest.raises(ValueError, match="t must be a finite 1-D array with at least two samples"):
        recovery_report.plot_recovery_pair(t, q, q, q, q, tmp_path / "p.png", title="t", boundaries=(), xlabel="x")
    assert not list(tmp_path.iterdir())
