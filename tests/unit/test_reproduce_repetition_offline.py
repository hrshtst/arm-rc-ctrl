# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""CI-001: the pilot reproduction's steps checked against the committed records without the evidence store.

The store-bound readers (payload digests, the study payload, the refits, the
figure renderers) are replaced by fakes at the module seams; every comparison
the steps make against the committed records runs for real.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from arm_rc_ctrl.execution import load_execution
from arm_rc_ctrl.experiments import reproduce_repetition as rr
from arm_rc_ctrl.experiments.repetition_accounting import account_pilot, load_accounting
from arm_rc_ctrl.experiments.repetition_diagnosis import load_diagnosis
from arm_rc_ctrl.experiments.repetition_evaluation import ModelEvidence, load_pointer, numerical_binding
from arm_rc_ctrl.experiments.repetition_fixture import (
    DOCS,
    PLANAR_DIGESTS,
    PLANAR_SCENARIOS,
    PLANAR_TRACKER,
    CraftedSimulator,
    PlanarFixture,
    build_pilot_runner,
    exploratory_provenance,
)
from arm_rc_ctrl.experiments.repetition_numerics import PanelContext, load_validation
from arm_rc_ctrl.experiments.repetition_panel import load_panel
from arm_rc_ctrl.experiments.repetition_recipes import ArmSpec, panel_arms
from arm_rc_ctrl.experiments.repetition_report import ANIMATION_DIR, PLOT_DIR, ReportInputs, load_report
from arm_rc_ctrl.experiments.repetition_timing import load_timing
from arm_rc_ctrl.experiments.reproduce_1a import Check, ReproductionError
from arm_rc_ctrl.experiments.reproduce_repetition import (
    DECLARED_COMPARISONS,
    RESIMULATION_RULE,
    STEPS,
    RepetitionReproduction,
    Reproducer,
    main,
    prescribed_fits,
    reproduce,
    run_from_checkout,
)
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from collections.abc import Callable

    from arm_rc_ctrl.experiments.repetition_panel import PanelManifest
    from arm_rc_ctrl.experiments.repetition_report import RepetitionReport, Representative

MODULE = "arm_rc_ctrl.experiments.reproduce_repetition"
REPORT = DOCS / "repetition_report_v1.json"
MANIFEST = DOCS / "panel_manifest_v1.json"
VALIDATION = DOCS / "numerical_validation_v1.json"
CRAFTED_ABORT = (49.0, 49.0)
TIGHT_ABORT = (0.05, 0.05)
DEVELOPMENT = "task_1a_recovery_dev_v1.toml"


def _returns(value: object) -> Callable[..., object]:
    """A stand-in that ignores its arguments and returns ``value``."""

    def stub(*_args: object, **_kwargs: object) -> object:
        return value

    return stub


def _bare(tmp_path: Path, *, docs: Path = DOCS) -> Reproducer:
    """A reproducer over ``docs`` with a throwaway store (never the configured external root)."""
    (tmp_path / "store-root").mkdir(parents=True, exist_ok=True)
    reproducer = Reproducer(tmp_path / "scratch", False, None, docs, None)  # noqa: FBT003 - positional dataclass field
    reproducer.store = StorageRoot(tmp_path / "store-root", repositories=(rr.REPO,))
    return reproducer


def _record_panel(rebuilt: list[str]) -> Callable[..., None]:
    def rebuild(_self: Reproducer, manifest: PanelManifest, _store: StorageRoot) -> None:
        rebuilt.append(manifest.experiment)

    return rebuild


def _copy_docs(tmp_path: Path) -> Path:
    """A private copy of the committed pilot records to tamper with."""
    docs = tmp_path / "docs"
    shutil.copytree(DOCS, docs)
    return docs


# --- environment and storage ---------------------------------------------------------------


def _environment_seams(monkeypatch: pytest.MonkeyPatch, report: RepetitionReport, *, identity: str) -> None:
    """Pins, builds, lock, and launcher as the evidence recorded them; the process reports ``identity``."""
    monkeypatch.setattr(f"{MODULE}.verify_builds", _returns(["skelarm", "rclib"]))
    monkeypatch.setattr(f"{MODULE}.submodule_revisions", _returns(list(report.provenance.submodules)))
    monkeypatch.setattr(f"{MODULE}.sha256_file", _returns(report.provenance.lock_sha256))
    monkeypatch.setattr(f"{MODULE}.require_canonical", lambda: None)
    monkeypatch.setattr(
        f"{MODULE}.collect_execution",
        _returns(SimpleNamespace(identity=identity, check_canonical=lambda: None)),
    )


def test_environment_binds_pins_lock_and_the_canonical_execution_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the evidence's pins and lock, only the canonical execution identity passes; any other is named."""
    report = load_report(REPORT)
    canonical = load_execution(DOCS / "execution_environment_v1.json")
    _environment_seams(monkeypatch, report, identity=canonical.identity)
    reproducer = _bare(tmp_path / "ok")
    detail = reproducer.environment()
    assert "2 build identities verified" in detail
    assert canonical.identity[:12] in detail
    assert reproducer.inputs["canonical_execution_identity"] == canonical.identity
    assert reproducer.inputs["reproduction_execution_identity"] == canonical.identity
    assert reproducer.inputs["evidence_project_commit"] == report.provenance.project_commit
    assert reproducer.report == report
    assert reproducer.execution == canonical

    _environment_seams(monkeypatch, report, identity="f" * 64)
    other = _bare(tmp_path / "other")
    with pytest.raises(ReproductionError, match=r"execution environment ffffffffffff, not the canonical"):
        other.environment()
    assert other.inputs["reproduction_execution_identity"] == "f" * 64


def test_environment_refuses_other_pins_another_lock_or_an_unbound_execution_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each binding the environment step checks fails with its own message before anything is launched."""
    report = load_report(REPORT)
    canonical = load_execution(DOCS / "execution_environment_v1.json")
    _environment_seams(monkeypatch, report, identity=canonical.identity)
    monkeypatch.setattr(f"{MODULE}.submodule_revisions", _returns([]))
    with pytest.raises(ReproductionError, match="submodule pins differ"):
        _bare(tmp_path / "pins").environment()

    _environment_seams(monkeypatch, report, identity=canonical.identity)
    monkeypatch.setattr(f"{MODULE}.sha256_file", _returns("0" * 64))
    with pytest.raises(ReproductionError, match=r"uv\.lock digest 000000000000 differs"):
        _bare(tmp_path / "lock").environment()

    _environment_seams(monkeypatch, report, identity=canonical.identity)
    monkeypatch.setattr(f"{MODULE}.load_execution", _returns(SimpleNamespace(identity="e" * 64)))
    with pytest.raises(ReproductionError, match="canonical execution identity is not the committed"):
        _bare(tmp_path / "record").environment()


def test_storage_creates_a_scratch_store_under_the_scratch_directory(tmp_path: Path) -> None:
    """The configured store is used as given and the scratch store lives in the scratch directory."""
    configured = StorageRoot(tmp_path, repositories=(rr.REPO,))
    (tmp_path / "scratch").mkdir()
    reproducer = Reproducer(tmp_path / "scratch", False, configured, DOCS, None)  # noqa: FBT003
    assert reproducer.storage() == "external storage root resolved; scratch store created"
    assert reproducer.store is configured
    assert reproducer.scratch_store is not None
    assert (tmp_path / "scratch" / "store").is_dir()
    assert reproducer.inputs["storage_root"] == "<configured external root>"


# --- records ------------------------------------------------------------------------------


def test_records_verify_the_committed_bindings_and_pointers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed report sources, C11 binding, incident record, and evidence pointers all verify."""
    rebuilt: list[str] = []
    monkeypatch.setattr(Reproducer, "_rebuild_panel", _record_panel(rebuilt))
    reproducer = _bare(tmp_path)
    reproducer.report = load_report(REPORT)
    detail = reproducer.records()
    accounting = load_accounting(DOCS / "pilot_execution_v1.json")
    pointers = len(accounting.models) + len(accounting.banks)
    assert f"{pointers} evidence pointers" in detail
    assert f"{len(reproducer.report.sources)} report sources" in detail
    assert rebuilt == [load_panel(MANIFEST).experiment]
    assert reproducer.inputs["panel_manifest_sha256"] == sha256_file(MANIFEST)
    assert reproducer.inputs["numerical_validation_sha256"] == sha256_file(VALIDATION)
    assert reproducer.inputs["retention_incident_sha256"] == sha256_file(DOCS / rr.RETENTION_INCIDENT)
    candidate = reproducer.inputs["c11_exception_candidate"]
    assert candidate[:12] in detail
    failed = [c for c in load_validation(VALIDATION).comparisons if not c.passed]
    assert [c.candidate_identity for c in failed] == [candidate]


def test_records_refuse_every_broken_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed source, an unbound validation, accounting, diagnosis, or timing record is refused by name."""
    report = load_report(REPORT)
    validation = load_validation(VALIDATION)
    accounting = load_accounting(DOCS / "pilot_execution_v1.json")
    name = next(iter(report.sources))
    tampered = replace(report, sources={**report.sources, name: "0" * 64})
    with pytest.raises(ReproductionError, match=f"report source {name} has digest"):
        _bare(tmp_path / "source")._check_bindings(tampered, validation, accounting)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    unbound = replace(validation, panel_manifest_sha256="0" * 64)
    with pytest.raises(ReproductionError, match="does not bind the committed panel manifest"):
        _bare(tmp_path / "validation")._check_bindings(report, unbound, accounting)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    other = replace(accounting, numerical_validation_sha256="0" * 64)
    with pytest.raises(ReproductionError, match="does not bind the committed numerical validation"):
        _bare(tmp_path / "accounting")._check_bindings(report, validation, other)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    diagnosis = load_diagnosis(DOCS / "numerical_validation_v1_diagnosis.json")
    monkeypatch.setattr(f"{MODULE}.load_diagnosis", _returns(replace(diagnosis, validation_sha256="0" * 64)))
    with pytest.raises(ReproductionError, match="does not bind the committed numerical validation"):
        _bare(tmp_path / "diagnosis")._check_bindings(report, validation, accounting)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    monkeypatch.undo()
    timing = load_timing(DOCS / "timing_smoke_check_v1.json")
    monkeypatch.setattr(f"{MODULE}.load_timing", _returns(replace(timing, panel_manifest_sha256="0" * 64)))
    with pytest.raises(ReproductionError, match="timing smoke check does not bind"):
        _bare(tmp_path / "timing")._check_bindings(report, validation, accounting)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_records_require_the_single_c11_exception_and_the_incident_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The C11 exception is exactly the validation's one retained failure, and the incident record must exist."""
    report = load_report(REPORT)
    validation = load_validation(VALIDATION)
    no_exception = replace(report, equivalence=tuple(e for e in report.equivalence if not e.accepted_exception))
    with pytest.raises(ReproductionError, match="exactly one failed comparison"):
        _bare(tmp_path / "c11")._check_c11_and_incident(no_exception, validation)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    binding = numerical_binding(VALIDATION, root=rr.REPO)
    monkeypatch.setattr(
        f"{MODULE}.numerical_binding",
        _returns(replace(binding, exception_candidate_identity="c" * 64)),
    )
    with pytest.raises(ReproductionError, match="not the C11-bound comparison"):
        _bare(tmp_path / "retained")._check_c11_and_incident(report, validation)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    monkeypatch.undo()
    docs = _copy_docs(tmp_path)
    (docs / rr.RETENTION_INCIDENT).unlink()
    with pytest.raises(ReproductionError, match=r"retention incident record .* is missing"):
        _bare(tmp_path / "incident", docs=docs)._check_c11_and_incident(report, validation)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_records_refuse_a_pointer_that_disagrees_with_the_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model and replay-bank pointers must name the accounting's identities and payloads."""
    accounting = load_accounting(DOCS / "pilot_execution_v1.json")
    assert _bare(tmp_path / "ok")._check_pointers(accounting) == len(accounting.models) + len(accounting.banks)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def tampered(kind: str) -> Callable[[Path], object]:
        def load(path: Path) -> object:
            pointer = load_pointer(path)
            return (
                replace(pointer, payload=replace(pointer.payload, sha256="0" * 64))
                if path.name.startswith(kind)
                else pointer
            )

        return load

    monkeypatch.setattr(f"{MODULE}.load_pointer", tampered("model"))
    with pytest.raises(ReproductionError, match=r"the pointer of .* disagrees with the accounting"):
        _bare(tmp_path / "model")._check_pointers(accounting)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(f"{MODULE}.load_pointer", tampered("replay"))
    with pytest.raises(ReproductionError, match=r"replay bank pointer at warm-up .* disagrees"):
        _bare(tmp_path / "bank")._check_pointers(accounting)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def _panel_seams(monkeypatch: pytest.MonkeyPatch, rebuilt: object) -> list[str]:
    """Fake the study payload behind the panel; the committed dataset record still loads for real."""
    calls: list[str] = []

    def pointer(path: Path) -> str:
        calls.append(path.name)
        return "pointer"

    monkeypatch.setattr(f"{MODULE}.load_report_pointer", pointer)
    monkeypatch.setattr(f"{MODULE}.open_stored_report", _returns(SimpleNamespace(protocol_file="p")))
    monkeypatch.setattr(f"{MODULE}.load_recovery_search", _returns("protocol"))
    monkeypatch.setattr(f"{MODULE}.load_ablation", _returns("ablation"))
    monkeypatch.setattr(f"{MODULE}.resolve_panel", _returns("entries"))
    monkeypatch.setattr(f"{MODULE}.build_panel_manifest", _returns(rebuilt))
    return calls


def test_panel_must_re_resolve_identically_from_the_study_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identical re-resolution records the source payload; a different one or a non-recovery dataset fails."""
    manifest = load_panel(MANIFEST)
    calls = _panel_seams(monkeypatch, manifest)
    reproducer = _bare(tmp_path / "ok")
    reproducer._rebuild_panel(manifest, cast("StorageRoot", reproducer.store))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert reproducer.inputs["source_study_payload_sha256"] == manifest.source.payload.sha256
    assert calls == [Path(manifest.source.pointer_file).name]

    _panel_seams(monkeypatch, replace(manifest, provenance=exploratory_provenance()))
    with pytest.raises(ReproductionError, match="re-resolved from the study payload differs"):
        _bare(tmp_path / "differs")._rebuild_panel(manifest, cast("StorageRoot", reproducer.store))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(f"{MODULE}.load_processed_record", _returns(SimpleNamespace()))
    with pytest.raises(ReproductionError, match="is not a recovery dataset record"):
        _bare(tmp_path / "dataset")._rebuild_panel(manifest, cast("StorageRoot", reproducer.store))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_payloads_step_reports_what_it_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The step verifies runs and the fit cache through the store and counts the augmented fits apart."""
    context = SimpleNamespace(manifest=SimpleNamespace(configs=SimpleNamespace(dataset="processed-x")))
    inputs = SimpleNamespace(models={"a": 1, "b": 2}, banks={0.5: 3})
    prescribed = {"v": SimpleNamespace(source="validation"), "e": SimpleNamespace(source="evidence")}
    monkeypatch.setattr(f"{MODULE}.PanelContext", SimpleNamespace(load=_returns(context)))
    monkeypatch.setattr(f"{MODULE}.build_report_inputs", _returns(inputs))
    monkeypatch.setattr(f"{MODULE}.verify_run_payloads", _returns((7, 3 * 2**20)))
    monkeypatch.setattr(f"{MODULE}.prescribed_fits", _returns(prescribed))
    monkeypatch.setattr(f"{MODULE}.verify_fit_cache", _returns(len(prescribed)))
    reproducer = _bare(tmp_path)
    with pytest.raises(ReproductionError, match="requires an earlier step"):
        reproducer.payloads()
    reproducer.current_execution = cast("Any", SimpleNamespace(identity="i"))
    detail = reproducer.payloads()
    fits = len(load_validation(VALIDATION).fits)
    assert detail.startswith("dataset processed-x, 2 model manifests, 1 replay banks, 7 run payloads (3.0 MiB")
    assert f"and 2 cached fits ({fits} of the validation, 1 augmented fits bound by the evidence)" in detail
    assert reproducer.inputs["dataset"] == "processed-x"
    assert reproducer.prescribed == prescribed


# --- tables and assets ----------------------------------------------------------------------


def _report_seams(monkeypatch: pytest.MonkeyPatch, report: RepetitionReport) -> None:
    """The store-derived report and accounting equal the committed ones; the renderings run for real."""
    monkeypatch.setattr(f"{MODULE}.representatives", _returns(list(report.representatives)))
    monkeypatch.setattr(f"{MODULE}.build_report", _returns(report))
    committed = load_accounting(DOCS / "pilot_execution_v1.json")
    monkeypatch.setattr(f"{MODULE}.account_pilot", _returns(committed))


def _tables_reproducer(tmp_path: Path, docs: Path = DOCS) -> Reproducer:
    reproducer = _bare(tmp_path, docs=docs)
    reproducer.report = load_report(REPORT)
    reproducer.report_inputs = cast("ReportInputs", SimpleNamespace())
    return reproducer


def test_tables_re_render_every_committed_markdown_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the report and accounting re-derived, every committed Markdown rendering is reproduced exactly."""
    report = load_report(REPORT)
    _report_seams(monkeypatch, report)
    reproducer = _tables_reproducer(tmp_path)
    detail = reproducer.tables()
    assert f"{len(report.outcomes)} outcome, {len(report.paired)} paired" in detail
    assert f"{len(report.representatives)} representatives, the accounting, and 6 Markdown renderings" in detail
    assert reproducer.reps == report.representatives


def test_tables_refuse_a_changed_table_accounting_or_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Any re-derived difference names the record it was found in."""
    report = load_report(REPORT)
    _report_seams(monkeypatch, report)
    fewer = replace(report, animation_rule=report.animation_rule + " (edited)")
    monkeypatch.setattr(f"{MODULE}.build_report", _returns(fewer))
    with pytest.raises(ReproductionError, match="report tables: 1 categorical differences"):
        _tables_reproducer(tmp_path / "tables").tables()

    _report_seams(monkeypatch, report)
    moved = [replace(r, warmup_s=r.warmup_s + 1.0) for r in report.representatives]
    monkeypatch.setattr(f"{MODULE}.representatives", _returns(moved))
    with pytest.raises(ReproductionError, match="representatives"):
        _tables_reproducer(tmp_path / "reps").tables()

    _report_seams(monkeypatch, report)
    committed = load_accounting(DOCS / "pilot_execution_v1.json")
    assert account_pilot is not rr.account_pilot  # the seam is patched, the real function is untouched
    monkeypatch.setattr(f"{MODULE}.account_pilot", _returns(replace(committed, provenance=exploratory_provenance())))
    with pytest.raises(ReproductionError, match="accounting"):
        _tables_reproducer(tmp_path / "accounting").tables()

    _report_seams(monkeypatch, report)
    docs = _copy_docs(tmp_path)
    (docs / "repetition_report_v1.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(ReproductionError, match=r"differs from the committed repetition_report_v1\.md"):
        _tables_reproducer(tmp_path / "report-md", docs).tables()
    shutil.copyfile(DOCS / "repetition_report_v1.md", docs / "repetition_report_v1.md")
    (docs / "timing_smoke_check_v1.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(ReproductionError, match=r"rendered timing_smoke_check_v1\.md differs"):
        _tables_reproducer(tmp_path / "timing-md", docs).tables()


def _asset_seams(monkeypatch: pytest.MonkeyPatch, report: RepetitionReport, *, corrupt: str | None = None) -> None:
    """Renderers that write the committed bytes (one file optionally corrupted)."""

    def copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data + b"\0" if target.name == corrupt else data)

    def plots(_inputs: object, reps: list[Representative], out: Path) -> tuple[list[str], list[Representative]]:
        for name in report.plots:
            copy(DOCS / PLOT_DIR / name, out / name)
        return list(report.plots), reps

    def animations(_inputs: object, reps: list[Representative], out: Path) -> list[Representative]:
        for name in report.animations:
            copy(DOCS / ANIMATION_DIR / name, out / name)
        return reps

    monkeypatch.setattr(f"{MODULE}.representatives", _returns(list(report.representatives)))
    monkeypatch.setattr(f"{MODULE}.write_plots", plots)
    monkeypatch.setattr(f"{MODULE}.write_animations", animations)
    monkeypatch.setattr(f"{MODULE}.build_report", _returns(report))


def test_assets_compare_every_regenerated_file_byte_for_byte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical bytes pass; a single changed byte in a figure or an animation is named."""
    report = load_report(REPORT)
    assert report.plots
    assert report.animations
    _asset_seams(monkeypatch, report)
    reproducer = _tables_reproducer(tmp_path / "ok")
    reproducer.reps = report.representatives
    assert reproducer.assets() == (
        f"{len(report.plots)} figures and {len(report.animations)} animations regenerated byte-for-byte; "
        "the report re-assembles"
    )
    _asset_seams(monkeypatch, report, corrupt=report.plots[0])
    with pytest.raises(ReproductionError, match=f"figure {report.plots[0]} does not regenerate"):
        _tables_reproducer(tmp_path / "plot").assets()
    _asset_seams(monkeypatch, report, corrupt=report.animations[-1])
    reproducer = _tables_reproducer(tmp_path / "animation")
    reproducer.reps = report.representatives
    with pytest.raises(ReproductionError, match=f"animation {report.animations[-1]} does not regenerate"):
        reproducer.assets()


def test_assets_refuse_a_different_set_of_figures_or_animations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regenerated inventory must equal the committed one, and so must the re-assembled report."""
    report = load_report(REPORT)
    _asset_seams(monkeypatch, report)
    monkeypatch.setattr(f"{MODULE}.write_plots", _returns((list(report.plots[1:]), list(report.representatives))))
    with pytest.raises(ReproductionError, match=r"regenerated figures .* differ from the committed set"):
        _tables_reproducer(tmp_path / "plots").assets()

    _asset_seams(monkeypatch, report)
    no_animation = [replace(r, rc_animation=None, replay_animation=None) for r in report.representatives]
    monkeypatch.setattr(f"{MODULE}.write_animations", _returns(no_animation))
    with pytest.raises(ReproductionError, match="regenerated animations \\[\\] differ"):
        _tables_reproducer(tmp_path / "animations").assets()

    _asset_seams(monkeypatch, report)
    monkeypatch.setattr(f"{MODULE}.build_report", _returns(replace(report, animation_rule="edited")))
    with pytest.raises(ReproductionError, match="report: 1 categorical differences"):
        _tables_reproducer(tmp_path / "report").assets()


# --- steps over the planar fixture ------------------------------------------------------------


@pytest.fixture(scope="module")
def crafted(fixture: PlanarFixture) -> ReportInputs:
    """Crafted sweeps of every behavioral arm of the fixture entry (no replay bank evidence on disk)."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=CRAFTED_ABORT, simulate_fn=CraftedSimulator(f.samples))
    models = {f"{f.entry.label}/{arm.label}": runner.evaluate(f.entry, arm) for arm in panel_arms() if arm.behavioral}
    return cast("ReportInputs", SimpleNamespace(models=models, banks={f.entry.warmup_s: runner.replay_bank(f.entry)}))


def _planar(f: PlanarFixture, inputs: ReportInputs, scratch: Path, *, exploratory: bool = False) -> Reproducer:
    reproducer = Reproducer(
        scratch,
        exploratory,
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


def _evidence_fits(crafted: ReportInputs) -> dict[str, rr.PrescribedFit]:
    validation = load_validation(VALIDATION)
    return {k: v for k, v in prescribed_fits(validation, crafted).items() if v.source == "evidence"}


def test_recipes_re_derive_every_prescribed_fit_from_the_cache(
    fixture: PlanarFixture, crafted: ReportInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each arm's identity, ridge rule, training construction, and reservoir re-derive from the cached recipe.

    The fixture fits only the behavioral arms, so the panel is narrowed to them.
    """
    f = fixture
    monkeypatch.setattr(f"{MODULE}.panel_arms", lambda: tuple(a for a in panel_arms() if a.behavioral))
    prescribed = _evidence_fits(crafted)
    reproducer = _planar(f, crafted, tmp_path / "ok")
    reproducer.context = cast(
        "PanelContext", SimpleNamespace(manifest=SimpleNamespace(entries=(f.entry,)), inputs=f.inputs)
    )
    reproducer.prescribed = prescribed
    detail = reproducer.recipes()
    assert detail.startswith(f"{len(prescribed)} prescribed fit identities, solver rules")
    assert f"(0 recorded by the validation, {len(prescribed)} augmented fits by the evidence)" in detail

    extra = next(iter(prescribed.values()))
    reproducer.prescribed = {**prescribed, "unexpected": replace(extra, identity="unexpected")}
    with pytest.raises(
        ReproductionError, match=f"{len(prescribed)} prescribed fits re-derived but the evidence records"
    ):
        reproducer.recipes()
    first_identity = next(iter(prescribed))
    reproducer.prescribed = {k: v for k, v in prescribed.items() if k != first_identity}
    with pytest.raises(ReproductionError, match="is not recorded by the evidence"):
        reproducer.recipes()
    other = replace(prescribed[first_identity], solver_alpha=prescribed[first_identity].solver_alpha * 2.0)
    reproducer.prescribed = {**prescribed, first_identity: other}
    with pytest.raises(ReproductionError, match="cached readout settings are not the arm's prescription"):
        reproducer.recipes()
    reproducer.prescribed = prescribed
    monkeypatch.setattr(f"{MODULE}.training_spec_for_arm", _returns(None))
    with pytest.raises(ReproductionError, match="cached recipe's training construction is not the arm's"):
        reproducer.recipes()


def test_metrics_context_refuses_drifted_configurations_scenarios_or_trackers(
    fixture: PlanarFixture, crafted: ReportInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scenarios, configurations, and trackers the replay banks bound must be the ones rebuilt now."""
    f = fixture
    ((warmup, bank),) = crafted.banks.items()
    assert warmup == f.entry.warmup_s
    assert bank.conditions.development_sha256 == sha256_file(f.root / "configs" / "evaluations" / DEVELOPMENT)
    with monkeypatch.context() as patched:
        patched.setattr(f"{MODULE}.sha256_file", _returns("0" * 64))
        with pytest.raises(ReproductionError, match="no longer hashes to what the replay banks bound"):
            _planar(f, crafted, tmp_path / "config")._metrics_context()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    fewer = _planar(f, crafted, tmp_path / "scenarios")
    fewer.scenarios = PLANAR_SCENARIOS[:-1]
    with pytest.raises(ReproductionError, match="development scenarios no longer resolve"):
        fewer._metrics_context()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    trackers = _planar(f, crafted, tmp_path / "trackers")
    trackers.tracker_digests = dict.fromkeys(PLANAR_DIGESTS, "0" * 64)
    with pytest.raises(ReproductionError, match="frozen trackers differ"):
        trackers._metrics_context()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    derived = _planar(f, crafted, tmp_path / "derived")
    derived.scenarios = None  # derived from the development levels, which the planar banks did not bind
    with pytest.raises(ReproductionError, match="development scenarios no longer resolve"):
        derived._metrics_context()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    ok = _planar(f, crafted, tmp_path / "ok")
    context = ok._metrics_context()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    # Built once, then shared by the metrics and the re-simulation.
    assert ok._metrics_context() is context  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ReproductionError, match="unknown scenario 'nope'"):
        context.case("nope")
    assert context.case(PLANAR_SCENARIOS[1].scenario_id) == (1, PLANAR_SCENARIOS[1])


def test_metrics_refuse_a_model_whose_replay_bank_is_not_committed(
    fixture: PlanarFixture, crafted: ReportInputs, tmp_path: Path
) -> None:
    """A model manifest naming a replay bank outside the evidence cannot be recomputed."""
    f = fixture
    label, evidence = next(iter(crafted.models.items()))
    orphan = cast("ReportInputs", SimpleNamespace(models={label: replace(evidence, replay_bank="b" * 64)}, banks={}))
    with pytest.raises(ReproductionError, match=f"{label}: its replay bank bbbbbbbbbbbb is not among"):
        _planar(f, orphan, tmp_path).metrics()


def test_the_scratch_runner_writes_to_the_scratch_store_under_the_rebuilt_conditions(
    fixture: PlanarFixture, crafted: ReportInputs, tmp_path: Path
) -> None:
    """The rebuilt runner targets the scratch store and reproduces the committed evaluation conditions."""
    f = fixture
    reproducer = _planar(f, crafted, tmp_path, exploratory=True)
    runner = reproducer._scratch_runner()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert runner.store is reproducer.scratch_store
    assert runner.root == f.root
    ((warmup, bank),) = crafted.banks.items()
    assert warmup == f.entry.warmup_s
    assert runner.replay_conditions(f.entry) == bank.conditions
    reproducer._check_conditions(runner, crafted, {f.entry.label: f.entry})  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    label, evidence = next(iter(crafted.models.items()))
    other = replace(evidence, conditions=replace(evidence.conditions, settling_band_rad=1.0))
    changed = cast("ReportInputs", SimpleNamespace(models={label: other}, banks={}))
    with pytest.raises(ReproductionError, match="rebuilt evaluation conditions are not the committed evidence's"):
        reproducer._check_conditions(runner, changed, {f.entry.label: f.entry})  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(scope="module")
def tight(fixture: PlanarFixture) -> ModelEvidence:
    """One real sweep under a tight abort: the first RC pair aborts infeasible and the rest stay unexecuted."""
    f = fixture
    runner = build_pilot_runner(f, velocity_abort=TIGHT_ABORT)
    runner.replay_bank(f.entry)
    evidence = runner.evaluate(f.entry, ArmSpec("absolute", "S"))
    assert evidence.status == "rc_gate_failure"
    return evidence


def test_a_rerun_must_carry_a_run_and_keep_the_stored_status(
    fixture: PlanarFixture, crafted: ReportInputs, tight: ModelEvidence, tmp_path: Path
) -> None:
    """A missing run on either side or a changed status fails; the stored pair compared with itself passes."""
    f = fixture
    reproducer = _planar(f, crafted, tmp_path)
    ((_, bank),) = crafted.banks.items()
    executed = next(p for p in bank.pairs if p.run is not None)
    store = f.store
    reproducer._compare_rerun(store, store, executed, executed, "same")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    unexecuted = next(p for p in tight.pairs if p.status == "unexecuted")
    with pytest.raises(ReproductionError, match="both the stored and the re-simulated pair must carry a run"):
        reproducer._compare_rerun(store, store, executed, unexecuted, "missing")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    feasible = next(p for e in crafted.models.values() for p in e.pairs if p.status == "completed")
    infeasible = next(p for p in tight.pairs if p.status == "infeasible")
    assert feasible.run is not None
    assert infeasible.run is not None
    same_arrays = replace(infeasible, run=replace(infeasible.run, arrays_sha256=feasible.run.arrays_sha256))
    with pytest.raises(ReproductionError, match="re-simulated status 'infeasible' != stored 'completed'"):
        reproducer._compare_rerun(store, store, feasible, same_arrays, "status")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


# --- orchestration and the command line ---------------------------------------------------------


def test_reproduce_keeps_going_through_every_step_when_asked(tmp_path: Path) -> None:
    """With --keep-going every step reports, in order, even after the environment step failed."""
    store_root = tmp_path / "store"
    store_root.mkdir()
    result = reproduce(
        scratch=tmp_path / "scratch",
        docs=tmp_path / "no-docs",
        keep_going=True,
        store=StorageRoot(store_root, repositories=(rr.REPO,)),
    )
    assert [c.name for c in result.checks] == list(STEPS)
    outcomes = {c.name: c.ok for c in result.checks}
    assert outcomes["storage"] is True
    assert outcomes["environment"] is False
    assert outcomes["records"] is False
    assert "requires an earlier step" in next(c for c in result.checks if c.name == "records").detail
    assert result.inputs["storage_root"] == "<configured external root>"
    assert not result.ok


def test_step_records_a_passing_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A step that returns names what it verified."""
    reproducer = _bare(tmp_path)
    monkeypatch.setattr(reproducer, "tables", lambda: "all tables")
    check = reproducer.step("tables")
    assert (check.name, check.ok, check.detail) == ("tables", True, "all tables")


def test_checkout_reproduction_stops_at_a_failed_preparation_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed ``uv sync`` returns its exit status and never launches the inner reproduction."""

    def fake_git(*args: str) -> str:
        if args[0] == "rev-parse":
            return "0123456789abcdef0123456789abcdef01234567\n"
        if args[0] == "worktree":
            Path(args[3]).mkdir(parents=True)
        return ""

    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(list(command))
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(f"{MODULE}._git", fake_git)
    monkeypatch.setattr(f"{MODULE}.subprocess.run", fake_run)
    assert run_from_checkout(tmp_path / "scratch", "HEAD", []) == 3
    assert commands == [["uv", "sync", "--locked"]]
    assert "'uv sync --locked' failed in" in capsys.readouterr().out


def test_git_returns_the_command_output() -> None:
    """A successful git command returns its standard output."""
    head = rr._git("rev-parse", "--verify", "HEAD^{commit}").strip()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert len(head) == 40
    assert int(head, 16) >= 0


def _result(*, ok: bool) -> RepetitionReproduction:
    return RepetitionReproduction(
        started_at="2026-09-29T00:00:00+00:00",
        checks=tuple(Check(name, ok=ok, detail="d", elapsed_s=0.0) for name in STEPS),
        inputs={"k": "v"},
        environment={"machine": "x86_64"},
        max_deviation=0.0,
        elapsed_s=1.0,
        comparisons=DECLARED_COMPARISONS,
        resimulation_rule=RESIMULATION_RULE,
        coverage=None,
        doc005=(),
        gates=(),
    )


def test_main_forwards_its_options_and_writes_the_summary_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command passes every option to the reproduction and writes the summary and audit it printed."""
    received: dict[str, object] = {}

    def fake_reproduce(**kwargs: object) -> RepetitionReproduction:
        received.update(kwargs)
        return _result(ok=True)

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(f"{MODULE}.reproduce", fake_reproduce)
    summary, audit = tmp_path / "summary.json", tmp_path / "audit.md"
    argv = [
        "--scratch", str(tmp_path / "scratch"), "--summary", str(summary), "--audit", str(audit),
        "--keep-going", "--exploratory", "--skip-resimulation", "--skip-fits", "--doc005", "--doc005-cpus", "16-17",
        "--gates",
    ]  # fmt: skip
    assert main(argv) == 0
    assert received == {
        "scratch": tmp_path / "scratch",
        "keep_going": True,
        "exploratory": True,
        "skip_resimulation": True,
        "skip_fits": True,
        "doc005": [("canonical-affinity", None), ("cpus-16-17", (16, 17))],
        "gates": True,
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert json.loads(summary.read_text(encoding="utf-8")) == printed
    note = audit.read_text(encoding="utf-8")
    assert note.startswith("# Task 1-a repeated-demonstration pilot reproduction")
    assert "--doc005-cpus 16-17" in note


def test_main_without_a_scratch_uses_a_fresh_temporary_directory_and_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --scratch a fresh directory is created; a failed reproduction exits 1 and writes nothing else."""
    received: dict[str, object] = {}

    def fake_reproduce(**kwargs: object) -> RepetitionReproduction:
        received.update(kwargs)
        return _result(ok=False)

    fresh = tmp_path / "fresh"
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(f"{MODULE}.tempfile.mkdtemp", _returns(str(fresh)))
    monkeypatch.setattr(f"{MODULE}.reproduce", fake_reproduce)
    assert main([]) == 1
    assert received["scratch"] == fresh
    assert received["doc005"] == []
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert sorted(p.name for p in tmp_path.iterdir()) == []
