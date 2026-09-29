# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""CI-001: the task 1-a reproduction's steps checked against the committed records without the evidence store.

The store-bound work (payload digests, preprocessing, refits, the confirmatory
rerun, the evidence worktree) is replaced by fakes at the module seams; every
comparison against the committed records runs for real.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from arm_rc_ctrl.data.records import ArtifactRecord, ProcessedDatasetRecord, RawDemonstrationRecord, load_record
from arm_rc_ctrl.experiments import reproduce_1a as r1
from arm_rc_ctrl.experiments.reproduce_1a import (
    CONFIRMATORY_REPORT,
    DOCS,
    STEPS,
    Check,
    Reproducer,
    Reproduction,
    ReproductionError,
    audit_markdown,
    compare_suites,
    main,
    prepare_evidence_checkout,
    prepare_scratch,
    reproduce,
    run_from_evidence,
)
from arm_rc_ctrl.experiments.robustness import RobustnessSuite, load_suite
from arm_rc_ctrl.rc.recipe import load_recipe
from arm_rc_ctrl.storage import StorageRoot

if TYPE_CHECKING:
    from collections.abc import Callable

    from arm_rc_ctrl.experiments.perturbations import PerturbationClass

MODULE = "arm_rc_ctrl.experiments.reproduce_1a"
REPO = r1.REPO


def _returns(value: object) -> Callable[..., object]:
    """A stand-in that ignores its arguments and returns ``value``."""

    def stub(*_args: object, **_kwargs: object) -> object:
        return value

    return stub


@pytest.fixture(scope="module")
def suite() -> RobustnessSuite:
    """The committed confirmatory suite (a repository file, no store)."""
    return load_suite(CONFIRMATORY_REPORT)


def _store(root: Path) -> StorageRoot:
    root.mkdir(parents=True, exist_ok=True)
    return StorageRoot(root, repositories=(REPO,))


def _reproducer(tmp_path: Path, *, exploratory: bool = True, tolerance: float = 0.0) -> Reproducer:
    classes: tuple[PerturbationClass, ...] = ("nominal",)
    scratch = prepare_scratch(tmp_path / "scratch")
    return Reproducer(scratch, classes, tolerance, exploratory, _store(tmp_path / "configured"), DOCS, None)


def _records(tmp_path: Path, suite: RobustnessSuite, **kwargs: Any) -> Reproducer:  # noqa: ANN401 - forwarded
    """A reproducer past the records step (which reads only committed files)."""
    reproducer = _reproducer(tmp_path, **kwargs)
    reproducer.suite = suite
    assert reproducer.step("storage").ok
    check = reproducer.step("records")
    assert check.ok, check.detail
    return reproducer


# --- comparison -----------------------------------------------------------------------------


def test_comparison_treats_nan_as_equal_only_to_nan_and_names_structural_differences() -> None:
    """Two NaNs agree; NaN against a number, a missing key, or a changed kind is a categorical difference."""
    differences: list[str] = []
    assert r1._compare("a", math.nan, math.nan, differences) == 0.0  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert differences == []
    assert r1._compare("a", math.nan, 1.0, differences) == 0.0  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert differences == ["a: nan vs 1.0"]
    differences.clear()
    worst = r1._compare("r", {"x": 1.0, "y": 2.0}, {"x": 1.5, "z": 2.0}, differences)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert worst == 0.5
    assert differences == ["r.y: present in one report only", "r.z: present in one report only"]
    differences.clear()
    assert r1._compare("r", {"x": 1.0}, [1.0], differences) == 0.0  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    assert differences == ["r: dict vs list"]


def test_compare_suites_names_a_rebuilt_run_the_committed_suite_lacks(suite: RobustnessSuite) -> None:
    """A rebuilt run with no committed counterpart is a difference, never silently skipped."""
    first = suite.runs[0]
    stray = SimpleNamespace(arm=first.arm, scenario_id="not-a-scenario", report=first.report)
    rebuilt = cast("RobustnessSuite", SimpleNamespace(runs=(first, stray)))
    assert compare_suites(suite, rebuilt) == (0.0, [f"{first.arm}/not-a-scenario: not in the committed suite"])


def test_an_existing_empty_scratch_directory_is_accepted(tmp_path: Path) -> None:
    """A fresh, empty directory is used as it is."""
    (tmp_path / "empty").mkdir()
    assert prepare_scratch(tmp_path / "empty") == (tmp_path / "empty").resolve()


# --- environment, storage, records ------------------------------------------------------------


def _pins(monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite, lock: str) -> None:
    monkeypatch.setattr(f"{MODULE}.verify_builds", _returns(["skelarm"]))
    monkeypatch.setattr(f"{MODULE}.submodule_revisions", _returns(list(suite.provenance.submodules)))
    monkeypatch.setattr(f"{MODULE}.sha256_file", _returns(lock))


def test_environment_requires_the_evidence_lock_and_records_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite
) -> None:
    """At the evidence pins a different uv.lock is refused; the evidence lock passes and the checkout is recorded."""
    _pins(monkeypatch, suite, "0" * 64)
    with pytest.raises(ReproductionError, match=r"uv\.lock digest 000000000000 differs from the evidence's"):
        _reproducer(tmp_path / "lock").environment()
    _pins(monkeypatch, suite, suite.provenance.lock_sha256)
    reproducer = _reproducer(tmp_path / "ok")
    detail = reproducer.environment()
    assert detail.startswith("1 build identities verified; submodules")
    assert reproducer.inputs["lock_sha256"] == suite.provenance.lock_sha256
    assert reproducer.inputs["evidence_project_commit"] == suite.provenance.project_commit
    assert len(reproducer.inputs["reproduction_project_commit"]) == 40
    assert reproducer.inputs["reproduction_project_dirty"] in {"True", "False"}


def test_storage_uses_the_configured_root_or_opens_the_configured_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An injected store is used as given; otherwise the storage configuration is opened."""
    given = _reproducer(tmp_path / "given")
    configured = given.configured_store
    assert given.storage() == "external storage root resolved"
    assert given.store is configured
    opened = _store(tmp_path / "opened")
    monkeypatch.setattr(f"{MODULE}.open_storage", lambda: opened)
    fresh = Reproducer(tmp_path / "s2", ("nominal",), 0.0, True, None, DOCS, None)  # noqa: FBT003
    fresh.storage()
    assert fresh.store is opened
    assert fresh.inputs["storage_root"] == "<configured external root>"


def test_records_resolve_the_committed_recipe_dataset_and_run_pointers(tmp_path: Path, suite: RobustnessSuite) -> None:
    """The committed records name one recipe, its dataset, the raw demonstration, and every run pointer."""
    reproducer = _records(tmp_path, suite)
    assert reproducer.recipe is not None
    assert reproducer.recipe.name == suite.recipe
    assert reproducer.processed is not None
    assert reproducer.processed.artifact.artifact_id == suite.reference_artifact
    assert reproducer.raw is not None
    assert reproducer.inputs["raw"] == reproducer.raw.artifact.artifact_id
    assert reproducer.inputs["recipe"] == suite.recipe_file
    assert reproducer.records().endswith(f"{len(suite.runs)} run pointers")


def _suite_like(suite: RobustnessSuite, **overrides: object) -> RobustnessSuite:
    fields = {k: getattr(suite, k) for k in ("recipe", "recipe_file", "reference_artifact", "runs")}
    return cast("RobustnessSuite", SimpleNamespace(**{**fields, **overrides}))


def test_records_refuse_another_recipe_dataset_or_a_missing_pointer(tmp_path: Path, suite: RobustnessSuite) -> None:
    """The recipe must be the evidence's, its dataset the reference artifact, and every run must have a pointer."""
    reproducer = _reproducer(tmp_path / "recipe")
    reproducer.suite = _suite_like(suite, recipe="other-recipe")
    with pytest.raises(ReproductionError, match="the evidence used 'other-recipe'"):
        reproducer.records()
    reproducer = _reproducer(tmp_path / "dataset")
    reproducer.suite = _suite_like(suite, reference_artifact="processed-other")
    with pytest.raises(ReproductionError, match="not the evidence's reference artifact"):
        reproducer.records()
    missing = SimpleNamespace(run_id="run-20000101-000000000000", pointer="data/records/runs/missing.toml")
    reproducer = _reproducer(tmp_path / "pointer")
    reproducer.suite = _suite_like(suite, runs=(*suite.runs[:2], missing))
    with pytest.raises(
        ReproductionError, match=r"1 evidence runs lack a pointer .* \(first: run-20000101-000000000000\)"
    ):
        reproducer.records()
    assert reproducer.recipe is None  # nothing is recorded from a refused step


def test_a_step_without_its_predecessor_fails_by_name(tmp_path: Path) -> None:
    """Running a step before the one that resolves its input is a named failure, not a crash."""
    check = _reproducer(tmp_path).step("payloads")
    assert not check.ok
    assert check.detail == "ReproductionError: step 'payloads' requires an earlier step that did not run"


# --- payloads, data, model ------------------------------------------------------------------


def test_payloads_verify_the_raw_processed_and_every_run_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite
) -> None:
    """Every payload the records name is verified against its recorded digest."""
    reproducer = _records(tmp_path, suite)
    verified: list[str] = []

    def verify(_store: StorageRoot, artifact: ArtifactRecord) -> None:
        verified.append(artifact.artifact_id)

    monkeypatch.setattr(f"{MODULE}.verify_payload", verify)
    detail = reproducer.payloads()
    assert detail == f"raw, processed, and {len(suite.runs)} run payloads verified against their recorded digests"
    assert reproducer.raw is not None
    assert verified[:2] == [reproducer.raw.artifact.artifact_id, suite.reference_artifact]
    assert len(verified) == 2 + len(suite.runs)


def _processed_digest(record: ProcessedDatasetRecord) -> str:
    return record.artifact.payload.sha256


def test_data_rebuilds_the_dataset_in_the_scratch_store_and_compares_its_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite
) -> None:
    """The raw payload is copied into a scratch store, preprocessed there, and the digest must be the committed one."""
    reproducer = _records(tmp_path / "ok", suite)
    raw = cast("RawDemonstrationRecord", reproducer.raw)
    processed = cast("ProcessedDatasetRecord", reproducer.processed)
    store = cast("StorageRoot", reproducer.store)
    store.path(raw.artifact.payload.uri, mode="write").write_bytes(b"raw payload bytes")
    calls: list[dict[str, object]] = []

    def preprocess(*args: object, **kwargs: object) -> object:
        calls.append({"args": args, **kwargs})
        digest = _processed_digest(processed) if len(calls) == 1 else "f" * 64
        return SimpleNamespace(record=SimpleNamespace(artifact=SimpleNamespace(payload=SimpleNamespace(sha256=digest))))

    monkeypatch.setattr(f"{MODULE}.preprocess_demonstration", preprocess)
    detail = reproducer.data()
    assert detail == f"processed dataset rebuilt with digest {_processed_digest(processed)[:12]} (identical)"
    scratch_store = reproducer.scratch_store
    assert scratch_store is not None
    assert scratch_store.path(raw.artifact.payload.uri, mode="read").read_bytes() == b"raw payload bytes"
    (call,) = calls
    assert call["store"] is scratch_store
    assert call["exploratory"] is True
    assert call["records_root"] == reproducer.scratch / "repo"
    assert (reproducer.scratch / "repo" / "data" / "records" / "processed").is_dir()

    other = _records(tmp_path / "differs", suite)
    cast("StorageRoot", other.store).path(raw.artifact.payload.uri, mode="write").write_bytes(b"raw payload bytes")
    with pytest.raises(ReproductionError, match=r"rebuilt dataset digest ffffffffffff differs from the committed"):
        other.data()
    assert other.scratch_store is None


def test_model_refits_the_recipe_on_the_verified_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite
) -> None:
    """The recipe refits on the samples loaded from the verified processed payload."""
    reproducer = _records(tmp_path, suite)
    processed = cast("ProcessedDatasetRecord", reproducer.processed)
    samples = object()
    refits: list[dict[str, object]] = []

    def verify(_store: StorageRoot, artifact: ArtifactRecord) -> Path:
        return Path(artifact.artifact_id)

    def load(path: Path) -> object:
        return samples if path.name == processed.artifact.artifact_id else None

    monkeypatch.setattr(f"{MODULE}.verify_payload", verify)
    monkeypatch.setattr(f"{MODULE}.load_samples", load)

    def refit(training: dict[str, object]) -> tuple[object, object]:
        refits.append(training)
        return object(), SimpleNamespace(rmse=0.0123)

    reproducer.recipe = cast("Any", SimpleNamespace(refit=refit))
    assert reproducer.model() == "recipe refitted; fit RMSE 0.0123 rad reproduced"
    assert refits == [{processed.artifact.artifact_id: samples}]
    assert reproducer.samples is samples


# --- the confirmatory rerun and the report ------------------------------------------------------


def _evaluation_ready(tmp_path: Path, suite: RobustnessSuite, *, tolerance: float = 0.0) -> Reproducer:
    reproducer = _records(tmp_path, suite, exploratory=False, tolerance=tolerance)
    processed = cast("ProcessedDatasetRecord", reproducer.processed)
    cast("StorageRoot", reproducer.store).path(processed.artifact.payload.uri, mode="write").write_bytes(b"processed")
    reproducer.scratch_store = _store(tmp_path / "scratch-store")
    reproducer.samples = cast("Any", object())
    return reproducer


def _rerun_seams(monkeypatch: pytest.MonkeyPatch, rebuilt: RobustnessSuite) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def run(*args: object, **kwargs: object) -> RobustnessSuite:
        calls.append({"args": args, **kwargs})
        return rebuilt

    monkeypatch.setattr(f"{MODULE}.load_confirmatory", _returns(SimpleNamespace(scenario="scenario.toml")))
    monkeypatch.setattr(f"{MODULE}.load_scenario", "loaded {}".format)
    monkeypatch.setattr(f"{MODULE}.load_frozen_baseline", "tracker {}".format)
    monkeypatch.setattr(f"{MODULE}.run_robustness", run)
    return calls


def _bumped(suite: RobustnessSuite, delta: float) -> RobustnessSuite:
    first = suite.runs[0]
    assert first.report.joint_rmse is not None
    rmse = replace(first.report.joint_rmse, aggregate=first.report.joint_rmse.aggregate + delta)
    run = replace(first, report=replace(first.report, joint_rmse=rmse))
    return replace(suite, runs=(run, *suite.runs[1:]), aggregates=(), effects=())


def test_the_confirmatory_rerun_replays_the_recorded_scenarios_and_matches_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite
) -> None:
    """The rerun uses the recorded arms and scenarios under its own label; an identical suite deviates by zero."""
    reproducer = _evaluation_ready(tmp_path, suite)
    calls = _rerun_seams(monkeypatch, suite)
    detail = reproducer.evaluation()
    assert detail == f"{len(suite.runs)} runs re-evaluated; largest deviation 0.000e+00 (tolerance 0.000e+00)"
    assert reproducer.max_deviation == 0.0
    (call,) = calls
    assert call["label"] == "confirmatory-rerun"
    assert call["exploratory"] is False
    assert call["scenarios"] == suite.scenarios
    assert call["arms"] == suite.arms
    assert call["classes"] == ("nominal",)
    assert call["store"] is reproducer.scratch_store
    assert call["trackers"] == {a.tracker: f"tracker {a.tracker}" for a in suite.arms}
    processed = cast("ProcessedDatasetRecord", reproducer.processed)
    copied = cast("StorageRoot", reproducer.scratch_store).path(processed.artifact.payload.uri, mode="read")
    assert copied.read_bytes() == b"processed"


def test_the_confirmatory_rerun_enforces_its_tolerance_and_categorical_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite
) -> None:
    """A float deviation passes only within the tolerance; any categorical difference fails regardless."""
    _rerun_seams(monkeypatch, _bumped(suite, 1e-6))
    strict = _evaluation_ready(tmp_path / "strict", suite)
    with pytest.raises(
        ReproductionError, match=r"largest metric deviation 1\.000e-06 exceeds the tolerance 0\.000e\+00"
    ):
        strict.evaluation()
    assert strict.max_deviation == pytest.approx(1e-6)
    tolerant = _evaluation_ready(tmp_path / "tolerant", suite, tolerance=1e-5)
    assert tolerant.evaluation().endswith("(tolerance 1.000e-05)")

    first = suite.runs[0]
    stray = SimpleNamespace(arm=first.arm, scenario_id="not-a-scenario", report=first.report)
    _rerun_seams(monkeypatch, cast("RobustnessSuite", SimpleNamespace(runs=(stray,))))
    with pytest.raises(ReproductionError, match="1 categorical differences"):
        _evaluation_ready(tmp_path / "categorical", suite, tolerance=1.0).evaluation()


def test_the_confirmatory_rerun_refuses_an_exploratory_checkout(tmp_path: Path, suite: RobustnessSuite) -> None:
    """--exploratory lets the other steps run but never the confirmatory rerun."""
    reproducer = _records(tmp_path, suite, exploratory=True)
    with pytest.raises(ReproductionError, match="requires a clean checkout"):
        reproducer.evaluation()


def test_the_report_re_renders_the_committed_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed report re-renders identically from the committed evidence; any change is named."""
    assert _reproducer(tmp_path / "ok").report() == "report re-rendered identically from the committed evidence"
    monkeypatch.setattr(f"{MODULE}.render_report", _returns("# edited\n"))
    with pytest.raises(ReproductionError, match=r"differs from the committed report\.md"):
        _reproducer(tmp_path / "edited").report()


# --- steps, orchestration, the audit --------------------------------------------------------------


def test_a_passing_step_records_its_detail_without_machine_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absolute paths in a step's detail are reduced to their basenames."""
    reproducer = _reproducer(tmp_path)
    monkeypatch.setattr(reproducer, "report", lambda: f"wrote {tmp_path}/deep/result.json")
    check = reproducer.step("report")
    assert (check.name, check.ok, check.detail) == ("report", True, "wrote result.json")


def test_reproduce_runs_every_step_and_keeps_going_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All passing steps make a passing reproduction; failures stop it unless --keep-going."""

    def passing(self: Reproducer, name: str) -> Check:
        self.inputs[name] = "ok"
        return Check(name, ok=True, detail="ok", elapsed_s=0.0)

    monkeypatch.setattr(Reproducer, "step", passing)
    result = reproduce(scratch=tmp_path / "pass", classes=("nominal",))
    assert result.ok
    assert [c.name for c in result.checks] == list(STEPS)
    assert result.inputs == dict.fromkeys(STEPS, "ok")
    note = audit_markdown(result, command="python -m arm_rc_ctrl.experiments.reproduce_1a")
    assert "- Outcome: PASS" in note
    assert "Steps not run" not in note

    def failing(_self: Reproducer, name: str) -> Check:
        return Check(name, ok=False, detail="no", elapsed_s=0.0)

    monkeypatch.setattr(Reproducer, "step", failing)
    kept = reproduce(scratch=tmp_path / "kept", classes=("nominal",), keep_going=True)
    assert [c.name for c in kept.checks] == list(STEPS)
    stopped = reproduce(scratch=tmp_path / "stopped", classes=("nominal",))
    assert [c.name for c in stopped.checks] == ["environment"]


# --- the evidence worktree ----------------------------------------------------------------------


class _WorktreeGit:
    """A git stand-in: ``worktree add`` creates the checkout, ``submodule status`` reports the given pins."""

    def __init__(self, suite: RobustnessSuite, *, revisions: dict[str, str] | None = None, script: bool = True) -> None:
        recorded = {s.name: (s.checked_out or s.recorded) for s in suite.provenance.submodules}
        self.status = {**recorded, **(revisions or {})}
        self.script = script
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> str:
        self.calls.append(args)
        if args[:2] == ("worktree", "add"):
            checkout = Path(args[3])
            (checkout / "scripts").mkdir(parents=True)
            if self.script:
                (checkout / "scripts" / "reproduce_1a.py").write_text("", encoding="utf-8")
        if args[-2:] == ("submodule", "status"):
            return "".join(f" {rev} third_party/{name} (heads/main)\n" for name, rev in self.status.items())
        return ""


def test_the_evidence_checkout_is_at_the_audit_commit_with_the_evidence_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite, capsys: pytest.CaptureFixture[str]
) -> None:
    """The worktree is created at the recorded audit commit and every submodule pin is re-verified."""
    audit = json.loads((DOCS / "reproduction_audit.json").read_text(encoding="utf-8"))
    git = _WorktreeGit(suite)
    monkeypatch.setattr(f"{MODULE}._git", git)
    checkout, commit = prepare_evidence_checkout(tmp_path / "scratch")
    assert commit == audit["inputs"]["reproduction_project_commit"]
    assert checkout == (tmp_path / "scratch").resolve() / "evidence"
    assert git.calls[0] == ("worktree", "add", "--detach", str(checkout), commit)
    assert "git worktree remove --force" in capsys.readouterr().out


def test_the_evidence_checkout_refuses_other_pins_or_a_missing_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: RobustnessSuite
) -> None:
    """A submodule at another revision or a checkout without the reproduction script is refused."""
    name = suite.provenance.submodules[0].name
    monkeypatch.setattr(f"{MODULE}._git", _WorktreeGit(suite, revisions={name: "0" * 40}))
    with pytest.raises(ReproductionError, match=f"evidence checkout has {name} at 000000000000"):
        prepare_evidence_checkout(tmp_path / "pins", keep=True)
    monkeypatch.setattr(f"{MODULE}._git", _WorktreeGit(suite, script=False))
    with pytest.raises(ReproductionError, match=r"carries no scripts/reproduce_1a\.py"):
        prepare_evidence_checkout(tmp_path / "script", keep=True)


def test_git_returns_its_output_or_names_the_failed_command() -> None:
    """A successful git call returns standard output; a failing one raises with the command named."""
    assert len(r1._git("rev-parse", "--verify", "HEAD^{commit}").strip()) == 40  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ReproductionError, match="git rev-parse --verify no-such-ref-0123 failed"):
        r1._git("rev-parse", "--verify", "no-such-ref-0123")  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_run_from_evidence_syncs_rebuilds_and_runs_the_inner_reproduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed preparation stage returns its status; otherwise the inner command's status is returned."""
    checkout = tmp_path / "evidence"
    monkeypatch.setattr(f"{MODULE}.prepare_evidence_checkout", _returns((checkout, "c" * 40)))
    commands: list[tuple[list[str], object]] = []
    codes = iter([4])

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append((list(command), kwargs.get("cwd")))
        return SimpleNamespace(returncode=next(codes, 0))

    monkeypatch.setattr(f"{MODULE}.subprocess.run", run)
    assert run_from_evidence(tmp_path / "scratch", ["--classes", "nominal"]) == 4
    assert [c for c, _ in commands] == [["uv", "sync", "--locked"]]
    assert "'uv sync --locked' failed in" in capsys.readouterr().out

    commands.clear()
    codes = iter([0, 0, 7])
    assert run_from_evidence(tmp_path / "scratch", ["--classes", "nominal"]) == 7
    assert [cwd for _, cwd in commands] == [checkout] * 3
    inner = commands[-1][0]
    assert inner[:5] == ["uv", "run", "--locked", "python", "scripts/reproduce_1a.py"]
    assert inner[inner.index("--scratch") + 1] == str(tmp_path / "scratch" / "inner")
    assert inner[-2:] == ["--classes", "nominal"]
    assert f"reproducing at {'c' * 12}" in capsys.readouterr().out


def test_main_forwards_its_options_to_the_evidence_worktree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--from-evidence forwards classes, tolerance, absolute output paths, and --keep-going."""
    received: list[tuple[Path, list[str]]] = []
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    def run(scratch: Path, forwarded: list[str]) -> int:
        received.append((scratch, list(forwarded)))
        return 0

    monkeypatch.setattr(f"{MODULE}.run_from_evidence", run)
    monkeypatch.chdir(tmp_path)
    argv = ["--from-evidence", "--scratch", "s", "--classes", "nominal", "--summary", "sum.json", "--audit", "a.md"]
    assert main([*argv, "--keep-going", "--tolerance", "1e-9"]) == 0
    ((scratch, forwarded),) = received
    assert scratch == Path("s")
    assert forwarded == [
        "--classes", "nominal", "--tolerance", "1e-09",
        "--summary", str(tmp_path / "sum.json"), "--audit", str(tmp_path / "a.md"), "--keep-going",
    ]  # fmt: skip
    received.clear()
    assert main(["--from-evidence", "--scratch", "s", "--classes", "nominal"]) == 0
    assert received[0][1] == ["--classes", "nominal", "--tolerance", "0.0"]


def test_main_prints_a_passing_summary_without_writing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing reproduction exits 0 and only prints its summary when no output file is requested."""
    passing = Reproduction(
        started_at="2026-09-29T00:00:00+00:00",
        checks=tuple(Check(name, ok=True, detail="ok", elapsed_s=0.0) for name in STEPS),
        inputs={},
        environment={},
        max_deviation=0.0,
        elapsed_s=0.0,
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(f"{MODULE}.reproduce", _returns(passing))
    monkeypatch.chdir(tmp_path)
    assert main(["--scratch", "s", "--classes", "nominal"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert list(tmp_path.iterdir()) == []


def test_the_committed_recipe_file_names_the_evidence_recipe(suite: RobustnessSuite) -> None:
    """The records step's premise holds for the committed evidence: its recipe file names its recipe."""
    assert load_recipe(REPO / suite.recipe_file).name == suite.recipe
    (source,) = load_recipe(REPO / suite.recipe_file).datasets
    processed = load_record(REPO / source.record, ProcessedDatasetRecord)
    assert processed.artifact.artifact_id == suite.reference_artifact
