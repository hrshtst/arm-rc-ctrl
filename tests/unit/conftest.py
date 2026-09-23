# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Shared unit-test fixtures: the planar repetition fixture store bound to this process's affinity."""

from __future__ import annotations

import importlib
import os
import shutil
from typing import TYPE_CHECKING

import pytest

from arm_rc_ctrl.execution import AffinityRequest, collect_execution
from arm_rc_ctrl.experiments import manual_evaluation, manual_figures, manual_results
from arm_rc_ctrl.experiments.manual_fixture import NOW as MANUAL_NOW
from arm_rc_ctrl.experiments.manual_fixture import (
    ManualFixture,
    ManualStudyEvidence,
    build_manual_fixture,
    manual_four_arms,
    manual_narrowed,
)
from arm_rc_ctrl.experiments.manual_search import load_manual_search
from arm_rc_ctrl.experiments.repetition_fixture import NOW, PlanarFixture, build_planar_fixture
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import ENV_VAR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> PlanarFixture:
    """Build the planar recovery dataset in a temporary root/store and declare this process's affinity canonical."""
    base = tmp_path_factory.mktemp("repetition")
    importlib.import_module("rclib")  # the OpenMP probe must see the runtime the workers will also load
    request = AffinityRequest("explicit", tuple(sorted(os.sched_getaffinity(0))))
    env = {**os.environ, **request.environment(), ENV_VAR: str(base / "store")}
    execution = collect_execution(
        command="python -m arm_rc_ctrl.experiments.repetition_numerics validate",
        env=env,
        effective_cpus=request.cpus,
        now=NOW,
    )
    if not execution.canonical:  # pragma: no cover - only on a multithreaded test runner
        pytest.skip("the test process's numerical runtimes are not single-threaded")
    return build_planar_fixture(base, execution=execution, env=env)


@pytest.fixture
def pinned_environment(fixture: PlanarFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this process look launched through the pinned launcher with the fixture's store."""
    for name, value in fixture.env.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(scope="module")
def manual_fixture(tmp_path_factory: pytest.TempPathFactory) -> ManualFixture:
    """Build the ten manual demonstrations, their banks, and the frozen study manifest in a temporary root."""
    base = tmp_path_factory.mktemp("manual")
    importlib.import_module("rclib")  # the OpenMP probe must see the runtime the workers will also load
    request = AffinityRequest("explicit", tuple(sorted(os.sched_getaffinity(0))))
    env = {**os.environ, **request.environment(), ENV_VAR: str(base / "store")}
    execution = collect_execution(
        command="python -m arm_rc_ctrl.experiments.manual_numerics validate",
        env=env,
        effective_cpus=request.cpus,
        now=MANUAL_NOW,
    )
    if not execution.canonical:  # pragma: no cover - only on a multithreaded test runner
        pytest.skip("the test process's numerical runtimes are not single-threaded")
    return build_manual_fixture(base, execution=execution, env=env)


# --- the manual study's evidence, evaluated once per test module -----------------------------------


@pytest.fixture(scope="module")
def study(manual_fixture: ManualFixture, tmp_path_factory: pytest.TempPathFactory) -> Iterator[ManualStudyEvidence]:
    """Run the real evaluation command over the four illustrated arms, once for every test in a module."""
    base = tmp_path_factory.mktemp("manual-study")
    with pytest.MonkeyPatch.context() as patch:
        for name, value in manual_fixture.env.items():
            patch.setenv(name, value)
        built = ManualStudyEvidence(manual_fixture, base)
        patch.setattr(manual_evaluation, "repository_root", lambda: manual_fixture.root)
        patch.setattr(manual_evaluation, "evaluation_entries", manual_four_arms)
        patch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
        assert manual_evaluation.main(built.run_argv()) == 0
        yield built


@pytest.fixture
def patch_manual(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Point a command at the fixture study: its environment, its root, and the two narrowed scenarios."""

    def patch(study: ManualStudyEvidence, *modules: object) -> None:
        for name, value in study.f.env.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(manual_evaluation, "evaluation_scenarios", manual_narrowed)
        for module in modules:
            monkeypatch.setattr(module, "repository_root", lambda: study.f.root)

    return patch


@pytest.fixture
def derived(
    study: ManualStudyEvidence,
    capsys: pytest.CaptureFixture[str],
    patch_manual: Callable[..., None],
) -> Iterator[Path]:
    """Derive this study's evidence into a fresh directory and return it."""
    patch_manual(study, manual_results, manual_figures)
    # Inside the fixture root, where the real study keeps them: the audit's bundle cites repository paths.
    output = study.f.root / "docs" / "experiments" / "task_1a_manual_demonstration" / "results"
    assert manual_results.main(study.derive_argv(output)) == 0
    capsys.readouterr()
    yield output
    shutil.rmtree(output)  # one derivation per test: the command refuses to overwrite its own outputs


# --- the manual ESN search over the fixture study (M3MS) -------------------------------------------

SEARCH_PROTOCOL_FILE = repository_root() / "configs/studies/manual_esn_search_v1.toml"


def _fixture_protocol(f: ManualFixture, evaluation: Path) -> Path:
    """A search protocol over the fixture study, written where the fixture root is the repository."""
    filters = f.root / "configs" / "evaluations" / "fixture_filters.toml"
    filters.parent.mkdir(parents=True, exist_ok=True)
    filters.write_text(
        'name = "fixture-filters"\ntracker = "../controllers/task_1a_pd_v2.toml"\n\n'
        "[estimator]\nvelocity_cutoff_hz = 20.0\nacceleration_cutoff_hz = 8.0\nmax_dt_ratio = 3.0\n",
        encoding="utf-8",
    )
    body = SEARCH_PROTOCOL_FILE.read_text(encoding="utf-8")
    replacements = {
        'study = "../../docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json"': (
            f'study = "{f.manifest_file}"'
        ),
        'model = "../models/esn_task_1a_v4.toml"': f'model = "{f.root / f.manifest.model.path}"',
        'scenario = "../tasks/task_1a_manual_v2.toml"': f'scenario = "{f.scenario_file}"',
        'filters = "../evaluations/task_1a_nominal_v4.toml"': f'filters = "{filters}"',
        "velocity_cutoff_hz = 29.980411525699598": "velocity_cutoff_hz = 20.0",
        "acceleration_cutoff_hz = 10.938122239871603": "acceleration_cutoff_hz = 8.0",
        'evaluation = "../evaluations/task_1a_manual_dev_v1.toml"': f'evaluation = "{evaluation}"',
    }
    for old, new in replacements.items():
        assert old in body, old
        body = body.replace(old, new, 1)
    target = f.root / "configs" / "studies" / "fixture_search.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


@pytest.fixture
def fixture_search(
    manual_fixture: ManualFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ManualSearchProtocol:
    """The committed protocol, re-pointed at the fixture study, with that root as the repository."""
    base = tmp_path / "study"
    base.mkdir(parents=True, exist_ok=True)
    evidence = ManualStudyEvidence(manual_fixture, base)
    target = _fixture_protocol(manual_fixture, evidence.evaluation)
    # Only the protocol's own view of the repository moves: provenance keeps the real checkout, which
    # is what the fixture study's own evidence was recorded under.
    monkeypatch.setattr("arm_rc_ctrl.experiments.manual_search.repository_root", lambda: manual_fixture.root)
    return load_manual_search(target)
