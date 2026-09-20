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
from arm_rc_ctrl.experiments.repetition_fixture import NOW, PlanarFixture, build_planar_fixture
from arm_rc_ctrl.storage import ENV_VAR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


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
