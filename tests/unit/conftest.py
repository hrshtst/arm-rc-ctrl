# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Shared unit-test fixtures: the planar repetition fixture store bound to this process's affinity."""

from __future__ import annotations

import importlib
import os

import pytest

from arm_rc_ctrl.execution import AffinityRequest, collect_execution
from arm_rc_ctrl.experiments.manual_fixture import NOW as MANUAL_NOW
from arm_rc_ctrl.experiments.manual_fixture import ManualFixture, build_manual_fixture
from arm_rc_ctrl.experiments.repetition_fixture import NOW, PlanarFixture, build_planar_fixture
from arm_rc_ctrl.storage import ENV_VAR


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
