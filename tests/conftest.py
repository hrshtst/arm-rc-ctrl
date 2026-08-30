# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Process-wide test settings applied before any domain library is imported."""

from __future__ import annotations

import os

# skelarm imports PyQt6 at package import; never require a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# rclib may use OpenMP; one thread keeps reductions bitwise deterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # placed after the environment settings on purpose; pytest itself imports no domain library


def pytest_addoption(parser: pytest.Parser) -> None:
    """Regression tests can rewrite their committed expectations instead of comparing against them."""
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="rewrite the committed baseline replay expectations under tests/fixtures/regression",
    )


@pytest.fixture
def update_baselines(request: pytest.FixtureRequest) -> bool:
    """Whether the run was asked to rewrite committed regression expectations."""
    return bool(request.config.getoption("--update-baselines"))
