# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-009: the committed canonical execution record is strict, pinned, single-threaded, and rebuilds in place."""

from __future__ import annotations

import importlib
import platform
from datetime import UTC, datetime

import pytest

from arm_rc_ctrl.execution import (
    THREAD_VARIABLES,
    AffinityRequest,
    CoreTopology,
    collect_execution,
    load_execution,
    render_execution_markdown,
)
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

DOCS = repository_root() / "docs" / "experiments" / "task_1a_repeated_demonstration"
RECORD = DOCS / "execution_environment_v1.json"
MARKDOWN = DOCS / "execution_environment_v1.md"


def test_canonical_record_is_pinned_single_threaded_and_verified() -> None:
    """The strict load re-derives both verification flags; the record describes the P-core, one-thread environment."""
    record = load_execution(RECORD)
    assert record.canonical
    assert record.policy == "p-cores"
    assert record.role == "main"
    assert record.topology.hybrid
    assert record.requested_cpus == record.effective_cpus == record.topology.core_types["core"]
    assert record.thread_environment == dict.fromkeys(THREAD_VARIABLES, "1")
    assert record.blas.library is not None
    assert record.blas.corename is not None
    assert record.blas.num_threads == 1
    assert record.openmp.max_threads == 1  # rclib's OpenMP runtime, loaded by the record command
    assert set(record.topology.l2_cache_kib) == set(record.topology.core_types)
    assert record.command.startswith("python -m arm_rc_ctrl.execution record")
    assert "/" not in record.command  # no machine path in the recorded launch command
    assert render_execution_markdown(record) == MARKDOWN.read_text(encoding="utf-8")


def test_canonical_record_rebuilds_on_the_same_machine() -> None:
    """On the machine and kernel that produced it, collecting the environment again yields the same identity."""
    committed = load_execution(RECORD)
    topology = CoreTopology.from_sysfs()
    if topology != committed.topology or platform.release() != committed.release:
        pytest.skip("the canonical record belongs to another machine or kernel")
    importlib.import_module("rclib")
    request = AffinityRequest.resolve("p-cores", topology)
    rebuilt = collect_execution(
        command=committed.command,
        env=request.environment(),
        topology=topology,
        effective_cpus=committed.effective_cpus,
        now=datetime.now(tz=UTC),
    )
    assert rebuilt.identity == committed.identity
