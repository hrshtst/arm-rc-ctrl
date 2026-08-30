"""Planted failing test, not collected by the normal run (no ``test_`` prefix)."""

from __future__ import annotations

import arm_rc_ctrl


def test_deliberate_failure() -> None:
    """Fail on purpose so the runner is proven to execute assertions."""
    assert arm_rc_ctrl.__version__ == "not-a-version"
