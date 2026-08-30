# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-001: the package installs into the ``uv`` environment and imports."""

from __future__ import annotations

from importlib.metadata import version


def test_package_imports_and_reports_installed_version() -> None:
    """``arm_rc_ctrl`` imports and its version matches the installed distribution."""
    import arm_rc_ctrl

    assert arm_rc_ctrl.__version__ == version("arm-rc-ctrl")
    assert arm_rc_ctrl.__version__ == "0.1.0.dev0"
