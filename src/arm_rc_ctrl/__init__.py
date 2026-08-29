# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reservoir-computing target generators for robot-arm control.

This package owns the learning policy, research protocol, experiment
configuration, metrics, tuning, and reproducibility tooling described in
``docs/PLAN.md``. Reservoir computing itself comes from ``rclib``, planar arm
simulation from ``skelarm``, and the CRANE-X7 bridge from ``rtctrl``.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("arm-rc-ctrl")

__all__ = ["__version__"]
