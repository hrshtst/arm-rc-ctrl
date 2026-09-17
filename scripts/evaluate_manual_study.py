#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Evaluate the manual-demonstration study against paired replay of each parent (M3MAN-008).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.experiments.manual_evaluation`.
"""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.manual_evaluation import main

if __name__ == "__main__":
    sys.exit(main())
