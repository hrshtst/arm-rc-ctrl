#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Render the repeated-demonstration pilot's report tables, task-clock figures, and animations (M3REP-007)."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.repetition_report import main

if __name__ == "__main__":
    sys.exit(main())
