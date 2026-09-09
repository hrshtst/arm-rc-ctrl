#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Resolve and freeze the repeated-demonstration pilot's six-configuration panel (M3REP-001)."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.repetition_panel import main

if __name__ == "__main__":
    sys.exit(main())
