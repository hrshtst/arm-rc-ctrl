#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the task 1-a repeated-demonstration pilot from the committed records (M3REP-008)."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.reproduce_repetition import main

if __name__ == "__main__":
    sys.exit(main())
