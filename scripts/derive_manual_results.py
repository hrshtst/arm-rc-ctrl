#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Derive the machine-readable evidence of the manual-demonstration study (M3MAN-010).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.experiments.manual_results`.
"""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.manual_results import main

if __name__ == "__main__":
    sys.exit(main())
