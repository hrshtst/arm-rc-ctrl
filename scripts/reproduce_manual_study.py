#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Audit the manual-demonstration study's evidence from a clean checkout (M3MAN-011).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.experiments.manual_audit`.
"""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.manual_audit import main

if __name__ == "__main__":
    sys.exit(main())
