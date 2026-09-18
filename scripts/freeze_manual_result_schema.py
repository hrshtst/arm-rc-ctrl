#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Freeze the result schema of the manual-demonstration evidence (M3MAN-009).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.experiments.manual_schema`.
"""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.manual_schema import main

if __name__ == "__main__":
    sys.exit(main())
