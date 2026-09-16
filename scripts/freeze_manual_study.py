#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Freeze the manual-demonstration study manifest binding all 186 model identities (M3MAN-007).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.experiments.manual_study`.
"""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.manual_study import main

if __name__ == "__main__":
    sys.exit(main())
