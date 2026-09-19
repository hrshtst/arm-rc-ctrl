#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""List, plot or animate the manual study's illustrated cases from their committed inputs (M3MAN-010).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.experiments.manual_figures`.
"""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.manual_figures import main

if __name__ == "__main__":
    sys.exit(main())
