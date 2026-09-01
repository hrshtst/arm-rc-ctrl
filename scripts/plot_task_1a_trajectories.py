#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Generate representative task 1-a joint-trajectory plots."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.trajectory_plots import main

if __name__ == "__main__":
    sys.exit(main())
