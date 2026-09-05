#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Plot task 1-a recovery augmentation trajectories in task space."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.augmentation_plots import main

if __name__ == "__main__":
    sys.exit(main())
