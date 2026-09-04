#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the task 1-a recovery result (negative-result path) from the committed records."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.reproduce_recovery import main

if __name__ == "__main__":
    sys.exit(main())
