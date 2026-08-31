#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the task 1-a result from the committed records (see ``arm_rc_ctrl.experiments.reproduce_1a``)."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.reproduce_1a import main

if __name__ == "__main__":
    sys.exit(main())
