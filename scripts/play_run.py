#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Export one verified run to a temporary log and play it with the pinned ``skelarm`` player."""

from __future__ import annotations

import sys

from arm_rc_ctrl.experiments.playback import main_play

if __name__ == "__main__":
    sys.exit(main_play())
