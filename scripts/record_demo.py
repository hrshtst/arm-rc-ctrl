# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Launch the pinned skelarm trajectory recorder for a manual-demonstration session (M3MAN-003, I7).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.data.recording`.
"""

from __future__ import annotations

from arm_rc_ctrl.data.recording import main

if __name__ == "__main__":
    raise SystemExit(main())
