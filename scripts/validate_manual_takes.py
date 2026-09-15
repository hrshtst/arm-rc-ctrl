# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Validate a batch of saved manual takes offline and update the bank manifest (M3MAN-002).

Thin entry point; the logic lives in :mod:`arm_rc_ctrl.experiments.manual_bank`.
"""

from __future__ import annotations

from arm_rc_ctrl.experiments.manual_bank import main

if __name__ == "__main__":
    raise SystemExit(main())
