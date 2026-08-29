# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Process-wide test settings applied before any domain library is imported."""

from __future__ import annotations

import os

# skelarm imports PyQt6 at package import; never require a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# rclib may use OpenMP; one thread keeps reductions bitwise deterministic.
os.environ.setdefault("OMP_NUM_THREADS", "1")
