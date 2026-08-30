# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Typed TOML configuration loading with strict validation."""

from __future__ import annotations

from arm_rc_ctrl.config.loader import ConfigError, from_mapping, load_config, to_mapping

__all__ = ["ConfigError", "from_mapping", "load_config", "to_mapping"]
