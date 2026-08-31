# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Flattening nested records into dotted scalar keys (for tracking and study summaries)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["flatten_scalars"]


def flatten_scalars(prefix: str, value: object, out: dict[str, object]) -> None:
    """Flatten nested mappings/sequences into dotted keys; scalars keep their type."""
    if isinstance(value, dict):
        for key, item in cast("dict[str, object]", value).items():
            flatten_scalars(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        for i, item in enumerate(items):
            flatten_scalars(f"{prefix}.{i}" if prefix else str(i), item, out)
    else:
        out[prefix] = value
