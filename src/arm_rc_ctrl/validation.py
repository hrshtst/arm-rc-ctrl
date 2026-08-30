# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Small validators shared by provenance, dependency, and artifact-record schemas."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime, timedelta

__all__ = [
    "COMMIT_HEX_LENGTH",
    "MD5_HEX_LENGTH",
    "SHA256_HEX_LENGTH",
    "is_hex",
    "require_finite",
    "validate_utc_timestamp",
]

COMMIT_HEX_LENGTH = 40
SHA256_HEX_LENGTH = 64
MD5_HEX_LENGTH = 32


def is_hex(value: str, length: int) -> bool:
    """Whether ``value`` is exactly ``length`` lowercase hexadecimal characters."""
    return len(value) == length and all(c in "0123456789abcdef" for c in value)


def validate_utc_timestamp(value: str, field: str = "created_at") -> datetime:
    """Require an ISO 8601 timestamp that is timezone-aware, in UTC, at second precision."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"{field} is not an ISO 8601 timestamp: {value!r}"
        raise ValueError(msg) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        msg = f"{field} must be timezone-aware in UTC, got {value!r}"
        raise ValueError(msg)
    if parsed.isoformat(timespec="seconds") != value:
        msg = f"{field} must have second precision with a +00:00 offset, got {value!r}"
        raise ValueError(msg)
    return parsed


def require_finite(values: Iterable[float], field: str) -> None:
    """Reject NaN or infinite entries."""
    for i, value in enumerate(values):
        if not math.isfinite(value):
            msg = f"{field}[{i}] must be finite, got {value!r}"
            raise ValueError(msg)
