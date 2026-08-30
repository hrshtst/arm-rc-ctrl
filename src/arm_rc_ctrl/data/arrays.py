# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic array digests and ``.npz`` payload I/O."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.provenance import sha256_bytes

__all__ = ["array_digest", "load_npz", "save_npz"]


def array_digest(array: NDArray[Any]) -> str:
    """SHA-256 over dtype, shape, and the C-contiguous bytes of an array."""
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}|{contiguous.shape}|".encode()
    return sha256_bytes(header + contiguous.tobytes())


def save_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    """Write arrays to an uncompressed ``.npz`` archive (never pickled)."""
    # numpy's savez stub cannot express typed **kwargs alongside allow_pickle.
    np.savez(path, **cast("dict[str, Any]", dict(arrays)))


def load_npz(path: Path, expected: tuple[str, ...]) -> dict[str, NDArray[Any]]:
    """Load an ``.npz`` archive that must contain exactly ``expected`` arrays."""
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(archive.files)
        if set(names) != set(expected):
            msg = f"{path}: expected arrays {sorted(expected)}, found {sorted(names)}"
            raise ValueError(msg)
        return {name: cast("NDArray[Any]", archive[name]) for name in expected}
