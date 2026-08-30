# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Training-only normalization statistics (``docs/PLAN.md`` sections 5.1 and 7.3).

Statistics are fitted on an explicit set of training rows and persisted in
the processed record / model recipe as :class:`~arm_rc_ctrl.data.records.Normalization`.
Columns whose standard deviation is at or below ``near_zero`` get scale 1.0,
and their indices are recorded so the replacement is visible in reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.records import ChannelStats, Normalization

__all__ = ["DEFAULT_NEAR_ZERO", "Normalizer", "fit_normalization"]

DEFAULT_NEAR_ZERO: Final = 1e-8
"""Standard deviations at or below this value are replaced by 1.0."""


def fit_normalization(
    arrays: Mapping[str, NDArray[Any]],
    channels: tuple[str, ...],
    *,
    fitted_on: tuple[str, ...],
    training_rows: NDArray[np.bool_],
    near_zero: float = DEFAULT_NEAR_ZERO,
) -> Normalization:
    """Fit per-column mean and scale of ``channels`` over ``training_rows`` only.

    Parameters
    ----------
    arrays : Mapping[str, NDArray]
        Canonical dataset arrays (``(N, k)`` per channel).
    channels : tuple[str, ...]
        Channels to normalize (a subset of the normalizable arrays).
    fitted_on : tuple[str, ...]
        Artifact IDs of the datasets the rows come from (recorded, never inferred).
    training_rows : NDArray[np.bool_]
        Boolean mask of length ``N`` selecting the training rows. Evaluation
        rows never influence the statistics.
    near_zero : float, optional
        Threshold at or below which a standard deviation is replaced by 1.0.
    """
    mask = np.asarray(training_rows)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        msg = "training_rows must be a 1-D boolean mask"
        raise ValueError(msg)
    if not mask.any():
        msg = "training_rows selects no samples"
        raise ValueError(msg)
    if not (near_zero >= 0 and near_zero < float("inf")):
        msg = f"near_zero must be non-negative and finite, got {near_zero!r}"
        raise ValueError(msg)
    stats: dict[str, ChannelStats] = {}
    for name in channels:
        if name not in arrays:
            msg = f"channel {name!r} is not in the dataset"
            raise ValueError(msg)
        data = np.asarray(arrays[name], dtype=np.float64)
        if data.ndim != 2 or data.shape[0] != mask.shape[0]:  # noqa: PLR2004
            msg = f"channel {name!r} must have shape ({mask.shape[0]}, k), got {data.shape}"
            raise ValueError(msg)
        if data.shape[1] == 0:
            continue  # nothing to normalize (e.g. task_code with dimension 0)
        rows = data[mask]
        if not np.all(np.isfinite(rows)):
            msg = f"channel {name!r} has non-finite training values"
            raise ValueError(msg)
        mean = np.mean(rows, axis=0)
        std = np.std(rows, axis=0)
        replaced = tuple(int(i) for i in np.flatnonzero(std <= near_zero))
        scale = std.copy()
        scale[list(replaced)] = 1.0
        stats[name] = ChannelStats(
            mean=tuple(float(v) for v in mean),
            scale=tuple(float(v) for v in scale),
            replaced_near_zero=replaced,
        )
    return Normalization(fitted_on=fitted_on, channels=stats)


@dataclass(frozen=True)
class Normalizer:
    """Apply and invert recorded normalization statistics."""

    normalization: Normalization

    def transform(self, name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``(values - mean) / scale`` for channel ``name``."""
        mean, scale = self._params(name, values)
        return np.ascontiguousarray((values - mean) / scale, dtype=np.float64)

    def inverse(self, name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``values * scale + mean`` for channel ``name``."""
        mean, scale = self._params(name, values)
        return np.ascontiguousarray(values * scale + mean, dtype=np.float64)

    def _params(self, name: str, values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        try:
            stats = self.normalization.channels[name]
        except KeyError:
            msg = f"no normalization statistics for channel {name!r}"
            raise KeyError(msg) from None
        width = len(stats.mean)
        if values.ndim not in (1, 2) or values.shape[-1] != width:
            msg = f"channel {name!r} expects {width} columns, got shape {values.shape}"
            raise ValueError(msg)
        return np.asarray(stats.mean, dtype=np.float64), np.asarray(stats.scale, dtype=np.float64)
