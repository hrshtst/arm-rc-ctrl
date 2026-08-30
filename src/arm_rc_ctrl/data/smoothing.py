# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Zero-phase smoothing of offline demonstrations.

Offline data may be filtered forward and backward (``scipy.signal.sosfiltfilt``)
so that the result has no phase shift; the magnitude response is squared, so
the effective attenuation is that of a filter of twice the configured order.
Online estimators (``docs/PLAN.md`` section 5.4) must not use this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

__all__ = ["SmoothingConfig", "smooth"]

_MAX_ORDER: Final = 16


@dataclass(frozen=True)
class SmoothingConfig:
    """Smoothing method and parameters (``method = "none"`` disables filtering)."""

    method: Literal["none", "butterworth"] = "butterworth"
    cutoff_hz: float = 5.0
    order: int = 4
    """Order of the one-way Butterworth prototype; zero-phase application doubles it."""

    def __post_init__(self) -> None:
        """Validate cutoff and order."""
        if not (self.cutoff_hz > 0 and self.cutoff_hz < float("inf")):
            msg = f"cutoff_hz must be positive and finite, got {self.cutoff_hz!r}"
            raise ValueError(msg)
        if not 1 <= self.order <= _MAX_ORDER:
            msg = f"order must be in [1, {_MAX_ORDER}], got {self.order}"
            raise ValueError(msg)

    @property
    def label(self) -> str:
        """Record label of the method."""
        return "none" if self.method == "none" else "butterworth-zero-phase"

    def parameters(self) -> dict[str, float]:
        """Record parameters of the method."""
        if self.method == "none":
            return {}
        return {"order": float(self.order), "cutoff_hz": self.cutoff_hz}


def smooth(signal: NDArray[np.float64], sample_rate_hz: float, config: SmoothingConfig) -> NDArray[np.float64]:
    """Return a zero-phase smoothed copy of ``signal`` (samples along axis 0).

    Parameters
    ----------
    signal : NDArray[np.float64]
        Shape ``(N,)`` or ``(N, k)``; columns are filtered independently.
    sample_rate_hz : float
        Uniform sampling rate of ``signal``.
    config : SmoothingConfig
        Method and parameters.

    Raises
    ------
    ValueError
        If the signal is not finite, the cutoff is not below the Nyquist
        frequency, or the signal is too short for the filter's edge padding.
    """
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim not in (1, 2) or data.shape[0] == 0:
        msg = f"signal must have shape (N,) or (N, k) with N > 0, got {data.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(data)):
        msg = "signal contains non-finite values; smoothing never repairs data"
        raise ValueError(msg)
    if not (sample_rate_hz > 0 and sample_rate_hz < float("inf")):
        msg = f"sample_rate_hz must be positive and finite, got {sample_rate_hz!r}"
        raise ValueError(msg)
    if config.method == "none":
        return data.copy()
    nyquist = 0.5 * sample_rate_hz
    if config.cutoff_hz >= nyquist:
        msg = f"cutoff_hz {config.cutoff_hz} must be below the Nyquist frequency {nyquist} Hz"
        raise ValueError(msg)
    sos = cast(
        "NDArray[np.float64]", butter(config.order, config.cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    )
    # sosfiltfilt pads by odd extension; it needs strictly more samples than the pad length.
    padlen = 3 * (2 * sos.shape[0] + 1)
    if data.shape[0] <= padlen:
        msg = (
            f"signal has {data.shape[0]} samples; zero-phase filtering of order {config.order} needs more than {padlen}"
        )
        raise ValueError(msg)
    filtered = cast("NDArray[Any]", sosfiltfilt(sos, data, axis=0, padlen=padlen))
    return np.ascontiguousarray(filtered, dtype=np.float64)
