# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Typed echo-state-network construction on top of ``rclib`` (docs/PLAN.md section 5.2).

``EsnModel`` wraps one ``rclib.ESN`` with a random sparse reservoir and a ridge
readout. The readout consumes the reservoir state only and carries its own
bias term (``rclib``'s convention: no input pass-through and no manually
added second bias). Reservoir states are harvested one control sample at a
time with :meth:`EsnModel.advance`; the readout is fitted on harvested states
with :meth:`EsnModel.fit_readout`, which reproduces ``rclib``'s single-sequence
``fit`` bit for bit while allowing explicit per-episode resets.

UP-005: the pinned ``rclib`` seeds its reservoir weights but scales them with a
power iteration started from ``Eigen::Random()``, which draws from the C
library's ``rand()`` and is never re-seeded, so a second reservoir built in the
same process differs. Until the upstream fix lands, ``EsnModel`` re-seeds the C
library generator from the reservoir seed immediately before construction, which
restores bitwise in-process reproducibility (verified by ``tests/unit/test_esn.py``).
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.validation import require_finite

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["EsnConfig", "EsnModel", "ReadoutConfig", "ReservoirConfig", "ensure_single_thread", "require_single_thread"]

type Solver = Literal["auto", "cholesky", "dual_cholesky", "conjugate_gradient", "conjugate_gradient_implicit"]
_SOLVERS: Final = ("auto", "cholesky", "dual_cholesky", "conjugate_gradient", "conjugate_gradient_implicit")


@dataclass(frozen=True)
class ReservoirConfig:
    """Random sparse reservoir hyperparameters (mirrors ``rclib.reservoirs.RandomSparse``)."""

    n_neurons: int
    spectral_radius: float
    sparsity: float
    leak_rate: float
    input_scaling: float
    seed: int
    include_bias: bool = True

    def __post_init__(self) -> None:
        """Mirror ``rclib``'s constructor constraints so failures name the configuration key."""
        require_finite((self.spectral_radius, self.sparsity, self.leak_rate, self.input_scaling), "reservoir")
        checks = (
            (self.n_neurons > 0, "reservoir.n_neurons must be positive"),
            (self.spectral_radius >= 0, "reservoir.spectral_radius must be non-negative"),
            (0 <= self.sparsity <= 1, "reservoir.sparsity must be in [0, 1]"),
            (0 < self.leak_rate <= 1, "reservoir.leak_rate must be in (0, 1]"),
            (self.input_scaling >= 0, "reservoir.input_scaling must be non-negative"),
            (self.seed >= 0, "reservoir.seed must be non-negative"),
        )
        for ok, message in checks:
            if not ok:
                raise ValueError(message)


@dataclass(frozen=True)
class ReadoutConfig:
    """Ridge readout hyperparameters (mirrors ``rclib.readouts.Ridge``)."""

    alpha: float
    solver: Solver = "cholesky"
    include_bias: bool = True
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        """Regularization is non-negative and the solver is one ``rclib`` offers."""
        require_finite((self.alpha, self.tolerance), "readout")
        if self.alpha < 0:
            msg = "readout.alpha must be non-negative"
            raise ValueError(msg)
        if self.solver not in _SOLVERS:
            msg = f"readout.solver must be one of {list(_SOLVERS)}, got {self.solver!r}"
            raise ValueError(msg)
        if self.tolerance <= 0:
            msg = "readout.tolerance must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class EsnConfig:
    """Complete ESN description: one reservoir and one ridge readout."""

    reservoir: ReservoirConfig
    readout: ReadoutConfig


def require_single_thread() -> None:
    """Fail unless OpenMP is pinned to one thread; multi-threaded reductions are not bitwise reproducible."""
    if os.environ.get("OMP_NUM_THREADS") != "1":
        msg = "set OMP_NUM_THREADS=1 before importing rclib: reservoir computations must stay bitwise reproducible"
        raise RuntimeError(msg)


def ensure_single_thread() -> None:
    """Pin OpenMP to one thread for this process (command-line boundary), or fail if it is pinned otherwise.

    Commands call this before importing ``rclib`` and before collecting provenance,
    so a plain shell reproduces the documented results without exporting anything;
    an explicit different value is an error rather than silently overridden.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    require_single_thread()


def _reseed_c_library(seed: int) -> None:
    """UP-005 guard: re-seed the C library generator ``rclib``'s power iteration starts from."""
    ctypes.CDLL(None).srand(ctypes.c_uint(seed % (2**32)))


def _vector(values: NDArray[np.float64], name: str, width: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (width,):
        msg = f"{name} must have shape ({width},), got {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return array


class EsnModel:
    """One ``rclib`` ESN with fixed input/output widths and explicit state control."""

    def __init__(self, config: EsnConfig, *, input_dim: int, output_dim: int) -> None:
        if input_dim < 1 or output_dim < 1:
            msg = f"input_dim and output_dim must be >= 1, got {input_dim} and {output_dim}"
            raise ValueError(msg)
        require_single_thread()
        import rclib

        self._config = config
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._fitted = False
        reservoir = config.reservoir
        _reseed_c_library(reservoir.seed)
        self._model = rclib.ESN()
        self._model.add_reservoir(
            rclib.reservoirs.RandomSparse(
                n_neurons=reservoir.n_neurons,
                spectral_radius=reservoir.spectral_radius,
                sparsity=reservoir.sparsity,
                leak_rate=reservoir.leak_rate,
                input_scaling=reservoir.input_scaling,
                include_bias=reservoir.include_bias,
                seed=reservoir.seed,
            )
        )
        readout = config.readout
        self._model.set_readout(
            rclib.readouts.Ridge(
                readout.alpha, include_bias=readout.include_bias, solver=readout.solver, tolerance=readout.tolerance
            )
        )
        self._reservoir: Any = self._model.get_reservoir(0)
        # rclib exposes no public accessor for the readout object; the private attribute is stable in the pin.
        self._readout: Any = cast("Any", self._model)._cpp_model.getReadout()  # noqa: SLF001

    @property
    def config(self) -> EsnConfig:
        """The hyperparameters the model was built from."""
        return self._config

    @property
    def input_dim(self) -> int:
        """Width of the input vectors."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Width of the readout output."""
        return self._output_dim

    @property
    def n_neurons(self) -> int:
        """Reservoir size (width of the harvested states)."""
        return self._config.reservoir.n_neurons

    @property
    def fitted(self) -> bool:
        """Whether the readout has been fitted."""
        return self._fitted

    def reset(self) -> None:
        """Zero the reservoir state (episode boundary)."""
        self._model.reset_reservoirs()

    def state(self) -> NDArray[np.float64]:
        """A copy of the current reservoir state, shape ``(n_neurons,)``."""
        return np.array(self._reservoir.getState(), dtype=np.float64).reshape(-1)

    def advance(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Drive the reservoir with one input sample and return the new state (copy)."""
        row = _vector(u, "input", self._input_dim)[None, :]
        return np.array(self._reservoir.advance(row), dtype=np.float64).reshape(-1)

    def fit_readout(self, states: NDArray[np.float64], targets: NDArray[np.float64]) -> None:
        """Fit the ridge readout on harvested reservoir states (rows) and their targets."""
        x = np.asarray(states, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_neurons:  # noqa: PLR2004
            msg = f"states must have shape (M, {self.n_neurons}), got {x.shape}"
            raise ValueError(msg)
        if y.shape != (x.shape[0], self._output_dim):
            msg = f"targets must have shape ({x.shape[0]}, {self._output_dim}), got {y.shape}"
            raise ValueError(msg)
        if x.shape[0] < 1 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            msg = "states and targets must be non-empty and finite"
            raise ValueError(msg)
        self._readout.fit(np.ascontiguousarray(x), np.ascontiguousarray(y))
        self._fitted = True

    def fit_sequence(self, inputs: NDArray[np.float64], targets: NDArray[np.float64], *, washout_len: int) -> None:
        """``rclib``'s single-sequence fit (reset, drive, ridge on the rows after ``washout_len``); reference path."""
        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self._input_dim or y.shape != (x.shape[0], self._output_dim):  # noqa: PLR2004
            msg = f"inputs must be (M, {self._input_dim}) and targets (M, {self._output_dim}), got {x.shape}, {y.shape}"
            raise ValueError(msg)
        if not 0 <= washout_len < x.shape[0]:
            msg = f"washout_len must be in [0, {x.shape[0]}), got {washout_len}"
            raise ValueError(msg)
        self._model.reset_reservoirs()
        self._model.fit(x, y, washout_len=washout_len)
        self._fitted = True

    def readout(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the fitted readout on one reservoir state."""
        if not self._fitted:
            msg = "the readout has not been fitted"
            raise RuntimeError(msg)
        row = _vector(state, "state", self.n_neurons)[None, :]
        return np.array(self._readout.predict(row), dtype=np.float64).reshape(-1)

    def step(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance with one input and read out the prediction (``rclib``'s online prediction)."""
        return self.readout(self.advance(u))

    def predict_sequence(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reset, then predict a whole input sequence teacher-forced (``rclib``'s batch ``predict``)."""
        if not self._fitted:
            msg = "the readout has not been fitted"
            raise RuntimeError(msg)
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self._input_dim:  # noqa: PLR2004
            msg = f"inputs must have shape (M, {self._input_dim}), got {x.shape}"
            raise ValueError(msg)
        raw = cast("Sequence[Sequence[float]]", self._model.predict(x, reset_state_before_predict=True))
        prediction = np.array(raw, dtype=np.float64).reshape(x.shape[0], self._output_dim)
        return prediction
