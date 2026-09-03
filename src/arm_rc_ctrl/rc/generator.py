# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The reservoir-computing target generator (docs/PLAN.md sections 5.1, 5.3, 5.4, and 8).

``RcTargetGenerator`` turns measured joint state into the desired joint state
for the low-level tracker. Every input is built from *actual* feedback with the
recipe's encoder — never from the generator's previous prediction. During the
priming interval :meth:`prime` drives the reservoir with the measured state
while the desired position stays at the initial posture; from then on
:meth:`step` reads out the next desired position and the causal estimator
supplies desired velocity and acceleration. A non-finite prediction, or one
outside the configured joint bounds, is rejected as a :class:`GeneratorError`.
In the residual mode (recovery plan section 6.1) the readout is an increment
``r`` and the commanded position is the composed ``q_measured + r``; the raw
increment is exposed as its own telemetry channel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.controllers.contracts import DesiredJointState, GeneratorError, RobotState, TargetGeneratorBase

if TYPE_CHECKING:
    from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator
    from arm_rc_ctrl.rc.esn import EsnModel
    from arm_rc_ctrl.rc.teacher_forcing import InputEncoder

__all__ = ["RcTargetGenerator"]


class RcTargetGenerator(TargetGeneratorBase):
    """ESN target generator with an input encoder, a fitted model, and a derivative estimator."""

    def __init__(
        self,
        model: EsnModel,
        encoder: InputEncoder,
        estimator: CausalDerivativeEstimator,
        *,
        position_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        output: str = "absolute",
    ) -> None:
        super().__init__()
        if output not in ("absolute", "increment"):
            msg = f"unsupported output {output!r}; supported: 'absolute', 'increment'"
            raise ValueError(msg)
        self._output = output
        if not model.fitted:
            msg = "the ESN readout must be fitted before it can generate targets"
            raise ValueError(msg)
        if encoder.input_dim != model.input_dim or model.output_dim != encoder.dof or estimator.dof != encoder.dof:
            msg = (
                f"incompatible widths: encoder input {encoder.input_dim}/dof {encoder.dof}, "
                f"model input {model.input_dim}/output {model.output_dim}, estimator dof {estimator.dof}"
            )
            raise ValueError(msg)
        self._model = model
        self._encoder = encoder
        self._estimator = estimator
        self._bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        if position_bounds is not None:
            lower = np.asarray(position_bounds[0], dtype=np.float64)
            upper = np.asarray(position_bounds[1], dtype=np.float64)
            if lower.shape != (encoder.dof,) or upper.shape != (encoder.dof,) or np.any(lower >= upper):
                msg = f"position_bounds must be two ({encoder.dof},) vectors with lower < upper"
                raise ValueError(msg)
            if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
                msg = "position_bounds must be finite"
                raise ValueError(msg)
            self._bounds = (lower, upper)
        self._hold: NDArray[np.float64] | None = None
        self._last: dict[str, NDArray[np.float64]] = {}

    @property
    def model(self) -> EsnModel:
        """The fitted ESN."""
        return self._model

    @property
    def hold_posture(self) -> NDArray[np.float64] | None:
        """The initial posture held during priming (``None`` before a reset)."""
        return self._hold

    @property
    def last(self) -> dict[str, NDArray[np.float64]]:
        """Telemetry of the last prime/step: input, reservoir state norm, prediction, derivative channels."""
        return dict(self._last)

    def _reset(self, initial_state: RobotState) -> None:
        self._model.reset()
        self._estimator.reset()
        self._hold = initial_state.q
        self._last = {}

    def _encode(self, state: RobotState, task_code: NDArray[np.float64]) -> NDArray[np.float64]:
        code = task_code if self._encoder.task_code_dim else None
        if self._encoder.task_code_dim and task_code.shape[0] != self._encoder.task_code_dim:
            msg = f"task_code must have {self._encoder.task_code_dim} entries, got {task_code.shape[0]}"
            raise GeneratorError(msg, category="shape")
        return self._encoder.encode(state.q, state.dq, code)

    def _telemetry(
        self,
        u: NDArray[np.float64],
        prediction: NDArray[np.float64],
        mode: float,
        *,
        increment: NDArray[np.float64] | None = None,
    ) -> None:
        estimate = self._estimator.last
        channels = estimate.channels() if estimate is not None else {}
        generating = mode >= 1.0
        state_norm = float(np.linalg.norm(self._model.state()))
        dof = self._encoder.dof
        self._last = {
            "esn_input": u,
            "esn_state_norm": np.array([state_norm]),
            "q_generated": prediction,
            "generating": np.array([mode]),
            # M3R task-time telemetry: the readout channel exists only while active; the warm-up
            # channels exist only while the readout is inactive (never a hold command as readout).
            "generator_output_q": np.array(prediction, dtype=np.float64) if generating else np.full(dof, np.nan),
            "generator_increment_q": (
                np.array(increment, dtype=np.float64) if generating and increment is not None else np.full(dof, np.nan)
            ),
            "warmup_state_norm": np.array([np.nan if generating else state_norm]),
            "warmup_esn_input": np.full(u.shape[0], np.nan) if generating else np.array(u, dtype=np.float64),
            **channels,
        }

    def _prime(self, state: RobotState, task_code: NDArray[np.float64]) -> None:
        if self._hold is None:  # pragma: no cover - reset() always sets it
            msg = "prime() called before reset()"
            raise GeneratorError(msg)
        u = self._encode(state, task_code)
        self._model.advance(u)
        self._estimator.update(state.t, self._hold)
        self._telemetry(u, self._hold, 0.0)

    def _step(self, state: RobotState, task_code: NDArray[np.float64]) -> DesiredJointState:
        u = self._encode(state, task_code)
        prediction = np.asarray(self._model.step(u), dtype=np.float64)
        if prediction.shape != (self._encoder.dof,):
            msg = f"the ESN produced a target of shape {prediction.shape}, expected ({self._encoder.dof},)"
            raise GeneratorError(msg, category="shape")
        if not np.all(np.isfinite(prediction)):
            msg = f"the ESN produced a non-finite target {prediction.tolist()} at t = {state.t} s"
            raise GeneratorError(msg, category="non_finite")
        command = np.asarray(state.q, dtype=np.float64) + prediction if self._output == "increment" else prediction
        if self._bounds is not None:
            lower, upper = self._bounds
            if np.any(command < lower) or np.any(command > upper):
                msg = f"the ESN target {command.tolist()} leaves the joint bounds at t = {state.t} s"
                raise GeneratorError(msg, category="bounds")
        estimate = self._estimator.update(state.t, command)
        self._telemetry(u, command, 1.0, increment=prediction if self._output == "increment" else None)
        return DesiredJointState(command, estimate.dq, estimate.ddq)
