# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic paired AR(1) training augmentation (M3R-004; recovery plan section 5, decision D1).

Each synthetic episode adds a smooth, bounded, seeded position perturbation to
the cropped demonstration: seeded Gaussian innovations pass through an AR(1)
filter (stationary initialization ``z_0 ~ N(0, sigma^2 I)``), and the same
latent draw produces the **matched pair** of arms — the non-decaying arm keeps
the perturbation uncontracted through the movement, the contractive arm scales
it by the endpoint-distance envelope ``clip(d_tip/d_tip0, 0, 1)^gamma`` — so
both arms share episode seeds and amplitudes by construction, and an attempt
whose either variant violates a physical or configured bound is rejected as a
whole. Velocity is recomputed from each augmented position sequence with the
versioned derivative policy; velocity is never perturbed independently.

**Locked terminal taper (M3R-004 protocol detail, fixed and non-tuned):** both
arms multiply the same taper ``s(clip((t_z - t) / 0.2 s, 0, 1))`` with the
smoothstep ``s(x) = x^2 (3 - 2x)`` and ``t_z`` = dwell onset - 0.1 s. The
perturbation therefore decays C^1-continuously over the final 0.2 s of that
window and is exactly zero from 0.1 s before dwell onset through the episode
end — exact zero strictly before the dwell, zero throughout the dwell, and no
discontinuous step that would inject a velocity spike.

The AR(1) coefficient ``phi`` is an augmentation parameter and must never be
named or logged as the ESN spectral radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.recovery import TaskIntervals
from arm_rc_ctrl.scenario import ScenarioConfig, endpoint_positions, joint_limits

__all__ = [
    "APPROVED_GAMMA",
    "APPROVED_N_SYNTHETIC",
    "APPROVED_PHI",
    "APPROVED_SIGMA_RAD",
    "SEED_NAMESPACE",
    "TAPER_DURATION_S",
    "TAPER_ZERO_MARGIN_S",
    "AugmentationConfig",
    "AugmentationError",
    "AugmentationResult",
    "AugmentedEpisode",
    "EpisodeArrays",
    "Rejection",
    "contraction_envelope",
    "generate_augmentation",
    "terminal_taper",
]

SEED_NAMESPACE: Final = 415926535
"""Leading entropy word of every augmentation stream: ``default_rng([SEED_NAMESPACE, seed_bank, attempt])``.

Allocated 2026-09-03 for the augmentation namespace only. Deliberately not a
date: it is disjoint from every ``YYYYMMDD``-shaped seed (including M3's
confirmatory seeds 20260901-20260905) and from the separately allocated M3R
development and confirmatory evaluation namespaces; the three-word streams are
additionally disjoint from M3's two-word ``[seed, stream]`` scenario streams.
"""

TAPER_DURATION_S: Final = 0.2
"""Length of the locked smoothstep decay window (fixed, non-tuned)."""

TAPER_ZERO_MARGIN_S: Final = 0.1
"""The perturbation is exactly zero from this long before dwell onset (fixed, non-tuned)."""

APPROVED_N_SYNTHETIC: Final = frozenset({16, 32, 64})
"""Approved accepted-episode budgets (D1); anchor 64."""
APPROVED_SIGMA_RAD: Final = frozenset({0.01, 0.025, 0.05, 0.10})
"""Approved marginal perturbation scales in rad (D1); anchor 0.05. Constant across a configuration."""
APPROVED_PHI: Final = frozenset({0.98, 0.99, 0.995})
"""Approved AR(1) coefficients (plan section 5); anchor 0.99."""
APPROVED_GAMMA: Final = frozenset({0.5, 1.0, 2.0})
"""Approved envelope exponents (D1); anchor 1."""

type Family = Literal["non_decaying", "contractive"]
_FAMILIES: Final[tuple[Family, ...]] = ("non_decaying", "contractive")
_ENDPOINT_TOLERANCE_M: Final = 1e-9
"""Slack on the endpoint-radius bound (forward kinematics of a valid posture may exceed it by float dust)."""


class AugmentationError(RuntimeError):
    """The episode cannot be augmented under the locked protocol and configured budget."""


@dataclass(frozen=True)
class AugmentationConfig:
    """One augmentation configuration on the approved D1 grids."""

    n_synthetic: int
    """Accepted synthetic episodes (the original episode 0 is not counted)."""
    sigma_rad: float
    phi: float
    gamma: float
    seed_bank: int
    """Shared seed-bank identifier; both augmented study formulations reuse the same banks."""
    attempt_budget: int
    """Maximum seeded attempts before the configuration fails (never resample indefinitely)."""
    max_abs_perturbation_rad: float | None = None
    """Optional configured bound on ``|delta|``; violations reject the attempt, never clip silently."""

    def __post_init__(self) -> None:
        """Reject values outside the approved protocol grids and inconsistent budgets."""
        grids: tuple[tuple[str, float, frozenset[float] | frozenset[int]], ...] = (
            ("n_synthetic", self.n_synthetic, APPROVED_N_SYNTHETIC),
            ("sigma_rad", self.sigma_rad, APPROVED_SIGMA_RAD),
            ("phi", self.phi, APPROVED_PHI),
            ("gamma", self.gamma, APPROVED_GAMMA),
        )
        for name, value, approved in grids:
            if value not in approved:
                msg = f"{name} must be one of the approved values {sorted(approved)}, got {value!r}"
                raise ValueError(msg)
        if self.seed_bank < 0:
            msg = f"seed_bank must be non-negative, got {self.seed_bank}"
            raise ValueError(msg)
        if self.attempt_budget < self.n_synthetic:
            msg = f"attempt_budget must be at least n_synthetic={self.n_synthetic}, got {self.attempt_budget}"
            raise ValueError(msg)
        bound = self.max_abs_perturbation_rad
        if bound is not None and not (bound > 0 and bound < float("inf")):
            msg = f"max_abs_perturbation_rad must be positive and finite, got {bound!r}"
            raise ValueError(msg)


def _frozen(array: NDArray[np.float64]) -> NDArray[np.float64]:
    copy = np.ascontiguousarray(np.asarray(array, dtype=np.float64)).copy()
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class EpisodeArrays:
    """Position, recomputed velocity, and perturbation of one episode variant, with realized statistics."""

    q: NDArray[np.float64]
    dq: NDArray[np.float64]
    delta: NDArray[np.float64]
    delta_rms_rad: float
    delta_peak_rad: float

    def __post_init__(self) -> None:
        """Freeze the arrays and require consistent shapes."""
        for name in ("q", "dq", "delta"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        if self.q.shape != self.dq.shape or self.q.shape != self.delta.shape or self.q.ndim != 2:  # noqa: PLR2004
            msg = f"q, dq, delta must share one (N, dof) shape, got {self.q.shape}, {self.dq.shape}, {self.delta.shape}"
            raise ValueError(msg)


def _episode_arrays(q: NDArray[np.float64], dq: NDArray[np.float64], delta: NDArray[np.float64]) -> EpisodeArrays:
    return EpisodeArrays(
        q=q,
        dq=dq,
        delta=delta,
        delta_rms_rad=float(np.sqrt(np.mean(delta**2))),
        delta_peak_rad=float(np.max(np.abs(delta))),
    )


@dataclass(frozen=True)
class AugmentedEpisode:
    """One accepted synthetic episode: the matched non-decaying/contractive pair from one latent draw."""

    episode: int
    """1-based accepted-episode number (episode 0 is the original demonstration)."""
    attempt: int
    """1-based seeded attempt index; the stream is ``[SEED_NAMESPACE, seed_bank, attempt]``."""
    non_decaying: EpisodeArrays
    contractive: EpisodeArrays


@dataclass(frozen=True)
class Rejection:
    """One rejected attempt variant and why."""

    attempt: int
    family: Family
    reason: str


@dataclass(frozen=True)
class AugmentationResult:
    """Every accepted episode of one configuration, with rejection accounting and digests."""

    config: AugmentationConfig
    dwell_start_s: float
    derivative_method: str
    original: EpisodeArrays
    episodes: tuple[AugmentedEpisode, ...]
    rejections: tuple[Rejection, ...]
    attempts_used: int

    def digests(self) -> dict[str, str]:
        """SHA-256 of every generated array, keyed ``episode-XXX/<family>/<name>`` (plus ``original/...``)."""
        out: dict[str, str] = {}
        for name in ("q", "dq", "delta"):
            out[f"original/{name}"] = array_digest(getattr(self.original, name))
        for episode in self.episodes:
            for family in _FAMILIES:
                arrays: EpisodeArrays = getattr(episode, family)
                for name in ("q", "dq", "delta"):
                    out[f"episode-{episode.episode:03d}/{family}/{name}"] = array_digest(getattr(arrays, name))
        return out


def terminal_taper(t: NDArray[np.float64], dwell_start_s: float) -> NDArray[np.float64]:
    """The locked shared terminal taper evaluated on the task clock.

    ``1`` until the window opens, a C^1 smoothstep decay over
    :data:`TAPER_DURATION_S`, and exactly ``0`` from
    ``dwell_start_s - TAPER_ZERO_MARGIN_S`` onward.

    Raises
    ------
    AugmentationError
        If the movement is too short to contain the taper window.
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        msg = f"t must be a non-empty finite 1-D array, got shape {times.shape}"
        raise AugmentationError(msg)
    zero_from = dwell_start_s - TAPER_ZERO_MARGIN_S
    window_start = zero_from - TAPER_DURATION_S
    if window_start <= float(times[0]):
        msg = (
            f"the movement is too short for the locked terminal taper: the window would open at "
            f"{window_start!r} s, at or before the first sample {float(times[0])!r} s"
        )
        raise AugmentationError(msg)
    x = np.clip((zero_from - times) / TAPER_DURATION_S, 0.0, 1.0)
    taper = x * x * (3.0 - 2.0 * x)
    taper[times <= window_start] = 1.0
    taper[times >= zero_from] = 0.0
    return taper


def contraction_envelope(scenario: ScenarioConfig, q_ref: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """The target-distance envelope ``clip(d_tip/d_tip0, 0, 1)^gamma`` along the reference.

    Raises
    ------
    AugmentationError
        If the reference starts at the target (the normalizing distance vanishes).
    """
    tip = endpoint_positions(scenario, np.asarray(q_ref, dtype=np.float64))
    target = np.asarray(scenario.task.target, dtype=np.float64)
    diff = tip - target[None, :]
    distance = np.sqrt(np.sum(diff * diff, axis=1))
    d0 = float(distance[0])
    if not d0 > 0:
        msg = f"the reference already starts at the target (d_tip0 = {d0!r}); the envelope is undefined"
        raise AugmentationError(msg)
    return np.asarray(np.clip(distance / d0, 0.0, 1.0) ** gamma, dtype=np.float64)


def _latent(config: AugmentationConfig, attempt: int, n: int, dof: int) -> NDArray[np.float64]:
    """The seeded stationary AR(1) latent process of one attempt (independent per attempt)."""
    rng = np.random.default_rng([SEED_NAMESPACE, config.seed_bank, attempt])
    z = np.empty((n, dof), dtype=np.float64)
    z[0] = config.sigma_rad * rng.standard_normal(dof)
    innovations = config.sigma_rad * np.sqrt(1.0 - config.phi**2) * rng.standard_normal((n - 1, dof))
    for k in range(n - 1):
        z[k + 1] = config.phi * z[k] + innovations[k]
    return z


def _validity_problem(
    scenario: ScenarioConfig,
    q: NDArray[np.float64],
    dq: NDArray[np.float64],
    delta: NDArray[np.float64],
    bound: float | None,
) -> str | None:
    """The first violated physical or configured limit, or ``None`` when the variant is valid."""
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
        return "non-finite augmented sample"
    limits = joint_limits(scenario)
    lower = np.asarray(limits.lower, dtype=np.float64)
    upper = np.asarray(limits.upper, dtype=np.float64)
    if bool(np.any((q < lower) | (q > upper))):
        return "joint limits violated"
    speed = np.asarray(scenario.limits.velocity, dtype=np.float64)
    if bool(np.any(np.abs(dq) > speed)):
        return "velocity limit violated"
    tip = endpoint_positions(scenario, q)
    reach = np.sqrt(np.sum(tip * tip, axis=1))
    if bool(np.any(reach > scenario.limits.endpoint_radius + _ENDPOINT_TOLERANCE_M)):
        return "endpoint radius violated"
    if bound is not None and float(np.max(np.abs(delta))) > bound:
        return f"perturbation bound {bound!r} rad exceeded"
    return None


def generate_augmentation(
    t: NDArray[np.float64],
    q_ref: NDArray[np.float64],
    task: TaskIntervals,
    scenario: ScenarioConfig,
    config: AugmentationConfig,
    *,
    derivatives: DerivativeConfig,
) -> AugmentationResult:
    """Generate the accepted matched episode pairs of one configuration, deterministically.

    Attempts are seeded ``[SEED_NAMESPACE, seed_bank, attempt]`` for
    ``attempt = 1, 2, ...`` up to the configured budget; an attempt is accepted
    only when **both** family variants are physically valid, keeping the arms
    matched. Velocity is recomputed from each augmented position sequence with
    the given derivative policy (the recovery dataset's versioned policy).

    Raises
    ------
    AugmentationError
        If the attempt budget is exhausted before ``n_synthetic`` acceptances.
    """
    times = np.asarray(t, dtype=np.float64)
    positions = np.asarray(q_ref, dtype=np.float64)
    if positions.ndim != 2 or times.ndim != 1 or positions.shape[0] != times.shape[0]:  # noqa: PLR2004
        msg = f"t and q_ref must have shapes (N,) and (N, dof), got {times.shape} and {positions.shape}"
        raise AugmentationError(msg)
    period = scenario.timing.dt
    taper = terminal_taper(times, task.dwell[0])
    envelopes: dict[Family, NDArray[np.float64]] = {
        "non_decaying": taper,
        "contractive": contraction_envelope(scenario, positions, config.gamma) * taper,
    }
    dq_ref, _ = differentiate(positions, period, derivatives)
    original = _episode_arrays(positions, dq_ref, np.zeros_like(positions))

    accepted: list[AugmentedEpisode] = []
    rejections: list[Rejection] = []
    attempts_used = 0
    for attempt in range(1, config.attempt_budget + 1):
        if len(accepted) == config.n_synthetic:
            break
        attempts_used = attempt
        z = _latent(config, attempt, positions.shape[0], positions.shape[1])
        pair: dict[Family, EpisodeArrays] = {}
        failed = False
        for family in _FAMILIES:
            delta = envelopes[family][:, None] * z
            q_aug = positions + delta
            dq_aug, _ = differentiate(q_aug, period, derivatives)
            reason = _validity_problem(scenario, q_aug, dq_aug, delta, config.max_abs_perturbation_rad)
            if reason is not None:
                rejections.append(Rejection(attempt=attempt, family=family, reason=reason))
                failed = True
            else:
                pair[family] = _episode_arrays(q_aug, dq_aug, delta)
        if failed:
            continue
        accepted.append(
            AugmentedEpisode(
                episode=len(accepted) + 1,
                attempt=attempt,
                non_decaying=pair["non_decaying"],
                contractive=pair["contractive"],
            )
        )
    if len(accepted) < config.n_synthetic:
        msg = (
            f"attempt budget {config.attempt_budget} exhausted: accepted {len(accepted)} of "
            f"{config.n_synthetic} episodes with {len(rejections)} rejection(s)"
        )
        raise AugmentationError(msg)
    return AugmentationResult(
        config=config,
        dwell_start_s=task.dwell[0],
        derivative_method=derivatives.label,
        original=original,
        episodes=tuple(accepted),
        rejections=tuple(rejections),
        attempts_used=attempts_used,
    )
