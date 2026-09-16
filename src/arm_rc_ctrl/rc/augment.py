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

**Manual protocol (M3MAN-006, manual plan section 11; a separate, versioned
arm):** the manual demonstrations are complete recordings whose first sample
and final dwell are the recorded boundaries, so their envelope is exactly zero
on the first sample, ramps in with a C^1 smoothstep over the frozen
:data:`MANUAL_RAMP_DURATION_S`, and keeps the same terminal taper. Its streams
carry a parent term (clarification I8), so two demonstrations of one seed bank
never draw the same latent process. The inherited recovery families above are
untouched by it: they still perturb their first sample.

The AR(1) coefficient ``phi`` is an augmentation parameter and must never be
named or logged as the ESN spectral radius.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.recovery import TaskIntervals
from arm_rc_ctrl.scenario import (
    LimitsConfig,
    RobotConfig,
    ScenarioConfig,
    robot_endpoint_positions,
    robot_joint_limits,
)

__all__ = [
    "APPROVED_GAMMA",
    "APPROVED_N_SYNTHETIC",
    "APPROVED_PHI",
    "APPROVED_SIGMA_RAD",
    "MANUAL_ATTEMPT_BUDGET",
    "MANUAL_ENVELOPE_VERSION",
    "MANUAL_GAMMA",
    "MANUAL_N_SYNTHETIC",
    "MANUAL_PHI",
    "MANUAL_RAMP_DURATION_S",
    "MANUAL_SEED_NAMESPACE",
    "MANUAL_SEED_WORD",
    "MANUAL_SIGMA_RAD",
    "SEED_NAMESPACE",
    "TAPER_DURATION_S",
    "TAPER_ZERO_MARGIN_S",
    "AugmentationConfig",
    "AugmentationError",
    "AugmentationResult",
    "AugmentedEpisode",
    "EndpointTarget",
    "EpisodeArrays",
    "ManualAugmentationConfig",
    "ManualAugmentationResult",
    "ManualAugmentedEpisode",
    "ManualBudgetExhaustedError",
    "Rejection",
    "TaskGeometry",
    "contraction_envelope",
    "entropy_word",
    "generate_augmentation",
    "generate_manual_augmentation",
    "manual_boundary_envelope",
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


class EndpointTarget(Protocol):
    """The only part of a task section the augmentation reads: the endpoint target."""

    @property
    def target(self) -> tuple[float, ...]: ...  # noqa: D102 - protocol member


class TaskGeometry(Protocol):
    """Structural view of a task configuration: the arm, its limits, and the endpoint target.

    Both :class:`~arm_rc_ctrl.scenario.ScenarioConfig` and the manual protocol's
    :class:`~arm_rc_ctrl.data.manual_scenario.ManualScenarioConfig` satisfy it.
    Their ``task`` and ``timing`` sections are different classes carrying
    different rules, but the generator needs neither: it reads the robot (for
    forward kinematics), the shared limits, and the target.
    """

    @property
    def robot(self) -> RobotConfig: ...  # noqa: D102 - protocol member

    @property
    def limits(self) -> LimitsConfig: ...  # noqa: D102 - protocol member

    @property
    def task(self) -> EndpointTarget: ...  # noqa: D102 - protocol member


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


def contraction_envelope(scenario: TaskGeometry, q_ref: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """The target-distance envelope ``clip(d_tip/d_tip0, 0, 1)^gamma`` along the reference.

    Raises
    ------
    AugmentationError
        If the reference starts at the target (the normalizing distance vanishes).
    """
    tip = robot_endpoint_positions(scenario.robot, np.asarray(q_ref, dtype=np.float64))
    target = np.asarray(scenario.task.target, dtype=np.float64)
    diff = tip - target[None, :]
    distance = np.sqrt(np.sum(diff * diff, axis=1))
    d0 = float(distance[0])
    if not d0 > 0:
        msg = f"the reference already starts at the target (d_tip0 = {d0!r}); the envelope is undefined"
        raise AugmentationError(msg)
    return np.asarray(np.clip(distance / d0, 0.0, 1.0) ** gamma, dtype=np.float64)


def _ar1(rng: np.random.Generator, sigma_rad: float, phi: float, n: int, dof: int) -> NDArray[np.float64]:
    """The stationary AR(1) latent process of one seeded stream (``z_0 ~ N(0, sigma^2 I)``)."""
    z = np.empty((n, dof), dtype=np.float64)
    z[0] = sigma_rad * rng.standard_normal(dof)
    innovations = sigma_rad * np.sqrt(1.0 - phi**2) * rng.standard_normal((n - 1, dof))
    for k in range(n - 1):
        z[k + 1] = phi * z[k] + innovations[k]
    return z


def _latent(config: AugmentationConfig, attempt: int, n: int, dof: int) -> NDArray[np.float64]:
    """The seeded stationary AR(1) latent process of one attempt (independent per attempt)."""
    rng = np.random.default_rng([SEED_NAMESPACE, config.seed_bank, attempt])
    return _ar1(rng, config.sigma_rad, config.phi, n, dof)


def _validity_problem(
    scenario: TaskGeometry,
    q: NDArray[np.float64],
    dq: NDArray[np.float64],
    delta: NDArray[np.float64],
    bound: float | None,
) -> str | None:
    """The first violated physical or configured limit, or ``None`` when the variant is valid."""
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
        return "non-finite augmented sample"
    limits = robot_joint_limits(scenario.robot, scenario.limits)
    lower = np.asarray(limits.lower, dtype=np.float64)
    upper = np.asarray(limits.upper, dtype=np.float64)
    if bool(np.any((q < lower) | (q > upper))):
        return "joint limits violated"
    speed = np.asarray(scenario.limits.velocity, dtype=np.float64)
    if bool(np.any(np.abs(dq) > speed)):
        return "velocity limit violated"
    tip = robot_endpoint_positions(scenario.robot, q)
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


# --- the manual protocol's contractive arm (M3MAN-006; manual plan section 11) --------------

MANUAL_SEED_NAMESPACE: Final = "task_1a_manual_v1/contractive/v1"
"""Frozen namespace label of every manual contractive stream (owner-frozen 2026-09-16).

A label rather than a number: the stream's leading entropy word is derived from
it by :func:`entropy_word`, so the namespace stays readable in records while the
word stays disjoint from every allocated seed.
"""

_ENTROPY_BYTES: Final = 8
"""Bytes of the SHA-256 digest that make up a derived entropy word."""


def entropy_word(text: str) -> int:
    """The stable entropy word of a label: the leading eight bytes of its SHA-256.

    Independent of the interpreter's hash randomization, of worker scheduling,
    and of the order parents are generated in, so a stream depends on its
    labels alone.

    Raises
    ------
    ValueError
        If the label is empty.
    """
    if not text.strip():
        msg = "an entropy word needs a non-empty label"
        raise ValueError(msg)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:_ENTROPY_BYTES], "big")


MANUAL_SEED_WORD: Final = entropy_word(MANUAL_SEED_NAMESPACE)
"""Leading entropy word of every manual stream: ``[MANUAL_SEED_WORD, parent, seed_bank, attempt]`` (I8)."""

MANUAL_ENVELOPE_VERSION: Final = "boundary_ramp_v1"
"""Version label of the manual envelope, recorded with every bank."""
MANUAL_RAMP_DURATION_S: Final = 0.5
"""Frozen ramp-in duration measured from task time zero (I14; fixed, non-tuned)."""
MANUAL_N_SYNTHETIC: Final = 9
"""Accepted synthetic episodes per parent (frozen)."""
MANUAL_SIGMA_RAD: Final = 0.05
"""Marginal perturbation scale in rad (frozen)."""
MANUAL_PHI: Final = 0.99
"""AR(1) coefficient (frozen); never the ESN spectral radius."""
MANUAL_GAMMA: Final = 1.0
"""Contraction-envelope exponent (frozen)."""
MANUAL_ATTEMPT_BUDGET: Final = 36
"""Seeded attempts a parent may use before its bank is reported as a failure (frozen)."""


def manual_boundary_envelope(
    t: NDArray[np.float64], dwell_start_s: float, *, ramp_duration_s: float = MANUAL_RAMP_DURATION_S
) -> NDArray[np.float64]:
    """The versioned manual boundary envelope evaluated on the task clock.

    Exactly ``0`` on the first recorded sample, a C^1 smoothstep ramp in over
    ``ramp_duration_s``, ``1`` in between, and the inherited
    :func:`terminal_taper` at the end — so the recorded start and the recorded
    final dwell are preserved exactly and no step injects a velocity spike.

    Raises
    ------
    AugmentationError
        If the dwell onset lies outside the recording, if the movement is too
        short for the terminal taper, or if the recording has no room for the
        ramp before that taper window opens (insufficient transition support).
    """
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        msg = f"t must be a non-empty finite 1-D array, got shape {times.shape}"
        raise AugmentationError(msg)
    if not (ramp_duration_s > 0 and ramp_duration_s < float("inf")):
        msg = f"ramp_duration_s must be positive and finite, got {ramp_duration_s!r}"
        raise AugmentationError(msg)
    start = float(times[0])
    if not start < dwell_start_s <= float(times[-1]):
        msg = f"the dwell onset {dwell_start_s!r} s must lie inside the recording [{start!r}, {float(times[-1])!r}] s"
        raise AugmentationError(msg)
    taper = terminal_taper(times, dwell_start_s)
    window_start = dwell_start_s - TAPER_ZERO_MARGIN_S - TAPER_DURATION_S
    ramp_end = start + ramp_duration_s
    if ramp_end > window_start:
        msg = (
            f"the recording has insufficient transition support: the {ramp_duration_s!r} s ramp from the first "
            f"sample ends at {ramp_end!r} s, after the terminal taper window opens at {window_start!r} s"
        )
        raise AugmentationError(msg)
    x = np.clip((times - start) / ramp_duration_s, 0.0, 1.0)
    ramp = x * x * (3.0 - 2.0 * x)
    ramp[times >= ramp_end] = 1.0
    return np.asarray(ramp * taper, dtype=np.float64)


@dataclass(frozen=True)
class ManualAugmentationConfig:
    """The manual protocol's frozen contractive configuration for one parent (owner-frozen 2026-09-16)."""

    parent: str
    """Stable identifier of the parent demonstration (its processed artifact ID); part of every stream."""
    seed_bank: int
    """Shared seed-bank identifier; banks are reused across reservoir configurations."""
    n_synthetic: int = MANUAL_N_SYNTHETIC
    sigma_rad: float = MANUAL_SIGMA_RAD
    phi: float = MANUAL_PHI
    gamma: float = MANUAL_GAMMA
    attempt_budget: int = MANUAL_ATTEMPT_BUDGET
    ramp_duration_s: float = MANUAL_RAMP_DURATION_S

    def __post_init__(self) -> None:
        """Everything but the parent and the seed bank is frozen: deviations are refused, never applied."""
        frozen: tuple[tuple[str, float, float], ...] = (
            ("n_synthetic", self.n_synthetic, MANUAL_N_SYNTHETIC),
            ("sigma_rad", self.sigma_rad, MANUAL_SIGMA_RAD),
            ("phi", self.phi, MANUAL_PHI),
            ("gamma", self.gamma, MANUAL_GAMMA),
            ("attempt_budget", self.attempt_budget, MANUAL_ATTEMPT_BUDGET),
            ("ramp_duration_s", self.ramp_duration_s, MANUAL_RAMP_DURATION_S),
        )
        for name, value, expected in frozen:
            if value != expected:
                msg = f"{name} is frozen at {expected!r} for the manual protocol, got {value!r}"
                raise ValueError(msg)
        if not self.parent.strip():
            msg = "parent must identify the demonstration the bank is grown from"
            raise ValueError(msg)
        if self.seed_bank < 0:
            msg = f"seed_bank must be non-negative, got {self.seed_bank}"
            raise ValueError(msg)


class ManualBudgetExhaustedError(AugmentationError):
    """A manual parent could not fill its bank within the frozen attempt budget."""

    def __init__(self, message: str, *, rejections: tuple[Rejection, ...], attempts_used: int, accepted: int) -> None:
        """Carry the complete rejection accounting, so the failure stays reviewable evidence."""
        super().__init__(message)
        self.rejections = rejections
        self.attempts_used = attempts_used
        self.accepted = accepted


@dataclass(frozen=True)
class ManualAugmentedEpisode:
    """One accepted manual synthetic episode; the manual study has the contractive arm only."""

    episode: int
    """1-based accepted-episode number (episode 0 is the recorded demonstration)."""
    attempt: int
    """1-based seeded attempt index; the stream is ``[MANUAL_SEED_WORD, parent, seed_bank, attempt]``."""
    arrays: EpisodeArrays


@dataclass(frozen=True)
class ManualAugmentationResult:
    """Every accepted episode of one parent, with rejection accounting and digests."""

    config: ManualAugmentationConfig
    dwell_start_s: float
    period_s: float
    derivative_method: str
    envelope_version: str
    seed_namespace: str
    original: EpisodeArrays
    episodes: tuple[ManualAugmentedEpisode, ...]
    rejections: tuple[Rejection, ...]
    attempts_used: int

    def __post_init__(self) -> None:
        """The accepted episodes are numbered in acceptance order and none shares an attempt."""
        numbers = [episode.episode for episode in self.episodes]
        attempts = [episode.attempt for episode in self.episodes]
        if numbers != list(range(1, len(self.episodes) + 1)) or attempts != sorted(set(attempts)):
            msg = f"episodes must be numbered 1..n in increasing attempt order, got {numbers} and {attempts}"
            raise ValueError(msg)

    def digests(self) -> dict[str, str]:
        """SHA-256 of every generated array, keyed ``episode-XXX/<name>`` (plus ``parent/...``)."""
        out: dict[str, str] = {}
        for name in ("q", "dq", "delta"):
            out[f"parent/{name}"] = array_digest(getattr(self.original, name))
        for episode in self.episodes:
            for name in ("q", "dq", "delta"):
                out[f"episode-{episode.episode:03d}/{name}"] = array_digest(getattr(episode.arrays, name))
        return out


def _manual_latent(config: ManualAugmentationConfig, attempt: int, n: int, dof: int) -> NDArray[np.float64]:
    """The seeded AR(1) latent process of one manual attempt, separated per parent (I8)."""
    stream = [MANUAL_SEED_WORD, entropy_word(config.parent), config.seed_bank, attempt]
    return _ar1(np.random.default_rng(stream), config.sigma_rad, config.phi, n, dof)


def generate_manual_augmentation(
    t: NDArray[np.float64],
    q_ref: NDArray[np.float64],
    scenario: TaskGeometry,
    config: ManualAugmentationConfig,
    *,
    dwell_start_s: float,
    period_s: float,
    derivatives: DerivativeConfig,
) -> ManualAugmentationResult:
    """Generate one parent's accepted contractive episodes, deterministically.

    Attempts are seeded ``[MANUAL_SEED_WORD, parent, seed_bank, attempt]`` for
    ``attempt = 1, 2, ...`` up to the frozen budget. The manual study has no
    non-decaying arm, so an attempt is accepted when its contractive variant is
    valid at the task's own limits; a violation rejects the attempt with its
    reason and is never clipped or retimed. Velocity is recomputed from each
    augmented position sequence with the parent's derivative policy.

    Raises
    ------
    AugmentationError
        If the arrays disagree, or the recording cannot carry the ramp and the
        terminal taper.
    ManualBudgetExhaustedError
        If the attempt budget is exhausted before ``n_synthetic`` acceptances;
        the error carries every rejection.
    """
    times = np.asarray(t, dtype=np.float64)
    positions = np.asarray(q_ref, dtype=np.float64)
    if positions.ndim != 2 or times.ndim != 1 or positions.shape[0] != times.shape[0]:  # noqa: PLR2004
        msg = f"t and q_ref must have shapes (N,) and (N, dof), got {times.shape} and {positions.shape}"
        raise AugmentationError(msg)
    boundary = manual_boundary_envelope(times, dwell_start_s, ramp_duration_s=config.ramp_duration_s)
    envelope = contraction_envelope(scenario, positions, config.gamma) * boundary
    dq_ref, _ = differentiate(positions, period_s, derivatives)
    original = _episode_arrays(positions, dq_ref, np.zeros_like(positions))

    accepted: list[ManualAugmentedEpisode] = []
    rejections: list[Rejection] = []
    attempts_used = 0
    for attempt in range(1, config.attempt_budget + 1):
        if len(accepted) == config.n_synthetic:
            break
        attempts_used = attempt
        z = _manual_latent(config, attempt, positions.shape[0], positions.shape[1])
        delta = envelope[:, None] * z
        q_aug = positions + delta
        dq_aug, _ = differentiate(q_aug, period_s, derivatives)
        reason = _validity_problem(scenario, q_aug, dq_aug, delta, None)
        if reason is not None:
            rejections.append(Rejection(attempt=attempt, family="contractive", reason=reason))
            continue
        accepted.append(
            ManualAugmentedEpisode(
                episode=len(accepted) + 1, attempt=attempt, arrays=_episode_arrays(q_aug, dq_aug, delta)
            )
        )
    if len(accepted) < config.n_synthetic:
        msg = (
            f"attempt budget {config.attempt_budget} exhausted for parent {config.parent}: accepted "
            f"{len(accepted)} of {config.n_synthetic} episodes with {len(rejections)} rejection(s)"
        )
        raise ManualBudgetExhaustedError(
            msg, rejections=tuple(rejections), attempts_used=attempts_used, accepted=len(accepted)
        )
    return ManualAugmentationResult(
        config=config,
        dwell_start_s=dwell_start_s,
        period_s=period_s,
        derivative_method=derivatives.label,
        envelope_version=MANUAL_ENVELOPE_VERSION,
        seed_namespace=MANUAL_SEED_NAMESPACE,
        original=original,
        episodes=tuple(accepted),
        rejections=tuple(rejections),
        attempts_used=attempts_used,
    )
