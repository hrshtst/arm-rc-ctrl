# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Deterministic robustness scenarios (``docs/PLAN.md`` section 9.2; M3-007, M3-008).

The five classes of the robustness protocol are generated from a locked
confirmatory protocol as a pure function of its levels and seeds, so every
method receives exactly the same scenario IDs, initial postures, and force
pulses:

1. ``nominal`` — the demonstrated initial posture, no disturbance;
2. ``posture_small`` — per seed, ``draws_per_seed`` joint-space offsets with a
   uniformly random direction and a Euclidean norm equal to the small level;
3. ``posture_large`` — the same construction at the larger held-out level
   (an independent random stream per class);
4. ``force`` — one finite-duration endpoint force pulse per locked direction;
5. ``combined`` — every small posture draw paired with one force direction
   (cycling through the locked directions in draw order).

A grid mode (unit directions times a magnitude) serves pilots and fixtures;
random mode is what the confirmatory suite uses. Offsets are checked against
the joint limits before any scenario is issued.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

import numpy as np

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.experiments.confirmatory import ForcePulseLevels, PostureLevels
from arm_rc_ctrl.experiments.disturbances import ForcePulse

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


__all__ = [
    "CLASS_ORDER",
    "DevelopmentRobustness",
    "PerturbationClass",
    "RobustnessLevels",
    "RobustnessScenario",
    "check_offsets",
    "force_scenarios",
    "load_development_robustness",
    "posture_grid",
    "posture_random",
    "robustness_scenarios",
]

type PerturbationClass = Literal["nominal", "posture_small", "posture_large", "force", "combined"]


class RobustnessLevels(Protocol):
    """What a protocol must provide: seeds and the posture/force levels (locked or development)."""

    @property
    def name(self) -> str: ...  # noqa: D102 - protocol member

    @property
    def seeds(self) -> tuple[int, ...]: ...  # noqa: D102 - protocol member

    @property
    def posture(self) -> PostureLevels: ...  # noqa: D102 - protocol member

    @property
    def force(self) -> ForcePulseLevels: ...  # noqa: D102 - protocol member


CLASS_ORDER: Final[tuple[PerturbationClass, ...]] = ("nominal", "posture_small", "posture_large", "force", "combined")
_CLASS_STREAM: Final = {"posture_small": 2, "posture_large": 3}
"""Sub-seed of each random posture class (the protocol class number), so classes draw independent streams."""


@dataclass(frozen=True)
class RobustnessScenario:
    """One scenario of the robustness suite, identical for every method."""

    scenario_id: str
    kind: PerturbationClass
    offset: tuple[float, ...]
    """Joint-space offset (rad) added to the demonstrated initial posture (zeros when unperturbed)."""
    seed: int | None = None
    draw: int | None = None
    magnitude_rad: float | None = None
    force_magnitude_n: float | None = None
    """Endpoint force pulse (polar description; ``None`` without a pulse)."""
    force_start_s: float | None = None
    force_duration_s: float | None = None
    direction_deg: float | None = None

    def __post_init__(self) -> None:
        """The ID is non-empty, the offset finite, and the pulse description complete or absent."""
        if not self.scenario_id.strip():
            msg = "scenario_id must not be empty"
            raise ValueError(msg)
        if not all(math.isfinite(v) for v in self.offset):
            msg = f"offset must be finite, got {self.offset}"
            raise ValueError(msg)
        parts = (self.force_magnitude_n, self.force_start_s, self.force_duration_s, self.direction_deg)
        if any(v is None for v in parts) and any(v is not None for v in parts[:3]):
            msg = "a force pulse needs magnitude, start, duration, and direction"
            raise ValueError(msg)
        self.pulse  # noqa: B018 - validates the pulse description through ForcePulse

    @property
    def pulse(self) -> ForcePulse | None:
        """The endpoint force pulse of this scenario, if any."""
        if self.force_magnitude_n is None or self.force_start_s is None or self.force_duration_s is None:
            return None
        return ForcePulse.from_polar(
            self.force_start_s, self.force_duration_s, self.force_magnitude_n, self.direction_deg or 0.0
        )

    def initial_q(self, nominal: Sequence[float]) -> tuple[float, ...]:
        """The perturbed initial posture."""
        if len(nominal) != len(self.offset):
            msg = f"nominal posture has {len(nominal)} joints, offset {len(self.offset)}"
            raise ValueError(msg)
        return tuple(float(a + b) for a, b in zip(nominal, self.offset, strict=True))


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        msg = "direction must be non-zero"
        raise ValueError(msg)
    return vector / norm


def posture_random(
    magnitude_rad: float, seed: int, dof: int, count: int, *, stream: int
) -> tuple[tuple[float, ...], ...]:
    """``count`` offsets of norm ``magnitude_rad`` with uniformly random directions from ``(seed, stream)``."""
    if magnitude_rad <= 0 or dof < 1 or count < 1:
        msg = "posture_random needs magnitude_rad > 0, dof >= 1, count >= 1"
        raise ValueError(msg)
    rng = np.random.default_rng([seed, stream])
    offsets: list[tuple[float, ...]] = []
    for _ in range(count):
        direction = _unit(rng.standard_normal(dof))
        offsets.append(tuple(float(v) for v in magnitude_rad * direction))
    return tuple(offsets)


def posture_grid(magnitude_rad: float, directions: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Offsets of norm ``magnitude_rad`` along the normalized ``directions`` (grid mode)."""
    if magnitude_rad <= 0 or not directions:
        msg = "posture_grid needs magnitude_rad > 0 and at least one direction"
        raise ValueError(msg)
    offsets: list[tuple[float, ...]] = []
    for direction in directions:
        unit = _unit(np.asarray(direction, dtype=np.float64))
        offsets.append(tuple(float(v) for v in magnitude_rad * unit))
    return tuple(offsets)


def check_offsets(
    offsets: Sequence[Sequence[float]],
    nominal: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> None:
    """Fail when a perturbed posture leaves the joint limits."""
    for index, offset in enumerate(offsets):
        if len(offset) != len(nominal):
            msg = f"offset {index} has {len(offset)} entries, expected {len(nominal)}"
            raise ValueError(msg)
        for joint, (q0, d, lo, hi) in enumerate(zip(nominal, offset, lower, upper, strict=True)):
            if not lo <= q0 + d <= hi:
                msg = f"offset {index} puts joint {joint} at {q0 + d:.4f} rad, outside [{lo}, {hi}]"
                raise ValueError(msg)


def _posture_class(
    kind: PerturbationClass, levels: PostureLevels, seeds: Sequence[int], dof: int
) -> tuple[RobustnessScenario, ...]:
    magnitude = levels.small_magnitude_rad if kind == "posture_small" else levels.large_magnitude_rad
    label = "small" if kind == "posture_small" else "large"
    scenarios: list[RobustnessScenario] = []
    for seed in seeds:
        offsets = posture_random(magnitude, seed, dof, levels.draws_per_seed, stream=_CLASS_STREAM[kind])
        scenarios.extend(
            RobustnessScenario(
                f"posture-{label}-{seed}-{draw:02d}", kind, offset, seed=seed, draw=draw, magnitude_rad=magnitude
            )
            for draw, offset in enumerate(offsets)
        )
    return tuple(scenarios)


def force_scenarios(levels: ForcePulseLevels, dof: int) -> tuple[RobustnessScenario, ...]:
    """Class 4: one pulse per locked direction at the nominal posture."""
    zeros = (0.0,) * dof
    return tuple(
        RobustnessScenario(
            f"force-{levels.magnitude_n:g}N-{direction:03.0f}deg",
            "force",
            zeros,
            force_magnitude_n=levels.magnitude_n,
            force_start_s=levels.start_s,
            force_duration_s=levels.duration_s,
            direction_deg=direction,
        )
        for direction in levels.directions_deg
    )


def robustness_scenarios(
    protocol: RobustnessLevels,
    *,
    nominal: Sequence[float],
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    classes: Sequence[PerturbationClass] = CLASS_ORDER,
) -> tuple[RobustnessScenario, ...]:
    """Every scenario of the requested classes, in protocol order, checked against the joint limits."""
    dof = len(nominal)
    small = _posture_class("posture_small", protocol.posture, protocol.seeds, dof)
    large = _posture_class("posture_large", protocol.posture, protocol.seeds, dof)
    forces = force_scenarios(protocol.force, dof)
    combined = tuple(
        RobustnessScenario(
            f"combined-{s.seed}-{s.draw:02d}-{f.direction_deg:03.0f}deg",
            "combined",
            s.offset,
            seed=s.seed,
            draw=s.draw,
            magnitude_rad=s.magnitude_rad,
            force_magnitude_n=f.force_magnitude_n,
            force_start_s=f.force_start_s,
            force_duration_s=f.force_duration_s,
            direction_deg=f.direction_deg,
        )
        for s, f in ((s, forces[i % len(forces)]) for i, s in enumerate(small))
    )
    by_class: dict[PerturbationClass, tuple[RobustnessScenario, ...]] = {
        "nominal": (RobustnessScenario("nominal", "nominal", (0.0,) * dof),),
        "posture_small": small,
        "posture_large": large,
        "force": forces,
        "combined": combined,
    }
    unknown = [c for c in classes if c not in by_class]
    if unknown:
        msg = f"unknown perturbation classes {unknown}"
        raise ValueError(msg)
    scenarios = tuple(s for kind in CLASS_ORDER if kind in classes for s in by_class[kind])
    if lower is not None and upper is not None:
        check_offsets([s.offset for s in scenarios], nominal, lower, upper)
    ids = [s.scenario_id for s in scenarios]
    if len(set(ids)) != len(ids):  # pragma: no cover - IDs are built from distinct seeds, draws, and directions
        msg = "scenario IDs must be unique"
        raise ValueError(msg)
    return scenarios


@dataclass(frozen=True)
class DevelopmentRobustness:
    """Development-only robustness levels and seeds (never the locked confirmatory ones)."""

    name: str
    scenario: Path
    seeds: tuple[int, ...]
    posture: PostureLevels
    force: ForcePulseLevels

    def __post_init__(self) -> None:
        """Name and seeds are valid."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            msg = f"seeds must be a non-empty list of distinct non-negative integers, got {self.seeds}"
            raise ValueError(msg)


def load_development_robustness(path: Path) -> DevelopmentRobustness:
    """Load and validate development robustness levels."""
    return load_config(path, DevelopmentRobustness)
