# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""The locked confirmatory protocol of a task (docs/PLAN.md sections 9.2 and 10).

M1-028 locks the perturbation levels selected by the pilot together with the
confirmatory seed list. M3 implements the generators and runs the suite; this
module only defines and checks the protocol so that nothing can drift from
the pilot's justification or reuse a development seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from arm_rc_ctrl.config import load_config
from arm_rc_ctrl.validation import require_finite

if TYPE_CHECKING:
    from collections.abc import Iterable

    from arm_rc_ctrl.experiments.perturbation_pilot import PilotReport

__all__ = ["ConfirmatoryProtocol", "ForcePulseLevels", "PostureLevels", "check_against_pilot", "load_confirmatory"]


@dataclass(frozen=True)
class PostureLevels:
    """Initial-posture perturbation classes 2 and 3 of the robustness protocol."""

    small_magnitude_rad: float
    """Euclidean norm (rad) of the small joint-space offsets drawn with a confirmatory seed."""
    large_magnitude_rad: float
    """Euclidean norm (rad) of the larger held-out joint-space offsets."""
    draws_per_seed: int
    """Number of random postures drawn per confirmatory seed and level."""

    def __post_init__(self) -> None:
        """Magnitudes are positive, ordered, and draws are counted."""
        require_finite((self.small_magnitude_rad, self.large_magnitude_rad), "posture")
        if not 0 < self.small_magnitude_rad <= self.large_magnitude_rad:
            msg = "posture magnitudes must satisfy 0 < small <= large"
            raise ValueError(msg)
        if self.draws_per_seed < 1:
            msg = f"draws_per_seed must be >= 1, got {self.draws_per_seed}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ForcePulseLevels:
    """Endpoint force pulse class 4 (and, combined with postures, class 5)."""

    magnitude_n: float
    start_s: float
    duration_s: float
    directions_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        """Magnitude and window are positive; directions are finite and distinct."""
        require_finite((self.magnitude_n, self.start_s, self.duration_s, *self.directions_deg), "force")
        if self.magnitude_n <= 0 or self.start_s < 0 or self.duration_s <= 0:
            msg = "force pulse needs magnitude_n > 0, start_s >= 0, duration_s > 0"
            raise ValueError(msg)
        if not self.directions_deg or len(set(self.directions_deg)) != len(self.directions_deg):
            msg = f"directions_deg must be non-empty and distinct, got {self.directions_deg}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ConfirmatoryProtocol:
    """Locked confirmatory perturbation levels and seeds of one task."""

    name: str
    scenario: Path
    locked: bool
    pilot_report: Path
    """The pilot report (JSON) that justifies the levels."""
    seeds: tuple[int, ...]
    """Confirmatory seeds; never used by development studies."""
    posture: PostureLevels
    force: ForcePulseLevels

    def __post_init__(self) -> None:
        """Name, seed list, and lock flag are valid."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            msg = f"seeds must be a non-empty list of distinct non-negative integers, got {self.seeds}"
            raise ValueError(msg)
        if not self.locked:
            msg = "a confirmatory protocol must be locked before it is used"
            raise ValueError(msg)

    def forbid_seeds(self, development_seeds: Iterable[int], label: str) -> None:
        """Fail when any confirmatory seed also appears in a development study."""
        shared = sorted(set(self.seeds) & set(development_seeds))
        if shared:
            msg = f"confirmatory seeds {shared} are also used by {label}"
            raise ValueError(msg)


def load_confirmatory(path: Path) -> ConfirmatoryProtocol:
    """Load and validate a locked confirmatory protocol."""
    return load_config(path, ConfirmatoryProtocol)


def check_against_pilot(protocol: ConfirmatoryProtocol, report: PilotReport) -> None:
    """Fail unless the locked levels equal the pilot's selection and the pilot is confirmatory-grade."""
    selection = report.selection
    pairs = (
        ("posture.small_magnitude_rad", protocol.posture.small_magnitude_rad, selection.posture_small_rad),
        ("posture.large_magnitude_rad", protocol.posture.large_magnitude_rad, selection.posture_large_rad),
        ("force.magnitude_n", protocol.force.magnitude_n, selection.force_magnitude_n),
        ("force.start_s", protocol.force.start_s, selection.force_start_s),
        ("force.duration_s", protocol.force.duration_s, selection.force_duration_s),
    )
    for name, locked, selected in pairs:
        if not math.isclose(locked, selected, rel_tol=0.0, abs_tol=0.0):
            msg = f"{name} = {locked!r} differs from the pilot selection {selected!r}"
            raise ValueError(msg)
    if protocol.force.directions_deg != selection.force_directions_deg:
        msg = f"force.directions_deg {protocol.force.directions_deg} differ from the pilot's"
        msg += f" {selection.force_directions_deg}"
        raise ValueError(msg)
    if report.provenance.project_dirty:
        msg = "the pilot report was produced from a dirty worktree and cannot justify a lock"
        raise ValueError(msg)
    if protocol.scenario.name != Path(report.scenario_file).name:
        msg = f"protocol scenario {protocol.scenario.name} differs from the pilot's {report.scenario_file}"
        raise ValueError(msg)
