# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the deterministic re-simulation subset, enumerated before the study runs.

M3MAN-011 re-simulates a sample from a clean checkout to show the runs reproduce
bitwise. Sampling only works if the sample is fixed in advance: chosen
afterwards, it could be the runs that happened to reproduce. So the subset is
the 24 models the budget subset measures, both frozen trackers, and five
scenarios taken as the first of each perturbation class in the study's own
order -- 240 model runs beside 60 replay runs, 300 in all.

Only re-simulation is sampled. Artifact verification and metric recomputation
still cover the complete evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.experiments.manual_timing import budget_entries

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_study import StudyManifest, StudyModel
    from arm_rc_ctrl.experiments.perturbations import RobustnessScenario

__all__ = [
    "RESIMULATION_CLASSES",
    "ResimulationSubset",
    "resimulation_subset",
]

RESIMULATION_CLASSES: Final = ("nominal", "posture_small", "posture_large", "force", "combined")
"""One scenario from each perturbation class, so the sample spans what the study varies."""


@dataclass(frozen=True)
class ResimulationSubset:
    """The runs a clean-checkout audit re-simulates, named before any of them exist."""

    models: tuple[StudyModel, ...]
    scenarios: tuple[RobustnessScenario, ...]
    trackers: tuple[str, ...]
    n_banks: int
    """Replay banks the models share: one per configuration at the fixed parent."""

    @property
    def n_rc_runs(self) -> int:
        """Model runs to re-simulate."""
        return len(self.models) * len(self.trackers) * len(self.scenarios)

    @property
    def n_replay_runs(self) -> int:
        """Replay baselines to re-simulate beside them."""
        return self.n_banks * len(self.trackers) * len(self.scenarios)

    @property
    def n_runs(self) -> int:
        """Every run in the sample."""
        return self.n_rc_runs + self.n_replay_runs

    def run_identities(self) -> tuple[tuple[str, str, str], ...]:
        """Every model run as ``(model label, scenario id, tracker)``, in frozen order.

        The audit needs to know which runs to redo, so each is named here rather
        than derived from whatever the execution happens to have produced.
        """
        return tuple(
            (model.label, scenario.scenario_id, tracker)
            for model in self.models
            for scenario in self.scenarios
            for tracker in self.trackers
        )


def resimulation_subset(
    manifest: StudyManifest,
    *,
    scenarios: Sequence[RobustnessScenario],
    trackers: Sequence[str],
    classes: Sequence[str] = RESIMULATION_CLASSES,
) -> ResimulationSubset:
    """Enumerate the frozen re-simulation sample over the study's own locked cases.

    The locked cases come from the caller, which already resolved them against
    the root the study is bound to: the development draws are named by the
    evaluation configuration rather than by the manifest, so rediscovering them
    here would invent a second path convention beside the one the study context
    owns.

    A class the study does not contain is an error rather than a quietly smaller
    sample, because the audit's strength is exactly what it covers.
    """
    chosen: list[RobustnessScenario] = []
    for wanted in classes:
        first = next((case for case in scenarios if case.kind == wanted), None)
        if first is None:
            msg = f"no scenario of class {wanted!r} among the {len(scenarios)} locked cases"
            raise ValueError(msg)
        chosen.append(first)
    models = budget_entries(manifest)
    banks = len({(model.configuration, model.arm.assignment) for model in models if model.arm.assignment is not None})
    return ResimulationSubset(models=models, scenarios=tuple(chosen), trackers=tuple(trackers), n_banks=banks)
