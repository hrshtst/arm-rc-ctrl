# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: what the manual-demonstration study will cost, projected from a measured subset.

The projection multiplies the study's own counts by the means a smoke check
measured. The counts are derived from the frozen study rather than written
here: six configurations of 31 arms are 186 models, and because replay is
driven through the derivative policy of the configuration it is paired against
(owner decision 2026-09-17), each configuration keeps its own bank per parent,
so the replay side is 60 banks and not the three warm-up banks the repeated
demonstration pilot had.

It is a projection and not a bound. Means measured on one configuration are
multiplied by the maximum run counts, while reservoir sizes, recording lengths
and storage overhead vary; a run that aborts early costs less, and an
infeasible model still costs its fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_study import ARM_COUNT, CONFIGURATION_COUNT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.experiments.manual_evaluation import ManualModelTiming, ManualRunTiming

__all__ = [
    "PARENT_COUNT",
    "TIMING_SCHEMA_VERSION",
    "ManualRunStats",
    "ManualStudyProjection",
    "project_study",
    "summarize_timings",
]

TIMING_SCHEMA_VERSION: Final = 1
PARENT_COUNT: Final = len(ASSIGNMENTS)
"""The ten locked demonstrations; one replay bank per parent per configuration."""


@dataclass(frozen=True)
class ManualRunStats:
    """Per-run wall-time statistics of one arm (seconds), so a projection built on means can be judged."""

    arm: str
    runs: int
    mean_simulate_s: float
    median_simulate_s: float
    max_simulate_s: float
    mean_persist_s: float
    mean_bytes: float


@dataclass(frozen=True)
class ManualStudyProjection:
    """The full study projected from the measured means of a subset (not a guaranteed bound)."""

    configurations: int
    arms: int
    models: int
    pairs_per_model: int
    parents: int
    replay_banks: int
    """``configurations x parents``: a bank belongs to one parent under one derivative policy."""
    rc_runs: int
    replay_runs: int
    total_runs: int
    rc_run_seconds: float
    replay_run_seconds: float
    fit_seconds: float
    """One fit per model of the whole study, at the mean measured fit cost."""
    total_seconds: float
    storage_bytes: int
    completed_models: int
    """Models whose evidence already exists after the measuring invocation."""
    remaining_seconds: float

    def __post_init__(self) -> None:
        """The counts are consistent and nothing is negative."""
        if self.models != self.configurations * self.arms:
            msg = f"{self.models} models is not {self.configurations} configurations of {self.arms} arms"
            raise ValueError(msg)
        if self.replay_banks != self.configurations * self.parents:
            msg = f"{self.replay_banks} banks is not {self.configurations} configurations of {self.parents} parents"
            raise ValueError(msg)
        if self.total_runs != self.rc_runs + self.replay_runs:
            msg = f"{self.total_runs} runs is not {self.rc_runs} RC and {self.replay_runs} replay"
            raise ValueError(msg)
        if min(self.total_seconds, self.remaining_seconds, self.storage_bytes, self.completed_models) < 0:
            msg = "a projection has no negative figures"
            raise ValueError(msg)


def summarize_timings(runs: Sequence[ManualRunTiming]) -> tuple[ManualRunStats, ...]:
    """Per-arm statistics of the measured runs; an arm with no measurements is omitted."""
    stats: list[ManualRunStats] = []
    for arm in ("replay", "rc"):
        selected = [run for run in runs if run.arm == arm]
        if not selected:
            continue
        stats.append(
            ManualRunStats(
                arm=arm,
                runs=len(selected),
                mean_simulate_s=float(mean(run.simulate_seconds for run in selected)),
                median_simulate_s=float(median(run.simulate_seconds for run in selected)),
                max_simulate_s=float(max(run.simulate_seconds for run in selected)),
                mean_persist_s=float(mean(run.persist_seconds for run in selected)),
                mean_bytes=float(mean(run.run_bytes for run in selected)),
            )
        )
    return tuple(stats)


def project_study(
    models: Sequence[ManualModelTiming],
    runs: Sequence[ManualRunTiming],
    *,
    pairs_per_model: int,
    completed_models: int,
    configurations: int = CONFIGURATION_COUNT,
    arms: int = ARM_COUNT,
    parents: int = PARENT_COUNT,
) -> ManualStudyProjection:
    """Project the whole study from the per-run and per-fit means measured here.

    An arm this invocation did not measure contributes nothing rather than
    dividing by zero, and what the invocation already completed is taken off
    the remaining estimate, so a resumed full run is not quoted its cost twice.
    """
    if pairs_per_model < 0 or completed_models < 0:
        msg = f"pairs_per_model and completed_models are non-negative, got {pairs_per_model} and {completed_models}"
        raise ValueError(msg)
    rc = [run for run in runs if run.arm == "rc"]
    replay = [run for run in runs if run.arm == "replay"]
    rc_seconds = float(mean(run.simulate_seconds + run.persist_seconds for run in rc)) if rc else 0.0
    replay_seconds = float(mean(run.simulate_seconds + run.persist_seconds for run in replay)) if replay else 0.0
    rc_bytes = float(mean(run.run_bytes for run in rc)) if rc else 0.0
    replay_bytes = float(mean(run.run_bytes for run in replay)) if replay else 0.0
    fit_per_model = float(mean(model.fit_seconds for model in models)) if models else 0.0
    total_models = configurations * arms
    replay_banks = configurations * parents
    rc_runs = total_models * pairs_per_model
    replay_runs = replay_banks * pairs_per_model
    fit_seconds = total_models * fit_per_model
    total = rc_runs * rc_seconds + replay_runs * replay_seconds + fit_seconds
    per_model = pairs_per_model * rc_seconds + fit_per_model
    remaining = max(0.0, total - completed_models * per_model - len(replay) * replay_seconds)
    return ManualStudyProjection(
        configurations=configurations,
        arms=arms,
        models=total_models,
        pairs_per_model=pairs_per_model,
        parents=parents,
        replay_banks=replay_banks,
        rc_runs=rc_runs,
        replay_runs=replay_runs,
        total_runs=rc_runs + replay_runs,
        rc_run_seconds=rc_seconds,
        replay_run_seconds=replay_seconds,
        fit_seconds=fit_seconds,
        total_seconds=total,
        storage_bytes=int(rc_runs * rc_bytes + replay_runs * replay_bytes),
        completed_models=completed_models,
        remaining_seconds=remaining,
    )
