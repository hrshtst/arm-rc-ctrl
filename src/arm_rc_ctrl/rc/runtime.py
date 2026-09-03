# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Rebuild a runtime target generator from a model recipe and the external store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arm_rc_ctrl.controllers.estimator import CausalDerivativeEstimator, EstimatorConfig
from arm_rc_ctrl.data.records import Normalization, ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.rc.generator import RcTargetGenerator
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.rc.recipe import ModelRecipe
    from arm_rc_ctrl.scenario import ScenarioConfig
    from arm_rc_ctrl.storage import StorageRoot

__all__ = ["generator_from_recipe", "load_training_samples"]


def load_training_samples(
    recipe: ModelRecipe, store: StorageRoot, *, records_root: Path | None = None
) -> dict[str, SampleSet]:
    """Resolve every dataset of the recipe through its Git record and the store, binding each to the recipe.

    Every record must carry the recipe's artifact identity, joint and task-code
    widths, and preprocessing; the recipe's input transform must re-derive from
    the recorded normalization it claims to come from; every payload must match
    its record.
    """
    root = repository_root() if records_root is None else records_root
    samples: dict[str, SampleSet] = {}
    normalizations: dict[str, Normalization] = {}
    for source in recipe.datasets:
        record = load_record(root / source.record, ProcessedDatasetRecord)
        recipe.check_dataset_record(source, record)
        loaded = load_samples(verify_payload(store, record.artifact))
        record.check_samples(loaded)
        samples[source.artifact_id] = loaded
        if record.normalization is not None:
            normalizations[source.artifact_id] = record.normalization
    recipe.check_transform_source(normalizations)
    return samples


def generator_from_recipe(
    recipe: ModelRecipe,
    samples: Mapping[str, SampleSet],
    *,
    estimator: EstimatorConfig,
    position_bounds: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    scenario: ScenarioConfig | None = None,
) -> RcTargetGenerator:
    """Refit the recipe (verifying its fit report) and wrap the model as a target generator.

    Augmented recipes regenerate their synthetic episodes and need ``scenario``.
    """
    model, _report = recipe.refit(samples, scenario=scenario)
    bounds = None
    if position_bounds is not None:
        bounds = (np.asarray(position_bounds[0], dtype=np.float64), np.asarray(position_bounds[1], dtype=np.float64))
    return RcTargetGenerator(
        model,
        recipe.encoder(),
        CausalDerivativeEstimator(estimator, recipe.dof),
        position_bounds=bounds,
        output="increment" if recipe.training.target == "increment_q" else "absolute",
    )
