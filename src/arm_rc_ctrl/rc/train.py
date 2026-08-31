# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Task 1-a training command: datasets in, deterministic model recipe and provenance out (M2-011).

Usage::

    python -m arm_rc_ctrl.rc.train --model configs/models/esn_task_1a.toml \\
        --dataset data/records/processed/<id>.toml --report docs/experiments/task_1a/<name>.json

The recipe is written to ``data/records/models/<recipe id>.toml`` unless
``--recipe`` names another file; the ID is content-addressed from the recipe.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.config import load_config, to_mapping
from arm_rc_ctrl.data.records import ProcessedDatasetRecord, load_record, verify_payload
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    validate_utc_timestamp,
)
from arm_rc_ctrl.rc.esn import EsnConfig, EsnModel
from arm_rc_ctrl.rc.recipe import DatasetSource, ModelRecipe, create_recipe, write_recipe
from arm_rc_ctrl.rc.teacher_forcing import InputTransform, TransformPolicy
from arm_rc_ctrl.rc.training import FitReport
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arm_rc_ctrl.data.samples import SampleSet

__all__ = [
    "InputTransformSpec",
    "ModelConfig",
    "TrainingReport",
    "TrainingResult",
    "load_model_config",
    "main",
    "recipe_id",
    "train_task",
]

REPORT_SCHEMA_VERSION: Final = 1
RECIPE_DIRECTORY: Final = Path("data") / "records" / "models"


@dataclass(frozen=True)
class InputTransformSpec:
    """Input transform policy of a model configuration."""

    policy: TransformPolicy
    q_scale: float | None = None
    dq_scale: float | None = None

    def __post_init__(self) -> None:
        """Fixed scales are required (and positive) exactly for the ``fixed_scale`` policy."""
        scales = (self.q_scale, self.dq_scale)
        if self.policy == "fixed_scale":
            if any(v is None or not (v > 0) for v in scales):
                msg = "input_transform.q_scale and dq_scale must be positive for the fixed_scale policy"
                raise ValueError(msg)
        elif any(v is not None for v in scales):
            msg = "input_transform scales are only meaningful for the fixed_scale policy"
            raise ValueError(msg)

    @property
    def fixed_scales(self) -> dict[str, float] | None:
        """The per-channel scales for ``InputTransform.derive``."""
        if self.policy != "fixed_scale":
            return None
        return {"q": float(self.q_scale or 0.0), "dq": float(self.dq_scale or 0.0)}


@dataclass(frozen=True)
class ModelConfig:
    """A model configuration file (``configs/models/*.toml``)."""

    name: str
    esn: EsnConfig
    input_transform: InputTransformSpec

    def __post_init__(self) -> None:
        """The name identifies the recipe family."""
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)


def load_model_config(path: Path) -> ModelConfig:
    """Load and validate a model configuration."""
    return load_config(path, ModelConfig)


@dataclass(frozen=True)
class TrainingReport:
    """Curated, Git-tracked training summary."""

    recipe_id: str
    recipe_file: str
    model_config: str
    datasets: tuple[str, ...]
    fit: FitReport
    refit_verified: bool
    """Whether rebuilding and refitting the written recipe reproduced the fit report within tolerance."""
    provenance: ProvenanceRecord
    schema_version: int = REPORT_SCHEMA_VERSION


@dataclass(frozen=True)
class TrainingResult:
    """Outputs of :func:`train_task`."""

    recipe: ModelRecipe
    model: EsnModel
    report: TrainingReport
    samples: dict[str, SampleSet]


def recipe_id(recipe: ModelRecipe, created_at: str) -> str:
    """Content-addressed recipe ID ``model-<YYYYMMDD>-<12 hex of the canonical recipe JSON>``."""
    stamp = validate_utc_timestamp(created_at)
    digest = hashlib.sha256(canonical_json(to_mapping(recipe)).encode("utf-8")).hexdigest()
    return f"model-{stamp:%Y%m%d}-{digest[:12]}"


def _repo_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        msg = f"{path} lies outside the records root {root}"
        raise ValueError(msg)
    return resolved.relative_to(root).as_posix()


def train_task(
    config: ModelConfig,
    config_file: Path,
    dataset_records: Sequence[Path],
    *,
    store: StorageRoot,
    exploratory: bool,
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.rc.train",
    records_root: Path | None = None,
) -> TrainingResult:
    """Validate the datasets, train the readout, and build the recipe, report, and provenance.

    Every dataset must come from one scenario with one joint count, task-code
    width, and preprocessing, and must carry normalization statistics; the
    input transform derives from the first dataset. The written recipe is
    rebuilt and refitted as a self-check before the report claims it.
    """
    if not dataset_records:
        msg = "at least one processed dataset record is required"
        raise ValueError(msg)
    root = repository_root() if records_root is None else records_root.resolve()
    records = [load_record(path, ProcessedDatasetRecord) for path in dataset_records]
    first = records[0]
    if first.normalization is None:
        msg = f"dataset {first.artifact.artifact_id} records no normalization statistics"
        raise ValueError(msg)
    for record in records[1:]:
        if (
            record.scenario != first.scenario
            or record.dof != first.dof
            or record.task_code_dim != first.task_code_dim
            or record.preprocessing != first.preprocessing
        ):
            msg = (
                f"dataset {record.artifact.artifact_id} does not share the scenario, widths, and preprocessing "
                f"of {first.artifact.artifact_id}"
            )
            raise ValueError(msg)
    ids = [record.artifact.artifact_id for record in records]
    if len(set(ids)) != len(ids):
        msg = f"datasets must be distinct, got {ids}"
        raise ValueError(msg)
    samples: dict[str, SampleSet] = {}
    sources: list[DatasetSource] = []
    references: list[ArtifactReference] = []
    for path, record in zip(dataset_records, records, strict=True):
        loaded = load_samples(verify_payload(store, record.artifact))
        record.check_samples(loaded)
        samples[record.artifact.artifact_id] = loaded
        payload = record.artifact.payload
        sources.append(DatasetSource(record.artifact.artifact_id, payload.sha256, _repo_relative(path, root)))
        references.append(ArtifactReference(payload.uri, payload.sha256, payload.size))
    transform = InputTransform.derive(
        config.input_transform.policy, first.normalization, fixed_scales=config.input_transform.fixed_scales
    )
    model_config_file = (
        _repo_relative(config_file, repository_root())
        if config_file.resolve().is_relative_to(repository_root())
        else config_file.name
    )
    resolved = {
        "model": to_mapping(config),
        "model_config": model_config_file,
        "transform": to_mapping(transform),
        "datasets": ids,
        "command": command,
    }
    provenance = collect_provenance(
        resolved,
        seeds={"reservoir": config.esn.reservoir.seed},
        artifacts=references,
        exploratory=exploratory,
        now=now,
    )
    require_clean_for_confirmatory(provenance)
    recipe, model = create_recipe(
        config.name,
        config.esn,
        sources=sources,
        samples=samples,
        dof=first.dof,
        task_code_dim=first.task_code_dim,
        preprocessing=first.preprocessing,
        transform=transform,
    )
    for source, record in zip(sources, records, strict=True):
        recipe.check_dataset_record(source, record)
    recipe.check_transform_source({first.artifact.artifact_id: first.normalization})
    recipe.refit(samples)  # the self-check: a recipe that cannot reproduce its fit is never written
    identifier = recipe_id(recipe, provenance.created_at)
    report = TrainingReport(
        recipe_id=identifier,
        recipe_file=(RECIPE_DIRECTORY / f"{identifier}.toml").as_posix(),
        model_config=model_config_file,
        datasets=tuple(ids),
        fit=recipe.fit,
        refit_verified=True,
        provenance=provenance,
    )
    return TrainingResult(recipe, model, report, samples)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the task 1-a ESN and write its recipe.")
    parser.add_argument("--model", type=Path, required=True, help="model configuration (configs/models/*.toml)")
    parser.add_argument(
        "--dataset", type=Path, action="append", required=True, help="processed dataset record (repeatable)"
    )
    parser.add_argument("--report", type=Path, required=True, help="training report JSON to write (must not exist)")
    parser.add_argument(
        "--recipe", type=Path, default=None, help="recipe file to write (default: data/records/models/<id>.toml)"
    )
    parser.add_argument("--records-root", type=Path, default=None, help="root the dataset records are relative to")
    parser.add_argument(
        "--exploratory", action="store_true", help="allow a dirty worktree (result is not confirmatory)"
    )
    args = parser.parse_args(argv)
    if Path(args.report).exists():
        msg = f"refusing to overwrite {args.report}"
        raise FileExistsError(msg)
    config_file = Path(args.model)
    config = load_model_config(config_file)
    result = train_task(
        config,
        config_file,
        [Path(p) for p in args.dataset],
        store=open_storage(),
        exploratory=bool(args.exploratory),
        now=datetime.now(tz=UTC),
        command=command_line("arm_rc_ctrl.rc.train", sys.argv[1:] if argv is None else argv),
        records_root=None if args.records_root is None else Path(args.records_root),
    )
    report = result.report
    if args.recipe is not None:
        recipe_file = Path(args.recipe)
        report = dataclasses.replace(report, recipe_file=recipe_file.name)  # records never carry machine paths
    else:
        recipe_file = repository_root() / report.recipe_file
    recipe_file.parent.mkdir(parents=True, exist_ok=True)
    write_recipe(recipe_file, result.recipe)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(to_mapping(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "recipe_id": report.recipe_id,
                "recipe_file": str(recipe_file),
                "datasets": list(report.datasets),
                "loss_rows": report.fit.loss_rows,
                "rmse": report.fit.rmse,
                "constant_rmse": report.fit.constant_rmse,
                "refit_verified": report.refit_verified,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
