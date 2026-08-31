# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Mandatory MLflow run logging into the external store (docs/PLAN.md section 11; M3-001).

Every curated run is logged to a file-backed MLflow tracking store under
``armrc://mlflow/`` with its resolved configuration, dependency revisions,
payload digests, seeds, scalar metrics, and artifacts (report, run summary,
provenance, pointer record, model recipe, plots). Logging is idempotent per
run ID, offline (MLflow telemetry is disabled), and never the only record: the
Git pointer record and the run directory remain authoritative.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # research data stays local (set before mlflow is imported)
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.records import to_toml
from arm_rc_ctrl.experiments.plots import plot_run
from arm_rc_ctrl.experiments.run_record import RUN_SUMMARY_FILE
from arm_rc_ctrl.experiments.scalars import flatten_scalars
from arm_rc_ctrl.metrics.report import report_to_json

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arm_rc_ctrl.experiments.run_record import LoadedRun
    from arm_rc_ctrl.metrics.report import RunReport
    from arm_rc_ctrl.provenance import ProvenanceRecord
    from arm_rc_ctrl.storage import StorageRoot

__all__ = ["MLFLOW_BUCKET", "RUN_ID_TAG", "TRACKING_URI", "MlflowTracker", "TrackedRun"]

MLFLOW_BUCKET: Final = "mlflow"
TRACKING_URI: Final = f"armrc://{MLFLOW_BUCKET}/tracking"
"""Logical location of the MLflow tracking store (SQLite database plus artifact directory)."""
DATABASE_FILE: Final = "mlflow.db"
ARTIFACT_DIRECTORY: Final = "artifacts"
RUN_ID_TAG: Final = "armrc.run_id"
_PARAM_LIMIT: Final = 6000
_KEY_FORBIDDEN: Final = re.compile(r"[^A-Za-z0-9_.\-/ ]")


def _key(name: str) -> str:
    """An MLflow-legal key (alphanumerics, ``_ - . /`` and spaces)."""
    return _KEY_FORBIDDEN.sub("_", name)


def _provenance_params(provenance: ProvenanceRecord) -> dict[str, object]:
    params: dict[str, object] = {}
    flatten_scalars("config", provenance.config, params)
    params["provenance.project_commit"] = provenance.project_commit
    params["provenance.project_dirty"] = provenance.project_dirty
    params["provenance.lock_sha256"] = provenance.lock_sha256
    params["provenance.config_sha256"] = provenance.config_sha256
    params["provenance.created_at"] = provenance.created_at
    for submodule in provenance.submodules:
        params[f"revisions.{submodule.name}"] = submodule.checked_out or submodule.recorded
    for build in provenance.builds:
        params[f"builds.{build.name}"] = f"{build.version}@{build.source_commit}"
    for name, seed in provenance.seeds.items():
        params[f"seeds.{name}"] = seed
    for i, artifact in enumerate(provenance.artifacts):
        params[f"artifacts.{i}.uri"] = artifact.uri
        params[f"artifacts.{i}.sha256"] = artifact.sha256
    return params


def _report_metrics(report: RunReport) -> dict[str, float]:
    flat: dict[str, object] = {}
    flatten_scalars("", to_mapping(report), flat)
    return {
        _key(key): float(value)
        for key, value in flat.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _tags(run: LoadedRun, report: RunReport) -> dict[str, str]:
    summary = run.summary
    return {
        RUN_ID_TAG: run.pointer.artifact.artifact_id,
        "armrc.method": summary.method,
        "armrc.scenario": summary.scenario,
        "armrc.kind": summary.kind,
        "armrc.termination": summary.termination.kind,
        "armrc.success": str(summary.outcome.success),
        "armrc.project_commit": summary.provenance.project_commit,
        "armrc.project_dirty": str(summary.provenance.project_dirty),
        "armrc.reference_artifact": report.reference_artifact,
    }


@dataclass(frozen=True)
class TrackedRun:
    """Where a run landed in MLflow."""

    experiment_id: str
    mlflow_run_id: str
    tracking_uri: str
    created: bool
    """``False`` when the run ID had already been logged (nothing was re-logged)."""


class MlflowTracker:
    """MLflow client on a SQLite tracking database and artifact directory in the store's ``mlflow`` bucket."""

    def __init__(self, store: StorageRoot) -> None:
        self._root = store.path(TRACKING_URI, mode="write")
        self._root.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{self._root / DATABASE_FILE}"
        self.artifact_root = self._root / ARTIFACT_DIRECTORY
        logging.getLogger("mlflow.store.db.utils").setLevel(logging.WARNING)  # schema migrations are routine
        self._client: Any = MlflowClient(tracking_uri=self.tracking_uri)

    @property
    def client(self) -> MlflowClient:
        """The underlying client (for queries)."""
        return cast("MlflowClient", self._client)

    def experiment_id(self, name: str) -> str:
        """Create or fetch the experiment named ``name``."""
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            return str(experiment.experiment_id)
        location = (self.artifact_root / _key(name).replace("/", "_")).resolve().as_uri()
        return str(self._client.create_experiment(name, artifact_location=location))

    def find(self, run_id: str) -> str | None:
        """The MLflow run ID already holding ``run_id``, if any."""
        for experiment in self._client.search_experiments():
            hits = self._client.search_runs([experiment.experiment_id], filter_string=f"tags.{RUN_ID_TAG} = '{run_id}'")
            if hits:
                return str(hits[0].info.run_id)
        return None

    def log_run(
        self,
        run: LoadedRun,
        report: RunReport,
        *,
        experiment: str,
        recipe: object | None = None,
        pointer_file: Path | None = None,
        extra_files: Mapping[str, Path] | None = None,
    ) -> TrackedRun:
        """Log one curated run (parameters, metrics, tags, artifacts, plots); idempotent per run ID."""
        run_id = run.pointer.artifact.artifact_id
        existing = self.find(run_id)
        experiment_id = self.experiment_id(experiment)
        if existing is not None:
            return TrackedRun(experiment_id, existing, self.tracking_uri, created=False)
        params = _provenance_params(run.summary.provenance)
        params["run.arrays_sha256"] = run.summary.arrays_sha256
        params["run.n_samples"] = run.arrays.n_samples
        tags = _tags(run, report)
        created = self._client.create_run(experiment_id, run_name=run_id, tags=tags)
        mlflow_run_id = str(created.info.run_id)
        self._client.log_batch(
            mlflow_run_id,
            metrics=[Metric(key, value, 0, 0) for key, value in _report_metrics(report).items()],
            params=[Param(_key(key), str(value)[:_PARAM_LIMIT]) for key, value in params.items()],
            tags=[RunTag(key, value) for key, value in tags.items()],
        )
        self._log_files(mlflow_run_id, run, report, recipe=recipe, pointer_file=pointer_file, extra=extra_files)
        self._client.set_terminated(mlflow_run_id, status="FINISHED")
        return TrackedRun(experiment_id, mlflow_run_id, self.tracking_uri, created=True)

    def _log_files(
        self,
        mlflow_run_id: str,
        run: LoadedRun,
        report: RunReport,
        *,
        recipe: object | None,
        pointer_file: Path | None,
        extra: Mapping[str, Path] | None,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="armrc-mlflow-") as scratch:
            staging = Path(scratch)
            (staging / "report.json").write_text(report_to_json(report) + "\n", encoding="utf-8")
            (staging / "provenance.json").write_text(run.summary.provenance.to_json() + "\n", encoding="utf-8")
            (staging / "pointer.toml").write_text(to_toml(run.pointer), encoding="utf-8")
            copies: dict[str, Path] = {"run.json": run.directory / RUN_SUMMARY_FILE, **(dict(extra) if extra else {})}
            if pointer_file is not None:
                copies["pointer_record.toml"] = pointer_file
            for name, source in copies.items():
                if source.is_file():
                    (staging / name).write_bytes(source.read_bytes())
            if recipe is not None:
                (staging / "recipe.toml").write_text(to_toml(recipe), encoding="utf-8")
            plots = staging / "plots"
            title = f"{run.pointer.artifact.artifact_id} ({run.summary.method})"
            plot_run(run, tuple(run.summary.target), plots / "tracking.png", title=title)
            for file in sorted(staging.iterdir()):
                if file.is_file():
                    self._client.log_artifact(mlflow_run_id, str(file))
            self._client.log_artifacts(mlflow_run_id, str(plots), artifact_path="plots")

    def find_by_tags(self, experiment_id: str, tags: Mapping[str, str]) -> str | None:
        """The run in ``experiment_id`` carrying every tag in ``tags``, if any."""
        clauses = " and ".join(f"tags.`{key}` = '{value}'" for key, value in tags.items())
        hits = self._client.search_runs([experiment_id], filter_string=clauses, max_results=1)
        return str(hits[0].info.run_id) if hits else None

    def start_run(self, experiment_id: str, *, name: str, tags: Mapping[str, str], params: Mapping[str, object]) -> str:
        """Create a run with tags and parameters; it stays RUNNING until :meth:`finish`."""
        created = self._client.create_run(experiment_id, run_name=name, tags=dict(tags))
        run_id = str(created.info.run_id)
        self._client.log_batch(
            run_id, params=[Param(_key(key), str(value)[:_PARAM_LIMIT]) for key, value in params.items()]
        )
        return run_id

    def log_metrics(self, run_id: str, metrics: Mapping[str, float], *, step: int = 0) -> None:
        """Log scalar metrics at ``step``."""
        self._client.log_batch(run_id, metrics=[Metric(_key(k), float(v), 0, step) for k, v in metrics.items()])

    def log_series(self, run_id: str, name: str, values: Sequence[float]) -> None:
        """Log ``values`` as one metric with steps ``0..len-1``."""
        self._client.log_batch(run_id, metrics=[Metric(_key(name), float(v), 0, i) for i, v in enumerate(values)])

    def log_text(self, run_id: str, name: str, text: str) -> None:
        """Log ``text`` as the artifact ``name``."""
        with tempfile.TemporaryDirectory(prefix="armrc-mlflow-") as scratch:
            path = Path(scratch) / name
            path.write_text(text, encoding="utf-8")
            self._client.log_artifact(run_id, str(path))

    def finish(self, run_id: str) -> None:
        """Mark a run finished."""
        self._client.set_terminated(run_id, status="FINISHED")

    def logged_artifacts(self, mlflow_run_id: str) -> list[str]:
        """Artifact paths of a logged run (top level and ``plots/``)."""
        names = [str(a.path) for a in self._client.list_artifacts(mlflow_run_id)]
        if "plots" in names:
            names.remove("plots")
            names.extend(str(a.path) for a in self._client.list_artifacts(mlflow_run_id, "plots"))
        return sorted(names)

    def params(self, mlflow_run_id: str) -> dict[str, str]:
        """Logged parameters of a run."""
        return dict(self._client.get_run(mlflow_run_id).data.params)

    def metrics(self, mlflow_run_id: str) -> dict[str, float]:
        """Logged metrics of a run."""
        return dict(self._client.get_run(mlflow_run_id).data.metrics)

    def tags(self, mlflow_run_id: str) -> dict[str, str]:
        """Logged tags of a run."""
        return dict(self._client.get_run(mlflow_run_id).data.tags)
