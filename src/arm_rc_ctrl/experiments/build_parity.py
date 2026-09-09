# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Numerical parity of two dependency builds on fixed fits (UP-007; repetition plan section 12, C1).

A submodule pin advance rebuilds ``rclib`` (and whatever it vendors, Eigen
included). Recipes bind the commit they were made with and refuse a refit
under another pin, so the question "does the new build still compute the same
numbers" must be answered separately and kept as evidence. ``probe`` refits the
frozen task 1-a recipes through the explicit identity override plus two
deterministic fixture recipes of the repetition pilot and writes their
teacher-forced predictions and fit reports; ``compare`` takes two probe
directories, one per build, and records whether every prediction is bitwise
identical and every fit report equal. The comparison is exact, not a
tolerance: a build that changes any prediction by one ulp is reported as
different, and the decision about it belongs to the owner.

Command line::

    python -m arm_rc_ctrl.experiments.build_parity probe --output <dir> [--fixture-only]
    python -m arm_rc_ctrl.experiments.build_parity compare --old <dir> --new <dir>
        --output <docs>/rclib_build_parity_v1.json --markdown <docs>/rclib_build_parity_v1.md [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.normalization import fit_normalization
from arm_rc_ctrl.data.records import Preprocessing
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.provenance import (
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
)
from arm_rc_ctrl.rc.esn import EsnConfig, ReadoutConfig, ReservoirConfig, ensure_single_thread
from arm_rc_ctrl.rc.recipe import (
    DatasetSource,
    RclibIdentity,
    TrainingSpec,
    TrainingValidation,
    create_recipe,
    load_recipe,
)
from arm_rc_ctrl.rc.runtime import load_training_samples
from arm_rc_ctrl.rc.teacher_forcing import InputTransform
from arm_rc_ctrl.rc.training import FitReport, predict_episode
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import endpoint_positions, load_scenario
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "FIXTURE_CASES",
    "PARITY_SCHEMA_VERSION",
    "SUMMARY_FILE",
    "BuildParityReport",
    "CaseParity",
    "ProbeCase",
    "ProbeSummary",
    "compare_probes",
    "fixture_cases",
    "load_parity",
    "load_probe",
    "main",
    "parity_to_json",
    "probe_fixture_recipes",
    "probe_frozen_recipes",
    "render_parity_markdown",
    "write_probe",
]

PARITY_SCHEMA_VERSION: Final = 1
SUMMARY_FILE: Final = "summary.json"
FIXTURE_CASES: Final = ("fixture-absolute-R17", "fixture-residual-R17")
_FIXTURE_SCENARIO: Final = Path("tests") / "fixtures" / "configs" / "planar_2dof_fixture.toml"
_FIXTURE_SOURCE: Final = DatasetSource("processed-20260830-555555555555", "ab" * 32, "data/records/processed/x.toml")
_N: Final = 101
_DT: Final = 0.01
_MOVE_END_S: Final = 0.8


@dataclass(frozen=True)
class ProbeCase:
    """One probed fit: its recorded report and, for frozen recipes, whether the refit reproduced it bitwise."""

    report: FitReport
    exact: bool | None = None
    """``True``/``False`` for frozen recipes (refit equals the recorded fit report exactly); ``None`` for fixtures."""


@dataclass(frozen=True)
class ProbeSummary:
    """The ``summary.json`` of one probe directory: the installed ``rclib`` and one entry per case."""

    rclib_installed: RclibIdentity
    cases: dict[str, ProbeCase]

    def __post_init__(self) -> None:
        """At least one case."""
        if not self.cases:
            msg = "a probe summary needs at least one case"
            raise ValueError(msg)

    def to_json(self) -> str:
        """``{"rclib_installed": ..., "<case>": {"report": ..., "exact": ...}, ...}``."""
        mapping: dict[str, object] = {"rclib_installed": to_mapping(self.rclib_installed)}
        for label, case in self.cases.items():
            entry: dict[str, object] = {"report": to_mapping(case.report)}
            if case.exact is not None:
                entry["exact"] = case.exact
            mapping[label] = entry
        return json.dumps(mapping, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ProbeSummary:
        """Strictly parse the summary layout above."""
        mapping = cast("dict[str, object]", json.loads(text))
        rclib = from_mapping(cast("dict[str, object]", mapping.pop("rclib_installed")), RclibIdentity)
        cases = {label: from_mapping(cast("dict[str, object]", entry), ProbeCase) for label, entry in mapping.items()}
        return cls(rclib_installed=rclib, cases=cases)


def _fixture_samples(scenario_file: Path) -> SampleSet:
    scenario = load_scenario(scenario_file)
    derivatives = DerivativeConfig(method="central")
    t = np.arange(_N, dtype=np.float64) * _DT
    start = np.array(scenario.task.initial_q)
    goal = np.array([0.8, 0.4])
    s = np.clip(t / _MOVE_END_S, 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (goal - start)[None, :]
    dq, ddq = differentiate(q, _DT, derivatives)
    tip = endpoint_positions(scenario, q)
    dtip, ddtip = differentiate(tip, _DT, derivatives)
    phase = np.where(t < _MOVE_END_S, 1, 2).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((_N, 0)), phase)


def fixture_cases(root: Path | None = None) -> dict[str, tuple[NDArray[np.float64], FitReport]]:
    """Fit the two deterministic fixture recipes (absolute and residual exact repetition at K = 17)."""
    root = repository_root() if root is None else root
    scenario_file = root / _FIXTURE_SCENARIO
    scenario = load_scenario(scenario_file)
    samples = _fixture_samples(scenario_file)
    normalization = fit_normalization(
        samples.arrays(),
        ("q", "dq"),
        fitted_on=(_FIXTURE_SOURCE.artifact_id,),
        training_rows=np.ones(_N, dtype=np.bool_),
    )
    transform = InputTransform.derive("fixed_scale", normalization, fixed_scales={"q": 0.3, "dq": 4.0})
    esn = EsnConfig(
        reservoir=ReservoirConfig(
            n_neurons=40, spectral_radius=0.85, sparsity=0.9, leak_rate=0.4, input_scaling=0.4, seed=23
        ),
        readout=ReadoutConfig(alpha=0.02),
    )
    validation = TrainingValidation.from_scenario(scenario, scenario_file, root=root)
    preprocessing = Preprocessing(
        resample_period_s=_DT, smoothing="none", smoothing_params={}, derivative_method="central-difference"
    )
    out: dict[str, tuple[NDArray[np.float64], FitReport]] = {}
    for label, target in zip(FIXTURE_CASES, ("next_q", "increment_q"), strict=True):
        spec = TrainingSpec(
            washout="warmup_hold",
            warmup_s=0.25,
            target=target,
            additional_repeats=16,
            base_alpha=0.02,
            regularization_rule="base",
        )
        recipe, model = create_recipe(
            label,
            esn,
            sources=[_FIXTURE_SOURCE],
            samples={_FIXTURE_SOURCE.artifact_id: samples},
            dof=2,
            task_code_dim=0,
            preprocessing=preprocessing,
            transform=transform,
            training=spec,
            scenario=scenario,
            validation=validation,
        )
        episode = recipe.episodes({_FIXTURE_SOURCE.artifact_id: samples}, scenario=scenario)[0]
        out[label] = (predict_episode(model, episode), recipe.fit)
    return out


def probe_fixture_recipes(output: Path, *, root: Path | None = None) -> ProbeSummary:
    """Write the fixture-only probe of the installed build to ``output``."""
    cases: dict[str, ProbeCase] = {}
    predictions: dict[str, NDArray[np.float64]] = {}
    for label, (prediction, report) in fixture_cases(root).items():
        predictions[label] = prediction
        cases[label] = ProbeCase(report=report)
    summary = ProbeSummary(rclib_installed=RclibIdentity.current(), cases=cases)
    write_probe(output, summary, predictions)
    return summary


def probe_frozen_recipes(output: Path, *, root: Path | None = None) -> ProbeSummary:
    """Write the full probe: frozen recipes from the store (identity override) plus the fixture recipes."""
    root = repository_root() if root is None else root
    store = open_storage()
    cases: dict[str, ProbeCase] = {}
    predictions: dict[str, NDArray[np.float64]] = {}
    for path in sorted((root / "data" / "records" / "models").glob("model-*.toml")):
        recipe = load_recipe(path)
        samples = load_training_samples(recipe, store, records_root=root)
        model, report = recipe.refit(samples, installed=recipe.rclib)  # parity across pins: override the identity
        predictions[path.stem] = predict_episode(model, recipe.episodes(samples)[0])
        cases[path.stem] = ProbeCase(report=report, exact=report == recipe.fit)
    for label, (prediction, report) in fixture_cases(root).items():
        predictions[label] = prediction
        cases[label] = ProbeCase(report=report)
    summary = ProbeSummary(rclib_installed=RclibIdentity.current(), cases=cases)
    write_probe(output, summary, predictions)
    return summary


def write_probe(output: Path, summary: ProbeSummary, predictions: dict[str, NDArray[np.float64]]) -> None:
    """Write ``<case>.npy`` per case and ``summary.json`` (refusing to overwrite an existing probe)."""
    if set(predictions) != set(summary.cases):
        msg = "predictions and summary cases must name the same cases"
        raise ValueError(msg)
    if (output / SUMMARY_FILE).exists():
        msg = f"refusing to overwrite the probe in {output}"
        raise FileExistsError(msg)
    output.mkdir(parents=True, exist_ok=True)
    for label, prediction in predictions.items():
        np.save(output / f"{label}.npy", np.ascontiguousarray(prediction, dtype=np.float64))
    (output / SUMMARY_FILE).write_text(summary.to_json() + "\n", encoding="utf-8")


def load_probe(directory: Path) -> tuple[ProbeSummary, dict[str, NDArray[np.float64]]]:
    """Read a probe directory back (strictly)."""
    summary = ProbeSummary.from_json((directory / SUMMARY_FILE).read_text(encoding="utf-8"))
    predictions = {
        label: np.asarray(np.load(directory / f"{label}.npy", allow_pickle=False), dtype=np.float64)
        for label in summary.cases
    }
    return summary, predictions


@dataclass(frozen=True)
class CaseParity:
    """Old-build versus new-build outcome of one probed fit."""

    case: str
    predictions_bitwise_equal: bool
    max_abs_prediction_diff: float
    prediction_sha256_old: str
    prediction_sha256_new: str
    fit_report_equal: bool
    rmse_old: float
    rmse_new: float
    exact_old: bool | None
    exact_new: bool | None
    """For frozen recipes: whether each build's refit reproduced the recorded fit report exactly."""

    def __post_init__(self) -> None:
        """The flags re-derive from the digests and figures."""
        if self.predictions_bitwise_equal != (self.prediction_sha256_old == self.prediction_sha256_new):
            msg = f"{self.case}: predictions_bitwise_equal contradicts the prediction digests"
            raise ValueError(msg)
        if self.predictions_bitwise_equal and self.max_abs_prediction_diff != 0.0:
            msg = f"{self.case}: bitwise-equal predictions cannot differ"
            raise ValueError(msg)


@dataclass(frozen=True)
class BuildParityReport:
    """The committed comparison of two builds on the probed fits."""

    rclib_old: RclibIdentity
    rclib_new: RclibIdentity
    cases: tuple[CaseParity, ...]
    all_bitwise_equal: bool
    provenance: ProvenanceRecord
    schema_version: int = field(default=PARITY_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """The summary flag re-derives from the cases."""
        if self.schema_version != PARITY_SCHEMA_VERSION:
            msg = f"unsupported parity schema_version {self.schema_version}"
            raise ValueError(msg)
        if not self.cases:
            msg = "a parity report compares at least one case"
            raise ValueError(msg)
        if self.all_bitwise_equal != all(c.predictions_bitwise_equal and c.fit_report_equal for c in self.cases):
            msg = "all_bitwise_equal contradicts the cases"
            raise ValueError(msg)


def compare_probes(old: Path, new: Path, *, provenance: ProvenanceRecord) -> BuildParityReport:
    """Compare two probe directories case by case; the case sets must be identical."""
    old_summary, old_predictions = load_probe(old)
    new_summary, new_predictions = load_probe(new)
    if set(old_summary.cases) != set(new_summary.cases):
        msg = f"probe case sets differ: {sorted(old_summary.cases)} vs {sorted(new_summary.cases)}"
        raise ValueError(msg)
    cases: list[CaseParity] = []
    for label in sorted(old_summary.cases):
        a, b = old_predictions[label], new_predictions[label]
        if a.shape != b.shape:
            msg = f"{label}: prediction shapes differ ({a.shape} vs {b.shape})"
            raise ValueError(msg)
        digest_old, digest_new = array_digest(a), array_digest(b)
        cases.append(
            CaseParity(
                case=label,
                predictions_bitwise_equal=digest_old == digest_new,
                max_abs_prediction_diff=float(np.max(np.abs(a - b))),
                prediction_sha256_old=digest_old,
                prediction_sha256_new=digest_new,
                fit_report_equal=old_summary.cases[label].report == new_summary.cases[label].report,
                rmse_old=old_summary.cases[label].report.rmse,
                rmse_new=new_summary.cases[label].report.rmse,
                exact_old=old_summary.cases[label].exact,
                exact_new=new_summary.cases[label].exact,
            )
        )
    return BuildParityReport(
        rclib_old=old_summary.rclib_installed,
        rclib_new=new_summary.rclib_installed,
        cases=tuple(cases),
        all_bitwise_equal=all(c.predictions_bitwise_equal and c.fit_report_equal for c in cases),
        provenance=provenance,
    )


def parity_to_json(report: BuildParityReport) -> str:
    """Canonical JSON of the report."""
    return canonical_json(to_mapping(report))


def load_parity(path: Path) -> BuildParityReport:
    """Strictly rebuild a report from JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), BuildParityReport)


def _fmt(value: bool | None) -> str:  # noqa: FBT001 - a tri-state flag is rendered, not a switch
    return "n/a" if value is None else str(value)


def render_parity_markdown(report: BuildParityReport) -> str:
    """The Markdown rendering of the report."""
    lines = [
        "# rclib build parity (v1)",
        "",
        (
            f"Old build: rclib {report.rclib_old.version} at `{report.rclib_old.commit[:12]}`; new build: rclib "
            f"{report.rclib_new.version} at `{report.rclib_new.commit[:12]}`; project commit "
            f"`{report.provenance.project_commit[:12]}`{' (dirty)' if report.provenance.project_dirty else ''}."
        ),
        "",
        (
            "Every case refits one recipe under each build and compares the teacher-forced predictions of its "
            "first episode bitwise and its fit report exactly. Frozen recipes are refitted through the explicit "
            "identity override (they bind the commit they were made with); `exact` says whether each build's "
            "refit reproduced the recorded fit report bitwise, which a difference in the recording environment, "
            "not in the build, can already prevent."
        ),
        "",
        f"- All predictions bitwise equal and all fit reports equal: **{report.all_bitwise_equal}**.",
        "",
        (
            "| case | predictions bitwise equal | max abs diff | fit report equal | rmse old | rmse new "
            "| exact old | exact new |"
        ),
        "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    lines.extend(
        f"| {c.case} | {c.predictions_bitwise_equal} | {c.max_abs_prediction_diff:.3e} | {c.fit_report_equal} "
        f"| {c.rmse_old!r} | {c.rmse_new!r} | {_fmt(c.exact_old)} | {_fmt(c.exact_new)} |"
        for c in report.cases
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Probe and compare the numerical parity of two dependency builds.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    probe = subparsers.add_parser("probe", help="refit the frozen and fixture recipes with the installed build")
    probe.add_argument("--output", type=Path, required=True, help="probe directory to write (must not exist)")
    probe.add_argument("--fixture-only", action="store_true", help="skip the frozen recipes (no external store)")
    compare = subparsers.add_parser("compare", help="compare two probe directories and write the evidence")
    compare.add_argument("--old", type=Path, required=True)
    compare.add_argument("--new", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True, help="parity JSON to write (must not exist)")
    compare.add_argument("--markdown", type=Path, required=True, help="parity Markdown to write (must not exist)")
    compare.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    ensure_single_thread()
    if args.subcommand == "probe":
        output = Path(str(args.output))
        summary = probe_fixture_recipes(output) if bool(args.fixture_only) else probe_frozen_recipes(output)
        print(json.dumps({"rclib": to_mapping(summary.rclib_installed), "cases": sorted(summary.cases)}, indent=2))
        return 0
    for target in (args.output, args.markdown):
        if Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    old, new = Path(str(args.old)), Path(str(args.new))
    resolved = {
        "old": json.loads((old / SUMMARY_FILE).read_text(encoding="utf-8")),
        "new": json.loads((new / SUMMARY_FILE).read_text(encoding="utf-8")),
        "command": command_line("arm_rc_ctrl.experiments.build_parity", list(sys.argv[1:] if argv is None else argv)),
    }
    provenance = collect_provenance(
        resolved, seeds={}, artifacts=[], exploratory=bool(args.exploratory), now=datetime.now(tz=UTC)
    )
    require_clean_for_confirmatory(provenance)
    report = compare_probes(old, new, provenance=provenance)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(parity_to_json(report) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_parity_markdown(report), encoding="utf-8")
    print(json.dumps({"all_bitwise_equal": report.all_bitwise_equal, "cases": len(report.cases)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
