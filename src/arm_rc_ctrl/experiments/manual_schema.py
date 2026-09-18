# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the frozen result schema of the manual-demonstration evidence (plan section 7.1).

The machine-readable evidence ships with documented field definitions and
units, so an independent reader can reproduce the analysis without asking the
author what a column meant. The structure here is derived from the dataclasses
the sweep actually writes, and the definitions are authored beside that
derivation: a field with no definition, or an optional field that does not say
what its absence means, refuses to freeze at all. A new field therefore cannot
reach the evidence undescribed, and a changed field fails the regression lock
rather than silently refreshing the committed artifact -- changing it is a
version decision.

Numbers carry a unit and the scope they aggregate over, because a count without
a denominator cannot be read back correctly by anyone but its author.
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments import manual_evaluation as records
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.provenance import canonical_json

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "RESULT_RECORDS",
    "RESULT_SCHEMA_VERSION",
    "FieldSpec",
    "RecordSpec",
    "ResultSchema",
    "describe_records",
    "load_schema",
    "main",
    "render_schema_markdown",
    "result_schema",
    "schema_to_json",
]

RESULT_SCHEMA_VERSION: Final = 1
RESULT_RECORDS: Final = (
    "ManualRunConditions",
    "ManualEvidencePointer",
    "ManualModelEvidence",
    "ManualReplayBank",
    "ManualPairRecord",
    "ManualRunOutcome",
    "ManualRunArtifact",
    "ManualFitBinding",
    "ManualDwellReport",
    "GeneratedReferenceReport",
    "TriggerOutcome",
)
"""The evaluation evidence the sweep writes; the timing report is its own artifact."""

_PAIRS: Final = "pairs"
_SECONDS: Final = "s"

# (definition, unit, aggregation scope or denominator, what absence means)
_DEFINITIONS: Final[dict[str, dict[str, tuple[str, str, str, str]]]] = {
    "ManualRunConditions": {
        "evaluation_name": ("Name the evaluation configuration declares for this protocol.", "", "", ""),
        "evaluation_file": ("Path of the evaluation configuration, relative to the repository root.", "", "", ""),
        "evaluation_sha256": ("SHA-256 of the evaluation configuration as read.", "", "", ""),
        "development_file": ("Path of the locked development-robustness draws.", "", "", ""),
        "development_sha256": (
            "SHA-256 of the locked draws: the scenarios are only these if this file is.",
            "",
            "",
            "",
        ),
        "scenario_file": ("Path of the manual task configuration.", "", "", ""),
        "scenario_sha256": ("SHA-256 of the task configuration as read.", "", "", ""),
        "horizon_s": (
            (
                "Evaluation horizon measured from activation; completion is judged against it and "
                "never against a demonstration's length."
            ),
            _SECONDS,
            "one run",
            "",
        ),
        "trigger_hold_s": (
            "Continuous target dwell the arm must hold before the force pulse is armed.",
            _SECONDS,
            "one run",
            "",
        ),
        "trigger_duration_s": ("Length of the force pulse once it fires.", _SECONDS, "one run", ""),
        "trigger_magnitude_n": ("Magnitude of the force pulse.", "N", "one run", ""),
        "dwell_min_duration_s": (
            "Uninterrupted final dwell a run must achieve to count as completed.",
            _SECONDS,
            "one run",
            "",
        ),
        "dwell_tolerance_m": (
            "Radius of the target region every dwell sample must lie inside.",
            "m",
            "one dwell sample",
            "",
        ),
        "dwell_max_velocity_rad_s": (
            (
                "Maximum absolute joint speed allowed at every dwell sample; the acquisition rule, "
                "carried explicitly so a run records the rule it was judged by."
            ),
            "rad/s",
            "one dwell sample, maximum over joints",
            "",
        ),
        "replay_velocity_cutoff_hz": (
            (
                "Velocity cutoff of the causal derivative estimator replay is driven through: its "
                "paired configuration's own, so both arms filter alike."
            ),
            "Hz",
            "one run",
            "",
        ),
        "replay_acceleration_cutoff_hz": ("Acceleration cutoff of that same estimator.", "Hz", "one run", ""),
        "velocity_abort": (
            "Per-joint speed at which a run is aborted as unsafe.",
            "rad/s",
            "per joint, one sample",
            "",
        ),
        "trackers": ("SHA-256 of each frozen tracker's gains, by name.", "", "", ""),
        "tracker_order": (
            "The trackers in evaluation order; the JSON form sorts mapping keys, so the order is explicit here.",
            "",
            "",
            "",
        ),
        "scenario_ids": ("Identifiers of the development cases evaluated, in evaluation order.", "", "", ""),
        "warmup_s": (
            "Hold at the reset posture before activation; the reference begins after it.",
            _SECONDS,
            "one run",
            "",
        ),
        "execution_identity": (
            "Identity of the canonical execution environment every run under these conditions was keyed in.",
            "",
            "",
            "",
        ),
    },
    "ManualEvidencePointer": {
        "schema": ("Schema tag of the pointer record.", "", "", ""),
        "experiment": ("Experiment label the evidence belongs to.", "", "", ""),
        "kind": ("Either `model` or `replay`: which kind of manifest this points at.", "", "", ""),
        "identity": ("Identity of the manifest pointed at.", "", "", ""),
        "label": ("Name of the model or replay bank, as the repository files it.", "", "", ""),
        "status": ("`feasible` when every pair completed, `infeasible` otherwise.", "", "", ""),
        "payload": ("Store reference and digest of the manifest itself.", "", "", ""),
        "n_pairs": ("Pairs the manifest records.", _PAIRS, "one manifest", ""),
        "n_completed": (
            "Pairs whose run reached the horizon and met every criterion.",
            _PAIRS,
            "one manifest; of n_pairs",
            "",
        ),
        "n_infeasible": ("Pairs that failed at least one criterion.", _PAIRS, "one manifest; of n_pairs", ""),
    },
    "ManualModelEvidence": {
        "identity": ("Key this evidence is stored and served under: its fit and its conditions together.", "", "", ""),
        "label": ("`<configuration>/<arm>`, the model this evidence is of.", "", "", ""),
        "conditions": ("The protocol conditions every pair here was run under.", "", "", ""),
        "fit": (
            "The fit these runs were produced from, bound by identity and digests.",
            "",
            "",
            "a record naming no fit; every evidence the sweep writes binds one.",
        ),
        "assignment": (
            "The parent this model is paired against.",
            "",
            "",
            "the all-ten arm, which trains on the whole bank and has no single parent.",
        ),
        "replay_bank": (
            "Identity of the replay bank this model's pairs are compared against.",
            "",
            "",
            "the all-ten arm, which is paired against no single bank.",
        ),
        "status": ("`feasible` when every pair completed, `infeasible` otherwise.", "", "", ""),
        "pairs": ("Every scenario and tracker pair of this model, in evaluation order.", "", "", ""),
        "n_pairs": ("Pairs recorded: scenarios times trackers.", _PAIRS, "one model", ""),
        "n_completed": ("Pairs that met every criterion.", _PAIRS, "one model; of n_pairs", ""),
        "n_infeasible": ("Pairs that failed at least one criterion.", _PAIRS, "one model; of n_pairs", ""),
        "n_unexecuted": (
            (
                "Pairs recorded as not executed. A completed sweep emits zero: every scenario is "
                "attempted from a fresh reset (D6), so nothing is skipped because an earlier scenario "
                "failed. An interrupted sweep's missing runs are simply absent from the record and are "
                "never turned into unexecuted pairs."
            ),
            _PAIRS,
            "one model; of n_pairs",
            "",
        ),
        "schema_version": ("Version of the evidence schema this record follows.", "version", "one record", ""),
    },
    "ManualReplayBank": {
        "conditions": ("The protocol conditions every baseline here was run under.", "", "", ""),
        "assignment": ("The demonstration these baselines replay.", "", "", ""),
        "pairs": ("Every scenario and tracker pair of this bank, in evaluation order.", "", "", ""),
    },
    "ManualPairRecord": {
        "index": ("Position of this scenario in the evaluation order.", "index", "one model's scenario list", ""),
        "scenario_id": ("Identifier of the development case.", "", "", ""),
        "kind": ("Perturbation family of the case, such as `nominal` or `posture_small`.", "", "", ""),
        "tracker": ("Which frozen tracker ran the pair.", "", "", ""),
        "arm": ("`rc` for the generated reference, `replay` for the recorded one.", "", "", ""),
        "status": ("`completed` when the run met every criterion, `infeasible` otherwise.", "", "", ""),
        "initial_q": ("Joint angles the run started from, after the case's perturbation.", "rad", "per joint", ""),
        "outcome": (
            "What the run achieved and why it did not succeed when it did not.",
            "",
            "",
            "a pair that was not simulated; a simulated pair always carries one.",
        ),
        "run": (
            "Where the stored run lives and what it contains.",
            "",
            "",
            "a pair that was not simulated; a simulated pair always carries one.",
        ),
        "pulse_start_s": (
            "When the pulse actually fired on the run clock, rather than when the levels prescribed it.",
            _SECONDS,
            "one run clock",
            "no pulse fired in this run.",
        ),
    },
    "ManualRunOutcome": {
        "completed": ("The run reached the configured horizon measured from activation.", "", "", ""),
        "dwell": (
            "The dwell over the whole active segment: earliest, longest, final, and departures after holds.",
            "",
            "",
            "",
        ),
        "post_pulse_dwell": (
            "The dwell measured from the pulse end, which is what a force case must satisfy.",
            "",
            "",
            "no pulse fired, so the whole active segment is judged instead.",
        ),
        "generated": (
            "The generated reference judged by the same rules as the actual motion.",
            "",
            "",
            "a replay run, which carries no readout to judge.",
        ),
        "saturation_fraction": (
            "Share of active-segment samples in which any joint's requested torque reached its limit.",
            "fraction in [0, 1]",
            "active segment; denominator is its sample count",
            "",
        ),
        "torque_rms": (
            "Root mean square of the applied torque over the active segment.",
            "N m",
            "active segment, over joints and samples",
            "the run aborted before its active segment began, so there is nothing to average.",
        ),
        "trigger": (
            "Whether the pulse fired, and early enough to leave room for the dwell it disturbs.",
            "",
            "",
            "a case that prescribes no pulse.",
        ),
        "success": ("Every criterion this sweep judges was met.", "", "", ""),
        "reason": (
            "Why the run did not succeed, naming the cause rather than a consequence.",
            "",
            "",
            "the run succeeded.",
        ),
    },
    "ManualRunArtifact": {
        "artifact_id": ("Identity of the stored run.", "", "", ""),
        "uri": ("Store location of the run summary.", "", "", ""),
        "sha256": ("SHA-256 of the run summary as written.", "", "", ""),
        "size": ("Size of the run summary.", "bytes", "one run summary", ""),
        "arrays_sha256": ("SHA-256 of the run's arrays, as the summary records it.", "", "", ""),
        "sources": (
            "The demonstrations this run's model trained on: all ten for the all-ten arm, one otherwise.",
            "",
            "",
            "",
        ),
    },
    "ManualFitBinding": {
        "identity": ("Identity of the fit these runs were produced from.", "", "", ""),
        "configuration": ("Inherited configuration the fit belongs to.", "", "", ""),
        "arm": ("Training arm of the fit.", "", "", ""),
        "solver_alpha": (
            "Ridge parameter the readout was solved at: the configuration's base alpha times the episode count.",
            "ridge alpha",
            "one readout solve",
            "",
        ),
        "recipe_sha256": ("SHA-256 of the recipe as written.", "", "", ""),
        "weights_sha256": ("Digest of the fitted readout weights, over dtype, shape and bytes.", "", "", ""),
    },
    "ManualDwellReport": {
        "ok": (
            "The run ending at the last sample satisfies the dwell rule for its full required duration.",
            "",
            "",
            "",
        ),
        "final_samples": ("Samples of the dwell that ends at the last sample.", "samples", "the final dwell", ""),
        "final_duration_s": ("Duration of that final dwell.", _SECONDS, "the final dwell", ""),
        "longest_samples": (
            "Samples of the longest qualifying dwell anywhere in the active segment.",
            "samples",
            "the longest qualifying dwell",
            "",
        ),
        "longest_duration_s": ("Duration of that longest dwell.", _SECONDS, "the longest qualifying dwell", ""),
        "earliest_start_s": (
            "Start of the first dwell that reached the required length.",
            _SECONDS,
            "run clock",
            "no dwell ever reached the required length.",
        ),
        "departures_after_hold": (
            "Qualifying holds that ended before the last sample: the arm reached the target and then left it.",
            "count",
            "active segment",
            "",
        ),
    },
    "GeneratedReferenceReport": {
        "within_position_limits": ("Every generated joint command lay inside the joint limits.", "", "", ""),
        "within_speed_limits": ("Every generated joint speed lay inside the speed limits.", "", "", ""),
        "within_workspace": (
            "The generated endpoint stayed inside the reachable workspace.",
            "",
            "",
            (
                "the generated command left the joint limits, so forward kinematics would have "
                "answered for a clamped trajectory rather than the commanded one."
            ),
        ),
        "dwell": (
            "The dwell the generated reference itself achieves, judged by the rule the actual motion is judged by.",
            "",
            "",
            "the generated command was not evaluable; see within_position_limits.",
        ),
    },
    "TriggerOutcome": {
        "ok": ("The pulse fired and left room for the dwell it disturbs.", "", "", ""),
        "triggered": ("A pulse fired at all.", "", "", ""),
        "pulse_start_s": ("When the pulse began on the run clock.", _SECONDS, "one run clock", "no pulse fired."),
        "pulse_end_s": ("When the pulse ended on the run clock.", _SECONDS, "one run clock", "no pulse fired."),
        "reason": (
            "Why the case cannot count as a disturbance-recovery test.",
            "",
            "",
            "the timing rule was satisfied.",
        ),
    },
}


@dataclass(frozen=True)
class FieldSpec:
    """One field of one record: what it is, what it is in, and what its absence says."""

    name: str
    type: str
    optional: bool
    definition: str
    unit: str
    scope: str
    """What the figure aggregates over, and the denominator of a share or a count."""
    absent: str


@dataclass(frozen=True)
class RecordSpec:
    """One record of the evidence, with every field it carries."""

    name: str
    summary: str
    fields: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class ResultSchema:
    """The frozen description of the evidence the manual sweep writes."""

    experiment: str
    schema_version: int
    records: tuple[RecordSpec, ...]

    def __post_init__(self) -> None:
        """The document names the experiment and schema it belongs to."""
        if self.schema_version != RESULT_SCHEMA_VERSION or self.experiment != EXPERIMENT_LABEL:
            msg = f"unsupported result schema {self.schema_version} or experiment {self.experiment!r}"
            raise ValueError(msg)
        if [record.name for record in self.records] != list(RESULT_RECORDS):
            msg = "the schema must describe exactly the evaluation records, in their declared order"
            raise ValueError(msg)


def describe_records() -> tuple[RecordSpec, ...]:
    """Derive every record from the dataclasses, refusing any field that was never described."""
    described: list[RecordSpec] = []
    for name in RESULT_RECORDS:
        cls = getattr(records, name)
        texts = _DEFINITIONS.get(name, {})
        specs: list[FieldSpec] = []
        for handle in dc.fields(cls):
            declared = handle.type if isinstance(handle.type, str) else str(handle.type)
            optional = "None" in declared
            if handle.name not in texts:
                msg = f"{name}.{handle.name} has no definition; a field cannot be frozen undescribed"
                raise ValueError(msg)
            definition, unit, scope, absent = texts[handle.name]
            if not definition.strip():
                msg = f"{name}.{handle.name} has an empty definition"
                raise ValueError(msg)
            if optional and not absent.strip():
                msg = f"{name}.{handle.name} is optional and must say what its absence means"
                raise ValueError(msg)
            if declared.split(" |")[0] in {"int", "float"} and not (unit.strip() and scope.strip()):
                msg = f"{name}.{handle.name} is numeric and must declare a unit and what it aggregates over"
                raise ValueError(msg)
            specs.append(
                FieldSpec(
                    name=handle.name,
                    type=declared,
                    optional=optional,
                    definition=definition,
                    unit=unit,
                    scope=scope,
                    absent=absent,
                )
            )
        summary = " ".join((cls.__doc__ or "").split()).split(". ")[0].rstrip(".")
        described.append(RecordSpec(name=name, summary=summary, fields=tuple(specs)))
    return tuple(described)


def result_schema() -> ResultSchema:
    """The schema as the code defines it now."""
    return ResultSchema(experiment=EXPERIMENT_LABEL, schema_version=RESULT_SCHEMA_VERSION, records=describe_records())


def schema_to_json(schema: ResultSchema) -> str:
    """Canonical JSON of the frozen schema."""
    return canonical_json(to_mapping(schema))


def load_schema(path: Path) -> ResultSchema:
    """Strictly rebuild the schema from its JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ResultSchema)


def _row(spec: FieldSpec) -> str:
    """One table row. A declared type may contain a union pipe, which would otherwise split the cell."""
    declared = spec.type.replace("|", "\\|")
    return (
        f"| `{spec.name}` | `{declared}` | {spec.unit or '—'} | {spec.scope or '—'} "
        f"| {spec.definition} | {spec.absent or '—'} |"
    )


def render_schema_markdown(schema: ResultSchema) -> str:
    """The Markdown rendering: one compact table per record."""
    lines = [
        f"# Task 1-a manual-demonstration result schema (v{schema.schema_version})",
        "",
        (
            f"Field definitions and units of the machine-readable evidence `{schema.experiment}` writes "
            f"(plan section 7.1). Generated from the records themselves; a changed field fails the regression "
            f"lock rather than refreshing this document, so changing one is a version decision."
        ),
        "",
    ]
    for record in schema.records:
        lines += [
            f"## `{record.name}`",
            "",
            f"{record.summary}.",
            "",
            "| field | type | unit | scope | meaning | absent |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines.extend(_row(spec) for spec in record.fields)
        lines.append("")
    return "\n".join(lines)


def _freeze(args: argparse.Namespace) -> int:
    """Write the frozen schema and its rendering, refusing to replace a version that already exists."""
    schema = result_schema()
    docs = Path(cast("str", args.docs))
    output = docs / f"result_schema_v{schema.schema_version}.json"
    markdown = docs / f"result_schema_v{schema.schema_version}.md"
    for target in (output, markdown):
        if target.exists():
            msg = (
                f"refusing to overwrite {target}: a changed field is a version decision, so raise "
                f"RESULT_SCHEMA_VERSION and freeze the new version beside this one"
            )
            raise FileExistsError(msg)
    docs.mkdir(parents=True, exist_ok=True)
    output.write_text(schema_to_json(schema) + "\n", encoding="utf-8")
    markdown.write_text(render_schema_markdown(schema), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(schema.records),
                "fields": sum(len(record.fields) for record in schema.records),
                "schema_version": schema.schema_version,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Freeze the result schema of the manual-demonstration evidence.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    freeze = subparsers.add_parser("freeze", help="write the frozen schema and its Markdown rendering")
    freeze.add_argument("--docs", type=str, required=True, help="the experiment's documentation directory")
    args = parser.parse_args(argv)
    return _freeze(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
