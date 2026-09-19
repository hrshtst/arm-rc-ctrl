# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009, M3MAN-010: the frozen result schema of the manual-demonstration evidence (plan section 7.1).

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
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL
from arm_rc_ctrl.provenance import canonical_json

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "RECORDS_BY_VERSION",
    "RESULT_RECORDS",
    "RESULT_SCHEMA_VERSION",
    "SUPPORTED_RESULT_SCHEMAS",
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

RESULT_SCHEMA_VERSION: Final = 3
"""Version 3 adds the derived-evidence records; versions 1 and 2 are retained exactly as frozen."""

SUPPORTED_RESULT_SCHEMAS: Final = (1, 2, 3)

_EVALUATION_RECORDS: Final = (
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

_HANDOFF_RECORDS: Final = (
    "ModelAccount",
    "BankAccount",
    "StudyAccounting",
    "IllustrationCase",
    "Selection",
    "ResimulationSubset",
)
"""What the handoff freezes describe: the accounting lines, the illustration rule's output, and the audit sample."""

_DERIVED_RECORDS: Final = (
    "ManualRunRow",
    "ManualContrastRow",
    "ManualContrastSummary",
    "ManualArmSummary",
    "ManualSelection",
    "ManualSelections",
    "ManualFigureRun",
    "ManualFigureCase",
    "ManualFigureInputs",
    "ManualResultInputs",
    "ManualResultTable",
    "ManualResultDocument",
    "ManualResults",
)
"""M3MAN-010's derived evidence: the per-run and per-comparison tables, summaries, selections and figure inputs."""

RECORDS_BY_VERSION: Final[dict[int, tuple[str, ...]]] = {
    1: _EVALUATION_RECORDS,
    2: _EVALUATION_RECORDS + _HANDOFF_RECORDS,
    3: _EVALUATION_RECORDS + _HANDOFF_RECORDS + _DERIVED_RECORDS,
}
"""Each frozen version's records. A version already frozen is never reordered or edited."""

RESULT_RECORDS: Final = RECORDS_BY_VERSION[RESULT_SCHEMA_VERSION]

_EXPERIMENTS: Final = "arm_rc_ctrl.experiments"
_RECORD_MODULES: Final[dict[str, str]] = {
    **dict.fromkeys(_EVALUATION_RECORDS, f"{_EXPERIMENTS}.manual_evaluation"),
    **dict.fromkeys(("ModelAccount", "BankAccount", "StudyAccounting"), f"{_EXPERIMENTS}.manual_accounting"),
    **dict.fromkeys(("IllustrationCase", "Selection"), f"{_EXPERIMENTS}.manual_representative"),
    "ResimulationSubset": f"{_EXPERIMENTS}.manual_resimulation",
    **dict.fromkeys(
        ("ManualContrastRow", "ManualContrastSummary", "ManualArmSummary"), f"{_EXPERIMENTS}.manual_contrasts"
    ),
    **dict.fromkeys(("ManualFigureRun", "ManualFigureCase", "ManualFigureInputs"), f"{_EXPERIMENTS}.manual_figures"),
    **dict.fromkeys(
        (
            "ManualRunRow",
            "ManualSelection",
            "ManualSelections",
            "ManualResultInputs",
            "ManualResultTable",
            "ManualResultDocument",
            "ManualResults",
        ),
        f"{_EXPERIMENTS}.manual_results",
    ),
}
"""A record is resolved against the module that defines it, imported by name when it is described, since
the derived-evidence modules read this schema themselves."""

_PAIRS: Final = "pairs"
_SECONDS: Final = "s"
_MODELS: Final = "the study's 186 models"
_ONE_MODEL: Final = "one model's line"
_ONE_BANK: Final = "one replay bank's line"
_ONE_DOC: Final = "one accounting document"
"""Denominators the handoff records aggregate over, named once so the wording cannot drift."""
_ONE_RUN: Final = "one run"
_ACTIVE: Final = "one run's active segment, activation to last sample"
_RUN_CLOCK: Final = "one run's clock, zero at its first sample"
_TASK_CLOCK: Final = "one run's task clock, zero at activation"
_NOT_RUN: Final = "the run was not simulated: its model has no evidence, or the pair was not executed."
_NO_ACTIVE: Final = "the run was not simulated, or it aborted before activation and has no active segment."
_ONE_ROW: Final = "one comparison row"
_CLASS: Final = "one configuration, tracker and class"
_ONE_TABLE: Final = "one stored table"
_ONE_INDEX: Final = "one results index, over all runs"

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
    "ModelAccount": {
        "label": ("Model label `<configuration>/<arm>`, as the frozen manifest names it.", "", "", ""),
        "configuration": ("Configuration the model belongs to, one of the six frozen.", "", "", ""),
        "arm": ("Arm within the configuration, such as `S/D01`, `M10`, `R10/D01` or `C10/D01`.", "", "", ""),
        "present": (
            "Whether the study holds evidence for this model. False is a recorded absence, not an error.",
            "",
            "",
            "",
        ),
        "identity": (
            "Evidence identity the manifest carries: the digest over the fit identity and the conditions.",
            "",
            "",
            "The model has no evidence yet; `present` is false and every other stated fact is absent with it.",
        ),
        "fit_identity": (
            "Cache identity of the fit this model's evidence was produced from.",
            "",
            "",
            "No evidence exists for this model.",
        ),
        "assignment": (
            "Parent demonstration the model is paired against, such as `D01`.",
            "",
            "",
            "Either the model has no evidence, or it is the all-ten arm, which is paired against no single parent.",
        ),
        "status": (
            "The manifest's verdict for the model: `feasible` when every pair completed, else `infeasible`.",
            "",
            "",
            "No evidence exists for this model.",
        ),
        "n_pairs": (
            "Pairs the model's evidence records. The denominator of the three counts below, which sum to it.",
            "pairs",
            _ONE_MODEL,
            "",
        ),
        "n_completed": ("Pairs that completed and were judged successful.", "pairs", f"{_ONE_MODEL}, of `n_pairs`", ""),
        "n_infeasible": (
            "Pairs that ran and were judged infeasible, by the horizon, dwell, abort or saturation rules.",
            "pairs",
            f"{_ONE_MODEL}, of `n_pairs`",
            "",
        ),
        "n_unexecuted": (
            (
                "Pairs the protocol names but the evidence does not record as run. A completed sweep emits "
                "zero, because every scenario is attempted from a fresh reset and none is skipped after a "
                "failure; an interrupted sweep's missing runs are NOT counted here, because they were never "
                "recorded as pairs."
            ),
            "pairs",
            f"{_ONE_MODEL}, of `n_pairs`",
            "",
        ),
        "execution_identity": (
            "Execution environment the model's runs were keyed in.",
            "",
            "",
            "No evidence exists for this model.",
        ),
        "payload": (
            "Store reference and digest of the manifest this line was read from.",
            "",
            "",
            "No evidence exists for this model.",
        ),
    },
    "BankAccount": {
        "identity": ("Bank identity: the digest over the conditions and the parent.", "", "", ""),
        "assignment": ("Parent demonstration whose direct replay this bank holds.", "", "", ""),
        "warmup_s": (
            "Warm-up the bank was built under; banks of one parent differ by it.",
            "s",
            _ONE_BANK,
            "",
        ),
        "velocity_cutoff_hz": (
            "Causal velocity cutoff replay was driven through, from the paired configuration's estimator.",
            "Hz",
            _ONE_BANK,
            "",
        ),
        "acceleration_cutoff_hz": (
            "Causal acceleration cutoff replay was driven through, from the same estimator.",
            "Hz",
            _ONE_BANK,
            "",
        ),
        "n_pairs": ("Replay pairs the bank records. The denominator of the two counts below.", "pairs", _ONE_BANK, ""),
        "n_completed": ("Replay pairs judged successful.", "pairs", f"{_ONE_BANK}, of `n_pairs`", ""),
        "n_infeasible": ("Replay pairs judged infeasible.", "pairs", f"{_ONE_BANK}, of `n_pairs`", ""),
        "execution_identity": ("Execution environment the bank's runs were keyed in.", "", "", ""),
        "payload": ("Store reference and digest of the bank manifest this line was read from.", "", "", ""),
    },
    "StudyAccounting": {
        "experiment": ("Experiment label the accounting belongs to.", "", "", ""),
        "canonical_execution_identity": (
            "The execution identity every admissible run must be keyed in, for `all_bind_canonical_execution`.",
            "",
            "",
            "",
        ),
        "models": ("One line per study model, present or absent, in the frozen manifest order.", "", "", ""),
        "banks": ("One line per replay bank the study produced.", "", "", ""),
        "n_models": (
            "Model lines the accounting carries: the study's full count, not only those with evidence.",
            "models",
            _MODELS,
            "",
        ),
        "n_present": ("Models whose evidence exists.", "models", f"{_ONE_DOC}, of `n_models`", ""),
        "n_missing": (
            "Models with no evidence. Stated rather than omitted, so an incomplete study is legible as incomplete.",
            "models",
            f"{_ONE_DOC}, of `n_models`",
            "",
        ),
        "missing": ("Labels of the models with no evidence, so the gap is named and not merely counted.", "", "", ""),
        "statuses": (
            "Count of present models by their recorded status, such as `feasible` or `infeasible`.",
            "",
            "",
            "",
        ),
        "n_rc_runs": ("Model pairs summed over every present model.", "pairs", f"{_ONE_DOC}, over present models", ""),
        "n_replay_runs": ("Replay pairs summed over every bank.", "pairs", f"{_ONE_DOC}, over banks", ""),
        "all_bind_canonical_execution": (
            "Whether every present model and bank is keyed in the canonical execution identity.",
            "",
            "",
            "",
        ),
        "complete": ("Whether every study model has evidence. False while any line is absent.", "", "", ""),
        "provenance": ("Reproducibility record of the invocation that produced the accounting.", "", "", ""),
        "schema_version": (
            "Version of the accounting record itself; a changed field is a version decision.",
            "version",
            _ONE_DOC,
            "",
        ),
    },
    "IllustrationCase": {
        "scenario_id": ("Identifier of the scenario the report illustrates.", "", "", ""),
        "categories": (
            "Every category that selected this case, in the rule's declared order; a case may satisfy several.",
            "",
            "",
            "",
        ),
        "arms": (
            "Each comparison arm's verdict on this scenario, so the figure shows all four beside the replay baseline.",
            "",
            "",
            "",
        ),
    },
    "Selection": {
        "cases": ("The cases the frozen rule chose, deduplicated, in the rule's declared order.", "", "", ""),
        "categories": ("Scenario ids each category selected, before deduplication across categories.", "", "", ""),
        "absent": (
            "Categories that occurred nowhere in the results, stated so a reader sees the rule looked and found none.",
            "",
            "",
            "",
        ),
    },
    "ResimulationSubset": {
        "models": ("The models the audit re-simulates, in the frozen manifest order.", "", "", ""),
        "scenarios": (
            "One locked scenario from each perturbation class, in the frozen class order.",
            "",
            "",
            "",
        ),
        "trackers": ("The trackers each model and scenario is run under, in evaluation order.", "", "", ""),
        "n_banks": (
            "Replay banks the chosen models share: one per configuration at the fixed parent.",
            "banks",
            "the re-simulation sample",
            "",
        ),
    },
    "ManualRunRow": {
        "source": ("`rc` for a model's run, `replay` for a direct replay of a demonstration.", "", "", ""),
        "configuration": ("Inherited configuration the run belongs to, one of the six frozen.", "", "", ""),
        "arm": ("Arm as the study names it: `M10`, `S/D01`, `R10/D01`, `C10/D01`, or `replay/D01`.", "", "", ""),
        "arm_kind": ("Arm without its parent: `S`, `M10`, `R10`, `C10` or `replay`.", "", "", ""),
        "parent": (
            "Demonstration the arm belongs to.",
            "",
            "",
            "the all-ten arm, which trains on the whole bank and has no single parent.",
        ),
        "model_label": ("`<configuration>/<arm>`, the model or replay bank the run is of.", "", "", ""),
        "evidence_identity": (
            "Identity of the model evidence or replay bank manifest the run was read from.",
            "",
            "",
            "the model has no evidence, so no manifest describes the run.",
        ),
        "tracker": ("Frozen tracker that ran the pair.", "", "", ""),
        "scenario_id": ("Identifier of the development case.", "", "", ""),
        "scenario_class": (
            "Perturbation class of the case: `nominal`, `posture_small`, `posture_large`, `force` or `combined`.",
            "",
            "",
            "",
        ),
        "scenario_index": ("Position of the case in the frozen scenario order.", "index", "the 65 locked cases", ""),
        "status": (
            (
                "`completed` (met every criterion), `infeasible` (failed at least one), `unexecuted` (recorded as not "
                "run) or `unavailable` (the model has no evidence). Only the first two are runs."
            ),
            "",
            "",
            "",
        ),
        "success": ("The run met every criterion the sweep judges.", "", "", _NOT_RUN),
        "reason": (
            "Why the run did not succeed, naming the cause rather than a consequence.",
            "",
            "",
            "the run succeeded, or it was not simulated.",
        ),
        "initial_q": (
            "Joint angles the run started from, after the case's perturbation, one per joint (rad).",
            "",
            "",
            "the model has no evidence, so no pair records the start.",
        ),
        "horizon_completed": ("The run reached the configured horizon measured from activation.", "", "", _NOT_RUN),
        "termination_kind": ("How the run ended, from the run's own termination record.", "", "", _NOT_RUN),
        "termination_time_s": ("When the run ended.", "s", _RUN_CLOCK, _NOT_RUN),
        "termination_limit": (
            "The limit an abort tripped.",
            "",
            "",
            "the run did not abort on a limit, or it was not simulated.",
        ),
        "termination_joint": (
            "The joint whose limit tripped.",
            "index",
            "joints of the arm, from 0",
            "no joint limit tripped, or the run was not simulated.",
        ),
        "termination_value": (
            "The value that crossed the limit.",
            "the limit's unit: rad, rad/s or N m",
            "the tripping sample",
            "no limit tripped, or the run was not simulated.",
        ),
        "termination_bound": (
            "The bound the value crossed.",
            "the limit's unit: rad, rad/s or N m",
            "the tripped limit",
            "no limit tripped, or the run was not simulated.",
        ),
        "termination_failure": (
            "The failure kind the termination recorded.",
            "",
            "",
            "the run ended without a failure, or it was not simulated.",
        ),
        "termination_detail": (
            "The termination record's free-text detail.",
            "",
            "",
            "the record carries no detail, or the run was not simulated.",
        ),
        "dwell_ok": ("The run ends with a qualifying dwell of the required length.", "", "", _NOT_RUN),
        "dwell_final_s": ("Duration of the dwell that ends at the last sample.", "s", "the final dwell", _NOT_RUN),
        "dwell_longest_s": ("Duration of the longest qualifying dwell.", "s", _ACTIVE, _NOT_RUN),
        "dwell_earliest_start_s": (
            "Start of the first dwell that reached the required length.",
            "s",
            _RUN_CLOCK,
            "no dwell reached the required length, or the run was not simulated.",
        ),
        "departures_after_hold": (
            "Qualifying holds that ended before the last sample: the arm reached the target and left it.",
            "count",
            _ACTIVE,
            _NOT_RUN,
        ),
        "post_pulse_dwell_ok": (
            "The dwell measured from the pulse end satisfies the rule; what a force case must meet.",
            "",
            "",
            "no pulse fired, so the whole active segment is judged, or the run was not simulated.",
        ),
        "post_pulse_dwell_final_s": (
            "Duration of the final dwell measured from the pulse end.",
            "s",
            "the final dwell after the pulse",
            "no pulse fired, or the run was not simulated.",
        ),
        "trigger_ok": (
            "The pulse fired and left room for the dwell it disturbs.",
            "",
            "",
            "the case prescribes no pulse, or the run was not simulated.",
        ),
        "pulse_triggered": (
            "A pulse fired at all.",
            "",
            "",
            "the case prescribes no pulse, or the run was not simulated.",
        ),
        "pulse_start_s": ("When the pulse began.", "s", _RUN_CLOCK, "no pulse fired, or the run was not simulated."),
        "pulse_end_s": ("When the pulse ended.", "s", _RUN_CLOCK, "no pulse fired, or the run was not simulated."),
        "trigger_reason": (
            "Why the case cannot count as a disturbance-recovery test.",
            "",
            "",
            "the timing rule was met, the case prescribes no pulse, or the run was not simulated.",
        ),
        "generated_within_position_limits": (
            "Every generated joint command lay inside the joint limits.",
            "",
            "",
            "a replay run, which carries no readout, or the run was not simulated.",
        ),
        "generated_within_speed_limits": (
            "Every generated joint speed lay inside the speed limits.",
            "",
            "",
            "a replay run, or the run was not simulated.",
        ),
        "generated_within_workspace": (
            "The generated endpoint stayed inside the reachable workspace.",
            "",
            "",
            "a replay run, a generated command outside the joint limits, or a run that was not simulated.",
        ),
        "generated_dwell_ok": (
            "The generated reference itself ends with a qualifying dwell.",
            "",
            "",
            "a replay run, a generated command that was not evaluable, or a run that was not simulated.",
        ),
        "generated_dwell_final_s": (
            "Duration of the generated reference's final dwell.",
            "s",
            "the generated reference's final dwell",
            "a replay run, a generated command that was not evaluable, or a run that was not simulated.",
        ),
        "saturation_fraction": (
            "Share of active samples in which any joint's requested torque reached its limit.",
            "fraction in [0, 1]",
            f"{_ACTIVE}; denominator is its sample count",
            _NOT_RUN,
        ),
        "torque_rms_nm": (
            (
                "Root mean square over active samples of the applied torque vector's Euclidean norm, joints "
                "combined, as the sweep recorded it; it can exceed any single joint's limit."
            ),
            "N m",
            _ACTIVE,
            _NO_ACTIVE,
        ),
        "activation_s": ("When the task activated: the end of the warm-up.", "s", _RUN_CLOCK, _NOT_RUN),
        "duration_s": ("Length of the run as recorded.", "s", _RUN_CLOCK, _NOT_RUN),
        "n_samples": ("Samples the run recorded, warm-up included.", "samples", _ONE_RUN, _NOT_RUN),
        "n_active_samples": ("Samples from activation to the last sample.", "samples", _ACTIVE, _NOT_RUN),
        "final_endpoint_error_m": (
            "Distance from the endpoint to the target at the last sample.",
            "m",
            "the last sample",
            _NO_ACTIVE,
        ),
        "time_to_final_dwell_s": (
            "Time from activation to the start of the dwell that ends the run, when that dwell is long enough.",
            "s",
            _TASK_CLOCK,
            "the run's dwell failed, or the run was not simulated.",
        ),
        "departure_latency_s": (
            (
                "Time from activation to the first sample whose endpoint lies farther than the departure radius "
                "(the task's 1 cm dwell radius) from its position at activation."
            ),
            "s",
            _TASK_CLOCK,
            "the endpoint never left the departure radius, or the run has no active segment.",
        ),
        "peak_speed_rad_s": ("Largest measured joint speed magnitude.", "rad/s", f"{_ACTIVE}, over joints", _NO_ACTIVE),
        "peak_acceleration_rad_s2": (
            (
                "Largest measured joint acceleration magnitude: the finite difference of measured velocity between "
                "consecutive active samples."
            ),
            "rad/s^2",
            f"{_ACTIVE}, over joints",
            _NO_ACTIVE,
        ),
        "peak_reference_speed_rad_s": (
            (
                "Largest commanded joint speed magnitude: the generated reference for an RC run, the replayed "
                "recording through the causal estimator for a replay run."
            ),
            "rad/s",
            f"{_ACTIVE}, over joints",
            _NO_ACTIVE,
        ),
        "peak_reference_acceleration_rad_s2": (
            "Largest commanded joint acceleration magnitude, from the same reference.",
            "rad/s^2",
            f"{_ACTIVE}, over joints",
            _NO_ACTIVE,
        ),
        "activation_jump_rad": (
            "Euclidean norm of the first active command minus the measured posture at activation.",
            "rad",
            "the activation sample",
            _NO_ACTIVE,
        ),
        "tracking_error_rms_rad": (
            "Root mean square of the stored tracking error, over joints and samples.",
            "rad",
            _ACTIVE,
            _NO_ACTIVE,
        ),
        "tracking_error_max_rad": (
            "Largest stored tracking-error magnitude.",
            "rad",
            f"{_ACTIVE}, over joints",
            _NO_ACTIVE,
        ),
        "torque_peak_nm": (
            "Largest applied joint torque magnitude.",
            "N m",
            f"{_ACTIVE}, over joints",
            _NO_ACTIVE,
        ),
        "run_artifact_id": ("Identity of the stored run.", "", "", _NOT_RUN),
        "run_uri": ("Store location of the run summary.", "", "", _NOT_RUN),
        "run_sha256": ("SHA-256 of the run summary as written.", "", "", _NOT_RUN),
        "run_size": ("Size of the run summary.", "bytes", "one run summary", _NOT_RUN),
        "arrays_sha256": ("SHA-256 of the run's arrays file, as the summary records it.", "", "", _NOT_RUN),
        "sources": (
            "The demonstrations the run's model trained on (the replayed one for a replay run), by artifact ID.",
            "",
            "",
            _NOT_RUN,
        ),
        "fit_identity": (
            "Identity of the fit the run's model was produced from.",
            "",
            "",
            "a replay run, which has no fit, or a model with no evidence.",
        ),
    },
    "ManualContrastRow": {
        "configuration": ("Configuration both arms belong to; comparisons never pool configurations.", "", "", ""),
        "tracker": ("Tracker both arms ran under; comparisons never pool trackers.", "", "", ""),
        "scenario_class": ("Class of scenarios compared, or `all` for the 65 together.", "", "", ""),
        "contrast": (
            (
                "`M10-S`, `M10-R10`, `C10-R10`, `M10-C10` (plan section 6), or an arm against its parent's replay: "
                "`S-replay`, `R10-replay`, `C10-replay`, `M10-replay`."
            ),
            "",
            "",
            "",
        ),
        "parent": (
            "The parent whose arms are compared; the all-ten arm is the same model for every parent.",
            "",
            "",
            "",
        ),
        "arm_a": ("First arm, as the study names it; the difference is a minus b.", "", "", ""),
        "arm_b": ("Second arm, as the study names it.", "", "", ""),
        "status": (
            (
                "`complete` when both arms have every scenario of the class, `partial` when they share only some, "
                "`unavailable` when they share none."
            ),
            "",
            "",
            "",
        ),
        "n_scenarios": ("Scenarios the class holds.", "scenarios", _ONE_ROW, ""),
        "n_shared": (
            "Scenarios both arms were run on: the denominator of every count below.",
            "scenarios",
            f"{_ONE_ROW}, of `n_scenarios`",
            "",
        ),
        "successes_a": ("Shared scenarios arm a succeeded on.", "scenarios", f"{_ONE_ROW}, of `n_shared`", ""),
        "successes_b": ("Shared scenarios arm b succeeded on.", "scenarios", f"{_ONE_ROW}, of `n_shared`", ""),
        "difference": (
            "Paired success-count difference, `successes_a - successes_b`, equal to `improved - worsened`.",
            "scenarios",
            f"{_ONE_ROW}, of `n_shared`",
            "",
        ),
        "improved": (
            "Shared scenarios arm a succeeded on and arm b failed.",
            "scenarios",
            f"{_ONE_ROW}, of `n_shared`",
            "",
        ),
        "worsened": (
            "Shared scenarios arm b succeeded on and arm a failed.",
            "scenarios",
            f"{_ONE_ROW}, of `n_shared`",
            "",
        ),
        "tied_success": ("Shared scenarios both arms succeeded on.", "scenarios", f"{_ONE_ROW}, of `n_shared`", ""),
        "tied_failure": ("Shared scenarios both arms failed.", "scenarios", f"{_ONE_ROW}, of `n_shared`", ""),
    },
    "ManualContrastSummary": {
        "configuration": ("Configuration of the ten comparisons.", "", "", ""),
        "tracker": ("Tracker of the ten comparisons.", "", "", ""),
        "scenario_class": ("Class of scenarios compared, or `all`.", "", "", ""),
        "contrast": ("The contrast summarised, as in `ManualContrastRow`.", "", "", ""),
        "arm_a": ("Kind of the first arm.", "", "", ""),
        "arm_b": ("Kind of the second arm.", "", "", ""),
        "n_scenarios": ("Scenarios each parent's comparison is over.", "scenarios", "one parent's comparison", ""),
        "n_parents": (
            "Parents whose comparison is complete: the denominator of the figures below.",
            "parents",
            "the ten parents",
            "",
        ),
        "differences": (
            (
                "Every parent's paired success-count difference in parent order D01 to D10, absent (null) where "
                "that parent's comparison is not complete."
            ),
            "",
            "",
            "never absent as a whole; an element is null where that parent's comparison is not complete.",
        ),
        "median": (
            "Median of the complete differences.",
            "scenarios",
            "complete parents, of `n_parents`",
            "no parent's comparison is complete.",
        ),
        "minimum": (
            "Smallest complete difference.",
            "scenarios",
            "complete parents, of `n_parents`",
            "no parent's comparison is complete.",
        ),
        "maximum": (
            "Largest complete difference.",
            "scenarios",
            "complete parents, of `n_parents`",
            "no parent's comparison is complete.",
        ),
        "parents_improved": ("Parents whose difference is positive.", "parents", "of `n_parents`", ""),
        "parents_worsened": ("Parents whose difference is negative.", "parents", "of `n_parents`", ""),
        "parents_tied": ("Parents whose difference is zero.", "parents", "of `n_parents`", ""),
    },
    "ManualArmSummary": {
        "configuration": ("Configuration of the arm.", "", "", ""),
        "tracker": ("Tracker the runs ran under.", "", "", ""),
        "scenario_class": ("Class of scenarios summed over, or `all`.", "", "", ""),
        "arm_kind": (
            "`S`, `M10`, `R10`, `C10` or `replay`; the all-ten arm is one model and counted once.",
            "",
            "",
            "",
        ),
        "n_scenarios": ("Scenarios the class holds.", "scenarios", _CLASS, ""),
        "n_models": (
            (
                "Models, or replay banks, with a verdict on every scenario of the class: ten for a parented arm, "
                "one for the all-ten arm."
            ),
            "models",
            _CLASS,
            "",
        ),
        "n_runs": (
            "Runs in the total: `n_models` times `n_scenarios`. The denominator of `successes`.",
            "runs",
            _CLASS,
            "",
        ),
        "excluded_runs": (
            "Existing runs of models left out of the total because another of their runs in the class is missing.",
            "runs",
            _CLASS,
            "",
        ),
        "unavailable_runs": ("Runs the protocol names that do not exist.", "runs", _CLASS, ""),
        "successes": ("Runs in the total that met every criterion.", "runs", f"{_CLASS}, of `n_runs`", ""),
        "per_model": (
            (
                "Each model's successes over the class, in parent order (one entry for the all-ten arm), absent (null) "
                "for a model left out."
            ),
            "",
            "",
            "never absent as a whole; an element is null for a model left out of the total.",
        ),
        "median": ("Median of the per-model successes.", "runs", "models in the total", "no model is in the total."),
        "minimum": ("Smallest per-model success count.", "runs", "models in the total", "no model is in the total."),
        "maximum": ("Largest per-model success count.", "runs", "models in the total", "no model is in the total."),
    },
    "ManualSelection": {
        "configuration": ("Configuration the frozen rule was applied to.", "", "", ""),
        "tracker": ("Tracker the frozen rule was applied under.", "", "", ""),
        "arms": ("The compared arm labels, as the rule was frozen with them.", "", "", ""),
        "omitted": (
            "Scenarios left out because one of the compared arms has no verdict there; empty in a complete study.",
            "",
            "",
            "",
        ),
        "selection": ("The rule's selection for this configuration and tracker.", "", "", ""),
    },
    "ManualSelections": {
        "experiment": ("Experiment label the selections belong to.", "", "", ""),
        "rule_sha256": ("SHA-256 of the frozen representative rule applied.", "", "", ""),
        "ordering_sha256": ("SHA-256 of the run ordering whose scenario order broke ties.", "", "", ""),
        "applications": ("One application per configuration and tracker, in the rule's order.", "", "", ""),
        "schema_version": ("Version of the selections record.", "version", "one selections document", ""),
    },
    "ManualFigureRun": {
        "role": ("`S`, `M10`, `R10`, `C10` or `replay`: the run's place in the comparison.", "", "", ""),
        "label": ("Model or replay-bank label of the run.", "", "", ""),
        "success": ("The run's verdict.", "", "", ""),
        "pulse_start_s": ("When the run's pulse began.", "s", _RUN_CLOCK, "no pulse fired in this run."),
        "pulse_end_s": ("When the run's pulse ended.", "s", _RUN_CLOCK, "no pulse fired in this run."),
        "run": ("The stored run, bound by the digests of its summary and arrays.", "", "", ""),
    },
    "ManualFigureCase": {
        "case_id": ("`<configuration>__<tracker>__<scenario>`, how a rendering command names the case.", "", "", ""),
        "configuration": ("Configuration of the case.", "", "", ""),
        "tracker": ("Tracker of the case.", "", "", ""),
        "scenario_id": ("Scenario of the case.", "", "", ""),
        "scenario_class": ("Perturbation class of the scenario.", "", "", ""),
        "categories": ("Every rule category that chose the case, in the rule's order.", "", "", ""),
        "activation_s": ("Activation time shared by the case's runs: task time zero.", "s", _RUN_CLOCK, ""),
        "teacher_assignment": ("Parent whose recorded demonstration the figure draws.", "", "", ""),
        "teacher": ("The recorded demonstration, bound by its record and payload digest.", "", "", ""),
        "runs": ("The runs drawn: the rule's four arms and the parent's replay, where they exist.", "", "", ""),
        "missing": ("Roles the case names whose runs do not exist; empty in a complete study.", "", "", ""),
    },
    "ManualFigureInputs": {
        "experiment": ("Experiment label the inputs belong to.", "", "", ""),
        "scenario_file": ("Repository-relative path of the manual task configuration used for kinematics.", "", "", ""),
        "scenario_sha256": ("SHA-256 of that task configuration.", "", "", ""),
        "cases": ("Every case the frozen rule chose, in selection order.", "", "", ""),
        "schema_version": ("Version of the figure-inputs record.", "version", "one figure-inputs document", ""),
    },
    "ManualResultInputs": {
        "study_manifest_sha256": ("SHA-256 of the frozen study manifest.", "", "", ""),
        "evaluation_sha256": ("SHA-256 of the evaluation configuration every run was keyed under.", "", "", ""),
        "run_ordering_sha256": ("SHA-256 of the frozen run ordering.", "", "", ""),
        "representative_rule_sha256": ("SHA-256 of the frozen representative-case rule.", "", "", ""),
        "result_schema_sha256": ("SHA-256 of the result schema these outputs follow.", "", "", ""),
        "evidence_sha256": (
            "SHA-256 over every evidence pointer's file name and file digest, in name order.",
            "",
            "",
            "",
        ),
        "n_pointers": ("Evidence pointers read.", "pointers", "the evidence directory", ""),
    },
    "ManualResultTable": {
        "name": ("Name of the table, with its version.", "", "", ""),
        "record": ("Result-schema record each row follows.", "", "", ""),
        "payload": ("Store location, SHA-256 and size of the CSV file.", "", "", ""),
        "n_rows": ("Rows the table holds, header excluded.", "rows", _ONE_TABLE, ""),
        "columns": ("The header, in order: the record's fields.", "", "", ""),
    },
    "ManualResultDocument": {
        "name": ("File name, beside the results index.", "", "", ""),
        "record": ("Result-schema record the document holds.", "", "", ""),
        "sha256": ("SHA-256 of the file as written.", "", "", ""),
        "size": ("Size of the file.", "bytes", "one document", ""),
        "n_entries": (
            "Rows of a CSV document, or the lines, applications or cases of a JSON one.",
            "entries",
            "one document",
            "",
        ),
    },
    "ManualResults": {
        "experiment": ("Experiment label the derived evidence belongs to.", "", "", ""),
        "version": (
            "Version of the derived outputs; a changed output is a new version beside this one.",
            "version",
            "one results index",
            "",
        ),
        "result_schema_version": (
            "Result schema version whose records the outputs follow.",
            "version",
            "one results index",
            "",
        ),
        "inputs": ("The frozen inputs read, bound by digest.", "", "", ""),
        "tables": ("The tables kept in the external store.", "", "", ""),
        "documents": ("The committed documents beside this index.", "", "", ""),
        "n_models": ("Study models with evidence.", "models", "the study's 186 models", ""),
        "n_replay_banks": ("Replay banks keyed to a configuration and parent.", "banks", "the study's 60 banks", ""),
        "n_rc_runs": ("RC runs simulated.", "runs", _ONE_INDEX, ""),
        "n_replay_runs": ("Replay runs simulated.", "runs", _ONE_INDEX, ""),
        "n_unavailable_runs": ("Rows naming a run that was not simulated.", "runs", _ONE_INDEX, ""),
        "n_rc_successes": ("RC runs that met every criterion.", "runs", f"{_ONE_INDEX}, of `n_rc_runs`", ""),
        "n_replay_successes": (
            "Replay runs that met every criterion.",
            "runs",
            f"{_ONE_INDEX}, of `n_replay_runs`",
            "",
        ),
        "departure_radius_m": (
            "Radius used for `departure_latency_s`: the dwell radius every run was judged under.",
            "m",
            "every run",
            "",
        ),
        "command": ("The command line that derived the outputs.", "", "", ""),
        "provenance": ("Reproducibility record of that invocation.", "", "", ""),
        "schema_version": ("Version of the results index record itself.", "version", "one results index", ""),
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
        if self.schema_version not in SUPPORTED_RESULT_SCHEMAS or self.experiment != EXPERIMENT_LABEL:
            msg = f"unsupported result schema {self.schema_version} or experiment {self.experiment!r}"
            raise ValueError(msg)
        if [record.name for record in self.records] != list(RECORDS_BY_VERSION[self.schema_version]):
            msg = (
                f"a schema {self.schema_version} document describes exactly that version's records, "
                f"in their declared order"
            )
            raise ValueError(msg)


def describe_records(version: int = RESULT_SCHEMA_VERSION) -> tuple[RecordSpec, ...]:
    """Derive one version's records from the dataclasses, refusing any field that was never described."""
    if version not in SUPPORTED_RESULT_SCHEMAS:
        msg = f"unsupported result schema {version}; known versions are {SUPPORTED_RESULT_SCHEMAS}"
        raise ValueError(msg)
    described: list[RecordSpec] = []
    for name in RECORDS_BY_VERSION[version]:
        cls = getattr(importlib.import_module(_RECORD_MODULES[name]), name)
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


def result_schema(version: int = RESULT_SCHEMA_VERSION) -> ResultSchema:
    """The schema as the code defines it, at one frozen version."""
    return ResultSchema(
        experiment=EXPERIMENT_LABEL,
        schema_version=version,
        records=describe_records(version),
    )


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
    schema = result_schema(int(cast("int", args.schema_version)))
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
    freeze.add_argument(
        "--schema-version",
        dest="schema_version",
        type=int,
        default=RESULT_SCHEMA_VERSION,
        help="the version to freeze; an already frozen version is refused rather than replaced",
    )
    args = parser.parse_args(argv)
    return _freeze(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
