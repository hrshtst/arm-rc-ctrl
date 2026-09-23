# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-005: freeze the three highest nominal scores once the search has finished.

The freeze is the last thing the search's evidence decides, so it is taken only
when that evidence is complete and consistent:

* the search has **stopped**: no reservation is pending, and a cap is spent
  (the 100-trial cap, or the shared time or storage ceiling, which is reported
  as the reason rather than enlarged);
* the Optuna study and the retained records **agree** trial by trial: the same
  numbers, a COMPLETE trial for each scored outcome with the same value, a
  FAIL for each failed one, nothing still running, and the parameters the
  study recorded are the point the trial reserved;
* each chosen trial's evidence is **verified again** against inputs
  reconstructed from the protocol and the study, and its recounted score must
  equal the one recorded.

Selection itself is `select_configurations`: descending nominal score, then the
earliest trial, one configuration per distinct point, unscored candidates never
chosen, and a shortfall reported rather than filled. Only nominal success and
trial order enter it; no perturbed case has been evaluated when it runs. The
record calls the result the *highest nominal scores*, never the best
configurations: a two-run score does not rank quality or robustness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from optuna.trial import TrialState

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_sampled import SampledPoint, sampled_point
from arm_rc_ctrl.experiments.manual_search import (
    REQUIRED_CONFIGURATIONS,
    REQUIRED_RUNS,
    SELECTION_LABEL,
    ScoredTrial,
    load_manual_search,
    protocol_digest,
    select_configurations,
)
from arm_rc_ctrl.experiments.manual_search_run import (
    STUDY_NAME,
    BudgetLedger,
    SearchInputs,
    TrialOutcome,
    TrialReservation,
    TrialResult,
    ledger_of,
    pending_reservations,
    read_record,
    read_result,
    search_stopped,
    trial_directories,
    verified_outcome,
)
from arm_rc_ctrl.experiments.studies import PrunerSpec, close_study, open_study
from arm_rc_ctrl.provenance import canonical_json, sha256_bytes
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from arm_rc_ctrl.experiments.manual_search import ManualSearchProtocol

__all__ = [
    "FrozenConfiguration",
    "ManualSearchFreeze",
    "freeze_digest",
    "freeze_mismatches",
    "freeze_search",
    "read_freeze",
    "render_freeze",
    "reverify_outcome",
    "study_mismatches",
    "write_freeze",
]

SCHEMA: Final = 1


@dataclass(frozen=True)
class FrozenConfiguration:
    """One frozen configuration: its rank, the trial that represents it, and where its evidence lives."""

    rank: int
    trial: int
    configuration: str
    point: SampledPoint
    score: float
    successes: int
    runs: int
    fit_identity: str
    evidence_identity: str
    evidence: str
    """The model evidence manifest the search verified, re-verified when the freeze was taken."""


def freeze_mismatches(
    chosen: Sequence[FrozenConfiguration], *, shortfall: int, n_configurations: int, label: str
) -> list[str]:
    """Why a set of frozen configurations does not follow the selection rule, if it does not.

    Checked where a freeze is constructed, so a record that was edited, or
    assembled any other way than by the rule, cannot be loaded as one.
    """
    found: list[str] = []
    if label != SELECTION_LABEL:
        found.append(f"the frozen configurations are the {SELECTION_LABEL!r}, not {label!r}")
    if n_configurations != REQUIRED_CONFIGURATIONS:
        found.append(f"{REQUIRED_CONFIGURATIONS} configurations are frozen, not {n_configurations}")
    if shortfall < 0 or len(chosen) + shortfall != n_configurations:
        found.append(f"{len(chosen)} chosen with a shortfall of {shortfall} is not {n_configurations}")
    if [item.rank for item in chosen] != list(range(1, len(chosen) + 1)):
        found.append(f"ranks run 1, 2, 3 in order, got {[item.rank for item in chosen]}")
    order = [(-item.score, item.trial) for item in chosen]
    if order != sorted(order) or len({item.trial for item in chosen}) != len(chosen):
        found.append("the chosen trials are ordered by descending score, then ascending trial, each once")
    if len({item.point for item in chosen}) != len(chosen):
        found.append("the chosen configurations are distinct parameter points")
    for item in chosen:
        if item.runs != REQUIRED_RUNS or not 0 <= item.successes <= item.runs:
            found.append(f"trial {item.trial}: {item.successes} of {item.runs} is not a two-tracker result")
        elif item.score != item.successes / item.runs:
            found.append(f"trial {item.trial}: score {item.score} is not {item.successes} of {item.runs}")
    return found


@dataclass(frozen=True)
class ManualSearchFreeze:
    """The frozen selection and the search it was taken from."""

    schema: int
    protocol_sha256: str
    study: str
    label: str
    order: str
    n_configurations: int
    trial_cap: int
    finalized: int
    scored: int
    failed: int
    score_counts: dict[str, int]
    """Scored trials by nominal success fraction, keyed as written ("0.0", "0.5", "1.0")."""
    stopped: tuple[str, ...]
    """Why the search stopped: the caps that were spent."""
    ledger: BudgetLedger
    chosen: tuple[FrozenConfiguration, ...]
    shortfall: int

    def __post_init__(self) -> None:
        """A freeze follows the selection rule and accounts for every trial it was taken from."""
        found = freeze_mismatches(
            self.chosen, shortfall=self.shortfall, n_configurations=self.n_configurations, label=self.label
        )
        if self.schema != SCHEMA:
            found.append(f"schema {SCHEMA}, got {self.schema}")
        if self.scored + self.failed != self.finalized or sum(self.score_counts.values()) != self.scored:
            found.append(f"{self.scored} scored and {self.failed} failed do not make {self.finalized} trials")
        if self.finalized != self.ledger.trials:
            found.append(f"the ledger counts {self.ledger.trials} trials, the freeze {self.finalized}")
        if not self.stopped:
            found.append("a freeze is taken from a search that stopped, and records why")
        if found:
            raise ValueError("; ".join(found))


def study_mismatches(
    protocol: ManualSearchProtocol,
    study_trials: Mapping[int, tuple[str, float | None, Mapping[str, object]]],
    outcomes: Mapping[int, TrialOutcome],
    reservations: Mapping[int, TrialReservation],
) -> list[str]:
    """Every way the Optuna study and the retained records disagree; empty when they are the same search."""
    found: list[str] = []
    if set(study_trials) != set(outcomes):
        found.append(f"the study holds trials {sorted(study_trials)}, the records {sorted(outcomes)}")
    for number in sorted(set(study_trials) & set(outcomes)):
        state, value, params = study_trials[number]
        outcome = outcomes[number]
        expected = TrialState.COMPLETE.name if outcome.state == "scored" else TrialState.FAIL.name
        if state != expected:
            found.append(f"trial {number} is {state} in the study but {outcome.state} in the records")
        elif outcome.state == "scored" and value != outcome.score:
            found.append(f"trial {number} has value {value} in the study but score {outcome.score} in the records")
        reservation = reservations.get(number)
        if reservation is not None:
            try:
                recorded = sampled_point(protocol.space, cast("dict[str, float]", dict(params)))
            except (ValueError, KeyError) as error:
                found.append(f"trial {number}'s study parameters are not a point of the space: {error}")
                continue
            if recorded != reservation.point:
                found.append(f"trial {number} reserved {reservation.point}, the study recorded {recorded}")
        elif outcome.state == "scored":
            found.append(f"trial {number} is scored without a reservation naming its point")
    return found


def reverify_outcome(
    store: StorageRoot,
    reservation: TrialReservation,
    outcome: TrialOutcome,
    result: TrialResult,
    *,
    protocol: ManualSearchProtocol,
    inputs: SearchInputs,
) -> None:
    """Verify one trial's evidence again and require the verdict the search recorded.

    The recorded outcome is what the search concluded; the freeze does not
    take it on trust. The worker's report is scored afresh, against inputs
    reconstructed from the protocol and the study, and every field a
    selection depends on must come out the same.
    """
    again = verified_outcome(store, reservation, result, seconds=outcome.seconds, protocol=protocol, inputs=inputs)
    fields = ("state", "score", "successes", "runs", "evidence")
    differing = [name for name in fields if getattr(again, name) != getattr(outcome, name)]
    if differing:
        msg = f"trial {reservation.trial}: re-verification disagrees with the record on {', '.join(differing)}"
        raise ValueError(msg)


def trial_directories_by_number(store: StorageRoot) -> dict[int, Path]:
    """Every retained trial directory, keyed by its trial number."""
    return {int(directory.name.removeprefix("trial-")): directory for directory in trial_directories(store)}


def _study_trials(
    protocol: ManualSearchProtocol, store: StorageRoot
) -> dict[int, tuple[str, float | None, Mapping[str, object]]]:
    study = open_study(
        store,
        STUDY_NAME,
        protocol_sha256=protocol_digest(protocol),
        sampler=protocol.sampler,
        pruner=PrunerSpec(kind="none"),
        direction="maximize",
    )
    try:
        return {trial.number: (trial.state.name, trial.value, dict(trial.params)) for trial in study.trials}
    finally:
        close_study(study)


def freeze_search(protocol: ManualSearchProtocol, store: StorageRoot, *, inputs: SearchInputs) -> ManualSearchFreeze:
    """Freeze the highest nominal scores of a finished, consistent search, verifying each one again."""
    pending = pending_reservations(store)
    if pending:
        msg = f"trials {[r.trial for r in pending]} are still pending: finish the search before freezing it"
        raise ValueError(msg)
    ledger = ledger_of(store)
    stopped = search_stopped(ledger, protocol.budget)
    if not stopped:
        msg = (
            f"the search has {ledger.trials} of {protocol.budget.trials} trials and no cap is spent: "
            "a freeze is taken only when the search has stopped"
        )
        raise ValueError(msg)
    directories = trial_directories_by_number(store)
    outcomes = {
        number: read_record(directory / "outcome.json", TrialOutcome)
        for number, directory in directories.items()
        if (directory / "outcome.json").is_file()
    }
    reservations = {
        number: read_record(directory / "reservation.json", TrialReservation)
        for number, directory in directories.items()
        if (directory / "reservation.json").is_file()
    }
    disagreements = study_mismatches(protocol, _study_trials(protocol, store), outcomes, reservations)
    if disagreements:
        raise ValueError("the study and the records disagree: " + "; ".join(disagreements))
    candidates = [
        ScoredTrial(
            number=number,
            score=outcomes[number].score,
            point=cast("dict[str, float]", to_mapping(reservation.point)),
        )
        for number, reservation in sorted(reservations.items())
    ]
    selection = select_configurations(candidates, n_configurations=protocol.selection.n_configurations)
    chosen: list[FrozenConfiguration] = []
    for rank, trial in enumerate(selection.chosen, start=1):
        reservation, outcome = reservations[trial.number], outcomes[trial.number]
        result = read_result(directories[trial.number] / "result.json")
        reverify_outcome(store, reservation, outcome, result, protocol=protocol, inputs=inputs)
        chosen.append(
            FrozenConfiguration(
                rank=rank,
                trial=trial.number,
                configuration=reservation.configuration,
                point=reservation.point,
                score=cast("float", outcome.score),
                successes=outcome.successes,
                runs=outcome.runs,
                fit_identity=reservation.fit_identity,
                evidence_identity=reservation.evidence_identity,
                evidence=cast("str", outcome.evidence),
            )
        )
    scored = [outcome for outcome in outcomes.values() if outcome.state == "scored"]
    return ManualSearchFreeze(
        schema=SCHEMA,
        protocol_sha256=protocol_digest(protocol),
        study=STUDY_NAME,
        label=protocol.selection.label,
        order=protocol.selection.order,
        n_configurations=protocol.selection.n_configurations,
        trial_cap=protocol.budget.trials,
        finalized=len(outcomes),
        scored=len(scored),
        failed=len(outcomes) - len(scored),
        score_counts=dict(sorted(Counter(repr(cast("float", o.score)) for o in scored).items())),
        stopped=tuple(stopped),
        ledger=ledger,
        chosen=tuple(chosen),
        shortfall=selection.shortfall,
    )


def freeze_json(freeze: ManualSearchFreeze) -> str:
    """The freeze as canonical JSON."""
    return canonical_json(to_mapping(freeze)) + "\n"


def freeze_digest(freeze: ManualSearchFreeze) -> str:
    """The digest the comparison binds: the freeze's canonical JSON."""
    return sha256_bytes(freeze_json(freeze).encode("utf-8"))


def write_freeze(path: Path, freeze: ManualSearchFreeze) -> None:
    """Write a freeze once; a frozen selection is versioned, never rewritten."""
    if path.exists():
        msg = f"{path} already holds a freeze; a frozen selection is never rewritten"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(freeze_json(freeze), encoding="utf-8")


def read_freeze(path: Path) -> ManualSearchFreeze:
    """Read a freeze strictly: a record that breaks the selection rule is refused where it is loaded."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ManualSearchFreeze)


def render_freeze(freeze: ManualSearchFreeze) -> str:
    """A human-readable account of the freeze; the JSON is the record."""
    counts = ", ".join(f"{count} at {score}" for score, count in freeze.score_counts.items())
    lines = [
        "<!-- Generated by `python -m arm_rc_ctrl.experiments.manual_search_freeze`; do not edit. -->",
        "",
        "# M3MS-005 configuration freeze",
        "",
        (
            f"Protocol `{freeze.protocol_sha256[:12]}`, freeze digest `{freeze_digest(freeze)[:12]}`. "
            f"The search finalized {freeze.finalized} of its {freeze.trial_cap} trials "
            f"({freeze.scored} scored: {counts}; {freeze.failed} failed) and stopped because "
            f"{'; '.join(freeze.stopped)}. It charged {freeze.ledger.seconds / 3600.0:.2f} h and "
            f"{freeze.ledger.stored_bytes / 1024**3:.3f} GiB of the shared ceiling."
        ),
        "",
        (
            f"These are the {freeze.n_configurations} {freeze.label}: descending nominal success fraction over the "
            "two fixed trackers, then the earliest trial, one per distinct parameter point. The score is a two-run "
            "nominal verdict; it does not rank trajectory quality or robustness, and no perturbed case has been "
            "evaluated. Each chosen trial's evidence was verified again when the freeze was taken."
        ),
        "",
        (
            "| Rank | Trial | Score | Neurons | Spectral radius | Sparsity | Leak rate | Input scaling | alpha_0 | "
            "Warm-up (s) |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {c.rank} | {c.trial} | {c.score:g} | {c.point.n_neurons} | {c.point.spectral_radius:.6g} | "
        f"{c.point.sparsity:.6g} | {c.point.leak_rate:.6g} | {c.point.input_scaling:.6g} | "
        f"{c.point.alpha_0:.6g} | {c.point.warmup_s:g} |"
        for c in freeze.chosen
    ]
    lines += [
        "",
        "Shortfall: "
        + ("none." if not freeze.shortfall else f"{freeze.shortfall}; the cap was not enlarged to fill it."),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: freeze the finished search's highest nominal scores."""
    parser = argparse.ArgumentParser(description="Freeze the manual ESN search's highest nominal scores.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="the freeze as JSON; never overwritten")
    parser.add_argument("--markdown", required=True, type=Path, help="the freeze rendered for reading")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    output, markdown = cast("Path", args.output), cast("Path", args.markdown)
    for path in (output, markdown):
        if path.exists():
            msg = f"{path} already exists; a frozen selection is never rewritten"
            raise FileExistsError(msg)
    protocol = load_manual_search(cast("Path", args.protocol))
    inputs = SearchInputs(protocol, root=repository_root())
    frozen = freeze_search(protocol, open_storage(), inputs=inputs)
    write_freeze(output, frozen)
    markdown.write_text(render_freeze(frozen), encoding="utf-8")
    print(render_freeze(frozen))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
