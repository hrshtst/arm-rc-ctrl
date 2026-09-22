# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MS-003: the resumable nominal-only search, its scope guards and its budget accounting.

The parent opens (or resumes) one Optuna study keyed by the protocol's identity
and runs trials **serially**. Each trial draws a point from the approved space,
turns it into a sampled configuration, and hands it to a worker **process**,
which fits the all-ten arm and evaluates the nominal case under both fixed
trackers. The worker's score is the nominal success fraction; nothing else
reaches the optimizer.

Two properties are enforced on both sides of that boundary rather than assumed:

* **Scope.** The parent restricts the evaluation to ``("nominal",)`` and the
  worker checks the scenarios it was actually given. A parent-side preflight
  alone would not survive a worker invoked by hand or by a resumed study.
* **Budget.** Trials, elapsed seconds and stored bytes accumulate on the study
  itself, so a resumed invocation continues the same accounting instead of
  starting a fresh allowance. Work stops at a cap; it is never enlarged.

Numerical work runs in the worker process, which inherits the whole canonical
affinity set with one thread each (plan section 7). Worker count is never
implemented by narrowing a worker's CPU set: the execution identity includes it,
and evidence produced under a narrowed set is not this study's.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

from optuna.trial import TrialState

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.manual_evaluation import prepare_runner
from arm_rc_ctrl.experiments.manual_sampled import (
    SampledPoint,
    arm_of,
    sampled_configuration,
    sampled_entry,
    sampled_point,
)
from arm_rc_ctrl.experiments.manual_search import (
    NOMINAL_SCENARIOS,
    load_manual_search,
    nominal_success_fraction,
    protocol_digest,
)
from arm_rc_ctrl.experiments.studies import PrunerSpec, close_study, open_study
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot, open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optuna

    from arm_rc_ctrl.experiments.manual_search import ManualSearchBudget, ManualSearchProtocol, ManualSearchSpace

__all__ = [
    "STUDY_NAME",
    "BudgetLedger",
    "TrialResult",
    "budget_complaints",
    "evaluate_trial",
    "ledger_of",
    "nominal_scope_mismatches",
    "read_result",
    "record_trial",
    "run_search",
    "spawn_trial",
    "suggest_sampled_point",
    "trial_command",
]

STUDY_NAME: Final = "manual-esn-search-v1"
"""The study's name in the Optuna bucket; its identity is the protocol digest, checked on resume."""
_TRIALS_ATTR: Final = "armrc.spent_trials"
_SECONDS_ATTR: Final = "armrc.spent_seconds"
_BYTES_ATTR: Final = "armrc.spent_bytes"
_MODULE: Final = "arm_rc_ctrl.experiments.manual_search_run"
_GIB: Final = 1024**3


def nominal_scope_mismatches(scenarios: Sequence[str]) -> list[str]:
    """Why a scenario list is not the optimizer's nominal-only scope, if it is not.

    Checked by the parent before it schedules anything and by the worker with
    the list it actually received: a perturbed case reaching the objective
    would undo the separation the whole experiment rests on.
    """
    actual = tuple(scenarios)
    if actual != NOMINAL_SCENARIOS:
        return [
            (
                f"the optimizer evaluates {NOMINAL_SCENARIOS} and nothing else, got {actual}: "
                "no perturbed scenario may reach the score, a tie-break or pruning"
            )
        ]
    return []


@dataclass(frozen=True)
class BudgetLedger:
    """What this search has spent so far, across every invocation that resumed it."""

    trials: int
    seconds: float
    stored_bytes: int

    def __post_init__(self) -> None:
        """Spend is never negative, and a resumed ledger never forgets what was already spent."""
        if self.trials < 0 or self.seconds < 0.0 or self.stored_bytes < 0:
            msg = f"a ledger records non-negative spend, got {self}"
            raise ValueError(msg)

    def plus(self, *, trials: int = 0, seconds: float = 0.0, stored_bytes: int = 0) -> BudgetLedger:
        """The ledger after one more trial's spend."""
        return BudgetLedger(
            trials=self.trials + trials,
            seconds=self.seconds + seconds,
            stored_bytes=self.stored_bytes + stored_bytes,
        )


def budget_complaints(ledger: BudgetLedger, budget: ManualSearchBudget) -> list[str]:
    """Every cap this ledger has reached. A non-empty answer stops scheduling; it never raises the cap."""
    complaints: list[str] = []
    if ledger.trials >= budget.trials:
        complaints.append(f"the {budget.trials}-trial cap is spent ({ledger.trials} trials)")
    if ledger.seconds >= budget.hours * 3600.0:
        complaints.append(f"the {budget.hours:g} h cap is spent ({ledger.seconds / 3600.0:.2f} h)")
    if ledger.stored_bytes >= budget.gib * _GIB:
        complaints.append(f"the {budget.gib:g} GiB cap is spent ({ledger.stored_bytes / _GIB:.2f} GiB)")
    return complaints


def ledger_of(study: optuna.Study) -> BudgetLedger:
    """The spend recorded on the study, which a resume continues rather than resets."""
    return BudgetLedger(
        trials=int(cast("int", study.user_attrs.get(_TRIALS_ATTR, 0))),
        seconds=float(cast("float", study.user_attrs.get(_SECONDS_ATTR, 0.0))),
        stored_bytes=int(cast("int", study.user_attrs.get(_BYTES_ATTR, 0))),
    )


def record_trial(study: optuna.Study, ledger: BudgetLedger) -> None:
    """Persist the spend on the study, so the next invocation reads it back."""
    study.set_user_attr(_TRIALS_ATTR, ledger.trials)
    study.set_user_attr(_SECONDS_ATTR, ledger.seconds)
    study.set_user_attr(_BYTES_ATTR, ledger.stored_bytes)


def suggest_sampled_point(space: ManualSearchSpace, trial: optuna.Trial) -> SampledPoint:
    """Draw one point of the approved space, then read it back through the protocol's own validation."""
    params: dict[str, float] = {
        "n_neurons": trial.suggest_int(
            "n_neurons", space.n_neurons.low, space.n_neurons.high, step=space.n_neurons.step
        ),
        "spectral_radius": trial.suggest_float(
            "spectral_radius", space.spectral_radius.low, space.spectral_radius.high
        ),
        "sparsity": trial.suggest_float("sparsity", space.sparsity.low, space.sparsity.high),
        "leak_rate": trial.suggest_float("leak_rate", space.leak_rate.low, space.leak_rate.high, log=True),
        "input_scaling": trial.suggest_float(
            "input_scaling", space.input_scaling.low, space.input_scaling.high, log=True
        ),
        "alpha_0": trial.suggest_float("alpha_0", space.alpha_0.low, space.alpha_0.high, log=True),
        "warmup_s": trial.suggest_categorical("warmup_s", list(space.warmup_s)),
    }
    return sampled_point(space, params)


@dataclass(frozen=True)
class TrialResult:
    """What a worker reports back about one candidate: never a trajectory, only its verdicts."""

    trial: int
    successes: int
    runs: int
    statuses: tuple[str, ...]
    scenarios: tuple[str, ...]
    seconds: float
    stored_bytes: int
    evidence: str | None = None
    """Where the worker left this candidate's evidence, retained whatever the verdict was."""
    failure: str | None = None
    """Why the candidate has no score: a fit failure or an interruption, retained as itself."""

    def __post_init__(self) -> None:
        """A reported result is internally consistent and stays inside the nominal scope."""
        if self.trial < 0 or self.runs < 0 or self.successes < 0 or self.successes > self.runs:
            msg = f"trial {self.trial}: {self.successes} successes of {self.runs} runs is not a result"
            raise ValueError(msg)
        if len(self.statuses) != self.runs:
            msg = f"trial {self.trial}: {len(self.statuses)} statuses for {self.runs} runs"
            raise ValueError(msg)
        mismatches = nominal_scope_mismatches(self.scenarios)
        if mismatches:
            msg = f"trial {self.trial}: {'; '.join(mismatches)}"
            raise ValueError(msg)

    @property
    def score(self) -> float | None:
        """The nominal success fraction, or ``None`` when this candidate was never scored."""
        if self.failure is not None:
            return None
        return nominal_success_fraction(self.successes, runs=self.runs)


def read_result(path: Path) -> TrialResult:
    """Read a worker's report strictly; an unreadable one is a failed trial, never a zero."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), TrialResult)


def trial_command(
    protocol_file: Path, *, trial: int, point: SampledPoint, output: Path, root: Path, exploratory: bool = False
) -> list[str]:
    """The worker invocation for one trial: its own process, its own scope, its own report.

    The worker is given the point rather than a seed for the sampler, so it
    reproduces exactly the candidate the parent drew, and it is given the
    scenarios explicitly so it can refuse anything but the nominal case.
    """
    return [
        sys.executable,
        "-m",
        _MODULE,
        "evaluate-trial",
        "--protocol",
        str(protocol_file),
        "--trial",
        str(trial),
        "--point",
        json.dumps(to_mapping(point), sort_keys=True),
        "--scenarios",
        *NOMINAL_SCENARIOS,
        "--output",
        str(output),
        "--root",
        str(root),
        *(["--exploratory"] if exploratory else []),
    ]


def evaluate_trial(
    protocol: ManualSearchProtocol,
    *,
    trial: int,
    point: SampledPoint,
    scenarios: Sequence[str],
    root: Path,
    argv: Sequence[str],
    exploratory: bool = False,
) -> TrialResult:
    """Fit the all-ten arm at one sampled configuration and judge its nominal runs.

    This runs in the worker process. The scenarios it was actually given are
    checked here, not only where they were chosen, and the configuration is
    rebuilt from the point so the worker never depends on the parent having
    described it correctly.
    """
    mismatches = nominal_scope_mismatches(scenarios)
    if mismatches:
        msg = "; ".join(mismatches)
        raise ValueError(msg)
    configuration = sampled_configuration(protocol, point, trial=trial)
    args = argparse.Namespace(
        study=str(protocol.study),
        evaluation=str(protocol.comparison.evaluation),
        exploratory=exploratory,
        argv=list(argv),
    )
    prepared = prepare_runner(
        args,
        role="worker",
        root=root,
        module=_MODULE,
        scenario_ids=tuple(scenarios),
        sampled=(configuration,),
    )
    manifest = prepared.context.manifest
    entry = sampled_entry(manifest, configuration, arm_of("M10"))
    started = time.perf_counter()
    evidence = prepared.runner.evaluate(entry, warmup_s=configuration.warmup_s)
    statuses = tuple(pair.status for pair in evidence.pairs)
    return TrialResult(
        trial=trial,
        successes=sum(1 for pair in evidence.pairs if pair.outcome is not None and pair.outcome.success),
        runs=len(evidence.pairs),
        statuses=statuses,
        scenarios=tuple(scenarios),
        seconds=time.perf_counter() - started,
        stored_bytes=_evidence_bytes(prepared.runner.store, evidence),
        evidence=evidence.identity,
    )


def _evidence_bytes(store: StorageRoot, evidence: Any) -> int:  # noqa: ANN401 - the evidence record of one model
    """How much this candidate's runs added to the store, counted from the records themselves."""
    del store
    return sum(pair.run.size for pair in evidence.pairs if pair.run is not None)


def spawn_trial(command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    """Run one trial's worker as its own process, inheriting this process's pinned environment.

    The worker is a process rather than a thread because reservoir
    construction re-seeds a process-global generator: two constructions that
    interleave in one interpreter can differ from the same seed.
    """
    return subprocess.run(list(command), check=False, capture_output=True)


def run_search(
    protocol: ManualSearchProtocol, protocol_file: Path, *, root: Path, scratch: Path, exploratory: bool = False
) -> BudgetLedger:
    """Run or resume the search, serially, until a cap is reached or the trial budget is spent.

    Every trial is a worker process. A worker that fails leaves a failed trial
    behind: it consumed a slot, its partial work stays where it was written,
    and the search does not hand its number to a fresh candidate.
    """
    store = open_storage()
    study = open_study(
        store,
        STUDY_NAME,
        protocol_sha256=protocol_digest(protocol),
        sampler=protocol.sampler,
        pruner=PrunerSpec(kind="none"),
        direction="maximize",
    )
    try:
        ledger = ledger_of(study)
        while True:
            complaints = budget_complaints(ledger, protocol.budget)
            if complaints:
                print(json.dumps({"stopped": complaints, "spent": to_mapping(ledger)}, indent=2))
                return ledger
            trial = study.ask()
            point = suggest_sampled_point(protocol.space, trial)
            output = scratch / f"trial-{trial.number:04d}.json"
            started = time.perf_counter()
            completed = spawn_trial(
                trial_command(
                    protocol_file,
                    trial=trial.number,
                    point=point,
                    output=output,
                    root=root,
                    exploratory=exploratory,
                )
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0 or not output.is_file():
                tail = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
                study.tell(trial, state=TrialState.FAIL)
                ledger = ledger.plus(trials=1, seconds=elapsed)
                record_trial(study, ledger)
                print(json.dumps({"trial": trial.number, "failed": " | ".join(tail)[:500]}, indent=2))
                continue
            result = read_result(output)
            score = result.score
            if score is None:
                study.tell(trial, state=TrialState.FAIL)
            else:
                study.tell(trial, score)
            ledger = ledger.plus(trials=1, seconds=elapsed, stored_bytes=result.stored_bytes)
            record_trial(study, ledger)
            print(json.dumps({"trial": trial.number, "score": score, "statuses": list(result.statuses)}, indent=2))
    finally:
        close_study(study)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: the parent search and the per-trial worker."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Run or resume the manual-demonstration ESN search.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    search = subparsers.add_parser("search", help="run or resume the approved search")
    search.add_argument("--protocol", required=True, type=Path, help="the frozen search protocol")
    search.add_argument("--scratch", required=True, type=Path, help="where per-trial reports are written")
    search.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree (fixtures, trials)")
    worker = subparsers.add_parser("evaluate-trial", help="evaluate one candidate in this process")
    worker.add_argument("--protocol", required=True, type=Path)
    worker.add_argument("--trial", required=True, type=int)
    worker.add_argument("--point", required=True, help="the sampled point as JSON")
    worker.add_argument("--scenarios", nargs="+", required=True, help="the scenarios this worker may evaluate")
    worker.add_argument("--output", required=True, type=Path)
    worker.add_argument("--root", required=True, type=Path)
    worker.add_argument("--exploratory", action="store_true", help="tolerate a dirty worktree (fixtures, trials)")
    args = parser.parse_args(argv)
    protocol = load_manual_search(Path(cast("Path", args.protocol)))
    if args.subcommand == "search":
        scratch = Path(cast("Path", args.scratch))
        scratch.mkdir(parents=True, exist_ok=True)
        run_search(
            protocol,
            Path(cast("Path", args.protocol)),
            root=repository_root(),
            scratch=scratch,
            exploratory=bool(args.exploratory),
        )
        return 0
    point = from_mapping(cast("dict[str, object]", json.loads(cast("str", args.point))), SampledPoint)
    result = evaluate_trial(
        protocol,
        trial=int(cast("int", args.trial)),
        point=point,
        scenarios=tuple(cast("list[str]", args.scenarios)),
        root=Path(cast("Path", args.root)),
        argv=argv,
        exploratory=bool(args.exploratory),
    )
    output = Path(cast("Path", args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_mapping(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
