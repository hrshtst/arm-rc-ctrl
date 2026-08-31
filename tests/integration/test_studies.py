# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Optuna studies in the external storage root: creation, resume, pruning, deterministic selection (M3-002)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import optuna
import pytest

from arm_rc_ctrl.experiments.studies import (
    PrunerSpec,
    SamplerSpec,
    StudyMismatchError,
    close_study,
    open_study,
    run_trials,
    select_best,
    study_uri,
    summarize,
    summary_from_json,
    summary_to_json,
)
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.storage import StorageRoot

pytestmark = pytest.mark.integration

SAMPLER = SamplerSpec(seed=7, n_startup_trials=3)
PRUNER = PrunerSpec(n_startup_trials=2)
DIGEST = "0" * 64
STEPS = 4


def objective(trial: optuna.Trial) -> float:
    """A bowl with per-step intermediate reports that get worse the farther ``x`` is from the optimum."""
    x = trial.suggest_float("x", -2.0, 2.0)
    alpha = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
    n = trial.suggest_int("n", 10, 50, step=10)
    trial.set_user_attr("components", {"x2": x * x, "alpha": alpha, "n": n})
    for step in range(STEPS):
        trial.report(x * x * (step + 1), step)
        if trial.should_prune():
            raise optuna.TrialPruned
    return x * x


def make_store(base: Path, name: str = "store") -> StorageRoot:
    """A store with an empty root."""
    root = base / name
    root.mkdir()
    return StorageRoot(root, repositories=(repository_root(),))


def params_of(study: optuna.Study) -> list[tuple[int, str, dict[str, object]]]:
    """The stored trial sequence (number, state, params)."""
    return [(t.number, t.state.name, dict(t.params)) for t in study.trials]


def test_study_name_is_validated() -> None:
    """Study names are simple file stems under armrc://optuna/."""
    assert study_uri("esn-1a_v2.study") == "armrc://optuna/esn-1a_v2.study.db"
    for bad in ("", ".hidden", "a/b", "a b", "-x"):
        with pytest.raises(ValueError, match="study name"):
            study_uri(bad)


def test_specs_validate_their_counts() -> None:
    """Negative seeds and counts are rejected."""
    with pytest.raises(ValueError, match="seed"):
        SamplerSpec(seed=-1)
    with pytest.raises(ValueError, match="n_startup_trials"):
        SamplerSpec(seed=1, n_startup_trials=0)
    with pytest.raises(ValueError, match="non-negative"):
        PrunerSpec(n_warmup_steps=-1)
    assert isinstance(SamplerSpec(seed=1, kind="random").build(), optuna.samplers.RandomSampler)
    assert isinstance(PrunerSpec(kind="none").build(), optuna.pruners.NopPruner)


def test_study_creates_prunes_resumes_and_selects_deterministically(tmp_path: Path) -> None:
    """The study lives in the store, prunes with the median rule, resumes to its budget, and reruns identically."""
    store = make_store(tmp_path)
    study = open_study(store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER)
    assert (store.root / "optuna" / "bowl.db").is_file()
    assert run_trials(study, objective, budget=6) == 6
    first = params_of(study)
    assert len(first) == 6
    assert study.user_attrs["armrc.protocol_sha256"] == DIGEST
    states = {state for _, state, _ in first}
    assert states == {"COMPLETE", "PRUNED"}
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    assert all(0 < len(t.intermediate_values) < STEPS for t in pruned)
    assert all(t.user_attrs["components"]["x2"] == t.params["x"] ** 2 for t in study.trials)

    # Resume: the same identity reopens the study; trials already stored are kept and only the remainder runs.
    resumed = open_study(store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER)
    assert params_of(resumed) == first
    assert run_trials(resumed, objective, budget=6) == 0
    assert run_trials(resumed, objective, budget=9) == 3
    assert params_of(resumed)[:6] == first
    assert len(resumed.trials) == 9

    # Deterministic: a fresh store with the same seed and the same open/resume pattern reproduces the sequence.
    other_store = make_store(tmp_path, "other")
    other = open_study(other_store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER)
    run_trials(other, objective, budget=6)
    other = open_study(other_store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER)
    run_trials(other, objective, budget=9)
    assert params_of(other) == params_of(resumed)
    # ... while an uninterrupted run of nine trials diverges after the interruption point (documented).
    straight = open_study(
        make_store(tmp_path, "straight"), "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER
    )
    run_trials(straight, objective, budget=9)
    assert params_of(straight)[:6] == first
    assert params_of(straight) != params_of(resumed)

    # Two resumes from one stored state agree with each other.
    copy_a = make_store(tmp_path, "copy_a")
    copy_b = make_store(tmp_path, "copy_b")
    for copy in (copy_a, copy_b):
        (copy.root / "optuna").mkdir()
        shutil.copy(store.root / "optuna" / "bowl.db", copy.root / "optuna" / "bowl.db")
    continued: list[list[tuple[int, str, dict[str, object]]]] = []
    for copy in (copy_a, copy_b):
        study_copy = open_study(copy, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER)
        run_trials(study_copy, objective, budget=12)
        continued.append(params_of(study_copy))
    assert continued[0] == continued[1]
    assert len(continued[0]) == 12

    for opened in (study, other, straight):
        close_study(opened)  # disposes of the SQLite connections; a second call is a no-op
        close_study(opened)
    best = select_best(resumed)
    complete = [t for t in resumed.trials if t.state.name == "COMPLETE"]
    assert best.value == min(t.value for t in complete if t.value is not None)
    # An eligibility rule restricts the selection (and is named in the summary).
    later = select_best(resumed, eligible=lambda t: t.number > best.number)
    assert later.number > best.number
    restricted = summarize(resumed, eligible=lambda t: t.number > best.number, selection_rule="later")
    assert (restricted.best_number, restricted.selection_rule) == (later.number, "later")
    assert summarize(resumed, eligible=lambda _t: False).best_number is None
    with pytest.raises(ValueError, match="no eligible completed trial"):
        select_best(resumed, eligible=lambda _t: False)
    summary = summarize(resumed)
    assert summary.storage == "armrc://optuna/bowl.db"
    assert summary.best_number == best.number
    assert summary.n_complete + summary.n_pruned == 9
    assert summary.identity["armrc.direction"] == "minimize"
    assert summary.trials[0].params == {k: float(cast("float", v)) for k, v in first[0][2].items()}
    assert summary.trials[0].metrics["components.x2"] == summary.trials[0].params["x"] ** 2
    assert summary_from_json(summary_to_json(summary)) == summary


def test_resume_refuses_a_different_identity(tmp_path: Path) -> None:
    """A stored study is not continued under another protocol digest, sampler, pruner, or direction."""
    store = make_store(tmp_path)
    study = open_study(store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER)
    run_trials(study, objective, budget=1)
    with pytest.raises(StudyMismatchError, match=r"armrc\.protocol_sha256"):
        open_study(store, "bowl", protocol_sha256="1" * 64, sampler=SAMPLER, pruner=PRUNER)
    with pytest.raises(StudyMismatchError, match=r"armrc\.sampler"):
        open_study(store, "bowl", protocol_sha256=DIGEST, sampler=SamplerSpec(seed=8), pruner=PRUNER)
    with pytest.raises(StudyMismatchError, match=r"armrc\.pruner"):
        open_study(store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PrunerSpec(kind="none"))
    with pytest.raises(StudyMismatchError, match=r"armrc\.direction"):
        open_study(store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER, direction="maximize")
    assert len(open_study(store, "bowl", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PRUNER).trials) == 1


def test_selection_breaks_ties_by_trial_number_and_needs_a_completed_trial(tmp_path: Path) -> None:
    """Equal objectives select the earliest trial; pruned-only studies have no selection."""
    store = make_store(tmp_path)
    study = open_study(store, "ties", protocol_sha256=DIGEST, sampler=SAMPLER, pruner=PrunerSpec(kind="none"))
    with pytest.raises(ValueError, match="no eligible completed trial"):
        select_best(study)
    study.enqueue_trial({"x": 0.5, "alpha": 0.1, "n": 10})
    study.enqueue_trial({"x": -0.5, "alpha": 0.1, "n": 10})
    run_trials(study, objective, budget=2)
    assert [t.value for t in study.trials] == [0.25, 0.25]
    assert select_best(study).number == 0
    assert summarize(study).best_number == 0

    def failing(trial: optuna.Trial) -> float:
        trial.suggest_float("x", -1.0, 1.0)
        msg = "boom"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        run_trials(study, failing, budget=3)
    with pytest.raises(ValueError, match="budget"):
        run_trials(study, objective, budget=0)
