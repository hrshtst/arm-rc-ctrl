# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-006: one contractive bank per locked demonstration, recorded portably and shareable across models.

A bank binds its parent by the committed record's payload and array digests,
records every accepted episode and every rejection, and regenerates bit for bit
in any order. A parent that cannot fill its bank is a reported failure that
never stops the other parents.
"""

from __future__ import annotations

import dataclasses
import json
import re

import numpy as np
import pytest

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual_scenario import (
    ManualScenarioConfig,
    load_manual_scenario,
    manual_endpoint_positions,
)
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.experiments.manual_augmentation import (
    MANUAL_AUGMENTATION_SCHEMA_VERSION,
    MANUAL_PROTOCOL,
    BankGeneration,
    BankGenerationRecord,
    EpisodeDigests,
    ManualParent,
    ParentBank,
    ParentBankRecord,
    ParentFailure,
    RejectionRecord,
    check_bank_reproducible,
    generate_banks,
    generate_parent_bank,
    manual_parents,
)
from arm_rc_ctrl.experiments.manual_bank import load_bank_manifest
from arm_rc_ctrl.provenance import canonical_json
from arm_rc_ctrl.rc.augment import (
    MANUAL_ENVELOPE_VERSION,
    MANUAL_N_SYNTHETIC,
    MANUAL_RAMP_DURATION_S,
    MANUAL_SEED_NAMESPACE,
)
from arm_rc_ctrl.rc.recipe import DatasetSource
from arm_rc_ctrl.repo import repository_root

REPO_ROOT = repository_root()
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "configs"
BANK_FILE = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration" / "bank" / "bank_v1.json"
SCENARIO = load_manual_scenario(FIXTURES / "planar_2dof_manual_fixture.toml")
DERIVATIVES = DerivativeConfig(method="central")
DT = 0.01
N = 201
DWELL_START_S = 1.5
HOLD_S = 0.2
GOAL_Q = (0.8, 0.4)
SEED_BANK = 1
IDS = ("processed-20260916-0000000000a1", "processed-20260916-0000000000b2")
RECORD_DIRECTORY = "data/records/processed"


def _samples(
    goal: tuple[float, float] = GOAL_Q,
    *,
    n: int = N,
    dwell_start_s: float = DWELL_START_S,
    move_end_s: float | None = None,
) -> SampleSet:
    """A complete manual recording: pre-roll hold, smooth reach, and the final dwell at ``goal``.

    ``move_end_s`` ends the reach before the dwell onset, which makes the same
    movement faster; the recording then holds still until the dwell.
    """
    t = np.arange(n, dtype=np.float64) * DT
    start = np.asarray(SCENARIO.task.initial_q, dtype=np.float64)
    end = dwell_start_s if move_end_s is None else move_end_s
    s = np.clip((t - HOLD_S) / (end - HOLD_S), 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (np.asarray(goal, dtype=np.float64) - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = manual_endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < HOLD_S, 0, np.where(t < dwell_start_s, 1, 2)).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((n, 0), dtype=np.float64), phase)


def _parent(
    assignment: str = "D01",
    *,
    artifact_id: str = IDS[0],
    samples: SampleSet | None = None,
    dwell_start_s: float = DWELL_START_S,
) -> ManualParent:
    """A parent bound to the digests of ``samples`` (the identity a committed record would carry)."""
    sample_set = _samples() if samples is None else samples
    return ManualParent(
        assignment=assignment,
        dataset=DatasetSource(artifact_id, "ab" * 32, f"{RECORD_DIRECTORY}/{artifact_id}.toml"),
        dwell_start_s=dwell_start_s,
        n_samples=sample_set.n_samples,
        period_s=DT,
        derivative_method=DERIVATIVES.label,
        q_sha256=array_digest(sample_set.q),
        dq_sha256=array_digest(sample_set.dq),
    )


def _tamper(record: ParentBankRecord) -> ParentBankRecord:
    """The record with the first episode's position digest replaced by the second episode's."""
    episodes = (dataclasses.replace(record.episodes[0], q_sha256=record.episodes[1].q_sha256), *record.episodes[1:])
    return dataclasses.replace(record, episodes=episodes)


def _slow_scenario(velocity: float) -> ManualScenarioConfig:
    """The fixture with both joints' speed limit (and the acquisition bound with it) lowered."""
    return dataclasses.replace(
        SCENARIO,
        limits=dataclasses.replace(SCENARIO.limits, velocity=(velocity, velocity)),
        acquisition=dataclasses.replace(SCENARIO.acquisition, velocity_bound_rad_s=velocity),
    )


def _bank(parent: ManualParent, samples: SampleSet) -> ParentBank:
    outcome = generate_parent_bank(parent, samples, SCENARIO, seed_bank=SEED_BANK)
    assert isinstance(outcome, ParentBank)
    return outcome


# --- parents from the locked manifest ------------------------------------------------------


def test_parents_resolve_from_the_locked_bank_with_digest_bound_identities() -> None:
    """``D01`` .. ``D10`` resolve to their committed records: payload digest, arrays, dwell, and grid."""
    parents = manual_parents(load_bank_manifest(BANK_FILE), root=REPO_ROOT)
    assert [parent.assignment for parent in parents] == [f"D{i:02d}" for i in range(1, 11)]
    assert len({parent.identifier for parent in parents}) == len(parents)
    for parent in parents:
        assert parent.identifier.startswith("processed-")
        assert parent.dataset.record == f"{RECORD_DIRECTORY}/{parent.identifier}.toml"
        assert parent.period_s == DT
        assert parent.derivative_method == "central-difference"
        assert parent.dwell_start_s > 0.0
        assert parent.n_samples > parent.dwell_start_s / parent.period_s
        assert len(parent.q_sha256) == 64
        assert len(parent.dq_sha256) == 64


# --- one parent's bank ---------------------------------------------------------------------


def test_parent_bank_records_every_accepted_episode_and_rejection() -> None:
    """The record carries the protocol, the parent identity, the episodes' digests, and the accounting."""
    samples = _samples()
    bank = _bank(_parent(), samples)
    record = bank.record
    assert record.schema_version == MANUAL_AUGMENTATION_SCHEMA_VERSION
    assert record.protocol == MANUAL_PROTOCOL
    assert record.assignment == "D01"
    assert record.dataset.artifact_id == IDS[0]
    assert record.envelope == MANUAL_ENVELOPE_VERSION
    assert record.seed_namespace == MANUAL_SEED_NAMESPACE
    assert record.config.ramp_duration_s == MANUAL_RAMP_DURATION_S
    assert record.config.parent == IDS[0]
    assert record.config.seed_bank == SEED_BANK
    assert record.dwell_start_s == DWELL_START_S
    assert record.parent_q_sha256 == array_digest(samples.q)
    assert len(record.episodes) == MANUAL_N_SYNTHETIC == len(bank.episodes)
    assert record.accepted_attempts == tuple(e.attempt for e in record.episodes)
    assert record.attempts_used >= MANUAL_N_SYNTHETIC
    assert all(rejection.attempt <= record.attempts_used for rejection in record.rejections)
    for digests, arrays in zip(record.episodes, bank.episodes, strict=True):
        assert digests.q_sha256 == array_digest(arrays.q)
        assert digests.dq_sha256 == array_digest(arrays.dq)
    assert len({d.q_sha256 for d in record.episodes}) == MANUAL_N_SYNTHETIC


def test_bank_record_round_trips_as_portable_json() -> None:
    """The record is versioned, JSON round-trips strictly, and names no machine path."""
    generation = generate_banks((_parent(),), {IDS[0]: _samples()}, SCENARIO, seed_bank=SEED_BANK)
    assert isinstance(generation, BankGeneration)
    record = generation.record
    text = json.dumps(to_mapping(record), indent=2, sort_keys=True)
    assert from_mapping(json.loads(text), BankGenerationRecord) == record
    assert not re.search(r"(?<![\w./])/(?:home|tmp|Users|mnt)/", text)
    assert record.schema_version == MANUAL_AUGMENTATION_SCHEMA_VERSION
    assert record.seed_namespace == MANUAL_SEED_NAMESPACE
    assert record.failures == ()
    assert generation.complete


def test_regenerating_a_bank_reproduces_identical_digests() -> None:
    """Banks are shareable across reservoir configurations: regeneration is bit for bit, and checkable."""
    parent, samples = _parent(), _samples()
    first, second = _bank(parent, samples), _bank(parent, samples)
    assert first.record == second.record
    assert check_bank_reproducible(first.record, parent, samples, SCENARIO) == ()
    for a, b in zip(first.episodes, second.episodes, strict=True):
        assert np.array_equal(a.q, b.q)
        assert np.array_equal(a.dq, b.dq)


def test_a_tampered_digest_is_reported_by_the_reproducibility_check() -> None:
    """The check compares every recorded digest and names what differs instead of passing silently."""
    parent, samples = _parent(), _samples()
    record = _bank(parent, samples).record
    tampered = _tamper(record)
    mismatches = check_bank_reproducible(tampered, parent, samples, SCENARIO)
    assert mismatches
    assert any("episode 1" in mismatch for mismatch in mismatches)


# --- many parents --------------------------------------------------------------------------


def test_generation_order_does_not_change_any_parent_bank() -> None:
    """Each parent's bank depends on its own identity only, never on scheduling or task ordering."""
    first, second = _parent("D01", artifact_id=IDS[0]), _parent("D02", artifact_id=IDS[1])
    samples = {IDS[0]: _samples(), IDS[1]: _samples()}
    forward = generate_banks((first, second), samples, SCENARIO, seed_bank=SEED_BANK)
    backward = generate_banks((second, first), samples, SCENARIO, seed_bank=SEED_BANK)
    assert [bank.record for bank in forward.banks] == [bank.record for bank in reversed(backward.banks)]
    # Identical recordings, different parents: the I8 parent term keeps the two banks apart.
    assert forward.banks[0].record.bank_sha256 != forward.banks[1].record.bank_sha256


def test_a_parent_that_cannot_fill_its_bank_is_reported_and_the_others_continue() -> None:
    """Budget exhaustion is a reported failure with its rejection reasons, not an abort.

    Both recordings are legal under the shared speed limit, but the faster one
    leaves no headroom for any perturbation, so none of its attempts survives.
    """
    slow, fast = _samples(), _samples(move_end_s=0.85)
    probe = _bank(_parent("D01", artifact_id=IDS[0]), slow)
    headroom = sorted(float(np.max(np.abs(arrays.dq))) for arrays in probe.episodes)[-3]
    scenario = _slow_scenario(headroom)
    good = _parent("D01", artifact_id=IDS[0], samples=slow)
    hurried = _parent("D02", artifact_id=IDS[1], samples=fast)
    generation = generate_banks((good, hurried), {IDS[0]: slow, IDS[1]: fast}, scenario, seed_bank=SEED_BANK)
    assert [bank.record.assignment for bank in generation.banks] == ["D01"]
    assert len(generation.failures) == 1
    failure = generation.failures[0]
    assert isinstance(failure, ParentFailure)
    assert failure.assignment == "D02"
    assert failure.dataset.artifact_id == IDS[1]
    assert len(failure.rejections) == failure.config.attempt_budget
    assert {rejection.reason for rejection in failure.rejections} == {"velocity limit violated"}
    assert {rejection.attempt for rejection in failure.rejections} == set(range(1, failure.config.attempt_budget + 1))
    assert failure.attempts_used == failure.config.attempt_budget
    assert failure.accepted == 0
    assert "attempt budget" in failure.reason
    assert not generation.complete
    assert generation.record.failures == (failure,)
    assert generation.record.banks == (generation.banks[0].record,)


def test_a_parent_too_short_for_the_ramp_is_reported_with_its_reason() -> None:
    """Insufficient transition support names the reason and never alters the recorded timing."""
    short = _samples(n=61, dwell_start_s=0.5)
    parent = _parent("D03", artifact_id=IDS[0], samples=short, dwell_start_s=0.5)
    outcome = generate_parent_bank(parent, short, SCENARIO, seed_bank=SEED_BANK)
    assert isinstance(outcome, ParentFailure)
    assert "ramp" in outcome.reason
    assert outcome.attempts_used == 0
    assert outcome.rejections == ()


def test_samples_that_do_not_match_the_parent_identity_are_refused() -> None:
    """A payload that is not the recorded one is a provenance error, never a silently different bank."""
    parent = _parent()
    with pytest.raises(ValueError, match="digest"):
        generate_parent_bank(parent, _samples(goal=(0.7, 0.5)), SCENARIO, seed_bank=SEED_BANK)
    with pytest.raises(ValueError, match="samples"):
        generate_parent_bank(parent, _samples(n=151), SCENARIO, seed_bank=SEED_BANK)
    with pytest.raises(ValueError, match=IDS[1]):
        generate_banks((parent,), {IDS[1]: _samples()}, SCENARIO, seed_bank=SEED_BANK)


def test_the_bank_digest_binds_the_configuration_and_every_episode() -> None:
    """``bank_sha256`` is the canonical witness other configurations compare against."""
    bank = _bank(_parent(), _samples())
    record = bank.record
    payload: dict[str, object] = {
        "assignment": record.assignment,
        "config": to_mapping(record.config),
        "dataset": to_mapping(record.dataset),
        "envelope": record.envelope,
        "episodes": [to_mapping(episode) for episode in record.episodes],
        "protocol": record.protocol,
        "seed_namespace": record.seed_namespace,
    }
    assert canonical_json(payload)
    assert len(record.bank_sha256) == 64
    assert record.bank_sha256 != record.parent_q_sha256


# --- the records refuse invalid data instead of correcting it ------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"assignment": "d1"}, "bank position"),
        ({"n_samples": 0}, "n_samples"),
        ({"dwell_start_s": 0.0}, "dwell_start_s"),
        ({"period_s": float("inf")}, "period_s"),
        ({"derivative_method": "finite-difference"}, "derivative policy"),
        ({"q_sha256": "ab"}, "q_sha256"),
        ({"dq_sha256": "zz" * 32}, "dq_sha256"),
    ],
)
def test_invalid_parents_are_refused(changes: dict[str, object], message: str) -> None:
    """A parent binds a bank position, a grid, and the recorded digests, or it is not a parent."""
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(_parent(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"episode": 0}, "positive"),
        ({"attempt": 0}, "positive"),
        ({"q_sha256": ""}, "q_sha256"),
    ],
)
def test_invalid_episode_digests_are_refused(changes: dict[str, object], message: str) -> None:
    """Every recorded episode names a positive episode, its attempt, and full digests."""
    digests = EpisodeDigests(episode=1, attempt=1, q_sha256="ab" * 32, dq_sha256="cd" * 32)
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(digests, **changes)


@pytest.mark.parametrize(("attempt", "reason"), [(0, "joint limits violated"), (1, "  ")])
def test_invalid_rejections_are_refused(attempt: int, reason: str) -> None:
    """A rejection is only evidence when it names both the attempt and the reason."""
    with pytest.raises(ValueError, match="rejection"):
        RejectionRecord(attempt=attempt, reason=reason)


def test_bank_records_refuse_inconsistent_accounting() -> None:
    """The record is self-checking: schema, parent binding, episode count, and attempt accounting."""
    record = _bank(_parent(), _samples()).record
    budget = record.config.attempt_budget
    cases: list[tuple[dict[str, object], str]] = [
        ({"schema_version": 2}, "schema_version"),
        ({"protocol": " "}, "protocol"),
        ({"assignment": "bank"}, "bank position"),
        ({"config": dataclasses.replace(record.config, parent=IDS[1])}, "is not the dataset"),
        ({"episodes": record.episodes[:-1], "accepted_attempts": record.accepted_attempts[:-1]}, "complete bank"),
        ({"accepted_attempts": tuple(reversed(record.accepted_attempts))}, "increasing order"),
        ({"episodes": tuple(reversed(record.episodes))}, "accepted_attempts"),
        ({"episodes": (dataclasses.replace(record.episodes[0], episode=2), *record.episodes[1:])}, "numbered 1"),
        ({"attempts_used": 1}, "attempts_used"),
        ({"attempts_used": budget + 1}, "attempts_used"),
        ({"rejections": (RejectionRecord(record.attempts_used + 1, "late"),)}, "distinct attempts"),
        ({"rejections": (RejectionRecord(record.accepted_attempts[0], "accepted too"),)}, "distinct attempts"),
        ({"bank_sha256": "ff"}, "bank_sha256"),
        ({"parent_q_sha256": "ff"}, "parent_q_sha256"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(record, **changes)


def test_failures_and_generations_refuse_inconsistent_records() -> None:
    """A failure never claims a complete bank, and one parent never appears twice in a pass."""
    parent = _parent()
    config = parent.config(seed_bank=SEED_BANK)
    failure = ParentFailure(
        schema_version=MANUAL_AUGMENTATION_SCHEMA_VERSION,
        protocol=MANUAL_PROTOCOL,
        assignment="D01",
        dataset=parent.dataset,
        config=config,
        reason="attempt budget exhausted",
        attempts_used=config.attempt_budget,
        accepted=0,
        rejections=(RejectionRecord(1, "velocity limit violated"),),
    )
    for changes, message in (
        ({"schema_version": 7}, "schema_version"),
        ({"reason": " "}, "reason"),
        ({"accepted": MANUAL_N_SYNTHETIC}, "is not a failure"),
        ({"attempts_used": config.attempt_budget + 1}, "outside the budget"),
        ({"attempts_used": -1}, "outside the budget"),
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(failure, **changes)

    record = _bank(parent, _samples()).record
    with pytest.raises(ValueError, match="appears once"):
        BankGenerationRecord(
            schema_version=MANUAL_AUGMENTATION_SCHEMA_VERSION,
            protocol=MANUAL_PROTOCOL,
            seed_namespace=MANUAL_SEED_NAMESPACE,
            envelope=MANUAL_ENVELOPE_VERSION,
            seed_bank=SEED_BANK,
            banks=(record,),
            failures=(failure,),
        )
    with pytest.raises(ValueError, match="seed bank"):
        BankGenerationRecord(
            schema_version=MANUAL_AUGMENTATION_SCHEMA_VERSION,
            protocol=MANUAL_PROTOCOL,
            seed_namespace=MANUAL_SEED_NAMESPACE,
            envelope=MANUAL_ENVELOPE_VERSION,
            seed_bank=SEED_BANK + 1,
            banks=(record,),
            failures=(),
        )
    with pytest.raises(ValueError, match="episodes"):
        ParentBank(record=record, episodes=())


def test_the_check_reports_a_failed_regeneration_and_a_changed_accounting() -> None:
    """Regeneration that cannot complete, or that used other attempts, is reported rather than passed."""
    record = _bank(_parent(), _samples()).record
    short = _samples(n=61, dwell_start_s=0.5)
    short_parent = _parent("D01", samples=short, dwell_start_s=0.5)
    failed = check_bank_reproducible(record, short_parent, short, SCENARIO)
    assert len(failed) == 1
    assert "regeneration failed" in failed[0]
    assert "ramp" in failed[0]

    claimed = dataclasses.replace(record, attempts_used=record.config.attempt_budget)
    mismatches = check_bank_reproducible(claimed, _parent(), _samples(), SCENARIO)
    assert any("attempts_used" in mismatch for mismatch in mismatches)
