# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Contractive augmentation banks of the locked manual demonstrations (M3MAN-006; manual plan section 11).

One bank per parent ``D01`` .. ``D10``: nine accepted synthetic episodes grown
from the recorded demonstration with the frozen contractive protocol
(:class:`~arm_rc_ctrl.rc.augment.ManualAugmentationConfig`), recorded as a
versioned, portable :class:`ParentBankRecord` that binds

- the parent's digest-bound dataset identity (artifact, payload digest, and the
  repository-relative record it came from) and its recorded array digests,
- the configuration, the envelope version, its ramp duration, and the seed
  namespace, so the generating protocol is never inferred from a file name,
- every accepted attempt with the per-episode digests of ``q`` and ``dq``, and
  every rejection with its attempt and reason.

Banks are shared across reservoir configurations: a bank depends on the parent
and the seed bank alone, never on the model it will train or on the order the
parents were generated in, and :func:`check_bank_reproducible` re-derives one
against its record. A parent that cannot fill its bank within the frozen budget
is a reported :class:`ParentFailure` carrying its rejection reasons — the other
parents are still generated.

Payloads live in the external store; this module takes the parent arrays from
its caller and verifies them against the committed record's digests, so nothing
here depends on a machine path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.manual import ManualDatasetRecord
from arm_rc_ctrl.data.records import load_record
from arm_rc_ctrl.experiments.manual_recipes import bank_sources
from arm_rc_ctrl.provenance import canonical_json, portable_config, sha256_bytes
from arm_rc_ctrl.rc.augment import (
    MANUAL_ENVELOPE_VERSION,
    MANUAL_SEED_NAMESPACE,
    AugmentationError,
    ManualAugmentationConfig,
    ManualBudgetExhaustedError,
    generate_manual_augmentation,
)
from arm_rc_ctrl.rc.recipe import DatasetSource, derivative_config
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.validation import SHA256_HEX_LENGTH, is_hex

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from arm_rc_ctrl.data.derivatives import DerivativeConfig
    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.experiments.manual_bank import BankManifest
    from arm_rc_ctrl.rc.augment import EpisodeArrays, TaskGeometry

__all__ = [
    "MANUAL_AUGMENTATION_SCHEMA_VERSION",
    "MANUAL_PROTOCOL",
    "BankGeneration",
    "BankGenerationRecord",
    "EpisodeDigests",
    "ManualParent",
    "ParentBank",
    "ParentBankRecord",
    "ParentFailure",
    "RejectionRecord",
    "check_bank_reproducible",
    "generate_banks",
    "generate_parent_bank",
    "manual_parents",
]

MANUAL_AUGMENTATION_SCHEMA_VERSION: Final = 1
"""Version of the bank-record schema (independent of the demonstration bank's own schema)."""
MANUAL_PROTOCOL: Final = "task_1a_manual_v1"
"""Protocol every manual bank belongs to."""

_RECORD_DIRECTORY: Final = "data/records/processed"
_PENDING_DIGEST: Final = "0" * SHA256_HEX_LENGTH
"""Stand-in a record carries only between its construction and the stamping of its own identity digest."""
_ASSIGNMENT_RE: Final = re.compile(r"^D\d{2}$")


def _require_digest(value: str, label: str) -> None:
    if not is_hex(value, SHA256_HEX_LENGTH):
        msg = f"{label} must be 64 lowercase hex characters, got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class ManualParent:
    """One locked demonstration a bank is grown from, bound to its committed record."""

    assignment: str
    """``D01`` .. ``D10``: the demonstration's position in the locked bank."""
    dataset: DatasetSource
    """Digest-bound dataset identity: artifact ID, payload digest, and repository-relative record."""
    dwell_start_s: float
    """Onset of the recorded final dwell on the task clock (``motion.dwell_start_s``)."""
    n_samples: int
    period_s: float
    """Grid period of the processed recording (``preprocessing.resample_period_s``)."""
    derivative_method: str
    """The parent's recorded derivative policy; synthetic velocity is recomputed with it."""
    q_sha256: str
    """Digest of the recorded positions, from the committed record's array specification."""
    dq_sha256: str

    def __post_init__(self) -> None:
        """Validate the assignment, the grid, and the recorded digests."""
        if _ASSIGNMENT_RE.match(self.assignment) is None:
            msg = f"assignment must be a bank position such as 'D01', got {self.assignment!r}"
            raise ValueError(msg)
        if self.n_samples < 1:
            msg = f"{self.assignment}: n_samples must be positive, got {self.n_samples}"
            raise ValueError(msg)
        for name, value in (("dwell_start_s", self.dwell_start_s), ("period_s", self.period_s)):
            if not (value > 0 and value < float("inf")):
                msg = f"{self.assignment}: {name} must be positive and finite, got {value!r}"
                raise ValueError(msg)
        derivative_config(self.derivative_method)  # refuses an unknown policy label
        _require_digest(self.q_sha256, f"{self.assignment}: q_sha256")
        _require_digest(self.dq_sha256, f"{self.assignment}: dq_sha256")

    @property
    def identifier(self) -> str:
        """The stable parent identifier every stream carries: the processed artifact ID."""
        return self.dataset.artifact_id

    @property
    def derivatives(self) -> DerivativeConfig:
        """The derivative scheme the parent's recorded policy names."""
        return derivative_config(self.derivative_method)

    def config(self, *, seed_bank: int) -> ManualAugmentationConfig:
        """The frozen contractive configuration of this parent in ``seed_bank``."""
        return ManualAugmentationConfig(parent=self.identifier, seed_bank=seed_bank)


def manual_parents(bank: BankManifest, *, root: Path | None = None) -> tuple[ManualParent, ...]:
    """Resolve the locked bank's assignments to their committed manual datasets, in bank order.

    The dataset identity comes from :func:`~arm_rc_ctrl.experiments.manual_recipes.bank_sources`
    (payload digest, repository-relative record), and the dwell onset, grid,
    derivative policy, and array digests come from that record — so a bank
    binds the recorded bytes and never a machine path.

    Raises
    ------
    ValueError
        If the bank is incomplete or a record does not describe its dataset.
    """
    sources = bank_sources(bank, root=root)
    base = repository_root() if root is None else root
    parents: list[ManualParent] = []
    for assignment in sorted(sources):
        source = sources[assignment]
        record = load_record(base / source.record, ManualDatasetRecord)
        parents.append(
            ManualParent(
                assignment=assignment,
                dataset=source,
                dwell_start_s=record.motion.dwell_start_s,
                n_samples=record.n_samples,
                period_s=record.preprocessing.resample_period_s,
                derivative_method=record.preprocessing.derivative_method,
                # The manual record schema requires exactly the canonical arrays, in order.
                q_sha256=record.arrays["q"].sha256,
                dq_sha256=record.arrays["dq"].sha256,
            )
        )
    return tuple(parents)


@dataclass(frozen=True)
class EpisodeDigests:
    """Identity of one accepted synthetic episode: its attempt and its array digests."""

    episode: int
    attempt: int
    q_sha256: str
    dq_sha256: str

    def __post_init__(self) -> None:
        """Validate the numbering and the digests."""
        if self.episode < 1 or self.attempt < 1:
            msg = f"episode and attempt must be positive, got {self.episode} and {self.attempt}"
            raise ValueError(msg)
        _require_digest(self.q_sha256, f"episode {self.episode}: q_sha256")
        _require_digest(self.dq_sha256, f"episode {self.episode}: dq_sha256")


@dataclass(frozen=True)
class RejectionRecord:
    """One rejected attempt and why it was rejected (never clipped, never retimed)."""

    attempt: int
    reason: str

    def __post_init__(self) -> None:
        """A rejection names an attempt and a reason."""
        if self.attempt < 1 or not self.reason.strip():
            msg = f"a rejection needs a positive attempt and a reason, got {self.attempt} and {self.reason!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ParentBankRecord:
    """The portable record of one parent's contractive bank."""

    schema_version: int
    protocol: str
    assignment: str
    dataset: DatasetSource
    config: ManualAugmentationConfig
    envelope: str
    seed_namespace: str
    dwell_start_s: float
    period_s: float
    derivative_method: str
    parent_q_sha256: str
    parent_dq_sha256: str
    episodes: tuple[EpisodeDigests, ...]
    accepted_attempts: tuple[int, ...]
    attempts_used: int
    rejections: tuple[RejectionRecord, ...]
    bank_sha256: str
    """Digest over the canonical identity of this bank; two runs agree exactly or the bank changed."""

    def __post_init__(self) -> None:
        """Validate the schema, the parent binding, and the attempt accounting."""
        if self.schema_version != MANUAL_AUGMENTATION_SCHEMA_VERSION:
            msg = f"unsupported schema_version {self.schema_version}; expected {MANUAL_AUGMENTATION_SCHEMA_VERSION}"
            raise ValueError(msg)
        if not self.protocol.strip() or _ASSIGNMENT_RE.match(self.assignment) is None:
            msg = f"a bank record needs a protocol and a bank position, got {self.protocol!r}, {self.assignment!r}"
            raise ValueError(msg)
        if self.config.parent != self.dataset.artifact_id:
            msg = f"the configuration's parent {self.config.parent!r} is not the dataset {self.dataset.artifact_id!r}"
            raise ValueError(msg)
        if len(self.episodes) != self.config.n_synthetic:
            msg = f"{self.assignment}: a complete bank has {self.config.n_synthetic} episodes, got {len(self.episodes)}"
            raise ValueError(msg)
        self._check_attempts()
        _require_digest(self.parent_q_sha256, f"{self.assignment}: parent_q_sha256")
        _require_digest(self.parent_dq_sha256, f"{self.assignment}: parent_dq_sha256")
        _require_digest(self.bank_sha256, f"{self.assignment}: bank_sha256")

    def _check_attempts(self) -> None:
        accepted = tuple(episode.attempt for episode in self.episodes)
        if self.accepted_attempts != accepted or list(accepted) != sorted(set(accepted)):
            msg = (
                f"{self.assignment}: accepted_attempts {list(self.accepted_attempts)} must be the episodes' "
                f"attempts {list(accepted)} in increasing order"
            )
            raise ValueError(msg)
        if [episode.episode for episode in self.episodes] != list(range(1, len(self.episodes) + 1)):
            msg = f"{self.assignment}: episodes must be numbered 1..{len(self.episodes)} in acceptance order"
            raise ValueError(msg)
        if not accepted or self.attempts_used < max(accepted) or self.attempts_used > self.config.attempt_budget:
            msg = (
                f"{self.assignment}: attempts_used {self.attempts_used} must cover the accepted attempts and stay "
                f"within the budget {self.config.attempt_budget}"
            )
            raise ValueError(msg)
        rejected = [rejection.attempt for rejection in self.rejections]
        if any(attempt > self.attempts_used for attempt in rejected) or set(rejected) & set(accepted):
            msg = f"{self.assignment}: rejections must name distinct attempts within {self.attempts_used}"
            raise ValueError(msg)


@dataclass(frozen=True)
class ParentBank:
    """One parent's generated bank: the portable record and the arrays it digests."""

    record: ParentBankRecord
    episodes: tuple[EpisodeArrays, ...]

    def __post_init__(self) -> None:
        """The arrays are exactly the episodes the record describes."""
        if len(self.episodes) != len(self.record.episodes):
            msg = f"{self.record.assignment}: {len(self.episodes)} episodes for {len(self.record.episodes)} records"
            raise ValueError(msg)


@dataclass(frozen=True)
class ParentFailure:
    """A parent whose bank could not be generated, with everything the diagnosis needs."""

    schema_version: int
    protocol: str
    assignment: str
    dataset: DatasetSource
    config: ManualAugmentationConfig
    reason: str
    attempts_used: int
    accepted: int
    rejections: tuple[RejectionRecord, ...]

    def __post_init__(self) -> None:
        """A failure names a reason and never claims a complete bank."""
        if self.schema_version != MANUAL_AUGMENTATION_SCHEMA_VERSION:
            msg = f"unsupported schema_version {self.schema_version}; expected {MANUAL_AUGMENTATION_SCHEMA_VERSION}"
            raise ValueError(msg)
        if not self.reason.strip():
            msg = f"{self.assignment}: a failure must carry its reason"
            raise ValueError(msg)
        if self.accepted >= self.config.n_synthetic:
            msg = f"{self.assignment}: {self.accepted} accepted episodes is not a failure"
            raise ValueError(msg)
        if self.attempts_used < 0 or self.attempts_used > self.config.attempt_budget:
            msg = f"{self.assignment}: attempts_used {self.attempts_used} lies outside the budget"
            raise ValueError(msg)


@dataclass(frozen=True)
class BankGenerationRecord:
    """The portable record of one generation pass over a set of parents."""

    schema_version: int
    protocol: str
    seed_namespace: str
    envelope: str
    seed_bank: int
    banks: tuple[ParentBankRecord, ...]
    failures: tuple[ParentFailure, ...]

    def __post_init__(self) -> None:
        """Every parent appears once, under one seed bank."""
        assignments = [bank.assignment for bank in self.banks] + [f.assignment for f in self.failures]
        if len(set(assignments)) != len(assignments):
            msg = f"each parent appears once per generation, got {assignments}"
            raise ValueError(msg)
        if any(bank.config.seed_bank != self.seed_bank for bank in self.banks):
            msg = f"every bank must be generated in seed bank {self.seed_bank}"
            raise ValueError(msg)


@dataclass(frozen=True)
class BankGeneration:
    """The outcome of generating banks for a set of parents: the banks, and the reported failures."""

    seed_bank: int
    banks: tuple[ParentBank, ...]
    failures: tuple[ParentFailure, ...]

    @property
    def complete(self) -> bool:
        """Whether every parent produced a complete bank."""
        return not self.failures

    @property
    def record(self) -> BankGenerationRecord:
        """The portable record of this pass."""
        return BankGenerationRecord(
            schema_version=MANUAL_AUGMENTATION_SCHEMA_VERSION,
            protocol=MANUAL_PROTOCOL,
            seed_namespace=MANUAL_SEED_NAMESPACE,
            envelope=MANUAL_ENVELOPE_VERSION,
            seed_bank=self.seed_bank,
            banks=tuple(bank.record for bank in self.banks),
            failures=self.failures,
        )


def _check_identity(parent: ManualParent, samples: SampleSet) -> None:
    """Refuse arrays that are not the recording the parent's committed record describes."""
    if samples.n_samples != parent.n_samples:
        msg = (
            f"{parent.assignment}: the payload has {samples.n_samples} samples, but {parent.identifier} records "
            f"{parent.n_samples}"
        )
        raise ValueError(msg)
    for name, recorded in (("q", parent.q_sha256), ("dq", parent.dq_sha256)):
        actual = array_digest(getattr(samples, name))
        if actual != recorded:
            msg = (
                f"{parent.assignment}: the payload's {name} digest {actual} does not match the digest {recorded} "
                f"recorded for {parent.identifier}"
            )
            raise ValueError(msg)


def record_digest(record: ParentBankRecord) -> str:
    """The canonical identity of one bank, derived from the record's own fields.

    Everything that decides which episodes the bank holds and why enters the
    digest: the parent binding, the protocol and its versions, the grid, the
    derivative policy, the seed namespace, every episode, and the whole
    rejection history. A record whose metadata was edited therefore no longer
    matches the digest it carries, which :func:`check_bank_reproducible`
    re-derives rather than trusting.
    """
    payload: dict[str, object] = {
        "accepted_attempts": list(record.accepted_attempts),
        "assignment": record.assignment,
        "attempts_used": record.attempts_used,
        "config": to_mapping(record.config),
        "dataset": to_mapping(record.dataset),
        "derivative_method": record.derivative_method,
        "dwell_start_s": record.dwell_start_s,
        "envelope": record.envelope,
        "episodes": [to_mapping(episode) for episode in record.episodes],
        "parent_dq_sha256": record.parent_dq_sha256,
        "parent_q_sha256": record.parent_q_sha256,
        "period_s": record.period_s,
        "protocol": record.protocol,
        "rejections": [to_mapping(rejection) for rejection in record.rejections],
        "schema_version": record.schema_version,
        "seed_namespace": record.seed_namespace,
    }
    return sha256_bytes(canonical_json(portable_config(payload)).encode("utf-8"))


def _failure(
    parent: ManualParent,
    config: ManualAugmentationConfig,
    *,
    reason: str,
    attempts_used: int,
    accepted: int,
    rejections: tuple[RejectionRecord, ...],
) -> ParentFailure:
    return ParentFailure(
        schema_version=MANUAL_AUGMENTATION_SCHEMA_VERSION,
        protocol=MANUAL_PROTOCOL,
        assignment=parent.assignment,
        dataset=parent.dataset,
        config=config,
        reason=reason,
        attempts_used=attempts_used,
        accepted=accepted,
        rejections=rejections,
    )


def generate_parent_bank(
    parent: ManualParent, samples: SampleSet, scenario: TaskGeometry, *, seed_bank: int
) -> ParentBank | ParentFailure:
    """Generate one parent's contractive bank, or report why it could not be generated.

    Raises
    ------
    ValueError
        If ``samples`` is not the recording the parent's committed record binds.
    """
    config = parent.config(seed_bank=seed_bank)
    _check_identity(parent, samples)
    try:
        result = generate_manual_augmentation(
            samples.t,
            samples.q,
            scenario,
            config,
            dwell_start_s=parent.dwell_start_s,
            period_s=parent.period_s,
            derivatives=parent.derivatives,
        )
    except ManualBudgetExhaustedError as exc:
        return _failure(
            parent,
            config,
            reason=str(exc),
            attempts_used=exc.attempts_used,
            accepted=exc.accepted,
            rejections=tuple(RejectionRecord(r.attempt, r.reason) for r in exc.rejections),
        )
    except AugmentationError as exc:
        return _failure(parent, config, reason=str(exc), attempts_used=0, accepted=0, rejections=())
    episodes = tuple(
        EpisodeDigests(
            episode=episode.episode,
            attempt=episode.attempt,
            q_sha256=array_digest(episode.arrays.q),
            dq_sha256=array_digest(episode.arrays.dq),
        )
        for episode in result.episodes
    )
    provisional = ParentBankRecord(
        schema_version=MANUAL_AUGMENTATION_SCHEMA_VERSION,
        protocol=MANUAL_PROTOCOL,
        assignment=parent.assignment,
        dataset=parent.dataset,
        config=config,
        envelope=result.envelope_version,
        seed_namespace=result.seed_namespace,
        dwell_start_s=parent.dwell_start_s,
        period_s=parent.period_s,
        derivative_method=result.derivative_method,
        parent_q_sha256=parent.q_sha256,
        parent_dq_sha256=parent.dq_sha256,
        episodes=episodes,
        accepted_attempts=tuple(episode.attempt for episode in episodes),
        attempts_used=result.attempts_used,
        rejections=tuple(RejectionRecord(r.attempt, r.reason) for r in result.rejections),
        bank_sha256=_PENDING_DIGEST,
    )
    # The digest covers every other field, so it is stamped once the record carrying them exists.
    record = replace(provisional, bank_sha256=record_digest(provisional))
    return ParentBank(record=record, episodes=tuple(episode.arrays for episode in result.episodes))


def generate_banks(
    parents: Sequence[ManualParent],
    samples: Mapping[str, SampleSet],
    scenario: TaskGeometry,
    *,
    seed_bank: int,
) -> BankGeneration:
    """Generate every parent's bank; a parent that cannot fill its bank is reported, never fatal.

    Raises
    ------
    ValueError
        If a parent's payload is missing, or a parent appears twice.
    """
    banks: list[ParentBank] = []
    failures: list[ParentFailure] = []
    for parent in parents:
        payload = samples.get(parent.identifier)
        if payload is None:
            msg = (
                f"no payload for parent {parent.assignment} ({parent.identifier}); the supplied payloads are "
                f"{sorted(samples)}"
            )
            raise ValueError(msg)
        outcome = generate_parent_bank(parent, payload, scenario, seed_bank=seed_bank)
        if isinstance(outcome, ParentBank):
            banks.append(outcome)
        else:
            failures.append(outcome)
    generation = BankGeneration(seed_bank=seed_bank, banks=tuple(banks), failures=tuple(failures))
    generation.record  # noqa: B018 - validates that each parent appears once under this seed bank
    return generation


def check_bank_reproducible(
    record: ParentBankRecord, parent: ManualParent, samples: SampleSet, scenario: TaskGeometry
) -> tuple[str, ...]:
    """Re-derive a bank from its parent and report every difference from ``record`` (empty when identical)."""
    mismatches: list[str] = []
    if record.bank_sha256 != record_digest(record):
        mismatches.append(f"{record.assignment}: bank_sha256 does not match the record's own contents")
    outcome = generate_parent_bank(parent, samples, scenario, seed_bank=record.config.seed_bank)
    if isinstance(outcome, ParentFailure):
        return (*mismatches, f"{record.assignment}: regeneration failed: {outcome.reason}")
    fresh = outcome.record
    mismatches.extend(_identity_mismatches(record, fresh))
    # Both sides hold exactly ``n_synthetic`` episodes: the record validates it, and so does the fresh bank.
    for recorded, regenerated in zip(record.episodes, fresh.episodes, strict=True):
        for name, was, now in (
            ("attempt", str(recorded.attempt), str(regenerated.attempt)),
            ("position digest", recorded.q_sha256, regenerated.q_sha256),
            ("velocity digest", recorded.dq_sha256, regenerated.dq_sha256),
        ):
            if was != now:
                mismatches.append(f"episode {recorded.episode}: {name} {was} regenerated as {now}")
    mismatches.extend(_rejection_mismatches(record, fresh))
    return tuple(mismatches)


_IDENTITY_FIELDS: Final = (
    "schema_version",
    "protocol",
    "assignment",
    "dataset",
    "config",
    "envelope",
    "seed_namespace",
    "dwell_start_s",
    "period_s",
    "derivative_method",
    "parent_q_sha256",
    "parent_dq_sha256",
    "accepted_attempts",
    "attempts_used",
    "bank_sha256",
)
"""Everything a record states about its bank; a regeneration reproduces all of it or the bank changed."""


def _identity_mismatches(record: ParentBankRecord, fresh: ParentBankRecord) -> list[str]:
    """Every recorded identity field the regenerated bank did not reproduce."""
    return [
        f"{record.assignment}: {name} {getattr(record, name)!r} regenerated as {getattr(fresh, name)!r}"
        for name in _IDENTITY_FIELDS
        if getattr(record, name) != getattr(fresh, name)
    ]


def _rejection_mismatches(record: ParentBankRecord, fresh: ParentBankRecord) -> list[str]:
    """Every difference in the rejection history: rejections are evidence, not a count."""
    was, now = record.rejections, fresh.rejections
    if len(was) != len(now):
        return [f"{record.assignment}: rejections {len(was)} regenerated as {len(now)}"]
    return [
        f"{record.assignment}: rejection of attempt {a.attempt} {a.reason!r} regenerated as "
        f"attempt {b.attempt} {b.reason!r}"
        for a, b in zip(was, now, strict=True)
        if a != b
    ]
