# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the handoff freezes -- what will run, how illustrations are chosen, and what an audit redoes.

Three artifacts, each versioned on its own and each written before execution:

``run_ordering_v1``
    The 186 models in manifest order, the 65 locked scenarios, the tracker order
    and the 60 replay banks by protocol key, with the accounting totals a
    complete study must meet. It is compact: the runs are the stated nesting
    over these lists, expanded deterministically.
``representative_rule_v1``
    How the report will choose which cases to illustrate: the arms, the
    categories and what each selects, the tie-breaking order, deduplication, and
    what an absent category means. The selections themselves come afterwards.
``resimulation_subset_v1``
    The explicit ordered identities of the 240 RC and 60 replay runs a
    clean-checkout audit re-simulates.

None holds a result. Each binds the files it was derived from by digest -- the
study manifest, the evaluation configuration and result schema v2 -- and names
the v2 records it relates to rather than restating their definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario
from arm_rc_ctrl.experiments.manual_evaluation import evaluation_scenarios, load_manual_evaluation_config
from arm_rc_ctrl.experiments.manual_recipes import ASSIGNMENTS
from arm_rc_ctrl.experiments.manual_representative import ILLUSTRATION_ARMS, ILLUSTRATION_CATEGORIES
from arm_rc_ctrl.experiments.manual_resimulation import RESIMULATION_CLASSES, resimulation_subset
from arm_rc_ctrl.experiments.manual_schema import RECORDS_BY_VERSION, load_schema
from arm_rc_ctrl.experiments.manual_study import EXPERIMENT_LABEL, load_study
from arm_rc_ctrl.experiments.manual_timing import BUDGET_PARENT, budget_entries
from arm_rc_ctrl.experiments.perturbations import load_development_robustness
from arm_rc_ctrl.experiments.recovery_search import RECOVERY_TRACKERS
from arm_rc_ctrl.provenance import canonical_json, sha256_file
from arm_rc_ctrl.validation import is_hex

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from arm_rc_ctrl.experiments.manual_study import StudyManifest
    from arm_rc_ctrl.experiments.perturbations import RobustnessScenario

__all__ = [
    "HANDOFF_VERSIONS",
    "REFERENCED_SCHEMA_VERSION",
    "REPRESENTATIVE_RULE_VERSION",
    "RESIMULATION_SUBSET_VERSION",
    "RUN_ORDERING_VERSION",
    "ExpectedAccounting",
    "HandoffInputs",
    "IllustrationCategory",
    "RcRun",
    "ReplayBankKey",
    "ReplayRun",
    "RepresentativeRule",
    "ResimulationFreeze",
    "RunOrdering",
    "handoff_inputs",
    "handoff_to_json",
    "load_handoff",
    "main",
    "render_handoff_markdown",
    "representative_rule",
    "resimulation_freeze",
    "run_ordering",
]

RUN_ORDERING_VERSION: Final = 1
REPRESENTATIVE_RULE_VERSION: Final = 1
RESIMULATION_SUBSET_VERSION: Final = 1
HANDOFF_VERSIONS: Final = {
    "run_ordering": RUN_ORDERING_VERSION,
    "representative_rule": REPRESENTATIVE_RULE_VERSION,
    "resimulation_subset": RESIMULATION_SUBSET_VERSION,
}
"""Each artifact's own version: a changed field in one does not version the others."""

REFERENCED_SCHEMA_VERSION: Final = 2
"""The result schema these v1 freezes reference. Pinned, so a later schema cannot make them unloadable."""

_SHA256_HEX: Final = 64
_ORDERING_DESCRIBES: Final = ("StudyAccounting", "ModelAccount", "BankAccount")
_RULE_DESCRIBES: Final = ("Selection", "IllustrationCase")
_RESIMULATION_DESCRIBES: Final = ("ResimulationSubset",)

_CATEGORY_SELECTS: Final = {
    "nominal": "the scenario `nominal`, always, so every configuration and tracker shows its nominal comparison",
    "all_ten_wins": (
        "the first scenario in frozen order where the all-ten arm (M10) succeeds and the singleton (S) fails"
    ),
    "singleton_wins": (
        "the first scenario in frozen order where the singleton (S) succeeds and the all-ten arm (M10) fails"
    ),
    "first_failure": "the first scenario in frozen order where any of the four comparison arms fails",
}
"""What each category selects. A category the code declares but this does not describe cannot be frozen."""

_EXPANSION: Final = (
    "RC runs are model-major: for each model in `models`, each scenario in `scenarios`, each tracker in "
    "`trackers`. Replay runs are bank-major: for each bank in `replay_banks`, each scenario, each tracker. "
    "A model with an assignment is paired against the bank of its own configuration and assignment; the "
    "all-ten arm is paired against none."
)
_APPLICATION: Final = (
    "Applied separately to each configuration under each tracker. One application's verdicts cover one "
    "configuration under one tracker, and each scenario must carry every comparison arm exactly once: "
    "verdicts spanning trackers, a repeated arm, a missing arm, or a scenario outside the frozen order are "
    "refused rather than merged."
)
_TIE_BREAKING: Final = (
    "The frozen scenario order of the run ordering this rule binds by digest. A category selects the FIRST "
    "qualifying scenario in that order, so it selects at most one."
)
_DEDUPLICATION: Final = (
    "A scenario chosen by more than one category is shown once, in frozen scenario order, recording every "
    "category that chose it in the declared category order."
)
_ABSENT: Final = (
    "A category that matches no scenario is listed in the selection's `absent`, never silently omitted, so a "
    "reader sees that the rule looked and found none."
)
_BASELINE: Final = (
    "Each selected case shows all four comparison arms beside the replay baseline of the parent under that "
    "case's configuration: the bank keyed (configuration, parent) in the run ordering."
)


def _require_hex(name: str, value: str) -> None:
    if not is_hex(value, _SHA256_HEX):
        msg = f"{name} must be 64 lowercase hex characters, got {value!r}"
        raise ValueError(msg)


def _require_header(experiment: str, version: int, expected: int, what: str) -> None:
    if experiment != EXPERIMENT_LABEL or version != expected:
        msg = f"unsupported {what} {version} or experiment {experiment!r}"
        raise ValueError(msg)


def _require_described(describes: tuple[str, ...]) -> None:
    known = set(RECORDS_BY_VERSION[REFERENCED_SCHEMA_VERSION])
    if not describes or not set(describes) <= known:
        msg = f"a freeze names result schema v{REFERENCED_SCHEMA_VERSION} records it relates to, got {describes}"
        raise ValueError(msg)


def _require_distinct(name: str, values: tuple[str, ...]) -> None:
    if not values or len(set(values)) != len(values):
        msg = f"{name} must be non-empty and name each item once"
        raise ValueError(msg)


@dataclass(frozen=True)
class HandoffInputs:
    """The files a freeze was derived from, bound by digest so it cannot be read against others."""

    study_manifest_sha256: str
    evaluation_sha256: str
    result_schema_sha256: str
    result_schema_version: int

    def __post_init__(self) -> None:
        """Digests are digests, and the schema is the one these freezes reference."""
        for name in ("study_manifest_sha256", "evaluation_sha256", "result_schema_sha256"):
            _require_hex(name, cast("str", getattr(self, name)))
        if self.result_schema_version != REFERENCED_SCHEMA_VERSION:
            msg = (
                f"the handoff freezes reference result schema version {REFERENCED_SCHEMA_VERSION}, "
                f"got version {self.result_schema_version}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class ReplayBankKey:
    """One replay bank by what the protocol decides about it, not by an identity that also hashes the environment."""

    configuration: str
    assignment: str
    warmup_s: float
    velocity_cutoff_hz: float
    acceleration_cutoff_hz: float


@dataclass(frozen=True)
class ExpectedAccounting:
    """The totals a complete study's accounting must meet, named as the accounting names them."""

    n_models: int
    n_replay_banks: int
    pairs_per_model: int
    n_rc_runs: int
    n_replay_runs: int
    n_runs: int


@dataclass(frozen=True)
class RunOrdering:
    """The study's complete run order, compactly: lists whose stated nesting is every run."""

    experiment: str
    inputs: HandoffInputs
    describes: tuple[str, ...]
    models: tuple[str, ...]
    scenarios: tuple[str, ...]
    trackers: tuple[str, ...]
    replay_banks: tuple[ReplayBankKey, ...]
    expansion: str
    expected: ExpectedAccounting
    schema_version: int = field(default=RUN_ORDERING_VERSION)

    def __post_init__(self) -> None:
        """Every list names each item once, and the expected totals follow from the lists."""
        _require_header(self.experiment, self.schema_version, RUN_ORDERING_VERSION, "run ordering")
        _require_described(self.describes)
        _require_distinct("models", self.models)
        _require_distinct("scenarios", self.scenarios)
        _require_distinct("trackers", self.trackers)
        keys = [(bank.configuration, bank.assignment) for bank in self.replay_banks]
        if len(set(keys)) != len(keys):
            msg = "a replay bank is keyed more than once"
            raise ValueError(msg)
        pairs = len(self.scenarios) * len(self.trackers)
        derived = ExpectedAccounting(
            n_models=len(self.models),
            n_replay_banks=len(self.replay_banks),
            pairs_per_model=pairs,
            n_rc_runs=len(self.models) * pairs,
            n_replay_runs=len(self.replay_banks) * pairs,
            n_runs=(len(self.models) + len(self.replay_banks)) * pairs,
        )
        if derived != self.expected:
            msg = "the expected accounting totals do not re-derive from the ordered lists"
            raise ValueError(msg)

    def rc_runs(self) -> Iterator[tuple[str, str, str]]:
        """Every model run as ``(model, scenario, tracker)``, in the frozen nesting."""
        for model in self.models:
            for scenario in self.scenarios:
                for tracker in self.trackers:
                    yield (model, scenario, tracker)

    def replay_runs(self) -> Iterator[tuple[str, str, str, str]]:
        """Every replay run as ``(configuration, assignment, scenario, tracker)``, in the frozen nesting."""
        for bank in self.replay_banks:
            for scenario in self.scenarios:
                for tracker in self.trackers:
                    yield (bank.configuration, bank.assignment, scenario, tracker)


@dataclass(frozen=True)
class IllustrationCategory:
    """One category the illustration rule looks for, and what it selects."""

    name: str
    selects: str


@dataclass(frozen=True)
class RepresentativeRule:
    """How the report chooses the cases it illustrates, frozen before any verdict exists."""

    experiment: str
    inputs: HandoffInputs
    ordering_sha256: str
    describes: tuple[str, ...]
    arms: tuple[str, ...]
    arm_labels: tuple[str, ...]
    """The comparison arms as labelled at the parent: parent-specific arms take it, the all-ten arm has none."""
    parent: str
    configurations: tuple[str, ...]
    trackers: tuple[str, ...]
    categories: tuple[IllustrationCategory, ...]
    application: str
    tie_breaking: str
    deduplication: str
    absent: str
    baseline: str
    schema_version: int = field(default=REPRESENTATIVE_RULE_VERSION)

    def __post_init__(self) -> None:
        """The rule is complete: every list is distinct and every category says what it selects."""
        _require_header(self.experiment, self.schema_version, REPRESENTATIVE_RULE_VERSION, "representative rule")
        _require_described(self.describes)
        _require_hex("ordering_sha256", self.ordering_sha256)
        for name in ("arms", "arm_labels", "configurations", "trackers"):
            _require_distinct(name, cast("tuple[str, ...]", getattr(self, name)))
        if len(self.arm_labels) != len(self.arms):
            msg = "every comparison arm has exactly one label at the parent"
            raise ValueError(msg)
        _require_distinct("categories", tuple(category.name for category in self.categories))
        if not all(category.selects.strip() for category in self.categories):
            msg = "every category states what it selects"
            raise ValueError(msg)


@dataclass(frozen=True)
class RcRun:
    """One model run the audit re-simulates."""

    model: str
    scenario_id: str
    tracker: str


@dataclass(frozen=True)
class ReplayRun:
    """One replay run the audit re-simulates, named by the bank's protocol key."""

    configuration: str
    assignment: str
    scenario_id: str
    tracker: str


@dataclass(frozen=True)
class ResimulationFreeze:
    """The runs a clean-checkout audit re-simulates, each named before any of them exists."""

    experiment: str
    inputs: HandoffInputs
    ordering_sha256: str
    describes: tuple[str, ...]
    classes: tuple[str, ...]
    scenarios: tuple[str, ...]
    """One locked scenario per class, in class order."""
    trackers: tuple[str, ...]
    parent: str
    models: tuple[str, ...]
    rc_runs: tuple[RcRun, ...]
    replay_runs: tuple[ReplayRun, ...]
    n_rc_runs: int
    n_replay_runs: int
    n_runs: int
    """Stated explicitly: the subset's own counts are properties, which the result schema does not describe."""
    schema_version: int = field(default=RESIMULATION_SUBSET_VERSION)

    def __post_init__(self) -> None:
        """The identities are the stated product, in order, and the counts are theirs."""
        _require_header(self.experiment, self.schema_version, RESIMULATION_SUBSET_VERSION, "re-simulation subset")
        _require_described(self.describes)
        _require_hex("ordering_sha256", self.ordering_sha256)
        if len(self.scenarios) != len(self.classes):
            msg = "the subset takes exactly one scenario per class"
            raise ValueError(msg)
        product = tuple(
            RcRun(model=model, scenario_id=scenario, tracker=tracker)
            for model in self.models
            for scenario in self.scenarios
            for tracker in self.trackers
        )
        if self.rc_runs != product:
            msg = "the RC identities are not models x scenarios x trackers in the frozen order"
            raise ValueError(msg)
        if len(set(self.replay_runs)) != len(self.replay_runs):
            msg = "a replay run is named more than once"
            raise ValueError(msg)
        counts = (len(self.rc_runs), len(self.replay_runs), len(self.rc_runs) + len(self.replay_runs))
        if counts != (self.n_rc_runs, self.n_replay_runs, self.n_runs):
            msg = f"the stated counts {(self.n_rc_runs, self.n_replay_runs, self.n_runs)} are not {counts}"
            raise ValueError(msg)


# --- building them ---------------------------------------------------------------------------------


def handoff_inputs(study_file: Path, evaluation_file: Path, result_schema_file: Path) -> HandoffInputs:
    """Bind the three input files by digest, refusing a result schema other than the one referenced."""
    schema = load_schema(result_schema_file)
    if schema.schema_version != REFERENCED_SCHEMA_VERSION:
        msg = (
            f"{result_schema_file.name} is result schema version {schema.schema_version}; the handoff freezes "
            f"reference version {REFERENCED_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    return HandoffInputs(
        study_manifest_sha256=sha256_file(study_file),
        evaluation_sha256=sha256_file(evaluation_file),
        result_schema_sha256=sha256_file(result_schema_file),
        result_schema_version=schema.schema_version,
    )


def _replay_banks(manifest: StudyManifest) -> tuple[ReplayBankKey, ...]:
    """The banks as configuration x assignment, checked against what the models actually pair with."""
    banks = tuple(
        ReplayBankKey(
            configuration=configuration.label,
            assignment=assignment,
            warmup_s=configuration.warmup_s,
            velocity_cutoff_hz=configuration.velocity_cutoff_hz,
            acceleration_cutoff_hz=configuration.acceleration_cutoff_hz,
        )
        for configuration in manifest.configurations
        for assignment in ASSIGNMENTS
    )
    paired = {(entry.configuration, entry.arm.assignment) for entry in manifest.entries if entry.arm.assignment}
    if paired != {(bank.configuration, bank.assignment) for bank in banks}:
        msg = "the banks the models pair against are not every configuration crossed with every assignment"
        raise ValueError(msg)
    mismatched = sorted(
        entry.label
        for entry in manifest.entries
        if entry.warmup_s != manifest.configuration(entry.configuration).warmup_s
    )
    if mismatched:
        msg = f"models whose warm-up differs from their configuration's: {mismatched}"
        raise ValueError(msg)
    return banks


def run_ordering(
    manifest: StudyManifest, *, scenarios: Sequence[str], trackers: Sequence[str], inputs: HandoffInputs
) -> RunOrdering:
    """Freeze the study's run order from the manifest, the locked scenarios and the trackers."""
    models = tuple(entry.label for entry in manifest.entries)
    banks = _replay_banks(manifest)
    pairs = len(scenarios) * len(trackers)
    return RunOrdering(
        experiment=EXPERIMENT_LABEL,
        inputs=inputs,
        describes=_ORDERING_DESCRIBES,
        models=models,
        scenarios=tuple(scenarios),
        trackers=tuple(trackers),
        replay_banks=banks,
        expansion=_EXPANSION,
        expected=ExpectedAccounting(
            n_models=len(models),
            n_replay_banks=len(banks),
            pairs_per_model=pairs,
            n_rc_runs=len(models) * pairs,
            n_replay_runs=len(banks) * pairs,
            n_runs=(len(models) + len(banks)) * pairs,
        ),
    )


def _arm_labels(manifest: StudyManifest, parent: str) -> tuple[str, ...]:
    """The comparison arms as the first configuration labels them at ``parent``, in manifest order."""
    first = manifest.configurations[0].label
    labels = tuple(
        entry.label.removeprefix(f"{first}/")
        for entry in budget_entries(manifest, parent=parent)
        if entry.configuration == first
    )
    if len(labels) != len(ILLUSTRATION_ARMS):
        msg = f"expected one label per comparison arm {ILLUSTRATION_ARMS} at {parent}, got {labels}"
        raise ValueError(msg)
    return labels


def representative_rule(
    manifest: StudyManifest, *, trackers: Sequence[str], inputs: HandoffInputs, ordering_sha256: str
) -> RepresentativeRule:
    """Freeze the illustration rule as the code applies it, refusing a category it cannot describe."""
    undescribed = [name for name in ILLUSTRATION_CATEGORIES if name not in _CATEGORY_SELECTS]
    if undescribed:
        msg = f"categories the code declares but the freeze does not describe: {undescribed}"
        raise ValueError(msg)
    return RepresentativeRule(
        experiment=EXPERIMENT_LABEL,
        inputs=inputs,
        ordering_sha256=ordering_sha256,
        describes=_RULE_DESCRIBES,
        arms=ILLUSTRATION_ARMS,
        arm_labels=_arm_labels(manifest, BUDGET_PARENT),
        parent=BUDGET_PARENT,
        configurations=tuple(configuration.label for configuration in manifest.configurations),
        trackers=tuple(trackers),
        categories=tuple(
            IllustrationCategory(name=name, selects=_CATEGORY_SELECTS[name]) for name in ILLUSTRATION_CATEGORIES
        ),
        application=_APPLICATION,
        tie_breaking=_TIE_BREAKING,
        deduplication=_DEDUPLICATION,
        absent=_ABSENT,
        baseline=_BASELINE,
    )


def resimulation_freeze(
    manifest: StudyManifest,
    *,
    cases: Sequence[RobustnessScenario],
    trackers: Sequence[str],
    inputs: HandoffInputs,
    ordering_sha256: str,
) -> ResimulationFreeze:
    """Freeze the audit sample's explicit identities from the subset rule's own enumeration."""
    subset = resimulation_subset(manifest, scenarios=cases, trackers=trackers)
    banks = tuple(
        dict.fromkeys(
            (model.configuration, model.arm.assignment) for model in subset.models if model.arm.assignment is not None
        )
    )
    if len(banks) != subset.n_banks:
        msg = f"the subset's models pair against {len(banks)} banks, not the {subset.n_banks} it states"
        raise ValueError(msg)
    replay = tuple(
        ReplayRun(configuration=configuration, assignment=assignment, scenario_id=case.scenario_id, tracker=tracker)
        for configuration, assignment in banks
        for case in subset.scenarios
        for tracker in subset.trackers
    )
    rc = tuple(
        RcRun(model=model, scenario_id=scenario, tracker=tracker)
        for model, scenario, tracker in subset.run_identities()
    )
    return ResimulationFreeze(
        experiment=EXPERIMENT_LABEL,
        inputs=inputs,
        ordering_sha256=ordering_sha256,
        describes=_RESIMULATION_DESCRIBES,
        classes=RESIMULATION_CLASSES,
        scenarios=tuple(case.scenario_id for case in subset.scenarios),
        trackers=tuple(subset.trackers),
        parent=BUDGET_PARENT,
        models=tuple(model.label for model in subset.models),
        rc_runs=rc,
        replay_runs=replay,
        n_rc_runs=subset.n_rc_runs,
        n_replay_runs=subset.n_replay_runs,
        n_runs=subset.n_runs,
    )


# --- serialization and rendering -------------------------------------------------------------------


def handoff_to_json(record: RunOrdering | RepresentativeRule | ResimulationFreeze) -> str:
    """Canonical JSON of one freeze."""
    return canonical_json(to_mapping(record))


def load_handoff[T](path: Path, cls: type[T]) -> T:
    """Strictly rebuild one freeze from its JSON."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), cls)


def _inputs_table(inputs: HandoffInputs) -> list[str]:
    return [
        "| bound input | sha256 |",
        "| --- | --- |",
        f"| study manifest | `{inputs.study_manifest_sha256}` |",
        f"| evaluation configuration | `{inputs.evaluation_sha256}` |",
        f"| result schema v{inputs.result_schema_version} | `{inputs.result_schema_sha256}` |",
    ]


def _describes_line(describes: tuple[str, ...], version: int) -> str:
    names = ", ".join(f"`{name}`" for name in describes)
    return f"Record definitions are not restated here: see result schema v{version}, bound by digest below ({names})."


def _render_ordering(record: RunOrdering) -> list[str]:
    e = record.expected
    by_configuration: dict[str, list[str]] = {}
    for label in record.models:
        configuration, _, arm = label.partition("/")
        by_configuration.setdefault(configuration, []).append(arm)
    banks: dict[str, list[ReplayBankKey]] = {}
    for bank in record.replay_banks:
        banks.setdefault(bank.configuration, []).append(bank)
    return [
        f"# Task 1-a manual-demonstration run ordering (v{record.schema_version})",
        "",
        "Frozen before execution. The runs are the stated nesting over the lists below; nothing here is a result.",
        _describes_line(record.describes, record.inputs.result_schema_version),
        "",
        *_inputs_table(record.inputs),
        "",
        "## Expected accounting",
        "",
        "| figure | value |",
        "| --- | ---: |",
        f"| models | {e.n_models:,} |",
        f"| replay banks | {e.n_replay_banks:,} |",
        f"| pairs per model or bank | {e.pairs_per_model:,} |",
        f"| RC runs | {e.n_rc_runs:,} |",
        f"| replay runs | {e.n_replay_runs:,} |",
        f"| runs in total | {e.n_runs:,} |",
        "",
        "## Expansion",
        "",
        record.expansion,
        "",
        f"## Models ({e.n_models}, manifest order)",
        "",
        *(f"- `{name}`: {', '.join(f'`{arm}`' for arm in arms)}" for name, arms in by_configuration.items()),
        "",
        f"## Scenarios ({len(record.scenarios)}, locked order)",
        "",
        *(f"{index}. `{scenario}`" for index, scenario in enumerate(record.scenarios, start=1)),
        "",
        "## Trackers",
        "",
        *(f"{index}. `{tracker}`" for index, tracker in enumerate(record.trackers, start=1)),
        "",
        f"## Replay banks ({e.n_replay_banks}, configuration x assignment)",
        "",
        "| configuration | assignments | warm-up (s) | velocity cutoff (Hz) | acceleration cutoff (Hz) |",
        "| --- | --- | ---: | ---: | ---: |",
        *(
            f"| `{name}` | {', '.join(bank.assignment for bank in group)} | {group[0].warmup_s:g} | "
            f"{group[0].velocity_cutoff_hz:.6f} | {group[0].acceleration_cutoff_hz:.6f} |"
            for name, group in banks.items()
        ),
    ]


def _render_rule(record: RepresentativeRule) -> list[str]:
    return [
        f"# Task 1-a manual-demonstration representative-case rule (v{record.schema_version})",
        "",
        (
            "Frozen before execution. It chooses which cases the report ILLUSTRATES; aggregate conclusions "
            "come from the whole study, never from these cases. The selections are derived after execution."
        ),
        _describes_line(record.describes, record.inputs.result_schema_version),
        "",
        *_inputs_table(record.inputs),
        f"| run ordering (tie-breaking order) | `{record.ordering_sha256}` |",
        "",
        "## Scope",
        "",
        (
            f"- Comparison arms: {', '.join(f'`{label}`' for label in record.arm_labels)} "
            f"(kinds {', '.join(f'`{arm}`' for arm in record.arms)}, parent `{record.parent}`)."
        ),
        f"- Configurations: {', '.join(f'`{name}`' for name in record.configurations)}.",
        f"- Trackers: {', '.join(f'`{name}`' for name in record.trackers)}.",
        "",
        "## Categories (in declared order)",
        "",
        "| category | selects |",
        "| --- | --- |",
        *(f"| `{category.name}` | {category.selects} |" for category in record.categories),
        "",
        "## Application",
        "",
        record.application,
        "",
        "## Tie-breaking",
        "",
        record.tie_breaking,
        "",
        "## Deduplication",
        "",
        record.deduplication,
        "",
        "## Absent categories",
        "",
        record.absent,
        "",
        "## Replay baseline",
        "",
        record.baseline,
    ]


def _render_resimulation(record: ResimulationFreeze) -> list[str]:
    return [
        f"# Task 1-a manual-demonstration re-simulation subset (v{record.schema_version})",
        "",
        (
            "Frozen before execution: the runs a clean-checkout audit re-simulates, each named before it "
            "exists. The identities are listed explicitly; the JSON beside this file is authoritative."
        ),
        _describes_line(record.describes, record.inputs.result_schema_version),
        "",
        *_inputs_table(record.inputs),
        f"| run ordering | `{record.ordering_sha256}` |",
        "",
        "## Counts",
        "",
        "| figure | value |",
        "| --- | ---: |",
        f"| models | {len(record.models)} |",
        f"| scenarios (one per class) | {len(record.scenarios)} |",
        f"| trackers | {len(record.trackers)} |",
        f"| RC runs | {record.n_rc_runs} |",
        f"| replay runs | {record.n_replay_runs} |",
        f"| runs in total | {record.n_runs} |",
        "",
        "## Scenarios",
        "",
        "| class | scenario |",
        "| --- | --- |",
        *(f"| `{kind}` | `{scenario}` |" for kind, scenario in zip(record.classes, record.scenarios, strict=True)),
        "",
        f"## RC runs ({record.n_rc_runs}, model x scenario x tracker)",
        "",
        "| # | model | scenario | tracker |",
        "| ---: | --- | --- | --- |",
        *(
            f"| {index} | `{run.model}` | `{run.scenario_id}` | `{run.tracker}` |"
            for index, run in enumerate(record.rc_runs, start=1)
        ),
        "",
        f"## Replay runs ({record.n_replay_runs}, bank x scenario x tracker)",
        "",
        "| # | configuration | assignment | scenario | tracker |",
        "| ---: | --- | --- | --- | --- |",
        *(
            f"| {index} | `{run.configuration}` | `{run.assignment}` | `{run.scenario_id}` | `{run.tracker}` |"
            for index, run in enumerate(record.replay_runs, start=1)
        ),
    ]


def render_handoff_markdown(record: RunOrdering | RepresentativeRule | ResimulationFreeze) -> str:
    """The Markdown rendering of one freeze, generated from the record and never edited beside it."""
    if isinstance(record, RunOrdering):
        lines = _render_ordering(record)
    elif isinstance(record, RepresentativeRule):
        lines = _render_rule(record)
    else:
        lines = _render_resimulation(record)
    return "\n".join(lines) + "\n"


# --- the command -----------------------------------------------------------------------------------

_ARTIFACTS: Final = tuple(HANDOFF_VERSIONS)


def _bound_ordering(docs: Path, kind: str, trusted: RunOrdering) -> str:
    """The digest of the frozen ordering a later freeze cites, refusing any that is not the trusted one.

    Checking only the ordering's input digests let an ordering reordered under
    intact bindings through: the dependent freeze then cited that file while
    taking its scenarios from the configuration, so what it cited and what it
    used could disagree. The loaded ordering is compared field by field with the
    one the trusted inputs produce, and every field that differs is named.
    """
    ordering_file = docs / f"run_ordering_v{RUN_ORDERING_VERSION}.json"
    if not ordering_file.exists():
        msg = f"{ordering_file.name} must be frozen first: the {kind} orders its scenarios by it"
        raise FileNotFoundError(msg)
    loaded = load_handoff(ordering_file, RunOrdering)
    differing = [each.name for each in fields(RunOrdering) if getattr(loaded, each.name) != getattr(trusted, each.name)]
    if differing:
        msg = (
            f"{ordering_file.name} does not match the ordering the trusted inputs produce, so the {kind} cannot cite "
            f"it (fields that differ: {', '.join(differing)})"
        )
        raise ValueError(msg)
    return sha256_file(ordering_file)


def _freeze(args: argparse.Namespace) -> int:
    """Write one frozen artifact and its rendering, refusing to replace a version that already exists."""
    kind = cast("str", args.artifact)
    docs = Path(cast("str", args.docs))
    version = HANDOFF_VERSIONS[kind]
    output, markdown = docs / f"{kind}_v{version}.json", docs / f"{kind}_v{version}.md"
    for target in (output, markdown):
        if target.exists():
            msg = f"refusing to overwrite {target}: a frozen version is never replaced; a change is a new version"
            raise FileExistsError(msg)
    study_file = Path(cast("str", args.study))
    evaluation_file = Path(cast("str", args.evaluation))
    inputs = handoff_inputs(study_file, evaluation_file, Path(cast("str", args.result_schema)))
    manifest = load_study(study_file)
    config = load_manual_evaluation_config(evaluation_file)
    cases = evaluation_scenarios(load_development_robustness(config.development), load_manual_scenario(config.scenario))
    # The ordering as the trusted inputs produce it: frozen as it is, or the reference a dependent
    # freeze compares the committed ordering against before it may cite that ordering.
    trusted = run_ordering(
        manifest, scenarios=[case.scenario_id for case in cases], trackers=RECOVERY_TRACKERS, inputs=inputs
    )
    record: RunOrdering | RepresentativeRule | ResimulationFreeze
    if kind == "run_ordering":
        record = trusted
    elif kind == "representative_rule":
        digest = _bound_ordering(docs, kind, trusted)
        record = representative_rule(manifest, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=digest)
    else:
        digest = _bound_ordering(docs, kind, trusted)
        record = resimulation_freeze(
            manifest, cases=cases, trackers=RECOVERY_TRACKERS, inputs=inputs, ordering_sha256=digest
        )
    docs.mkdir(parents=True, exist_ok=True)
    output.write_text(handoff_to_json(record) + "\n", encoding="utf-8")
    markdown.write_text(render_handoff_markdown(record), encoding="utf-8")
    print(json.dumps({"artifact": kind, "version": version, "output": str(output)}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Freeze the manual-demonstration handoff artifacts.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    freeze = subparsers.add_parser("freeze", help="write one frozen handoff artifact and its Markdown rendering")
    freeze.add_argument("--artifact", choices=_ARTIFACTS, required=True, help="which artifact to freeze")
    freeze.add_argument("--study", type=str, required=True, help="frozen study manifest JSON")
    freeze.add_argument("--evaluation", type=str, required=True, help="manual evaluation config TOML")
    freeze.add_argument(
        "--result-schema", dest="result_schema", type=str, required=True, help="the result schema v2 JSON"
    )
    freeze.add_argument("--docs", type=str, required=True, help="the experiment's documentation directory")
    args = parser.parse_args(argv)
    return _freeze(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
