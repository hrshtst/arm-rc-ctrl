# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-007 (stage 2): the frozen study manifest binds all 186 model identities of the manual pilot.

Six inherited panel configurations crossed with the 31 approved arms give 186
models. The manifest binds every one of them to the locked demonstration bank,
the frozen contractive banks, the equal-episode weights, and the canonical
execution environment, and re-derives every invariant when it is loaded instead
of trusting what the document claims.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.arrays import array_digest
from arm_rc_ctrl.data.derivatives import DerivativeConfig, differentiate
from arm_rc_ctrl.data.manual_scenario import load_manual_scenario, manual_endpoint_positions
from arm_rc_ctrl.data.samples import SampleSet
from arm_rc_ctrl.execution import (
    AffinityRequest,
    BlasInfo,
    CoreTopology,
    ExecutionEnvironmentError,
    ExecutionRecord,
    OpenMpInfo,
    Probes,
    collect_execution,
)
from arm_rc_ctrl.experiments.manual_augmentation import MANUAL_PROTOCOL, ManualParent
from arm_rc_ctrl.experiments.manual_bank import BankManifest, TakeMeasurements, TakeVerdict, write_bank_manifest
from arm_rc_ctrl.experiments.manual_recipes import (
    ASSIGNMENTS,
    CONTRACTIVE_ARM,
    COPIES,
    MANUAL_ANCHOR,
    ROWS_REFERENCE,
    SYNTHETIC_EPISODES,
    TRANSFORM_SOURCE,
    ArmAccounting,
    ManualAnchor,
    ManualArmSpec,
    arm_accounting,
    esn_for_arm,
    fit_identity,
    manual_arms,
)
from arm_rc_ctrl.experiments.manual_study import (
    EXPERIMENT_LABEL,
    MODEL_COUNT,
    STUDY_SCHEMA_VERSION,
    ContractiveBank,
    StudyManifest,
    StudyMismatchError,
    build_study_manifest,
    contractive_banks,
    frozen_transform,
    load_study,
    main,
    render_study_markdown,
    study_to_json,
)
from arm_rc_ctrl.experiments.repetition_panel import load_panel
from arm_rc_ctrl.provenance import DirtyWorktreeError, ProvenanceRecord, collect_provenance, sha256_file
from arm_rc_ctrl.rc.esn import EsnConfig
from arm_rc_ctrl.rc.recipe import DatasetSource, RclibIdentity, TrainingValidation
from arm_rc_ctrl.rc.train import load_model_config
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from arm_rc_ctrl.experiments.manual_augmentation import ParentBankRecord

REPO_ROOT = repository_root()
PANEL_FILE = REPO_ROOT / "docs" / "experiments" / "task_1a_repeated_demonstration" / "panel_manifest_v1.json"
PANEL = load_panel(PANEL_FILE)
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "configs"
SCENARIO_FILE = FIXTURES / "planar_2dof_manual_fixture.toml"
SCENARIO = load_manual_scenario(SCENARIO_FILE)
PREPROCESSING_FILE = FIXTURES / "manual_derive_fixture.toml"
MODEL_FILE = REPO_ROOT / PANEL.configs.model_file
MODEL = load_model_config(MODEL_FILE)
TRANSFORM = frozen_transform(MODEL, root=REPO_ROOT)
VALIDATION = TrainingValidation.from_scenario(SCENARIO, SCENARIO_FILE, root=REPO_ROOT)
RCLIB = RclibIdentity.current()

DT = 0.01
DWELL_START_S = 1.5
HOLD_S = 0.2
GOAL_Q = (0.8, 0.4)
SEED_BANK = 3
DERIVATIVES = DerivativeConfig(method="central")
RECORD_DIRECTORY = "data/records/processed"
IDS = tuple(f"processed-20260916-{0xA00000000000 + index:012x}" for index in range(len(ASSIGNMENTS)))
NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)
TOPOLOGY = CoreTopology(model="fixture-cpu", online=(0, 1), core_types={}, core_ids={}, l2_cache_kib={})
PROBES = Probes(blas=lambda: BlasInfo(None, None, None, None, None), openmp=lambda: OpenMpInfo(None, None))


def _samples(n: int) -> SampleSet:
    """A complete manual recording of ``n`` samples: pre-roll, a smooth reach, and the final dwell at the target."""
    t = np.arange(n, dtype=np.float64) * DT
    start = np.asarray(SCENARIO.task.initial_q, dtype=np.float64)
    s = np.clip((t - HOLD_S) / (DWELL_START_S - HOLD_S), 0.0, 1.0)
    blend = s * s * (3.0 - 2.0 * s)
    q = start[None, :] + blend[:, None] * (np.asarray(GOAL_Q, dtype=np.float64) - start)[None, :]
    dq, ddq = differentiate(q, DT, DERIVATIVES)
    tip = manual_endpoint_positions(SCENARIO, q)
    dtip, ddtip = differentiate(tip, DT, DERIVATIVES)
    phase = np.where(t < HOLD_S, 0, np.where(t < DWELL_START_S, 1, 2)).astype(np.int64)
    return SampleSet(t, q, dq, ddq, tip, dtip, ddtip, np.zeros((n, 0), dtype=np.float64), phase)


# Ten recordings of deliberately unequal length, so every episode weight 400 / L_i differs.
SAMPLES = {artifact: _samples(201 + 4 * index) for index, artifact in enumerate(IDS)}
LOSS_ROWS = {name: SAMPLES[IDS[index]].n_samples - 1 for index, name in enumerate(ASSIGNMENTS)}


def _parent(index: int) -> ManualParent:
    """The parent identity the committed record of demonstration ``index`` would carry."""
    artifact = IDS[index]
    samples = SAMPLES[artifact]
    return ManualParent(
        assignment=ASSIGNMENTS[index],
        dataset=DatasetSource(artifact, array_digest(samples.t), f"{RECORD_DIRECTORY}/{artifact}.toml"),
        dwell_start_s=DWELL_START_S,
        n_samples=samples.n_samples,
        period_s=DT,
        derivative_method=DERIVATIVES.label,
        q_sha256=array_digest(samples.q),
        dq_sha256=array_digest(samples.dq),
    )


PARENTS = tuple(_parent(index) for index in range(len(ASSIGNMENTS)))


def _execution(*, model: str = "fixture-cpu", cpus: tuple[int, ...] = (0, 1)) -> ExecutionRecord:
    """A canonical execution record of a pinned, single-threaded process (no sysfs and no probes of this machine)."""
    request = AffinityRequest("explicit", cpus)
    return collect_execution(
        command="python -m arm_rc_ctrl.experiments.manual_study",
        env=request.environment(),
        topology=dataclasses.replace(TOPOLOGY, model=model),
        effective_cpus=cpus,
        probes=PROBES,
        now=NOW,
    )


EXECUTION = _execution()


def _bank_manifest() -> BankManifest:
    """The locked bank the parents came from: ten accepted takes and one retained rejection."""
    takes = [
        TakeVerdict(
            batch=1,
            attempt=index + 1,
            source_file=f"reach_{index + 1:03d}.sklog.npz",
            accepted=True,
            reasons=(),
            raw_artifact_id=f"raw-20260916-{0xB00000000000 + index:012x}",
            processed_artifact_id=IDS[index],
            payload_sha256=array_digest(SAMPLES[IDS[index]].t),
            q_sha256=array_digest(SAMPLES[IDS[index]].q),
            duplicate_of=None,
            assignment=ASSIGNMENTS[index],
            measurements=TakeMeasurements(),
        )
        for index in range(len(ASSIGNMENTS))
    ]
    takes.append(
        TakeVerdict(
            batch=2,
            attempt=len(ASSIGNMENTS) + 1,
            source_file=f"reach_{len(ASSIGNMENTS) + 1:03d}.sklog.npz",
            accepted=False,
            reasons=("final dwell shorter than 1.00 s",),
            raw_artifact_id=f"raw-20260916-{0xB0000000000B:012x}",
            processed_artifact_id=None,
            payload_sha256="cd" * 32,
            q_sha256="ef" * 32,
            duplicate_of=None,
            assignment=None,
            measurements=TakeMeasurements(),
        )
    )
    return BankManifest(
        bank_schema_version=1,
        protocol=MANUAL_PROTOCOL,
        session="fixture-01",
        scenario_path=SCENARIO_FILE.relative_to(REPO_ROOT).as_posix(),
        scenario_sha256=sha256_file(SCENARIO_FILE),
        derive_config_path=PREPROCESSING_FILE.relative_to(REPO_ROOT).as_posix(),
        derive_config_sha256=sha256_file(PREPROCESSING_FILE),
        required=len(ASSIGNMENTS),
        takes=tuple(takes),
        updated_at="2026-09-16T12:00:00+00:00",
    )


@pytest.fixture(scope="module")
def bank_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The locked bank manifest of the fixture demonstrations, written where the freeze would read it."""
    path = tmp_path_factory.mktemp("bank") / "bank_v1.json"
    write_bank_manifest(path, _bank_manifest())
    return path


@pytest.fixture(scope="module")
def banks() -> dict[str, ParentBankRecord]:
    """One frozen contractive bank per parent at the study's seed bank."""
    return contractive_banks(PARENTS, SAMPLES, SCENARIO, seed_bank=SEED_BANK)


@pytest.fixture(scope="module")
def provenance() -> ProvenanceRecord:
    """One exploratory provenance record shared by the fixture manifests."""
    return collect_provenance({}, seeds={"contractive_seed_bank": SEED_BANK}, exploratory=True)


def _build(
    bank_file: Path,
    banks: Mapping[str, ParentBankRecord],
    provenance: ProvenanceRecord,
    *,
    seed_bank: int = SEED_BANK,
    execution: ExecutionRecord | None = None,
    parents: tuple[ManualParent, ...] = PARENTS,
    model_file: Path = MODEL_FILE,
) -> StudyManifest:
    """Freeze the fixture study, overriding one input at a time."""
    return build_study_manifest(
        panel=PANEL,
        panel_file=PANEL_FILE,
        bank=_bank_manifest(),
        bank_file=bank_file,
        scenario_file=SCENARIO_FILE,
        preprocessing_file=PREPROCESSING_FILE,
        model=MODEL,
        model_file=model_file,
        parents=parents,
        banks=banks,
        transform=TRANSFORM,
        validation=VALIDATION,
        execution=EXECUTION if execution is None else execution,
        provenance=provenance,
        seed_bank=seed_bank,
        rclib=RCLIB,
        root=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def manifest(bank_file: Path, banks: dict[str, ParentBankRecord], provenance: ProvenanceRecord) -> StudyManifest:
    """The frozen study manifest of the fixture bank."""
    return _build(bank_file, banks, provenance)


def test_manifest_enumerates_the_186_models_of_the_approved_scope(manifest: StudyManifest) -> None:
    """Six configurations times 31 arms: 60 singletons, 6 all-ten models, 60 duplication controls, 60 banks."""
    assert manifest.experiment == EXPERIMENT_LABEL == "task_1a_manual_v1"
    assert manifest.schema_version == STUDY_SCHEMA_VERSION
    assert len(manifest.configurations) == 6
    assert [c.label for c in manifest.configurations] == [e.label for e in PANEL.entries]
    assert [c.source_trial for c in manifest.configurations] == [17, 136, 53, 1, 0, 28]
    assert len(manifest.entries) == MODEL_COUNT == 186 == 6 * 31

    pairs = [(entry.configuration, entry.arm.label) for entry in manifest.entries]
    assert len(set(pairs)) == MODEL_COUNT
    assert pairs == [(c.label, arm.label) for c in manifest.configurations for arm in manual_arms()]
    assert len({entry.fit_identity for entry in manifest.entries}) == MODEL_COUNT
    assert all(len(entry.fit_identity) == 64 for entry in manifest.entries)

    kinds = {arm: sum(1 for entry in manifest.entries if entry.arm.arm == arm) for arm in ("S", "M10", "R10", "C10")}
    assert kinds == {"S": 60, "M10": 6, "R10": 60, "C10": 60}
    assert sum(kinds.values()) == MODEL_COUNT
    assert manifest.models_by_arm == kinds
    assert manifest.entry("feasible-best", "S/D01").label == "feasible-best/S/D01"
    assert [d.artifact_id for d in manifest.datasets(manifest.entry("feasible-best", "M10"))] == list(IDS)
    assert manifest.configuration("feasible-worst").source_trial == 53
    with pytest.raises(KeyError):
        manifest.entry("feasible-best", "M100")  # the whole-bank duplication control stays deferred (D4)
    with pytest.raises(KeyError):
        manifest.configuration("feasible-nowhere")


def test_solver_alphas_are_the_episode_count_times_the_configuration_alpha(manifest: StudyManifest) -> None:
    """``S`` fits one episode at ``alpha_0``; every other arm fits ten at ``10 alpha_0`` with equal episode weight."""
    by_label = {c.label: c for c in manifest.configurations}
    panel_alphas = {e.label: e.base_alpha for e in PANEL.entries}
    for entry in manifest.entries:
        configuration = by_label[entry.configuration]
        accounting, esn = manifest.accounting(entry), manifest.esn(entry)
        count = 1 if entry.arm.arm == "S" else 10
        assert entry.arm.count == count == accounting.episodes == entry.episodes
        assert configuration.base_alpha == panel_alphas[entry.configuration]
        assert accounting.base_alpha == configuration.base_alpha
        assert accounting.solver_alpha == count * configuration.base_alpha
        assert esn.readout.alpha == accounting.solver_alpha
        assert esn.readout.explicit_bias is True
        assert esn.readout.include_bias is False
        assert esn.reservoir == configuration.reservoir
        assert entry.warmup_s == configuration.warmup_s
        assert entry.source_trial == configuration.source_trial
        assert accounting.regularization_lambda == configuration.base_alpha / ROWS_REFERENCE
        assert accounting.total_loss_weight == pytest.approx(ROWS_REFERENCE * count)
        assert accounting.weight_reference_rows == MANUAL_ANCHOR.weight_reference_rows

    single = manifest.accounting(manifest.entry("feasible-best", "S/D02"))
    assert single.loss_rows == (LOSS_ROWS["D02"],)
    assert single.row_weights == (ROWS_REFERENCE / LOSS_ROWS["D02"],)
    everything = manifest.accounting(manifest.entry("feasible-best", "M10"))
    assert everything.loss_rows == tuple(LOSS_ROWS[name] for name in ASSIGNMENTS)
    assert len(set(everything.row_weights)) == len(ASSIGNMENTS)  # unequal recordings, unequal weights
    control = manifest.accounting(manifest.entry("feasible-best", "R10/D02"))
    assert control.loss_rows == (LOSS_ROWS["D02"],) * COPIES
    assert (control.unique_sources, control.copies) == (1, COPIES - 1)
    grown = manifest.accounting(manifest.entry("feasible-best", "C10/D02"))
    assert grown.synthetic == SYNTHETIC_EPISODES
    assert grown.total_loss_weight == pytest.approx(everything.total_loss_weight)


def test_contractive_entries_bind_their_parents_bank_and_seed_bank(
    manifest: StudyManifest, bank_file: Path, provenance: ProvenanceRecord, banks: dict[str, ParentBankRecord]
) -> None:
    """Every ``C10`` entry names its parent's bank digest; another seed bank gives other digests and other keys."""
    contractive = [entry for entry in manifest.entries if entry.arm.arm == CONTRACTIVE_ARM]
    assert len(contractive) == 60
    assert all(entry.contractive is None for entry in manifest.entries if entry.arm.arm != CONTRACTIVE_ARM)
    digests: set[str] = set()
    for entry in contractive:
        construction = entry.contractive
        assert construction is not None
        assert construction.assignment == entry.arm.assignment
        assert construction.parent == manifest.datasets(entry)[0].artifact_id
        assert construction.seed_bank == manifest.seed_bank == SEED_BANK
        assert construction.dwell_start_s == DWELL_START_S
        assert construction.bank_sha256 == banks[construction.assignment].bank_sha256
        assert construction.spec.seed_bank == SEED_BANK
        digests.add(construction.bank_sha256)
    assert len(digests) == len(ASSIGNMENTS)  # one bank per parent, never one bank reused

    shifted_banks = contractive_banks(PARENTS, SAMPLES, SCENARIO, seed_bank=SEED_BANK + 1)
    assert all(shifted_banks[name].bank_sha256 != banks[name].bank_sha256 for name in ASSIGNMENTS)
    shifted = _build(bank_file, shifted_banks, provenance, seed_bank=SEED_BANK + 1)
    for entry in manifest.entries:
        other = shifted.entry(entry.configuration, entry.arm.label)
        if entry.arm.arm == CONTRACTIVE_ARM:
            assert other.fit_identity != entry.fit_identity  # the seed bank is part of the key
        else:
            assert other.fit_identity == entry.fit_identity  # recorded-data arms never depend on it


def test_every_entry_binds_the_canonical_execution_identity(manifest: StudyManifest, tmp_path: Path) -> None:
    """The environment is part of every fit key; a foreign or non-canonical record is refused."""
    assert manifest.execution.canonical
    assert {entry.execution_identity for entry in manifest.entries} == {manifest.execution.identity}
    foreign = _execution(model="another-cpu")
    assert foreign.identity != manifest.execution.identity
    with pytest.raises(StudyMismatchError, match="execution identity"):
        dataclasses.replace(manifest, execution=foreign)
    unmanaged = collect_execution(
        command="python -m arm_rc_ctrl.experiments.manual_study",
        env={},
        topology=TOPOLOGY,
        effective_cpus=(0, 1),
        probes=PROBES,
        now=NOW,
    )
    assert not unmanaged.canonical
    with pytest.raises(ExecutionEnvironmentError, match="canonical"):
        dataclasses.replace(manifest, execution=unmanaged)
    document = json.loads(study_to_json(manifest))
    document["entries"][0]["execution_identity"] = "c" * 64
    file = tmp_path / "study.json"
    file.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="execution identity"):  # the strict loader reports it as a document error
        load_study(file)


def test_manifest_round_trips_and_renders_markdown(manifest: StudyManifest, tmp_path: Path) -> None:
    """Canonical JSON reloads to the same manifest, names no machine path, and the Markdown re-renders exactly."""
    text = study_to_json(manifest)
    file = tmp_path / "study_manifest_v1.json"
    file.write_text(text + "\n", encoding="utf-8")
    loaded = load_study(file)
    assert loaded == manifest
    assert study_to_json(loaded) == text
    # The sorted-key document cannot carry the transform's channel mapping, so it is recorded channel-wise.
    assert loaded.transform.transform == TRANSFORM
    assert [channel.name for channel in loaded.transform.channels] == ["q", "dq"]
    assert not re.search(r"(?<![\w./])/(?:home|tmp|Users|mnt)/", text)
    markdown = render_study_markdown(manifest)
    assert not re.search(r"(?<![\w./])/(?:home|tmp|Users|mnt)/", markdown)
    assert render_study_markdown(loaded) == markdown
    rendering = tmp_path / "study_manifest_v1.md"
    rendering.write_text(markdown, encoding="utf-8")
    assert rendering.read_text(encoding="utf-8") == render_study_markdown(load_study(file))
    for required in (
        "# Task 1-a manual study manifest (v1)",
        "## Sources",
        "## Configurations",
        "## Arms",
        "## Demonstrations",
        "## Models",
        "## Provenance",
        "186 models",
        "feasible-best",
        TRANSFORM_SOURCE.artifact_id,
        "| D01 |",
        "10 alpha_0",
    ):
        assert required in markdown, required


def test_tampered_manifests_are_refused_on_load(manifest: StudyManifest, tmp_path: Path) -> None:
    """A changed count, alpha, arm, duplicate pair, missing bank digest, or stale key never loads as the study."""
    file = tmp_path / "tampered.json"

    def reload(document: object) -> StudyManifest:
        file.write_text(json.dumps(document), encoding="utf-8")
        return load_study(file)

    # The header records each demonstration once; a recording without a loss row fits nothing.
    counted = json.loads(study_to_json(manifest))
    counted["demonstrations"][0]["loss_rows"] = 0
    with pytest.raises(ValueError, match="loss_rows"):
        reload(counted)

    # Two bank positions sharing one dataset would let M10 fit the same recording twice.
    repeated = json.loads(study_to_json(manifest))
    repeated["demonstrations"][1]["dataset"] = json.loads(json.dumps(repeated["demonstrations"][0]["dataset"]))
    with pytest.raises(ValueError, match="distinct dataset"):
        reload(repeated)

    out_of_order = json.loads(study_to_json(manifest))
    out_of_order["demonstrations"].reverse()
    with pytest.raises(ValueError, match="bank order"):
        reload(out_of_order)

    # A dataset the header no longer binds by its committed payload no longer re-derives the keys it was hashed into.
    payload = json.loads(study_to_json(manifest))
    payload["demonstrations"][0]["dataset"]["payload_sha256"] = "ab" * 32
    with pytest.raises(ValueError, match="does not re-derive"):
        reload(payload)

    foreign_parent = json.loads(study_to_json(manifest))
    foreign_parent["demonstrations"][0]["dataset"]["artifact_id"] = "processed-20260916-000000000000"
    with pytest.raises(ValueError, match="bank was grown from"):
        reload(foreign_parent)

    # The readout solver and the inherited reservoir are bound once and rebuild every entry's fitted configuration.
    solver = json.loads(study_to_json(manifest))
    solver["readout"]["solver"] = "dual_cholesky"
    with pytest.raises(ValueError, match="does not re-derive"):
        reload(solver)

    reservoir = json.loads(study_to_json(manifest))
    reservoir["configurations"][0]["reservoir"]["n_neurons"] += 1
    with pytest.raises(ValueError, match="does not re-derive"):
        reload(reservoir)

    # The fixed-alpha diagnostic stays deferred (D4): the header binds the count-scaled ridge rule, and no
    # document can state a ten-episode arm that trains at alpha_0.
    fixed_alpha = json.loads(study_to_json(manifest))
    fixed_alpha["anchor"]["regularization_rule"] = "base"
    with pytest.raises(ValueError, match="count_scaled"):
        reload(fixed_alpha)

    duplicated = json.loads(study_to_json(manifest))
    duplicated["entries"][1] = json.loads(json.dumps(duplicated["entries"][0]))
    with pytest.raises(ValueError, match="twice"):
        reload(duplicated)

    deferred = json.loads(study_to_json(manifest))
    deferred["entries"][0]["arm"]["arm"] = "M100"
    with pytest.raises(ValueError, match="arm must be one of"):
        reload(deferred)

    without_bank = json.loads(study_to_json(manifest))
    grown = next(entry for entry in without_bank["entries"] if entry["arm"]["arm"] == CONTRACTIVE_ARM)
    grown["contractive"] = None
    with pytest.raises(ValueError, match="bank digest"):
        reload(without_bank)

    stale = json.loads(study_to_json(manifest))
    stale["entries"][0]["fit_identity"] = "d" * 64
    with pytest.raises(ValueError, match="does not re-derive"):
        reload(stale)

    reordered = json.loads(study_to_json(manifest))
    reordered["entries"].reverse()
    with pytest.raises(ValueError, match="report order"):
        reload(reordered)


def test_entries_reference_the_header_instead_of_embedding_their_fit_inputs(manifest: StudyManifest) -> None:
    """An entry stores what identifies it; its ESN, datasets, and accounting rebuild from the header it references."""
    document = json.loads(study_to_json(manifest))
    assert {key for entry in document["entries"] for key in entry} == {
        "arm",
        "configuration",
        "contractive",
        "execution_identity",
        "fit_identity",
        "source_trial",
        "warmup_s",
    }
    assert [record["assignment"] for record in document["demonstrations"]] == list(ASSIGNMENTS)
    assert document["readout"] == to_mapping(MODEL.esn.readout)
    assert manifest.sources == {name: parent.dataset for name, parent in zip(ASSIGNMENTS, PARENTS, strict=True)}
    assert manifest.loss_rows == LOSS_ROWS

    entry = manifest.entry("feasible-middle", "R10/D03")
    configuration = manifest.configuration("feasible-middle")
    assert manifest.datasets(entry) == (PARENTS[2].dataset,)
    assert manifest.esn(entry) == esn_for_arm(
        EsnConfig(reservoir=configuration.reservoir, readout=MODEL.esn.readout),
        entry.arm,
        base_alpha=configuration.base_alpha,
    )
    assert manifest.accounting(entry) == arm_accounting(
        entry.arm, base_alpha=configuration.base_alpha, loss_rows=LOSS_ROWS
    )
    # The recorded identity is the witness of the rebuild: it was hashed from exactly these values.
    assert entry.fit_identity == fit_identity(
        configuration=entry.configuration,
        arm=entry.arm,
        warmup_s=entry.warmup_s,
        base_alpha=configuration.base_alpha,
        esn=manifest.esn(entry),
        datasets=manifest.datasets(entry),
        transform=TRANSFORM,
        validation=VALIDATION,
        rclib_commit=RCLIB.commit,
        execution_identity=EXECUTION.identity,
    )


def test_the_loader_refuses_a_rebuild_that_no_longer_matches_the_arms(
    manifest: StudyManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rebuilt inputs are checked, not trusted: another ridge rule, accounting, or bias layout never loads."""
    from arm_rc_ctrl.experiments import manual_study

    # A ten-episode arm fitted at alpha_0 is the fixed-alpha diagnostic that stays deferred (D4).
    monkeypatch.setattr(manual_study, "RIDGE_RULE", "count_divided")
    with pytest.raises(StudyMismatchError, match="solver parameter"):
        dataclasses.replace(manifest)
    monkeypatch.undo()

    inherited_accounting = manual_study.arm_accounting

    def foreign_accounting(
        arm: ManualArmSpec, *, base_alpha: float, loss_rows: Mapping[str, int], anchor: ManualAnchor = MANUAL_ANCHOR
    ) -> ArmAccounting:
        """The accounting of a different arm than the one that was asked for."""
        other = ManualArmSpec("M10") if arm.label != "M10" else ManualArmSpec("S", ASSIGNMENTS[0])
        return inherited_accounting(other, base_alpha=base_alpha, loss_rows=loss_rows, anchor=anchor)

    monkeypatch.setattr(manual_study, "arm_accounting", foreign_accounting)
    with pytest.raises(StudyMismatchError, match="not the arm"):
        dataclasses.replace(manifest)
    monkeypatch.undo()

    inherited_esn = manual_study.esn_for_arm

    def implicit_bias(
        base: EsnConfig, arm: ManualArmSpec, *, base_alpha: float, anchor: ManualAnchor = MANUAL_ANCHOR
    ) -> EsnConfig:
        """The historical implicit-bias layout the equal-episode weighting replaced (I3)."""
        fitted = inherited_esn(base, arm, base_alpha=base_alpha, anchor=anchor)
        readout = dataclasses.replace(fitted.readout, explicit_bias=None, include_bias=True)
        return EsnConfig(reservoir=fitted.reservoir, readout=readout)

    monkeypatch.setattr(manual_study, "esn_for_arm", implicit_bias)
    with pytest.raises(StudyMismatchError, match="explicit-bias readout"):
        dataclasses.replace(manifest)


def test_build_refuses_sources_that_do_not_match_the_bank_or_the_panel(
    bank_file: Path, banks: dict[str, ParentBankRecord], provenance: ProvenanceRecord
) -> None:
    """Banks from another seed bank, a parent outside the locked bank, and a moved model file all fail the binding."""
    with pytest.raises(ValueError, match="seed bank"):
        _build(bank_file, banks, provenance, seed_bank=SEED_BANK + 1)
    foreign = dataclasses.replace(PARENTS[0], dataset=dataclasses.replace(PARENTS[0].dataset, artifact_id=IDS[1]))
    with pytest.raises(ValueError, match="locked bank"):
        _build(bank_file, banks, provenance, parents=(foreign, *PARENTS[1:]))
    with pytest.raises(ValueError, match="model configuration"):
        _build(bank_file, banks, provenance, model_file=FIXTURES / "esn_fixture.toml")
    incomplete = {name: record for name, record in banks.items() if name != "D07"}
    with pytest.raises(ValueError, match="D07"):
        _build(bank_file, incomplete, provenance)
    stale = dict(banks)
    stale["D03"] = dataclasses.replace(banks["D03"], bank_sha256="ab" * 32)
    with pytest.raises(ValueError, match="bank_sha256"):
        _build(bank_file, stale, provenance)


def test_frozen_transform_copies_the_historical_scripted_statistics() -> None:
    """The input transform is the predecessor's physical one, digest-bound and never computed from a take (I8)."""
    assert TRANSFORM.derived_from == (TRANSFORM_SOURCE.artifact_id,)
    assert TRANSFORM.policy == MODEL.input_transform.policy == "fixed_scale"
    assert TRANSFORM.fixed_scales == MODEL.input_transform.fixed_scales
    assert MANUAL_ANCHOR.transform_source == TRANSFORM_SOURCE
    assert frozen_transform(MODEL, root=REPO_ROOT) == TRANSFORM


def test_main_refuses_an_existing_output_and_a_dirty_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bank_file: Path
) -> None:
    """The command never overwrites a frozen artifact and never freezes a study from a modified checkout."""
    from arm_rc_ctrl.experiments import manual_study

    output = tmp_path / "study_manifest_v1.json"
    markdown = tmp_path / "study_manifest_v1.md"
    argv = [
        "--panel",
        str(PANEL_FILE),
        "--bank",
        str(bank_file),
        "--scenario",
        str(SCENARIO_FILE),
        "--preprocessing",
        str(PREPROCESSING_FILE),
        "--seed-bank",
        str(SEED_BANK),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
    ]
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
    output.unlink()

    def dirty(_root: Path) -> tuple[str, bool]:
        """The worktree state of a modified checkout."""
        return "0" * 40, True

    monkeypatch.setattr(manual_study, "worktree_state", dirty)
    with pytest.raises(DirtyWorktreeError, match="clean checkout"):
        main(argv)
    monkeypatch.setattr(manual_study, "require_canonical", lambda: None)
    exploratory = [*argv, "--exploratory"]
    exploratory[1] = str(tmp_path / "absent_panel.json")
    with pytest.raises(FileNotFoundError):
        main(exploratory)  # the dirty worktree is tolerated; the run fails on the missing panel instead
    assert not output.exists()


def test_contractive_bank_entries_validate_their_fields() -> None:
    """A recorded bank names a bank position, its parent, a seed bank, the dwell onset, and a full digest."""
    valid = ContractiveBank(
        assignment="D01", parent=IDS[0], seed_bank=SEED_BANK, dwell_start_s=DWELL_START_S, bank_sha256="ab" * 32
    )
    assert valid.spec.seed_bank == SEED_BANK
    assert valid.spec.dwell_start_s == DWELL_START_S
    for changes, message in (
        ({"assignment": "D11"}, "bank position"),
        ({"parent": " "}, "parent"),
        ({"seed_bank": -1}, "seed_bank"),
        ({"dwell_start_s": 0.0}, "dwell_start_s"),
        ({"bank_sha256": "ab"}, "bank_sha256"),
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(valid, **changes)
    assert ManualArmSpec(CONTRACTIVE_ARM, "D01").synthetic == SYNTHETIC_EPISODES
