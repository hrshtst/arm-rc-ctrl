# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Validate and visualize every augmentation family before training (M3R-005; recovery plan gate 3).

A reproducible development command generates every configuration of the
approved D1 grid from the recovery dataset and checks, per configuration and
family: deterministic regeneration, realized-amplitude bounds, temporal
smoothness, AR(1) lag-1 correlation, the exact contractive envelope identity,
exact dwell collapse through the locked terminal taper, episode separation,
and rejection accounting; globally it checks the dataset source binding and
seed-bank separation. Failures are recorded in the report — never swallowed —
and the command exits non-zero after writing it. Augmentation streams use the
dedicated ``[SEED_NAMESPACE, seed_bank, attempt]`` namespace, so no development
or confirmatory evaluation seed is consumed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.derivatives import DerivativeConfig
from arm_rc_ctrl.data.records import load_record, verify_payload
from arm_rc_ctrl.data.recovery import RecoveryDatasetRecord
from arm_rc_ctrl.data.samples import load_samples
from arm_rc_ctrl.provenance import (
    ArtifactReference,
    ProvenanceRecord,
    canonical_json,
    collect_provenance,
    command_line,
    require_clean_for_confirmatory,
    sha256_bytes,
)
from arm_rc_ctrl.rc.augment import (
    APPROVED_GAMMA,
    APPROVED_N_SYNTHETIC,
    APPROVED_PHI,
    APPROVED_SIGMA_RAD,
    SEED_NAMESPACE,
    TAPER_DURATION_S,
    TAPER_ZERO_MARGIN_S,
    AugmentationConfig,
    AugmentationError,
    AugmentationResult,
    EpisodeArrays,
    Rejection,
    contraction_envelope,
    generate_augmentation,
    terminal_taper,
)
from arm_rc_ctrl.scenario import load_scenario
from arm_rc_ctrl.storage import open_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arm_rc_ctrl.data.samples import SampleSet
    from arm_rc_ctrl.scenario import ScenarioConfig

__all__ = [
    "ANCHOR",
    "REPORT_SCHEMA_VERSION",
    "AugmentationValidationReport",
    "CheckResult",
    "ConfigurationReport",
    "FamilyStats",
    "approved_grid",
    "main",
    "plot_configuration",
    "report_to_json",
    "report_to_markdown",
    "validate_augmentation",
]

REPORT_SCHEMA_VERSION: Final = 2

ANCHOR: Final = AugmentationConfig(n_synthetic=64, sigma_rad=0.05, phi=0.99, gamma=1.0, seed_bank=1, attempt_budget=256)
"""The D1 anchor configuration ``(64, 0.05, phi=0.99, gamma=1)``; its episodes are the plotted set."""

_FAMILIES: Final = ("non_decaying", "contractive")
_CORRELATION_TOLERANCE: Final = 0.05
"""Allowed deviation of the pooled lag-1 autocorrelation estimate from the configured phi."""
_PEAK_SIGMA_FACTOR: Final = 6.0
"""Realized peaks beyond this multiple of sigma indicate a broken latent process."""
_ENVELOPE_TOLERANCE_RAD: Final = 1e-12
"""Allowed float discrepancy of the contractive = envelope * non-decaying identity."""
_CHECK_NAMES: Final = (
    "determinism",
    "bounds",
    "smoothness",
    "correlation",
    "envelope",
    "dwell-collapse",
    "episode-separation",
    "rejection-accounting",
)


def approved_grid(*, seed_bank: int = 1, attempt_factor: int = 4) -> tuple[AugmentationConfig, ...]:
    """Every approved D1 combination, with a finite attempt budget proportional to the episode count."""
    return tuple(
        AugmentationConfig(
            n_synthetic=n, sigma_rad=sigma, phi=phi, gamma=gamma, seed_bank=seed_bank, attempt_budget=attempt_factor * n
        )
        for n in sorted(APPROVED_N_SYNTHETIC)
        for sigma in sorted(APPROVED_SIGMA_RAD)
        for phi in sorted(APPROVED_PHI)
        for gamma in sorted(APPROVED_GAMMA)
    )


@dataclass(frozen=True)
class CheckResult:
    """One named validation check and its outcome; failures carry the measured detail."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class FamilyStats:
    """Realized per-episode perturbation statistics of one family."""

    family: Literal["non_decaying", "contractive"]
    rms_min: float
    rms_median: float
    rms_max: float
    peak_min: float
    peak_median: float
    peak_max: float


@dataclass(frozen=True)
class ConfigurationReport:
    """Checks, statistics, and full rejection accounting of one augmentation configuration."""

    config: AugmentationConfig
    attempts_used: int
    rejected_attempts: int
    """Distinct seeded attempts rejected (an attempt is rejected whole when either family fails)."""
    family_rejections: int
    """Family-level rejection records (one attempt can contribute up to two)."""
    rejections: tuple[Rejection, ...]
    """Every rejection with its attempt index, family, and reason (rejected attempts stay reviewable)."""
    families: tuple[FamilyStats, ...]
    checks: tuple[CheckResult, ...]
    digests_sha256: str
    """SHA-256 over the canonical JSON of every generated-array digest (the determinism witness)."""

    def __post_init__(self) -> None:
        """The counts must agree with the persisted rejection records."""
        attempts = {rejection.attempt for rejection in self.rejections}
        if self.family_rejections != len(self.rejections) or self.rejected_attempts != len(attempts):
            msg = (
                f"rejection accounting mismatch: {self.rejected_attempts} attempts / "
                f"{self.family_rejections} family records vs {len(attempts)} / {len(self.rejections)} persisted"
            )
            raise ValueError(msg)

    @property
    def passed(self) -> bool:
        """Whether every check of this configuration passed."""
        return all(check.passed for check in self.checks)


@dataclass(frozen=True)
class AugmentationValidationReport:
    """The complete augmentation validation outcome (human-reviewable via :func:`report_to_markdown`)."""

    dataset: str
    payload_sha256: str
    scenario_name: str
    derivative_method: str
    seed_namespace: int
    taper_duration_s: float
    taper_zero_margin_s: float
    dwell_start_s: float
    configurations: tuple[ConfigurationReport, ...]
    checks: tuple[CheckResult, ...]
    """Dataset-global checks (source binding, seed-bank separation)."""
    passed: bool
    provenance: ProvenanceRecord
    schema_version: int = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require the recorded outcome to equal the conjunction of every check."""
        if self.schema_version != REPORT_SCHEMA_VERSION:
            msg = f"unsupported report schema_version {self.schema_version}; expected {REPORT_SCHEMA_VERSION}"
            raise ValueError(msg)
        expected = all(c.passed for c in self.checks) and all(cfg.passed for cfg in self.configurations)
        if self.passed != expected:
            msg = f"passed={self.passed} contradicts the recorded checks (expected {expected})"
            raise ValueError(msg)


def _derivative_config(label: str) -> DerivativeConfig:
    methods = {"central-difference": "central", "cubic-spline": "spline"}
    if label not in methods:
        msg = f"unknown derivative policy label {label!r}; expected one of {sorted(methods)}"
        raise ValueError(msg)
    return DerivativeConfig(method=cast('Literal["central", "spline"]', methods[label]))


def _family_stats(result: AugmentationResult, family: str) -> FamilyStats:
    arrays = [cast("EpisodeArrays", getattr(episode, family)) for episode in result.episodes]
    rms = [a.delta_rms_rad for a in arrays]
    peak = [a.delta_peak_rad for a in arrays]
    return FamilyStats(
        family=cast('Literal["non_decaying", "contractive"]', family),
        rms_min=min(rms),
        rms_median=statistics.median(rms),
        rms_max=max(rms),
        peak_min=min(peak),
        peak_median=statistics.median(peak),
        peak_max=max(peak),
    )


def _check(name: str, passed: bool, detail: str) -> CheckResult:  # noqa: FBT001
    return CheckResult(name=name, passed=passed, detail=detail)


def _configuration_checks(
    result: AugmentationResult,
    second: AugmentationResult,
    samples: SampleSet,
    scenario: ScenarioConfig,
    config: AugmentationConfig,
) -> tuple[CheckResult, ...]:
    """Every per-configuration check, in the documented order."""
    t = samples.t
    taper = terminal_taper(t, result.dwell_start_s)
    untapered = taper == 1.0
    zero_from = result.dwell_start_s - TAPER_ZERO_MARGIN_S
    tail = t >= zero_from
    variants = [
        (episode, cast("EpisodeArrays", getattr(episode, family)))
        for episode in result.episodes
        for family in _FAMILIES
    ]

    deterministic = result.digests() == second.digests()
    max_peak = max(arrays.delta_peak_rad for _, arrays in variants)
    peak_bound = _PEAK_SIGMA_FACTOR * config.sigma_rad
    max_step = max(float(np.max(np.abs(np.diff(arrays.delta, axis=0)))) for _, arrays in variants)

    numerator = 0.0
    denominator = 0.0
    for episode in result.episodes:
        z = episode.non_decaying.delta[untapered]
        numerator += float(np.sum(z[:-1] * z[1:]))
        denominator += float(np.sum(z[:-1] * z[:-1]))
    correlation = numerator / denominator if denominator > 0 else float("nan")

    envelope_curve = contraction_envelope(scenario, samples.q, config.gamma)
    residual = max(
        float(np.max(np.abs(episode.contractive.delta - envelope_curve[:, None] * episode.non_decaying.delta)))
        for episode in result.episodes
    )

    collapse_ok = all(
        bool(np.all(arrays.delta[tail] == 0.0)) and np.array_equal(arrays.q[tail], samples.q[tail])
        for _, arrays in variants
    )

    digests = result.digests()
    delta_digests = [
        digests[f"episode-{e.episode:03d}/{family}/delta"] for e in result.episodes for family in _FAMILIES
    ]
    unique = len(set(delta_digests)) == 2 * len(result.episodes)
    accepted_attempts = {e.attempt for e in result.episodes}
    rejected_attempts = {r.attempt for r in result.rejections}
    accounting = (
        len(result.episodes) == config.n_synthetic
        and result.attempts_used <= config.attempt_budget
        and not (accepted_attempts & rejected_attempts)
    )

    return (
        _check("determinism", deterministic, "regeneration digests match" if deterministic else "digests differ"),
        _check("bounds", max_peak <= peak_bound, f"max peak {max_peak:.6g} rad vs bound {peak_bound:.6g} rad"),
        _check("smoothness", max_step < config.sigma_rad, f"max step {max_step:.6g} rad vs sigma {config.sigma_rad}"),
        _check(
            "correlation",
            abs(correlation - config.phi) <= _CORRELATION_TOLERANCE,
            f"lag-1 estimate {correlation:.4f} vs phi {config.phi}",
        ),
        _check("envelope", residual <= _ENVELOPE_TOLERANCE_RAD, f"max identity residual {residual:.3g} rad"),
        _check(
            "dwell-collapse",
            collapse_ok,
            f"delta exactly zero from {zero_from!r} s through the dwell" if collapse_ok else "nonzero tail sample",
        ),
        _check(
            "episode-separation",
            unique,
            "all per-episode delta digests unique" if unique else "duplicate episode digests",
        ),
        _check(
            "rejection-accounting",
            accounting,
            (
                f"{len(result.episodes)} accepted in {result.attempts_used} attempts; "
                f"{len({r.attempt for r in result.rejections})} attempt(s) rejected "
                f"({len(result.rejections)} family record(s))"
            ),
        ),
    )


def validate_augmentation(
    record: RecoveryDatasetRecord,
    samples: SampleSet,
    scenario: ScenarioConfig,
    configurations: Sequence[AugmentationConfig],
    *,
    provenance: ProvenanceRecord,
) -> AugmentationValidationReport:
    """Generate and check every configuration; failures are recorded, never raised."""
    derivatives = _derivative_config(record.preprocessing.derivative_method)
    task = record.crop.task
    reports: list[ConfigurationReport] = []
    first_result: AugmentationResult | None = None
    for config in configurations:
        try:
            result = generate_augmentation(samples.t, samples.q, task, scenario, config, derivatives=derivatives)
            second = generate_augmentation(samples.t, samples.q, task, scenario, config, derivatives=derivatives)
        except AugmentationError as exc:
            reports.append(
                ConfigurationReport(
                    config=config,
                    attempts_used=0,
                    rejected_attempts=0,
                    family_rejections=0,
                    rejections=(),
                    families=(),
                    checks=(_check("generation", passed=False, detail=str(exc)),),
                    digests_sha256=sha256_bytes(b""),
                )
            )
            continue
        if first_result is None:
            first_result = result
        checks = _configuration_checks(result, second, samples, scenario, config)
        reports.append(
            ConfigurationReport(
                config=config,
                attempts_used=result.attempts_used,
                rejected_attempts=len({rejection.attempt for rejection in result.rejections}),
                family_rejections=len(result.rejections),
                rejections=tuple(result.rejections),
                families=tuple(_family_stats(result, family) for family in _FAMILIES),
                checks=checks,
                digests_sha256=sha256_bytes(canonical_json(result.digests()).encode("utf-8")),
            )
        )

    global_checks: list[CheckResult] = []
    try:
        record.check_samples(samples)
        global_checks.append(
            _check("source-binding", passed=True, detail=f"payload matches {record.artifact.artifact_id}")
        )
    except ValueError as exc:
        global_checks.append(_check("source-binding", passed=False, detail=str(exc)))
    if configurations and first_result is not None:
        base = configurations[0]
        alt = dataclasses.replace(base, seed_bank=base.seed_bank + 1)
        derivative = _derivative_config(record.preprocessing.derivative_method)
        try:
            other = generate_augmentation(samples.t, samples.q, task, scenario, alt, derivatives=derivative)
            separated = other.digests() != first_result.digests()
            global_checks.append(
                _check(
                    "bank-separation",
                    separated,
                    f"banks {base.seed_bank} and {alt.seed_bank} " + ("differ" if separated else "collide"),
                )
            )
        except AugmentationError as exc:
            global_checks.append(_check("bank-separation", passed=False, detail=str(exc)))
    passed = all(c.passed for c in global_checks) and all(r.passed for r in reports)
    return AugmentationValidationReport(
        dataset=record.artifact.artifact_id,
        payload_sha256=record.artifact.payload.sha256,
        scenario_name=scenario.name,
        derivative_method=record.preprocessing.derivative_method,
        seed_namespace=SEED_NAMESPACE,
        taper_duration_s=TAPER_DURATION_S,
        taper_zero_margin_s=TAPER_ZERO_MARGIN_S,
        dwell_start_s=task.dwell[0],
        configurations=tuple(reports),
        checks=tuple(global_checks),
        passed=passed,
        provenance=provenance,
    )


def report_to_json(report: AugmentationValidationReport) -> str:
    """Canonical JSON of the report."""
    return canonical_json(to_mapping(report))


def report_to_markdown(report: AugmentationValidationReport) -> str:
    """Human-reviewable Markdown summary; every failing check appears with its detail."""
    lines = [
        "# Augmentation validation",
        "",
        f"- Dataset: `{report.dataset}` (payload `{report.payload_sha256[:12]}`)",
        f"- Scenario: {report.scenario_name}; derivative policy: {report.derivative_method}",
        f"- Seed namespace: `[{report.seed_namespace}, seed_bank, attempt]` (no evaluation seed consumed)",
        (
            f"- Locked taper: smoothstep over {report.taper_duration_s} s, exact zero from "
            f"{report.taper_zero_margin_s} s before dwell onset (dwell starts at {report.dwell_start_s} s)"
        ),
        f"- Outcome: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        "## Global checks",
        "",
        "| check | outcome | detail |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |" for c in report.checks)
    lines += [
        "",
        "## Configurations",
        "",
        "| N_aug | sigma (rad) | phi | gamma | attempts | rejected attempts | family rejections | "
        + " | ".join(_CHECK_NAMES)
        + " |",
        "| --- | --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in _CHECK_NAMES) + " |",
    ]
    for cfg in report.configurations:
        outcomes = {c.name: c for c in cfg.checks}
        cells = [("PASS" if outcomes[name].passed else "FAIL") if name in outcomes else "n/a" for name in _CHECK_NAMES]
        lines.append(
            f"| {cfg.config.n_synthetic} | {cfg.config.sigma_rad} | {cfg.config.phi} | {cfg.config.gamma} "
            f"| {cfg.attempts_used} | {cfg.rejected_attempts} | {cfg.family_rejections} | " + " | ".join(cells) + " |"
        )
    rejected = [cfg for cfg in report.configurations if cfg.rejections]
    if rejected:
        lines += [
            "",
            "## Rejected attempts",
            "",
            "Rejections are expected under the bounded resampling protocol; they are recorded, never hidden.",
            "",
        ]
        for cfg in rejected:
            label = f"({cfg.config.n_synthetic}, {cfg.config.sigma_rad}, {cfg.config.phi}, {cfg.config.gamma})"
            lines.extend(
                f"- {label} attempt {rejection.attempt}, {rejection.family}: {rejection.reason}"
                for rejection in cfg.rejections
            )
    failures = [
        (f"configuration ({cfg.config.n_synthetic}, {cfg.config.sigma_rad}, {cfg.config.phi}, {cfg.config.gamma})", c)
        for cfg in report.configurations
        for c in cfg.checks
        if not c.passed
    ] + [("global", c) for c in report.checks if not c.passed]
    if failures:
        lines += ["", "## Failures", ""]
        lines.extend(f"- **FAIL** {where}: {c.name} — {c.detail}" for where, c in failures)
    lines += [
        "",
        "## Realized amplitudes (per family, rad)",
        "",
        "| N_aug | sigma | phi | gamma | family | rms median | rms max | peak median | peak max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cfg in report.configurations:
        for family in cfg.families:
            lines.append(
                f"| {cfg.config.n_synthetic} | {cfg.config.sigma_rad} | {cfg.config.phi} | {cfg.config.gamma} "
                f"| {family.family} | {family.rms_median:.5f} | {family.rms_max:.5f} "
                f"| {family.peak_median:.5f} | {family.peak_max:.5f} |"
            )
    return "\n".join(lines) + "\n"


def plot_configuration(result: AugmentationResult, t: NDArray[np.float64], out_dir: Path, *, stem: str) -> list[Path]:
    """Write the four review panels of one configuration; returns the written paths in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    taper = terminal_taper(t, result.dwell_start_s)
    zero_from = result.dwell_start_s - TAPER_ZERO_MARGIN_S
    written: list[Path] = []
    for family in _FAMILIES:
        fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax = cast("Any", axis)  # matplotlib's Axes kwargs are untyped
        for episode in result.episodes:
            arrays = cast("EpisodeArrays", getattr(episode, family))
            ax.plot(t, arrays.delta[:, 0], linewidth=0.6, alpha=0.5)
        ax.axvline(zero_from - TAPER_DURATION_S, color="k", linestyle=":", label="taper window")
        ax.axvline(zero_from, color="k", linestyle="--", label="exact zero")
        ax.axvline(result.dwell_start_s, color="r", linestyle="--", label="dwell onset")
        twin = ax.twinx()
        twin.plot(t, taper, "k-", linewidth=1.0)
        twin.set_ylabel("taper")
        ax.set_xlabel("task time (s)")
        ax.set_ylabel("delta joint 1 (rad)")
        ax.set_title(f"{family} perturbations")
        ax.legend(loc="upper right", fontsize="small")
        path = out_dir / f"{stem}_{family}.png"
        cast("Any", fig).savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True, sharex=True)
    for joint, ax in enumerate(axes):
        for episode in result.episodes:
            ax.plot(t, episode.contractive.q[:, joint], linewidth=0.6, alpha=0.4)
        ax.plot(t, result.original.q[:, joint], "k-", linewidth=1.5, label="reference")
        ax.axvline(result.dwell_start_s, color="r", linestyle="--")
        ax.set_ylabel(f"q[{joint}] (rad)")
        ax.legend(loc="best", fontsize="small")
    axes[-1].set_xlabel("task time (s)")
    axes[0].set_title("contractive augmented positions vs reference")
    path = out_dir / f"{stem}_positions.png"
    cast("Any", fig).savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax = cast("Any", axis)  # matplotlib's Axes kwargs are untyped
    for family, marker in zip(_FAMILIES, ("o", "x"), strict=True):
        rms = [cast("EpisodeArrays", getattr(e, family)).delta_rms_rad for e in result.episodes]
        peak = [cast("EpisodeArrays", getattr(e, family)).delta_peak_rad for e in result.episodes]
        ax.scatter(rms, peak, marker=marker, label=family, alpha=0.7)
    ax.set_xlabel("episode delta RMS (rad)")
    ax.set_ylabel("episode delta peak (rad)")
    ax.set_title("realized per-episode amplitudes")
    ax.legend(loc="best", fontsize="small")
    path = out_dir / f"{stem}_statistics.png"
    cast("Any", fig).savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; writes the report set, then exits by outcome."""
    parser = argparse.ArgumentParser(description="Validate and visualize the recovery augmentation families.")
    parser.add_argument("--dataset", type=Path, required=True, help="Git-tracked recovery dataset record (TOML)")
    parser.add_argument("--scenario", type=Path, required=True, help="scenario TOML the dataset was derived under")
    parser.add_argument("--report", type=Path, required=True, help="JSON report to write (must not exist)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown to write (must not exist)")
    parser.add_argument("--plots", type=Path, default=None, help="optional directory for the anchor plot set")
    parser.add_argument("--seed-bank", type=int, default=1, help="shared augmentation seed bank (default 1)")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)
    for target in (args.report, args.markdown, args.plots):
        if target is not None and Path(target).exists():
            msg = f"{target} already exists; reports are immutable (choose a new versioned name)"
            raise FileExistsError(msg)

    store = open_storage()
    record = load_record(Path(args.dataset), RecoveryDatasetRecord)
    scenario = load_scenario(Path(args.scenario))
    record.check_scenario(Path(args.scenario))
    payload = verify_payload(store, record.artifact)
    samples = load_samples(payload)
    grid = approved_grid(seed_bank=args.seed_bank)
    command = command_line(
        "arm_rc_ctrl.experiments.augmentation_validation", argv if argv is not None else sys.argv[1:]
    )
    resolved = {
        "dataset": record.artifact.artifact_id,
        "scenario": to_mapping(scenario),
        "grid": [to_mapping(config) for config in grid],
        "seed_bank": args.seed_bank,
        "command": command,
    }
    reference = ArtifactReference(
        record.artifact.payload.uri, record.artifact.payload.sha256, record.artifact.payload.size
    )
    provenance = collect_provenance(resolved, seeds={}, artifacts=[reference], exploratory=args.exploratory)
    require_clean_for_confirmatory(provenance)

    report = validate_augmentation(record, samples, scenario, grid, provenance=provenance)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report_to_json(report) + "\n", encoding="utf-8")
    if args.markdown is not None:
        Path(args.markdown).write_text(report_to_markdown(report), encoding="utf-8")
    if args.plots is not None:
        anchor = dataclasses.replace(ANCHOR, seed_bank=args.seed_bank)
        derivatives = _derivative_config(record.preprocessing.derivative_method)
        result = generate_augmentation(
            samples.t, samples.q, record.crop.task, scenario, anchor, derivatives=derivatives
        )
        plot_configuration(result, samples.t, Path(args.plots), stem=Path(args.report).stem)
    failed = [
        f"{cfg.config.n_synthetic}/{cfg.config.sigma_rad}/{cfg.config.phi}/{cfg.config.gamma}:{c.name}"
        for cfg in report.configurations
        for c in cfg.checks
        if not c.passed
    ] + [f"global:{c.name}" for c in report.checks if not c.passed]
    print(
        json.dumps(
            {"passed": report.passed, "configurations": len(report.configurations), "failed_checks": failed}, indent=2
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
