# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

r"""Two-tracker paired suite: generator effects versus tracker effects (M2-017).

Given the paired RC/replay reports under PD and under computed torque — four
runs of one reference with one recipe — every metric decomposes into the
generator effect within a tracker (``rc - replay``) and the tracker effect
within an arm (``computed_torque - pd``). All four reports share the reference
windows and metric definitions, so the decomposition is exact.

Usage::

    python -m arm_rc_ctrl.experiments.paired_suite --pd <paired_pd.json> --computed-torque <paired_ct.json> \\
        --report <json> [--markdown <md>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.experiments.paired import PairedReport, load_paired_report
from arm_rc_ctrl.metrics.report import RunReport

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["EffectDecomposition", "PairedSuite", "load_paired_suite", "main", "suite_to_markdown"]

SUITE_SCHEMA_VERSION: Final = 1


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


@dataclass(frozen=True)
class EffectDecomposition:
    """One metric across the four runs and its generator/tracker effects."""

    name: str
    unit: str
    replay_pd: float | None
    rc_pd: float | None
    replay_ct: float | None
    rc_ct: float | None

    @property
    def generator_effect_pd(self) -> float | None:
        """``rc - replay`` under PD."""
        return _diff(self.rc_pd, self.replay_pd)

    @property
    def generator_effect_ct(self) -> float | None:
        """``rc - replay`` under computed torque."""
        return _diff(self.rc_ct, self.replay_ct)

    @property
    def tracker_effect_replay(self) -> float | None:
        """``computed_torque - pd`` for direct replay."""
        return _diff(self.replay_ct, self.replay_pd)

    @property
    def tracker_effect_rc(self) -> float | None:
        """``computed_torque - pd`` for the RC generator."""
        return _diff(self.rc_ct, self.rc_pd)


@dataclass(frozen=True)
class PairedSuite:
    """The PD and computed-torque pairs of one reference and recipe with the effect decomposition."""

    scenario: str
    reference_artifact: str
    recipe: str
    pd: PairedReport
    computed_torque: PairedReport
    effects: tuple[EffectDecomposition, ...] = ()
    schema_version: int = field(default=SUITE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Both pairs describe the same reference and recipe with identical windows and metric sets."""
        if self.schema_version != SUITE_SCHEMA_VERSION:
            msg = f"unsupported paired suite schema version {self.schema_version}"
            raise ValueError(msg)
        pd, ct = self.pd, self.computed_torque
        if pd.tracker != "pd" or ct.tracker != "computed_torque":
            msg = f"the suite needs a pd pair and a computed_torque pair, got {pd.tracker!r} and {ct.tracker!r}"
            raise ValueError(msg)
        for name, value_pd, value_ct in (
            ("scenario", pd.scenario, ct.scenario),
            ("reference_artifact", pd.reference_artifact, ct.reference_artifact),
            ("recipe", pd.recipe, ct.recipe),
        ):
            if value_pd != value_ct or value_pd != getattr(self, name):
                own = getattr(self, name)
                msg = f"{name} differs between the pairs and the suite: {value_pd!r}, {value_ct!r}, {own!r}"
                raise ValueError(msg)
        if pd.rc.windows != ct.rc.windows:
            msg = f"metric windows differ between trackers: {pd.rc.windows} vs {ct.rc.windows}"
            raise ValueError(msg)
        pd_names = [m.name for m in pd.metrics]
        if pd_names != [m.name for m in ct.metrics]:
            msg = "the pairs compare different metric sets"
            raise ValueError(msg)
        effects = tuple(
            EffectDecomposition(a.name, a.unit, a.replay, a.rc, b.replay, b.rc)
            for a, b in zip(pd.metrics, ct.metrics, strict=True)
        )
        if self.effects and self.effects != effects:
            msg = "stored effect decompositions do not match the paired reports they were derived from"
            raise ValueError(msg)
        object.__setattr__(self, "effects", effects)


def load_paired_suite(path: Path) -> PairedSuite:
    """Strictly rebuild a suite from JSON."""
    return from_mapping(json.loads(path.read_text(encoding="utf-8")), PairedSuite)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}" if math.isfinite(value) else str(value)


def _row(cells: Sequence[object]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _run_line(label: str, report: RunReport) -> str:
    return _row([label, report.run_id, report.termination_kind, report.success])


def suite_to_markdown(suite: PairedSuite) -> str:
    """A Markdown table of the four runs per metric with the generator and tracker effects."""
    columns = [
        "Metric",
        "Unit",
        "replay+pd",
        "rc+pd",
        "replay+ct",
        "rc+ct",
        "gen. effect (pd)",
        "gen. effect (ct)",
        "tracker effect (replay)",
        "tracker effect (rc)",
    ]
    header = [
        "# Paired suite: generator effect vs tracker effect",
        "",
        f"Scenario `{suite.scenario}`, reference `{suite.reference_artifact}`, recipe `{suite.recipe}`.",
        "Generator effect = RC - replay within one tracker; tracker effect = computed torque - PD within one arm.",
        "",
        _row(["Run", "Run ID", "Termination", "Success"]),
        "|---|---|---|---|",
        _run_line("replay+pd", suite.pd.replay),
        _run_line("rc+pd", suite.pd.rc),
        _run_line("replay+computed_torque", suite.computed_torque.replay),
        _run_line("rc+computed_torque", suite.computed_torque.rc),
        "",
        _row(columns),
        "|" + "---|" * len(columns),
    ]
    rows = [
        _row(
            [
                e.name,
                e.unit or "-",
                _fmt(e.replay_pd),
                _fmt(e.rc_pd),
                _fmt(e.replay_ct),
                _fmt(e.rc_ct),
                _fmt(e.generator_effect_pd),
                _fmt(e.generator_effect_ct),
                _fmt(e.tracker_effect_replay),
                _fmt(e.tracker_effect_rc),
            ]
        )
        for e in suite.effects
    ]
    return "\n".join([*header, *rows, ""])


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Combine the PD and computed-torque paired reports into one suite.")
    parser.add_argument("--pd", type=Path, required=True, help="paired report JSON under PD")
    parser.add_argument("--computed-torque", type=Path, required=True, help="paired report JSON under computed torque")
    parser.add_argument("--report", type=Path, required=True, help="suite JSON to write (must not exist)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional Markdown table to write (must not exist)")
    args = parser.parse_args(argv)
    for target in (args.report, args.markdown):
        if target is not None and Path(target).exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    pd = load_paired_report(Path(args.pd))
    ct = load_paired_report(Path(args.computed_torque))
    suite = PairedSuite(pd.scenario, pd.reference_artifact, pd.recipe, pd, ct)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_mapping(suite), indent=2, sort_keys=True, allow_nan=False)
    Path(args.report).write_text(text + "\n", encoding="utf-8")
    if args.markdown is not None:
        Path(args.markdown).write_text(suite_to_markdown(suite), encoding="utf-8")
    rmse = next(e for e in suite.effects if e.name == "joint_rmse")
    print(
        json.dumps(
            {
                "joint_rmse": {
                    "replay_pd": rmse.replay_pd,
                    "rc_pd": rmse.rc_pd,
                    "replay_ct": rmse.replay_ct,
                    "rc_ct": rmse.rc_ct,
                },
                "generator_effect_pd": rmse.generator_effect_pd,
                "generator_effect_ct": rmse.generator_effect_ct,
                "tracker_effect_replay": rmse.tracker_effect_replay,
                "tracker_effect_rc": rmse.tracker_effect_rc,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
