# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-007: the browser overview of the repetition pilot stays bound to its generated report and assets (C4)."""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

import pytest

from arm_rc_ctrl.experiments.repetition_accounting import load_accounting
from arm_rc_ctrl.experiments.repetition_report import ANIMATION_DIR, PLOT_DIR, load_report, render_report_markdown
from arm_rc_ctrl.experiments.repetition_timing import load_timing
from arm_rc_ctrl.provenance import sha256_file
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

DOCS = repository_root() / "docs" / "experiments" / "task_1a_repeated_demonstration"
OVERVIEW = DOCS / "overview.html"
REPORT = DOCS / "repetition_report_v1.json"


class Overview(HTMLParser):
    """Collect the visible evidence fields, identifiers, and every local resource the page references."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.images: list[str] = []
        self.fields: dict[str, str] = {}
        self.active_field: str | None = None
        self.feed(OVERVIEW.read_text(encoding="utf-8"))
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track identifiers, evidence fields, and resources."""
        values = dict(attrs)
        identity = values.get("id")
        if identity is not None:
            assert identity not in self.ids, identity
            self.ids.add(identity)
        field = values.get("data-evidence")
        if field is not None:
            assert field not in self.fields, field
            self.active_field = field
            self.fields[field] = ""
        for attr in ("href", "src"):
            value = values.get(attr)
            if value is not None:
                self.links.append(value)
        if tag == "img":
            assert values.get("alt"), "every figure needs a description"
            assert values.get("src")
            self.images.append(str(values["src"]))

    def handle_data(self, data: str) -> None:
        """Evidence fields hold the text the reader sees."""
        if self.active_field is not None:
            self.fields[self.active_field] += data

    def handle_endtag(self, tag: str) -> None:  # noqa: ARG002 - HTMLParser callback signature
        """Evidence fields contain plain text only."""
        self.active_field = None


def _g(value: float | None) -> str:
    """The report tables' number format (four significant digits, ``n/a`` for an absent value)."""
    return "n/a" if value is None else f"{value:.4g}"


def test_overview_binds_the_report_counts_and_caveat() -> None:
    """Counts, the C11 caveat, the abort limits, and the cost figures on the page are the report's."""
    page = Overview()
    report = load_report(REPORT)
    assert page.fields["feasible"] == str(report.n_feasible)
    assert page.fields["configurations"] == str(len(report.outcomes))
    assert page.fields["rc_gate_failure"] == str(report.n_rc_gate_failure)
    assert page.fields["crossed_historical"] == str(report.n_crossed_historical)
    assert page.fields["velocity_abort"] == f"{report.velocity_abort[0]:g}"
    assert page.fields["historical_limit"] == f"{report.historical_velocity_limit[0]:g}"
    assert page.fields["peak_rss_mib"] == f"{report.peak_rss_bytes / 2**20:.1f}"
    exception = [e for e in report.equivalence if e.accepted_exception]
    assert len(exception) == 1
    assert page.fields["exception_max_abs"] == f"{exception[0].max_abs:.2e}"
    assert page.fields["equivalence_passed"] == str(sum(1 for e in report.equivalence if e.passed))
    for entry in ("feasible-best", "feasible-middle", "feasible-worst"):
        for arm in ("absolute/S", "absolute/R/K65", "absolute/R-scaled/K65"):
            row = next(o for o in report.outcomes if o.panel_label == entry and o.arm == arm)
            assert page.fields[f"{entry}:{arm}:status"] == row.status
    assert page.fields["representative_rule"] == report.representative_rule
    assert page.fields["c11"] == report.c11_caveat
    assert render_report_markdown(report) == (DOCS / "repetition_report_v1.md").read_text(encoding="utf-8")


def test_overview_is_offline_and_every_link_resolves() -> None:
    """No remote resource; every relative link and fragment resolves to a committed file or an id on the page."""
    page = Overview()
    for link in page.links:
        parts = urlsplit(link)
        assert not parts.scheme, link
        assert not parts.netloc, link
        if parts.path:
            target = DOCS / unquote(parts.path)
            assert target.is_file(), link
            assert target.stat().st_size > 0, link
        if parts.fragment:
            assert parts.fragment in page.ids, link
    assert page.images
    assert all(image.startswith((f"{PLOT_DIR}/", f"{ANIMATION_DIR}/")) for image in page.images)


def test_overview_shows_the_report_assets_and_representatives() -> None:
    """Every plot and animation the report names is committed; the shown animations are the rule's runs."""
    report = load_report(REPORT)
    page = Overview()
    for plot in report.plots:
        assert (DOCS / PLOT_DIR / plot).is_file()
    for animation in report.animations:
        assert (DOCS / ANIMATION_DIR / animation).is_file()
        assert f"{ANIMATION_DIR}/{animation}" in page.links
    shown = [r for r in report.representatives if r.rc_animation is not None]
    assert [r.arm for r in shown] == ["absolute/S", "absolute/R/K65"]
    for rep in shown:
        assert rep.rc_run is not None
        assert page.fields[f"animation:{rep.arm}:run"] == rep.rc_run
    for name, digest in report.sources.items():
        assert sha256_file(DOCS / name) == digest


def test_overview_binds_the_derived_counts_comparisons_and_costs() -> None:
    """Speed, augmentation, residual, paired-comparison, cost, and execution figures on the page are the records'."""
    page = Overview()
    report = load_report(REPORT)
    assert page.fields["crossed_historical_repeat"] == str(report.n_crossed_historical)
    assert page.fields["aborted_models"] == str(sum(1 for s in report.speeds if s.aborts > 0))
    feasible_crossed = sum(1 for o in report.outcomes if o.status == "feasible" and o.crossed_historical)
    assert page.fields["feasible_crossed"] == str(feasible_crossed)
    augmented = [o for o in report.outcomes if "/A-" in o.arm]
    assert len(augmented) == 36
    assert page.fields["augmented_feasible"] == str(sum(1 for o in augmented if o.status == "feasible"))
    residual = [o for o in report.outcomes if o.formulation == "residual"]
    assert len(residual) == 42
    assert page.fields["residual_first_pair_failures"] == str(sum(1 for o in residual if o.n_completed == 0))
    gap, jump = "early_gap_integral", "activation_jump_rad"
    for entry in ("feasible-best", "feasible-middle", "feasible-worst", "failure-joint-velocity"):
        row = next(
            p for p in report.paired if p.panel_label == entry and p.comparison == "absolute/R/K65 vs absolute/S"
        )
        prefix = f"paired:{entry}:R"
        assert page.fields[f"{prefix}:shared"] == str(row.shared_pairs)
        assert page.fields[f"{prefix}:gap_left"] == _g(row.left_median[gap])
        assert page.fields[f"{prefix}:gap_right"] == _g(row.right_median[gap])
        assert page.fields[f"{prefix}:gap_diff"] == _g(row.signed_difference_median[gap])
        assert page.fields[f"{prefix}:jump_left"] == _g(row.left_median[jump])
        assert page.fields[f"{prefix}:jump_right"] == _g(row.right_median[jump])
        assert page.fields[f"{prefix}:jump_diff"] == _g(row.signed_difference_median[jump])
        assert page.fields[f"{prefix}:worst_left"] == _g(row.left_worst_cell)
        assert page.fields[f"{prefix}:worst_right"] == _g(row.right_worst_cell)
        scaled = next(
            p for p in report.paired if p.panel_label == entry and p.comparison == "absolute/R-scaled/K65 vs absolute/S"
        )
        assert not scaled.verdict_changed
        assert page.fields[f"paired:{entry}:R-scaled:gap_diff"] == _g(scaled.signed_difference_median[gap])
    shown = [c for c in report.costs if c.count in (1, 65)]
    assert len(shown) == 10
    for cost in shown:
        assert page.fields[f"cost:{cost.arm}:fit"] == f"{cost.fit_seconds_mean:.2f}"
        assert page.fields[f"cost:{cost.arm}:eval"] == f"{cost.evaluation_seconds_mean:.1f}"
    timing = load_timing(DOCS / "timing_smoke_check_v1.json")
    assert page.fields["smoke_runs"] == str(timing.runs_this_invocation)
    assert page.fields["smoke_wall_s"] == f"{timing.wall_seconds:.0f}"
    assert timing.peak_rss_bytes == report.peak_rss_bytes
    accounting = load_accounting(DOCS / "pilot_execution_v1.json")
    assert page.fields["panel_rc_runs"] == str(accounting.n_rc_runs)
    assert page.fields["panel_replay_runs"] == str(accounting.n_replay_runs)
    assert page.fields["panel_unexecuted"] == str(accounting.n_unexecuted_pairs)
    assert accounting.n_crossed_historical == report.n_crossed_historical
    replay = {r.replay_run for r in report.representatives if r.replay_animation is not None}
    assert len(replay) == 1
    assert page.fields["animation:replay:run"] == replay.pop()
