# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Store-free bindings for the expert HTML presentation, not new scientific evidence."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

from arm_rc_ctrl.experiments.manual_expert_report import CASES, configuration_table, summary_table, verify_report
from arm_rc_ctrl.experiments.manual_figures import load_figure_inputs
from arm_rc_ctrl.experiments.manual_report import DOCUMENT_LOCATION, load_report_data
from arm_rc_ctrl.repo import repository_root


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if tag in {"a", "img", "script", "link"} and name in {"src", "href"} and value:
                self.targets.append(value)
            if name == "id" and value:
                self.ids.add(value)


def test_expert_report_binds_its_sources_inputs_assets_and_generated_tables() -> None:
    """Presentation bytes and the two generated tables remain tied to their evidence."""
    docs = repository_root() / DOCUMENT_LOCATION
    report = docs / "expert_report"
    verify_report(report)
    data = load_report_data(docs)
    page = (report / "index.html").read_text(encoding="utf-8")
    assert summary_table(data) in page
    assert configuration_table(data) in page
    assert page.count('class="player"') == len(CASES)
    assert page.count('data-result="') == len(data.study.configurations) * 2


def test_every_playback_binds_the_frozen_cases_and_all_five_run_payloads() -> None:
    """Every displayed run retains the frozen figure input's identity and verdict."""
    docs = repository_root() / DOCUMENT_LOCATION
    figures = load_figure_inputs(docs / "results/figure_inputs_v1.json")
    cases = json.loads((docs / "expert_report/cases.json").read_text())
    assert [c["case_id"] for c in cases] == [s.case_id for s in CASES]
    for case in cases:
        bound = next(c for c in figures.cases if c.case_id == case["case_id"])
        assert [r["role"] for r in case["runs"]] == [r.role for r in bound.runs]
        for run, original in zip(case["runs"], bound.runs, strict=True):
            assert run["sha256"] == original.run.sha256
            assert run["arrays_sha256"] == original.run.arrays_sha256
            assert run["success"] == original.success
            assert run["uri"] == original.run.uri


def test_expert_report_local_links_and_navigation_resolve() -> None:
    """Offline assets, repository evidence links and within-page navigation resolve."""
    report = repository_root() / DOCUMENT_LOCATION / "expert_report"
    parser = _Links()
    parser.feed((report / "index.html").read_text(encoding="utf-8"))
    for target in parser.targets:
        parts = urlsplit(target)
        if parts.scheme or parts.netloc:
            continue
        if parts.path:
            assert (report / unquote(parts.path)).is_file(), target
        elif parts.fragment:
            assert parts.fragment in parser.ids, target
