# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""DOC-005: the browser overview remains linked to the frozen development evidence."""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

from arm_rc_ctrl.experiments.recovery_ablation import load_ablation
from arm_rc_ctrl.experiments.recovery_freeze import load_freeze
from arm_rc_ctrl.experiments.recovery_representative import load_representatives
from arm_rc_ctrl.repo import repository_root

DOCS = repository_root() / "docs" / "experiments" / "task_1a_state_conditioned_recovery"


class Overview(HTMLParser):
    """Collect visible evidence fields and resource references, including motion templates."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.images: list[str] = []
        self.fields: dict[str, str] = {}
        self.active_field: str | None = None
        self.feed((DOCS / "overview.html").read_text(encoding="utf-8"))
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track local resources and the plain-text numeric evidence fields."""
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
            assert values.get("alt"), "Every scientific image needs a description"
            assert values.get("src")
            self.images.append(str(values["src"]))

    def handle_data(self, data: str) -> None:
        """Capture the text a reader sees, rather than a hidden numeric duplicate."""
        if self.active_field is not None:
            self.fields[self.active_field] += data

    def handle_endtag(self, tag: str) -> None:  # noqa: ARG002 - HTMLParser callback signature
        """Evidence fields contain plain text only."""
        self.active_field = None


def test_overview_counts_and_trial_17_cells_match_evidence() -> None:
    """A favorable median cannot obscure the failing per-scenario consistency gate."""
    page = Overview()
    ablation = load_ablation(DOCS / "development_ablation_v2.json")
    freeze = load_freeze(DOCS / "model_freeze_v2.json")
    assert page.fields["feasible"] == str(freeze.n_candidates)
    assert page.fields["eligible"] == str(freeze.n_eligible)
    for arm in ablation.arms:
        assert page.fields[f"{arm.formulation}:feasible"] == str(arm.n_feasible)
        assert page.fields[f"{arm.formulation}:budget"] == str(arm.budget)
    trial = next(candidate for candidate in ablation.candidates if candidate.number == 17)
    for key, cell in trial.cells.items():
        assert page.fields[f"{key}:gap"] == f"{cell.gap_median:.4f}"
        assert page.fields[f"{key}:jump"] == f"{cell.jump_median:.4f}"
        assert page.fields[f"{key}:count"] == f"{cell.improving_both} / {cell.n}"
        assert page.fields[f"{key}:passes"] == ("Pass" if cell.passes else "Fail")
    maximum = max(candidate.cells["posture_small:pd_v2"].improving_both for candidate in ablation.candidates)
    assert page.fields["small_pd_maximum"] == str(maximum)


def test_overview_is_offline_and_all_links_resolve() -> None:
    """Opening the HTML directly needs no CDN, server, or external evidence payload."""
    page = Overview()
    assert page.images
    for link in page.links:
        url = urlsplit(link)
        assert not url.scheme, link
        assert not url.netloc, link
        if url.path:
            target = DOCS / unquote(url.path)
            assert target.is_file(), link
            assert target.stat().st_size > 0, link
        elif url.fragment:
            assert url.fragment in page.ids, link


def test_overview_motion_pairs_are_the_recorded_pd_representatives() -> None:
    """All four motion examples cite the exact run identity beside each existing GIF."""
    page = Overview()
    record = load_representatives(DOCS / "recovery_representative_v1.json")
    for pair in record.pairs:
        if pair.tracker != "pd_v2":
            continue
        for arm, run in (("rc", pair.rc_run), ("replay", pair.replay_run)):
            assert page.fields[f"{pair.kind}:{arm}:run"] == run
            assert f"animations/{pair.kind}_{arm}_pd.gif" in page.images
