#!/usr/bin/env python3
# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""One-off migration of the M3 task 1-a records to machine-independent identities (M3 CI finding, 2026-08-31).

Before this migration the ESN protocol digest and the study/suite-level
provenance configurations were computed from the absolute config paths the
loader resolves, so they embedded this machine's checkout path and could not
be re-derived elsewhere. The migration rewrites, in the committed JSON
reports only, (1) ``protocol_sha256`` (and the study summary's identity
attribute) to the canonical digest of the committed protocol file, (2)
``provenance.config_json``/``config_sha256`` with every absolute repository
path made repository-relative, and (3) the derived Markdown twins. No measurement, run ID, pointer, scenario,
or trial value changes. The external Optuna databases keep their original
identity attribute (documented in ``docs/TASKS.md``).

Run once from the repository root::

    python scripts/migrate_m3_records.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace

from arm_rc_ctrl.experiments.esn_search import load_esn_search, protocol_digest
from arm_rc_ctrl.experiments.esn_stability import load_stability, stability_to_json
from arm_rc_ctrl.experiments.esn_stability import render_markdown as render_stability_markdown
from arm_rc_ctrl.experiments.esn_study import load_report, report_to_json
from arm_rc_ctrl.experiments.esn_study import render_markdown as render_study_markdown
from arm_rc_ctrl.experiments.robustness import load_suite, suite_to_json
from arm_rc_ctrl.provenance import ProvenanceRecord, canonical_json, portable_config, sha256_bytes
from arm_rc_ctrl.repo import repository_root

REPO = repository_root()
DOCS = REPO / "docs" / "experiments" / "task_1a"
PROTOCOLS = {
    "esn-search-1a": REPO / "configs" / "studies" / "esn_search_1a.toml",
    "esn-search-1a-v2": REPO / "configs" / "studies" / "esn_search_1a_v2.toml",
}
FROZEN_CONFIGS = (
    REPO / "configs" / "models" / "esn_task_1a_v3.toml",
    REPO / "configs" / "evaluations" / "task_1a_nominal_v3.toml",
    REPO / "configs" / "models" / "esn_task_1a_v4.toml",
    REPO / "configs" / "evaluations" / "task_1a_nominal_v4.toml",
)


def portable_provenance(provenance: ProvenanceRecord) -> ProvenanceRecord:
    """The provenance with a portable configuration (JSON and digest recomputed)."""
    config = portable_config(json.loads(provenance.config_json))
    text = canonical_json(config)
    return replace(provenance, config_json=text, config_sha256=sha256_bytes(text.encode("utf-8")))


def migrate_studies(digests: dict[str, str]) -> list[str]:
    """Rewrite the ESN study reports' digests and provenance."""
    changed: list[str] = []
    for file in sorted(DOCS.glob("esn_search*.json")):
        report = load_report(file)
        digest = digests[report.protocol]
        identity = {**report.summary.identity, "armrc.protocol_sha256": digest}
        migrated = replace(
            report,
            protocol_sha256=digest,
            summary=replace(report.summary, identity=identity),
            provenance=portable_provenance(report.provenance),
        )
        text = report_to_json(migrated) + "\n"
        if text != file.read_text(encoding="utf-8"):
            file.write_text(text, encoding="utf-8")
            changed.append(file.name)
    return changed


def migrate_panels(digests: dict[str, str]) -> list[str]:
    """Rewrite the stability panels' digests and provenance."""
    changed: list[str] = []
    for file in sorted(DOCS.glob("esn_stability*.json")):
        panel = load_stability(file)
        migrated = replace(
            panel, protocol_sha256=digests[panel.protocol], provenance=portable_provenance(panel.provenance)
        )
        text = stability_to_json(migrated) + "\n"
        if text != file.read_text(encoding="utf-8"):
            file.write_text(text, encoding="utf-8")
            changed.append(file.name)
    return changed


def migrate_suites() -> list[str]:
    """Rewrite the robustness suites' provenance."""
    changed: list[str] = []
    for file in sorted(DOCS.glob("robustness_*.json")):
        suite = load_suite(file)
        migrated = replace(suite, provenance=portable_provenance(suite.provenance))
        text = suite_to_json(migrated) + "\n"
        if text != file.read_text(encoding="utf-8"):
            file.write_text(text, encoding="utf-8")
            changed.append(file.name)
    return changed


def rerender_markdown() -> list[str]:
    """Re-render the Markdown twins of the migrated reports (derived files; the locks compare them)."""
    changed: list[str] = []
    for file in sorted(DOCS.glob("esn_search*.json")):
        text = render_study_markdown(load_report(file))
        target = file.with_suffix(".md")
        if text != target.read_text(encoding="utf-8"):
            target.write_text(text, encoding="utf-8")
            changed.append(target.name)
    for file in sorted(DOCS.glob("esn_stability*.json")):
        text = render_stability_markdown(load_stability(file))
        target = file.with_suffix(".md")
        if text != target.read_text(encoding="utf-8"):
            target.write_text(text, encoding="utf-8")
            changed.append(target.name)
    return changed


def migrate_frozen_config_comments(old_to_new: dict[str, str]) -> list[str]:
    """Update the digest prefixes the frozen configuration headers cite (comments only)."""
    changed: list[str] = []
    for file in FROZEN_CONFIGS:
        text = file.read_text(encoding="utf-8")
        updated = text
        for old, new in old_to_new.items():
            updated = re.sub(rf"(?<=digest ){re.escape(old[:12])}\b", new[:12], updated)
        if updated != text:
            file.write_text(updated, encoding="utf-8")
            changed.append(file.name)
    return changed


def main() -> int:
    """Migrate every affected record once; print what changed."""
    digests = {name: protocol_digest(load_esn_search(path)) for name, path in PROTOCOLS.items()}
    previous = {load_report(f).protocol: load_report(f).protocol_sha256 for f in sorted(DOCS.glob("esn_search*.json"))}
    old_to_new = {previous[name]: digests[name] for name in previous if previous[name] != digests[name]}
    changed = migrate_studies(digests) + migrate_panels(digests) + migrate_suites()
    changed += rerender_markdown() + migrate_frozen_config_comments(old_to_new)
    for name, digest in digests.items():
        before = previous.get(name, "-")
        print(f"{name}: {before[:12]} -> {digest[:12]}")
    print("changed:", ", ".join(changed) or "nothing")
    leftovers = [f.name for f in DOCS.glob("*.json") if str(REPO) in f.read_text(encoding="utf-8")]
    if leftovers:
        print("records still naming the checkout path:", leftovers)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
