# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3MAN-009: the frozen result schema describes exactly the evidence the sweep writes.

Plan section 7.1 requires the machine-readable evidence to ship with documented
field definitions and units. A document that can drift from the code is worse
than none, so the structural half is derived from the dataclasses themselves and
this lock fails if the committed document and the code disagree about a single
record, field, or declared type.
"""

from __future__ import annotations

import pytest

from arm_rc_ctrl.experiments.manual_schema import (
    RESULT_RECORDS,
    RESULT_SCHEMA_VERSION,
    describe_records,
    load_schema,
    render_schema_markdown,
)
from arm_rc_ctrl.repo import repository_root

pytestmark = pytest.mark.regression

REPO_ROOT = repository_root()
DOCS = REPO_ROOT / "docs" / "experiments" / "task_1a_manual_demonstration"
SCHEMA = DOCS / "result_schema_v1.json"
MARKDOWN = DOCS / "result_schema_v1.md"


def test_the_frozen_schema_matches_the_records_the_sweep_writes() -> None:
    """Every record, field, and declared type in the document is the one the code defines."""
    frozen = load_schema(SCHEMA)
    derived = describe_records()
    assert [record.name for record in frozen.records] == [record.name for record in derived]
    for committed, actual in zip(frozen.records, derived, strict=True):
        assert [f.name for f in committed.fields] == [f.name for f in actual.fields], committed.name
        assert [f.type for f in committed.fields] == [f.type for f in actual.fields], committed.name
        assert [f.optional for f in committed.fields] == [f.optional for f in actual.fields], committed.name


def test_every_frozen_field_carries_a_definition() -> None:
    """A field nobody defined is a column a reader has to guess at; the freeze refuses to ship one."""
    frozen = load_schema(SCHEMA)
    undefined = [
        f"{record.name}.{field.name}"
        for record in frozen.records
        for field in record.fields
        if not field.definition.strip()
    ]
    assert undefined == []


def test_an_optional_field_says_what_its_absence_means() -> None:
    """Nulls survive serialization, so each one has to mean something stated rather than inferred."""
    frozen = load_schema(SCHEMA)
    unexplained = [
        f"{record.name}.{field.name}"
        for record in frozen.records
        for field in record.fields
        if field.optional and not field.absent.strip()
    ]
    assert unexplained == []


def test_the_frozen_scope_is_the_evaluation_evidence() -> None:
    """The result schema covers what the sweep writes; the timing report is its own artifact."""
    frozen = load_schema(SCHEMA)
    assert [record.name for record in frozen.records] == list(RESULT_RECORDS)
    assert "ManualTimingReport" not in {record.name for record in frozen.records}


def test_the_markdown_renders_from_the_committed_schema() -> None:
    """The rendering is generated from the record, never hand-edited beside it."""
    frozen = load_schema(SCHEMA)
    assert render_schema_markdown(frozen) == MARKDOWN.read_text(encoding="utf-8")


def test_a_numeric_field_declares_its_unit_and_what_it_aggregates_over() -> None:
    """A number without a unit or a denominator cannot be read back correctly by anyone but its author."""
    frozen = load_schema(SCHEMA)
    numeric = [
        (record.name, field)
        for record in frozen.records
        for field in record.fields
        if field.type.split(" |")[0] in {"int", "float"}
    ]
    assert numeric, "the evidence carries numbers; this check would be vacuous otherwise"
    missing_unit = [f"{name}.{f.name}" for name, f in numeric if not f.unit.strip()]
    missing_scope = [f"{name}.{f.name}" for name, f in numeric if not f.scope.strip()]
    assert missing_unit == []
    assert missing_scope == []


def test_the_frozen_document_states_the_version_it_belongs_to() -> None:
    """A changed field is a version decision, not a silent refresh: the artifact names its own version."""
    frozen = load_schema(SCHEMA)
    assert frozen.schema_version == RESULT_SCHEMA_VERSION
    assert SCHEMA.name == f"result_schema_v{RESULT_SCHEMA_VERSION}.json"
    assert MARKDOWN.name == f"result_schema_v{RESULT_SCHEMA_VERSION}.md"


def test_the_unexecuted_count_is_defined_and_currently_zero_by_construction() -> None:
    """Section 7.1 wants not-executed reported; the sweep cannot produce it, and the schema says so."""
    frozen = load_schema(SCHEMA)
    evidence = next(record for record in frozen.records if record.name == "ManualModelEvidence")
    unexecuted = next(field for field in evidence.fields if field.name == "n_unexecuted")
    assert "zero" in unexecuted.definition.lower()
    assert unexecuted.scope.strip()
