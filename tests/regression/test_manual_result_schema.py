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
    RECORDS_BY_VERSION,
    RESULT_SCHEMA_VERSION,
    SUPPORTED_RESULT_SCHEMAS,
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
SCHEMA_V2 = DOCS / "result_schema_v2.json"
MARKDOWN_V2 = DOCS / "result_schema_v2.md"
FROZEN = ((1, SCHEMA, MARKDOWN), (2, SCHEMA_V2, MARKDOWN_V2))
"""Every frozen version. A new one is added here; an existing one is never edited."""


@pytest.mark.parametrize(("version", "schema", "markdown"), FROZEN)
def test_the_frozen_schema_matches_the_records_the_sweep_writes(version: int, schema: object, markdown: object) -> None:
    """Every record, field, and declared type in the document is the one the code defines."""
    del markdown
    frozen = load_schema(schema)  # type: ignore[arg-type]
    derived = describe_records(version)
    assert [record.name for record in frozen.records] == [record.name for record in derived]
    for committed, actual in zip(frozen.records, derived, strict=True):
        assert [f.name for f in committed.fields] == [f.name for f in actual.fields], committed.name
        assert [f.type for f in committed.fields] == [f.type for f in actual.fields], committed.name
        assert [f.optional for f in committed.fields] == [f.optional for f in actual.fields], committed.name


def test_every_frozen_field_carries_a_definition() -> None:
    """A field nobody defined is a column a reader has to guess at; the freeze refuses to ship one."""
    frozen = load_schema(SCHEMA_V2)
    undefined = [
        f"{record.name}.{field.name}"
        for record in frozen.records
        for field in record.fields
        if not field.definition.strip()
    ]
    assert undefined == []


def test_an_optional_field_says_what_its_absence_means() -> None:
    """Nulls survive serialization, so each one has to mean something stated rather than inferred."""
    frozen = load_schema(SCHEMA_V2)
    unexplained = [
        f"{record.name}.{field.name}"
        for record in frozen.records
        for field in record.fields
        if field.optional and not field.absent.strip()
    ]
    assert unexplained == []


@pytest.mark.parametrize(("version", "schema", "markdown"), FROZEN)
def test_each_version_describes_exactly_its_own_records(version: int, schema: object, markdown: object) -> None:
    """A frozen document carries its version's record list, and the timing report is never in it."""
    del markdown
    frozen = load_schema(schema)  # type: ignore[arg-type]
    assert [record.name for record in frozen.records] == list(RECORDS_BY_VERSION[version])
    assert "ManualTimingReport" not in {record.name for record in frozen.records}


@pytest.mark.parametrize(("version", "schema", "markdown"), FROZEN)
def test_the_markdown_renders_from_the_committed_schema(version: int, schema: object, markdown: object) -> None:
    """The rendering is generated from the record, never hand-edited beside it."""
    del version
    frozen = load_schema(schema)  # type: ignore[arg-type]
    assert render_schema_markdown(frozen) == markdown.read_text(encoding="utf-8")  # type: ignore[attr-defined]


def test_a_numeric_field_declares_its_unit_and_what_it_aggregates_over() -> None:
    """A number without a unit or a denominator cannot be read back correctly by anyone but its author."""
    frozen = load_schema(SCHEMA_V2)
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


@pytest.mark.parametrize(("version", "schema", "markdown"), FROZEN)
def test_the_frozen_document_states_the_version_it_belongs_to(version: int, schema: object, markdown: object) -> None:
    """A changed field is a version decision, not a silent refresh: each artifact names its own version."""
    frozen = load_schema(schema)  # type: ignore[arg-type]
    assert frozen.schema_version == version
    assert schema.name == f"result_schema_v{version}.json"  # type: ignore[attr-defined]
    assert markdown.name == f"result_schema_v{version}.md"  # type: ignore[attr-defined]
    assert version in SUPPORTED_RESULT_SCHEMAS


def test_version_one_is_retained_exactly_as_frozen() -> None:
    """Raising the version adds a document beside v1; it never edits the one already frozen."""
    assert RESULT_SCHEMA_VERSION == 2
    assert len(RECORDS_BY_VERSION[1]) == 11
    assert list(RECORDS_BY_VERSION[2][:11]) == list(RECORDS_BY_VERSION[1]), (
        "v2 extends v1's records rather than reordering them"
    )
    assert load_schema(SCHEMA).schema_version == 1


def test_the_new_records_are_the_handoff_definitions() -> None:
    """The six records the freezes reference by digest instead of duplicating."""
    added = set(RECORDS_BY_VERSION[2]) - set(RECORDS_BY_VERSION[1])
    assert added == {
        "ModelAccount",
        "BankAccount",
        "StudyAccounting",
        "IllustrationCase",
        "Selection",
        "ResimulationSubset",
    }


def test_the_unexecuted_count_is_defined_and_currently_zero_by_construction() -> None:
    """Section 7.1 wants not-executed reported; the sweep cannot produce it, and the schema says so."""
    frozen = load_schema(SCHEMA_V2)
    evidence = next(record for record in frozen.records if record.name == "ManualModelEvidence")
    unexecuted = next(field for field in evidence.fields if field.name == "n_unexecuted")
    assert "zero" in unexecuted.definition.lower()
    assert unexecuted.scope.strip()
