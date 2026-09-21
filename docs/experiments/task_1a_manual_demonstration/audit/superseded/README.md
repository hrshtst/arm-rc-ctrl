# Superseded reproduction audits

An audit that was run is kept, whether or not it passed. The records here were
produced by a real run over the committed evidence and were then superseded, so
they are **not** the audit of record for this experiment. The issued audits are
`../reproduction_audit_v1.{json,md}`, `../reproduction_audit_v2.{json,md}` and
`../reproduction_audit_v4.{json,md}`, the last being current. Nothing here is
edited: each record and its rendering stand as the run left them, and
`tests/regression/test_manual_audit_evidence.py` holds them to that.

## `reproduction_audit_v3` — superseded by v4

Run on 2026-09-21 from clean `9f47e81`, pinned to the P-cores. Every check of
the evidence passed: 39,562 checks over sources, demonstrations and their raw
records, manifests, fits, all 31,980 stored runs re-judged from their own
arrays, the aggregates, completeness and figures, and 300 of 300 frozen-subset
runs re-simulated bitwise.

It is superseded because its own `gates` step failed, and that one failure is
retained in the record. The failing gate was the `type_check` session, on four
basedpyright errors in the audit's own tests (a private helper read directly
and an untyped lambda) introduced by `9f47e81`. The record's failure text is
also a poor witness to that: this version quoted a failing gate's standard
output, while `nox` names the failing session on standard error, so the quoted
tail reads `100% tests passed` under a failed gate.

Both were fixed in `dfe3ad1` — both streams are quoted now, with a test — and
the audit was run again as version 4 from that checkout, which passed every
step. Version 3's number is retired with its run and is never reused.
