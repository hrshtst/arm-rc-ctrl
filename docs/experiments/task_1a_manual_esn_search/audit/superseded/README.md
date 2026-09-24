# Superseded comparison audit

An audit that was run is kept. `audit_v1.{json,md}` ran on 2026-09-24 from
clean `9d7407a`, pinned to the P-cores, and passed all seven steps. It is not
the audit of record: `../audit_v2.{json,md}` is. Nothing here is edited.

## `audit_v1`: superseded by v2

The owner's review of M3MS-007 found that v1 could pass with incorrect headline
results. It checked the index's bound inputs, documents and accounting, but
not its headline figures. Changed counts of RC successes, RC runs, unavailable
runs and the departure radius all passed, and so did a replaced `results_v1.md`.
v1 also read the index before its recording boundary, so a malformed index
raised instead of producing a failed record. Its completeness step also took
the model and bank counts from the index it was checking.

v2 recomputes every headline figure from the per-run table and compares it with
the index. It counts models and banks from the rows themselves, requires
`results_v1.md` to be the index's rendering, and reads the index inside the
audit, so an unreadable index is recorded as unavailable steps rather than
raised. The derived evidence itself was not changed: v1's passing steps and
the owner's independent checks both found it correct, and v2 audits the same
committed derivation.
