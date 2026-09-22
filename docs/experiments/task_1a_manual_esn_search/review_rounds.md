<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# M3MS review rounds

Owner review findings on this experiment's implementation, with what each one
reproduced and how it was closed. The task queue keeps one short row per task
and links here; this is the detail, recorded beside the experiment it belongs
to. Every finding was fixed with a reproducing test written first.

## M3MS-001, round 1 (2026-09-22): four P2 findings

The committed protocol matched the approved plan, but its validation and
identity guarantees did not hold for other inputs.

1. **Different study sources shared one identity.** `protocol_digest` reduced
   the study path to its file name before hashing, so two studies named
   `study_manifest_v1.json` produced the same identity and a resume could land
   on a changed study. The identity now keeps the study's repository-relative
   location — `config_digest` already makes an in-repository path portable — and
   binds the study's own content digest beside it.
2. **The loader accepted settings outside the approved scope:** an unshared
   ceiling, four parallel trials, two configurations, two comparison scenarios,
   one replay bank, another reservoir seed. `scope_mismatches` became the single
   place those invariants live, and `ManualSearchProtocol` refuses to exist
   while it returns anything. The resource caps remain free to tighten, which
   is the one relaxation the owner permitted.
3. **Filter verification was optional.** `filter_mismatches` existed but the
   loader never called it, so a velocity cutoff of 1 Hz loaded cleanly. The
   check now runs at the validated loading boundary and names the recorded
   value, its source file and the difference.
4. **Selection ranked evidence the objective cannot produce.** Scores outside
   `{0, 0.5, 1}` were accepted, including NaN, which sorted ahead of a real
   success; three points sharing trial number 0 were selected as three
   configurations; and `nominal_success_fraction(1, runs=1)` reported a full
   nominal success from one tracker. Trial numbers must now be distinct, scores
   are validated as counts of the judged runs, and the objective requires both
   tracker runs.

## M3MS-001, round 2 (2026-09-23): two P2 findings

The scope restrictions, the mandatory filter check and content-sensitive
identities were confirmed working. Two gaps remained.

1. **The selection denominator could still be overridden.** Validation used the
   caller's `runs`, so score 0.75 at four runs and 0.3 at zero runs were
   accepted. The denominator is now checked against the protocol's two tracker
   runs before any trial is examined, so an empty study cannot slip a wrong one
   through either.
2. **The whole-location guarantee was conditional.** A path outside the
   repository is reduced to a file name, so two external studies with identical
   names and contents still collided. The protocol now refuses a study outside
   the repository, which has no portable location, and refuses one that cannot
   be read: the identity always names a committed study by its
   repository-relative path and binds its content. The round also noted that
   the earlier location test changed other configuration paths by copying the
   tree; the identity tests now run against a scratch tree that the code treats
   as the repository, so location binding is exercised with every other field,
   including every other configuration path, unchanged.

A documentation correction from the same round: the queue's focus block still
named the superseded protocol digest. The digest is `029739812525`, recorded in
the task row and pinned by `tests/regression/test_manual_search_config.py`.

## M3MS-003, round 1 (2026-09-23): four findings, two P1

The nominal-scope guard held, but resume handling and resource enforcement did
not.

1. **P1 — interrupted work could exceed the trial cap instead of resuming.**
   The parent always asked for a new trial and recorded spend only afterwards,
   so an interruption left a RUNNING trial while the resume opened the next
   one, and an interruption between finalizing and accounting lost the spend.
   The parent now writes a **reservation** before a worker starts, recovers
   every pending reservation under its original trial number before scheduling
   anything new, and derives the ledger from the retained records rather than a
   counter that can lag. An infrastructure failure is no longer read as a
   failed fit: a worker that writes no report leaves its trial pending, while a
   fit failure reports itself and consumes its slot.
2. **P1 — storage accounting omitted most artifacts and all failed work.** Only
   `run.json` sizes were summed. The worker now measures the directories it
   actually wrote — each run's payloads, the evidence manifest and the fit —
   and reports them for a failed candidate too, so the shared ceiling measures
   what it claims.
3. **P2 — worker reports were scored without verification.** A report naming
   another trial, with successes that its own statuses contradicted and no
   evidence, was recorded as a completed trial. A report must now name the
   trial that was scheduled, and a scored candidate must point at a model
   evidence manifest that verifies against its digest and belongs to this
   trial's configuration; the successes are recounted from that evidence.
4. **P2 — the elapsed cap did not constrain a running worker.** The cap was
   only checked between trials. The remaining allowance, less the headroom kept
   to persist an in-flight result, is now the worker's timeout, and a worker
   that reaches it is an interruption whose work is retained for recovery.
