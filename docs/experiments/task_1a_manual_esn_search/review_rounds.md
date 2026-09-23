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

## M3MS-003, round 2 (2026-09-23): six findings, four P1

The reservation protocol held its shape, but every write boundary and every
resource path needed to be examined one interruption at a time.

1. **P1 — retries erased earlier elapsed time.** `spend.json` was overwritten
   per attempt and finalization charged only the last one. Spend now
   accumulates over attempts and the outcome charges the total, so three
   attempts of 10 s, 20 s and 1 s cost 31 s rather than 1 s.
2. **P1 — near the deadline the timeout was disabled.** A non-positive
   allowance became an unbounded worker. A non-positive allowance now stops
   scheduling, for pending retries as well as new trials, and `spawn_trial`
   itself refuses to start a worker without a real deadline.
3. **P1 — recovery did not reconcile the two stores.** A trial lost between
   being drawn and having its reservation published was orphaned, and an
   outcome published before the study was told left a trial running for ever.
   A reconciliation pass now runs before anything is scheduled: an outcome the
   study has not heard of is finalized from the record, a drawn-but-unreserved
   trial is adopted under its own number from its recorded parameters, and one
   with no usable parameters is abandoned explicitly and charged nothing.
4. **P1 — partial-run storage was uncharged.** Bytes were measured only from a
   completed evidence object. The reservation now records the fit and evidence
   identities *before* the work runs, so the parent discovers retained work
   from the fit cache, the evidence directory and the run-granular progress
   record — for failed and interrupted attempts alike — and the worker no
   longer reports a storage figure at all.
5. **P2 — storage failures were finalized as failed candidates.** Only a
   `ValueError` — the learner or the protocol refusing a candidate — is a
   failed trial now. An infrastructure error propagates, leaves no report, and
   the trial is retained for a retry.
6. **P2 — evidence verification checked a label.** The parent now rebuilds the
   expected entry from the protocol and the reserved point, requires the
   evidence to carry the reserved evidence and fit identities, and loads every
   run through the reader a resume uses, so a corrupted archive or a different
   input scaling is refused before the candidate is scored.

## M3MS-003, round 3 (2026-09-23): five findings, two P1

Reconciliation and per-attempt accounting held. What remained were the moments
nobody was left to measure or to name what had been written.

1. **P1 — a parent interruption still lost elapsed time.** The attempt's cost
   was measured by the parent and written only when the attempt ended, so a
   parent killed while its worker ran charged nothing for the time it ran. The
   attempt is now opened on disk *before* the worker starts, and an attempt
   with no measured cost is closed on the next resume by the wall clock it was
   in flight: five minutes lost to a killed parent are five minutes charged.
2. **P1 — runs published before their progress record were uncharged.** A run
   payload becomes visible when its staging directory is renamed into the runs
   bucket; the progress record names it only once its pair is complete. Between
   the two, nothing pointed at it. `write_run` now takes a `claim`, called with
   the run's identity after the run is accepted and before its payload is
   published, and the evaluation records those claims beside the progress file,
   so ownership exists before the payload does. Bytes staged before any claim
   belong to the one worker the serial protocol allows and are charged to the
   search. An unreadable progress or claims record is refused rather than read
   as an empty inventory: bytes that cannot be counted are not bytes that are
   not there.
3. **P2 — partially sampled trials became free replacements.** A trial drawn
   but lost before a single parameter was recorded was failed in the study and
   charged nothing, so an interruption mid-draw bought a replacement outside
   the approved count. Abandoning it now writes a retained failed outcome, so
   the ledger charges the slot it took.
4. **P2 — stored identities still substituted for evidence verification.** The
   parent checked the reserved fit and evidence identities, which are recorded
   fields and cannot vouch for the bindings beside them. It now rebuilds the
   study's own fit inputs and runs `verify_model_evidence` — the one
   implementation the sweep's resume and the audit also use — against them, so
   a candidate whose cached weights no longer match their digest is refused
   instead of scored. A scored candidate cannot be finalized at all without
   the protocol and those inputs.
5. **P2 — corrupt cached evidence was classified as a failed candidate.**
   `EvidenceIntegrityError` covered run payloads but not the fit cache, whose
   digest and shape checks raised a plain `ValueError` — the worker's failed-
   candidate branch. The error now lives with the fit cache, the lowest layer
   that serves stored evidence, and the fit readers raise it, so a corrupted
   cache propagates, leaves no report and stays recoverable.

## M3MS-003, round 4 (2026-09-23): three findings, two P1

The owner reproduced all three against temporary fixture stores.

1. **P1 — a fresh parent could not verify valid evidence.** The parent probed
   its execution environment before the numerical runtimes were loaded, so its
   identity omitted rclib's OpenMP runtime and the study it was resuming
   refused it — invisible in every in-process test, because the fixture loads
   those runtimes first. The parent no longer keeps a preamble of its own: its
   fit inputs are prepared through `prepare_runner`, the path a worker uses,
   which requires the canonical environment, loads the runtimes before the
   probe and binds the study. The successful path is now tested in a fresh
   interpreter, where it is the only place it can fail.
2. **P1 — staged storage became uncharged at finalization.** Unpublished
   payloads were charged only while a reservation was pending, so finalizing
   the trial that wrote them handed the search a fresh allowance while the
   bytes remained: a 1,000,000-byte staged payload dropped out of the ledger.
   They are charged to the search itself now, once per ledger and for as long
   as they are in the store, because no outcome counts them and finalizing a
   trial does not remove them.
3. **P2 — an undecodable cached payload was still a failed candidate.** The
   digest and shape checks raised `EvidenceIntegrityError`, but a decoding
   failure from `np.load` — or from the record's own JSON — escaped as an
   ordinary `ValueError`, which the worker reads as a terminal candidate
   failure. Every read of a cached payload now goes through one guard that
   turns anything stopping it from being read whole into an integrity fault,
   so a truncated or overwritten cache propagates and the trial stays
   recoverable.

## M3MS-004, round 1 (2026-09-23): two P2 findings and one wording correction

The owner independently verified the pilot. The study holds 10 COMPLETE
trials and there are 20 nominal runs (12 successful, 8 infeasible). All model
bindings, cached fits and run payloads verify, and provenance names clean,
non-exploratory `5a6cde2`. The owner confirmed 111.84 s charged and 15.14 MiB
retained, confirmed that the report's rows, the Optuna parameters and scores,
and the projection arithmetic agree, and confirmed that trials 1, 2 and 4
fix the selection. The findings concern the report generator:

1. **P2 — cached runs diluted the per-run projection.** The generator divided
   the measured simulation time and bytes by every recorded run. In the
   owner's reproduction, one cached two-run result halved the estimate from
   1.392 to 0.696 s per run, and halved bytes per run too. Rows now carry
   the worker's `simulated_runs`, and the projection divides by those runs
   only. With no simulated run, the cost is unavailable, never zero.
2. **P2 — an abandoned trial broke the report.** Reconciliation retains a trial
   lost mid-sampling as an outcome alone, and the row builder required a
   reservation, so report generation raised `FileNotFoundError`. That trial
   is now reported with its parameters and timing unavailable.
3. **Wording.** The comparison figure was called a lower bound. Nominal
   measurements do not bound perturbed runs, so it is now an "estimated
   runs-and-fits cost".

Each finding has a test written first, and each test fails against the
committed generator. v1 of the report moved unedited to
[`pilot/superseded/`](pilot/superseded/README.md). v2 was regenerated from the
same records and carries identical figures, because neither defect occurred
in this pilot. The owner recommends finishing the approved 100 trials: stopping
early would need an explicit amendment.

## M3MS-005, round 1 (2026-09-23): one P2

The owner independently confirmed the search and the committed freeze. There
are 100 scored trials and 200 nominal runs (144 successful, 56 infeasible),
with scores 70 × 1.0, 4 × 0.5 and 26 × 0.0. All model bindings, fits and run
payloads verify under the stated clean revisions. The spend is 1,084.004 s
and 145.591 MiB. The selection is trials 1, 2 and 4, and the committed digest
and Markdown match.

1. **P2: a loaded freeze was checked against itself, not against the
   search.** `read_freeze` enforced internal consistency only. The owner
   reproduced three self-consistent edits that it accepted:
   - trial 4 replaced by trial 7, carrying trial 7's genuine bindings;
   - a chosen point's input scaling changed while its fit identity was kept;
   - the search's recorded time and storage set to zero.

   `load_verified_freeze` now rebuilds the whole freeze from the protocol and
   the retained trial records, verifying each chosen trial's evidence again.
   It requires the stored record to equal the rebuilt one, and its bytes to
   equal the canonical form that the comparison binds. A digest computed from
   the edited document would have verified nothing.

   To keep that comparison stable, the frozen spend is now the trials' own
   recorded outcomes (`recorded_spend`), summed exactly as the ledger sums
   them. Bytes that an interrupted comparison leaves staged in the shared runs
   bucket therefore cannot make the frozen search look different. For the
   committed freeze the two are bitwise equal: 100 trials, 1,084.0039491942152 s
   and 152,662,951 bytes, with nothing staged. `selection_v1.json` therefore
   stands unchanged.

   Tests written first: each of the three edits passes `read_freeze` and is
   refused by the verified load; a re-serialized copy is refused on its bytes;
   later staging does not unfreeze the search (this test fails if the freeze
   uses the live ledger); and every verified load re-verifies the chosen
   evidence.
