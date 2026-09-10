<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Retention incident: cleared first timing smoke-check attempt (2026-09-10)

- **Experiment:** `task_1a_repetition_v1`, task M3REP-005 (timing smoke check).
- **Affected attempt:** the first `arm_rc_ctrl.experiments.repetition_timing smoke`
  invocation of 2026-09-10, launched pinned through the canonical launcher
  at project commit `c2c6d3c` (runner as of `9e4bff8`, timing module as of
  `c2c6d3c`), against the 0.25 s replay bank and the 20 behavioral arms of
  `feasible-best` (trial 17). The run ids it wrote are dated `run-20260909-…`
  because run ids carry the UTC provenance date.
- **What happened:** the attempt evaluated the replay bank (130 runs) and the
  13 absolute arms completely (12 feasible sweeps of 130 pairs and the
  augmented arms' first-pair failures), then started the residual arms. On
  `residual/S`, the first pair (`nominal [pd_v2]`) terminated with an
  `invalid_output` (the generator's target left the joint bounds at task time
  3.71 s). The velocity diagnostics record (`VelocityDiagnostics` in
  `experiments/velocity_diagnostics.py`) only allowed a terminal checked state
  together with a joint-speed abort detail, so it raised
  `an abort and its terminal checked state are recorded together` after the
  run had already been persisted, and the invocation exited with status 1
  before writing the timing report or any Git pointer.
- **Diagnostics defect and fix:** the invariant was wrong for every early
  termination that is not a speed abort (other limit violations, invalid
  states, controller failures). Commit `e309be2` keeps the terminal checked
  state for every run that does not complete, names the termination kind and
  detail, and records the abort detail only for joint-speed limit violations;
  pair records now also carry their measured timing so a resumed sweep keeps
  every original measurement.
- **Artifacts removed (2026-09-10, before the rerun):** 1,047 run records
  (`<store>/runs/run-20260909-…`, identified by the note prefix
  `task_1a_repetition_v1`: 130 replay runs and 917 RC runs, about 125 MiB),
  14 model evidence directories and one replay-bank directory under
  `<store>/reports/task_1a_repetition_v1/` (progress files and the 13
  completed model manifests). They were cleared so that the rerun would
  measure every run afresh in one uninterrupted invocation, because the
  runner as of `c2c6d3c` did not persist run timings and a resume would have
  served the completed runs without their measurements. No fit cache entry
  (`<store>/models/task_1a_repetition_v1/`), numerical-validation probe, or
  earlier study artifact was touched.
- **Surviving evidence:** the complete stdout/stderr log of the attempt is
  committed as [`incidents/smoke_attempt_1_2026-09-10.log`](incidents/smoke_attempt_1_2026-09-10.log)
  (machine paths replaced by `<repo>`, `<store>`, `<python>`); it lists every
  simulated pair in order and the traceback. The fit cache entries the attempt
  produced for the six augmented arms survive and were served to the rerun
  (their `fit_seconds` in the rerun's report are the attempt's measurements).
- **Unrecoverable:** the attempt's per-run arrays, outcomes, diagnostics, and
  in-memory timings; its wall time (about 6 minutes from the log timestamps
  is not recorded, only the order of runs). The store lives under a Dropbox
  folder, whose deleted-file history may still hold the removed run
  directories; recovering them needs the Dropbox interface and was not
  attempted from this session.
- **Relation to the rerun:** the second attempt (commit `e309be2`, evidence
  `timing_smoke_check_v1.{json,md}`) reproduced the same first failures on the
  same pairs (the simulation path is deterministic in the canonical
  environment) and is the recorded smoke check; it does not replace the fact
  that the first attempt failed, which this record preserves.
- **Policy going forward (owner decision C12):** interrupted or failed
  attempts are preserved or quarantined in place, never cleared; the runner
  deletes nothing, and a resume reuses or bypasses stored progress explicitly.
