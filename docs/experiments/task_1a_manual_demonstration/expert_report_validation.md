<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Expert HTML report validation — M3MAN-015

Completed 2026-09-24 on the isolated `docs/manual-expert-report` branch.
The [report](expert_report/index.html) is an additional presentation of the
accepted manual-demonstration study. It does not incorporate the concurrent
M3MS search or change the original report, results, audit, or protocol.

## Deliverable and provenance

The report contains 27 PNG figures and eight interactive, synchronized five-arm
case studies (40 displayed runs). Each case has task-space, joint-angle and
diagnostic time-series plots. Background, methods, results, discussion and
conclusions assume no knowledge of earlier project experiments.

The renderer reads the audited tables and verified payloads, reuses the existing
summary and joint/endpoint renderers, and makes no fit, simulation or store write.
The primary results and configuration tables are generated from the evidence.
The eight examples are an explicitly disclosed subset of the frozen illustrations;
they do not estimate the prevalence of the behaviours they show.

The final presentation was regenerated from clean commit
`8a6f746243263df42730d9604b95e7a6de0dba6c`. All 40 presentation files were
byte-identical to the tested draft. Only `manifest.json` changed, recording this
commit and `project_dirty: false`. The manifest binds sources, inputs, the stored
table and every output. [Reproduction and verification commands](expert_report/README.md)
are retained beside the HTML; authored sources live in
[`expert_report_source/`](expert_report_source/README.md).

## Automated validation

The full gate used the repository's `p-cores` execution launcher, with an
independent worktree environment and no `PYTHONPATH` override:

```sh
uv run --locked python -m arm_rc_ctrl.execution run --policy p-cores -- \
  uv run --locked nox -s deps lint type_check tests cpp pre_commit
```

All six sessions passed. Python: **3,509 passed, 1 skipped, 1 expected failure**;
coverage **90.46%**. Both C++ tests passed. The gate completed before the
implementation commit; the final follow-up changes only provenance and task
documentation. Focused presentation tests and hooks were repeated afterward.

Nine new tests cover display sampling, event/end retention, the generated results
table, sparse legend colours, output and source tampering, input bindings, local
HTML links, frozen case/run bindings, and complete regeneration from the store.
The integration test compares all 40 presentation outputs byte-for-byte. The
original 14 presentation tests remain in place.

The first full-gate attempt exposed isolated-worktree setup problems: a symlinked
environment executable directory, missing nested dependency checkouts, and a
`PYTHONPATH` override leaking into a historical reproduction checkout. These were
corrected using an independent environment and pinned dependency checkouts.
The three affected tests passed in a focused recheck, followed by the successful
full gate above. An intervening focused coverage run was insufficient for the
repository-wide threshold and was superseded by that full run. No threshold,
test, or production execution rule was relaxed.

## Payload and browser checks

An additional direct comparison checked all 40 displayed runs against their
stored arrays: display times, measured and desired joint angles, first/final
samples, pulse instants and run endpoints agree. Six-decimal display values were
checked within 0.000000501; final endpoint errors matched the audited table within
0.000000001 mm. These are presentation checks, not new analysis tolerances.
Scientific plots retain full-resolution samples; only playback is decimated.

Headless Chrome checks exercised all eight case objects and 40 canvases, image
loading, scrubbing, playback/pause, pulse display and early-abort handling. No
JavaScript errors occurred. Desktop and 390-pixel mobile screenshots were
visually inspected; mobile had no horizontal overflow. All resources are local,
with no CDN, automatic playback, or server requirement. Static plots remain
available without JavaScript.

The main checkout remained clean. Work and commits stayed in the separate
worktree so M3MS-006 could continue; nothing was pushed or merged into main.
