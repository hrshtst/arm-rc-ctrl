<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Standalone expert report — M3MAN-015

Open `index.html` in a browser. Keep this directory together: scripts and images
are local, with no CDN, network request, or web server needed. Links to the
source study records assume this directory is in its documented position in the
checkout; the report text, plots and playback can be shared as this directory alone. Eight case studies
have synchronized five-arm playback, full-resolution task-space plots, joint-angle
plots, and diagnostic time series. Static plots remain available without JavaScript.

From the repository root, with the existing evidence store configured and mounted:

```sh
uv run --locked python scripts/render_manual_expert_report.py --output /tmp/manual-expert-report
```

Choose a new output directory; the renderer refuses to replace one. It reads the
closed study's audited tables and verified trajectory payloads. It performs no
training, simulation, search, or writes to the store. It reuses the original report's
summary renderer and the existing joint/endpoint renderer. The ten-teacher figure
is copied after checking its original presentation fingerprint.

For a store-free integrity check:

```sh
uv run --locked python scripts/render_manual_expert_report.py \
  --output docs/experiments/task_1a_manual_demonstration/expert_report --verify
```

Authoring sources are in `../expert_report_source/`; the renderer is
`src/arm_rc_ctrl/experiments/manual_expert_report.py`. Edit these, then regenerate.
`manifest.json` binds the sources, committed evidence, stored table and every
presentation file by SHA-256. `cases.json` binds the 40 displayed runs to their
stored summary and array digests. Its metrics come from the audited per-run table.
The primary results table and configuration table are generated from evidence.

All scientific plots use the full stored sample sequence. Playback uses every
tenth sample plus activation, force boundaries and the last sample; angles and
timestamps are rounded to six decimal places for display. It shows the nearest
preceding retained sample, without extrapolating past an abort. Display precision
is not an analysis tolerance. Each pulse follows its own recorded trigger time.

This is a new presentation of the accepted M3MAN findings, not a new analysis
protocol or an update incorporating the concurrent M3MS search. Eight examples
were chosen after results existed, from the 18 cases already selected by the
frozen illustration rule. All use D01; they illustrate mechanisms and do not
estimate their prevalence across parents. Aggregate tables supply that context.

Provenance is explicit: the manifest records the producing commit and whether
the working tree was dirty. Source hashes bind uncommitted authoring too; do not
interpret a dirty render as produced entirely by its recorded commit. A repeat
render can change this provenance while leaving all presentation bytes identical.
The original report, study evidence and audit records are preserved unchanged.
