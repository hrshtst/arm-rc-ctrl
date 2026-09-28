<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Reusable expert-report media — M3MAN-016

The exporter packages the eight animations from the [expert report](expert_report/index.html)
as ordinary GIF files, with their matching plots and captions. It reads the verified
presentation already in Git; it needs neither the external evidence store nor any
new simulation. The accepted study and the HTML presentation are unchanged.

## Reproduce

From the repository root, choose an output directory that does not exist:

```sh
uv run --locked python scripts/export_manual_expert_media.py --output /tmp/manual-expert-media
```

The command writes that directory and a sibling `manual-expert-media.zip`.
It refuses an existing directory or archive. Keep an interrupted export for
inspection and choose a new destination for the next invocation.

## Find matching files

| Folder | Configuration / case |
| --- | --- |
| `00_overview` | Teacher trajectories and overall/class success summaries |
| `01_rescue` | A / nominal; all-ten rescues a weak singleton |
| `02_nominal` | B / nominal; synthetic augmentation fails |
| `03_offset` | B / posture offset; singleton succeeds, all-ten fails |
| `04_ceiling` | C / nominal; all five arms succeed |
| `05_force` | C / computed torque and force; strict-tolerance recovery failure |
| `06_abort` | D / nominal; learned arms abort early |
| `07_force_abort` | E / computed torque and force; abort after reaching |
| `08_slow` | F / force; delayed singleton recovery succeeds |

Each numbered case has the same filenames:

- `animation.gif`: synchronized S, M10, R10, C10 and replay panels.
- `task-space.png`: measured and desired endpoint paths.
- `time-series.png`: error, speed, tracking and effort diagnostics.
- `joint-angles.png`: joint trajectories and endpoint error.
- `README.md`: original case caption and viewing notes.
- `case.json`: source case, outcome labels, pulse times and file fingerprints.

The matching PNGs are byte-for-byte copies of the report's full-resolution plots.
The GIFs use its decimated, six-decimal display samples: the last available
sample is held, without interpolation or extrapolation. They are illustrations,
not a new source for numerical measurements or estimates of success prevalence.
All eight cases use parent D01, as in the report.

## Playback and provenance

GIFs are 1500 × 450 pixels at nominally 20 frames/s, labelled **2× playback**.
They retain the warm-up and 30-second active window, then pause on the final frame
for one playback second before looping. Time is relative to activation. A run
that ended early remains grey at its last recorded state and shows its ending
time. PASS/FAIL labels describe the whole recorded run, not the current frame.
Force indicators use each run's actual recorded pulse window.

The bundle's `manifest.json` records the source report-manifest digest, exporter
and font digests, producing revision, dirty state, and every output fingerprint.
The original report manifest binds the display data to audited source artifacts.
The exporter lives in `src/arm_rc_ctrl/experiments/manual_expert_export.py`; the
script is a thin entry point. Tests cover GIF timing and looping, preceding-sample
selection, unchanged matching figures, source metadata, archive inventory and
refusal to overwrite an earlier export.

## Validation of the delivered export

Produced from clean commit `322c901` on the isolated `docs/manual-expert-report`
branch. The [export manifest](expert_media_manifest_v1.json) records all 52 output
fingerprints; all 52 reproduce byte-for-byte from the tested preview. All eight
GIFs were decoded and checked for dimensions, frame counts, loop settings and
playback durations. The ZIP contains all 53 files including its manifest and
passes its CRC check. Success, pulse and aborted-state frames were visually
inspected. No study or report asset was modified.

Five exporter tests plus the existing report tests pass. The full pinned gate
passed: 3,514 tests, 1 skipped, 1 expected failure, 90.48% coverage; dependency
verification, lint, type checks, C++ and pre-commit all passed. The final closure
adds only this record, the output manifest and task documentation.
