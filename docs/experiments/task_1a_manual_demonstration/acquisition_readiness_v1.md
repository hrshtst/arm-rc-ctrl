<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Task 1-a manual demonstrations: acquisition readiness and frozen settings

- **Experiment label:** `task_1a_manual_v1`
- **Frozen on:** 2026-09-16 by the owner, closing M3MAN-003, before any study take.
- **Evidence:** the excluded practice session `practice-pilot-03`, recorded by the
  owner with the v2 configurations and assessed offline in a throwaway store and
  records root. Practice payloads are disposable and are not retained; this
  summary and the versioned configurations are the retained readiness record.

## Frozen settings

| Setting | Frozen value | Where |
| --- | --- | --- |
| Acquisition rate | 50 Hz, actual timestamps retained | `configs/tasks/task_1a_manual_v2.toml`, `configs/recording/task_1a_manual_v2.toml` |
| Training/control grid | 0.01 s (100 Hz) for every comparison arm | `configs/tasks/task_1a_manual_v2.toml` |
| Maximum sample gap | 0.06 s of actual time | `configs/tasks/task_1a_manual_v2.toml` |
| Minimum recording duration | 1.0 s | `configs/tasks/task_1a_manual_v2.toml` |
| Start tolerance | 1e-12 rad against the reset posture | `configs/tasks/task_1a_manual_v2.toml` |
| Per-joint velocity bound | 6 rad/s (D3 canonical; the pilot stayed far below) | `configs/tasks/task_1a_manual_v2.toml` |
| Final dwell | 1.0 s continuous, 0.01 m radius, 0.05 rad/s per joint | `configs/tasks/task_1a_manual_v2.toml` |
| Smoothing | zero-phase Butterworth, 5 Hz, order 4, on the offset from the first sample with a 5 s point-reflection extension; start shift bound 1e-12 rad | `configs/preprocessing/manual_v2.toml` |
| Reconstruction / derivatives | linear interpolation from actual timestamps / central differences | `configs/preprocessing/manual_v2.toml` |
| Recorder overlays | current tip trail plus the most recently saved trail | `configs/recording/task_1a_manual_v2.toml` |
| Per-take timeout | 30 s | `configs/recording/task_1a_manual_v2.toml` |
| Contractive envelope ramp | 0.5 s from task time zero, terminal taper unchanged (I14) | this record; implemented by M3MAN-006 |
| Augmentation seed namespace | `task_1a_manual_v1/contractive/v1` | this record; implemented by M3MAN-006 |
| Execution environment | the canonical pinned launcher established by M3REP-009 | `docs/experiments/task_1a_repeated_demonstration/execution_environment_v1.json` |
| Panel and input transform | the six inherited configurations (trials 17, 136, 53, 1, 0, 28) and the historical scripted-data centers and scales (I8) | plan sections 4 and 5 |

## What the practice session showed

Ten takes were recorded and assessed; seven were accepted. The three rejections
failed only the dwell duration, at 0.58 s, 0.73 s and 0.92 s against the
required 1.0 s; none failed on position, speed, gaps or the start.

| Measurement | Observed over the ten takes | Frozen limit |
| --- | --- | --- |
| Median sample interval | 20.0 ms | 20 ms nominal |
| 99th percentile interval | 20.8 to 24.6 ms | |
| Largest sample gap | 39.9 ms | 60 ms |
| Late acquisition ticks per take | 0 to 2, without growth across the session | |
| Start shift after smoothing | 0.0 rad in every take | 1e-12 rad |
| First departure from the reset posture | 0.11 to 1.57 s | no minimum required |
| Peak processed joint speed | 0.31 to 0.46 rad/s | 6 rad/s |
| Endpoint error over the final second | 1.6 to 6.1 mm | 10 mm |
| Take length | 12.2 to 14.3 s | 30 s timeout |

The first practice session, recorded at 100 Hz with the full trail history,
showed acquisition ticks falling behind as saved trails accumulated, from 3 late
ticks in the first take to about 180 in the tenth, with gaps up to 73 ms. A
repaint benchmark grew by about 1 ms per drawn trail. That is consistent with
trail-drawing cost, although it was not established as the cause. With 50 Hz
acquisition and only the last saved trail drawn, the growth is absent.

## Operator note

The dwell tolerance is a 1 cm radius, about 20 to 75 mrad of joint motion at the
dwell posture and roughly 13 screen pixels across in a 1024 by 768 window. It was
kept unchanged because the closed-loop evaluation uses the same target region.
Holding for about two seconds before saving, and releasing the mouse button once
inside the region so the arm stops exactly, satisfies the rule comfortably.
