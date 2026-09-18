# Task 1-a manual-demonstration timing smoke check (v1)

Experiment `task_1a_manual_v1`, study manifest sha256 `b07b824c362a`, evaluation config sha256 `8780be5bac7f`, execution identity `a7f034c7aef4` (canonical), project commit `99e8b47b31da`.

Derived from the report with sha256 `676dc1d6d033` by code revision `4503e4aade28` at 2026-09-18T10:59:04+00:00. Reason: the original credited 24 restricted-sweep models as complete, deducting time the study cannot skip. No measurement was re-run; every measured figure below is the original's.

## Measured cost

- Wall time of this invocation: 0.03 h (116 s); 60 of 60 runs were simulated by it.
- Measured 24 model(s) over 24 entr(ies) and built 6 replay bank(s).
- Peak resident set size: 0.4 GiB for this process, 0.4 GiB for its waited-for children.
- Storage of the measured runs and this invocation's manifests: 0.0 GiB.

| arm | runs | mean simulate s | median simulate s | max simulate s | mean persist s | mean bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| replay | 12 | 1.370 | 1.370 | 1.685 | 0.003 | 594292 |
| rc | 48 | 1.378 | 1.280 | 1.894 | 0.005 | 737447 |

## Models

| model | fit | fit s | sweep s | runs |
| --- | --- | ---: | ---: | ---: |
| feasible-best/S/D01 | fitted now | 0.05 | 3.2 | 2 |
| feasible-best/M10 | fitted now | 0.32 | 3.1 | 2 |
| feasible-best/R10/D01 | fitted now | 0.31 | 3.1 | 2 |
| feasible-best/C10/D01 | fitted now | 1.64 | 3.1 | 2 |
| feasible-middle/S/D01 | fitted now | 0.14 | 3.3 | 2 |
| feasible-middle/M10 | fitted now | 1.19 | 3.3 | 2 |
| feasible-middle/R10/D01 | fitted now | 1.30 | 3.4 | 2 |
| feasible-middle/C10/D01 | fitted now | 2.61 | 2.0 | 2 |
| feasible-worst/S/D01 | fitted now | 0.12 | 3.3 | 2 |
| feasible-worst/M10 | fitted now | 0.86 | 3.3 | 2 |
| feasible-worst/R10/D01 | fitted now | 0.96 | 3.3 | 2 |
| feasible-worst/C10/D01 | fitted now | 2.29 | 3.4 | 2 |
| failure-actual-dwell/S/D01 | fitted now | 0.04 | 1.9 | 2 |
| failure-actual-dwell/M10 | fitted now | 0.24 | 1.9 | 2 |
| failure-actual-dwell/R10/D01 | fitted now | 0.26 | 1.9 | 2 |
| failure-actual-dwell/C10/D01 | fitted now | 1.56 | 1.8 | 2 |
| failure-joint-velocity/S/D01 | fitted now | 0.09 | 3.3 | 2 |
| failure-joint-velocity/M10 | fitted now | 0.69 | 3.3 | 2 |
| failure-joint-velocity/R10/D01 | fitted now | 0.76 | 3.3 | 2 |
| failure-joint-velocity/C10/D01 | fitted now | 2.05 | 3.2 | 2 |
| failure-generated-dwell/S/D01 | fitted now | 0.15 | 3.2 | 2 |
| failure-generated-dwell/M10 | fitted now | 0.79 | 3.2 | 2 |
| failure-generated-dwell/R10/D01 | fitted now | 0.89 | 3.3 | 2 |
| failure-generated-dwell/C10/D01 | fitted now | 2.19 | 3.2 | 2 |

## Full-study projection (measured means scaled to every run; an estimate, not a bound)

- 6 configurations x 31 arms = 186 models x 130 pairs = 24,180 RC runs at 1.38 s each.
- 6 configurations x 10 parents = 60 replay banks x 130 pairs = 7,800 replay runs at 1.37 s each.
- 31,980 runs in total; fits 0.05 h.
- Projected total: 12.31 h; storage about 20.9 GiB.
- Already complete after this check: 0 model(s); remaining about 12.31 h.

## Revised estimate

Derived, not re-run: every measured figure here is the original's. The locked protocol is 65 scenarios under 2 trackers, so the study is 31,980 runs, projecting 12.31 h of serial simulation and about 20.9 GiB of run data from the means this measurement established. This is a projection from a subset, not a guaranteed bound, and M3MAN-010 waits for the owner's budget approval.

## Limitations

- The projection multiplies maximum run counts by means measured on a subset. It is an estimate and not a guaranteed bound: reservoir sizes, recording lengths and storage overhead vary, an aborted run costs less, and an infeasible model still costs its fit.
- Timings are wall-clock in the canonical single-threaded execution environment of this machine (C10); another core type, thread setting, or machine measures differently.
