# Task 1-a manual-demonstration timing smoke check (v1)

Experiment `task_1a_manual_v1`, study manifest sha256 `b07b824c362a`, evaluation config sha256 `8780be5bac7f`, execution identity `a7f034c7aef4` (canonical), project commit `f25ba5534dcc`.

## Measured cost

- Wall time of this invocation: 0.03 h (120 s); 60 of 60 runs were simulated by it.
- Measured 24 model(s) over 24 entr(ies) and built 6 replay bank(s).
- Peak resident set size: 0.4 GiB for this process, 0.4 GiB for its waited-for children.
- Storage of the measured runs and this invocation's manifests: 0.0 GiB.

| arm | runs | mean simulate s | median simulate s | max simulate s | mean persist s | mean bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| replay | 12 | 1.412 | 1.402 | 1.769 | 0.006 | 594292 |
| rc | 48 | 1.366 | 1.273 | 1.900 | 0.005 | 737448 |

## Models

| model | fit | fit s | sweep s | runs |
| --- | --- | ---: | ---: | ---: |
| feasible-best/S/D01 | fitted now | 0.05 | 6.0 | 2 |
| feasible-best/M10 | fitted now | 0.25 | 3.1 | 2 |
| feasible-best/R10/D01 | fitted now | 0.28 | 3.2 | 2 |
| feasible-best/C10/D01 | fitted now | 1.59 | 3.1 | 2 |
| feasible-middle/S/D01 | fitted now | 0.16 | 6.1 | 2 |
| feasible-middle/M10 | fitted now | 1.10 | 3.2 | 2 |
| feasible-middle/R10/D01 | fitted now | 1.26 | 3.3 | 2 |
| feasible-middle/C10/D01 | fitted now | 2.57 | 1.9 | 2 |
| feasible-worst/S/D01 | fitted now | 0.11 | 6.4 | 2 |
| feasible-worst/M10 | fitted now | 0.85 | 3.4 | 2 |
| feasible-worst/R10/D01 | fitted now | 0.96 | 3.3 | 2 |
| feasible-worst/C10/D01 | fitted now | 2.30 | 3.3 | 2 |
| failure-actual-dwell/S/D01 | fitted now | 0.04 | 4.6 | 2 |
| failure-actual-dwell/M10 | fitted now | 0.22 | 1.9 | 2 |
| failure-actual-dwell/R10/D01 | fitted now | 0.26 | 1.8 | 2 |
| failure-actual-dwell/C10/D01 | fitted now | 1.57 | 1.8 | 2 |
| failure-joint-velocity/S/D01 | fitted now | 0.09 | 6.1 | 2 |
| failure-joint-velocity/M10 | fitted now | 0.63 | 3.2 | 2 |
| failure-joint-velocity/R10/D01 | fitted now | 0.75 | 3.4 | 2 |
| failure-joint-velocity/C10/D01 | fitted now | 2.10 | 3.3 | 2 |
| failure-generated-dwell/S/D01 | fitted now | 0.10 | 5.9 | 2 |
| failure-generated-dwell/M10 | fitted now | 0.76 | 3.2 | 2 |
| failure-generated-dwell/R10/D01 | fitted now | 0.85 | 3.1 | 2 |
| failure-generated-dwell/C10/D01 | fitted now | 2.13 | 3.1 | 2 |

## Full-study projection (measured means scaled to every run; an estimate, not a bound)

- 6 configurations x 31 arms = 186 models x 2 pairs = 372 RC runs at 1.37 s each.
- 6 configurations x 10 parents = 60 replay banks x 2 pairs = 120 replay runs at 1.42 s each.
- 492 runs in total; fits 0.05 h.
- Projected total: 0.23 h; storage about 0.3 GiB.
- Already complete after this check: 24 model(s); remaining about 0.21 h.

## Revised estimate

Measured 24 of 186 models and 6 replay bank(s) in 0.03 h on 2 pairs per model. Scaling those means to the whole study projects 0.23 h of serial simulation and about 0.3 GiB of run data for 492 runs, beside the 2026-09-15 planning estimate of about 13 h and 26-27 GB rescaled on 2026-09-17 for the revised replay count. Bounded parallel execution reduces elapsed time and not storage. This is a projection from a subset, not a guaranteed bound, and M3MAN-010 waits for the owner's budget approval.

## Limitations

- The projection multiplies maximum run counts by means measured on a subset. It is an estimate and not a guaranteed bound: reservoir sizes, recording lengths and storage overhead vary, an aborted run costs less, and an infeasible model still costs its fit.
- Timings are wall-clock in the canonical single-threaded execution environment of this machine (C10); another core type, thread setting, or machine measures differently.
