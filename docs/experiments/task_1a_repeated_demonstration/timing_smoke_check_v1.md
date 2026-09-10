# Task 1-a repetition timing smoke check (v1)

Experiment `task_1a_repetition_v1`, panel entry `feasible-best` (manifest sha256 `68bc850ef269`, evaluation config sha256 `7f9c24e01470`), execution identity `a7f034c7aef4` (canonical), project commit `e309be264e11`.

## Measured cost

- Wall time of this invocation: 0.07 h (268 s); 1053 of 1053 measured runs were simulated by it.
- Replay bank: 130 runs in 25 s; models: 20 (923 RC runs, 1677 unexecuted pairs).
- Peak resident set size (process-cumulative): 264.5 MiB; waited-for children: 264.5 MiB.
- Storage of the measured runs and this invocation's manifests: 125.5 MiB.

| arm | runs | mean simulate s | median simulate s | max simulate s | mean persist s | mean bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| replay | 130 | 0.188 | 0.191 | 0.240 | 0.002 | 93874 |
| rc | 923 | 0.205 | 0.176 | 0.267 | 0.017 | 126270 |

## Models

| model | status | fit | fit s | sweep s | RC runs | unexecuted | run bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| feasible-best/absolute/S | feasible | cache hit | 0.05 | 29.8 | 130 | 0 | 860116 |
| feasible-best/absolute/R/K17 | feasible | cache hit | 0.12 | 30.1 | 130 | 0 | 860116 |
| feasible-best/absolute/R-scaled/K17 | feasible | cache hit | 0.12 | 30.0 | 130 | 0 | 860116 |
| feasible-best/absolute/A-non-decaying/K17 | rc_gate_failure | cache hit | 1.20 | 1.4 | 1 | 129 | 6498 |
| feasible-best/absolute/A-contractive/K17 | rc_gate_failure | cache hit | 1.16 | 1.4 | 1 | 129 | 6498 |
| feasible-best/absolute/R/K33 | feasible | cache hit | 0.21 | 30.2 | 130 | 0 | 860116 |
| feasible-best/absolute/R-scaled/K33 | feasible | cache hit | 0.20 | 30.2 | 130 | 0 | 860116 |
| feasible-best/absolute/A-non-decaying/K33 | rc_gate_failure | cache hit | 2.33 | 2.5 | 1 | 129 | 6497 |
| feasible-best/absolute/A-contractive/K33 | rc_gate_failure | cache hit | 2.33 | 2.5 | 1 | 129 | 6498 |
| feasible-best/absolute/R/K65 | feasible | cache hit | 0.41 | 30.2 | 130 | 0 | 860116 |
| feasible-best/absolute/R-scaled/K65 | feasible | cache hit | 0.39 | 30.4 | 130 | 0 | 860116 |
| feasible-best/absolute/A-non-decaying/K65 | rc_gate_failure | cache hit | 4.58 | 4.6 | 1 | 129 | 6687 |
| feasible-best/absolute/A-contractive/K65 | rc_gate_failure | cache hit | 4.54 | 4.6 | 1 | 129 | 6687 |
| feasible-best/residual/S | rc_gate_failure | cache hit | 0.03 | 0.3 | 1 | 129 | 6873 |
| feasible-best/residual/R/K17 | rc_gate_failure | cache hit | 0.12 | 0.4 | 1 | 129 | 6634 |
| feasible-best/residual/R-scaled/K17 | rc_gate_failure | cache hit | 0.12 | 0.3 | 1 | 129 | 6875 |
| feasible-best/residual/R/K33 | rc_gate_failure | cache hit | 0.21 | 0.4 | 1 | 129 | 6873 |
| feasible-best/residual/R-scaled/K33 | rc_gate_failure | cache hit | 0.20 | 0.4 | 1 | 129 | 6875 |
| feasible-best/residual/R/K65 | rc_gate_failure | cache hit | 0.40 | 0.6 | 1 | 129 | 6927 |
| feasible-best/residual/R-scaled/K65 | rc_gate_failure | cache hit | 0.39 | 0.6 | 1 | 129 | 6875 |

## Full-panel projection (measured means scaled to every pair; not a guaranteed bound)

- 6 entries x 20 models x 130 pairs = 15600 RC runs at 0.22 s each; 3 replay banks x 130 = 390 replay runs at 0.19 s each; fits 0.03 h.
- Projected total: 1.02 h; storage about 1913.5 MiB.
- Already complete after this check: 20 models and this replay bank; remaining about 0.84 h.

## Revised engineering estimate

Measured on feasible-best: 20 models and one replay bank in 0.07 h; scaled to the six entries and three warm-ups the full panel is at most 1.02 h of wall time and about 1913.5 MiB of run storage in the canonical execution environment, with about 0.84 h remaining after this check; the earlier engineering estimate (plan section 10) stands or is revised accordingly, and M3REP-006 waits for the owner's budget approval (C8).

## Limitations

- The projection multiplies maximum run counts by mean costs measured on one entry; it is not a guaranteed bound: other reservoir sizes, warm-ups, and storage overhead can change the costs, while first-failure stopping lowers them.
- Timings are wall-clock in the canonical single-threaded execution environment of this machine (C10); another core type, thread setting, or machine measures differently.
