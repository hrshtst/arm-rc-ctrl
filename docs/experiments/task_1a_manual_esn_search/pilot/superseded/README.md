# Superseded timing-pilot report

A report that was issued is kept. `timing_pilot_v1.{json,md}` was generated on
2026-09-23 from the ten-trial pilot run at `5a6cde2` and committed in `7bd514c`.
It is not the report of record: `../timing_pilot_v2.{json,md}` is. Both were
generated from the same stored records and the same
[preflight](../preflight_v1.json), and nothing here is edited.

## `timing_pilot_v1` — superseded by v2

The owner's review of M3MS-004 found two P2 defects in the report generator
and one wording error; see the [review rounds](../../review_rounds.md).

- The per-run projection divided the measured simulation time and bytes by
  every recorded run, including runs served from the store without being
  simulated. A cached result would therefore have made runs look cheaper.
- A trial abandoned before any parameter was recorded has an outcome but no
  reservation, and the generator could not report it.
- It called the comparison figure a "lower bound". Nominal runs do not bound
  the cost of perturbed ones, so it is an estimate.

Neither defect occurred in this pilot: every trial had a reservation and every
run was simulated. v2 therefore carries the same counts, timings and
projections as v1. Its rows add each trial's `simulated_runs`, and it words
the comparison figure as an estimated runs-and-fits cost.
