# Task 1-a recovery development ablation (v1)

## Summary

- Dataset `processed-20260903-ce343c8ce6a5`; commit `835541c134ff`.
- Matched studies (identical budgets, scenarios, trackers, limits, metrics): `recovery-search-1a-contractive-v1`, `recovery-search-1a-no-augmentation-v1`, `recovery-search-1a-non-decaying-v1`.
- Eligible candidates under the section 7.3 rule: 0 of 134 feasible trials (ceil(0.75 * n) scenarios improving both paired metrics (15 of 20 per plan section 7.3)).

## Arms

| study | formulation | budget | stored | feasible | best trial | best worst-cell gap ratio |
| --- | --- | --- | --- | --- | --- | --- |
| recovery-search-1a-contractive-v1 | contractive | 500 | 500 | 0 | none | n/a |
| recovery-search-1a-no-augmentation-v1 | no_augmentation | 500 | 500 | 134 | 17 | 0.6704 |
| recovery-search-1a-non-decaying-v1 | non_decaying | 500 | 500 | 0 | none | n/a |

## Failure taxonomy

First failing gate of every infeasible trial (complete per-trial records live in the study
reports and their Optuna databases; nothing is discarded):

- `recovery-search-1a-contractive-v1`: Anchor `anchor-v4-tw1`: scenario 0 [pd_v2]: dwell:dwell_stationary.
    - dwell: 272
    - limit_violation:joint_velocity: 221
    - early_termination: 4
    - generated_dwell: 3
- `recovery-search-1a-no-augmentation-v1`: Anchor `anchor-v4-tw1`: scenario 23 [pd_v2]: limit_violation:joint_velocity.
    - dwell: 200
    - limit_violation:joint_velocity: 148
    - generated_dwell: 18
- `recovery-search-1a-non-decaying-v1`: Anchor `anchor-v4-tw1`: scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary.
    - limit_violation:joint_velocity: 258
    - dwell: 229
    - early_termination: 13

## Timing-only arm

Feasible trials by warm-up (D2 asks for the shortest duration passing the common gates):

| warm-up (s) | feasible trials |
| --- | --- |
| 0 | 123 |
| 0.25 | 9 |
| 1 | 2 |

## Eligible candidates (section 7.3)

Per class-by-tracker cell: median early command-gap ratio < 1, median activation-jump ratio < 1 (jump ratios re-derived from the trial-independent replay baselines), and improvement of both metrics in at least 15 of 20 scenarios.

No feasible trial satisfies every cell; this negative result is retained as-is.

## Limitations

- Synthetic-sample-count confound: the augmented arms train on 1 + N_aug episodes (17-65) against the timing-only arm's single demonstration, so augmentation family and training-set size change together; the matched grids bound but do not remove this confound.
- First-infeasible censoring: an infeasible trial stops at its first failing (scenario, tracker) pair, so the reason taxonomy counts first failures, not all failures a full sweep would find.
- Single scripted demonstration (approved decision D6): results do not establish a basin of attraction outside the augmented training tube, and a negative augmentation result here is valid evidence.
- Development data only: no confirmatory seed, level, or outcome was read, and the section 7.3 gates and their ordering are unchanged; candidate identification here does not select a model (M3R-015 does).
