# Task 1-a recovery model freeze (v1)

## Rule

Per posture class and frozen tracker pair independently: median early command-gap ratio < 1, median activation-jump ratio < 1, and both paired metrics improving in at least 15 of 20 scenarios; among eligible models, lexicographic selection on the worst class-by-tracker cell median of the early command-gap ratio, then the worst cell median of endpoint settling time, then the worst cell median of applied-torque RMS; reservoir-seed-panel stability precedes any freeze (recovery plan section 7.3; approved decision D5).

## Inputs

- Dataset `processed-20260903-ce343c8ce6a5`; commit `252c2ea05eca`.
- Development ablation `development_ablation_v1.json` (sha256 `6b327d77e2e4`).

| study | formulation | feasible trials | in the rule | note |
| --- | --- | --- | --- | --- |
| recovery-search-1a-contractive-v1 | contractive | 0 | yes |  |
| recovery-search-1a-no-augmentation-v1 | no_augmentation | 134 | yes |  |
| recovery-search-1a-non-decaying-v1 | non_decaying | 0 | yes |  |
| recovery-search-1a-residual-v1 | residual | 0 | no | exploratory per D4; retained as a negative, never predeclared for confirmatory inclusion |

## Outcome

- Candidates evaluated: 134; eligible under section 7.3: 0.
- **NEGATIVE RESULT — no model is frozen.** No feasible trial satisfies every class-by-tracker cell of the eligibility rule, so there is no selection and no reservoir-seed panel.
- The confirmatory gate stays closed: the locked suite must not run without an eligible frozen model or an owner-approved protocol revision (a new protocol version per D5).
