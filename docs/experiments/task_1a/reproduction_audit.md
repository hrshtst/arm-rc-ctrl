# Task 1-a reproduction audit

- Auditor: (to be filled by the auditor)
- Started: 2026-08-31T09:01:08.440591+00:00
- Command: `python -m arm_rc_ctrl.experiments.reproduce_1a --scratch reproduce --summary docs/experiments/task_1a/reproduction_audit.json --audit docs/experiments/task_1a/reproduction_audit.md`
- Elapsed: 131.1 s
- Outcome: PASS
- Largest metric deviation of the confirmatory rerun: 0.000e+00

## Environment

- OMP_NUM_THREADS: 1
- machine: x86_64
- platform: Linux-7.0.0-30-generic-x86_64-with-glibc2.43
- processor: -
- python: 3.12.11

## Inputs

- confirmatory_report: `robustness_confirmatory_v2_recipe_v4.json`
- lock_sha256: `ff69c024ff5fc381434e4758ca5ee10dc6923362b813785608a8171a097b5825`
- processed: `data/records/processed/processed-20260830-feaf73e6663c.toml`
- project_commit: `19b9af3f5540f95cb594eb19b9ab6fa4e32b9fac`
- raw: `raw-20260830-b5adde395f1c`
- recipe: `data/records/models/model-20260831-1b9477aaa246.toml`
- storage_root: `<configured external root>`

## Steps

| step | outcome | elapsed (s) | detail |
| --- | --- | --- | --- |
| environment | ok | 0.26 | 2 build identities verified; submodules ['rclib', 'rtctrl', 'skelarm'] and uv.lock match the evidence |
| storage | ok | 0.01 | external storage root resolved |
| records | ok | 0.08 | recipe esn-task-1a-v4, dataset processed-20260830-feaf73e6663c, raw raw-20260830-b5adde395f1c, 260 run pointers |
| payloads | ok | 0.11 | raw, processed, and 260 run payloads verified against their recorded digests |
| data | ok | 0.20 | processed dataset rebuilt with digest feaf73e6663c (identical) |
| model | ok | 0.08 | recipe refitted; fit RMSE 0.00173564 rad reproduced |
| evaluation | ok | 129.96 | 260 runs re-evaluated; largest deviation 0.000e+00 (tolerance 0.000e+00) |
| report | ok | 0.43 | report re-rendered identically from the committed evidence |
