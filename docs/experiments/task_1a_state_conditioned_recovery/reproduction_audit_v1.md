# Task 1-a recovery reproduction

- Started: 2026-09-04T04:46:03.065168+00:00; elapsed 62.6 s; ok: True.
- Command: `python -m arm_rc_ctrl.experiments.reproduce_recovery --scratch audit-inner --summary reproduction_audit_v1.json --audit reproduction_audit_v1.md`
- Executor machine: `Linux-7.0.0-30-generic-x86_64-with-glibc2.43 / x86_64`.
- Auditor: OpenAI Codex (independent same-host rerun at `0517fd7`; 2026-09-04) (cross-machine execution preferred; plan M3R-020).
- Largest metric deviation: 0.0.

| step | ok | elapsed (s) | detail |
| --- | --- | --- | --- |
| environment | True | 0.3 | 2 build identities verified; submodules ['rclib', 'rtctrl', 'skelarm'] and uv.lock match the evidence |
| storage | True | 0.0 | external storage root resolved |
| records | True | 0.1 | 4 study pointers, ablation, freeze, representative record, dataset processed-20260903-ce343c8ce6a5, raw raw-20260830-b5adde395f1c, and 16 run pointers verified |
| payloads | True | 1.3 | 4 study payloads, dataset, raw, and 16 run payloads verified; trial 17 bound |
| data | True | 0.3 | cropped recovery dataset rebuilt with digest ce343c8ce6a5 (identical) |
| episodes | True | 2.0 | anchor augmentation episodes regenerated with digest 173fcbb82952 (identical) |
| model | True | 50.7 | recipe refitted (fit RMSE 0.00203332 rad); trial 17 re-evaluated over 130 pairs; largest deviation 0.000e+00 |
| pairs | True | 8.0 | 16 runs rerun under the reproduction label; array digests identical; largest metric deviation 0.000e+00 (tolerance 0.000e+00) |
| report | True | 0.1 | ablation, freeze, and recovery reports re-rendered identically |

## Inputs

- dataset: `processed-20260903-ce343c8ce6a5`
- evidence_project_commit: `0e15b6270deb41bd64bb182e4caa705d99eac608`
- raw: `raw-20260830-b5adde395f1c`
- reproduction_project_commit: `7818c52db6e4b2e5fc887b8c2f9ffba4bc4a4471`
- reproduction_project_dirty: `False`
- storage_root: `<configured external root>`

## Environment

- OMP_NUM_THREADS: `1`
- machine: `x86_64`
- platform: `Linux-7.0.0-30-generic-x86_64-with-glibc2.43`
- processor: ``
- python: `3.12.11`
