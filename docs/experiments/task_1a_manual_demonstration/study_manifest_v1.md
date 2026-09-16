# Task 1-a manual study manifest (v1)

Experiment `task_1a_manual_v1`: the 186 models of the approved scope (6 inherited configurations crossed with the 31 arms of the locked demonstration bank), frozen before any arm is fitted (manual plan sections 4 and 5, decisions D4 and D5).

## Sources

- Repetition panel `docs/experiments/task_1a_repeated_demonstration/panel_manifest_v1.json` (sha256 `68bc850ef269`), read-only.
- Demonstration bank `docs/experiments/task_1a_manual_demonstration/bank/bank_v1.json` (sha256 `1e7ed86003fc`).
- Task configuration `configs/tasks/task_1a_manual_v2.toml` (`4abdad9b1504`) and preprocessing `configs/preprocessing/manual_v2.toml` (`fbfdde1609db`), both as the bank records them.
- Model configuration `configs/models/esn_task_1a_v4.toml` (`72a55213c7d5`), the file the panel bound: the readout solver and the frozen physical input transform.
- Input transform `fixed_scale` copied from `processed-20260830-feaf73e6663c` (payload `feaf73e6663c`, record `data/records/processed/processed-20260830-feaf73e6663c.toml`); no statistics come from a manual take (I8).
- Training construction: equal_episode weighting with 400 reference rows, the `count_scaled` ridge rule, target `next_q`, washout `warmup_hold`.
- Training validation `configs/tasks/task_1a_manual_v2.toml` (`4abdad9b1504`); contractive seed bank 1; rclib 0.1.0 (`61a29f0ce6fa`).
- Execution identity `79307eb3123c` (policy `p-cores`, canonical); every fit identity binds it (C10).

## Configurations

The inherited reservoir, warm-up, and `alpha_0` of each source trial; the estimator cutoffs are evaluation-side settings and never recipe fields.

| configuration | trial | warm-up (s) | alpha_0 | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | velocity_cutoff_hz | acceleration_cutoff_hz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feasible-best | 17 | 0.25 | 0.1191718250387115 | 100 | 1.263266875517793 | 0.9086681297381796 | 0.06544592224060722 | 0.4162819932167906 | 965 | 6.6694585741072885 | 5.590310542267199 |
| feasible-middle | 136 | 0 | 0.003097590569817245 | 400 | 1.299786136219192 | 0.6405712425492811 | 0.03857698270429774 | 0.2758773237273159 | 922 | 21.853688084124755 | 9.331655895077768 |
| feasible-worst | 53 | 1 | 0.004935486520748805 | 300 | 1.1483563502746237 | 0.8674824469976874 | 0.013523859112187989 | 0.022281641437870696 | 938 | 6.7310817038954465 | 5.1302452892157655 |
| failure-actual-dwell | 1 | 0.25 | 0.6191588897437532 | 100 | 1.1621910014621486 | 0.6097988097560109 | 0.014664019100094813 | 0.2114098851877163 | 340 | 28.457478731077835 | 5.730433263875364 |
| failure-joint-velocity | 0 | 1 | 0.002849478837743603 | 250 | 1.2944675208876626 | 0.9791076284866893 | 0.04089985548951509 | 0.021176881502572638 | 896 | 29.980411525699598 | 10.938122239871603 |
| failure-generated-dwell | 28 | 0 | 0.08365993429880747 | 300 | 1.118277115171816 | 0.7819437758127186 | 0.12632472327232913 | 0.044104986024179345 | 966 | 7.555972499674083 | 20.38331154329677 |

## Arms

Every configuration carries the same arms. `M100` and the fixed-alpha diagnostics stay deferred (D4), so no model trains ten episodes at `alpha_0`.

| arm | models | episodes K | solver alpha | unique sources | copies | synthetic |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| S | 60 | 1 | alpha_0 | 1 | 0 | 0 |
| M10 | 6 | 10 | 10 alpha_0 | 10 | 0 | 0 |
| R10 | 60 | 10 | 10 alpha_0 | 1 | 9 | 0 |
| C10 | 60 | 10 | 10 alpha_0 | 1 | 0 | 9 |

## Demonstrations

Each locked demonstration with the rows it contributes to a fit, its equal-episode weight 400 / L_i, and the frozen contractive bank grown from it.

| position | dataset | loss rows | row weight | dwell onset (s) | bank digest |
| --- | --- | ---: | ---: | ---: | --- |
| D01 | `processed-20260916-a8c94bb35358` | 1664 | 0.240385 | 14.57 | `17e999a294a6` |
| D02 | `processed-20260916-c33a21797eb3` | 1585 | 0.252366 | 13.91 | `3e4bb0ed2d5f` |
| D03 | `processed-20260916-d78ffcea290e` | 1398 | 0.286123 | 12.97 | `c1c0a43882e9` |
| D04 | `processed-20260916-6f06d3c33ea8` | 1287 | 0.3108 | 11.02 | `1685f0672c3f` |
| D05 | `processed-20260916-f5fc3e298697` | 1424 | 0.280899 | 13.08 | `b6e847d7b9bc` |
| D06 | `processed-20260916-bce2e493648f` | 1377 | 0.290487 | 10.08 | `2b65eecac0ae` |
| D07 | `processed-20260916-5ce1d9783a72` | 1364 | 0.293255 | 10.74 | `4f62d144580a` |
| D08 | `processed-20260916-87520abd2930` | 1387 | 0.288392 | 9.34 | `d4bb23e79b8a` |
| D09 | `processed-20260916-227ac2516270` | 1472 | 0.271739 | 10.36 | `9094c3b42fc0` |
| D10 | `processed-20260916-f4044d555347` | 1692 | 0.236407 | 12.88 | `8205a9bcae20` |

## Models

- 186 models: 60 `S`, 6 `M10`, 60 `R10`, 60 `C10`, one per (configuration, arm) pair over 6 configurations.
- Every fit identity binds the arm, the datasets, the transform, the training validation, the ridge scale, the contractive construction where there is one, the pinned rclib revision, and the execution identity; all of them are distinct and re-derived when this manifest is loaded.
- Entries reference this header instead of repeating it: each records its configuration, arm, warm-up, contractive bank, and identities, while its fitted ESN, its digest-bound datasets, and its accounting are rebuilt from the configurations, the bound readout, and the demonstrations above.

## Provenance

- Manifest: commit `d5f18405d389`, created 2026-09-16T11:04:32+00:00, Python 3.12.11, lock `ac9811f8142e`.
- Execution: p-cores on x86_64, Python 3.12.11, recorded 2026-09-16T11:04:27+00:00.
