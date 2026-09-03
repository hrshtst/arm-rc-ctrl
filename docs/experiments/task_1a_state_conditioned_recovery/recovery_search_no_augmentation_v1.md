# Recovery search `recovery-search-1a-no-augmentation-v1`

- Formulation `no_augmentation`; protocol `configs/studies/recovery_search_1a_no_augmentation_v1.toml` (digest `d3d3d087187d`), dataset `processed-20260903-ce343c8ce6a5`.
- Frozen trackers: `computed_torque` (digest `0ac3dff977cd`), `pd_v2` (digest `45f6e7a31490`).
- Budget 500; stored 500 trials (500 complete, 0 pruned); 134 feasible; this invocation ran 498.
- Provenance: commit `b7d29fe891cd`, sampler seed 20270101.

## Selection

Trial 17 with objective 0.6704 (worst class-by-tracker cell median of the early command-gap ratio).

| cell | median gap ratio |
| --- | --- |
| posture_large:computed_torque | 0.2335 |
| posture_large:pd_v2 | 0.5288 |
| posture_small:computed_torque | 0.3123 |
| posture_small:pd_v2 | 0.6704 |

| parameter | value |
| --- | --- |
| n_neurons | 100 |
| spectral_radius | 1.263266875517793 |
| sparsity | 0.9086681297381796 |
| leak_rate | 0.06544592224060722 |
| input_scaling | 0.4162819932167906 |
| seed | 965 |
| alpha | 0.1191718250387115 |
| velocity_cutoff_hz | 6.6694585741072885 |
| acceleration_cutoff_hz | 5.590310542267199 |
| warmup_s | 0.25 |

Development pairs of the selected trial:

| # | scenario | tracker | kind | gap ratio |
| --- | --- | --- | --- | --- |
| 0 | nominal | pd_v2 | nominal | n/a |
| 1 | nominal | computed_torque | nominal | n/a |
| 2 | posture-small-20261201-00 | pd_v2 | posture_small | 1.198 |
| 3 | posture-small-20261201-00 | computed_torque | posture_small | 0.4479 |
| 4 | posture-small-20261201-01 | pd_v2 | posture_small | 1.525 |
| 5 | posture-small-20261201-01 | computed_torque | posture_small | 0.3586 |
| 6 | posture-small-20261201-02 | pd_v2 | posture_small | 0.3483 |
| 7 | posture-small-20261201-02 | computed_torque | posture_small | 0.2527 |
| 8 | posture-small-20261201-03 | pd_v2 | posture_small | 0.8714 |
| 9 | posture-small-20261201-03 | computed_torque | posture_small | 0.2894 |
| 10 | posture-small-20261202-00 | pd_v2 | posture_small | 0.5536 |
| 11 | posture-small-20261202-00 | computed_torque | posture_small | 0.3118 |
| 12 | posture-small-20261202-01 | pd_v2 | posture_small | 1.469 |
| 13 | posture-small-20261202-01 | computed_torque | posture_small | 0.4448 |
| 14 | posture-small-20261202-02 | pd_v2 | posture_small | 0.4526 |
| 15 | posture-small-20261202-02 | computed_torque | posture_small | 0.2931 |
| 16 | posture-small-20261202-03 | pd_v2 | posture_small | 0.3982 |
| 17 | posture-small-20261202-03 | computed_torque | posture_small | 0.233 |
| 18 | posture-small-20261203-00 | pd_v2 | posture_small | 0.3321 |
| 19 | posture-small-20261203-00 | computed_torque | posture_small | 0.2331 |
| 20 | posture-small-20261203-01 | pd_v2 | posture_small | 1.831 |
| 21 | posture-small-20261203-01 | computed_torque | posture_small | 0.3981 |
| 22 | posture-small-20261203-02 | pd_v2 | posture_small | 0.5472 |
| 23 | posture-small-20261203-02 | computed_torque | posture_small | 0.298 |
| 24 | posture-small-20261203-03 | pd_v2 | posture_small | 1.386 |
| 25 | posture-small-20261203-03 | computed_torque | posture_small | 0.4498 |
| 26 | posture-small-20261204-00 | pd_v2 | posture_small | 0.5238 |
| 27 | posture-small-20261204-00 | computed_torque | posture_small | 0.2472 |
| 28 | posture-small-20261204-01 | pd_v2 | posture_small | 0.3325 |
| 29 | posture-small-20261204-01 | computed_torque | posture_small | 0.2325 |
| 30 | posture-small-20261204-02 | pd_v2 | posture_small | 0.7194 |
| 31 | posture-small-20261204-02 | computed_torque | posture_small | 0.2716 |
| 32 | posture-small-20261204-03 | pd_v2 | posture_small | 1.483 |
| 33 | posture-small-20261204-03 | computed_torque | posture_small | 0.3543 |
| 34 | posture-small-20261205-00 | pd_v2 | posture_small | 1.252 |
| 35 | posture-small-20261205-00 | computed_torque | posture_small | 0.4506 |
| 36 | posture-small-20261205-01 | pd_v2 | posture_small | 0.5608 |
| 37 | posture-small-20261205-01 | computed_torque | posture_small | 0.3128 |
| 38 | posture-small-20261205-02 | pd_v2 | posture_small | 1.515 |
| 39 | posture-small-20261205-02 | computed_torque | posture_small | 0.441 |
| 40 | posture-small-20261205-03 | pd_v2 | posture_small | 0.6214 |
| 41 | posture-small-20261205-03 | computed_torque | posture_small | 0.3176 |
| 42 | posture-large-20261201-00 | pd_v2 | posture_large | 0.3804 |
| 43 | posture-large-20261201-00 | computed_torque | posture_large | 0.2746 |
| 44 | posture-large-20261201-01 | pd_v2 | posture_large | 0.6381 |
| 45 | posture-large-20261201-01 | computed_torque | posture_large | 0.2184 |
| 46 | posture-large-20261201-02 | pd_v2 | posture_large | 0.5088 |
| 47 | posture-large-20261201-02 | computed_torque | posture_large | 0.2595 |
| 48 | posture-large-20261201-03 | pd_v2 | posture_large | 0.2578 |
| 49 | posture-large-20261201-03 | computed_torque | posture_large | 0.162 |
| 50 | posture-large-20261202-00 | pd_v2 | posture_large | 0.7187 |
| 51 | posture-large-20261202-00 | computed_torque | posture_large | 0.2313 |
| 52 | posture-large-20261202-01 | pd_v2 | posture_large | 0.2992 |
| 53 | posture-large-20261202-01 | computed_torque | posture_large | 0.1821 |
| 54 | posture-large-20261202-02 | pd_v2 | posture_large | 0.3969 |
| 55 | posture-large-20261202-02 | computed_torque | posture_large | 0.2357 |
| 56 | posture-large-20261202-03 | pd_v2 | posture_large | 1.413 |
| 57 | posture-large-20261202-03 | computed_torque | posture_large | 0.4515 |
| 58 | posture-large-20261203-00 | pd_v2 | posture_large | 1.149 |
| 59 | posture-large-20261203-00 | computed_torque | posture_large | 0.2986 |
| 60 | posture-large-20261203-01 | pd_v2 | posture_large | 0.1979 |
| 61 | posture-large-20261203-01 | computed_torque | posture_large | 0.1798 |
| 62 | posture-large-20261203-02 | pd_v2 | posture_large | 1.271 |
| 63 | posture-large-20261203-02 | computed_torque | posture_large | 0.3505 |
| 64 | posture-large-20261203-03 | pd_v2 | posture_large | 0.5489 |
| 65 | posture-large-20261203-03 | computed_torque | posture_large | 0.203 |
| 66 | posture-large-20261204-00 | pd_v2 | posture_large | 0.3287 |
| 67 | posture-large-20261204-00 | computed_torque | posture_large | 0.2073 |
| 68 | posture-large-20261204-01 | pd_v2 | posture_large | 1.172 |
| 69 | posture-large-20261204-01 | computed_torque | posture_large | 0.3162 |
| 70 | posture-large-20261204-02 | pd_v2 | posture_large | 0.2007 |
| 71 | posture-large-20261204-02 | computed_torque | posture_large | 0.1768 |
| 72 | posture-large-20261204-03 | pd_v2 | posture_large | 0.4933 |
| 73 | posture-large-20261204-03 | computed_torque | posture_large | 0.4274 |
| 74 | posture-large-20261205-00 | pd_v2 | posture_large | 1.385 |
| 75 | posture-large-20261205-00 | computed_torque | posture_large | 0.4513 |
| 76 | posture-large-20261205-01 | pd_v2 | posture_large | 0.282 |
| 77 | posture-large-20261205-01 | computed_torque | posture_large | 0.1739 |
| 78 | posture-large-20261205-02 | pd_v2 | posture_large | 0.604 |
| 79 | posture-large-20261205-02 | computed_torque | posture_large | 0.3146 |
| 80 | posture-large-20261205-03 | pd_v2 | posture_large | 0.5742 |
| 81 | posture-large-20261205-03 | computed_torque | posture_large | 0.2075 |
| 82 | force-12N-000deg | pd_v2 | force | n/a |
| 83 | force-12N-000deg | computed_torque | force | n/a |
| 84 | force-12N-090deg | pd_v2 | force | n/a |
| 85 | force-12N-090deg | computed_torque | force | n/a |
| 86 | force-12N-180deg | pd_v2 | force | n/a |
| 87 | force-12N-180deg | computed_torque | force | n/a |
| 88 | force-12N-270deg | pd_v2 | force | n/a |
| 89 | force-12N-270deg | computed_torque | force | n/a |
| 90 | combined-20261201-00-000deg | pd_v2 | combined | n/a |
| 91 | combined-20261201-00-000deg | computed_torque | combined | n/a |
| 92 | combined-20261201-01-090deg | pd_v2 | combined | n/a |
| 93 | combined-20261201-01-090deg | computed_torque | combined | n/a |
| 94 | combined-20261201-02-180deg | pd_v2 | combined | n/a |
| 95 | combined-20261201-02-180deg | computed_torque | combined | n/a |
| 96 | combined-20261201-03-270deg | pd_v2 | combined | n/a |
| 97 | combined-20261201-03-270deg | computed_torque | combined | n/a |
| 98 | combined-20261202-00-000deg | pd_v2 | combined | n/a |
| 99 | combined-20261202-00-000deg | computed_torque | combined | n/a |
| 100 | combined-20261202-01-090deg | pd_v2 | combined | n/a |
| 101 | combined-20261202-01-090deg | computed_torque | combined | n/a |
| 102 | combined-20261202-02-180deg | pd_v2 | combined | n/a |
| 103 | combined-20261202-02-180deg | computed_torque | combined | n/a |
| 104 | combined-20261202-03-270deg | pd_v2 | combined | n/a |
| 105 | combined-20261202-03-270deg | computed_torque | combined | n/a |
| 106 | combined-20261203-00-000deg | pd_v2 | combined | n/a |
| 107 | combined-20261203-00-000deg | computed_torque | combined | n/a |
| 108 | combined-20261203-01-090deg | pd_v2 | combined | n/a |
| 109 | combined-20261203-01-090deg | computed_torque | combined | n/a |
| 110 | combined-20261203-02-180deg | pd_v2 | combined | n/a |
| 111 | combined-20261203-02-180deg | computed_torque | combined | n/a |
| 112 | combined-20261203-03-270deg | pd_v2 | combined | n/a |
| 113 | combined-20261203-03-270deg | computed_torque | combined | n/a |
| 114 | combined-20261204-00-000deg | pd_v2 | combined | n/a |
| 115 | combined-20261204-00-000deg | computed_torque | combined | n/a |
| 116 | combined-20261204-01-090deg | pd_v2 | combined | n/a |
| 117 | combined-20261204-01-090deg | computed_torque | combined | n/a |
| 118 | combined-20261204-02-180deg | pd_v2 | combined | n/a |
| 119 | combined-20261204-02-180deg | computed_torque | combined | n/a |
| 120 | combined-20261204-03-270deg | pd_v2 | combined | n/a |
| 121 | combined-20261204-03-270deg | computed_torque | combined | n/a |
| 122 | combined-20261205-00-000deg | pd_v2 | combined | n/a |
| 123 | combined-20261205-00-000deg | computed_torque | combined | n/a |
| 124 | combined-20261205-01-090deg | pd_v2 | combined | n/a |
| 125 | combined-20261205-01-090deg | computed_torque | combined | n/a |
| 126 | combined-20261205-02-180deg | pd_v2 | combined | n/a |
| 127 | combined-20261205-02-180deg | computed_torque | combined | n/a |
| 128 | combined-20261205-03-270deg | pd_v2 | combined | n/a |
| 129 | combined-20261205-03-270deg | computed_torque | combined | n/a |

## Comparison points

| label | trial | objective | feasible | reason |
| --- | --- | --- | --- | --- |
| anchor-v4-tw1 | 0 | 10 | False | scenario 23 [pd_v2]: limit_violation:joint_velocity |

## Best feasible trials

| trial | objective | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | alpha | velocity_cutoff_hz | acceleration_cutoff_hz | warmup_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | 0.6704 | 100 | 1.263 | 0.9087 | 0.06545 | 0.4163 | 965 | 0.1192 | 6.669 | 5.59 | 0.25 |
| 400 | 0.7438 | 100 | 1.068 | 0.59 | 0.08967 | 0.07996 | 634 | 0.005252 | 15.42 | 5.823 | 0.25 |
| 30 | 0.8153 | 350 | 0.9867 | 0.5968 | 0.02663 | 0.1068 | 852 | 0.07998 | 18.67 | 5.699 | 1 |
| 107 | 0.8288 | 100 | 1.217 | 0.9502 | 0.02464 | 0.3263 | 940 | 0.002482 | 17.43 | 7.092 | 0.25 |
| 478 | 0.8559 | 350 | 1.056 | 0.5927 | 0.1174 | 0.1223 | 728 | 0.007286 | 6.977 | 7.262 | 0 |
| 474 | 0.8593 | 350 | 1.053 | 0.5693 | 0.09046 | 0.112 | 712 | 0.00498 | 6.934 | 7.342 | 0 |
| 307 | 0.8601 | 350 | 1.102 | 0.5368 | 0.1077 | 0.174 | 737 | 0.005692 | 15.45 | 5.325 | 0 |
| 318 | 0.8611 | 350 | 0.9557 | 0.83 | 0.09672 | 0.1898 | 639 | 0.004698 | 16.85 | 5.584 | 0 |
| 477 | 0.8628 | 350 | 1.059 | 0.5735 | 0.1096 | 0.121 | 730 | 0.005134 | 6.131 | 7.483 | 0 |
| 277 | 0.8656 | 350 | 0.9351 | 0.5682 | 0.08599 | 0.08676 | 730 | 0.005753 | 17.21 | 8.046 | 0 |

## Infeasible and pruned trials by reason

| reason | trials |
| --- | --- |
| limit_violation:joint_velocity | 148 |
| dwell:dwell_stationary | 110 |
| dwell:dwell_in_tolerance,dwell_stationary | 57 |
| dwell:dwell_in_tolerance | 33 |
| generated_dwell:generated_dwell_stationary | 18 |
