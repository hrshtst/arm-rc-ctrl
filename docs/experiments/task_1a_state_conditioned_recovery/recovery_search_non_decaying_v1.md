# Recovery search `recovery-search-1a-non-decaying-v1`

- Formulation `non_decaying`; protocol `configs/studies/recovery_search_1a_non_decaying_v1.toml` (digest `735d8bf6aea5`), dataset `processed-20260903-ce343c8ce6a5`.
- Frozen trackers: `computed_torque` (digest `0ac3dff977cd`), `pd_v2` (digest `45f6e7a31490`).
- Budget 500; stored 500 trials (500 complete, 0 pruned); 0 feasible; this invocation ran 500.
- Provenance: commit `b7d29fe891cd`, sampler seed 20270102.

## Selection

No feasible completed trial: nothing is selected.
## Comparison points

| label | trial | objective | feasible | reason |
| --- | --- | --- | --- | --- |
| anchor-v4-tw1 | 0 | 10 | False | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |

## Best feasible trials

| trial | objective | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | alpha | velocity_cutoff_hz | acceleration_cutoff_hz | warmup_s | n_synthetic | sigma_rad | phi | gamma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Infeasible and pruned trials by reason

| reason | trials |
| --- | --- |
| limit_violation:joint_velocity | 258 |
| dwell:dwell_in_tolerance,dwell_stationary | 124 |
| dwell:dwell_stationary | 80 |
| dwell:dwell_in_tolerance | 25 |
| early_termination:invalid_output:bounds | 13 |
