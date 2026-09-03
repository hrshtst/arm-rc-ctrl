# Recovery search `recovery-search-1a-contractive-v1`

- Formulation `contractive`; protocol `configs/studies/recovery_search_1a_contractive_v1.toml` (digest `67e8346013b5`), dataset `processed-20260903-ce343c8ce6a5`.
- Frozen trackers: `computed_torque` (digest `0ac3dff977cd`), `pd_v2` (digest `45f6e7a31490`).
- Budget 500; stored 500 trials (500 complete, 0 pruned); 0 feasible; this invocation ran 500.
- Provenance: commit `b7d29fe891cd`, sampler seed 20270103.

## Selection

No feasible completed trial: nothing is selected.
## Comparison points

| label | trial | objective | feasible | reason |
| --- | --- | --- | --- | --- |
| anchor-v4-tw1 | 0 | 10 | False | scenario 0 [pd_v2]: dwell:dwell_stationary |

## Best feasible trials

| trial | objective | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | alpha | velocity_cutoff_hz | acceleration_cutoff_hz | warmup_s | n_synthetic | sigma_rad | phi | gamma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Infeasible and pruned trials by reason

| reason | trials |
| --- | --- |
| limit_violation:joint_velocity | 221 |
| dwell:dwell_in_tolerance,dwell_stationary | 138 |
| dwell:dwell_stationary | 79 |
| dwell:dwell_in_tolerance | 55 |
| early_termination:invalid_output:bounds | 4 |
| generated_dwell:generated_dwell_stationary | 3 |
