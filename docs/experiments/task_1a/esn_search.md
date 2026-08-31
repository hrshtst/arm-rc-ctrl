# ESN search `esn-search-1a`

- Protocol: `configs/studies/esn_search_1a.toml` (digest `255ff4876c5f`), dataset `processed-20260830-feaf73e6663c`,
  tracker `pd_v2` (digest `45f6e7a31490`).
- Budget 120; stored 120 trials (110 complete, 10 pruned); 1 feasible; this invocation ran 120.
- Provenance: commit `732dc01f3f60`, sampler seed 20260921.

## Selection

Trial 13 with objective 0.02165 rad (median movement-window joint RMSE).

| parameter | value |
| --- | --- |
| n_neurons | 200 |
| spectral_radius | 1.0919869800080664 |
| sparsity | 0.654914582763362 |
| leak_rate | 0.05869162144717609 |
| input_scaling | 0.13930936236831867 |
| seed | 774 |
| alpha | 0.015036411476169492 |
| velocity_cutoff_hz | 14.175590436732154 |
| acceleration_cutoff_hz | 10.612032951422595 |

Development metrics of the selected trial:

| scenario | kind | movement RMSE (rad) | saturation | criteria |
| --- | --- | --- | --- | --- |
| 0 | posture | 0.0123 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 1 | posture | 0.03577 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 2 | posture | 0.02247 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 3 | posture | 0.01521 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 4 | posture | 0.009559 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 5 | posture | 0.03081 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 6 | posture | 0.01535 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 7 | posture | 0.02672 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 8 | posture | 0.01411 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 9 | posture | 0.05785 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 10 | posture | 0.05646 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 11 | posture | 0.01797 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 12 | posture | 0.008014 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 13 | posture | 0.04886 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 14 | posture | 0.03911 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 15 | posture | 0.0408 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 16 | posture | 0.03434 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 17 | force | 0.01387 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 18 | force | 0.02285 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 19 | force | 0.01748 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 20 | force | 0.00978 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 21 | force | 0.01653 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 22 | force | 0.02827 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 23 | force | 0.02165 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 24 | force | 0.01481 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |

## Comparison points

| label | trial | objective (rad) | feasible | reason |
| --- | --- | --- | --- | --- |
| anchor-alpha-1e-2 | 0 | 10 | False | scenario 2: dwell:dwell_in_tolerance |
| anchor-alpha-3e-2 | 1 | 10 | False | scenario 2: dwell:dwell_in_tolerance |
| anchor-alpha-1e-1 | 2 | 10 | False | scenario 2: dwell:dwell_in_tolerance |
| anchor-alpha-3e-1 | 3 | 10 | False | scenario 2: dwell:dwell_in_tolerance |

## Best feasible trials

| trial | objective (rad) | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | alpha | velocity_cutoff_hz | acceleration_cutoff_hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 0.02165 | 200 | 1.092 | 0.6549 | 0.05869 | 0.1393 | 774 | 0.01504 | 14.18 | 10.61 |

## Infeasible and pruned trials by reason

| reason | trials |
| --- | --- |
| dwell:dwell_in_tolerance | 55 |
| dwell:dwell_in_tolerance,dwell_stationary | 25 |
| dwell:dwell_stationary | 15 |
| limit_violation:joint_velocity | 14 |
| stopped by the pruner | 10 |
