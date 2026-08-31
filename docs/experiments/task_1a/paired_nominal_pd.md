# Paired nominal evaluation: `rc+pd` vs `replay+pd`

Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `model-20260831-4999bfe0e0ec.toml`.
Windows: move [1.0, 4.0] s, dwell [4.0, 5.0] s (identical for both runs).

| Run | Method | Run ID | Termination | Success | Failed criteria |
|---|---|---|---|---|---|
| RC | rc+pd | run-20260831-6bddfc9e086d | completed | True | - |
| replay | replay+pd | run-20260831-0007efbfaef0 | completed | True | - |

| Metric | Unit | RC | Replay | RC - replay | RC / replay | Better |
|---|---|---|---|---|---|---|
| joint_rmse | rad | 0.1265 | 0.000254 | 0.1262 | 497.9 | lower |
| dwell_in_tolerance_fraction | - | 1 | 1 | 0 | 1 | higher |
| dwell_longest_in_tolerance_s | s | 1 | 1 | 0 | 1 | higher |
| dwell_endpoint_rms | m | 0.0004284 | 6.093e-06 | 0.0004223 | 70.31 | lower |
| dwell_endpoint_max | m | 0.000494 | 2.173e-05 | 0.0004722 | 22.73 | lower |
| dwell_velocity_max | rad/s | 0.002023 | 0.0009219 | 0.001101 | 2.195 | lower |
| effort_torque_rms | N*m | 0.06436 | 0.01823 | 0.04613 | 3.531 | lower |
| effort_torque_peak | N*m | 0.3247 | 0.04584 | 0.2788 | 7.083 | lower |
| effort_saturation_fraction | - | 0 | 0 | 0 | - | lower |
| effort | N^2*m^2*s | 0.0415 | 0.003329 | 0.03817 | 12.47 | lower |
| move_coverage | - | 1 | 1 | 0 | 1 | higher |
| dwell_coverage | - | 1 | 1 | 0 | 1 | higher |
