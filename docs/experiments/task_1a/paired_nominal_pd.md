# Paired nominal evaluation: `rc+pd` vs `replay+pd`

Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `model-20260831-038a9b2c8432.toml`.
Windows: move [1.0, 4.0] s, dwell [4.0, 5.0] s (identical for both runs).

| Run | Method | Run ID | Termination | Success | Failed criteria |
|---|---|---|---|---|---|
| RC | rc+pd | run-20260831-3fef7abca22a | completed | True | - |
| replay | replay+pd | run-20260831-e5265c2773a6 | completed | True | - |

| Metric | Unit | RC | Replay | RC - replay | RC / replay | Better |
|---|---|---|---|---|---|---|
| joint_rmse | rad | 0.1325 | 0.000254 | 0.1323 | 521.9 | lower |
| dwell_in_tolerance_fraction | - | 1 | 1 | 0 | 1 | higher |
| dwell_longest_in_tolerance_s | s | 1 | 1 | 0 | 1 | higher |
| dwell_endpoint_rms | m | 0.0003912 | 6.093e-06 | 0.0003851 | 64.2 | lower |
| dwell_endpoint_max | m | 0.0004238 | 2.173e-05 | 0.0004021 | 19.5 | lower |
| dwell_velocity_max | rad/s | 0.002116 | 0.0009219 | 0.001194 | 2.295 | lower |
| effort_torque_rms | N*m | 0.07196 | 0.01823 | 0.05374 | 3.948 | lower |
| effort_torque_peak | N*m | 0.3736 | 0.04584 | 0.3277 | 8.15 | lower |
| effort_saturation_fraction | - | 0 | 0 | 0 | - | lower |
| effort | N^2*m^2*s | 0.05189 | 0.003329 | 0.04856 | 15.59 | lower |
| move_coverage | - | 1 | 1 | 0 | 1 | higher |
| dwell_coverage | - | 1 | 1 | 0 | 1 | higher |
