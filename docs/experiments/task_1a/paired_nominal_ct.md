# Paired nominal evaluation: `rc+computed_torque` vs `replay+computed_torque`

Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `model-20260831-4999bfe0e0ec.toml`.
Windows: move [1.0, 4.0] s, dwell [4.0, 5.0] s (identical for both runs).

| Run | Method | Run ID | Termination | Success | Failed criteria |
|---|---|---|---|---|---|
| RC | rc+computed_torque | run-20260831-49f320d19a50 | completed | True | - |
| replay | replay+computed_torque | run-20260831-8312074cb958 | completed | True | - |

| Metric | Unit | RC | Replay | RC - replay | RC / replay | Better |
|---|---|---|---|---|---|---|
| joint_rmse | rad | 0.1626 | 1.484e-05 | 0.1626 | 1.096e+04 | lower |
| dwell_in_tolerance_fraction | - | 1 | 1 | 0 | 1 | higher |
| dwell_longest_in_tolerance_s | s | 1 | 1 | 0 | 1 | higher |
| dwell_endpoint_rms | m | 0.0004402 | 5.474e-06 | 0.0004347 | 80.41 | lower |
| dwell_endpoint_max | m | 0.0004402 | 2.368e-05 | 0.0004165 | 18.59 | lower |
| dwell_velocity_max | rad/s | 7.527e-07 | 0.0007137 | -0.0007129 | 0.001055 | lower |
| effort_torque_rms | N*m | 0.08322 | 0.01813 | 0.06509 | 4.59 | lower |
| effort_torque_peak | N*m | 0.6074 | 0.04563 | 0.5617 | 13.31 | lower |
| effort_saturation_fraction | - | 0 | 0 | 0 | - | lower |
| effort | N^2*m^2*s | 0.06939 | 0.003294 | 0.0661 | 21.07 | lower |
| move_coverage | - | 1 | 1 | 0 | 1 | higher |
| dwell_coverage | - | 1 | 1 | 0 | 1 | higher |
