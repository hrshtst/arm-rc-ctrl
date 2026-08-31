# Paired nominal evaluation: `rc+computed_torque` vs `replay+computed_torque`

Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `model-20260831-038a9b2c8432.toml`.
Windows: move [1.0, 4.0] s, dwell [4.0, 5.0] s (identical for both runs).

| Run | Method | Run ID | Termination | Success | Failed criteria |
|---|---|---|---|---|---|
| RC | rc+computed_torque | run-20260831-2dce459a5acc | completed | True | - |
| replay | replay+computed_torque | run-20260831-86704cde37a8 | completed | True | - |

| Metric | Unit | RC | Replay | RC - replay | RC / replay | Better |
|---|---|---|---|---|---|---|
| joint_rmse | rad | 0.1714 | 1.484e-05 | 0.1714 | 1.155e+04 | lower |
| dwell_in_tolerance_fraction | - | 1 | 1 | 0 | 1 | higher |
| dwell_longest_in_tolerance_s | s | 1 | 1 | 0 | 1 | higher |
| dwell_endpoint_rms | m | 0.0004117 | 5.474e-06 | 0.0004062 | 75.21 | lower |
| dwell_endpoint_max | m | 0.0004117 | 2.368e-05 | 0.000388 | 17.38 | lower |
| dwell_velocity_max | rad/s | 2.67e-07 | 0.0007137 | -0.0007134 | 0.0003742 | lower |
| effort_torque_rms | N*m | 0.1009 | 0.01813 | 0.08273 | 5.563 | lower |
| effort_torque_peak | N*m | 0.5795 | 0.04563 | 0.5338 | 12.7 | lower |
| effort_saturation_fraction | - | 0 | 0 | 0 | - | lower |
| effort | N^2*m^2*s | 0.1019 | 0.003294 | 0.09865 | 30.95 | lower |
| move_coverage | - | 1 | 1 | 0 | 1 | higher |
| dwell_coverage | - | 1 | 1 | 0 | 1 | higher |
