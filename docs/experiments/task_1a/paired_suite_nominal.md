# Paired suite: generator effect vs tracker effect

Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `model-20260831-038a9b2c8432.toml`.
Generator effect = RC - replay within one tracker; tracker effect = computed torque - PD within one arm.

| Run | Run ID | Termination | Success |
|---|---|---|---|
| replay+pd | run-20260831-e5265c2773a6 | completed | True |
| rc+pd | run-20260831-3fef7abca22a | completed | True |
| replay+computed_torque | run-20260831-86704cde37a8 | completed | True |
| rc+computed_torque | run-20260831-2dce459a5acc | completed | True |

| Metric | Unit | replay+pd | rc+pd | replay+ct | rc+ct | gen. effect (pd) | gen. effect (ct) | tracker effect (replay) | tracker effect (rc) |
|---|---|---|---|---|---|---|---|---|---|
| joint_rmse | rad | 0.000254 | 0.1325 | 1.484e-05 | 0.1714 | 0.1323 | 0.1714 | -0.0002391 | 0.03885 |
| dwell_in_tolerance_fraction | - | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| dwell_longest_in_tolerance_s | s | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| dwell_endpoint_rms | m | 6.093e-06 | 0.0003912 | 5.474e-06 | 0.0004117 | 0.0003851 | 0.0004062 | -6.19e-07 | 2.05e-05 |
| dwell_endpoint_max | m | 2.173e-05 | 0.0004238 | 2.368e-05 | 0.0004117 | 0.0004021 | 0.000388 | 1.95e-06 | -1.211e-05 |
| dwell_velocity_max | rad/s | 0.0009219 | 0.002116 | 0.0007137 | 2.67e-07 | 0.001194 | -0.0007134 | -0.0002082 | -0.002116 |
| effort_torque_rms | N*m | 0.01823 | 0.07196 | 0.01813 | 0.1009 | 0.05374 | 0.08273 | -9.565e-05 | 0.0289 |
| effort_torque_peak | N*m | 0.04584 | 0.3736 | 0.04563 | 0.5795 | 0.3277 | 0.5338 | -0.000208 | 0.2059 |
| effort_saturation_fraction | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| effort | N^2*m^2*s | 0.003329 | 0.05189 | 0.003294 | 0.1019 | 0.04856 | 0.09865 | -3.484e-05 | 0.05005 |
| move_coverage | - | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| dwell_coverage | - | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
