# Paired suite: generator effect vs tracker effect

Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `model-20260831-4999bfe0e0ec.toml`.
Generator effect = RC - replay within one tracker; tracker effect = computed torque - PD within one arm.

| Run | Run ID | Termination | Success |
|---|---|---|---|
| replay+pd | run-20260831-0007efbfaef0 | completed | True |
| rc+pd | run-20260831-6bddfc9e086d | completed | True |
| replay+computed_torque | run-20260831-8312074cb958 | completed | True |
| rc+computed_torque | run-20260831-49f320d19a50 | completed | True |

| Metric | Unit | replay+pd | rc+pd | replay+ct | rc+ct | gen. effect (pd) | gen. effect (ct) | tracker effect (replay) | tracker effect (rc) |
|---|---|---|---|---|---|---|---|---|---|
| joint_rmse | rad | 0.000254 | 0.1265 | 1.484e-05 | 0.1626 | 0.1262 | 0.1626 | -0.0002391 | 0.03616 |
| dwell_in_tolerance_fraction | - | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| dwell_longest_in_tolerance_s | s | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| dwell_endpoint_rms | m | 6.093e-06 | 0.0004284 | 5.474e-06 | 0.0004402 | 0.0004223 | 0.0004347 | -6.19e-07 | 1.177e-05 |
| dwell_endpoint_max | m | 2.173e-05 | 0.000494 | 2.368e-05 | 0.0004402 | 0.0004722 | 0.0004165 | 1.95e-06 | -5.38e-05 |
| dwell_velocity_max | rad/s | 0.0009219 | 0.002023 | 0.0007137 | 7.527e-07 | 0.001101 | -0.0007129 | -0.0002082 | -0.002022 |
| effort_torque_rms | N*m | 0.01823 | 0.06436 | 0.01813 | 0.08322 | 0.04613 | 0.06509 | -9.565e-05 | 0.01886 |
| effort_torque_peak | N*m | 0.04584 | 0.3247 | 0.04563 | 0.6074 | 0.2788 | 0.5617 | -0.000208 | 0.2827 |
| effort_saturation_fraction | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| effort | N^2*m^2*s | 0.003329 | 0.0415 | 0.003294 | 0.06939 | 0.03817 | 0.0661 | -3.484e-05 | 0.02789 |
| move_coverage | - | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| dwell_coverage | - | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
