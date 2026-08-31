# Robustness suite `task-1a-robustness-dev-v2` (development)

- Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `esn-task-1a-v3`
  (`data/records/models/model-20260831-ea83321eeaa5.toml`), estimator cutoffs 14.18/10.61 Hz.
- 23 scenarios x 4 arms = 92 runs;
  protocol `configs/evaluations/task_1a_robustness_dev_v2.toml`; commit `2c57244dce8a`.

## Outcomes by arm and class

| arm | class | n | completed | successes | failures | joint RMSE median (rad) | joint RMSE max | saturation max | dwell in-tolerance median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rc+pd_v2 | nominal | 1 | 1 | 1 | - | 0.0123 | 0.0123 | 0 | 1 |
| rc+pd_v2 | posture_small | 6 | 6 | 6 | - | 0.02096 | 0.02698 | 0 | 1 |
| rc+pd_v2 | posture_large | 6 | 6 | 6 | - | 0.04884 | 0.06293 | 0 | 1 |
| rc+pd_v2 | force | 4 | 4 | 4 | - | 0.01926 | 0.02348 | 0 | 1 |
| rc+pd_v2 | combined | 6 | 6 | 6 | - | 0.0265 | 0.03281 | 0 | 1 |
| replay+pd_v2 | nominal | 1 | 1 | 1 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_small | 6 | 6 | 6 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_large | 6 | 6 | 6 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | force | 4 | 4 | 4 | - | 0.01082 | 0.01495 | 0 | 1 |
| replay+pd_v2 | combined | 6 | 6 | 6 | - | 0.01082 | 0.01495 | 0 | 1 |
| rc+computed_torque | nominal | 1 | 1 | 1 | - | 0.01332 | 0.01332 | 0 | 1 |
| rc+computed_torque | posture_small | 6 | 6 | 6 | - | 0.02203 | 0.02806 | 0 | 1 |
| rc+computed_torque | posture_large | 6 | 6 | 6 | - | 0.0498 | 0.06393 | 0.001996 | 1 |
| rc+computed_torque | force | 4 | 4 | 3 | completed x1 | 0.02718 | 0.07803 | 0 | 1 |
| rc+computed_torque | combined | 6 | 6 | 4 | completed x2 | 0.03335 | 0.07879 | 0 | 1 |
| replay+computed_torque | nominal | 1 | 1 | 1 | - | 1.484e-05 | 1.484e-05 | 0 | 1 |
| replay+computed_torque | posture_small | 6 | 6 | 6 | - | 1.73e-05 | 1.889e-05 | 0 | 1 |
| replay+computed_torque | posture_large | 6 | 6 | 6 | - | 4.178e-05 | 5.125e-05 | 0 | 1 |
| replay+computed_torque | force | 4 | 4 | 4 | - | 0.04236 | 0.08794 | 0 | 1 |
| replay+computed_torque | combined | 6 | 6 | 6 | - | 0.04236 | 0.08794 | 0 | 1 |

## Paired effects (RC minus replay, same tracker and scenario)

| tracker | class | metric | pairs | both succeeded | RC failures | replay failures | median RC | median replay | median difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pd_v2 | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.0123 rad | 0.000254 rad | 0.01204 rad |
| pd_v2 | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.002333 m | 6.093e-06 m | 0.002327 m |
| pd_v2 | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.02654 N*m | 0.01823 N*m | 0.008317 N*m |
| pd_v2 | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_small | joint_rmse | 6 | 6 | 0 | 0 | 0.02096 rad | 0.000254 rad | 0.02071 rad |
| pd_v2 | posture_small | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002428 m | 6.093e-06 m | 0.002422 m |
| pd_v2 | posture_small | effort_torque_rms | 6 | 6 | 0 | 0 | 0.1363 N*m | 0.1271 N*m | 0.008081 N*m |
| pd_v2 | posture_small | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_large | joint_rmse | 6 | 6 | 0 | 0 | 0.04884 rad | 0.000254 rad | 0.04859 rad |
| pd_v2 | posture_large | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002302 m | 6.093e-06 m | 0.002296 m |
| pd_v2 | posture_large | effort_torque_rms | 6 | 6 | 0 | 0 | 0.1798 N*m | 0.3247 N*m | -0.1571 N*m |
| pd_v2 | posture_large | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | force | joint_rmse | 4 | 4 | 0 | 0 | 0.01926 rad | 0.01082 rad | 0.005426 rad |
| pd_v2 | force | dwell_endpoint_rms | 4 | 4 | 0 | 0 | 0.002413 m | 6.092e-06 m | 0.002407 m |
| pd_v2 | force | effort_torque_rms | 4 | 4 | 0 | 0 | 0.4604 N*m | 0.4758 N*m | -0.01508 N*m |
| pd_v2 | force | effort_saturation_fraction | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | combined | joint_rmse | 6 | 6 | 0 | 0 | 0.0265 rad | 0.01082 rad | 0.0155 rad |
| pd_v2 | combined | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002439 m | 6.092e-06 m | 0.002433 m |
| pd_v2 | combined | effort_torque_rms | 6 | 6 | 0 | 0 | 0.4758 N*m | 0.4873 N*m | -0.0104 N*m |
| pd_v2 | combined | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.01332 rad | 1.484e-05 rad | 0.01331 rad |
| computed_torque | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.002041 m | 5.474e-06 m | 0.002035 m |
| computed_torque | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.04647 N*m | 0.01813 N*m | 0.02834 N*m |
| computed_torque | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | posture_small | joint_rmse | 6 | 6 | 0 | 0 | 0.02203 rad | 1.73e-05 rad | 0.02201 rad |
| computed_torque | posture_small | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002146 m | 5.474e-06 m | 0.002141 m |
| computed_torque | posture_small | effort_torque_rms | 6 | 6 | 0 | 0 | 0.3181 N*m | 0.1094 N*m | 0.1899 N*m |
| computed_torque | posture_small | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | posture_large | joint_rmse | 6 | 6 | 0 | 0 | 0.0498 rad | 4.178e-05 rad | 0.04975 rad |
| computed_torque | posture_large | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002062 m | 5.474e-06 m | 0.002057 m |
| computed_torque | posture_large | effort_torque_rms | 6 | 6 | 0 | 0 | 0.3761 N*m | 0.4701 N*m | -0.09393 N*m |
| computed_torque | posture_large | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0.000998 | 0 | 0.000998 |
| computed_torque | force | joint_rmse | 4 | 3 | 1 | 0 | 0.02718 rad | 0.007848 rad | 0.003063 rad |
| computed_torque | force | dwell_endpoint_rms | 4 | 3 | 1 | 0 | 0.002667 m | 9.034e-06 m | 0.002658 m |
| computed_torque | force | effort_torque_rms | 4 | 3 | 1 | 0 | 0.4709 N*m | 0.4733 N*m | -0.009749 N*m |
| computed_torque | force | effort_saturation_fraction | 4 | 3 | 1 | 0 | 0 | 0 | 0 |
| computed_torque | combined | joint_rmse | 6 | 4 | 2 | 0 | 0.03335 rad | 0.007244 rad | 0.01602 rad |
| computed_torque | combined | dwell_endpoint_rms | 6 | 4 | 2 | 0 | 0.002739 m | 9.034e-06 m | 0.00273 m |
| computed_torque | combined | effort_torque_rms | 6 | 4 | 2 | 0 | 0.5664 N*m | 0.4785 N*m | 0.08791 N*m |
| computed_torque | combined | effort_saturation_fraction | 6 | 4 | 2 | 0 | 0 | 0 | 0 |

## Failed runs (3)

| arm | scenario | termination | failed criteria | run |
| --- | --- | --- | --- | --- |
| rc+computed_torque | force-7.5N-030deg | completed | dwell_in_tolerance | `run-20260831-558edd9eb66d` |
| rc+computed_torque | combined-20261003-00-030deg | completed | dwell_in_tolerance | `run-20260831-a9ff331d1f52` |
| rc+computed_torque | combined-20261004-01-030deg | completed | dwell_in_tolerance | `run-20260831-ce0b47d2fa09` |
