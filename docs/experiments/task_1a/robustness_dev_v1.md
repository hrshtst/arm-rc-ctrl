# Robustness suite `task-1a-robustness-dev-v1` (development)

- Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `esn-task-1a-v3`
  (`data/records/models/model-20260831-ea83321eeaa5.toml`), estimator cutoffs 14.18/10.61 Hz.
- 23 scenarios x 4 arms = 92 runs;
  protocol `configs/evaluations/task_1a_robustness_dev_v1.toml`; commit `e1e4f4e78765`.

## Outcomes by arm and class

| arm | class | n | completed | successes | failures | joint RMSE median (rad) | joint RMSE max | saturation max | dwell in-tolerance median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rc+pd_v2 | nominal | 1 | 1 | 1 | - | 0.0123 | 0.0123 | 0 | 1 |
| rc+pd_v2 | posture_small | 6 | 6 | 6 | - | 0.02591 | 0.0326 | 0 | 1 |
| rc+pd_v2 | posture_large | 6 | 6 | 6 | - | 0.05161 | 0.05639 | 0 | 1 |
| rc+pd_v2 | force | 4 | 4 | 4 | - | 0.01909 | 0.02827 | 0 | 1 |
| rc+pd_v2 | combined | 6 | 6 | 6 | - | 0.03355 | 0.04265 | 0 | 1 |
| replay+pd_v2 | nominal | 1 | 1 | 1 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_small | 6 | 6 | 6 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_large | 6 | 6 | 6 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | force | 4 | 4 | 4 | - | 0.01364 | 0.01496 | 0 | 1 |
| replay+pd_v2 | combined | 6 | 6 | 6 | - | 0.01327 | 0.01496 | 0 | 1 |
| rc+computed_torque | nominal | 1 | 1 | 1 | - | 0.01332 | 0.01332 | 0 | 1 |
| rc+computed_torque | posture_small | 6 | 6 | 6 | - | 0.0258 | 0.03373 | 0 | 1 |
| rc+computed_torque | posture_large | 6 | 6 | 6 | - | 0.05195 | 0.05532 | 0.001996 | 1 |
| rc+computed_torque | force | 4 | 4 | 3 | completed x1 | 0.04755 | 0.1015 | 0 | 1 |
| rc+computed_torque | combined | 6 | 6 | 5 | completed x1 | 0.0555 | 0.1015 | 0 | 1 |
| replay+computed_torque | nominal | 1 | 1 | 1 | - | 1.484e-05 | 1.484e-05 | 0 | 1 |
| replay+computed_torque | posture_small | 6 | 6 | 6 | - | 2.232e-05 | 2.623e-05 | 0 | 1 |
| replay+computed_torque | posture_large | 6 | 6 | 6 | - | 4.381e-05 | 4.755e-05 | 0 | 1 |
| replay+computed_torque | force | 4 | 4 | 4 | - | 0.06803 | 0.08859 | 0 | 1 |
| replay+computed_torque | combined | 6 | 6 | 6 | - | 0.06803 | 0.08859 | 0 | 1 |

## Paired effects (RC minus replay, same tracker and scenario)

| tracker | class | metric | pairs | both succeeded | RC failures | replay failures | median RC | median replay | median difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pd_v2 | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.0123 rad | 0.000254 rad | 0.01204 rad |
| pd_v2 | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.002333 m | 6.093e-06 m | 0.002327 m |
| pd_v2 | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.02654 N*m | 0.01823 N*m | 0.008317 N*m |
| pd_v2 | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_small | joint_rmse | 6 | 6 | 0 | 0 | 0.02591 rad | 0.000254 rad | 0.02566 rad |
| pd_v2 | posture_small | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002341 m | 6.093e-06 m | 0.002334 m |
| pd_v2 | posture_small | effort_torque_rms | 6 | 6 | 0 | 0 | 0.0971 N*m | 0.1666 N*m | -0.06653 N*m |
| pd_v2 | posture_small | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_large | joint_rmse | 6 | 6 | 0 | 0 | 0.05161 rad | 0.000254 rad | 0.05136 rad |
| pd_v2 | posture_large | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002085 m | 6.093e-06 m | 0.002079 m |
| pd_v2 | posture_large | effort_torque_rms | 6 | 6 | 0 | 0 | 0.09549 N*m | 0.3214 N*m | -0.229 N*m |
| pd_v2 | posture_large | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | force | joint_rmse | 4 | 4 | 0 | 0 | 0.01909 rad | 0.01364 rad | 0.004979 rad |
| pd_v2 | force | dwell_endpoint_rms | 4 | 4 | 0 | 0 | 0.002418 m | 6.093e-06 m | 0.002412 m |
| pd_v2 | force | effort_torque_rms | 4 | 4 | 0 | 0 | 0.5306 N*m | 0.5445 N*m | -0.01385 N*m |
| pd_v2 | force | effort_saturation_fraction | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | combined | joint_rmse | 6 | 6 | 0 | 0 | 0.03355 rad | 0.01327 rad | 0.01991 rad |
| pd_v2 | combined | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002297 m | 6.093e-06 m | 0.002291 m |
| pd_v2 | combined | effort_torque_rms | 6 | 6 | 0 | 0 | 0.5434 N*m | 0.567 N*m | -0.01915 N*m |
| pd_v2 | combined | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.01332 rad | 1.484e-05 rad | 0.01331 rad |
| computed_torque | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.002041 m | 5.474e-06 m | 0.002035 m |
| computed_torque | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.04647 N*m | 0.01813 N*m | 0.02834 N*m |
| computed_torque | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | posture_small | joint_rmse | 6 | 6 | 0 | 0 | 0.0258 rad | 2.232e-05 rad | 0.02577 rad |
| computed_torque | posture_small | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.002064 m | 5.474e-06 m | 0.002059 m |
| computed_torque | posture_small | effort_torque_rms | 6 | 6 | 0 | 0 | 0.2337 N*m | 0.2038 N*m | 0.03124 N*m |
| computed_torque | posture_small | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | posture_large | joint_rmse | 6 | 6 | 0 | 0 | 0.05195 rad | 4.381e-05 rad | 0.0519 rad |
| computed_torque | posture_large | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.001809 m | 5.474e-06 m | 0.001804 m |
| computed_torque | posture_large | effort_torque_rms | 6 | 6 | 0 | 0 | 0.2275 N*m | 0.5019 N*m | -0.2832 N*m |
| computed_torque | posture_large | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | force | joint_rmse | 4 | 3 | 1 | 0 | 0.04755 rad | 0.04852 rad | 0.006329 rad |
| computed_torque | force | dwell_endpoint_rms | 4 | 3 | 1 | 0 | 0.003063 m | 5.57e-06 m | 0.003058 m |
| computed_torque | force | effort_torque_rms | 4 | 3 | 1 | 0 | 0.6306 N*m | 0.6372 N*m | 0.01622 N*m |
| computed_torque | force | effort_saturation_fraction | 4 | 3 | 1 | 0 | 0 | 0 | 0 |
| computed_torque | combined | joint_rmse | 6 | 5 | 1 | 0 | 0.0555 rad | 0.04852 rad | 0.00698 rad |
| computed_torque | combined | dwell_endpoint_rms | 6 | 5 | 1 | 0 | 0.003777 m | 5.57e-06 m | 0.003772 m |
| computed_torque | combined | effort_torque_rms | 6 | 5 | 1 | 0 | 0.6655 N*m | 0.6651 N*m | 0.005395 N*m |
| computed_torque | combined | effort_saturation_fraction | 6 | 5 | 1 | 0 | 0 | 0 | 0 |

## Failed runs (2)

| arm | scenario | termination | failed criteria | run |
| --- | --- | --- | --- | --- |
| rc+computed_torque | force-9N-225deg | completed | dwell_stationary | `run-20260831-0f7bc6d6d7f7` |
| rc+computed_torque | combined-20261001-00-045deg | completed | dwell_in_tolerance | `run-20260831-d0d6e7f2f638` |
