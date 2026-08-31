# Robustness suite `task-1a-robustness-dev-v2` (development)

- Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `esn-task-1a-v4`
  (`data/records/models/model-20260831-1b9477aaa246.toml`), estimator cutoffs 29.98/10.94 Hz.
- 23 scenarios x 4 arms = 92 runs;
  protocol `configs/evaluations/task_1a_robustness_dev_v2.toml`; commit `5c210b2658f9`.

## Outcomes by arm and class

| arm | class | n | completed | successes | failures | joint RMSE median (rad) | joint RMSE max | saturation max | dwell in-tolerance median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rc+pd_v2 | nominal | 1 | 1 | 1 | - | 0.002599 | 0.002599 | 0 | 1 |
| rc+pd_v2 | posture_small | 6 | 6 | 6 | - | 0.004088 | 0.004538 | 0 | 1 |
| rc+pd_v2 | posture_large | 6 | 6 | 6 | - | 0.007723 | 0.00874 | 0.001996 | 1 |
| rc+pd_v2 | force | 4 | 4 | 4 | - | 0.01101 | 0.01543 | 0 | 1 |
| rc+pd_v2 | combined | 6 | 6 | 6 | - | 0.01147 | 0.0158 | 0 | 1 |
| replay+pd_v2 | nominal | 1 | 1 | 1 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_small | 6 | 6 | 6 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_large | 6 | 6 | 6 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | force | 4 | 4 | 4 | - | 0.01082 | 0.01495 | 0 | 1 |
| replay+pd_v2 | combined | 6 | 6 | 6 | - | 0.01082 | 0.01495 | 0 | 1 |
| rc+computed_torque | nominal | 1 | 1 | 1 | - | 0.002615 | 0.002615 | 0 | 1 |
| rc+computed_torque | posture_small | 6 | 6 | 6 | - | 0.003794 | 0.004617 | 0.001996 | 1 |
| rc+computed_torque | posture_large | 6 | 6 | 6 | - | 0.009627 | 0.01991 | 0.001996 | 1 |
| rc+computed_torque | force | 4 | 4 | 4 | - | 0.0427 | 0.08862 | 0 | 1 |
| rc+computed_torque | combined | 6 | 6 | 6 | - | 0.04296 | 0.08867 | 0.001996 | 1 |
| replay+computed_torque | nominal | 1 | 1 | 1 | - | 1.484e-05 | 1.484e-05 | 0 | 1 |
| replay+computed_torque | posture_small | 6 | 6 | 6 | - | 1.73e-05 | 1.889e-05 | 0 | 1 |
| replay+computed_torque | posture_large | 6 | 6 | 6 | - | 4.178e-05 | 5.125e-05 | 0 | 1 |
| replay+computed_torque | force | 4 | 4 | 4 | - | 0.04236 | 0.08794 | 0 | 1 |
| replay+computed_torque | combined | 6 | 6 | 6 | - | 0.04236 | 0.08794 | 0 | 1 |

## Paired effects (RC minus replay, same tracker and scenario)

| tracker | class | metric | pairs | both succeeded | RC failures | replay failures | median RC | median replay | median difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pd_v2 | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.002599 rad | 0.000254 rad | 0.002345 rad |
| pd_v2 | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.001617 m | 6.093e-06 m | 0.001611 m |
| pd_v2 | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.02338 N*m | 0.01823 N*m | 0.005151 N*m |
| pd_v2 | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_small | joint_rmse | 6 | 6 | 0 | 0 | 0.004088 rad | 0.000254 rad | 0.003834 rad |
| pd_v2 | posture_small | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.001634 m | 6.093e-06 m | 0.001628 m |
| pd_v2 | posture_small | effort_torque_rms | 6 | 6 | 0 | 0 | 0.1831 N*m | 0.1271 N*m | 0.05918 N*m |
| pd_v2 | posture_small | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_large | joint_rmse | 6 | 6 | 0 | 0 | 0.007723 rad | 0.000254 rad | 0.007469 rad |
| pd_v2 | posture_large | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.001634 m | 6.093e-06 m | 0.001628 m |
| pd_v2 | posture_large | effort_torque_rms | 6 | 6 | 0 | 0 | 0.4305 N*m | 0.3247 N*m | 0.1193 N*m |
| pd_v2 | posture_large | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0.001996 | 0 | 0.001996 |
| pd_v2 | force | joint_rmse | 4 | 4 | 0 | 0 | 0.01101 rad | 0.01082 rad | 0.0002426 rad |
| pd_v2 | force | dwell_endpoint_rms | 4 | 4 | 0 | 0 | 0.001622 m | 6.092e-06 m | 0.001616 m |
| pd_v2 | force | effort_torque_rms | 4 | 4 | 0 | 0 | 0.4745 N*m | 0.4758 N*m | -0.000329 N*m |
| pd_v2 | force | effort_saturation_fraction | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | combined | joint_rmse | 6 | 6 | 0 | 0 | 0.01147 rad | 0.01082 rad | 0.0006716 rad |
| pd_v2 | combined | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.001623 m | 6.092e-06 m | 0.001617 m |
| pd_v2 | combined | effort_torque_rms | 6 | 6 | 0 | 0 | 0.5007 N*m | 0.4873 N*m | 0.01838 N*m |
| pd_v2 | combined | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.002615 rad | 1.484e-05 rad | 0.0026 rad |
| computed_torque | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.001585 m | 5.474e-06 m | 0.00158 m |
| computed_torque | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.02984 N*m | 0.01813 N*m | 0.01171 N*m |
| computed_torque | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | posture_small | joint_rmse | 6 | 6 | 0 | 0 | 0.003794 rad | 1.73e-05 rad | 0.003777 rad |
| computed_torque | posture_small | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.001597 m | 5.474e-06 m | 0.001592 m |
| computed_torque | posture_small | effort_torque_rms | 6 | 6 | 0 | 0 | 0.3581 N*m | 0.1094 N*m | 0.2742 N*m |
| computed_torque | posture_small | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0.000998 | 0 | 0.000998 |
| computed_torque | posture_large | joint_rmse | 6 | 6 | 0 | 0 | 0.009627 rad | 4.178e-05 rad | 0.009585 rad |
| computed_torque | posture_large | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.001597 m | 5.474e-06 m | 0.001592 m |
| computed_torque | posture_large | effort_torque_rms | 6 | 6 | 0 | 0 | 0.4914 N*m | 0.4701 N*m | 0.03617 N*m |
| computed_torque | posture_large | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0.001996 | 0 | 0.001996 |
| computed_torque | force | joint_rmse | 4 | 4 | 0 | 0 | 0.0427 rad | 0.04236 rad | 0.0002815 rad |
| computed_torque | force | dwell_endpoint_rms | 4 | 4 | 0 | 0 | 0.001598 m | 1.02e-05 m | 0.001588 m |
| computed_torque | force | effort_torque_rms | 4 | 4 | 0 | 0 | 0.4669 N*m | 0.4652 N*m | 0.001669 N*m |
| computed_torque | force | effort_saturation_fraction | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | combined | joint_rmse | 6 | 6 | 0 | 0 | 0.04296 rad | 0.04236 rad | 0.0007229 rad |
| computed_torque | combined | dwell_endpoint_rms | 6 | 6 | 0 | 0 | 0.001599 m | 1.02e-05 m | 0.001589 m |
| computed_torque | combined | effort_torque_rms | 6 | 6 | 0 | 0 | 0.568 N*m | 0.4726 N*m | 0.1154 N*m |
| computed_torque | combined | effort_saturation_fraction | 6 | 6 | 0 | 0 | 0.000998 | 0 | 0.000998 |

## Failed runs (0)

None.
