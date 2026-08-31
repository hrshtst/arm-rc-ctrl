# Robustness suite `task-1a-confirmatory-v2` (confirmatory)

- Scenario `task-1a-reach`, reference `processed-20260830-feaf73e6663c`, recipe `esn-task-1a-v4`
  (`data/records/models/model-20260831-1b9477aaa246.toml`), estimator cutoffs 29.98/10.94 Hz.
- 65 scenarios x 4 arms = 260 runs;
  protocol `configs/evaluations/task_1a_confirmatory_v2.toml`; commit `19b9af3f5540`.

## Outcomes by arm and class

| arm | class | n | completed | successes | failures | joint RMSE median (rad) | joint RMSE max | saturation max | dwell in-tolerance median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rc+pd_v2 | nominal | 1 | 1 | 1 | - | 0.002599 | 0.002599 | 0 | 1 |
| rc+pd_v2 | posture_small | 20 | 20 | 20 | - | 0.004716 | 0.006109 | 0.001996 | 1 |
| rc+pd_v2 | posture_large | 20 | 20 | 20 | - | 0.008515 | 0.01018 | 0.003992 | 1 |
| rc+pd_v2 | force | 4 | 4 | 4 | - | 0.01583 | 0.02445 | 0 | 1 |
| rc+pd_v2 | combined | 20 | 20 | 20 | - | 0.01667 | 0.02485 | 0.001996 | 1 |
| replay+pd_v2 | nominal | 1 | 1 | 1 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_small | 20 | 20 | 20 | - | 0.000254 | 0.000254 | 0 | 1 |
| replay+pd_v2 | posture_large | 20 | 20 | 20 | - | 0.000254 | 0.000254 | 0.001996 | 1 |
| replay+pd_v2 | force | 4 | 4 | 4 | - | 0.01561 | 0.02424 | 0 | 1 |
| replay+pd_v2 | combined | 20 | 20 | 20 | - | 0.01561 | 0.02424 | 0 | 1 |
| rc+computed_torque | nominal | 1 | 1 | 1 | - | 0.002615 | 0.002615 | 0 | 1 |
| rc+computed_torque | posture_small | 20 | 20 | 20 | - | 0.004394 | 0.01219 | 0.001996 | 1 |
| rc+computed_torque | posture_large | 20 | 20 | 20 | - | 0.009263 | 0.02003 | 0.003992 | 1 |
| rc+computed_torque | force | 4 | 4 | 4 | - | 0.06898 | 0.1424 | 0 | 1 |
| rc+computed_torque | combined | 20 | 20 | 20 | - | 0.06995 | 0.1428 | 0.001996 | 1 |
| replay+computed_torque | nominal | 1 | 1 | 1 | - | 1.484e-05 | 1.484e-05 | 0 | 1 |
| replay+computed_torque | posture_small | 20 | 20 | 20 | - | 2.523e-05 | 3.191e-05 | 0 | 1 |
| replay+computed_torque | posture_large | 20 | 20 | 20 | - | 3.523e-05 | 5.659e-05 | 0 | 1 |
| replay+computed_torque | force | 4 | 4 | 4 | - | 0.06898 | 0.1417 | 0 | 1 |
| replay+computed_torque | combined | 20 | 20 | 20 | - | 0.06898 | 0.1417 | 0 | 1 |

## Paired effects (RC minus replay, same tracker and scenario)

| tracker | class | metric | pairs | both succeeded | RC failures | replay failures | median RC | median replay | median difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pd_v2 | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.002599 rad | 0.000254 rad | 0.002345 rad |
| pd_v2 | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.001617 m | 6.093e-06 m | 0.001611 m |
| pd_v2 | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.02338 N*m | 0.01823 N*m | 0.005151 N*m |
| pd_v2 | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_small | joint_rmse | 20 | 20 | 0 | 0 | 0.004716 rad | 0.000254 rad | 0.004462 rad |
| pd_v2 | posture_small | dwell_endpoint_rms | 20 | 20 | 0 | 0 | 0.001617 m | 6.093e-06 m | 0.001611 m |
| pd_v2 | posture_small | effort_torque_rms | 20 | 20 | 0 | 0 | 0.3261 N*m | 0.1606 N*m | 0.1474 N*m |
| pd_v2 | posture_small | effort_saturation_fraction | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | posture_large | joint_rmse | 20 | 20 | 0 | 0 | 0.008515 rad | 0.000254 rad | 0.008261 rad |
| pd_v2 | posture_large | dwell_endpoint_rms | 20 | 20 | 0 | 0 | 0.001622 m | 6.093e-06 m | 0.001616 m |
| pd_v2 | posture_large | effort_torque_rms | 20 | 20 | 0 | 0 | 0.4522 N*m | 0.3424 N*m | 0.1044 N*m |
| pd_v2 | posture_large | effort_saturation_fraction | 20 | 20 | 0 | 0 | 0.001996 | 0 | 0.001996 |
| pd_v2 | force | joint_rmse | 4 | 4 | 0 | 0 | 0.01583 rad | 0.01561 rad | 0.0003061 rad |
| pd_v2 | force | dwell_endpoint_rms | 4 | 4 | 0 | 0 | 0.001616 m | 6.093e-06 m | 0.00161 m |
| pd_v2 | force | effort_torque_rms | 4 | 4 | 0 | 0 | 0.7288 N*m | 0.729 N*m | -0.0008357 N*m |
| pd_v2 | force | effort_saturation_fraction | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| pd_v2 | combined | joint_rmse | 20 | 20 | 0 | 0 | 0.01667 rad | 0.01561 rad | 0.0008848 rad |
| pd_v2 | combined | dwell_endpoint_rms | 20 | 20 | 0 | 0 | 0.001618 m | 6.093e-06 m | 0.001611 m |
| pd_v2 | combined | effort_torque_rms | 20 | 20 | 0 | 0 | 0.7939 N*m | 0.7563 N*m | 0.0498 N*m |
| pd_v2 | combined | effort_saturation_fraction | 20 | 20 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | nominal | joint_rmse | 1 | 1 | 0 | 0 | 0.002615 rad | 1.484e-05 rad | 0.0026 rad |
| computed_torque | nominal | dwell_endpoint_rms | 1 | 1 | 0 | 0 | 0.001585 m | 5.474e-06 m | 0.00158 m |
| computed_torque | nominal | effort_torque_rms | 1 | 1 | 0 | 0 | 0.02984 N*m | 0.01813 N*m | 0.01171 N*m |
| computed_torque | nominal | effort_saturation_fraction | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | posture_small | joint_rmse | 20 | 20 | 0 | 0 | 0.004394 rad | 2.523e-05 rad | 0.004368 rad |
| computed_torque | posture_small | dwell_endpoint_rms | 20 | 20 | 0 | 0 | 0.001599 m | 5.474e-06 m | 0.001594 m |
| computed_torque | posture_small | effort_torque_rms | 20 | 20 | 0 | 0 | 0.3467 N*m | 0.2441 N*m | 0.1023 N*m |
| computed_torque | posture_small | effort_saturation_fraction | 20 | 20 | 0 | 0 | 0.001996 | 0 | 0.001996 |
| computed_torque | posture_large | joint_rmse | 20 | 20 | 0 | 0 | 0.009263 rad | 3.523e-05 rad | 0.00924 rad |
| computed_torque | posture_large | dwell_endpoint_rms | 20 | 20 | 0 | 0 | 0.001593 m | 5.474e-06 m | 0.001588 m |
| computed_torque | posture_large | effort_torque_rms | 20 | 20 | 0 | 0 | 0.4835 N*m | 0.3707 N*m | 0.07209 N*m |
| computed_torque | posture_large | effort_saturation_fraction | 20 | 20 | 0 | 0 | 0.001996 | 0 | 0.001996 |
| computed_torque | force | joint_rmse | 4 | 4 | 0 | 0 | 0.06898 rad | 0.06898 rad | 0.0003625 rad |
| computed_torque | force | dwell_endpoint_rms | 4 | 4 | 0 | 0 | 0.001613 m | 5.776e-06 m | 0.001607 m |
| computed_torque | force | effort_torque_rms | 4 | 4 | 0 | 0 | 0.6918 N*m | 0.6884 N*m | 0.0002221 N*m |
| computed_torque | force | effort_saturation_fraction | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| computed_torque | combined | joint_rmse | 20 | 20 | 0 | 0 | 0.06995 rad | 0.06898 rad | 0.000813 rad |
| computed_torque | combined | dwell_endpoint_rms | 20 | 20 | 0 | 0 | 0.001613 m | 5.776e-06 m | 0.001608 m |
| computed_torque | combined | effort_torque_rms | 20 | 20 | 0 | 0 | 0.7818 N*m | 0.7327 N*m | 0.03941 N*m |
| computed_torque | combined | effort_saturation_fraction | 20 | 20 | 0 | 0 | 0.001996 | 0 | 0.001996 |

## Failed runs (0)

None.
