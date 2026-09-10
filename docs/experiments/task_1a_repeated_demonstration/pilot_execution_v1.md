# Task 1-a repetition pilot execution accounting (v1)

Experiment `task_1a_repetition_v1`: panel manifest sha256 `68bc850ef269`, numerical validation sha256 `40720b539903`, canonical execution identity `a7f034c7aef4`, project commit `de1d79e4b620`.

## Totals

- Behavioral configurations: 120 of 120 with evidence; missing: 0.
- Statuses: feasible 25, rc_gate_failure 95, replay_blocked 0, training_failure 0.
- RC runs executed: 4400; unexecuted pairs: 11200; replay-blocked pairs: 0; replay runs: 390 in 3 banks.
- Models that crossed the historical 6 rad/s limit in an executed RC run: 61.
- S-effective numerical reference fits (validated, never evaluated behaviorally): 36.
- Every manifest binds the canonical execution identity: True; every manifest carries the C11 caveat: True; accounting complete: **True**.

## Replay banks

| warm-up (s) | identity | pairs | completed | infeasible |
| ---: | --- | ---: | ---: | ---: |
| 0.25 | `e60fecac20a7` | 130 | 130 | 0 |
| 0 | `0a30281299cf` | 130 | 130 | 0 |
| 1 | `93caeea3d6c8` | 130 | 130 | 0 |

## Configurations

| entry | arm | status | completed | infeasible | blocked | unexecuted | crossed 6 rad/s | first failure |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| feasible-best | absolute/S | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-best | absolute/R/K17 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-best | absolute/R-scaled/K17 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-best | absolute/A-non-decaying/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | absolute/A-contractive/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | absolute/R/K33 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-best | absolute/R-scaled/K33 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-best | absolute/A-non-decaying/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance |
| feasible-best | absolute/A-contractive/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | absolute/R/K65 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-best | absolute/R-scaled/K65 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-best | absolute/A-non-decaying/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-best | absolute/A-contractive/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-best | residual/S | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual/R/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | residual/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual/R/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual/R/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-middle | absolute/S | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-middle | absolute/R/K17 | feasible | 130 | 0 | 0 | 0 | yes |  |
| feasible-middle | absolute/R-scaled/K17 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-middle | absolute/A-non-decaying/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute/A-contractive/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute/R/K33 | feasible | 130 | 0 | 0 | 0 | yes |  |
| feasible-middle | absolute/R-scaled/K33 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-middle | absolute/A-non-decaying/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute/A-contractive/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute/R/K65 | feasible | 130 | 0 | 0 | 0 | yes |  |
| feasible-middle | absolute/R-scaled/K65 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-middle | absolute/A-non-decaying/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute/A-contractive/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual/S | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual/R/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual/R/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual/R/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | absolute/S | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-worst | absolute/R/K17 | rc_gate_failure | 83 | 1 | 0 | 46 | no | scenario 41 [computed_torque]: dwell:dwell_in_tolerance |
| feasible-worst | absolute/R-scaled/K17 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-worst | absolute/A-non-decaying/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute/A-contractive/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute/R/K33 | rc_gate_failure | 83 | 1 | 0 | 46 | no | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-worst | absolute/R-scaled/K33 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-worst | absolute/A-non-decaying/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute/A-contractive/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute/R/K65 | rc_gate_failure | 8 | 1 | 0 | 121 | no | scenario 4 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute/R-scaled/K65 | feasible | 130 | 0 | 0 | 0 | no |  |
| feasible-worst | absolute/A-non-decaying/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute/A-contractive/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | residual/S | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual/R/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-worst | residual/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual/R/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-worst | residual/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual/R/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | absolute/S | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/R/K17 | rc_gate_failure | 83 | 1 | 0 | 46 | yes | scenario 41 [computed_torque]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/A-non-decaying/K17 | rc_gate_failure | 44 | 1 | 0 | 85 | no | scenario 22 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/A-contractive/K17 | rc_gate_failure | 83 | 1 | 0 | 46 | yes | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-actual-dwell | absolute/R/K33 | rc_gate_failure | 83 | 1 | 0 | 46 | no | scenario 41 [computed_torque]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/A-non-decaying/K33 | rc_gate_failure | 2 | 1 | 0 | 127 | no | scenario 1 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/A-contractive/K33 | rc_gate_failure | 83 | 1 | 0 | 46 | yes | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-actual-dwell | absolute/R/K65 | rc_gate_failure | 83 | 1 | 0 | 46 | no | scenario 41 [computed_torque]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/A-non-decaying/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute/A-contractive/K65 | rc_gate_failure | 83 | 1 | 0 | 46 | yes | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-actual-dwell | residual/S | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | residual/R/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-actual-dwell | residual/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | residual/R/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-actual-dwell | residual/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | residual/R/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-actual-dwell | residual/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | absolute/S | feasible | 130 | 0 | 0 | 0 | yes |  |
| failure-joint-velocity | absolute/R/K17 | feasible | 130 | 0 | 0 | 0 | yes |  |
| failure-joint-velocity | absolute/R-scaled/K17 | feasible | 130 | 0 | 0 | 0 | yes |  |
| failure-joint-velocity | absolute/A-non-decaying/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | absolute/A-contractive/K17 | rc_gate_failure | 82 | 1 | 0 | 47 | no | scenario 41 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-joint-velocity | absolute/R/K33 | feasible | 130 | 0 | 0 | 0 | yes |  |
| failure-joint-velocity | absolute/R-scaled/K33 | feasible | 130 | 0 | 0 | 0 | yes |  |
| failure-joint-velocity | absolute/A-non-decaying/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | absolute/A-contractive/K33 | rc_gate_failure | 6 | 1 | 0 | 123 | no | scenario 3 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | absolute/R/K65 | feasible | 130 | 0 | 0 | 0 | yes |  |
| failure-joint-velocity | absolute/R-scaled/K65 | feasible | 130 | 0 | 0 | 0 | yes |  |
| failure-joint-velocity | absolute/A-non-decaying/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-joint-velocity | absolute/A-contractive/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | residual/S | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual/R/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual/R/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual/R/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | absolute/S | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute/R/K17 | rc_gate_failure | 83 | 1 | 0 | 46 | no | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute/A-non-decaying/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute/A-contractive/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute/R/K33 | rc_gate_failure | 83 | 1 | 0 | 46 | no | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute/A-non-decaying/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute/A-contractive/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute/R/K65 | rc_gate_failure | 83 | 1 | 0 | 46 | no | scenario 41 [computed_torque]: dwell:dwell_in_tolerance |
| failure-generated-dwell | absolute/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute/A-non-decaying/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute/A-contractive/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | residual/S | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual/R/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | no | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-generated-dwell | residual/R-scaled/K17 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual/R/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual/R-scaled/K33 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual/R/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual/R-scaled/K65 | rc_gate_failure | 0 | 1 | 0 | 129 | yes | scenario 0 [pd_v2]: limit_violation:joint_velocity |

## Limitations

- This is an execution accounting, not a result: paired metrics, comparisons, and interpretation belong to the report (M3REP-007) and its review (M3REP-GATE).
- Feasibility here is under the pilot's 12 rad/s evaluation abort; a run that crossed the historical 6 rad/s limit is not a recovery-v1 success (plan section 7.1).
