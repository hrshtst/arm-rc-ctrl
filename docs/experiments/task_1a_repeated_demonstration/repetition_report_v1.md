# Task 1-a repetition pilot report tables (v1)

Experiment `task_1a_repetition_v1`; canonical execution identity `a7f034c7aef4`; project commit `78f358a63297`; evaluation abort [12.0, 12.0] rad/s with the historical limit [6.0, 6.0] rad/s reported as a diagnostic.

- Feasible configurations: 25 of 120; RC-gate failures: 95; configurations that crossed the historical limit in an executed run: 61.
- Representative rule: For every panel entry, the first pair of the fixed evaluation order (the nominal scenario under pd_v2) of the absolute S, R/K65, R-scaled/K65, and A-contractive/K65 arms and of the residual S arm, whatever its outcome. Declared at the start of M3REP-007, after the execution accounting was visible; the rule selects by position in the panel, never by result, so failures are shown as often as successes.
- Animation rule: The feasible-best entry's representative pair of the absolute S and absolute R/K65 arms (their RC runs) and the replay run they share, exported on the task clock so matching frames show matching task times.
- C11: the feasible-middle absolute R/K65 fit did not demonstrate numerical agreement with S-effective within the approved tolerance (largest prediction difference 2.93e-8 rad); the fit is retained unchanged, the exact-arithmetic ridge identity stands, and this small discrepancy does not establish that closed-loop behavior is unaffected.

## Paired outcomes

| entry | formulation | arm | K | status | completed | unexecuted | crossed 6 rad/s | peak rad/s | worst cell | first failure |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| feasible-best | absolute | S | 1 | feasible | 130 | 0 | no | 5.97 | 0.6704 |  |
| feasible-best | absolute | R/K17 | 17 | feasible | 130 | 0 | no | 5.9 | 0.7467 |  |
| feasible-best | absolute | R-scaled/K17 | 17 | feasible | 130 | 0 | no | 5.97 | 0.6704 |  |
| feasible-best | absolute | A-non-decaying/K17 | 17 | rc_gate_failure | 0 | 129 | no | 4.38 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | absolute | A-contractive/K17 | 17 | rc_gate_failure | 0 | 129 | no | 0.577 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | absolute | R/K33 | 33 | feasible | 130 | 0 | no | 5.9 | 0.7632 |  |
| feasible-best | absolute | R-scaled/K33 | 33 | feasible | 130 | 0 | no | 5.97 | 0.6704 |  |
| feasible-best | absolute | A-non-decaying/K33 | 33 | rc_gate_failure | 0 | 129 | no | 5.79 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance |
| feasible-best | absolute | A-contractive/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 7.92 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | absolute | R/K65 | 65 | feasible | 130 | 0 | no | 5.92 | 0.8012 |  |
| feasible-best | absolute | R-scaled/K65 | 65 | feasible | 130 | 0 | no | 5.97 | 0.6704 |  |
| feasible-best | absolute | A-non-decaying/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.1 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-best | absolute | A-contractive/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.4 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-best | residual | S | 1 | rc_gate_failure | 0 | 129 | yes | 7.38 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual | R/K17 | 17 | rc_gate_failure | 0 | 129 | no | 4.25 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-best | residual | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 7.38 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual | R/K33 | 33 | rc_gate_failure | 0 | 129 | no | 3.19 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 7.38 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual | R/K65 | 65 | rc_gate_failure | 0 | 129 | no | 4.35 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-best | residual | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 7.38 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-middle | absolute | S | 1 | feasible | 130 | 0 | no | 5.99 | 0.9602 |  |
| feasible-middle | absolute | R/K17 | 17 | feasible | 130 | 0 | yes | 6 | 0.9838 |  |
| feasible-middle | absolute | R-scaled/K17 | 17 | feasible | 130 | 0 | no | 5.99 | 0.9602 |  |
| feasible-middle | absolute | A-non-decaying/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 13.2 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute | A-contractive/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute | R/K33 | 33 | feasible | 130 | 0 | yes | 6.01 | 0.991 |  |
| feasible-middle | absolute | R-scaled/K33 | 33 | feasible | 130 | 0 | no | 5.99 | 0.9602 |  |
| feasible-middle | absolute | A-non-decaying/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute | A-contractive/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 13 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute | R/K65 | 65 | feasible | 130 | 0 | yes | 6.01 | 1.004 |  |
| feasible-middle | absolute | R-scaled/K65 | 65 | feasible | 130 | 0 | no | 5.99 | 0.9602 |  |
| feasible-middle | absolute | A-non-decaying/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 14.7 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | absolute | A-contractive/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 14.4 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual | S | 1 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual | R/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12.2 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual | R/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual | R/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-middle | residual | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | absolute | S | 1 | feasible | 130 | 0 | no | 5.98 | 1.286 |  |
| feasible-worst | absolute | R/K17 | 17 | rc_gate_failure | 83 | 46 | no | 5.98 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance |
| feasible-worst | absolute | R-scaled/K17 | 17 | feasible | 130 | 0 | no | 5.98 | 1.286 |  |
| feasible-worst | absolute | A-non-decaying/K17 | 17 | rc_gate_failure | 0 | 129 | no | 0.43 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute | A-contractive/K17 | 17 | rc_gate_failure | 0 | 129 | no | 0.41 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute | R/K33 | 33 | rc_gate_failure | 83 | 46 | no | 5.98 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| feasible-worst | absolute | R-scaled/K33 | 33 | feasible | 130 | 0 | no | 5.98 | 1.286 |  |
| feasible-worst | absolute | A-non-decaying/K33 | 33 | rc_gate_failure | 0 | 129 | no | 0.432 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute | A-contractive/K33 | 33 | rc_gate_failure | 0 | 129 | no | 0.413 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute | R/K65 | 65 | rc_gate_failure | 8 | 121 | no | 2.75 | n/a | scenario 4 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute | R-scaled/K65 | 65 | feasible | 130 | 0 | no | 5.98 | 1.286 |  |
| feasible-worst | absolute | A-non-decaying/K65 | 65 | rc_gate_failure | 0 | 129 | no | 0.433 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | absolute | A-contractive/K65 | 65 | rc_gate_failure | 0 | 129 | no | 0.415 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| feasible-worst | residual | S | 1 | rc_gate_failure | 0 | 129 | yes | 12 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual | R/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 9.22 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-worst | residual | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual | R/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 8.5 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| feasible-worst | residual | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual | R/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.6 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| feasible-worst | residual | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | absolute | S | 1 | rc_gate_failure | 0 | 129 | no | 4.61 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | R/K17 | 17 | rc_gate_failure | 83 | 46 | yes | 6.33 | n/a | scenario 41 [computed_torque]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | no | 4.61 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | A-non-decaying/K17 | 17 | rc_gate_failure | 44 | 85 | no | 3.78 | n/a | scenario 22 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | A-contractive/K17 | 17 | rc_gate_failure | 83 | 46 | yes | 6.8 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-actual-dwell | absolute | R/K33 | 33 | rc_gate_failure | 83 | 46 | no | 5.97 | n/a | scenario 41 [computed_torque]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | no | 4.61 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | A-non-decaying/K33 | 33 | rc_gate_failure | 2 | 127 | no | 1.19 | n/a | scenario 1 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | A-contractive/K33 | 33 | rc_gate_failure | 83 | 46 | yes | 7.06 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-actual-dwell | absolute | R/K65 | 65 | rc_gate_failure | 83 | 46 | no | 5.97 | n/a | scenario 41 [computed_torque]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | no | 4.61 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | A-non-decaying/K65 | 65 | rc_gate_failure | 0 | 129 | no | 0.49 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-actual-dwell | absolute | A-contractive/K65 | 65 | rc_gate_failure | 83 | 46 | yes | 7.46 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-actual-dwell | residual | S | 1 | rc_gate_failure | 0 | 129 | yes | 12.7 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | residual | R/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 11.1 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-actual-dwell | residual | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12.7 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | residual | R/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 11 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-actual-dwell | residual | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12.7 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-actual-dwell | residual | R/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 10.9 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-actual-dwell | residual | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.7 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | absolute | S | 1 | feasible | 130 | 0 | yes | 6.16 | 1.344 |  |
| failure-joint-velocity | absolute | R/K17 | 17 | feasible | 130 | 0 | yes | 6.08 | 1.034 |  |
| failure-joint-velocity | absolute | R-scaled/K17 | 17 | feasible | 130 | 0 | yes | 6.16 | 1.344 |  |
| failure-joint-velocity | absolute | A-non-decaying/K17 | 17 | rc_gate_failure | 0 | 129 | no | 0.539 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | absolute | A-contractive/K17 | 17 | rc_gate_failure | 82 | 47 | no | 5.63 | n/a | scenario 41 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-joint-velocity | absolute | R/K33 | 33 | feasible | 130 | 0 | yes | 6.09 | 1.045 |  |
| failure-joint-velocity | absolute | R-scaled/K33 | 33 | feasible | 130 | 0 | yes | 6.16 | 1.344 |  |
| failure-joint-velocity | absolute | A-non-decaying/K33 | 33 | rc_gate_failure | 0 | 129 | no | 0.565 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | absolute | A-contractive/K33 | 33 | rc_gate_failure | 6 | 123 | no | 0.465 | n/a | scenario 3 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | absolute | R/K65 | 65 | feasible | 130 | 0 | yes | 6.11 | 1.123 |  |
| failure-joint-velocity | absolute | R-scaled/K65 | 65 | feasible | 130 | 0 | yes | 6.16 | 1.344 |  |
| failure-joint-velocity | absolute | A-non-decaying/K65 | 65 | rc_gate_failure | 0 | 129 | no | 0.676 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-joint-velocity | absolute | A-contractive/K65 | 65 | rc_gate_failure | 0 | 129 | no | 0.476 | n/a | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | residual | S | 1 | rc_gate_failure | 0 | 129 | yes | 12.2 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual | R/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12.3 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12.2 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual | R/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12.9 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12.2 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual | R/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.5 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-joint-velocity | residual | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.2 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | absolute | S | 1 | rc_gate_failure | 0 | 129 | no | 2.56 | n/a | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute | R/K17 | 17 | rc_gate_failure | 83 | 46 | no | 5.99 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | no | 2.56 | n/a | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute | A-non-decaying/K17 | 17 | rc_gate_failure | 0 | 129 | no | 0.627 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute | A-contractive/K17 | 17 | rc_gate_failure | 0 | 129 | no | 0.509 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute | R/K33 | 33 | rc_gate_failure | 83 | 46 | no | 5.99 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | no | 2.56 | n/a | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute | A-non-decaying/K33 | 33 | rc_gate_failure | 0 | 129 | no | 0.744 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute | A-contractive/K33 | 33 | rc_gate_failure | 0 | 129 | no | 0.556 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute | R/K65 | 65 | rc_gate_failure | 83 | 46 | no | 5.99 | n/a | scenario 41 [computed_torque]: dwell:dwell_in_tolerance |
| failure-generated-dwell | absolute | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | no | 2.56 | n/a | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |
| failure-generated-dwell | absolute | A-non-decaying/K65 | 65 | rc_gate_failure | 0 | 129 | no | 1.92 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | absolute | A-contractive/K65 | 65 | rc_gate_failure | 0 | 129 | no | 0.658 | n/a | scenario 0 [pd_v2]: dwell:dwell_in_tolerance,dwell_stationary |
| failure-generated-dwell | residual | S | 1 | rc_gate_failure | 0 | 129 | yes | 12.9 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual | R/K17 | 17 | rc_gate_failure | 0 | 129 | no | 5.78 | n/a | scenario 0 [pd_v2]: early_termination:invalid_output:bounds |
| failure-generated-dwell | residual | R-scaled/K17 | 17 | rc_gate_failure | 0 | 129 | yes | 12.9 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual | R/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12.4 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual | R-scaled/K33 | 33 | rc_gate_failure | 0 | 129 | yes | 12.9 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual | R/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | residual | R-scaled/K65 | 65 | rc_gate_failure | 0 | 129 | yes | 12.9 | n/a | scenario 0 [pd_v2]: limit_violation:joint_velocity |

## Paired comparisons (medians over the scenario/tracker pairs both arms completed)

| entry | comparison | K | left status | right status | shared pairs | early gap left | early gap right | early gap diff | jump left | jump right | jump diff | worst cell left | worst cell right |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feasible-best | absolute/R/K17 vs absolute/S | 17 | feasible | feasible | 130 | 0.001803 | 0.002408 | -0.0009235 | 0.03821 | 0.04751 | -0.02053 | 0.7467 | 0.6704 |
| feasible-best | absolute/R-scaled/K17 vs absolute/S | 17 | feasible | feasible | 130 | 0.002408 | 0.002408 | -1.735e-14 | 0.04751 | 0.04751 | -2.905e-13 | 0.6704 | 0.6704 |
| feasible-best | absolute/A-non-decaying/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.7467 |
| feasible-best | absolute/A-contractive/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.7467 |
| feasible-best | absolute/R/K33 vs absolute/S | 33 | feasible | feasible | 130 | 0.001753 | 0.002408 | -0.0009936 | 0.03867 | 0.04751 | -0.02268 | 0.7632 | 0.6704 |
| feasible-best | absolute/R-scaled/K33 vs absolute/S | 33 | feasible | feasible | 130 | 0.002408 | 0.002408 | -4.799e-16 | 0.04751 | 0.04751 | 1.126e-14 | 0.6704 | 0.6704 |
| feasible-best | absolute/A-non-decaying/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.7632 |
| feasible-best | absolute/A-contractive/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.7632 |
| feasible-best | absolute/R/K65 vs absolute/S | 65 | feasible | feasible | 130 | 0.001827 | 0.002408 | -0.001017 | 0.03837 | 0.04751 | -0.02403 | 0.8012 | 0.6704 |
| feasible-best | absolute/R-scaled/K65 vs absolute/S | 65 | feasible | feasible | 130 | 0.002408 | 0.002408 | -1.72e-14 | 0.04751 | 0.04751 | -1.951e-13 | 0.6704 | 0.6704 |
| feasible-best | absolute/A-non-decaying/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.8012 |
| feasible-best | absolute/A-contractive/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.8012 |
| feasible-best | residual/R/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-best | residual/R-scaled/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-best | residual/R/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-best | residual/R-scaled/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-best | residual/R/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-best | residual/R-scaled/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-best | residual/S vs absolute/S | 1 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.6704 |
| feasible-best | residual/R/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.7467 |
| feasible-best | residual/R/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.7632 |
| feasible-best | residual/R/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.8012 |
| feasible-best | residual/R-scaled/K17 vs absolute/R-scaled/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.6704 |
| feasible-best | residual/R-scaled/K33 vs absolute/R-scaled/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.6704 |
| feasible-best | residual/R-scaled/K65 vs absolute/R-scaled/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.6704 |
| feasible-middle | absolute/R/K17 vs absolute/S | 17 | feasible | feasible | 130 | 0.005915 | 0.005676 | -9.107e-06 | 0.04975 | 0.05159 | -0.001321 | 0.9838 | 0.9602 |
| feasible-middle | absolute/R-scaled/K17 vs absolute/S | 17 | feasible | feasible | 130 | 0.005676 | 0.005676 | 3.214e-13 | 0.05159 | 0.05159 | 2.26e-12 | 0.9602 | 0.9602 |
| feasible-middle | absolute/A-non-decaying/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9838 |
| feasible-middle | absolute/A-contractive/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9838 |
| feasible-middle | absolute/R/K33 vs absolute/S | 33 | feasible | feasible | 130 | 0.005992 | 0.005676 | 1.8e-05 | 0.04984 | 0.05159 | -0.001467 | 0.991 | 0.9602 |
| feasible-middle | absolute/R-scaled/K33 vs absolute/S | 33 | feasible | feasible | 130 | 0.005676 | 0.005676 | 6.302e-13 | 0.05159 | 0.05159 | 2.183e-12 | 0.9602 | 0.9602 |
| feasible-middle | absolute/A-non-decaying/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.991 |
| feasible-middle | absolute/A-contractive/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.991 |
| feasible-middle | absolute/R/K65 vs absolute/S | 65 | feasible | feasible | 130 | 0.006093 | 0.005676 | 5.707e-05 | 0.04984 | 0.05159 | -0.001637 | 1.004 | 0.9602 |
| feasible-middle | absolute/R-scaled/K65 vs absolute/S | 65 | feasible | feasible | 130 | 0.005676 | 0.005676 | -1.358e-12 | 0.05159 | 0.05159 | -3.331e-12 | 0.9602 | 0.9602 |
| feasible-middle | absolute/A-non-decaying/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.004 |
| feasible-middle | absolute/A-contractive/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.004 |
| feasible-middle | residual/R/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-middle | residual/R-scaled/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-middle | residual/R/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-middle | residual/R-scaled/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-middle | residual/R/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-middle | residual/R-scaled/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-middle | residual/S vs absolute/S | 1 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9602 |
| feasible-middle | residual/R/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9838 |
| feasible-middle | residual/R/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.991 |
| feasible-middle | residual/R/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.004 |
| feasible-middle | residual/R-scaled/K17 vs absolute/R-scaled/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9602 |
| feasible-middle | residual/R-scaled/K33 vs absolute/R-scaled/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9602 |
| feasible-middle | residual/R-scaled/K65 vs absolute/R-scaled/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.9602 |
| feasible-worst | absolute/R/K17 vs absolute/S | 17 | rc_gate_failure | feasible | 83 | 0.004727 | 0.00611 | -0.0006399 | 0.0533 | 0.05721 | -0.00414 | n/a | 1.286 |
| feasible-worst | absolute/R-scaled/K17 vs absolute/S | 17 | feasible | feasible | 130 | 0.005498 | 0.005498 | 5.648e-14 | 0.05556 | 0.05556 | 7.464e-14 | 1.286 | 1.286 |
| feasible-worst | absolute/A-non-decaying/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | absolute/A-contractive/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | absolute/R/K33 vs absolute/S | 33 | rc_gate_failure | feasible | 83 | 0.004711 | 0.00611 | -0.0007447 | 0.05227 | 0.05721 | -0.005289 | n/a | 1.286 |
| feasible-worst | absolute/R-scaled/K33 vs absolute/S | 33 | feasible | feasible | 130 | 0.005498 | 0.005498 | 6.063e-14 | 0.05556 | 0.05556 | 1.236e-12 | 1.286 | 1.286 |
| feasible-worst | absolute/A-non-decaying/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | absolute/A-contractive/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | absolute/R/K65 vs absolute/S | 65 | rc_gate_failure | feasible | 8 | 0.003506 | 0.00345 | -0.0008637 | 0.04743 | 0.05342 | -0.007945 | n/a | 1.286 |
| feasible-worst | absolute/R-scaled/K65 vs absolute/S | 65 | feasible | feasible | 130 | 0.005498 | 0.005498 | 1.974e-13 | 0.05556 | 0.05556 | 1.669e-12 | 1.286 | 1.286 |
| feasible-worst | absolute/A-non-decaying/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | absolute/A-contractive/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R-scaled/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R-scaled/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R-scaled/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/S vs absolute/S | 1 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.286 |
| feasible-worst | residual/R/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| feasible-worst | residual/R-scaled/K17 vs absolute/R-scaled/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.286 |
| feasible-worst | residual/R-scaled/K33 vs absolute/R-scaled/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.286 |
| feasible-worst | residual/R-scaled/K65 vs absolute/R-scaled/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.286 |
| failure-actual-dwell | absolute/R/K17 vs absolute/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | absolute/R-scaled/K17 vs absolute/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | absolute/A-non-decaying/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 44 | 0.00209 | 0.004046 | -0.001567 | 0.03878 | 0.05096 | -0.0215 | n/a | n/a |
| failure-actual-dwell | absolute/A-contractive/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 83 | 0.002908 | 0.005024 | -0.001935 | 0.04529 | 0.06996 | -0.0262 | n/a | n/a |
| failure-actual-dwell | absolute/R/K33 vs absolute/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | absolute/R-scaled/K33 vs absolute/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | absolute/A-non-decaying/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 2 | 0.001197 | 0.0005956 | 0.0006011 | 0.01585 | 0.01214 | 0.003711 | n/a | n/a |
| failure-actual-dwell | absolute/A-contractive/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 83 | 0.0029 | 0.004641 | -0.002368 | 0.03538 | 0.06311 | -0.0288 | n/a | n/a |
| failure-actual-dwell | absolute/R/K65 vs absolute/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | absolute/R-scaled/K65 vs absolute/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | absolute/A-non-decaying/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | absolute/A-contractive/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 83 | 0.002152 | 0.004408 | -0.002603 | 0.02443 | 0.05835 | -0.03494 | n/a | n/a |
| failure-actual-dwell | residual/R/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R-scaled/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R-scaled/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R-scaled/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/S vs absolute/S | 1 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R-scaled/K17 vs absolute/R-scaled/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R-scaled/K33 vs absolute/R-scaled/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-actual-dwell | residual/R-scaled/K65 vs absolute/R-scaled/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-joint-velocity | absolute/R/K17 vs absolute/S | 17 | feasible | feasible | 130 | 0.004205 | 0.004719 | -0.0005564 | 0.04926 | 0.05007 | -0.004163 | 1.034 | 1.344 |
| failure-joint-velocity | absolute/R-scaled/K17 vs absolute/S | 17 | feasible | feasible | 130 | 0.004719 | 0.004719 | 3.252e-14 | 0.05007 | 0.05007 | 1.166e-12 | 1.344 | 1.344 |
| failure-joint-velocity | absolute/A-non-decaying/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.034 |
| failure-joint-velocity | absolute/A-contractive/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 82 | 0.0006583 | 0.004478 | -0.003885 | 0.007034 | 0.05085 | -0.0472 | n/a | 1.034 |
| failure-joint-velocity | absolute/R/K33 vs absolute/S | 33 | feasible | feasible | 130 | 0.004247 | 0.004719 | -0.0005433 | 0.04928 | 0.05007 | -0.003841 | 1.045 | 1.344 |
| failure-joint-velocity | absolute/R-scaled/K33 vs absolute/S | 33 | feasible | feasible | 130 | 0.004719 | 0.004719 | 5.818e-14 | 0.05007 | 0.05007 | 1.739e-12 | 1.344 | 1.344 |
| failure-joint-velocity | absolute/A-non-decaying/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.045 |
| failure-joint-velocity | absolute/A-contractive/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 6 | 0.0002821 | 0.001895 | -0.00139 | 0.002338 | 0.04704 | -0.04166 | n/a | 1.045 |
| failure-joint-velocity | absolute/R/K65 vs absolute/S | 65 | feasible | feasible | 130 | 0.004423 | 0.004719 | -0.0002777 | 0.04943 | 0.05007 | -0.002808 | 1.123 | 1.344 |
| failure-joint-velocity | absolute/R-scaled/K65 vs absolute/S | 65 | feasible | feasible | 130 | 0.004719 | 0.004719 | -2.841e-14 | 0.05007 | 0.05007 | 2.368e-13 | 1.344 | 1.344 |
| failure-joint-velocity | absolute/A-non-decaying/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.123 |
| failure-joint-velocity | absolute/A-contractive/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.123 |
| failure-joint-velocity | residual/R/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-joint-velocity | residual/R-scaled/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-joint-velocity | residual/R/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-joint-velocity | residual/R-scaled/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-joint-velocity | residual/R/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-joint-velocity | residual/R-scaled/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-joint-velocity | residual/S vs absolute/S | 1 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.344 |
| failure-joint-velocity | residual/R/K17 vs absolute/R/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.034 |
| failure-joint-velocity | residual/R/K33 vs absolute/R/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.045 |
| failure-joint-velocity | residual/R/K65 vs absolute/R/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.123 |
| failure-joint-velocity | residual/R-scaled/K17 vs absolute/R-scaled/K17 | 17 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.344 |
| failure-joint-velocity | residual/R-scaled/K33 vs absolute/R-scaled/K33 | 33 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.344 |
| failure-joint-velocity | residual/R-scaled/K65 vs absolute/R-scaled/K65 | 65 | rc_gate_failure | feasible | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 1.344 |
| failure-generated-dwell | absolute/R/K17 vs absolute/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/R-scaled/K17 vs absolute/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/A-non-decaying/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/A-contractive/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/R/K33 vs absolute/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/R-scaled/K33 vs absolute/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/A-non-decaying/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/A-contractive/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/R/K65 vs absolute/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/R-scaled/K65 vs absolute/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/A-non-decaying/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | absolute/A-contractive/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R-scaled/K17 vs residual/S | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R-scaled/K33 vs residual/S | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R-scaled/K65 vs residual/S | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/S vs absolute/S | 1 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R/K17 vs absolute/R/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R/K33 vs absolute/R/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R/K65 vs absolute/R/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R-scaled/K17 vs absolute/R-scaled/K17 | 17 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R-scaled/K33 vs absolute/R-scaled/K33 | 33 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| failure-generated-dwell | residual/R-scaled/K65 vs absolute/R-scaled/K65 | 65 | rc_gate_failure | rc_gate_failure | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Numerical equivalence (M3REP-003)

| entry | formulation | K | candidate | reference | quantity | max abs | max rel | coef fro rel | decision |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| feasible-best | absolute | 17 | R/K17 | S-effective/K17 | prediction | 3.714e-11 | 7.908e-11 | 6.662e-10 | pass |
| feasible-best | absolute | 17 | R-scaled/K17 | S | prediction | 2.019e-12 | 5.557e-12 | 4.664e-11 | pass |
| feasible-best | absolute | 33 | R/K33 | S-effective/K33 | prediction | 5.277e-11 | 1.044e-10 | 1.389e-09 | pass |
| feasible-best | absolute | 33 | R-scaled/K33 | S | prediction | 1.432e-12 | 4.093e-12 | 4.934e-11 | pass |
| feasible-best | absolute | 65 | R/K65 | S-effective/K65 | prediction | 1.756e-10 | 4.966e-10 | 3.371e-09 | pass |
| feasible-best | absolute | 65 | R-scaled/K65 | S | prediction | 1.515e-12 | 6.082e-12 | 5.375e-11 | pass |
| feasible-best | residual | 17 | R/K17 | S-effective/K17 | increment | 1.071e-13 | 3.755e-06 | 1.665e-10 | pass |
| feasible-best | residual | 17 | R/K17 | S-effective/K17 | command | 1.071e-13 | 1.266e-12 | 1.665e-10 | pass |
| feasible-best | residual | 17 | R-scaled/K17 | S | increment | 5.802e-15 | 1.385e-08 | 8.935e-12 | pass |
| feasible-best | residual | 17 | R-scaled/K17 | S | command | 5.829e-15 | 6.738e-14 | 8.935e-12 | pass |
| feasible-best | residual | 33 | R/K33 | S-effective/K33 | increment | 2.239e-13 | 1.768e-06 | 3.252e-10 | pass |
| feasible-best | residual | 33 | R/K33 | S-effective/K33 | command | 2.239e-13 | 2.695e-12 | 3.252e-10 | pass |
| feasible-best | residual | 33 | R-scaled/K33 | S | increment | 6.945e-15 | 1.366e-08 | 1.045e-11 | pass |
| feasible-best | residual | 33 | R-scaled/K33 | S | command | 6.994e-15 | 7.511e-14 | 1.045e-11 | pass |
| feasible-best | residual | 65 | R/K65 | S-effective/K65 | increment | 6.446e-13 | 9.777e-07 | 8.051e-10 | pass |
| feasible-best | residual | 65 | R/K65 | S-effective/K65 | command | 6.446e-13 | 4.484e-12 | 8.051e-10 | pass |
| feasible-best | residual | 65 | R-scaled/K65 | S | increment | 6.333e-15 | 1.757e-08 | 1.078e-11 | pass |
| feasible-best | residual | 65 | R-scaled/K65 | S | command | 6.328e-15 | 8.211e-14 | 1.078e-11 | pass |
| feasible-middle | absolute | 17 | R/K17 | S-effective/K17 | prediction | 2.881e-09 | 2.423e-09 | 3.290e-08 | pass |
| feasible-middle | absolute | 17 | R-scaled/K17 | S | prediction | 1.079e-10 | 1.668e-10 | 1.929e-09 | pass |
| feasible-middle | absolute | 33 | R/K33 | S-effective/K33 | prediction | 5.595e-09 | 4.745e-09 | 1.082e-07 | pass |
| feasible-middle | absolute | 33 | R-scaled/K33 | S | prediction | 1.524e-10 | 1.674e-10 | 3.311e-09 | pass |
| feasible-middle | absolute | 65 | R/K65 | S-effective/K65 | prediction | 2.929e-08 | 5.003e-08 | 3.906e-07 | **accepted exception (C11)** |
| feasible-middle | absolute | 65 | R-scaled/K65 | S | prediction | 4.456e-10 | 6.708e-10 | 6.256e-09 | pass |
| feasible-middle | residual | 17 | R/K17 | S-effective/K17 | increment | 6.073e-12 | 3.393e-05 | 9.539e-09 | pass |
| feasible-middle | residual | 17 | R/K17 | S-effective/K17 | command | 6.073e-12 | 3.960e-11 | 9.539e-09 | pass |
| feasible-middle | residual | 17 | R-scaled/K17 | S | increment | 2.540e-13 | 2.041e-06 | 5.794e-10 | pass |
| feasible-middle | residual | 17 | R-scaled/K17 | S | command | 2.540e-13 | 1.868e-12 | 5.794e-10 | pass |
| feasible-middle | residual | 33 | R/K33 | S-effective/K33 | increment | 1.861e-11 | 1.763e-04 | 1.770e-08 | pass |
| feasible-middle | residual | 33 | R/K33 | S-effective/K33 | command | 1.861e-11 | 6.470e-11 | 1.770e-08 | pass |
| feasible-middle | residual | 33 | R-scaled/K33 | S | increment | 2.440e-13 | 1.572e-06 | 5.602e-10 | pass |
| feasible-middle | residual | 33 | R-scaled/K33 | S | command | 2.440e-13 | 2.501e-12 | 5.602e-10 | pass |
| feasible-middle | residual | 65 | R/K65 | S-effective/K65 | increment | 1.641e-11 | 2.165e-04 | 3.579e-08 | pass |
| feasible-middle | residual | 65 | R/K65 | S-effective/K65 | command | 1.641e-11 | 1.194e-10 | 3.579e-08 | pass |
| feasible-middle | residual | 65 | R-scaled/K65 | S | increment | 2.696e-13 | 1.364e-06 | 5.842e-10 | pass |
| feasible-middle | residual | 65 | R-scaled/K65 | S | command | 2.696e-13 | 1.671e-12 | 5.842e-10 | pass |
| feasible-worst | absolute | 17 | R/K17 | S-effective/K17 | prediction | 8.556e-11 | 4.593e-10 | 6.135e-09 | pass |
| feasible-worst | absolute | 17 | R-scaled/K17 | S | prediction | 2.929e-12 | 1.488e-11 | 5.870e-10 | pass |
| feasible-worst | absolute | 33 | R/K33 | S-effective/K33 | prediction | 1.409e-10 | 5.989e-10 | 1.309e-08 | pass |
| feasible-worst | absolute | 33 | R-scaled/K33 | S | prediction | 3.328e-12 | 1.354e-11 | 6.528e-10 | pass |
| feasible-worst | absolute | 65 | R/K65 | S-effective/K65 | prediction | 4.804e-10 | 1.509e-09 | 3.778e-08 | pass |
| feasible-worst | absolute | 65 | R-scaled/K65 | S | prediction | 4.178e-12 | 2.006e-11 | 8.678e-10 | pass |
| feasible-worst | residual | 17 | R/K17 | S-effective/K17 | increment | 1.369e-12 | 1.362e-05 | 4.486e-09 | pass |
| feasible-worst | residual | 17 | R/K17 | S-effective/K17 | command | 1.369e-12 | 1.191e-11 | 4.486e-09 | pass |
| feasible-worst | residual | 17 | R-scaled/K17 | S | increment | 4.383e-14 | 7.152e-06 | 2.876e-10 | pass |
| feasible-worst | residual | 17 | R-scaled/K17 | S | command | 4.383e-14 | 3.437e-13 | 2.876e-10 | pass |
| feasible-worst | residual | 33 | R/K33 | S-effective/K33 | increment | 3.309e-12 | 2.729e-04 | 8.312e-09 | pass |
| feasible-worst | residual | 33 | R/K33 | S-effective/K33 | command | 3.309e-12 | 2.766e-11 | 8.312e-09 | pass |
| feasible-worst | residual | 33 | R-scaled/K33 | S | increment | 5.427e-14 | 6.119e-06 | 3.275e-10 | pass |
| feasible-worst | residual | 33 | R-scaled/K33 | S | command | 5.429e-14 | 5.192e-13 | 3.275e-10 | pass |
| feasible-worst | residual | 65 | R/K65 | S-effective/K65 | increment | 1.402e-11 | 1.925e-04 | 2.292e-08 | pass |
| feasible-worst | residual | 65 | R/K65 | S-effective/K65 | command | 1.402e-11 | 1.301e-10 | 2.292e-08 | pass |
| feasible-worst | residual | 65 | R-scaled/K65 | S | increment | 5.568e-14 | 1.425e-05 | 4.059e-10 | pass |
| feasible-worst | residual | 65 | R-scaled/K65 | S | command | 5.568e-14 | 4.721e-13 | 4.059e-10 | pass |
| failure-actual-dwell | absolute | 17 | R/K17 | S-effective/K17 | prediction | 1.575e-12 | 1.361e-12 | 5.635e-11 | pass |
| failure-actual-dwell | absolute | 17 | R-scaled/K17 | S | prediction | 1.783e-13 | 2.348e-13 | 3.836e-12 | pass |
| failure-actual-dwell | absolute | 33 | R/K33 | S-effective/K33 | prediction | 1.916e-11 | 1.664e-11 | 1.925e-10 | pass |
| failure-actual-dwell | absolute | 33 | R-scaled/K33 | S | prediction | 4.663e-13 | 4.055e-13 | 6.187e-12 | pass |
| failure-actual-dwell | absolute | 65 | R/K65 | S-effective/K65 | prediction | 4.786e-11 | 4.160e-11 | 6.284e-10 | pass |
| failure-actual-dwell | absolute | 65 | R-scaled/K65 | S | prediction | 5.598e-13 | 1.050e-12 | 1.030e-11 | pass |
| failure-actual-dwell | residual | 17 | R/K17 | S-effective/K17 | increment | 4.358e-15 | 1.343e-07 | 1.707e-11 | pass |
| failure-actual-dwell | residual | 17 | R/K17 | S-effective/K17 | command | 4.441e-15 | 1.104e-14 | 1.707e-11 | pass |
| failure-actual-dwell | residual | 17 | R-scaled/K17 | S | increment | 2.654e-16 | 5.914e-08 | 9.716e-13 | pass |
| failure-actual-dwell | residual | 17 | R-scaled/K17 | S | command | 2.637e-16 | 3.290e-15 | 9.716e-13 | pass |
| failure-actual-dwell | residual | 33 | R/K33 | S-effective/K33 | increment | 7.796e-15 | 4.479e-08 | 2.987e-11 | pass |
| failure-actual-dwell | residual | 33 | R/K33 | S-effective/K33 | command | 7.772e-15 | 2.832e-14 | 2.987e-11 | pass |
| failure-actual-dwell | residual | 33 | R-scaled/K33 | S | increment | 4.122e-16 | 5.025e-08 | 9.394e-13 | pass |
| failure-actual-dwell | residual | 33 | R-scaled/K33 | S | command | 4.163e-16 | 4.945e-15 | 9.394e-13 | pass |
| failure-actual-dwell | residual | 65 | R/K65 | S-effective/K65 | increment | 2.670e-14 | 1.615e-06 | 6.533e-11 | pass |
| failure-actual-dwell | residual | 65 | R/K65 | S-effective/K65 | command | 2.676e-14 | 1.745e-13 | 6.533e-11 | pass |
| failure-actual-dwell | residual | 65 | R-scaled/K65 | S | increment | 6.457e-16 | 5.902e-08 | 1.025e-12 | pass |
| failure-actual-dwell | residual | 65 | R-scaled/K65 | S | command | 6.384e-16 | 7.418e-15 | 1.025e-12 | pass |
| failure-joint-velocity | absolute | 17 | R/K17 | S-effective/K17 | prediction | 9.117e-11 | 2.436e-10 | 1.516e-08 | pass |
| failure-joint-velocity | absolute | 17 | R-scaled/K17 | S | prediction | 6.148e-12 | 1.603e-11 | 1.529e-09 | pass |
| failure-joint-velocity | absolute | 33 | R/K33 | S-effective/K33 | prediction | 2.290e-10 | 1.195e-09 | 2.587e-08 | pass |
| failure-joint-velocity | absolute | 33 | R-scaled/K33 | S | prediction | 8.288e-12 | 1.401e-11 | 1.507e-09 | pass |
| failure-joint-velocity | absolute | 65 | R/K65 | S-effective/K65 | prediction | 6.958e-10 | 3.603e-09 | 4.929e-08 | pass |
| failure-joint-velocity | absolute | 65 | R-scaled/K65 | S | prediction | 4.718e-12 | 1.827e-11 | 1.496e-09 | pass |
| failure-joint-velocity | residual | 17 | R/K17 | S-effective/K17 | increment | 1.991e-12 | 9.294e-05 | 8.553e-09 | pass |
| failure-joint-velocity | residual | 17 | R/K17 | S-effective/K17 | command | 1.991e-12 | 1.612e-11 | 8.553e-09 | pass |
| failure-joint-velocity | residual | 17 | R-scaled/K17 | S | increment | 7.800e-14 | 5.789e-07 | 5.276e-10 | pass |
| failure-joint-velocity | residual | 17 | R-scaled/K17 | S | command | 7.799e-14 | 7.375e-13 | 5.276e-10 | pass |
| failure-joint-velocity | residual | 33 | R/K33 | S-effective/K33 | increment | 4.755e-12 | 1.974e-05 | 1.713e-08 | pass |
| failure-joint-velocity | residual | 33 | R/K33 | S-effective/K33 | command | 4.756e-12 | 4.762e-11 | 1.713e-08 | pass |
| failure-joint-velocity | residual | 33 | R-scaled/K33 | S | increment | 1.540e-13 | 3.917e-07 | 5.442e-10 | pass |
| failure-joint-velocity | residual | 33 | R-scaled/K33 | S | command | 1.540e-13 | 1.199e-12 | 5.442e-10 | pass |
| failure-joint-velocity | residual | 65 | R/K65 | S-effective/K65 | increment | 1.059e-11 | 4.397e-05 | 3.338e-08 | pass |
| failure-joint-velocity | residual | 65 | R/K65 | S-effective/K65 | command | 1.059e-11 | 9.324e-11 | 3.338e-08 | pass |
| failure-joint-velocity | residual | 65 | R-scaled/K65 | S | increment | 7.939e-14 | 4.068e-07 | 5.270e-10 | pass |
| failure-joint-velocity | residual | 65 | R-scaled/K65 | S | command | 7.938e-14 | 5.501e-13 | 5.270e-10 | pass |
| failure-generated-dwell | absolute | 17 | R/K17 | S-effective/K17 | prediction | 1.686e-11 | 1.522e-11 | 9.830e-10 | pass |
| failure-generated-dwell | absolute | 17 | R-scaled/K17 | S | prediction | 8.726e-13 | 2.702e-12 | 6.447e-11 | pass |
| failure-generated-dwell | absolute | 33 | R/K33 | S-effective/K33 | prediction | 3.480e-11 | 8.603e-11 | 3.030e-09 | pass |
| failure-generated-dwell | absolute | 33 | R-scaled/K33 | S | prediction | 1.316e-12 | 4.379e-12 | 9.719e-11 | pass |
| failure-generated-dwell | absolute | 65 | R/K65 | S-effective/K65 | prediction | 3.617e-10 | 4.743e-10 | 1.164e-08 | pass |
| failure-generated-dwell | absolute | 65 | R-scaled/K65 | S | prediction | 5.221e-12 | 5.586e-12 | 1.881e-10 | pass |
| failure-generated-dwell | residual | 17 | R/K17 | S-effective/K17 | increment | 2.252e-13 | 1.482e-05 | 2.745e-10 | pass |
| failure-generated-dwell | residual | 17 | R/K17 | S-effective/K17 | command | 2.252e-13 | 2.213e-12 | 2.745e-10 | pass |
| failure-generated-dwell | residual | 17 | R-scaled/K17 | S | increment | 6.733e-15 | 2.194e-06 | 1.577e-11 | pass |
| failure-generated-dwell | residual | 17 | R-scaled/K17 | S | command | 6.717e-15 | 6.619e-14 | 1.577e-11 | pass |
| failure-generated-dwell | residual | 33 | R/K33 | S-effective/K33 | increment | 3.791e-13 | 4.308e-05 | 5.215e-10 | pass |
| failure-generated-dwell | residual | 33 | R/K33 | S-effective/K33 | command | 3.791e-13 | 3.959e-12 | 5.215e-10 | pass |
| failure-generated-dwell | residual | 33 | R-scaled/K33 | S | increment | 4.754e-15 | 1.827e-06 | 1.513e-11 | pass |
| failure-generated-dwell | residual | 33 | R-scaled/K33 | S | command | 4.746e-15 | 4.715e-14 | 1.513e-11 | pass |
| failure-generated-dwell | residual | 65 | R/K65 | S-effective/K65 | increment | 7.504e-13 | 3.310e-06 | 1.028e-09 | pass |
| failure-generated-dwell | residual | 65 | R/K65 | S-effective/K65 | command | 7.504e-13 | 6.436e-12 | 1.028e-09 | pass |
| failure-generated-dwell | residual | 65 | R-scaled/K65 | S | increment | 6.121e-15 | 1.968e-06 | 1.598e-11 | pass |
| failure-generated-dwell | residual | 65 | R-scaled/K65 | S | command | 6.134e-15 | 7.097e-14 | 1.598e-11 | pass |

## Speed diagnostics

| entry | arm | executed | crossed 6 rad/s | peak rad/s | aborts at 12 | above 6 warm-up s | above 6 movement s | above 6 dwell s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feasible-best | absolute/S | 130 | 0 | 5.97 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/R/K17 | 130 | 0 | 5.9 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/R-scaled/K17 | 130 | 0 | 5.97 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/A-non-decaying/K17 | 1 | 0 | 4.38 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/A-contractive/K17 | 1 | 0 | 0.577 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/R/K33 | 130 | 0 | 5.9 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/R-scaled/K33 | 130 | 0 | 5.97 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/A-non-decaying/K33 | 1 | 0 | 5.79 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/A-contractive/K33 | 1 | 1 | 7.92 | 0 | 0.00 | 0.12 | 0.18 |
| feasible-best | absolute/R/K65 | 130 | 0 | 5.92 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/R-scaled/K65 | 130 | 0 | 5.97 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | absolute/A-non-decaying/K65 | 1 | 1 | 12.1 | 1 | 0.00 | 0.38 | 0.00 |
| feasible-best | absolute/A-contractive/K65 | 1 | 1 | 12.4 | 1 | 0.00 | 0.09 | 0.00 |
| feasible-best | residual/S | 1 | 1 | 7.38 | 0 | 0.00 | 0.00 | 0.22 |
| feasible-best | residual/R/K17 | 1 | 0 | 4.25 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | residual/R-scaled/K17 | 1 | 1 | 7.38 | 0 | 0.00 | 0.00 | 0.22 |
| feasible-best | residual/R/K33 | 1 | 0 | 3.19 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | residual/R-scaled/K33 | 1 | 1 | 7.38 | 0 | 0.00 | 0.00 | 0.22 |
| feasible-best | residual/R/K65 | 1 | 0 | 4.35 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-best | residual/R-scaled/K65 | 1 | 1 | 7.38 | 0 | 0.00 | 0.00 | 0.22 |
| feasible-middle | absolute/S | 130 | 0 | 5.99 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-middle | absolute/R/K17 | 130 | 6 | 6 | 0 | 0.00 | 0.06 | 0.00 |
| feasible-middle | absolute/R-scaled/K17 | 130 | 0 | 5.99 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-middle | absolute/A-non-decaying/K17 | 1 | 1 | 13.2 | 1 | 0.00 | 0.06 | 0.00 |
| feasible-middle | absolute/A-contractive/K17 | 1 | 1 | 12.3 | 1 | 0.00 | 0.05 | 0.00 |
| feasible-middle | absolute/R/K33 | 130 | 6 | 6.01 | 0 | 0.00 | 0.06 | 0.00 |
| feasible-middle | absolute/R-scaled/K33 | 130 | 0 | 5.99 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-middle | absolute/A-non-decaying/K33 | 1 | 1 | 12.3 | 1 | 0.00 | 0.04 | 0.00 |
| feasible-middle | absolute/A-contractive/K33 | 1 | 1 | 13 | 1 | 0.00 | 0.05 | 0.00 |
| feasible-middle | absolute/R/K65 | 130 | 6 | 6.01 | 0 | 0.00 | 0.06 | 0.00 |
| feasible-middle | absolute/R-scaled/K65 | 130 | 0 | 5.99 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-middle | absolute/A-non-decaying/K65 | 1 | 1 | 14.7 | 1 | 0.00 | 0.06 | 0.00 |
| feasible-middle | absolute/A-contractive/K65 | 1 | 1 | 14.4 | 1 | 0.00 | 0.12 | 0.00 |
| feasible-middle | residual/S | 1 | 1 | 12.3 | 1 | 0.00 | 0.26 | 0.09 |
| feasible-middle | residual/R/K17 | 1 | 1 | 12.2 | 1 | 0.00 | 0.19 | 0.33 |
| feasible-middle | residual/R-scaled/K17 | 1 | 1 | 12.3 | 1 | 0.00 | 0.26 | 0.09 |
| feasible-middle | residual/R/K33 | 1 | 1 | 12 | 1 | 0.00 | 0.38 | 0.00 |
| feasible-middle | residual/R-scaled/K33 | 1 | 1 | 12.3 | 1 | 0.00 | 0.26 | 0.09 |
| feasible-middle | residual/R/K65 | 1 | 1 | 12.3 | 1 | 0.00 | 0.19 | 0.00 |
| feasible-middle | residual/R-scaled/K65 | 1 | 1 | 12.3 | 1 | 0.00 | 0.26 | 0.09 |
| feasible-worst | absolute/S | 130 | 0 | 5.98 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/R/K17 | 84 | 0 | 5.98 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/R-scaled/K17 | 130 | 0 | 5.98 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/A-non-decaying/K17 | 1 | 0 | 0.43 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/A-contractive/K17 | 1 | 0 | 0.41 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/R/K33 | 84 | 0 | 5.98 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/R-scaled/K33 | 130 | 0 | 5.98 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/A-non-decaying/K33 | 1 | 0 | 0.432 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/A-contractive/K33 | 1 | 0 | 0.413 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/R/K65 | 9 | 0 | 2.75 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/R-scaled/K65 | 130 | 0 | 5.98 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/A-non-decaying/K65 | 1 | 0 | 0.433 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | absolute/A-contractive/K65 | 1 | 0 | 0.415 | 0 | 0.00 | 0.00 | 0.00 |
| feasible-worst | residual/S | 1 | 1 | 12 | 1 | 0.00 | 0.13 | 0.39 |
| feasible-worst | residual/R/K17 | 1 | 1 | 9.22 | 0 | 0.00 | 0.07 | 0.24 |
| feasible-worst | residual/R-scaled/K17 | 1 | 1 | 12 | 1 | 0.00 | 0.13 | 0.39 |
| feasible-worst | residual/R/K33 | 1 | 1 | 8.5 | 0 | 0.00 | 0.24 | 0.04 |
| feasible-worst | residual/R-scaled/K33 | 1 | 1 | 12 | 1 | 0.00 | 0.13 | 0.39 |
| feasible-worst | residual/R/K65 | 1 | 1 | 12.6 | 1 | 0.00 | 0.15 | 0.00 |
| feasible-worst | residual/R-scaled/K65 | 1 | 1 | 12 | 1 | 0.00 | 0.13 | 0.39 |
| failure-actual-dwell | absolute/S | 1 | 0 | 4.61 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/R/K17 | 84 | 4 | 6.33 | 0 | 0.00 | 0.04 | 0.00 |
| failure-actual-dwell | absolute/R-scaled/K17 | 1 | 0 | 4.61 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/A-non-decaying/K17 | 45 | 0 | 3.78 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/A-contractive/K17 | 84 | 1 | 6.8 | 0 | 0.00 | 0.09 | 0.00 |
| failure-actual-dwell | absolute/R/K33 | 84 | 0 | 5.97 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/R-scaled/K33 | 1 | 0 | 4.61 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/A-non-decaying/K33 | 3 | 0 | 1.19 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/A-contractive/K33 | 84 | 1 | 7.06 | 0 | 0.00 | 0.11 | 0.00 |
| failure-actual-dwell | absolute/R/K65 | 84 | 0 | 5.97 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/R-scaled/K65 | 1 | 0 | 4.61 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/A-non-decaying/K65 | 1 | 0 | 0.49 | 0 | 0.00 | 0.00 | 0.00 |
| failure-actual-dwell | absolute/A-contractive/K65 | 84 | 1 | 7.46 | 0 | 0.00 | 0.12 | 0.00 |
| failure-actual-dwell | residual/S | 1 | 1 | 12.7 | 1 | 0.00 | 0.43 | 0.00 |
| failure-actual-dwell | residual/R/K17 | 1 | 1 | 11.1 | 0 | 0.00 | 0.29 | 0.00 |
| failure-actual-dwell | residual/R-scaled/K17 | 1 | 1 | 12.7 | 1 | 0.00 | 0.43 | 0.00 |
| failure-actual-dwell | residual/R/K33 | 1 | 1 | 11 | 0 | 0.00 | 0.28 | 0.00 |
| failure-actual-dwell | residual/R-scaled/K33 | 1 | 1 | 12.7 | 1 | 0.00 | 0.43 | 0.00 |
| failure-actual-dwell | residual/R/K65 | 1 | 1 | 10.9 | 0 | 0.00 | 0.27 | 0.00 |
| failure-actual-dwell | residual/R-scaled/K65 | 1 | 1 | 12.7 | 1 | 0.00 | 0.43 | 0.00 |
| failure-joint-velocity | absolute/S | 130 | 5 | 6.16 | 0 | 0.00 | 0.05 | 0.00 |
| failure-joint-velocity | absolute/R/K17 | 130 | 1 | 6.08 | 0 | 0.00 | 0.01 | 0.00 |
| failure-joint-velocity | absolute/R-scaled/K17 | 130 | 5 | 6.16 | 0 | 0.00 | 0.05 | 0.00 |
| failure-joint-velocity | absolute/A-non-decaying/K17 | 1 | 0 | 0.539 | 0 | 0.00 | 0.00 | 0.00 |
| failure-joint-velocity | absolute/A-contractive/K17 | 83 | 0 | 5.63 | 0 | 0.00 | 0.00 | 0.00 |
| failure-joint-velocity | absolute/R/K33 | 130 | 2 | 6.09 | 0 | 0.00 | 0.02 | 0.00 |
| failure-joint-velocity | absolute/R-scaled/K33 | 130 | 5 | 6.16 | 0 | 0.00 | 0.05 | 0.00 |
| failure-joint-velocity | absolute/A-non-decaying/K33 | 1 | 0 | 0.565 | 0 | 0.00 | 0.00 | 0.00 |
| failure-joint-velocity | absolute/A-contractive/K33 | 7 | 0 | 0.465 | 0 | 0.00 | 0.00 | 0.00 |
| failure-joint-velocity | absolute/R/K65 | 130 | 3 | 6.11 | 0 | 0.00 | 0.03 | 0.00 |
| failure-joint-velocity | absolute/R-scaled/K65 | 130 | 5 | 6.16 | 0 | 0.00 | 0.05 | 0.00 |
| failure-joint-velocity | absolute/A-non-decaying/K65 | 1 | 0 | 0.676 | 0 | 0.00 | 0.00 | 0.00 |
| failure-joint-velocity | absolute/A-contractive/K65 | 1 | 0 | 0.476 | 0 | 0.00 | 0.00 | 0.00 |
| failure-joint-velocity | residual/S | 1 | 1 | 12.2 | 1 | 0.00 | 0.44 | 0.21 |
| failure-joint-velocity | residual/R/K17 | 1 | 1 | 12.3 | 1 | 0.00 | 0.10 | 0.00 |
| failure-joint-velocity | residual/R-scaled/K17 | 1 | 1 | 12.2 | 1 | 0.00 | 0.44 | 0.21 |
| failure-joint-velocity | residual/R/K33 | 1 | 1 | 12.9 | 1 | 0.00 | 0.10 | 0.00 |
| failure-joint-velocity | residual/R-scaled/K33 | 1 | 1 | 12.2 | 1 | 0.00 | 0.44 | 0.21 |
| failure-joint-velocity | residual/R/K65 | 1 | 1 | 12.5 | 1 | 0.00 | 0.08 | 0.00 |
| failure-joint-velocity | residual/R-scaled/K65 | 1 | 1 | 12.2 | 1 | 0.00 | 0.44 | 0.21 |
| failure-generated-dwell | absolute/S | 1 | 0 | 2.56 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/R/K17 | 84 | 0 | 5.99 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/R-scaled/K17 | 1 | 0 | 2.56 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/A-non-decaying/K17 | 1 | 0 | 0.627 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/A-contractive/K17 | 1 | 0 | 0.509 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/R/K33 | 84 | 0 | 5.99 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/R-scaled/K33 | 1 | 0 | 2.56 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/A-non-decaying/K33 | 1 | 0 | 0.744 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/A-contractive/K33 | 1 | 0 | 0.556 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/R/K65 | 84 | 0 | 5.99 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/R-scaled/K65 | 1 | 0 | 2.56 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/A-non-decaying/K65 | 1 | 0 | 1.92 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | absolute/A-contractive/K65 | 1 | 0 | 0.658 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | residual/S | 1 | 1 | 12.9 | 1 | 0.00 | 0.64 | 0.00 |
| failure-generated-dwell | residual/R/K17 | 1 | 0 | 5.78 | 0 | 0.00 | 0.00 | 0.00 |
| failure-generated-dwell | residual/R-scaled/K17 | 1 | 1 | 12.9 | 1 | 0.00 | 0.64 | 0.00 |
| failure-generated-dwell | residual/R/K33 | 1 | 1 | 12.4 | 1 | 0.00 | 0.10 | 0.00 |
| failure-generated-dwell | residual/R-scaled/K33 | 1 | 1 | 12.9 | 1 | 0.00 | 0.64 | 0.00 |
| failure-generated-dwell | residual/R/K65 | 1 | 1 | 12 | 1 | 0.00 | 0.27 | 0.00 |
| failure-generated-dwell | residual/R-scaled/K65 | 1 | 1 | 12.9 | 1 | 0.00 | 0.64 | 0.00 |

## Measured cost by arm and count

Peak resident set size of the timing smoke check (process-cumulative, 20 models): 264.5 MiB; the panel run recorded no per-model memory figure.

| formulation | arm | K | models | fits timed | fit s mean | evaluation s mean | executed runs mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| absolute | S | 1 | 6 | 6 | 0.0394 | 21.7 | 87.0 |
| absolute | R/K17 | 17 | 6 | 6 | 0.307 | 27.1 | 107.0 |
| absolute | R-scaled/K17 | 17 | 6 | 6 | 0.303 | 21.5 | 87.0 |
| absolute | A-non-decaying/K17 | 17 | 6 | 6 | 1.35 | 1.8 | 8.3 |
| absolute | A-contractive/K17 | 17 | 6 | 6 | 1.35 | 6.9 | 28.5 |
| absolute | S-effective/K17 | 17 | 6 | 6 | 0.035 | 0.0 | 0.0 |
| absolute | R/K33 | 33 | 6 | 6 | 0.578 | 26.5 | 107.0 |
| absolute | R-scaled/K33 | 33 | 6 | 6 | 0.572 | 21.8 | 87.0 |
| absolute | A-non-decaying/K33 | 33 | 6 | 6 | 2.65 | 0.4 | 1.3 |
| absolute | A-contractive/K33 | 33 | 6 | 6 | 2.65 | 3.5 | 15.8 |
| absolute | S-effective/K33 | 33 | 6 | 6 | 0.0344 | 0.0 | 0.0 |
| absolute | R/K65 | 65 | 6 | 6 | 1.12 | 22.1 | 94.5 |
| absolute | R-scaled/K65 | 65 | 6 | 6 | 1.11 | 21.6 | 87.0 |
| absolute | A-non-decaying/K65 | 65 | 6 | 6 | 5.23 | 0.1 | 1.0 |
| absolute | A-contractive/K65 | 65 | 6 | 6 | 5.23 | 3.3 | 14.8 |
| absolute | S-effective/K65 | 65 | 6 | 6 | 0.0348 | 0.0 | 0.0 |
| residual | S | 1 | 6 | 6 | 0.0367 | 0.1 | 1.0 |
| residual | R/K17 | 17 | 6 | 6 | 0.31 | 0.1 | 1.0 |
| residual | R-scaled/K17 | 17 | 6 | 6 | 0.3 | 0.3 | 1.0 |
| residual | S-effective/K17 | 17 | 6 | 6 | 0.0354 | 0.0 | 0.0 |
| residual | R/K33 | 33 | 6 | 6 | 0.573 | 0.1 | 1.0 |
| residual | R-scaled/K33 | 33 | 6 | 6 | 0.571 | 0.1 | 1.0 |
| residual | S-effective/K33 | 33 | 6 | 6 | 0.0346 | 0.0 | 0.0 |
| residual | R/K65 | 65 | 6 | 6 | 1.12 | 0.1 | 1.0 |
| residual | R-scaled/K65 | 65 | 6 | 6 | 1.11 | 0.1 | 1.0 |
| residual | S-effective/K65 | 65 | 6 | 6 | 0.036 | 0.0 | 0.0 |

## Representatives

| entry | arm | pair | status | reason | RC run | replay run | plot | animations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feasible-best | absolute/S | nominal [pd_v2] | completed |  | run-20260910-79fe0985a4b1 | run-20260910-54e04960dcbb | trajectory_feasible-best__absolute__S.png | feasible-best__absolute__S__rc.gif, feasible-best__replay_pd_v2.gif |
| feasible-best | absolute/R/K65 | nominal [pd_v2] | completed |  | run-20260910-10f86520d538 | run-20260910-54e04960dcbb | trajectory_feasible-best__absolute__R__K65.png | feasible-best__absolute__R__K65__rc.gif, feasible-best__replay_pd_v2.gif |
| feasible-best | absolute/R-scaled/K65 | nominal [pd_v2] | completed |  | run-20260910-8b95b4d89d8d | run-20260910-54e04960dcbb | trajectory_feasible-best__absolute__R-scaled__K65.png |  |
| feasible-best | absolute/A-contractive/K65 | nominal [pd_v2] | infeasible | limit_violation:joint_velocity | run-20260910-fda843bf83ca | run-20260910-54e04960dcbb | trajectory_feasible-best__absolute__A-contractive__K65.png |  |
| feasible-best | residual/S | nominal [pd_v2] | infeasible | early_termination:invalid_output:bounds | run-20260910-4b19d25412c6 | run-20260910-54e04960dcbb | trajectory_feasible-best__residual__S.png |  |
| feasible-middle | absolute/S | nominal [pd_v2] | completed |  | run-20260910-a55ec4043444 | run-20260910-adca995e59cb | trajectory_feasible-middle__absolute__S.png |  |
| feasible-middle | absolute/R/K65 | nominal [pd_v2] | completed |  | run-20260910-ae2dbbc83249 | run-20260910-adca995e59cb | trajectory_feasible-middle__absolute__R__K65.png |  |
| feasible-middle | absolute/R-scaled/K65 | nominal [pd_v2] | completed |  | run-20260910-40a9e7eecc44 | run-20260910-adca995e59cb | trajectory_feasible-middle__absolute__R-scaled__K65.png |  |
| feasible-middle | absolute/A-contractive/K65 | nominal [pd_v2] | infeasible | limit_violation:joint_velocity | run-20260910-1c97e232fe44 | run-20260910-adca995e59cb | trajectory_feasible-middle__absolute__A-contractive__K65.png |  |
| feasible-middle | residual/S | nominal [pd_v2] | infeasible | limit_violation:joint_velocity | run-20260910-a1f8d840b7c3 | run-20260910-adca995e59cb | trajectory_feasible-middle__residual__S.png |  |
| feasible-worst | absolute/S | nominal [pd_v2] | completed |  | run-20260910-13b0350016ff | run-20260910-5b04c1b2d75b | trajectory_feasible-worst__absolute__S.png |  |
| feasible-worst | absolute/R/K65 | nominal [pd_v2] | completed |  | run-20260910-8b2864402409 | run-20260910-5b04c1b2d75b | trajectory_feasible-worst__absolute__R__K65.png |  |
| feasible-worst | absolute/R-scaled/K65 | nominal [pd_v2] | completed |  | run-20260910-aaefaca650d1 | run-20260910-5b04c1b2d75b | trajectory_feasible-worst__absolute__R-scaled__K65.png |  |
| feasible-worst | absolute/A-contractive/K65 | nominal [pd_v2] | infeasible | dwell:dwell_stationary | run-20260910-699542f24c3d | run-20260910-5b04c1b2d75b | trajectory_feasible-worst__absolute__A-contractive__K65.png |  |
| feasible-worst | residual/S | nominal [pd_v2] | infeasible | limit_violation:joint_velocity | run-20260910-5fbc2d702adf | run-20260910-5b04c1b2d75b | trajectory_feasible-worst__residual__S.png |  |
| failure-actual-dwell | absolute/S | nominal [pd_v2] | infeasible | dwell:dwell_stationary | run-20260910-a79b0e2b1acd | run-20260910-54e04960dcbb | trajectory_failure-actual-dwell__absolute__S.png |  |
| failure-actual-dwell | absolute/R/K65 | nominal [pd_v2] | completed |  | run-20260910-5f20de701560 | run-20260910-54e04960dcbb | trajectory_failure-actual-dwell__absolute__R__K65.png |  |
| failure-actual-dwell | absolute/R-scaled/K65 | nominal [pd_v2] | infeasible | dwell:dwell_stationary | run-20260910-d7e20d591ce9 | run-20260910-54e04960dcbb | trajectory_failure-actual-dwell__absolute__R-scaled__K65.png |  |
| failure-actual-dwell | absolute/A-contractive/K65 | nominal [pd_v2] | completed |  | run-20260910-da9d4b807c5b | run-20260910-54e04960dcbb | trajectory_failure-actual-dwell__absolute__A-contractive__K65.png |  |
| failure-actual-dwell | residual/S | nominal [pd_v2] | infeasible | limit_violation:joint_velocity | run-20260910-8e14b4a32370 | run-20260910-54e04960dcbb | trajectory_failure-actual-dwell__residual__S.png |  |
| failure-joint-velocity | absolute/S | nominal [pd_v2] | completed |  | run-20260910-d73d3b1082eb | run-20260910-5b04c1b2d75b | trajectory_failure-joint-velocity__absolute__S.png |  |
| failure-joint-velocity | absolute/R/K65 | nominal [pd_v2] | completed |  | run-20260910-03fb52817d75 | run-20260910-5b04c1b2d75b | trajectory_failure-joint-velocity__absolute__R__K65.png |  |
| failure-joint-velocity | absolute/R-scaled/K65 | nominal [pd_v2] | completed |  | run-20260910-f8d0d360d515 | run-20260910-5b04c1b2d75b | trajectory_failure-joint-velocity__absolute__R-scaled__K65.png |  |
| failure-joint-velocity | absolute/A-contractive/K65 | nominal [pd_v2] | infeasible | dwell:dwell_stationary | run-20260910-06b52f386f40 | run-20260910-5b04c1b2d75b | trajectory_failure-joint-velocity__absolute__A-contractive__K65.png |  |
| failure-joint-velocity | residual/S | nominal [pd_v2] | infeasible | limit_violation:joint_velocity | run-20260910-d7f02817da72 | run-20260910-5b04c1b2d75b | trajectory_failure-joint-velocity__residual__S.png |  |
| failure-generated-dwell | absolute/S | nominal [pd_v2] | infeasible | generated_dwell:generated_dwell_stationary | run-20260910-617e39d291a4 | run-20260910-adca995e59cb | trajectory_failure-generated-dwell__absolute__S.png |  |
| failure-generated-dwell | absolute/R/K65 | nominal [pd_v2] | completed |  | run-20260910-a76178494a85 | run-20260910-adca995e59cb | trajectory_failure-generated-dwell__absolute__R__K65.png |  |
| failure-generated-dwell | absolute/R-scaled/K65 | nominal [pd_v2] | infeasible | generated_dwell:generated_dwell_stationary | run-20260910-ae50472ef321 | run-20260910-adca995e59cb | trajectory_failure-generated-dwell__absolute__R-scaled__K65.png |  |
| failure-generated-dwell | absolute/A-contractive/K65 | nominal [pd_v2] | infeasible | dwell:dwell_in_tolerance,dwell_stationary | run-20260910-6b99b92e528e | run-20260910-adca995e59cb | trajectory_failure-generated-dwell__absolute__A-contractive__K65.png |  |
| failure-generated-dwell | residual/S | nominal [pd_v2] | infeasible | limit_violation:joint_velocity | run-20260910-45d92f049ec7 | run-20260910-adca995e59cb | trajectory_failure-generated-dwell__residual__S.png |  |

## Limitations

- The panel is six historical configurations of one demonstration; the augmentation anchor and seed bank are fixed; first-failure censoring hides later behavior (plan section 7.3).
- Feasibility is under the 12 rad/s evaluation abort; crossings of the historical 6 rad/s limit are reported apart and are not recovery-v1 successes.
- Signed differences use only the pairs both arms completed; comparisons with few shared pairs describe those pairs only.
