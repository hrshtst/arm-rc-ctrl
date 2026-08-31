# Perturbation pilot `perturbation-pilot-1a-v2`

Dataset `processed-20260830-feaf73e6663c`, scenario `configs/tasks/task_1a.toml`, commit `5b38e6614ec2`, baselines `computed_torque` (kp [808.611917423121, 277.88260825351], kd [9.613035006148348, 35.99857512413955]), `pd_v2` (kp [103.66390204385338, 26.057552331473403], kd [3.206354655071141, 0.11992321023424186]).

Rules: a level is *safe* when every baseline completes, meets every dwell criterion, and recovers (endpoint within tolerance of the reference) within 1 s (posture) / 1 s after the pulse (force, with at most 0 of the samples saturated); *nontrivial* when a baseline needs >= 0.1 s to recover (posture) or the endpoint deviates >= 0.01 m (force).

## Posture levels

| Magnitude (rad) | Baseline | Terminations | First failure | Success | Recovery max (s) | Peak deviation (m) | Peak torque fraction | Saturation | Peak velocity (rad/s) | Safe | Nontrivial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | computed_torque | completed | — | yes | 0 | 0.00949 | 0.182 | 0 | 0.461 | yes | no |
| 0.02 | pd_v2 | completed | — | yes | 0 | 0.00949 | 0.207 | 0 | 0.91 | yes | no |
| 0.05 | computed_torque | completed | — | yes | 0.13 | 0.0238 | 0.456 | 0 | 1.15 | yes | yes |
| 0.05 | pd_v2 | completed | — | yes | 0.05 | 0.0238 | 0.518 | 0 | 2.28 | yes | yes |
| 0.06 | computed_torque | completed | — | yes | 0.14 | 0.0286 | 0.547 | 0 | 1.38 | yes | yes |
| 0.06 | pd_v2 | completed | — | yes | 0.12 | 0.0286 | 0.622 | 0 | 2.74 | yes | yes |
| 0.07 | computed_torque | completed | — | yes | 0.23 | 0.0334 | 0.638 | 0 | 1.61 | yes | yes |
| 0.07 | pd_v2 | completed | — | yes | 0.13 | 0.0334 | 0.726 | 0 | 3.2 | yes | yes |
| 0.08 | computed_torque | completed | — | yes | 0.24 | 0.0382 | 0.73 | 0 | 1.84 | yes | yes |
| 0.08 | pd_v2 | completed | — | yes | 0.14 | 0.0382 | 0.829 | 0 | 3.67 | yes | yes |
| 0.09 | computed_torque | completed | — | yes | 0.25 | 0.043 | 0.821 | 0 | 2.07 | yes | yes |
| 0.09 | pd_v2 | completed | — | yes | 0.14 | 0.043 | 0.933 | 0 | 4.13 | yes | yes |
| 0.1 | computed_torque | completed | — | yes | 0.25 | 0.0479 | 0.912 | 0 | 2.3 | yes | yes |
| 0.1 | pd_v2 | completed | — | yes | 0.14 | 0.0479 | 1 | 0.002 | 4.59 | yes | yes |
| 0.15 | computed_torque | completed | — | yes | 0.36 | 0.0721 | 1 | 0.00399 | 3.35 | no | yes |
| 0.15 | pd_v2 | completed, limit_violation | joint_velocity joint 1: -6.125 > 6 at t = 0.02 s | no | — | 0.0721 | 1 | 0.5 | 5.86 | no | yes |
| 0.2 | computed_torque | completed | — | yes | 0.38 | 0.0965 | 1 | 0.016 | 4.34 | no | yes |
| 0.2 | pd_v2 | completed, limit_violation | joint_velocity joint 1: 6.71 > 6 at t = 0.01 s | no | — | 0.0965 | 1 | 1 | 5.57 | no | yes |
| 0.3 | computed_torque | completed, limit_violation | joint_velocity joint 1: -6.162 > 6 at t = 0.03 s | no | — | 0.146 | 1 | 1 | 5.5 | no | yes |
| 0.3 | pd_v2 | completed, limit_violation | joint_velocity joint 1: 7.301 > 6 at t = 0.01 s | no | — | 0.146 | 1 | 1 | 5.66 | no | yes |
| 0.5 | computed_torque | completed, limit_violation | joint_velocity joint 1: -7.406 > 6 at t = 0.02 s | no | — | 0.245 | 1 | 1 | 5.27 | no | yes |
| 0.5 | pd_v2 | limit_violation | joint_velocity joint 1: 8.706 > 6 at t = 0.01 s | no | — | 0.245 | 1 | 1 | 5.97 | no | yes |
| 0.8 | computed_torque | completed, limit_violation | joint_velocity joint 1: -7.406 > 6 at t = 0.02 s | no | — | 0.393 | 1 | 1 | 5.8 | no | yes |
| 0.8 | pd_v2 | limit_violation | joint_velocity joint 1: 11.15 > 6 at t = 0.01 s | no | — | 0.393 | 1 | 1 | 5.97 | no | yes |

## Force levels

| Magnitude (N) | Baseline | Terminations | First failure | Success | Recovery max (s) | Peak deviation (m) | Peak torque fraction | Saturation | Peak velocity (rad/s) | Safe | Nontrivial |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | computed_torque | completed | — | yes | 0 | 0.00778 | 0.0296 | 0 | 0.408 | yes | no |
| 0.5 | pd_v2 | completed | — | yes | 0 | 0.00266 | 0.0379 | 0 | 0.419 | yes | no |
| 1 | computed_torque | completed | — | yes | 0.28 | 0.0156 | 0.0565 | 0 | 0.489 | yes | yes |
| 1 | pd_v2 | completed | — | yes | 0 | 0.00513 | 0.074 | 0 | 0.464 | yes | yes |
| 2 | computed_torque | completed | — | yes | 0.36 | 0.0315 | 0.111 | 0 | 0.962 | yes | yes |
| 2 | pd_v2 | completed | — | yes | 0.1 | 0.0101 | 0.146 | 0 | 0.69 | yes | yes |
| 4 | computed_torque | completed | — | yes | 0.42 | 0.0636 | 0.22 | 0 | 1.92 | yes | yes |
| 4 | pd_v2 | completed | — | yes | 0.22 | 0.0199 | 0.291 | 0 | 1.36 | yes | yes |
| 8 | computed_torque | completed | — | yes | 0.54 | 0.127 | 0.434 | 0 | 3.9 | yes | yes |
| 8 | pd_v2 | completed | — | yes | 0.24 | 0.0395 | 0.587 | 0 | 2.71 | yes | yes |
| 10 | computed_torque | completed | — | yes | 0.56 | 0.158 | 0.538 | 0 | 4.93 | yes | yes |
| 10 | pd_v2 | completed | — | yes | 0.31 | 0.0492 | 0.735 | 0 | 3.38 | yes | yes |
| 12 | computed_torque | completed | — | yes | 0.57 | 0.185 | 0.642 | 0 | 5.99 | yes | yes |
| 12 | pd_v2 | completed | — | yes | 0.32 | 0.0588 | 0.884 | 0 | 4.05 | yes | yes |
| 14 | computed_torque | completed, limit_violation | joint_velocity joint 1: -6.039 > 6 at t = 2.03 s | no | — | 0.0554 | 0.751 | 0 | 5.74 | no | yes |
| 14 | pd_v2 | completed | — | yes | 0.32 | 0.069 | 1 | 0.00599 | 4.73 | no | yes |
| 16 | computed_torque | completed, limit_violation | joint_velocity joint 1: -6.92 > 6 at t = 2.03 s | no | — | 0.0645 | 0.86 | 0 | 5.61 | no | yes |
| 16 | pd_v2 | completed | — | yes | 0.32 | 0.091 | 1 | 0.018 | 5.4 | no | yes |
| 32 | computed_torque | limit_violation | joint_velocity joint 1: -6.894 > 6 at t = 2.01 s | no | — | 0.172 | 1 | 0.0929 | 5.88 | no | yes |
| 32 | pd_v2 | completed, limit_violation | joint_velocity joint 1: -6.895 > 6 at t = 2.01 s | no | — | 0.205 | 1 | 0.109 | 5.29 | no | yes |
| 64 | computed_torque | limit_violation | joint_velocity joint 1: -13.77 > 6 at t = 2.01 s | no | — | 0.0181 | 0.631 | 0 | 5.13 | no | yes |
| 64 | pd_v2 | limit_violation | joint_velocity joint 1: -13.77 > 6 at t = 2.01 s | no | — | 0.108 | 1 | 0.0288 | 5.84 | no | yes |

## Selection

- small posture perturbation: 0.05 rad (smallest safe nontrivial level)
- large held-out posture perturbation: 0.1 rad (largest safe level)
- endpoint force pulse: 12 N for 0.2 s from t = 2 s, directions [0.0, 90.0, 180.0, 270.0] deg (largest safe nontrivial level)

These values are locked in `configs/evaluations/task_1a_confirmatory.toml` and may not be used for tuning.
