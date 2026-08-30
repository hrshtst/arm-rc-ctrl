# Perturbation pilot `perturbation-pilot-1a`

Dataset `processed-20260830-feaf73e6663c`, scenario `configs/tasks/task_1a.toml`, commit `0b4fa29421f3`, baselines `computed_torque` (kp [808.611917423121, 277.88260825351], kd [9.613035006148348, 35.99857512413955]), `pd` (kp [95.28850531503508, 71.35814704646654], kd [7.85669277955735, 0.1557202290436201]).

Rules: a level is *safe* when every baseline completes, meets every dwell criterion, and recovers (endpoint within tolerance of the reference) within 1 s (posture) / 1 s after the pulse (force, with at most 0 of the samples saturated); *nontrivial* when a baseline needs >= 0.1 s to recover (posture) or the endpoint deviates >= 0.01 m (force).

## Posture levels

| Magnitude (rad) | Baseline | Terminations | Success | Recovery max (s) | Peak deviation (m) | Peak torque fraction | Saturation | Peak velocity (rad/s) | Safe | Nontrivial |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | computed_torque | completed | yes | 0 | 0.00949 | 0.182 | 0 | 0.461 | yes | no |
| 0.02 | pd | completed | yes | 0 | 0.00949 | 0.285 | 0 | 1.66 | yes | no |
| 0.05 | computed_torque | completed | yes | 0.13 | 0.0238 | 0.456 | 0 | 1.15 | yes | yes |
| 0.05 | pd | completed | yes | 0.06 | 0.0238 | 0.714 | 0 | 4.25 | yes | yes |
| 0.06 | computed_torque | completed | yes | 0.14 | 0.0286 | 0.547 | 0 | 1.38 | yes | yes |
| 0.06 | pd | completed | yes | 0.08 | 0.0286 | 0.86 | 0 | 5.14 | yes | yes |
| 0.07 | computed_torque | completed | yes | 0.23 | 0.0334 | 0.638 | 0 | 1.61 | no | yes |
| 0.07 | pd | completed, limit_violation | no | — | 0.0334 | 0.999 | 0 | 5.47 | no | yes |
| 0.08 | computed_torque | completed | yes | 0.24 | 0.0382 | 0.73 | 0 | 1.84 | no | yes |
| 0.08 | pd | completed, limit_violation | no | — | 0.0382 | 1 | 1 | 5.51 | no | yes |
| 0.09 | computed_torque | completed | yes | 0.25 | 0.043 | 0.821 | 0 | 2.07 | no | yes |
| 0.09 | pd | completed, limit_violation | no | — | 0.043 | 1 | 1 | 5.8 | no | yes |
| 0.1 | computed_torque | completed | yes | 0.25 | 0.0479 | 0.912 | 0 | 2.3 | no | yes |
| 0.1 | pd | completed, limit_violation | no | — | 0.0479 | 1 | 1 | 5.36 | no | yes |
| 0.15 | computed_torque | completed | yes | 0.36 | 0.0721 | 1 | 0.00399 | 3.35 | no | yes |
| 0.15 | pd | completed, limit_violation | no | — | 0.0721 | 1 | 1 | 5.19 | no | yes |
| 0.2 | computed_torque | completed | yes | 0.38 | 0.0965 | 1 | 0.016 | 4.34 | no | yes |
| 0.2 | pd | completed, limit_violation | no | — | 0.0965 | 1 | 1 | 5.04 | no | yes |
| 0.3 | computed_torque | completed, limit_violation | no | — | 0.146 | 1 | 1 | 5.5 | no | yes |
| 0.3 | pd | completed, limit_violation | no | — | 0.146 | 1 | 1 | 4.77 | no | yes |
| 0.5 | computed_torque | completed, limit_violation | no | — | 0.245 | 1 | 1 | 5.27 | no | yes |
| 0.5 | pd | completed, limit_violation | no | — | 0.245 | 1 | 1 | 5.87 | no | yes |
| 0.8 | computed_torque | completed, limit_violation | no | — | 0.393 | 1 | 1 | 5.8 | no | yes |
| 0.8 | pd | limit_violation | no | — | 0.393 | 1 | 1 | 5.23 | no | yes |

## Force levels

| Magnitude (N) | Baseline | Terminations | Success | Recovery max (s) | Peak deviation (m) | Peak torque fraction | Saturation | Peak velocity (rad/s) | Safe | Nontrivial |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | computed_torque | completed | yes | 0 | 0.00778 | 0.0296 | 0 | 0.408 | yes | no |
| 0.5 | pd | completed | yes | 0 | 0.0013 | 0.0392 | 0 | 0.401 | yes | no |
| 1 | computed_torque | completed | yes | 0.28 | 0.0156 | 0.0565 | 0 | 0.489 | yes | yes |
| 1 | pd | completed | yes | 0 | 0.00248 | 0.0767 | 0 | 0.431 | yes | yes |
| 2 | computed_torque | completed | yes | 0.36 | 0.0315 | 0.111 | 0 | 0.962 | yes | yes |
| 2 | pd | completed | yes | 0 | 0.00485 | 0.152 | 0 | 0.492 | yes | yes |
| 4 | computed_torque | completed | yes | 0.42 | 0.0636 | 0.22 | 0 | 1.92 | yes | yes |
| 4 | pd | completed | yes | 0 | 0.00958 | 0.301 | 0 | 0.879 | yes | yes |
| 8 | computed_torque | completed | yes | 0.54 | 0.127 | 0.434 | 0 | 3.9 | yes | yes |
| 8 | pd | completed | yes | 0.23 | 0.019 | 0.602 | 0 | 1.74 | yes | yes |
| 10 | computed_torque | completed | yes | 0.56 | 0.158 | 0.538 | 0 | 4.93 | yes | yes |
| 10 | pd | completed | yes | 0.24 | 0.0238 | 0.754 | 0 | 2.17 | yes | yes |
| 12 | computed_torque | completed | yes | 0.57 | 0.185 | 0.642 | 0 | 5.99 | yes | yes |
| 12 | pd | completed | yes | 0.24 | 0.0287 | 0.906 | 0 | 2.6 | yes | yes |
| 14 | computed_torque | completed, limit_violation | no | — | 0.0554 | 0.751 | 0 | 5.74 | no | yes |
| 14 | pd | completed | yes | 0.25 | 0.0335 | 1 | 0.00399 | 3.05 | no | yes |
| 16 | computed_torque | completed, limit_violation | no | — | 0.0645 | 0.86 | 0 | 5.61 | no | yes |
| 16 | pd | completed | yes | 0.29 | 0.0379 | 1 | 0.012 | 3.46 | no | yes |
| 32 | computed_torque | limit_violation | no | — | 0.172 | 1 | 0.0929 | 5.88 | no | yes |
| 32 | pd | completed, limit_violation | no | — | 0.119 | 1 | 0.0399 | 2.68 | no | yes |
| 64 | computed_torque | limit_violation | no | — | 0.0181 | 0.631 | 0 | 5.13 | no | yes |
| 64 | pd | limit_violation | no | — | 0.257 | 1 | 0.0651 | 5.86 | no | yes |

## Selection

- small posture perturbation: 0.05 rad (smallest safe nontrivial level)
- large held-out posture perturbation: 0.06 rad (largest safe level)
- endpoint force pulse: 12 N for 0.2 s from t = 2 s, directions [0.0, 90.0, 180.0, 270.0] deg (largest safe nontrivial level)

These values are locked in `configs/evaluations/task_1a_confirmatory.toml` and may not be used for tuning.
