# Task 1-a manual numerical copy controls (v1)

Experiment `task_1a_manual_v1`: the duplication controls of the frozen study `docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json` (sha256 `c8f1bdef0ed9`), rclib `0.1.0` at `61a29f0ce6fa`, project commit `70e320280a2f`.

## Outcome

- Overall: **all checks passed**.
- Fits: 120 (0 served from the cache) over 6 configuration(s), arms S, R10.
- Equivalence comparisons (R10_i against S_i, absolute output): 60 of 60 within atol 1e-08 rad and rtol 1e-08.
- Weighted normal-equation residuals: 120 of 120 at or below 1e-10.
- Per-episode rows and weights rebuilt from the study's accounting: 120 of 120 fits.
- Literal repetition (inputs, targets, weights, masks, warm-up, harvested features): 60 of 60 duplication controls identical.
- Probe isolation: 120 of 120 fits trained exactly their arm's datasets and 120 of 120 copied the transform from outside the takes.
- Fresh-process refits: 120 of 120 reproduced weights, states, and report bitwise.

## Execution environment

- Identity `a7f034c7aef4` (canonical): policy `p-cores`, 16 logical CPUs, BLAS `Haswell`, OpenMP max threads 1.
- Every fit and every worker refit bound this identity; a worker whose environment differed is recorded as a failure, never merged.

## Probes

Per configuration: the task rows of all ten locked demonstrations harvested from a reset reservoir after the configuration's warm-up, plus a fit's own contractive bank where it has one. Probe-only trajectories never enter a fit or the frozen input transform.

| configuration | bank | episodes | rows | states sha256 | identical across arms |
| --- | --- | ---: | ---: | --- | --- |
| feasible-best | manual | 10 | 14650 | `68b3a35fa6c2` | yes |
| feasible-middle | manual | 10 | 14650 | `998ebff98703` | yes |
| feasible-worst | manual | 10 | 14650 | `34a599d54ac4` | yes |
| failure-actual-dwell | manual | 10 | 14650 | `c118ddaa9866` | yes |
| failure-joint-velocity | manual | 10 | 14650 | `3ced76fa0a6f` | yes |
| failure-generated-dwell | manual | 10 | 14650 | `9c401781d360` | yes |

## Equivalence comparisons

| configuration | demonstration | candidate | reference | rows | max abs | max rel | worst bank | coef max abs | coef max rel | coef fro rel | decision |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| feasible-best | D01 | R10/D01 | S/D01 | 14650 | 1.732e-13 | 2.514e-13 | manual | 4.034e-12 | 6.983e-09 | 2.617e-11 | pass |
| feasible-best | D02 | R10/D02 | S/D02 | 14650 | 2.984e-13 | 2.593e-13 | manual | 7.731e-12 | 1.689e-09 | 3.229e-11 | pass |
| feasible-best | D03 | R10/D03 | S/D03 | 14650 | 3.830e-13 | 3.112e-13 | manual | 5.645e-12 | 1.256e-08 | 3.333e-11 | pass |
| feasible-best | D04 | R10/D04 | S/D04 | 14650 | 3.355e-13 | 4.127e-13 | manual | 4.124e-12 | 3.167e-09 | 2.965e-11 | pass |
| feasible-best | D05 | R10/D05 | S/D05 | 14650 | 1.520e-13 | 7.258e-13 | manual | 2.827e-12 | 9.200e-09 | 2.877e-11 | pass |
| feasible-best | D06 | R10/D06 | S/D06 | 14650 | 4.612e-13 | 3.675e-13 | manual | 3.136e-12 | 7.554e-09 | 2.878e-11 | pass |
| feasible-best | D07 | R10/D07 | S/D07 | 14650 | 2.012e-13 | 4.851e-13 | manual | 2.977e-12 | 3.593e-09 | 2.672e-11 | pass |
| feasible-best | D08 | R10/D08 | S/D08 | 14650 | 1.945e-13 | 3.009e-13 | manual | 3.211e-12 | 1.654e-08 | 2.794e-11 | pass |
| feasible-best | D09 | R10/D09 | S/D09 | 14650 | 2.753e-13 | 5.655e-13 | manual | 3.196e-12 | 4.808e-09 | 2.590e-11 | pass |
| feasible-best | D10 | R10/D10 | S/D10 | 14650 | 1.934e-13 | 1.532e-13 | manual | 3.972e-12 | 2.685e-09 | 2.366e-11 | pass |
| feasible-middle | D01 | R10/D01 | S/D01 | 14650 | 2.815e-11 | 5.737e-11 | manual | 2.436e-10 | 7.857e-06 | 9.713e-10 | pass |
| feasible-middle | D02 | R10/D02 | S/D02 | 14650 | 2.405e-11 | 4.118e-11 | manual | 2.584e-10 | 1.249e-06 | 1.064e-09 | pass |
| feasible-middle | D03 | R10/D03 | S/D03 | 14650 | 1.732e-11 | 2.803e-11 | manual | 2.521e-10 | 4.626e-06 | 1.077e-09 | pass |
| feasible-middle | D04 | R10/D04 | S/D04 | 14650 | 7.771e-11 | 8.753e-11 | manual | 3.039e-10 | 8.133e-06 | 1.139e-09 | pass |
| feasible-middle | D05 | R10/D05 | S/D05 | 14650 | 3.883e-11 | 2.770e-11 | manual | 2.451e-10 | 2.444e-06 | 1.093e-09 | pass |
| feasible-middle | D06 | R10/D06 | S/D06 | 14650 | 4.848e-11 | 6.134e-11 | manual | 2.848e-10 | 1.429e-06 | 1.124e-09 | pass |
| feasible-middle | D07 | R10/D07 | S/D07 | 14650 | 3.630e-11 | 2.872e-11 | manual | 2.745e-10 | 3.181e-06 | 1.122e-09 | pass |
| feasible-middle | D08 | R10/D08 | S/D08 | 14650 | 2.675e-11 | 6.176e-11 | manual | 2.310e-10 | 4.577e-07 | 1.102e-09 | pass |
| feasible-middle | D09 | R10/D09 | S/D09 | 14650 | 2.184e-11 | 5.113e-11 | manual | 3.644e-10 | 7.619e-07 | 1.073e-09 | pass |
| feasible-middle | D10 | R10/D10 | S/D10 | 14650 | 1.032e-11 | 2.052e-11 | manual | 2.567e-10 | 9.067e-07 | 1.027e-09 | pass |
| feasible-worst | D01 | R10/D01 | S/D01 | 14650 | 2.014e-12 | 5.469e-12 | manual | 1.403e-10 | 1.894e-07 | 3.372e-10 | pass |
| feasible-worst | D02 | R10/D02 | S/D02 | 14650 | 3.301e-12 | 2.783e-12 | manual | 2.420e-10 | 5.018e-06 | 3.596e-10 | pass |
| feasible-worst | D03 | R10/D03 | S/D03 | 14650 | 4.792e-12 | 3.623e-12 | manual | 1.553e-10 | 4.164e-07 | 3.590e-10 | pass |
| feasible-worst | D04 | R10/D04 | S/D04 | 14650 | 1.291e-11 | 9.527e-12 | manual | 1.345e-10 | 2.802e-06 | 3.594e-10 | pass |
| feasible-worst | D05 | R10/D05 | S/D05 | 14650 | 2.550e-12 | 3.829e-12 | manual | 1.183e-10 | 1.280e-05 | 4.417e-10 | pass |
| feasible-worst | D06 | R10/D06 | S/D06 | 14650 | 6.332e-12 | 5.311e-12 | manual | 1.299e-10 | 1.024e-06 | 3.795e-10 | pass |
| feasible-worst | D07 | R10/D07 | S/D07 | 14650 | 4.994e-12 | 3.991e-12 | manual | 1.128e-10 | 3.897e-07 | 4.461e-10 | pass |
| feasible-worst | D08 | R10/D08 | S/D08 | 14650 | 4.532e-12 | 5.208e-12 | manual | 1.429e-10 | 4.492e-07 | 3.872e-10 | pass |
| feasible-worst | D09 | R10/D09 | S/D09 | 14650 | 1.233e-11 | 1.024e-11 | manual | 1.774e-10 | 7.639e-07 | 4.622e-10 | pass |
| feasible-worst | D10 | R10/D10 | S/D10 | 14650 | 1.945e-12 | 3.573e-12 | manual | 1.225e-10 | 1.245e-07 | 3.358e-10 | pass |
| failure-actual-dwell | D01 | R10/D01 | S/D01 | 14650 | 1.756e-13 | 1.778e-13 | manual | 8.934e-13 | 3.162e-08 | 2.722e-12 | pass |
| failure-actual-dwell | D02 | R10/D02 | S/D02 | 14650 | 1.191e-12 | 1.320e-12 | manual | 3.267e-12 | 3.799e-10 | 5.458e-12 | pass |
| failure-actual-dwell | D03 | R10/D03 | S/D03 | 14650 | 1.084e-13 | 5.125e-13 | manual | 5.976e-13 | 8.596e-09 | 2.672e-12 | pass |
| failure-actual-dwell | D04 | R10/D04 | S/D04 | 14650 | 1.139e-13 | 1.068e-13 | manual | 6.202e-13 | 5.486e-10 | 2.707e-12 | pass |
| failure-actual-dwell | D05 | R10/D05 | S/D05 | 14650 | 1.077e-13 | 1.231e-13 | manual | 8.129e-13 | 1.500e-09 | 2.886e-12 | pass |
| failure-actual-dwell | D06 | R10/D06 | S/D06 | 14650 | 8.971e-14 | 2.936e-13 | manual | 5.657e-13 | 4.577e-10 | 2.453e-12 | pass |
| failure-actual-dwell | D07 | R10/D07 | S/D07 | 14650 | 2.287e-13 | 2.262e-13 | manual | 6.197e-13 | 1.144e-09 | 2.463e-12 | pass |
| failure-actual-dwell | D08 | R10/D08 | S/D08 | 14650 | 1.934e-13 | 1.917e-13 | manual | 6.830e-13 | 8.909e-10 | 2.723e-12 | pass |
| failure-actual-dwell | D09 | R10/D09 | S/D09 | 14650 | 6.173e-14 | 9.057e-14 | manual | 6.316e-13 | 3.010e-10 | 2.265e-12 | pass |
| failure-actual-dwell | D10 | R10/D10 | S/D10 | 14650 | 2.959e-13 | 3.899e-13 | manual | 8.102e-13 | 1.000e-09 | 2.414e-12 | pass |
| failure-joint-velocity | D01 | R10/D01 | S/D01 | 14650 | 4.325e-12 | 1.041e-11 | manual | 2.422e-10 | 8.669e-07 | 5.138e-10 | pass |
| failure-joint-velocity | D02 | R10/D02 | S/D02 | 14650 | 1.845e-12 | 7.819e-12 | manual | 2.473e-10 | 1.710e-05 | 5.317e-10 | pass |
| failure-joint-velocity | D03 | R10/D03 | S/D03 | 14650 | 4.251e-12 | 1.287e-11 | manual | 2.261e-10 | 9.836e-07 | 5.250e-10 | pass |
| failure-joint-velocity | D04 | R10/D04 | S/D04 | 14650 | 7.060e-12 | 1.795e-11 | manual | 2.055e-10 | 4.812e-07 | 5.580e-10 | pass |
| failure-joint-velocity | D05 | R10/D05 | S/D05 | 14650 | 6.870e-12 | 1.671e-11 | manual | 2.182e-10 | 2.424e-07 | 5.585e-10 | pass |
| failure-joint-velocity | D06 | R10/D06 | S/D06 | 14650 | 3.595e-12 | 1.092e-11 | manual | 1.763e-10 | 1.745e-06 | 5.192e-10 | pass |
| failure-joint-velocity | D07 | R10/D07 | S/D07 | 14650 | 3.767e-12 | 1.082e-11 | manual | 1.750e-10 | 1.019e-07 | 5.473e-10 | pass |
| failure-joint-velocity | D08 | R10/D08 | S/D08 | 14650 | 8.232e-12 | 1.771e-11 | manual | 1.847e-10 | 3.287e-07 | 6.181e-10 | pass |
| failure-joint-velocity | D09 | R10/D09 | S/D09 | 14650 | 9.601e-12 | 7.399e-12 | manual | 2.041e-10 | 6.000e-07 | 5.657e-10 | pass |
| failure-joint-velocity | D10 | R10/D10 | S/D10 | 14650 | 4.517e-12 | 8.848e-12 | manual | 2.033e-10 | 3.612e-07 | 4.690e-10 | pass |
| failure-generated-dwell | D01 | R10/D01 | S/D01 | 14650 | 1.417e-12 | 2.424e-12 | manual | 7.398e-12 | 6.729e-08 | 3.183e-11 | pass |
| failure-generated-dwell | D02 | R10/D02 | S/D02 | 14650 | 1.271e-11 | 1.300e-11 | manual | 2.084e-11 | 3.281e-08 | 3.694e-11 | pass |
| failure-generated-dwell | D03 | R10/D03 | S/D03 | 14650 | 1.321e-12 | 4.313e-12 | manual | 7.292e-12 | 3.556e-08 | 3.300e-11 | pass |
| failure-generated-dwell | D04 | R10/D04 | S/D04 | 14650 | 3.004e-13 | 1.257e-12 | manual | 8.602e-12 | 3.410e-08 | 3.390e-11 | pass |
| failure-generated-dwell | D05 | R10/D05 | S/D05 | 14650 | 1.162e-12 | 1.171e-12 | manual | 1.307e-11 | 4.604e-07 | 3.646e-11 | pass |
| failure-generated-dwell | D06 | R10/D06 | S/D06 | 14650 | 3.615e-13 | 2.089e-12 | manual | 7.487e-12 | 1.017e-08 | 3.247e-11 | pass |
| failure-generated-dwell | D07 | R10/D07 | S/D07 | 14650 | 2.363e-12 | 2.370e-12 | manual | 7.279e-12 | 2.750e-07 | 3.566e-11 | pass |
| failure-generated-dwell | D08 | R10/D08 | S/D08 | 14650 | 1.559e-12 | 1.565e-12 | manual | 7.389e-12 | 1.088e-08 | 3.282e-11 | pass |
| failure-generated-dwell | D09 | R10/D09 | S/D09 | 14650 | 6.208e-13 | 1.085e-12 | manual | 8.536e-12 | 6.617e-09 | 3.146e-11 | pass |
| failure-generated-dwell | D10 | R10/D10 | S/D10 | 14650 | 3.191e-12 | 3.293e-12 | manual | 1.165e-11 | 6.713e-09 | 3.481e-11 | pass |

## Fits

Every fit with its ridge scale, loss rows, teacher-forced error, the normalized residual and conditioning of the weighted system A = Xw^T Xw + alpha I (explicit bias column last), the accessor check over the matched probes, and its construction invariants.

| configuration | arm | alpha | loss rows | rmse | residual | cond2 | accessor | weighting | sources | transform | fit s | cache |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| feasible-best | S/D01 | 0.119172 | 1664 | 4.258e-03 | 1.813e-16 | 1.297e+05 | 1.332e-15 | pass | pass | pass | 0.06 | fit |
| feasible-best | S/D02 | 0.119172 | 1585 | 3.758e-03 | 1.992e-16 | 1.305e+05 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | S/D03 | 0.119172 | 1398 | 3.795e-03 | 2.458e-16 | 1.302e+05 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | S/D04 | 0.119172 | 1287 | 3.991e-03 | 2.122e-16 | 1.309e+05 | 1.332e-15 | pass | pass | pass | 0.03 | fit |
| feasible-best | S/D05 | 0.119172 | 1424 | 3.948e-03 | 2.461e-16 | 1.323e+05 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | S/D06 | 0.119172 | 1377 | 4.017e-03 | 2.146e-16 | 1.306e+05 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | S/D07 | 0.119172 | 1364 | 3.637e-03 | 2.525e-16 | 1.307e+05 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | S/D08 | 0.119172 | 1387 | 3.265e-03 | 2.031e-16 | 1.326e+05 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | S/D09 | 0.119172 | 1472 | 4.008e-03 | 2.325e-16 | 1.331e+05 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | S/D10 | 0.119172 | 1692 | 3.667e-03 | 1.751e-16 | 1.306e+05 | 1.110e-15 | pass | pass | pass | 0.04 | fit |
| feasible-best | R10/D01 | 1.19172 | 16640 | 4.258e-03 | 1.886e-16 | 1.297e+05 | 1.332e-15 | pass | pass | pass | 0.25 | fit |
| feasible-best | R10/D02 | 1.19172 | 15850 | 3.758e-03 | 1.855e-16 | 1.305e+05 | 1.332e-15 | pass | pass | pass | 0.27 | fit |
| feasible-best | R10/D03 | 1.19172 | 13980 | 3.795e-03 | 1.800e-16 | 1.302e+05 | 1.332e-15 | pass | pass | pass | 0.23 | fit |
| feasible-best | R10/D04 | 1.19172 | 12870 | 3.991e-03 | 1.657e-16 | 1.309e+05 | 1.554e-15 | pass | pass | pass | 0.20 | fit |
| feasible-best | R10/D05 | 1.19172 | 14240 | 3.948e-03 | 2.431e-16 | 1.323e+05 | 1.332e-15 | pass | pass | pass | 0.23 | fit |
| feasible-best | R10/D06 | 1.19172 | 13770 | 4.017e-03 | 1.655e-16 | 1.306e+05 | 1.332e-15 | pass | pass | pass | 0.21 | fit |
| feasible-best | R10/D07 | 1.19172 | 13640 | 3.637e-03 | 1.980e-16 | 1.307e+05 | 1.332e-15 | pass | pass | pass | 0.23 | fit |
| feasible-best | R10/D08 | 1.19172 | 13870 | 3.265e-03 | 1.669e-16 | 1.326e+05 | 1.332e-15 | pass | pass | pass | 0.23 | fit |
| feasible-best | R10/D09 | 1.19172 | 14720 | 4.008e-03 | 2.366e-16 | 1.331e+05 | 1.332e-15 | pass | pass | pass | 0.23 | fit |
| feasible-best | R10/D10 | 1.19172 | 16920 | 3.667e-03 | 1.849e-16 | 1.306e+05 | 1.110e-15 | pass | pass | pass | 0.27 | fit |
| feasible-middle | S/D01 | 0.00309759 | 1664 | 1.734e-03 | 5.176e-17 | 1.869e+07 | 1.332e-15 | pass | pass | pass | 0.20 | fit |
| feasible-middle | S/D02 | 0.00309759 | 1585 | 1.564e-03 | 5.172e-17 | 1.871e+07 | 8.882e-16 | pass | pass | pass | 0.13 | fit |
| feasible-middle | S/D03 | 0.00309759 | 1398 | 1.640e-03 | 6.411e-17 | 1.860e+07 | 1.110e-15 | pass | pass | pass | 0.13 | fit |
| feasible-middle | S/D04 | 0.00309759 | 1287 | 1.623e-03 | 9.005e-17 | 1.861e+07 | 1.110e-15 | pass | pass | pass | 0.12 | fit |
| feasible-middle | S/D05 | 0.00309759 | 1424 | 1.515e-03 | 1.016e-16 | 1.879e+07 | 1.110e-15 | pass | pass | pass | 0.12 | fit |
| feasible-middle | S/D06 | 0.00309759 | 1377 | 1.680e-03 | 7.075e-17 | 1.859e+07 | 8.882e-16 | pass | pass | pass | 0.11 | fit |
| feasible-middle | S/D07 | 0.00309759 | 1364 | 1.472e-03 | 1.002e-16 | 1.861e+07 | 8.882e-16 | pass | pass | pass | 0.12 | fit |
| feasible-middle | S/D08 | 0.00309759 | 1387 | 1.511e-03 | 7.291e-17 | 1.874e+07 | 7.772e-16 | pass | pass | pass | 0.11 | fit |
| feasible-middle | S/D09 | 0.00309759 | 1472 | 1.587e-03 | 6.838e-17 | 1.884e+07 | 1.110e-15 | pass | pass | pass | 0.12 | fit |
| feasible-middle | S/D10 | 0.00309759 | 1692 | 1.711e-03 | 9.193e-17 | 1.867e+07 | 9.992e-16 | pass | pass | pass | 0.14 | fit |
| feasible-middle | R10/D01 | 0.0309759 | 16640 | 1.734e-03 | 5.473e-17 | 1.869e+07 | 1.221e-15 | pass | pass | pass | 1.23 | fit |
| feasible-middle | R10/D02 | 0.0309759 | 15850 | 1.564e-03 | 1.163e-16 | 1.871e+07 | 9.992e-16 | pass | pass | pass | 1.20 | fit |
| feasible-middle | R10/D03 | 0.0309759 | 13980 | 1.640e-03 | 6.016e-17 | 1.860e+07 | 9.992e-16 | pass | pass | pass | 1.04 | fit |
| feasible-middle | R10/D04 | 0.0309759 | 12870 | 1.623e-03 | 7.663e-17 | 1.861e+07 | 1.110e-15 | pass | pass | pass | 0.98 | fit |
| feasible-middle | R10/D05 | 0.0309759 | 14240 | 1.515e-03 | 9.902e-17 | 1.879e+07 | 1.332e-15 | pass | pass | pass | 1.06 | fit |
| feasible-middle | R10/D06 | 0.0309759 | 13770 | 1.680e-03 | 5.453e-17 | 1.859e+07 | 8.882e-16 | pass | pass | pass | 1.02 | fit |
| feasible-middle | R10/D07 | 0.0309759 | 13640 | 1.472e-03 | 1.080e-16 | 1.861e+07 | 9.992e-16 | pass | pass | pass | 1.02 | fit |
| feasible-middle | R10/D08 | 0.0309759 | 13870 | 1.511e-03 | 6.354e-17 | 1.874e+07 | 7.772e-16 | pass | pass | pass | 1.03 | fit |
| feasible-middle | R10/D09 | 0.0309759 | 14720 | 1.587e-03 | 6.952e-17 | 1.884e+07 | 9.992e-16 | pass | pass | pass | 1.09 | fit |
| feasible-middle | R10/D10 | 0.0309759 | 16920 | 1.711e-03 | 1.181e-16 | 1.867e+07 | 9.992e-16 | pass | pass | pass | 1.25 | fit |
| feasible-worst | S/D01 | 0.00493549 | 1664 | 8.190e-03 | 4.588e-17 | 9.297e+06 | 2.220e-15 | pass | pass | pass | 0.11 | fit |
| feasible-worst | S/D02 | 0.00493549 | 1585 | 7.784e-03 | 4.914e-17 | 9.271e+06 | 1.998e-15 | pass | pass | pass | 0.11 | fit |
| feasible-worst | S/D03 | 0.00493549 | 1398 | 7.313e-03 | 4.310e-17 | 9.202e+06 | 2.442e-15 | pass | pass | pass | 0.10 | fit |
| feasible-worst | S/D04 | 0.00493549 | 1287 | 8.584e-03 | 4.745e-17 | 9.152e+06 | 2.220e-15 | pass | pass | pass | 0.09 | fit |
| feasible-worst | S/D05 | 0.00493549 | 1424 | 7.846e-03 | 5.186e-17 | 9.212e+06 | 1.554e-15 | pass | pass | pass | 0.09 | fit |
| feasible-worst | S/D06 | 0.00493549 | 1377 | 7.770e-03 | 5.600e-17 | 9.193e+06 | 1.998e-15 | pass | pass | pass | 0.09 | fit |
| feasible-worst | S/D07 | 0.00493549 | 1364 | 7.668e-03 | 5.262e-17 | 9.188e+06 | 1.776e-15 | pass | pass | pass | 0.09 | fit |
| feasible-worst | S/D08 | 0.00493549 | 1387 | 6.793e-03 | 4.317e-17 | 9.196e+06 | 1.776e-15 | pass | pass | pass | 0.10 | fit |
| feasible-worst | S/D09 | 0.00493549 | 1472 | 6.235e-03 | 5.442e-17 | 9.230e+06 | 2.220e-15 | pass | pass | pass | 0.09 | fit |
| feasible-worst | S/D10 | 0.00493549 | 1692 | 6.668e-03 | 3.987e-17 | 9.305e+06 | 1.998e-15 | pass | pass | pass | 0.11 | fit |
| feasible-worst | R10/D01 | 0.0493549 | 16640 | 8.190e-03 | 3.535e-17 | 9.297e+06 | 1.887e-15 | pass | pass | pass | 0.92 | fit |
| feasible-worst | R10/D02 | 0.0493549 | 15850 | 7.784e-03 | 4.045e-17 | 9.271e+06 | 2.442e-15 | pass | pass | pass | 0.88 | fit |
| feasible-worst | R10/D03 | 0.0493549 | 13980 | 7.313e-03 | 3.435e-17 | 9.202e+06 | 2.665e-15 | pass | pass | pass | 0.77 | fit |
| feasible-worst | R10/D04 | 0.0493549 | 12870 | 8.584e-03 | 3.845e-17 | 9.152e+06 | 2.442e-15 | pass | pass | pass | 0.71 | fit |
| feasible-worst | R10/D05 | 0.0493549 | 14240 | 7.846e-03 | 4.959e-17 | 9.212e+06 | 1.776e-15 | pass | pass | pass | 0.79 | fit |
| feasible-worst | R10/D06 | 0.0493549 | 13770 | 7.770e-03 | 6.708e-17 | 9.193e+06 | 1.776e-15 | pass | pass | pass | 0.76 | fit |
| feasible-worst | R10/D07 | 0.0493549 | 13640 | 7.668e-03 | 4.554e-17 | 9.188e+06 | 1.776e-15 | pass | pass | pass | 0.76 | fit |
| feasible-worst | R10/D08 | 0.0493549 | 13870 | 6.793e-03 | 3.909e-17 | 9.196e+06 | 1.776e-15 | pass | pass | pass | 0.77 | fit |
| feasible-worst | R10/D09 | 0.0493549 | 14720 | 6.235e-03 | 5.688e-17 | 9.230e+06 | 1.998e-15 | pass | pass | pass | 0.83 | fit |
| feasible-worst | R10/D10 | 0.0493549 | 16920 | 6.668e-03 | 4.447e-17 | 9.305e+06 | 1.887e-15 | pass | pass | pass | 0.93 | fit |
| failure-actual-dwell | S/D01 | 0.619159 | 1664 | 1.558e-02 | 1.411e-16 | 1.884e+04 | 8.882e-16 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | S/D02 | 0.619159 | 1585 | 1.455e-02 | 1.470e-16 | 1.884e+04 | 8.882e-16 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | S/D03 | 0.619159 | 1398 | 1.445e-02 | 1.816e-16 | 1.863e+04 | 8.882e-16 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | S/D04 | 0.619159 | 1287 | 1.601e-02 | 2.945e-16 | 1.850e+04 | 8.882e-16 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | S/D05 | 0.619159 | 1424 | 1.427e-02 | 3.512e-16 | 1.877e+04 | 1.110e-15 | pass | pass | pass | 0.03 | fit |
| failure-actual-dwell | S/D06 | 0.619159 | 1377 | 1.457e-02 | 1.465e-16 | 1.863e+04 | 1.332e-15 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | S/D07 | 0.619159 | 1364 | 1.381e-02 | 3.245e-16 | 1.862e+04 | 1.110e-15 | pass | pass | pass | 0.03 | fit |
| failure-actual-dwell | S/D08 | 0.619159 | 1387 | 1.467e-02 | 1.865e-16 | 1.878e+04 | 1.110e-15 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | S/D09 | 0.619159 | 1472 | 1.334e-02 | 1.580e-16 | 1.889e+04 | 1.110e-15 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | S/D10 | 0.619159 | 1692 | 1.501e-02 | 2.929e-16 | 1.900e+04 | 8.882e-16 | pass | pass | pass | 0.04 | fit |
| failure-actual-dwell | R10/D01 | 6.19159 | 16640 | 1.558e-02 | 1.459e-16 | 1.884e+04 | 1.332e-15 | pass | pass | pass | 0.24 | fit |
| failure-actual-dwell | R10/D02 | 6.19159 | 15850 | 1.455e-02 | 3.830e-16 | 1.884e+04 | 8.882e-16 | pass | pass | pass | 0.24 | fit |
| failure-actual-dwell | R10/D03 | 6.19159 | 13980 | 1.445e-02 | 1.893e-16 | 1.863e+04 | 8.882e-16 | pass | pass | pass | 0.20 | fit |
| failure-actual-dwell | R10/D04 | 6.19159 | 12870 | 1.601e-02 | 2.500e-16 | 1.850e+04 | 1.110e-15 | pass | pass | pass | 0.20 | fit |
| failure-actual-dwell | R10/D05 | 6.19159 | 14240 | 1.427e-02 | 3.588e-16 | 1.877e+04 | 1.110e-15 | pass | pass | pass | 0.21 | fit |
| failure-actual-dwell | R10/D06 | 6.19159 | 13770 | 1.457e-02 | 1.168e-16 | 1.863e+04 | 1.110e-15 | pass | pass | pass | 0.21 | fit |
| failure-actual-dwell | R10/D07 | 6.19159 | 13640 | 1.381e-02 | 3.895e-16 | 1.862e+04 | 8.882e-16 | pass | pass | pass | 0.20 | fit |
| failure-actual-dwell | R10/D08 | 6.19159 | 13870 | 1.467e-02 | 1.740e-16 | 1.878e+04 | 8.882e-16 | pass | pass | pass | 0.21 | fit |
| failure-actual-dwell | R10/D09 | 6.19159 | 14720 | 1.334e-02 | 2.054e-16 | 1.889e+04 | 1.110e-15 | pass | pass | pass | 0.22 | fit |
| failure-actual-dwell | R10/D10 | 6.19159 | 16920 | 1.501e-02 | 3.718e-16 | 1.900e+04 | 8.882e-16 | pass | pass | pass | 0.25 | fit |
| failure-joint-velocity | S/D01 | 0.00284948 | 1664 | 5.731e-03 | 3.719e-17 | 1.371e+07 | 2.665e-15 | pass | pass | pass | 0.09 | fit |
| failure-joint-velocity | S/D02 | 0.00284948 | 1585 | 5.319e-03 | 3.766e-17 | 1.370e+07 | 2.442e-15 | pass | pass | pass | 0.08 | fit |
| failure-joint-velocity | S/D03 | 0.00284948 | 1398 | 4.586e-03 | 4.450e-17 | 1.368e+07 | 2.665e-15 | pass | pass | pass | 0.08 | fit |
| failure-joint-velocity | S/D04 | 0.00284948 | 1287 | 5.370e-03 | 4.685e-17 | 1.366e+07 | 2.887e-15 | pass | pass | pass | 0.07 | fit |
| failure-joint-velocity | S/D05 | 0.00284948 | 1424 | 5.633e-03 | 4.689e-17 | 1.368e+07 | 2.442e-15 | pass | pass | pass | 0.08 | fit |
| failure-joint-velocity | S/D06 | 0.00284948 | 1377 | 5.735e-03 | 4.397e-17 | 1.368e+07 | 2.442e-15 | pass | pass | pass | 0.07 | fit |
| failure-joint-velocity | S/D07 | 0.00284948 | 1364 | 5.159e-03 | 4.334e-17 | 1.368e+07 | 2.220e-15 | pass | pass | pass | 0.07 | fit |
| failure-joint-velocity | S/D08 | 0.00284948 | 1387 | 4.185e-03 | 5.575e-17 | 1.368e+07 | 2.220e-15 | pass | pass | pass | 0.07 | fit |
| failure-joint-velocity | S/D09 | 0.00284948 | 1472 | 4.961e-03 | 5.552e-17 | 1.369e+07 | 2.220e-15 | pass | pass | pass | 0.08 | fit |
| failure-joint-velocity | S/D10 | 0.00284948 | 1692 | 4.772e-03 | 3.859e-17 | 1.371e+07 | 2.220e-15 | pass | pass | pass | 0.09 | fit |
| failure-joint-velocity | R10/D01 | 0.0284948 | 16640 | 5.731e-03 | 3.812e-17 | 1.371e+07 | 2.665e-15 | pass | pass | pass | 0.70 | fit |
| failure-joint-velocity | R10/D02 | 0.0284948 | 15850 | 5.319e-03 | 4.081e-17 | 1.370e+07 | 2.442e-15 | pass | pass | pass | 0.67 | fit |
| failure-joint-velocity | R10/D03 | 0.0284948 | 13980 | 4.586e-03 | 3.982e-17 | 1.368e+07 | 2.665e-15 | pass | pass | pass | 0.61 | fit |
| failure-joint-velocity | R10/D04 | 0.0284948 | 12870 | 5.370e-03 | 5.022e-17 | 1.366e+07 | 3.553e-15 | pass | pass | pass | 0.54 | fit |
| failure-joint-velocity | R10/D05 | 0.0284948 | 14240 | 5.633e-03 | 4.201e-17 | 1.368e+07 | 2.442e-15 | pass | pass | pass | 0.64 | fit |
| failure-joint-velocity | R10/D06 | 0.0284948 | 13770 | 5.735e-03 | 3.667e-17 | 1.368e+07 | 2.442e-15 | pass | pass | pass | 0.59 | fit |
| failure-joint-velocity | R10/D07 | 0.0284948 | 13640 | 5.159e-03 | 3.985e-17 | 1.368e+07 | 2.220e-15 | pass | pass | pass | 0.58 | fit |
| failure-joint-velocity | R10/D08 | 0.0284948 | 13870 | 4.185e-03 | 4.646e-17 | 1.368e+07 | 2.442e-15 | pass | pass | pass | 0.61 | fit |
| failure-joint-velocity | R10/D09 | 0.0284948 | 14720 | 4.961e-03 | 5.971e-17 | 1.369e+07 | 1.776e-15 | pass | pass | pass | 0.62 | fit |
| failure-joint-velocity | R10/D10 | 0.0284948 | 16920 | 4.772e-03 | 3.451e-17 | 1.371e+07 | 2.220e-15 | pass | pass | pass | 0.72 | fit |
| failure-generated-dwell | S/D01 | 0.0836599 | 1664 | 7.632e-03 | 5.485e-17 | 5.515e+05 | 8.882e-16 | pass | pass | pass | 0.11 | fit |
| failure-generated-dwell | S/D02 | 0.0836599 | 1585 | 7.274e-03 | 5.838e-17 | 5.512e+05 | 8.882e-16 | pass | pass | pass | 0.09 | fit |
| failure-generated-dwell | S/D03 | 0.0836599 | 1398 | 7.458e-03 | 7.066e-17 | 5.502e+05 | 8.882e-16 | pass | pass | pass | 0.09 | fit |
| failure-generated-dwell | S/D04 | 0.0836599 | 1287 | 7.308e-03 | 9.212e-17 | 5.497e+05 | 8.882e-16 | pass | pass | pass | 0.08 | fit |
| failure-generated-dwell | S/D05 | 0.0836599 | 1424 | 7.540e-03 | 1.079e-16 | 5.506e+05 | 8.882e-16 | pass | pass | pass | 0.09 | fit |
| failure-generated-dwell | S/D06 | 0.0836599 | 1377 | 7.689e-03 | 7.453e-17 | 5.501e+05 | 8.882e-16 | pass | pass | pass | 0.08 | fit |
| failure-generated-dwell | S/D07 | 0.0836599 | 1364 | 7.104e-03 | 9.685e-17 | 5.500e+05 | 8.882e-16 | pass | pass | pass | 0.09 | fit |
| failure-generated-dwell | S/D08 | 0.0836599 | 1387 | 7.096e-03 | 6.919e-17 | 5.504e+05 | 8.882e-16 | pass | pass | pass | 0.08 | fit |
| failure-generated-dwell | S/D09 | 0.0836599 | 1472 | 7.607e-03 | 7.062e-17 | 5.510e+05 | 8.882e-16 | pass | pass | pass | 0.09 | fit |
| failure-generated-dwell | S/D10 | 0.0836599 | 1692 | 7.887e-03 | 8.807e-17 | 5.513e+05 | 8.882e-16 | pass | pass | pass | 0.11 | fit |
| failure-generated-dwell | R10/D01 | 0.836599 | 16640 | 7.632e-03 | 7.260e-17 | 5.515e+05 | 1.110e-15 | pass | pass | pass | 0.84 | fit |
| failure-generated-dwell | R10/D02 | 0.836599 | 15850 | 7.274e-03 | 1.119e-16 | 5.512e+05 | 8.882e-16 | pass | pass | pass | 0.80 | fit |
| failure-generated-dwell | R10/D03 | 0.836599 | 13980 | 7.458e-03 | 6.267e-17 | 5.502e+05 | 1.110e-15 | pass | pass | pass | 0.70 | fit |
| failure-generated-dwell | R10/D04 | 0.836599 | 12870 | 7.308e-03 | 8.124e-17 | 5.497e+05 | 1.110e-15 | pass | pass | pass | 0.66 | fit |
| failure-generated-dwell | R10/D05 | 0.836599 | 14240 | 7.540e-03 | 1.066e-16 | 5.506e+05 | 8.882e-16 | pass | pass | pass | 0.78 | fit |
| failure-generated-dwell | R10/D06 | 0.836599 | 13770 | 7.689e-03 | 5.592e-17 | 5.501e+05 | 8.882e-16 | pass | pass | pass | 0.67 | fit |
| failure-generated-dwell | R10/D07 | 0.836599 | 13640 | 7.104e-03 | 1.121e-16 | 5.500e+05 | 8.882e-16 | pass | pass | pass | 0.67 | fit |
| failure-generated-dwell | R10/D08 | 0.836599 | 13870 | 7.096e-03 | 6.732e-17 | 5.504e+05 | 8.882e-16 | pass | pass | pass | 0.70 | fit |
| failure-generated-dwell | R10/D09 | 0.836599 | 14720 | 7.607e-03 | 6.770e-17 | 5.510e+05 | 8.882e-16 | pass | pass | pass | 0.74 | fit |
| failure-generated-dwell | R10/D10 | 0.836599 | 16920 | 7.887e-03 | 1.079e-16 | 5.513e+05 | 8.882e-16 | pass | pass | pass | 0.85 | fit |

## Literal repetition

| configuration | arm | episodes | inputs | targets | weights | masks | warm-up | features | decision |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| feasible-best | R10/D01 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D02 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D03 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D04 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D05 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D06 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D07 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D08 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D09 | 10 | same | same | same | same | same | same | pass |
| feasible-best | R10/D10 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D01 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D02 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D03 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D04 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D05 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D06 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D07 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D08 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D09 | 10 | same | same | same | same | same | same | pass |
| feasible-middle | R10/D10 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D01 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D02 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D03 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D04 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D05 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D06 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D07 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D08 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D09 | 10 | same | same | same | same | same | same | pass |
| feasible-worst | R10/D10 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D01 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D02 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D03 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D04 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D05 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D06 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D07 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D08 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D09 | 10 | same | same | same | same | same | same | pass |
| failure-actual-dwell | R10/D10 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D01 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D02 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D03 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D04 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D05 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D06 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D07 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D08 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D09 | 10 | same | same | same | same | same | same | pass |
| failure-joint-velocity | R10/D10 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D01 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D02 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D03 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D04 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D05 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D06 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D07 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D08 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D09 | 10 | same | same | same | same | same | same | pass |
| failure-generated-dwell | R10/D10 | 10 | same | same | same | same | same | same | pass |

## Fresh-process refits

| identity | environment | weights | max abs weight diff | states | fit report | rmse diff | s | decision |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| `1c1e0dd53182` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `949cc8429e20` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `01b2c0a4ee0a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `f3b8a558c458` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `3f42be175c7b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `83b3ec963aa7` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `549b7ad55cc0` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `e82c6a73c511` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `afb162a8bdba` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `ff3a02f15231` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `0a161ec59ba2` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.38 | pass |
| `ede327851818` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.34 | pass |
| `f23241417256` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.33 | pass |
| `4ae7747368ce` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.29 | pass |
| `17bc1d44a727` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.34 | pass |
| `2ef29c666caa` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.31 | pass |
| `59abf2047ee0` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.31 | pass |
| `7fc0e5180a8a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.32 | pass |
| `63621b590c06` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.32 | pass |
| `db1b0525f32c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.38 | pass |
| `528f505a21cf` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.20 | pass |
| `13876e2c676e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.20 | pass |
| `7cc957fbdaa6` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.19 | pass |
| `8cbbb1a02dce` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.17 | pass |
| `80a6bfa48189` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.22 | pass |
| `8e70561b6882` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.17 | pass |
| `85e8caffc1c6` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.17 | pass |
| `822e3b532f42` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.17 | pass |
| `e88eb0cf4b95` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.18 | pass |
| `7b109f363710` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.21 | pass |
| `7e459faabea5` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.73 | pass |
| `4e024387e4fe` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.62 | pass |
| `3b0cc72d33a5` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.48 | pass |
| `eedc0ee3d73f` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.33 | pass |
| `23106ce99d15` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.48 | pass |
| `3ce62751d412` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.44 | pass |
| `a3562cf12e98` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.42 | pass |
| `a17342a50b42` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.43 | pass |
| `103d576971c3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.62 | pass |
| `dd1f0dcabe72` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.77 | pass |
| `c1e531c25176` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.16 | pass |
| `f64ff27a15bd` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.15 | pass |
| `a7ca1d73f7e9` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `ac3c8db2831a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `503cc39987a8` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.14 | pass |
| `5eb377a4b0c3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `6cfe8efdd871` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.14 | pass |
| `834f43b1003d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.14 | pass |
| `046b6ebb574e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.14 | pass |
| `970f52f32646` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.16 | pass |
| `9ef4081bb07b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.30 | pass |
| `c8c4172e879f` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.23 | pass |
| `426e191f867e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.09 | pass |
| `c529d718aeae` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.02 | pass |
| `9673333b05fb` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.12 | pass |
| `f1e4fc4f2158` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.08 | pass |
| `7ca7d289c9e1` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.05 | pass |
| `8dd3dda72429` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.18 | pass |
| `80567ff082e1` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.15 | pass |
| `c07cac922a03` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.30 | pass |
| `1f2264595a7d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `201488773f83` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `f22e56d0ed46` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `67ce9938c15f` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `16e0e9794ac3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `8e3fb64014a3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `a595bae679de` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `af6b2070cb6a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `1663b751713e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `d75dae95acb2` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `4ba21bc56552` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.33 | pass |
| `45e533755054` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.33 | pass |
| `094690bdf04c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.30 | pass |
| `856efdb20f7b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.27 | pass |
| `1201118ba79b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.30 | pass |
| `e75e1a524373` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.30 | pass |
| `ee6d9e192c5c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.28 | pass |
| `f22095229a31` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.29 | pass |
| `0bcef98bd1a8` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.30 | pass |
| `089bcdc5280d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.35 | pass |
| `0023b930ffd8` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `79d948c82840` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `025c5ab13ddc` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.11 | pass |
| `c861ab0749ae` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.11 | pass |
| `92b90e3bb8a7` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.11 | pass |
| `427a1d9ae361` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.11 | pass |
| `5193c5f278fe` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.12 | pass |
| `4fcecd40fa2c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.12 | pass |
| `dd1c6adfbb6f` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.12 | pass |
| `e6367cdc423b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `f95271333a96` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.00 | pass |
| `fdaec10e7cbf` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.96 | pass |
| `9f27c3e37bac` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.86 | pass |
| `f5d8f7dded46` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.80 | pass |
| `12f64b88b512` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.88 | pass |
| `27bd83684a65` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.85 | pass |
| `e7ee6286757f` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.85 | pass |
| `16515280e011` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.87 | pass |
| `8e7001c05023` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.90 | pass |
| `b28252ab8b32` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.04 | pass |
| `f1317d608129` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.15 | pass |
| `46269fc1d63d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.14 | pass |
| `8a455fc382ac` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `5c8e5827ef6e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.12 | pass |
| `1733492e6a7a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.12 | pass |
| `9073f70b1d8b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.12 | pass |
| `f5083c186465` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `82a36f337e6b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.12 | pass |
| `b25cb0a31907` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.13 | pass |
| `86e56ac9b727` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.15 | pass |
| `5b6ae696a3ba` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.18 | pass |
| `a9643f7bf19d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.12 | pass |
| `49f42d0e50a9` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.99 | pass |
| `70af16d87c9d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.91 | pass |
| `4f5070dd5e17` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.02 | pass |
| `017406261e08` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.97 | pass |
| `257a8a18d2d7` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.96 | pass |
| `406d8be57d71` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.97 | pass |
| `3b42ed231572` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.04 | pass |
| `1c0e614ace42` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.19 | pass |

## Limitations

- The prediction tolerance bounds teacher-forced readout differences on fixed probes; it is not a closed-loop stability guarantee (manual plan section 5).
- The equivalence is a numerical identity of the weighted ridge problem; agreement here says nothing about whether ten demonstrations help, which the behavioral evaluation decides.
- Fresh-process bitwise reproducibility holds in the recorded execution environment only; another core type, thread setting, library build, or machine is a different environment (C10).
- The residual is evaluated with numpy on the harvested rows; the solve itself is rclib's LDLT on its own accumulation of the same scaled rows.
- The predecessor's accepted numerical exception is not carried over: a failure here is retained and must be diagnosed before any behavioral interpretation.
