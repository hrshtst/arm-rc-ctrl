# Task 1-a repetition numerical validation (v1)

Experiment `task_1a_repetition_v1`: ridge equivalences and fresh-process reproducibility of the pilot's fits (repetition plan section 6, D6) on the frozen panel `docs/experiments/task_1a_repeated_demonstration/panel_manifest_v1.json` (sha256 `68bc850ef269`), dataset `processed-20260903-ce343c8ce6a5` (payload sha256 `ce343c8ce6a5`), rclib `0.1.0` at `61a29f0ce6fa`, project commit `7843c239876b`.

## Outcome

- Overall: **FAILURES RETAINED**.
- Fits: 120 (0 served from the cache).
- Equivalence comparisons: 71 of 72 within atol 1e-08 rad and rtol 1e-08.
- Normal-equation residuals: 120 of 120 at or below 1e-10.
- Literal copies harvested bitwise identically: 120 of 120 fits.
- Matched absolute/residual state identity: 60 of 60 pairs; probe matrices identical across formulations for 6 of 6 entries.
- Fresh-process refits: 120 of 120 reproduced weights, states, and report bitwise.

## Execution environment

- Identity `a7f034c7aef4` (canonical): policy `p-cores`, CPUs 0-15 (16 logical CPUs), threads MKL_NUM_THREADS=1, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, BLAS `Haswell` via `libscipy_openblas64_-61654e39.so`, OpenMP max threads 1.
- Every fit and every worker refit bound this identity; a worker whose environment differed is recorded as a failure, never merged.

## Probes

Per entry: the original episode plus the two largest fixed augmented banks (64 synthetic episodes each; anchor sigma 0.05 rad, phi 0.99, gamma 1.0, seed bank 1) harvested through the entry's reservoir after its warm-up; task rows only; probes never enter a residual fit's training.

| entry | bank | episodes | rows | attempts | rejections | states sha256 | anchors sha256 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| failure-actual-dwell | original | 1 | 400 |  |  | `623015ef492b` | `a3bc7f86bfd8` |
| failure-actual-dwell | non_decaying | 64 | 25600 | 64 | 0 | `3b72cb150c83` | `ab89a907348b` |
| failure-actual-dwell | contractive | 64 | 25600 | 64 | 0 | `a7cefedd3114` | `e5e93c615fbc` |
| failure-generated-dwell | original | 1 | 400 |  |  | `e30918e8b3fe` | `a3bc7f86bfd8` |
| failure-generated-dwell | non_decaying | 64 | 25600 | 64 | 0 | `c1754b666db5` | `ab89a907348b` |
| failure-generated-dwell | contractive | 64 | 25600 | 64 | 0 | `8c5f2a069cc3` | `e5e93c615fbc` |
| failure-joint-velocity | original | 1 | 400 |  |  | `127b93227e36` | `a3bc7f86bfd8` |
| failure-joint-velocity | non_decaying | 64 | 25600 | 64 | 0 | `a11d6e89b252` | `ab89a907348b` |
| failure-joint-velocity | contractive | 64 | 25600 | 64 | 0 | `f53e21cf1865` | `e5e93c615fbc` |
| feasible-best | original | 1 | 400 |  |  | `0b35a475d114` | `a3bc7f86bfd8` |
| feasible-best | non_decaying | 64 | 25600 | 64 | 0 | `b6e1c9b42375` | `ab89a907348b` |
| feasible-best | contractive | 64 | 25600 | 64 | 0 | `266e85e25d5b` | `e5e93c615fbc` |
| feasible-middle | original | 1 | 400 |  |  | `db23d14e2348` | `a3bc7f86bfd8` |
| feasible-middle | non_decaying | 64 | 25600 | 64 | 0 | `03bbcf0e44d5` | `ab89a907348b` |
| feasible-middle | contractive | 64 | 25600 | 64 | 0 | `41655808f542` | `e5e93c615fbc` |
| feasible-worst | original | 1 | 400 |  |  | `74eab91374c6` | `a3bc7f86bfd8` |
| feasible-worst | non_decaying | 64 | 25600 | 64 | 0 | `f60cca1d29f5` | `ab89a907348b` |
| feasible-worst | contractive | 64 | 25600 | 64 | 0 | `37221cd46b39` | `e5e93c615fbc` |

## Equivalence comparisons

Candidate against reference on the entry's probe matrix; residual pairs compare increments and the commands reconstructed as anchor + increment with identical anchors. Coefficient columns are the differences of the fitted weights (max abs, max relative, Frobenius relative).

| entry | formulation | K | candidate | reference | quantity | rows | max abs | max rel | worst bank | coef max abs | coef max rel | coef fro rel | decision |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| feasible-best | absolute | 17 | R/K17 | S-effective/K17 | prediction | 51600 | 3.714e-11 | 7.908e-11 | non_decaying | 1.187e-10 | 1.216e-07 | 6.662e-10 | pass |
| feasible-best | absolute | 17 | R-scaled/K17 | S | prediction | 51600 | 2.019e-12 | 5.557e-12 | non_decaying | 6.674e-12 | 2.716e-09 | 4.664e-11 | pass |
| feasible-best | absolute | 33 | R/K33 | S-effective/K33 | prediction | 51600 | 5.277e-11 | 1.044e-10 | non_decaying | 1.974e-10 | 2.977e-07 | 1.389e-09 | pass |
| feasible-best | absolute | 33 | R-scaled/K33 | S | prediction | 51600 | 1.432e-12 | 4.093e-12 | non_decaying | 6.956e-12 | 2.768e-09 | 4.934e-11 | pass |
| feasible-best | absolute | 65 | R/K65 | S-effective/K65 | prediction | 51600 | 1.756e-10 | 4.966e-10 | non_decaying | 5.786e-10 | 1.763e-07 | 3.371e-09 | pass |
| feasible-best | absolute | 65 | R-scaled/K65 | S | prediction | 51600 | 1.515e-12 | 6.082e-12 | non_decaying | 6.744e-12 | 3.749e-09 | 5.375e-11 | pass |
| feasible-best | residual | 17 | R/K17 | S-effective/K17 | increment | 51600 | 1.071e-13 | 3.755e-06 | contractive | 2.953e-13 | 1.172e-06 | 1.665e-10 | pass |
| feasible-best | residual | 17 | R/K17 | S-effective/K17 | command | 51600 | 1.071e-13 | 1.266e-12 | non_decaying | 2.953e-13 | 1.172e-06 | 1.665e-10 | pass |
| feasible-best | residual | 17 | R-scaled/K17 | S | increment | 51600 | 5.802e-15 | 1.385e-08 | non_decaying | 1.518e-14 | 1.153e-09 | 8.935e-12 | pass |
| feasible-best | residual | 17 | R-scaled/K17 | S | command | 51600 | 5.829e-15 | 6.738e-14 | non_decaying | 1.518e-14 | 1.153e-09 | 8.935e-12 | pass |
| feasible-best | residual | 33 | R/K33 | S-effective/K33 | increment | 51600 | 2.239e-13 | 1.768e-06 | contractive | 9.326e-13 | 1.047e-07 | 3.252e-10 | pass |
| feasible-best | residual | 33 | R/K33 | S-effective/K33 | command | 51600 | 2.239e-13 | 2.695e-12 | non_decaying | 9.326e-13 | 1.047e-07 | 3.252e-10 | pass |
| feasible-best | residual | 33 | R-scaled/K33 | S | increment | 51600 | 6.945e-15 | 1.366e-08 | non_decaying | 2.667e-14 | 1.039e-09 | 1.045e-11 | pass |
| feasible-best | residual | 33 | R-scaled/K33 | S | command | 51600 | 6.994e-15 | 7.511e-14 | non_decaying | 2.667e-14 | 1.039e-09 | 1.045e-11 | pass |
| feasible-best | residual | 65 | R/K65 | S-effective/K65 | increment | 51600 | 6.446e-13 | 9.777e-07 | non_decaying | 3.120e-12 | 5.723e-07 | 8.051e-10 | pass |
| feasible-best | residual | 65 | R/K65 | S-effective/K65 | command | 51600 | 6.446e-13 | 4.484e-12 | non_decaying | 3.120e-12 | 5.723e-07 | 8.051e-10 | pass |
| feasible-best | residual | 65 | R-scaled/K65 | S | increment | 51600 | 6.333e-15 | 1.757e-08 | non_decaying | 1.921e-14 | 6.819e-10 | 1.078e-11 | pass |
| feasible-best | residual | 65 | R-scaled/K65 | S | command | 51600 | 6.328e-15 | 8.211e-14 | non_decaying | 1.921e-14 | 6.819e-10 | 1.078e-11 | pass |
| feasible-middle | absolute | 17 | R/K17 | S-effective/K17 | prediction | 51600 | 2.881e-09 | 2.423e-09 | non_decaying | 6.897e-09 | 2.224e-03 | 3.290e-08 | pass |
| feasible-middle | absolute | 17 | R-scaled/K17 | S | prediction | 51600 | 1.079e-10 | 1.668e-10 | non_decaying | 4.176e-10 | 9.742e-05 | 1.929e-09 | pass |
| feasible-middle | absolute | 33 | R/K33 | S-effective/K33 | prediction | 51600 | 5.595e-09 | 4.745e-09 | non_decaying | 2.270e-08 | 6.201e-03 | 1.082e-07 | pass |
| feasible-middle | absolute | 33 | R-scaled/K33 | S | prediction | 51600 | 1.524e-10 | 1.674e-10 | non_decaying | 6.849e-10 | 1.168e-04 | 3.311e-09 | pass |
| feasible-middle | absolute | 65 | R/K65 | S-effective/K65 | prediction | 51600 | 2.929e-08 | 5.003e-08 | non_decaying | 8.865e-08 | 4.676e-03 | 3.906e-07 | **FAIL** |
| feasible-middle | absolute | 65 | R-scaled/K65 | S | prediction | 51600 | 4.456e-10 | 6.708e-10 | non_decaying | 1.385e-09 | 8.734e-05 | 6.256e-09 | pass |
| feasible-middle | residual | 17 | R/K17 | S-effective/K17 | increment | 51600 | 6.073e-12 | 3.393e-05 | non_decaying | 3.236e-11 | 2.722e-05 | 9.539e-09 | pass |
| feasible-middle | residual | 17 | R/K17 | S-effective/K17 | command | 51600 | 6.073e-12 | 3.960e-11 | non_decaying | 3.236e-11 | 2.722e-05 | 9.539e-09 | pass |
| feasible-middle | residual | 17 | R-scaled/K17 | S | increment | 51600 | 2.540e-13 | 2.041e-06 | non_decaying | 5.716e-13 | 4.683e-07 | 5.794e-10 | pass |
| feasible-middle | residual | 17 | R-scaled/K17 | S | command | 51600 | 2.540e-13 | 1.868e-12 | non_decaying | 5.716e-13 | 4.683e-07 | 5.794e-10 | pass |
| feasible-middle | residual | 33 | R/K33 | S-effective/K33 | increment | 51600 | 1.861e-11 | 1.763e-04 | non_decaying | 5.952e-11 | 5.808e-05 | 1.770e-08 | pass |
| feasible-middle | residual | 33 | R/K33 | S-effective/K33 | command | 51600 | 1.861e-11 | 6.470e-11 | non_decaying | 5.952e-11 | 5.808e-05 | 1.770e-08 | pass |
| feasible-middle | residual | 33 | R-scaled/K33 | S | increment | 51600 | 2.440e-13 | 1.572e-06 | non_decaying | 6.182e-13 | 2.646e-07 | 5.602e-10 | pass |
| feasible-middle | residual | 33 | R-scaled/K33 | S | command | 51600 | 2.440e-13 | 2.501e-12 | non_decaying | 6.182e-13 | 2.646e-07 | 5.602e-10 | pass |
| feasible-middle | residual | 65 | R/K65 | S-effective/K65 | increment | 51600 | 1.641e-11 | 2.165e-04 | non_decaying | 1.529e-10 | 3.532e-05 | 3.579e-08 | pass |
| feasible-middle | residual | 65 | R/K65 | S-effective/K65 | command | 51600 | 1.641e-11 | 1.194e-10 | non_decaying | 1.529e-10 | 3.532e-05 | 3.579e-08 | pass |
| feasible-middle | residual | 65 | R-scaled/K65 | S | increment | 51600 | 2.696e-13 | 1.364e-06 | non_decaying | 6.448e-13 | 3.188e-07 | 5.842e-10 | pass |
| feasible-middle | residual | 65 | R-scaled/K65 | S | command | 51600 | 2.696e-13 | 1.671e-12 | non_decaying | 6.448e-13 | 3.188e-07 | 5.842e-10 | pass |
| feasible-worst | absolute | 17 | R/K17 | S-effective/K17 | prediction | 51600 | 8.556e-11 | 4.593e-10 | contractive | 3.131e-09 | 7.164e-06 | 6.135e-09 | pass |
| feasible-worst | absolute | 17 | R-scaled/K17 | S | prediction | 51600 | 2.929e-12 | 1.488e-11 | non_decaying | 1.789e-10 | 4.078e-07 | 5.870e-10 | pass |
| feasible-worst | absolute | 33 | R/K33 | S-effective/K33 | prediction | 51600 | 1.409e-10 | 5.989e-10 | non_decaying | 7.061e-09 | 4.717e-05 | 1.309e-08 | pass |
| feasible-worst | absolute | 33 | R-scaled/K33 | S | prediction | 51600 | 3.328e-12 | 1.354e-11 | non_decaying | 1.695e-10 | 2.550e-07 | 6.528e-10 | pass |
| feasible-worst | absolute | 65 | R/K65 | S-effective/K65 | prediction | 51600 | 4.804e-10 | 1.509e-09 | non_decaying | 2.873e-08 | 1.296e-04 | 3.778e-08 | pass |
| feasible-worst | absolute | 65 | R-scaled/K65 | S | prediction | 51600 | 4.178e-12 | 2.006e-11 | non_decaying | 3.833e-10 | 5.825e-07 | 8.678e-10 | pass |
| feasible-worst | residual | 17 | R/K17 | S-effective/K17 | increment | 51600 | 1.369e-12 | 1.362e-05 | non_decaying | 6.232e-11 | 1.466e-06 | 4.486e-09 | pass |
| feasible-worst | residual | 17 | R/K17 | S-effective/K17 | command | 51600 | 1.369e-12 | 1.191e-11 | non_decaying | 6.232e-11 | 1.466e-06 | 4.486e-09 | pass |
| feasible-worst | residual | 17 | R-scaled/K17 | S | increment | 51600 | 4.383e-14 | 7.152e-06 | non_decaying | 1.724e-12 | 1.286e-07 | 2.876e-10 | pass |
| feasible-worst | residual | 17 | R-scaled/K17 | S | command | 51600 | 4.383e-14 | 3.437e-13 | non_decaying | 1.724e-12 | 1.286e-07 | 2.876e-10 | pass |
| feasible-worst | residual | 33 | R/K33 | S-effective/K33 | increment | 51600 | 3.309e-12 | 2.729e-04 | contractive | 1.293e-10 | 2.173e-06 | 8.312e-09 | pass |
| feasible-worst | residual | 33 | R/K33 | S-effective/K33 | command | 51600 | 3.309e-12 | 2.766e-11 | non_decaying | 1.293e-10 | 2.173e-06 | 8.312e-09 | pass |
| feasible-worst | residual | 33 | R-scaled/K33 | S | increment | 51600 | 5.427e-14 | 6.119e-06 | contractive | 3.663e-12 | 1.236e-07 | 3.275e-10 | pass |
| feasible-worst | residual | 33 | R-scaled/K33 | S | command | 51600 | 5.429e-14 | 5.192e-13 | non_decaying | 3.663e-12 | 1.236e-07 | 3.275e-10 | pass |
| feasible-worst | residual | 65 | R/K65 | S-effective/K65 | increment | 51600 | 1.402e-11 | 1.925e-04 | non_decaying | 4.864e-10 | 9.624e-06 | 2.292e-08 | pass |
| feasible-worst | residual | 65 | R/K65 | S-effective/K65 | command | 51600 | 1.402e-11 | 1.301e-10 | non_decaying | 4.864e-10 | 9.624e-06 | 2.292e-08 | pass |
| feasible-worst | residual | 65 | R-scaled/K65 | S | increment | 51600 | 5.568e-14 | 1.425e-05 | non_decaying | 6.081e-12 | 1.356e-07 | 4.059e-10 | pass |
| feasible-worst | residual | 65 | R-scaled/K65 | S | command | 51600 | 5.568e-14 | 4.721e-13 | non_decaying | 6.081e-12 | 1.356e-07 | 4.059e-10 | pass |
| failure-actual-dwell | absolute | 17 | R/K17 | S-effective/K17 | prediction | 51600 | 1.575e-12 | 1.361e-12 | non_decaying | 2.179e-11 | 2.625e-08 | 5.635e-11 | pass |
| failure-actual-dwell | absolute | 17 | R-scaled/K17 | S | prediction | 51600 | 1.783e-13 | 2.348e-13 | non_decaying | 1.109e-12 | 3.773e-08 | 3.836e-12 | pass |
| failure-actual-dwell | absolute | 33 | R/K33 | S-effective/K33 | prediction | 51600 | 1.916e-11 | 1.664e-11 | non_decaying | 7.831e-11 | 5.646e-07 | 1.925e-10 | pass |
| failure-actual-dwell | absolute | 33 | R-scaled/K33 | S | prediction | 51600 | 4.663e-13 | 4.055e-13 | non_decaying | 1.761e-12 | 2.442e-08 | 6.187e-12 | pass |
| failure-actual-dwell | absolute | 65 | R/K65 | S-effective/K65 | prediction | 51600 | 4.786e-11 | 4.160e-11 | non_decaying | 3.506e-10 | 2.131e-06 | 6.284e-10 | pass |
| failure-actual-dwell | absolute | 65 | R-scaled/K65 | S | prediction | 51600 | 5.598e-13 | 1.050e-12 | non_decaying | 3.820e-12 | 1.303e-08 | 1.030e-11 | pass |
| failure-actual-dwell | residual | 17 | R/K17 | S-effective/K17 | increment | 51600 | 4.358e-15 | 1.343e-07 | contractive | 7.872e-14 | 5.117e-09 | 1.707e-11 | pass |
| failure-actual-dwell | residual | 17 | R/K17 | S-effective/K17 | command | 51600 | 4.441e-15 | 1.104e-14 | non_decaying | 7.872e-14 | 5.117e-09 | 1.707e-11 | pass |
| failure-actual-dwell | residual | 17 | R-scaled/K17 | S | increment | 51600 | 2.654e-16 | 5.914e-08 | non_decaying | 2.202e-15 | 2.140e-10 | 9.716e-13 | pass |
| failure-actual-dwell | residual | 17 | R-scaled/K17 | S | command | 51600 | 2.637e-16 | 3.290e-15 | non_decaying | 2.202e-15 | 2.140e-10 | 9.716e-13 | pass |
| failure-actual-dwell | residual | 33 | R/K33 | S-effective/K33 | increment | 51600 | 7.796e-15 | 4.479e-08 | contractive | 1.614e-13 | 1.481e-08 | 2.987e-11 | pass |
| failure-actual-dwell | residual | 33 | R/K33 | S-effective/K33 | command | 51600 | 7.772e-15 | 2.832e-14 | non_decaying | 1.614e-13 | 1.481e-08 | 2.987e-11 | pass |
| failure-actual-dwell | residual | 33 | R-scaled/K33 | S | increment | 51600 | 4.122e-16 | 5.025e-08 | non_decaying | 2.404e-15 | 1.418e-10 | 9.394e-13 | pass |
| failure-actual-dwell | residual | 33 | R-scaled/K33 | S | command | 51600 | 4.163e-16 | 4.945e-15 | non_decaying | 2.404e-15 | 1.418e-10 | 9.394e-13 | pass |
| failure-actual-dwell | residual | 65 | R/K65 | S-effective/K65 | increment | 51600 | 2.670e-14 | 1.615e-06 | non_decaying | 3.762e-13 | 7.582e-08 | 6.533e-11 | pass |
| failure-actual-dwell | residual | 65 | R/K65 | S-effective/K65 | command | 51600 | 2.676e-14 | 1.745e-13 | non_decaying | 3.762e-13 | 7.582e-08 | 6.533e-11 | pass |
| failure-actual-dwell | residual | 65 | R-scaled/K65 | S | increment | 51600 | 6.457e-16 | 5.902e-08 | non_decaying | 2.053e-15 | 1.943e-10 | 1.025e-12 | pass |
| failure-actual-dwell | residual | 65 | R-scaled/K65 | S | command | 51600 | 6.384e-16 | 7.418e-15 | non_decaying | 2.053e-15 | 1.943e-10 | 1.025e-12 | pass |
| failure-joint-velocity | absolute | 17 | R/K17 | S-effective/K17 | prediction | 51600 | 9.117e-11 | 2.436e-10 | contractive | 6.262e-09 | 1.513e-05 | 1.516e-08 | pass |
| failure-joint-velocity | absolute | 17 | R-scaled/K17 | S | prediction | 51600 | 6.148e-12 | 1.603e-11 | non_decaying | 3.520e-10 | 4.766e-07 | 1.529e-09 | pass |
| failure-joint-velocity | absolute | 33 | R/K33 | S-effective/K33 | prediction | 51600 | 2.290e-10 | 1.195e-09 | non_decaying | 1.185e-08 | 1.268e-04 | 2.587e-08 | pass |
| failure-joint-velocity | absolute | 33 | R-scaled/K33 | S | prediction | 51600 | 8.288e-12 | 1.401e-11 | non_decaying | 3.567e-10 | 9.195e-07 | 1.507e-09 | pass |
| failure-joint-velocity | absolute | 65 | R/K65 | S-effective/K65 | prediction | 51600 | 6.958e-10 | 3.603e-09 | non_decaying | 2.422e-08 | 2.887e-04 | 4.929e-08 | pass |
| failure-joint-velocity | absolute | 65 | R-scaled/K65 | S | prediction | 51600 | 4.718e-12 | 1.827e-11 | non_decaying | 3.801e-10 | 1.024e-06 | 1.496e-09 | pass |
| failure-joint-velocity | residual | 17 | R/K17 | S-effective/K17 | increment | 51600 | 1.991e-12 | 9.294e-05 | non_decaying | 1.429e-10 | 3.631e-06 | 8.553e-09 | pass |
| failure-joint-velocity | residual | 17 | R/K17 | S-effective/K17 | command | 51600 | 1.991e-12 | 1.612e-11 | non_decaying | 1.429e-10 | 3.631e-06 | 8.553e-09 | pass |
| failure-joint-velocity | residual | 17 | R-scaled/K17 | S | increment | 51600 | 7.800e-14 | 5.789e-07 | non_decaying | 3.759e-12 | 1.818e-07 | 5.276e-10 | pass |
| failure-joint-velocity | residual | 17 | R-scaled/K17 | S | command | 51600 | 7.799e-14 | 7.375e-13 | non_decaying | 3.759e-12 | 1.818e-07 | 5.276e-10 | pass |
| failure-joint-velocity | residual | 33 | R/K33 | S-effective/K33 | increment | 51600 | 4.755e-12 | 1.974e-05 | non_decaying | 5.861e-10 | 1.226e-05 | 1.713e-08 | pass |
| failure-joint-velocity | residual | 33 | R/K33 | S-effective/K33 | command | 51600 | 4.756e-12 | 4.762e-11 | non_decaying | 5.861e-10 | 1.226e-05 | 1.713e-08 | pass |
| failure-joint-velocity | residual | 33 | R-scaled/K33 | S | increment | 51600 | 1.540e-13 | 3.917e-07 | contractive | 4.834e-12 | 1.721e-07 | 5.442e-10 | pass |
| failure-joint-velocity | residual | 33 | R-scaled/K33 | S | command | 51600 | 1.540e-13 | 1.199e-12 | non_decaying | 4.834e-12 | 1.721e-07 | 5.442e-10 | pass |
| failure-joint-velocity | residual | 65 | R/K65 | S-effective/K65 | increment | 51600 | 1.059e-11 | 4.397e-05 | non_decaying | 1.051e-09 | 1.907e-05 | 3.338e-08 | pass |
| failure-joint-velocity | residual | 65 | R/K65 | S-effective/K65 | command | 51600 | 1.059e-11 | 9.324e-11 | non_decaying | 1.051e-09 | 1.907e-05 | 3.338e-08 | pass |
| failure-joint-velocity | residual | 65 | R-scaled/K65 | S | increment | 51600 | 7.939e-14 | 4.068e-07 | non_decaying | 3.858e-12 | 2.140e-07 | 5.270e-10 | pass |
| failure-joint-velocity | residual | 65 | R-scaled/K65 | S | command | 51600 | 7.938e-14 | 5.501e-13 | non_decaying | 3.858e-12 | 2.140e-07 | 5.270e-10 | pass |
| failure-generated-dwell | absolute | 17 | R/K17 | S-effective/K17 | prediction | 51600 | 1.686e-11 | 1.522e-11 | non_decaying | 3.437e-10 | 1.519e-05 | 9.830e-10 | pass |
| failure-generated-dwell | absolute | 17 | R-scaled/K17 | S | prediction | 51600 | 8.726e-13 | 2.702e-12 | non_decaying | 1.864e-11 | 6.987e-08 | 6.447e-11 | pass |
| failure-generated-dwell | absolute | 33 | R/K33 | S-effective/K33 | prediction | 51600 | 3.480e-11 | 8.603e-11 | non_decaying | 1.203e-09 | 5.911e-06 | 3.030e-09 | pass |
| failure-generated-dwell | absolute | 33 | R-scaled/K33 | S | prediction | 51600 | 1.316e-12 | 4.379e-12 | non_decaying | 3.626e-11 | 9.856e-08 | 9.719e-11 | pass |
| failure-generated-dwell | absolute | 65 | R/K65 | S-effective/K65 | prediction | 51600 | 3.617e-10 | 4.743e-10 | non_decaying | 3.829e-09 | 7.620e-05 | 1.164e-08 | pass |
| failure-generated-dwell | absolute | 65 | R-scaled/K65 | S | prediction | 51600 | 5.221e-12 | 5.586e-12 | non_decaying | 5.089e-11 | 2.781e-07 | 1.881e-10 | pass |
| failure-generated-dwell | residual | 17 | R/K17 | S-effective/K17 | increment | 51600 | 2.252e-13 | 1.482e-05 | contractive | 1.557e-12 | 1.342e-07 | 2.745e-10 | pass |
| failure-generated-dwell | residual | 17 | R/K17 | S-effective/K17 | command | 51600 | 2.252e-13 | 2.213e-12 | non_decaying | 1.557e-12 | 1.342e-07 | 2.745e-10 | pass |
| failure-generated-dwell | residual | 17 | R-scaled/K17 | S | increment | 51600 | 6.733e-15 | 2.194e-06 | contractive | 5.960e-14 | 2.484e-08 | 1.577e-11 | pass |
| failure-generated-dwell | residual | 17 | R-scaled/K17 | S | command | 51600 | 6.717e-15 | 6.619e-14 | non_decaying | 5.960e-14 | 2.484e-08 | 1.577e-11 | pass |
| failure-generated-dwell | residual | 33 | R/K33 | S-effective/K33 | increment | 51600 | 3.791e-13 | 4.308e-05 | non_decaying | 3.147e-12 | 2.095e-06 | 5.215e-10 | pass |
| failure-generated-dwell | residual | 33 | R/K33 | S-effective/K33 | command | 51600 | 3.791e-13 | 3.959e-12 | non_decaying | 3.147e-12 | 2.095e-06 | 5.215e-10 | pass |
| failure-generated-dwell | residual | 33 | R-scaled/K33 | S | increment | 51600 | 4.754e-15 | 1.827e-06 | contractive | 5.657e-14 | 1.294e-08 | 1.513e-11 | pass |
| failure-generated-dwell | residual | 33 | R-scaled/K33 | S | command | 51600 | 4.746e-15 | 4.715e-14 | non_decaying | 5.657e-14 | 1.294e-08 | 1.513e-11 | pass |
| failure-generated-dwell | residual | 65 | R/K65 | S-effective/K65 | increment | 51600 | 7.504e-13 | 3.310e-06 | non_decaying | 7.972e-12 | 3.450e-07 | 1.028e-09 | pass |
| failure-generated-dwell | residual | 65 | R/K65 | S-effective/K65 | command | 51600 | 7.504e-13 | 6.436e-12 | non_decaying | 7.972e-12 | 3.450e-07 | 1.028e-09 | pass |
| failure-generated-dwell | residual | 65 | R-scaled/K65 | S | increment | 51600 | 6.121e-15 | 1.968e-06 | contractive | 5.407e-14 | 5.019e-09 | 1.598e-11 | pass |
| failure-generated-dwell | residual | 65 | R-scaled/K65 | S | command | 51600 | 6.134e-15 | 7.097e-14 | non_decaying | 5.407e-14 | 5.019e-09 | 1.598e-11 | pass |

## Fits

Every fit with its solver parameter, loss rows, teacher-forced fit errors, normalized normal-equation residual, conditioning of A = X^T X + alpha I (bias column last), accessor check (largest difference between rclib's predict and x W over the probes), and fit time.

| entry | arm | alpha | loss rows | rmse | max abs error | residual | cond2 | accessor | copies | fit s | cache | identity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| feasible-best | absolute/S | 0.119172 | 400 | 2.033e-03 | 2.823e-02 | 2.343e-16 | 1.305e+05 | 1.332e-15 | identical | 0.05 | fit | `9baa91c0b2e6` |
| feasible-best | absolute/R/K17 | 0.119172 | 6800 | 5.886e-04 | 8.345e-03 | 2.225e-16 | 2.218e+06 | 1.110e-15 | identical | 0.12 | fit | `0889179f658a` |
| feasible-best | absolute/R-scaled/K17 | 2.02592 | 6800 | 2.033e-03 | 2.823e-02 | 2.409e-16 | 1.305e+05 | 1.332e-15 | identical | 0.12 | fit | `fc3a64b6c038` |
| feasible-best | absolute/S-effective/K17 | 0.00701011 | 400 | 5.886e-04 | 8.345e-03 | 1.874e-16 | 2.218e+06 | 1.110e-15 | identical | 0.02 | fit | `33c496e31d85` |
| feasible-best | absolute/R/K33 | 0.119172 | 13200 | 4.491e-04 | 6.269e-03 | 2.349e-16 | 4.306e+06 | 1.221e-15 | identical | 0.21 | fit | `f7272a008d1a` |
| feasible-best | absolute/R-scaled/K33 | 3.93267 | 13200 | 2.033e-03 | 2.823e-02 | 2.543e-16 | 1.305e+05 | 1.332e-15 | identical | 0.20 | fit | `c619744ebc40` |
| feasible-best | absolute/S-effective/K33 | 0.00361127 | 400 | 4.491e-04 | 6.269e-03 | 2.024e-16 | 4.306e+06 | 1.332e-15 | identical | 0.02 | fit | `8467fecca599` |
| feasible-best | absolute/R/K65 | 0.119172 | 26000 | 3.411e-04 | 4.674e-03 | 3.033e-16 | 8.482e+06 | 1.221e-15 | identical | 0.41 | fit | `026da308e431` |
| feasible-best | absolute/R-scaled/K65 | 7.74617 | 26000 | 2.033e-03 | 2.823e-02 | 2.725e-16 | 1.305e+05 | 1.110e-15 | identical | 0.39 | fit | `e9f58149e6f1` |
| feasible-best | absolute/S-effective/K65 | 0.00183341 | 400 | 3.411e-04 | 4.674e-03 | 1.651e-16 | 8.482e+06 | 1.332e-15 | identical | 0.02 | fit | `0e15bb55858e` |
| feasible-best | residual/S | 0.119172 | 400 | 3.544e-05 | 1.400e-04 | 6.274e-17 | 1.305e+05 | 4.337e-18 | identical | 0.03 | fit | `95050c77cb0c` |
| feasible-best | residual/R/K17 | 0.119172 | 6800 | 2.078e-05 | 8.528e-05 | 7.331e-17 | 2.218e+06 | 6.939e-18 | identical | 0.12 | fit | `c60a46959c07` |
| feasible-best | residual/R-scaled/K17 | 2.02592 | 6800 | 3.544e-05 | 1.400e-04 | 5.965e-17 | 1.305e+05 | 4.337e-18 | identical | 0.12 | fit | `4566bb910916` |
| feasible-best | residual/S-effective/K17 | 0.00701011 | 400 | 2.078e-05 | 8.528e-05 | 5.587e-17 | 2.218e+06 | 7.373e-18 | identical | 0.02 | fit | `b57b698a173c` |
| feasible-best | residual/R/K33 | 0.119172 | 13200 | 1.709e-05 | 7.393e-05 | 7.117e-17 | 4.306e+06 | 8.240e-18 | identical | 0.21 | fit | `d662c6f2b20b` |
| feasible-best | residual/R-scaled/K33 | 3.93267 | 13200 | 3.544e-05 | 1.400e-04 | 6.768e-17 | 1.305e+05 | 5.204e-18 | identical | 0.20 | fit | `bc151c899a18` |
| feasible-best | residual/S-effective/K33 | 0.00361127 | 400 | 1.709e-05 | 7.393e-05 | 5.370e-17 | 4.306e+06 | 7.806e-18 | identical | 0.02 | fit | `58c2ce25d06d` |
| feasible-best | residual/R/K65 | 0.119172 | 26000 | 1.399e-05 | 6.357e-05 | 9.222e-17 | 8.482e+06 | 8.674e-18 | identical | 0.40 | fit | `10a69922a954` |
| feasible-best | residual/R-scaled/K65 | 7.74617 | 26000 | 3.544e-05 | 1.400e-04 | 7.478e-17 | 1.305e+05 | 5.204e-18 | identical | 0.39 | fit | `926ddc8e0090` |
| feasible-best | residual/S-effective/K65 | 0.00183341 | 400 | 1.399e-05 | 6.357e-05 | 6.251e-17 | 8.482e+06 | 9.541e-18 | identical | 0.02 | fit | `cee6141d4b76` |
| feasible-middle | absolute/S | 0.00309759 | 400 | 2.399e-04 | 2.206e-03 | 6.334e-17 | 1.648e+07 | 8.882e-16 | identical | 0.05 | fit | `ffacc2933af2` |
| feasible-middle | absolute/R/K17 | 0.00309759 | 6800 | 1.096e-04 | 4.590e-04 | 1.071e-16 | 2.801e+08 | 8.882e-16 | identical | 0.50 | fit | `2479559b1e36` |
| feasible-middle | absolute/R-scaled/K17 | 0.052659 | 6800 | 2.399e-04 | 2.206e-03 | 1.055e-16 | 1.648e+07 | 9.992e-16 | identical | 0.50 | fit | `a82642e20c7c` |
| feasible-middle | absolute/S-effective/K17 | 0.000182211 | 400 | 1.096e-04 | 4.590e-04 | 7.280e-17 | 2.801e+08 | 8.882e-16 | identical | 0.05 | fit | `07040a490e38` |
| feasible-middle | absolute/R/K33 | 0.00309759 | 13200 | 9.417e-05 | 3.984e-04 | 1.811e-16 | 5.438e+08 | 8.882e-16 | identical | 0.97 | fit | `c6bfad9af37d` |
| feasible-middle | absolute/R-scaled/K33 | 0.10222 | 13200 | 2.399e-04 | 2.206e-03 | 1.822e-16 | 1.648e+07 | 8.882e-16 | identical | 0.96 | fit | `f0e41d6dcb9c` |
| feasible-middle | absolute/S-effective/K33 | 9.38664e-05 | 400 | 9.417e-05 | 3.984e-04 | 7.774e-17 | 5.438e+08 | 8.882e-16 | identical | 0.05 | fit | `435f6306b1e5` |
| feasible-middle | absolute/R/K65 | 0.00309759 | 26000 | 7.720e-05 | 3.358e-04 | 3.435e-16 | 1.071e+09 | 9.992e-16 | identical | 1.90 | fit | `6912bffcc9a5` |
| feasible-middle | absolute/R-scaled/K65 | 0.201343 | 26000 | 2.399e-04 | 2.206e-03 | 3.549e-16 | 1.648e+07 | 9.992e-16 | identical | 1.90 | fit | `458039a5d4b4` |
| feasible-middle | absolute/S-effective/K65 | 4.76552e-05 | 400 | 7.720e-05 | 3.358e-04 | 5.890e-17 | 1.071e+09 | 8.882e-16 | identical | 0.05 | fit | `cc62950ed9a0` |
| feasible-middle | residual/S | 0.00309759 | 400 | 1.334e-05 | 5.694e-05 | 3.429e-17 | 1.648e+07 | 1.128e-17 | identical | 0.05 | fit | `6c1927cdfd4b` |
| feasible-middle | residual/R/K17 | 0.00309759 | 6800 | 7.036e-06 | 3.481e-05 | 5.127e-17 | 2.801e+08 | 1.388e-17 | identical | 0.51 | fit | `c1cea5e24c0d` |
| feasible-middle | residual/R-scaled/K17 | 0.052659 | 6800 | 1.334e-05 | 5.694e-05 | 1.039e-16 | 1.648e+07 | 8.674e-18 | identical | 0.49 | fit | `5b343d828893` |
| feasible-middle | residual/S-effective/K17 | 0.000182211 | 400 | 7.036e-06 | 3.481e-05 | 4.121e-17 | 2.801e+08 | 1.648e-17 | identical | 0.05 | fit | `3a94ee11e2dd` |
| feasible-middle | residual/R/K33 | 0.00309759 | 13200 | 5.946e-06 | 3.045e-05 | 5.066e-17 | 5.438e+08 | 2.082e-17 | identical | 0.96 | fit | `b407cc1dbd07` |
| feasible-middle | residual/R-scaled/K33 | 0.10222 | 13200 | 1.334e-05 | 5.694e-05 | 5.423e-17 | 1.648e+07 | 1.041e-17 | identical | 0.95 | fit | `6af42ed29f7f` |
| feasible-middle | residual/S-effective/K33 | 9.38664e-05 | 400 | 5.946e-06 | 3.045e-05 | 3.292e-17 | 5.438e+08 | 2.168e-17 | identical | 0.05 | fit | `eb5d2a0ac40b` |
| feasible-middle | residual/R/K65 | 0.00309759 | 26000 | 5.081e-06 | 2.660e-05 | 4.696e-17 | 1.071e+09 | 2.515e-17 | identical | 1.89 | fit | `cfe7829b94df` |
| feasible-middle | residual/R-scaled/K65 | 0.201343 | 26000 | 1.334e-05 | 5.694e-05 | 5.210e-17 | 1.648e+07 | 8.674e-18 | identical | 1.88 | fit | `f71226d1d56a` |
| feasible-middle | residual/S-effective/K65 | 4.76552e-05 | 400 | 5.081e-06 | 2.660e-05 | 3.383e-17 | 1.071e+09 | 2.472e-17 | identical | 0.05 | fit | `d2e250cbb2c2` |
| feasible-worst | absolute/S | 0.00493549 | 400 | 4.593e-03 | 1.370e-02 | 6.028e-17 | 7.944e+06 | 2.442e-15 | identical | 0.04 | fit | `21bea7cdbaa1` |
| feasible-worst | absolute/R/K17 | 0.00493549 | 6800 | 1.634e-03 | 4.572e-03 | 4.345e-17 | 1.350e+08 | 4.219e-15 | identical | 0.43 | fit | `5f4e539297ad` |
| feasible-worst | absolute/R-scaled/K17 | 0.0839033 | 6800 | 4.593e-03 | 1.370e-02 | 6.870e-17 | 7.944e+06 | 2.220e-15 | identical | 0.41 | fit | `441b52f5553e` |
| feasible-worst | absolute/S-effective/K17 | 0.000290323 | 400 | 1.634e-03 | 4.572e-03 | 4.406e-17 | 1.350e+08 | 3.997e-15 | identical | 0.04 | fit | `a94886975636` |
| feasible-worst | absolute/R/K33 | 0.00493549 | 13200 | 1.405e-03 | 4.085e-03 | 4.725e-17 | 2.621e+08 | 3.997e-15 | identical | 0.80 | fit | `c12efc6f717e` |
| feasible-worst | absolute/R-scaled/K33 | 0.162871 | 13200 | 4.593e-03 | 1.370e-02 | 7.610e-17 | 7.944e+06 | 2.220e-15 | identical | 0.80 | fit | `1db0df533258` |
| feasible-worst | absolute/S-effective/K33 | 0.00014956 | 400 | 1.405e-03 | 4.085e-03 | 4.227e-17 | 2.621e+08 | 4.441e-15 | identical | 0.04 | fit | `b33e44b2495a` |
| feasible-worst | absolute/R/K65 | 0.00493549 | 26000 | 1.240e-03 | 4.289e-03 | 7.235e-17 | 5.163e+08 | 4.441e-15 | identical | 1.56 | fit | `c5edfc0c3d1d` |
| feasible-worst | absolute/R-scaled/K65 | 0.320807 | 26000 | 4.593e-03 | 1.370e-02 | 1.077e-16 | 7.944e+06 | 2.442e-15 | identical | 1.57 | fit | `4541eb224351` |
| feasible-worst | absolute/S-effective/K65 | 7.59306e-05 | 400 | 1.240e-03 | 4.289e-03 | 3.459e-17 | 5.163e+08 | 4.663e-15 | identical | 0.04 | fit | `62c85654854f` |
| feasible-worst | residual/S | 0.00493549 | 400 | 8.359e-05 | 2.602e-04 | 3.517e-17 | 7.944e+06 | 2.862e-17 | identical | 0.05 | fit | `23f2f9a816e0` |
| feasible-worst | residual/R/K17 | 0.00493549 | 6800 | 4.891e-05 | 1.639e-04 | 3.008e-17 | 1.350e+08 | 4.250e-17 | identical | 0.44 | fit | `9d98ca3f26db` |
| feasible-worst | residual/R-scaled/K17 | 0.0839033 | 6800 | 8.359e-05 | 2.602e-04 | 3.177e-17 | 7.944e+06 | 2.949e-17 | identical | 0.42 | fit | `c27575d1243c` |
| feasible-worst | residual/S-effective/K17 | 0.000290323 | 400 | 4.891e-05 | 1.639e-04 | 2.627e-17 | 1.350e+08 | 5.031e-17 | identical | 0.04 | fit | `830a484c1bd6` |
| feasible-worst | residual/R/K33 | 0.00493549 | 13200 | 4.163e-05 | 1.448e-04 | 2.949e-17 | 2.621e+08 | 5.031e-17 | identical | 0.80 | fit | `a6075b54b445` |
| feasible-worst | residual/R-scaled/K33 | 0.162871 | 13200 | 8.359e-05 | 2.602e-04 | 3.987e-17 | 7.944e+06 | 2.602e-17 | identical | 0.82 | fit | `edf1e3ee4d1d` |
| feasible-worst | residual/S-effective/K33 | 0.00014956 | 400 | 4.163e-05 | 1.448e-04 | 2.732e-17 | 2.621e+08 | 5.031e-17 | identical | 0.04 | fit | `df7b5d6a50eb` |
| feasible-worst | residual/R/K65 | 0.00493549 | 26000 | 3.490e-05 | 1.268e-04 | 4.528e-17 | 5.163e+08 | 7.459e-17 | identical | 1.58 | fit | `7934c494b929` |
| feasible-worst | residual/R-scaled/K65 | 0.320807 | 26000 | 8.359e-05 | 2.602e-04 | 5.342e-17 | 7.944e+06 | 3.209e-17 | identical | 1.57 | fit | `c3994ce4aa7c` |
| feasible-worst | residual/S-effective/K65 | 7.59306e-05 | 400 | 3.490e-05 | 1.268e-04 | 3.088e-17 | 5.163e+08 | 7.286e-17 | identical | 0.04 | fit | `420227666caa` |
| failure-actual-dwell | absolute/S | 0.619159 | 400 | 1.353e-02 | 1.060e-01 | 1.672e-16 | 1.461e+04 | 6.661e-16 | identical | 0.02 | fit | `8494f5e7d203` |
| failure-actual-dwell | absolute/R/K17 | 0.619159 | 6800 | 3.477e-03 | 1.919e-02 | 2.079e-16 | 2.483e+05 | 5.551e-16 | identical | 0.11 | fit | `1119979643f0` |
| failure-actual-dwell | absolute/R-scaled/K17 | 10.5257 | 6800 | 1.353e-02 | 1.060e-01 | 2.211e-16 | 1.461e+04 | 8.882e-16 | identical | 0.10 | fit | `9ad88b3b9036` |
| failure-actual-dwell | absolute/S-effective/K17 | 0.0364211 | 400 | 3.477e-03 | 1.919e-02 | 1.181e-16 | 2.483e+05 | 7.772e-16 | identical | 0.02 | fit | `5f386198ddcf` |
| failure-actual-dwell | absolute/R/K33 | 0.619159 | 13200 | 2.408e-03 | 1.214e-02 | 3.779e-16 | 4.821e+05 | 6.661e-16 | identical | 0.20 | fit | `d0bd216d547a` |
| failure-actual-dwell | absolute/R-scaled/K33 | 20.4322 | 13200 | 1.353e-02 | 1.060e-01 | 3.840e-16 | 1.461e+04 | 7.772e-16 | identical | 0.20 | fit | `3557f212a205` |
| failure-actual-dwell | absolute/S-effective/K33 | 0.0187624 | 400 | 2.408e-03 | 1.214e-02 | 1.134e-16 | 4.821e+05 | 8.882e-16 | identical | 0.02 | fit | `435ed41ba991` |
| failure-actual-dwell | absolute/R/K65 | 0.619159 | 26000 | 1.633e-03 | 7.407e-03 | 5.740e-16 | 9.495e+05 | 6.661e-16 | identical | 0.36 | fit | `5197cf29bdd8` |
| failure-actual-dwell | absolute/R-scaled/K65 | 40.2453 | 26000 | 1.353e-02 | 1.060e-01 | 5.757e-16 | 1.461e+04 | 7.772e-16 | identical | 0.37 | fit | `ed8819050ecc` |
| failure-actual-dwell | absolute/S-effective/K65 | 0.00952552 | 400 | 1.633e-03 | 7.407e-03 | 1.113e-16 | 9.495e+05 | 6.661e-16 | identical | 0.02 | fit | `a2325600757e` |
| failure-actual-dwell | residual/S | 0.619159 | 400 | 1.502e-04 | 4.458e-04 | 4.508e-17 | 1.461e+04 | 4.337e-18 | identical | 0.02 | fit | `3f66b23003c3` |
| failure-actual-dwell | residual/R/K17 | 0.619159 | 6800 | 6.097e-05 | 2.099e-04 | 6.162e-17 | 2.483e+05 | 5.204e-18 | identical | 0.11 | fit | `a119eedc5dfc` |
| failure-actual-dwell | residual/R-scaled/K17 | 10.5257 | 6800 | 1.502e-04 | 4.458e-04 | 6.381e-17 | 1.461e+04 | 3.469e-18 | identical | 0.11 | fit | `97dce7be0e8a` |
| failure-actual-dwell | residual/S-effective/K17 | 0.0364211 | 400 | 6.097e-05 | 2.099e-04 | 5.080e-17 | 2.483e+05 | 5.855e-18 | identical | 0.02 | fit | `fba7e1fd67df` |
| failure-actual-dwell | residual/R/K33 | 0.619159 | 13200 | 5.370e-05 | 1.845e-04 | 4.906e-17 | 4.821e+05 | 6.939e-18 | identical | 0.20 | fit | `9d0e61b11197` |
| failure-actual-dwell | residual/R-scaled/K33 | 20.4322 | 13200 | 1.502e-04 | 4.458e-04 | 6.848e-17 | 1.461e+04 | 3.469e-18 | identical | 0.19 | fit | `c20a071221a0` |
| failure-actual-dwell | residual/S-effective/K33 | 0.0187624 | 400 | 5.370e-05 | 1.845e-04 | 4.721e-17 | 4.821e+05 | 6.559e-18 | identical | 0.02 | fit | `52f2c5602a27` |
| failure-actual-dwell | residual/R/K65 | 0.619159 | 26000 | 4.775e-05 | 1.637e-04 | 7.171e-17 | 9.495e+05 | 7.047e-18 | identical | 0.37 | fit | `d31dcc4131e3` |
| failure-actual-dwell | residual/R-scaled/K65 | 40.2453 | 26000 | 1.502e-04 | 4.458e-04 | 9.081e-17 | 1.461e+04 | 3.469e-18 | identical | 0.36 | fit | `5ad9642f287b` |
| failure-actual-dwell | residual/S-effective/K65 | 0.00952552 | 400 | 4.775e-05 | 1.637e-04 | 4.930e-17 | 9.495e+05 | 6.234e-18 | identical | 0.02 | fit | `f27e610f9df6` |
| failure-joint-velocity | absolute/S | 0.00284948 | 400 | 1.734e-03 | 5.064e-03 | 9.048e-17 | 1.337e+07 | 2.887e-15 | identical | 0.04 | fit | `0a58490284b1` |
| failure-joint-velocity | absolute/R/K17 | 0.00284948 | 6800 | 6.341e-04 | 2.002e-03 | 5.417e-17 | 2.273e+08 | 2.665e-15 | identical | 0.34 | fit | `a10d3056e727` |
| failure-joint-velocity | absolute/R-scaled/K17 | 0.0484411 | 6800 | 1.734e-03 | 5.064e-03 | 1.027e-16 | 1.337e+07 | 2.887e-15 | identical | 0.33 | fit | `d285e93b1bac` |
| failure-joint-velocity | absolute/S-effective/K17 | 0.000167616 | 400 | 6.341e-04 | 2.002e-03 | 5.512e-17 | 2.273e+08 | 2.887e-15 | identical | 0.04 | fit | `a0bc5aae167d` |
| failure-joint-velocity | absolute/R/K33 | 0.00284948 | 13200 | 5.369e-04 | 1.762e-03 | 5.197e-17 | 4.413e+08 | 2.665e-15 | identical | 0.63 | fit | `fe344f401d32` |
| failure-joint-velocity | absolute/R-scaled/K33 | 0.0940328 | 13200 | 1.734e-03 | 5.064e-03 | 9.776e-17 | 1.337e+07 | 2.887e-15 | identical | 0.62 | fit | `7a0bc2001462` |
| failure-joint-velocity | absolute/S-effective/K33 | 8.63478e-05 | 400 | 5.369e-04 | 1.762e-03 | 5.431e-17 | 4.413e+08 | 2.665e-15 | identical | 0.04 | fit | `cefe203e23f1` |
| failure-joint-velocity | absolute/R/K65 | 0.00284948 | 26000 | 4.548e-04 | 1.540e-03 | 5.327e-17 | 8.692e+08 | 2.665e-15 | identical | 1.23 | fit | `4ac3c3d39c52` |
| failure-joint-velocity | absolute/R-scaled/K65 | 0.185216 | 26000 | 1.734e-03 | 5.064e-03 | 1.050e-16 | 1.337e+07 | 2.887e-15 | identical | 1.22 | fit | `33ae4d8620c5` |
| failure-joint-velocity | absolute/S-effective/K65 | 4.38381e-05 | 400 | 4.548e-04 | 1.540e-03 | 4.513e-17 | 8.692e+08 | 2.665e-15 | identical | 0.04 | fit | `2b4dbbf4b2e4` |
| failure-joint-velocity | residual/S | 0.00284948 | 400 | 5.392e-05 | 1.985e-04 | 3.233e-17 | 1.337e+07 | 2.038e-17 | identical | 0.04 | fit | `afd2896b2c40` |
| failure-joint-velocity | residual/R/K17 | 0.00284948 | 6800 | 2.739e-05 | 1.092e-04 | 3.328e-17 | 2.273e+08 | 2.949e-17 | identical | 0.34 | fit | `7c818a33602e` |
| failure-joint-velocity | residual/R-scaled/K17 | 0.0484411 | 6800 | 5.392e-05 | 1.985e-04 | 3.480e-17 | 1.337e+07 | 2.255e-17 | identical | 0.33 | fit | `3c2997896c0c` |
| failure-joint-velocity | residual/S-effective/K17 | 0.000167616 | 400 | 2.739e-05 | 1.092e-04 | 2.970e-17 | 2.273e+08 | 3.036e-17 | identical | 0.04 | fit | `587806335246` |
| failure-joint-velocity | residual/R/K33 | 0.00284948 | 13200 | 2.265e-05 | 9.370e-05 | 3.644e-17 | 4.413e+08 | 3.469e-17 | identical | 0.63 | fit | `9cbe9d608b38` |
| failure-joint-velocity | residual/R-scaled/K33 | 0.0940328 | 13200 | 5.392e-05 | 1.985e-04 | 3.709e-17 | 1.337e+07 | 2.082e-17 | identical | 0.62 | fit | `1dcf97142e6d` |
| failure-joint-velocity | residual/S-effective/K33 | 8.63478e-05 | 400 | 2.265e-05 | 9.370e-05 | 2.912e-17 | 4.413e+08 | 3.816e-17 | identical | 0.04 | fit | `9a3677b9e0f7` |
| failure-joint-velocity | residual/R/K65 | 0.00284948 | 26000 | 1.879e-05 | 8.013e-05 | 3.387e-17 | 8.692e+08 | 5.486e-17 | identical | 1.24 | fit | `17e0a23a6558` |
| failure-joint-velocity | residual/R-scaled/K65 | 0.185216 | 26000 | 5.392e-05 | 1.985e-04 | 3.468e-17 | 1.337e+07 | 1.908e-17 | identical | 1.21 | fit | `8aea2df94268` |
| failure-joint-velocity | residual/S-effective/K65 | 4.38381e-05 | 400 | 1.879e-05 | 8.013e-05 | 2.953e-17 | 8.692e+08 | 5.031e-17 | identical | 0.04 | fit | `d3d3e96470d2` |
| failure-generated-dwell | absolute/S | 0.0836599 | 400 | 5.112e-03 | 8.707e-02 | 7.250e-17 | 5.294e+05 | 8.882e-16 | identical | 0.04 | fit | `c5d8919835a4` |
| failure-generated-dwell | absolute/R/K17 | 0.0836599 | 6800 | 8.181e-04 | 8.316e-03 | 9.698e-17 | 8.999e+06 | 8.049e-16 | identical | 0.35 | fit | `c8ae48ede5c8` |
| failure-generated-dwell | absolute/R-scaled/K17 | 1.42222 | 6800 | 5.112e-03 | 8.707e-02 | 1.099e-16 | 5.294e+05 | 1.110e-15 | identical | 0.35 | fit | `4381cc9b17b3` |
| failure-generated-dwell | absolute/S-effective/K17 | 0.00492117 | 400 | 8.181e-04 | 8.316e-03 | 5.932e-17 | 8.999e+06 | 8.882e-16 | identical | 0.04 | fit | `5841538a0e79` |
| failure-generated-dwell | absolute/R/K33 | 0.0836599 | 13200 | 6.244e-04 | 4.671e-03 | 1.664e-16 | 1.747e+07 | 8.882e-16 | identical | 0.66 | fit | `acf08d4bb586` |
| failure-generated-dwell | absolute/R-scaled/K33 | 2.76078 | 13200 | 5.112e-03 | 8.707e-02 | 1.706e-16 | 5.294e+05 | 8.882e-16 | identical | 0.65 | fit | `6bb9291e7897` |
| failure-generated-dwell | absolute/S-effective/K33 | 0.00253515 | 400 | 6.244e-04 | 4.671e-03 | 6.141e-17 | 1.747e+07 | 8.882e-16 | identical | 0.04 | fit | `7a78a3974d31` |
| failure-generated-dwell | absolute/R/K65 | 0.0836599 | 26000 | 5.014e-04 | 2.583e-03 | 3.152e-16 | 3.441e+07 | 1.110e-15 | identical | 1.26 | fit | `2c8e82dd5352` |
| failure-generated-dwell | absolute/R-scaled/K65 | 5.4379 | 26000 | 5.112e-03 | 8.707e-02 | 3.294e-16 | 5.294e+05 | 8.882e-16 | identical | 1.24 | fit | `62a92a2becce` |
| failure-generated-dwell | absolute/S-effective/K65 | 0.00128708 | 400 | 5.014e-04 | 2.583e-03 | 5.853e-17 | 3.441e+07 | 1.110e-15 | identical | 0.04 | fit | `89a8fc45bb57` |
| failure-generated-dwell | residual/S | 0.0836599 | 400 | 1.261e-04 | 4.018e-04 | 2.859e-17 | 5.294e+05 | 1.388e-17 | identical | 0.04 | fit | `0762978f0fc3` |
| failure-generated-dwell | residual/R/K17 | 0.0836599 | 6800 | 5.522e-05 | 1.995e-04 | 2.658e-17 | 8.999e+06 | 2.515e-17 | identical | 0.35 | fit | `e8ef2d748487` |
| failure-generated-dwell | residual/R-scaled/K17 | 1.42222 | 6800 | 1.261e-04 | 4.018e-04 | 2.801e-17 | 5.294e+05 | 1.344e-17 | identical | 0.33 | fit | `8c1ee3cf0d0b` |
| failure-generated-dwell | residual/S-effective/K17 | 0.00492117 | 400 | 5.522e-05 | 1.995e-04 | 2.790e-17 | 8.999e+06 | 2.559e-17 | identical | 0.04 | fit | `3eb398ae8d5c` |
| failure-generated-dwell | residual/R/K33 | 0.0836599 | 13200 | 4.840e-05 | 1.824e-04 | 2.895e-17 | 1.747e+07 | 2.862e-17 | identical | 0.65 | fit | `f95a70636ec8` |
| failure-generated-dwell | residual/R-scaled/K33 | 2.76078 | 13200 | 1.261e-04 | 4.018e-04 | 2.759e-17 | 5.294e+05 | 1.301e-17 | identical | 0.65 | fit | `dda0b1774a6b` |
| failure-generated-dwell | residual/S-effective/K33 | 0.00253515 | 400 | 4.840e-05 | 1.824e-04 | 2.780e-17 | 1.747e+07 | 2.689e-17 | identical | 0.04 | fit | `7373098580ba` |
| failure-generated-dwell | residual/R/K65 | 0.0836599 | 26000 | 4.177e-05 | 1.621e-04 | 3.086e-17 | 3.441e+07 | 3.946e-17 | identical | 1.24 | fit | `57fa80ba0f61` |
| failure-generated-dwell | residual/R-scaled/K65 | 5.4379 | 26000 | 1.261e-04 | 4.018e-04 | 3.109e-17 | 5.294e+05 | 1.301e-17 | identical | 1.24 | fit | `82d58e138013` |
| failure-generated-dwell | residual/S-effective/K65 | 0.00128708 | 400 | 4.177e-05 | 1.621e-04 | 2.618e-17 | 3.441e+07 | 3.773e-17 | identical | 0.04 | fit | `0c5465e68449` |

## State identity across formulations

| entry | arm | absolute | residual | identical |
| --- | --- | --- | --- | --- |
| feasible-best | S | `9baa91c0b2e6` | `95050c77cb0c` | yes |
| feasible-best | R/K17 | `0889179f658a` | `c60a46959c07` | yes |
| feasible-best | R-scaled/K17 | `fc3a64b6c038` | `4566bb910916` | yes |
| feasible-best | S-effective/K17 | `33c496e31d85` | `b57b698a173c` | yes |
| feasible-best | R/K33 | `f7272a008d1a` | `d662c6f2b20b` | yes |
| feasible-best | R-scaled/K33 | `c619744ebc40` | `bc151c899a18` | yes |
| feasible-best | S-effective/K33 | `8467fecca599` | `58c2ce25d06d` | yes |
| feasible-best | R/K65 | `026da308e431` | `10a69922a954` | yes |
| feasible-best | R-scaled/K65 | `e9f58149e6f1` | `926ddc8e0090` | yes |
| feasible-best | S-effective/K65 | `0e15bb55858e` | `cee6141d4b76` | yes |
| feasible-middle | S | `ffacc2933af2` | `6c1927cdfd4b` | yes |
| feasible-middle | R/K17 | `2479559b1e36` | `c1cea5e24c0d` | yes |
| feasible-middle | R-scaled/K17 | `a82642e20c7c` | `5b343d828893` | yes |
| feasible-middle | S-effective/K17 | `07040a490e38` | `3a94ee11e2dd` | yes |
| feasible-middle | R/K33 | `c6bfad9af37d` | `b407cc1dbd07` | yes |
| feasible-middle | R-scaled/K33 | `f0e41d6dcb9c` | `6af42ed29f7f` | yes |
| feasible-middle | S-effective/K33 | `435f6306b1e5` | `eb5d2a0ac40b` | yes |
| feasible-middle | R/K65 | `6912bffcc9a5` | `cfe7829b94df` | yes |
| feasible-middle | R-scaled/K65 | `458039a5d4b4` | `f71226d1d56a` | yes |
| feasible-middle | S-effective/K65 | `cc62950ed9a0` | `d2e250cbb2c2` | yes |
| feasible-worst | S | `21bea7cdbaa1` | `23f2f9a816e0` | yes |
| feasible-worst | R/K17 | `5f4e539297ad` | `9d98ca3f26db` | yes |
| feasible-worst | R-scaled/K17 | `441b52f5553e` | `c27575d1243c` | yes |
| feasible-worst | S-effective/K17 | `a94886975636` | `830a484c1bd6` | yes |
| feasible-worst | R/K33 | `c12efc6f717e` | `a6075b54b445` | yes |
| feasible-worst | R-scaled/K33 | `1db0df533258` | `edf1e3ee4d1d` | yes |
| feasible-worst | S-effective/K33 | `b33e44b2495a` | `df7b5d6a50eb` | yes |
| feasible-worst | R/K65 | `c5edfc0c3d1d` | `7934c494b929` | yes |
| feasible-worst | R-scaled/K65 | `4541eb224351` | `c3994ce4aa7c` | yes |
| feasible-worst | S-effective/K65 | `62c85654854f` | `420227666caa` | yes |
| failure-actual-dwell | S | `8494f5e7d203` | `3f66b23003c3` | yes |
| failure-actual-dwell | R/K17 | `1119979643f0` | `a119eedc5dfc` | yes |
| failure-actual-dwell | R-scaled/K17 | `9ad88b3b9036` | `97dce7be0e8a` | yes |
| failure-actual-dwell | S-effective/K17 | `5f386198ddcf` | `fba7e1fd67df` | yes |
| failure-actual-dwell | R/K33 | `d0bd216d547a` | `9d0e61b11197` | yes |
| failure-actual-dwell | R-scaled/K33 | `3557f212a205` | `c20a071221a0` | yes |
| failure-actual-dwell | S-effective/K33 | `435ed41ba991` | `52f2c5602a27` | yes |
| failure-actual-dwell | R/K65 | `5197cf29bdd8` | `d31dcc4131e3` | yes |
| failure-actual-dwell | R-scaled/K65 | `ed8819050ecc` | `5ad9642f287b` | yes |
| failure-actual-dwell | S-effective/K65 | `a2325600757e` | `f27e610f9df6` | yes |
| failure-joint-velocity | S | `0a58490284b1` | `afd2896b2c40` | yes |
| failure-joint-velocity | R/K17 | `a10d3056e727` | `7c818a33602e` | yes |
| failure-joint-velocity | R-scaled/K17 | `d285e93b1bac` | `3c2997896c0c` | yes |
| failure-joint-velocity | S-effective/K17 | `a0bc5aae167d` | `587806335246` | yes |
| failure-joint-velocity | R/K33 | `fe344f401d32` | `9cbe9d608b38` | yes |
| failure-joint-velocity | R-scaled/K33 | `7a0bc2001462` | `1dcf97142e6d` | yes |
| failure-joint-velocity | S-effective/K33 | `cefe203e23f1` | `9a3677b9e0f7` | yes |
| failure-joint-velocity | R/K65 | `4ac3c3d39c52` | `17e0a23a6558` | yes |
| failure-joint-velocity | R-scaled/K65 | `33ae4d8620c5` | `8aea2df94268` | yes |
| failure-joint-velocity | S-effective/K65 | `2b4dbbf4b2e4` | `d3d3e96470d2` | yes |
| failure-generated-dwell | S | `c5d8919835a4` | `0762978f0fc3` | yes |
| failure-generated-dwell | R/K17 | `c8ae48ede5c8` | `e8ef2d748487` | yes |
| failure-generated-dwell | R-scaled/K17 | `4381cc9b17b3` | `8c1ee3cf0d0b` | yes |
| failure-generated-dwell | S-effective/K17 | `5841538a0e79` | `3eb398ae8d5c` | yes |
| failure-generated-dwell | R/K33 | `acf08d4bb586` | `f95a70636ec8` | yes |
| failure-generated-dwell | R-scaled/K33 | `6bb9291e7897` | `dda0b1774a6b` | yes |
| failure-generated-dwell | S-effective/K33 | `7a78a3974d31` | `7373098580ba` | yes |
| failure-generated-dwell | R/K65 | `2c8e82dd5352` | `57fa80ba0f61` | yes |
| failure-generated-dwell | R-scaled/K65 | `62a92a2becce` | `82d58e138013` | yes |
| failure-generated-dwell | S-effective/K65 | `89a8fc45bb57` | `0c5465e68449` | yes |

## Fresh-process refits

Each cached fit refitted in a new pinned interpreter from its recipe TOML and the digest-verified dataset only (the cache is consulted after the fit, for the comparison).

| identity | environment | weights | max abs weight diff | states | fit report | rmse diff | s | decision |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| `9baa91c0b2e6` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `0889179f658a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.16 | pass |
| `fc3a64b6c038` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.17 | pass |
| `33c496e31d85` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `f7272a008d1a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.28 | pass |
| `c619744ebc40` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.28 | pass |
| `8467fecca599` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `026da308e431` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.54 | pass |
| `e9f58149e6f1` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.57 | pass |
| `0e15bb55858e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `95050c77cb0c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `c60a46959c07` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.15 | pass |
| `4566bb910916` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.16 | pass |
| `b57b698a173c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `d662c6f2b20b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.29 | pass |
| `bc151c899a18` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.28 | pass |
| `58c2ce25d06d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `10a69922a954` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.54 | pass |
| `926ddc8e0090` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.53 | pass |
| `cee6141d4b76` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `ffacc2933af2` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.07 | pass |
| `2479559b1e36` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.69 | pass |
| `a82642e20c7c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.68 | pass |
| `07040a490e38` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.07 | pass |
| `c6bfad9af37d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.31 | pass |
| `f0e41d6dcb9c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.29 | pass |
| `435f6306b1e5` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.07 | pass |
| `6912bffcc9a5` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.54 | pass |
| `458039a5d4b4` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.56 | pass |
| `cc62950ed9a0` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.07 | pass |
| `6c1927cdfd4b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.07 | pass |
| `c1cea5e24c0d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.71 | pass |
| `5b343d828893` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.69 | pass |
| `3a94ee11e2dd` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `b407cc1dbd07` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.30 | pass |
| `6af42ed29f7f` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.29 | pass |
| `eb5d2a0ac40b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.16 | pass |
| `cfe7829b94df` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.58 | pass |
| `f71226d1d56a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.56 | pass |
| `d2e250cbb2c2` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `21bea7cdbaa1` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `5f4e539297ad` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.59 | pass |
| `441b52f5553e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.60 | pass |
| `a94886975636` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `c12efc6f717e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.11 | pass |
| `1db0df533258` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.15 | pass |
| `b33e44b2495a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `c5edfc0c3d1d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.18 | pass |
| `4541eb224351` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.20 | pass |
| `62c85654854f` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `23f2f9a816e0` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `9d98ca3f26db` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.58 | pass |
| `c27575d1243c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.60 | pass |
| `830a484c1bd6` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `a6075b54b445` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.12 | pass |
| `edf1e3ee4d1d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.11 | pass |
| `df7b5d6a50eb` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `7934c494b929` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.17 | pass |
| `c3994ce4aa7c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 2.19 | pass |
| `420227666caa` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `8494f5e7d203` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `1119979643f0` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.15 | pass |
| `9ad88b3b9036` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.16 | pass |
| `5f386198ddcf` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `d0bd216d547a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.37 | pass |
| `3557f212a205` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.26 | pass |
| `435ed41ba991` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `5197cf29bdd8` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.47 | pass |
| `ed8819050ecc` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.47 | pass |
| `a2325600757e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `3f66b23003c3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `a119eedc5dfc` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.14 | pass |
| `97dce7be0e8a` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.14 | pass |
| `fba7e1fd67df` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `9d0e61b11197` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.25 | pass |
| `c20a071221a0` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.24 | pass |
| `52f2c5602a27` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `d31dcc4131e3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.59 | pass |
| `5ad9642f287b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.46 | pass |
| `f27e610f9df6` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.03 | pass |
| `0a58490284b1` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `a10d3056e727` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.48 | pass |
| `d285e93b1bac` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.45 | pass |
| `a0bc5aae167d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `fe344f401d32` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.87 | pass |
| `7a0bc2001462` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.87 | pass |
| `cefe203e23f1` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `4ac3c3d39c52` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.70 | pass |
| `33ae4d8620c5` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.71 | pass |
| `2b4dbbf4b2e4` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `afd2896b2c40` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `7c818a33602e` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.49 | pass |
| `3c2997896c0c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.46 | pass |
| `587806335246` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `9cbe9d608b38` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.87 | pass |
| `1dcf97142e6d` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.88 | pass |
| `9a3677b9e0f7` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `17e0a23a6558` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.70 | pass |
| `8aea2df94268` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.71 | pass |
| `d3d3e96470d2` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `c5d8919835a4` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `c8ae48ede5c8` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.48 | pass |
| `4381cc9b17b3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.45 | pass |
| `5841538a0e79` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `acf08d4bb586` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.88 | pass |
| `6bb9291e7897` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.87 | pass |
| `7a78a3974d31` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `2c8e82dd5352` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.71 | pass |
| `62a92a2becce` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.72 | pass |
| `89a8fc45bb57` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `0762978f0fc3` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `e8ef2d748487` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.48 | pass |
| `8c1ee3cf0d0b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.45 | pass |
| `3eb398ae8d5c` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |
| `f95a70636ec8` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.88 | pass |
| `dda0b1774a6b` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.87 | pass |
| `7373098580ba` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.06 | pass |
| `57fa80ba0f61` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.71 | pass |
| `82d58e138013` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 1.72 | pass |
| `0c5465e68449` | same | bitwise | 0.000e+00 | bitwise | equal | 0.000e+00 | 0.05 | pass |

## Limitations

- The prediction tolerance bounds teacher-forced readout differences on fixed probes; it is not a closed-loop stability guarantee (plan section 6).
- The equivalences are numerical identities of the ridge problem; agreement here says nothing about the behavioral effect of repetition, which M3REP-004 to M3REP-007 evaluate.
- Fresh-process bitwise reproducibility holds in the recorded execution environment only; another core type, thread setting, library build, or machine is a different environment (C10, condition 5).
- The normal-equation residual is evaluated with numpy on the harvested rows; the solve itself is rclib's LDLT (`cholesky` option) on its own accumulation of the same rows.
