# Execution record

- Identity `a7f034c7aef4` (schema 1); role `main`; canonical.
- Command `python -m arm_rc_ctrl.execution record --output execution_environment_v1.json --markdown execution_environment_v1.md`; created 2026-09-09T08:17:37+00:00.

## Affinity

- Policy `p-cores`; requested CPUs `0-15`; effective CPUs `0-15`; verified: True.
- CPU `Intel(R) Core(TM) i9-14900KF`; online `0-31`; core types atom = 16-31, core = 0-15; L2 atom 4096 KiB, core 2048 KiB.

## Threading

- Variables: `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- BLAS `libscipy_openblas64_-61654e39.so`: config `OpenBLAS 0.3.34.0.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`; kernel `Haswell`; threads 1; parallel model 1.
- OpenMP `libgomp.so.1.0.0`: max threads 1; verified: True.

## Software

- Python 3.12.11 on Linux 7.0.0-31-generic (x86_64).
- Packages: arm-rc-ctrl 0.1.0.dev0, numpy 2.5.2, rclib 0.1.0, scipy 1.18.1, skelarm 0.4.1.
