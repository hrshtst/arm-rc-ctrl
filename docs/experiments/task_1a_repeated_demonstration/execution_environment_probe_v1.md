<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Execution-environment probe of the DOC-005 discrepancy (2026-09-09)

Retained commands and outputs behind clarification C10 (plan section 12).
They establish core-affinity-dependent numerical reproducibility on the
owner's machine; OpenBLAS dispatch is the leading explanation and is not yet
confirmed by backend diagnostics. No test, tolerance, or evidence was changed.

## Machine

- CPU: Intel Core i9-14900KF, 32 logical CPUs, 24 cores, hybrid topology
  (`/sys/devices/cpu_core/cpus` = `0-15`, `/sys/devices/cpu_atom/cpus` = `16-31`).
- OS: Linux 7.0.0-31-generic, x86_64, glibc 2.43; CPython 3.12.11.
- numpy BLAS: `scipy-openblas64` 0.3.34.0.0, configuration
  `OpenBLAS 0.3.34.0.0 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`;
  `threadpoolctl` is not installed, so effective thread counts were not read.
- Environment of every run below: `OMP_NUM_THREADS=1`, `QT_QPA_PLATFORM=offscreen`;
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` unset.

## Subject

`tests/integration/test_reproduce_1a.py::test_from_evidence_reproduces_the_nominal_class_end_to_end`:
the original task 1-a `--from-evidence` reproduction of the nominal
confirmatory class (evidence commit `19b9af3f5540`, reproduction checkout
`c9cb57403c1e`, recipe `model-20260831-1b9477aaa246`, dataset
`processed-20260830-feaf73e6663c`) compared against its exact-zero tolerance.
The inner reproduction runs in a fresh subprocess inside a temporary worktree.

## Observations (UTC)

| Run | Command | Outcome |
| --- | --- | --- |
| Full gate 1, unpinned, 06:16–06:29 | `uv run --locked nox` | test passed (2413 passed overall) |
| Full gate 2, unpinned, 06:31–06:43 | `uv run --locked nox` | test passed (2415 passed overall) |
| Full gate 3, unpinned, 07:19–07:31 | `uv run --locked nox` | **test failed**: `largest metric deviation 1.608e-10 exceeds the tolerance 0.000e+00`, `max_deviation` 1.608386757112612e-10; all other checks of the reproduction (environment, storage, records, payloads, data, model) passed |
| Isolated rerun, 07:33:05 | `uv run --locked pytest <test> -q -p no:cacheprovider` | passed (39.70 s) |
| Isolated rerun, 07:33:45 | same | passed (46.61 s) |
| Hash seeds, 07:35–07:38 | `PYTHONHASHSEED={0,1,2,3} uv run --locked pytest <test> -q -p no:cacheprovider` | passed four times (42.69 s, 42.38 s, 88.39 s, 40.67 s) |
| E-cores, 07:40:06 | `taskset -c 16-31 uv run --locked pytest <test> -q -p no:cacheprovider` | **failed**, `max_deviation` 1.608386757112612e-10 (47.81 s) |
| P-cores, 07:40:55 | `taskset -c 0-15 uv run --locked pytest <test> -q -p no:cacheprovider` | passed (31.31 s) |
| Full gate 4, P-cores, 07:42–07:53 | `taskset -c 0-15 uv run --locked nox` and `nox -s pre_commit` | all sessions green (2417 passed, 1 skipped, 1 xfailed; coverage 92 %) |

The deviation is the same value the DOC-005 review recorded during its
unrestricted rerun. It appears whenever the reproduction runs on the E-cores
and did not appear in eight runs on the P-cores or unpinned in isolation, so
the intermittent failures of the unpinned full suite are explained by the
scheduler placing the reproduction subprocess on an E-core under load.

## What this does and does not establish

- Established: the same binaries, inputs, and single-thread settings produce
  results that differ at the 1e-10 level between the two core types of this
  CPU, amplified by the closed-loop simulation.
- Leading explanation, unconfirmed: OpenBLAS `DYNAMIC_ARCH` selects
  kernels or blocking parameters from the core it starts on. Backend
  diagnostics (loaded kernel, effective threads) are part of the M3REP-009
  execution record.
- Not established: bitwise reproducibility across machines, or that pinning
  removes the discrepancy anywhere but on this machine.

Grep of the numerical paths found no set-ordered floating-point reductions,
consistent with the hash-seed runs passing.
