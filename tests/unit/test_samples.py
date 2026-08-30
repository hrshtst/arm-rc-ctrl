# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M1-002: canonical samples.npz arrays have fixed names, shapes, dtypes, and phase codes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_rc_ctrl.data.arrays import array_digest, load_npz, save_npz
from arm_rc_ctrl.data.samples import (
    ARRAY_NAMES,
    PHASE_CODES,
    PHASE_DWELL,
    PHASE_MOVE,
    PHASE_PRIME,
    SampleSet,
    load_samples,
    save_samples,
)
from arm_rc_ctrl.data.synthetic import synthetic_arrays as make_arrays
from arm_rc_ctrl.data.synthetic import synthetic_samples as make_samples


def test_canonical_names_and_phase_codes() -> None:
    """The array set and the phase encoding are fixed and documented."""
    assert ARRAY_NAMES == ("t", "q", "dq", "ddq", "tip", "dtip", "ddtip", "task_code", "phase")
    assert PHASE_CODES == {"prime": 0, "move": 1, "dwell": 2}
    assert (PHASE_PRIME, PHASE_MOVE, PHASE_DWELL) == (0, 1, 2)


def test_valid_sample_set_reports_dimensions() -> None:
    """Shapes, dtypes, and dimensions are derived from the arrays."""
    samples = make_samples(n=6, dof=2, task_dim=2, code_dim=0)
    assert (samples.n_samples, samples.dof, samples.task_dim, samples.task_code_dim) == (6, 2, 2, 0)
    assert samples.shapes()["task_code"] == (6, 0)
    assert samples.dtypes() == {**dict.fromkeys(ARRAY_NAMES[:-1], "float64"), "phase": "int64"}
    assert list(samples.arrays()) == list(ARRAY_NAMES)
    assert set(samples.digests()) == set(ARRAY_NAMES)
    with_codes = make_samples(n=4, dof=3, task_dim=2, code_dim=2)
    assert with_codes.task_code_dim == 2


def test_arrays_are_immutable_copies() -> None:
    """Stored arrays are read-only copies detached from the inputs."""
    arrays = make_arrays()
    samples = SampleSet.from_arrays(arrays)
    arrays["q"][0, 0] = 99.0
    assert samples.q[0, 0] != 99.0
    with pytest.raises(ValueError, match="read-only"):
        samples.q[0, 0] = 1.0


def test_from_arrays_requires_exact_name_set() -> None:
    """Missing or extra arrays are rejected."""
    arrays = make_arrays()
    del arrays["ddtip"]
    with pytest.raises(ValueError, match="expected arrays"):
        SampleSet.from_arrays(arrays)
    arrays = make_arrays()
    arrays["extra"] = arrays["t"]
    with pytest.raises(ValueError, match="expected arrays"):
        SampleSet.from_arrays(arrays)


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    [
        ("q", np.zeros((6, 2), dtype=np.float32), TypeError, "expected dtype float64, got float32"),
        ("phase", np.zeros(6, dtype=np.float64), TypeError, "expected dtype int64, got float64"),
        ("t", np.zeros((6, 1)), ValueError, "t and phase must be 1-D"),
        ("q", np.zeros((5, 2)), ValueError, r"q must have shape \(6, k\)"),
        ("q", np.zeros(6), ValueError, r"q must have shape \(6, k\)"),
        ("dq", np.zeros((6, 3)), ValueError, "q, dq, ddq must share a joint dimension"),
        ("q", np.zeros((6, 0)), ValueError, "q, dq, ddq must share a joint dimension >= 1"),
        ("dtip", np.zeros((6, 1)), ValueError, "tip, dtip, ddtip must share a task dimension"),
        ("phase", np.array([0, 1, 2, 3, 1, -1], dtype=np.int64), ValueError, r"undocumented codes \[-1, 3\]"),
        ("phase", np.zeros(5, dtype=np.int64), ValueError, r"phase must have shape \(6,\)"),
    ],
)
def test_schema_violations_are_rejected(name: str, value: NDArray[Any], error: type[Exception], message: str) -> None:
    """dtype, dimensionality, consistency, and phase-code violations all fail."""
    arrays = make_arrays()
    arrays[name] = value
    with pytest.raises(error, match=message):
        SampleSet.from_arrays(arrays)


def test_at_least_two_samples_are_required() -> None:
    """One sample cannot define a time step."""
    with pytest.raises(ValueError, match="at least 2 samples, got 1"):
        make_samples(n=1)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """samples.npz reproduces the arrays bitwise and keeps their digests."""
    samples = make_samples()
    path = tmp_path / "samples.npz"
    save_samples(path, samples)
    loaded = load_samples(path)
    for name in ARRAY_NAMES:
        assert np.array_equal(loaded.arrays()[name], samples.arrays()[name])
        assert loaded.arrays()[name].dtype == samples.arrays()[name].dtype
    assert loaded.digests() == samples.digests()


def test_npz_loader_requires_exact_arrays(tmp_path: Path) -> None:
    """An archive with missing or extra arrays is rejected."""
    path = tmp_path / "bad.npz"
    save_npz(path, {"t": np.zeros(2), "extra": np.zeros(2)})
    with pytest.raises(ValueError, match="expected arrays"):
        load_npz(path, ("t",))
    with pytest.raises(ValueError, match="expected arrays"):
        load_samples(path)


def test_array_digest_covers_dtype_shape_and_bytes() -> None:
    """Different dtype, shape, or values change the digest; layout does not."""
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert array_digest(a) == array_digest(np.asfortranarray(a))
    assert array_digest(a) != array_digest(a.reshape(3, 2))
    assert array_digest(a) != array_digest(a.astype(np.int64))
    assert array_digest(a) != array_digest(a + 1)
