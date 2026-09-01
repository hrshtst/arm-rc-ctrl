# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Representative-scenario selection and joint-trajectory plotting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arm_rc_ctrl.experiments.robustness import load_suite
from arm_rc_ctrl.experiments.trajectory_plots import (
    CURVE_ORDER,
    CURVE_STYLES,
    plot_joint_series,
    select_representatives,
)
from arm_rc_ctrl.repo import repository_root

if TYPE_CHECKING:
    from numpy.typing import NDArray

SUITE = repository_root() / "docs" / "experiments" / "task_1a" / "robustness_confirmatory_v2_recipe_v4.json"


def test_representative_scenarios_are_selected_by_primary_arm_class_median() -> None:
    """The deterministic median-distance rule selects the reviewed scenario from every class."""
    selected = select_representatives(load_suite(SUITE))
    assert {kind: run.scenario_id for kind, run in selected.items()} == {
        "nominal": "nominal",
        "posture_small": "posture-small-20260903-03",
        "posture_large": "posture-large-20260904-03",
        "force": "force-12N-270deg",
        "combined": "combined-20260901-03-270deg",
    }


def test_joint_series_plot_writes_two_joint_png_and_refuses_overwrite(tmp_path: Path) -> None:
    """The pure plotter writes four ordered series and does not silently replace output."""
    t = np.linspace(0.0, 5.0, 11, dtype=np.float64)
    reference = np.column_stack((0.1 * t, -0.05 * t))
    replay_actual = reference + 0.001
    rc_output = reference - 0.001
    rc_actual = reference - 0.002
    assert CURVE_ORDER == ("reference", "replay_actual", "rc_output", "rc_actual")
    assert CURVE_STYLES["rc_output"] == ("tab:green", "--", 1.8, "RC output")
    out = plot_joint_series(
        t,
        reference,
        replay_actual,
        rc_output,
        rc_actual,
        tmp_path / "plot.png",
        title="fixture",
    )
    assert out.is_file()
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plot_joint_series(t, reference, replay_actual, rc_output, rc_actual, out, title="fixture")


@pytest.mark.parametrize(
    ("t", "reference", "match"),
    [
        (np.array([0.0, 0.0]), np.zeros((2, 2)), "strictly increasing"),
        (np.array([0.0, 1.0]), np.zeros((3, 2)), "reference must have shape"),
        (np.array([0.0, 1.0]), np.full((2, 2), np.nan), "reference must be finite"),
    ],
)
def test_joint_series_plot_rejects_invalid_inputs(
    t: NDArray[np.float64],
    reference: NDArray[np.float64],
    match: str,
    tmp_path: Path,
) -> None:
    """Invalid clocks, shapes, and values fail before a plot is written."""
    with pytest.raises(ValueError, match=match):
        plot_joint_series(
            t,
            reference,
            np.zeros_like(reference),
            np.zeros_like(reference),
            np.zeros_like(reference),
            tmp_path / "bad.png",
            title="bad",
        )
