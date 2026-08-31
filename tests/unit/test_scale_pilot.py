# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M2-018: input-scale pilot protocol, cell feasibility, and region-based anchor selection."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from arm_rc_ctrl.experiments.closed_loop import EstimatorSpec
from arm_rc_ctrl.experiments.scale_pilot import (
    Cell,
    ScaleGrid,
    ScalePilotProtocol,
    SelectionRules,
    Variant,
    Variants,
    load_protocol,
    select_anchor,
    summarize_cells,
)
from arm_rc_ctrl.repo import repository_root

PROTOCOL = repository_root() / "configs" / "studies" / "input_scale_pilot_1a.toml"


def _protocol(q: tuple[float, ...] = (0.2, 0.3, 0.5), dq: tuple[float, ...] = (2.0, 4.0, 8.0)) -> ScalePilotProtocol:
    return ScalePilotProtocol(
        "unit",
        repository_root() / "configs" / "tasks" / "task_1a.toml",
        repository_root() / "configs" / "models" / "esn_task_1a.toml",
        ("pd_v2", "computed_torque"),
        ScaleGrid(q, dq),
        Variants((1e-2,), (31,)),
        EstimatorSpec(20.0, 20.0),
        SelectionRules(0.75),
    )


def _variant(
    cell: Cell, tracker: str, *, success: bool, rmse: float = 1e-3, alpha: float = 1e-2, seed: int = 31
) -> Variant:
    return Variant(
        cell=cell,
        ridge_alpha=alpha,
        reservoir_seed=seed,
        tracker=tracker,
        termination="completed" if success else "limit_violation",
        success=success,
        criteria={"completed": success, "dwell_in_tolerance": success, "dwell_stationary": success},
        move_joint_rmse=rmse if success else None,
        fit_rmse=1e-4,
        boundary_jump=1e-3,
        peak_velocity=1.0,
        saturation_fraction=0.0,
    )


def test_committed_protocol_is_the_development_sweep() -> None:
    """The committed protocol sweeps around the anchor with both frozen trackers and development settings only."""
    protocol = load_protocol(PROTOCOL)
    assert protocol.trackers == ("pd_v2", "computed_torque")
    assert protocol.grid.q_scales == (0.1, 0.15, 0.2, 0.3, 0.5)
    assert protocol.grid.dq_scales == (4.0, 6.0, 8.0, 12.0, 16.0)
    assert protocol.variants.reservoir_seeds == (31, 32)
    assert protocol.variants.ridge_alphas == (1e-2, 3e-2, 1e-1, 3e-1)
    assert protocol.selection.min_feasible_fraction == 0.75
    assert protocol.model.name == "esn_task_1a.toml"


def test_protocol_validation() -> None:
    """Axes are increasing positive values; variants and trackers are distinct; the threshold lies in (0, 1]."""
    with pytest.raises(ValueError, match="grid axes must be increasing"):
        ScaleGrid((0.5, 0.2), (2.0,))
    with pytest.raises(ValueError, match="distinct positive values"):
        ScaleGrid((0.2, 0.2), (2.0,))
    with pytest.raises(ValueError, match="distinct positive values"):
        ScaleGrid((0.2,), (0.0,))
    with pytest.raises(ValueError, match="reservoir_seeds must be non-empty and distinct"):
        Variants((1e-2,), (31, 31))
    with pytest.raises(ValueError, match="must lie in"):
        SelectionRules(0.0)
    with pytest.raises(ValueError, match="distinct baselines"):
        dataclasses.replace(_protocol(), trackers=("pd_v2", "pd_v2"))
    with pytest.raises(ValueError, match="name must not be empty"):
        dataclasses.replace(_protocol(), name=" ")


def test_cells_are_scored_by_feasible_fraction() -> None:
    """A cell's fraction counts successful variants; the median RMSE covers feasible variants only."""
    protocol = _protocol(q=(0.2, 0.3), dq=(2.0,))
    variants = [
        _variant(Cell(0.2, 2.0), "pd_v2", success=True, rmse=2e-3),
        _variant(Cell(0.2, 2.0), "computed_torque", success=False),
        _variant(Cell(0.3, 2.0), "pd_v2", success=True, rmse=1e-3),
        _variant(Cell(0.3, 2.0), "computed_torque", success=True, rmse=3e-3),
    ]
    cells = summarize_cells(protocol, variants)
    assert [c.feasible_fraction for c in cells] == [0.5, 1.0]
    assert [c.feasible for c in cells] == [False, True]
    assert cells[0].median_move_joint_rmse == 2e-3
    assert cells[1].median_move_joint_rmse == 2e-3
    with pytest.raises(ValueError, match="no variants for cell"):
        summarize_cells(protocol, variants[:2])


def test_anchor_needs_feasible_neighbours() -> None:
    """The best cell is rejected when a grid neighbour fails; a region-interior cell wins instead."""
    protocol = _protocol()
    grid = [Cell(q, dq) for q in protocol.grid.q_scales for dq in protocol.grid.dq_scales]

    # every cell feasible except (0.2, 2.0) and (0.5, 8.0); the corner (0.3, 4.0) is interior and fully feasible
    def rmse_of(cell: Cell) -> float:
        return 1e-3 if cell == Cell(0.3, 4.0) else 2e-3

    variants = [
        _variant(cell, tracker, success=cell not in (Cell(0.2, 2.0), Cell(0.5, 8.0)), rmse=rmse_of(cell))
        for cell in grid
        for tracker in protocol.trackers
    ]
    cells = summarize_cells(protocol, variants)
    selection = select_anchor(protocol, cells)
    assert (selection.q_scale, selection.dq_scale) == (0.3, 4.0)
    assert Cell(0.2, 2.0) not in selection.region
    assert len(selection.region) == 7
    # a lone feasible cell surrounded by failures is never the anchor
    lonely = [_variant(cell, tracker, success=cell == Cell(0.3, 4.0)) for cell in grid for tracker in protocol.trackers]
    with pytest.raises(ValueError, match="no feasible interior cell has four feasible grid neighbours"):
        select_anchor(protocol, summarize_cells(protocol, lonely))
    # a fully feasible boundary column never yields an anchor: boundary cells have fewer than four neighbours
    column = [_variant(cell, tracker, success=cell.q_scale == 0.2) for cell in grid for tracker in protocol.trackers]
    with pytest.raises(ValueError, match="no feasible interior cell"):
        select_anchor(protocol, summarize_cells(protocol, column))
    # the highest fraction wins over the lowest RMSE when neighbours are feasible
    mixed = [
        _variant(
            cell, tracker, success=not (cell == Cell(0.3, 4.0) and tracker == "computed_torque"), rmse=rmse_of(cell)
        )
        for cell in grid
        for tracker in protocol.trackers
    ]
    with pytest.raises(ValueError, match="no feasible interior cell"):
        select_anchor(protocol, summarize_cells(protocol, mixed))  # the only interior cell is now below threshold


def test_protocol_file_paths_resolve_next_to_the_file(tmp_path: Path) -> None:
    """Relative scenario/model paths resolve relative to the protocol file."""
    file = tmp_path / "pilot.toml"
    file.write_text(PROTOCOL.read_text().replace('scenario = "../tasks/task_1a.toml"', 'scenario = "task.toml"'))
    protocol = load_protocol(file)
    assert protocol.scenario == tmp_path / "task.toml"
