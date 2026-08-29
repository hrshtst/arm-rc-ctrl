# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M0-010: typed TOML loading rejects unknown keys, wrong types, and resolves relative paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from arm_rc_ctrl.config import ConfigError, from_mapping, load_config, to_mapping


@dataclass(frozen=True)
class Link:
    """Link parameters."""

    length: float
    mass: float
    name: str = "link"


@dataclass(frozen=True)
class Limits:
    """Joint limits."""

    torque: tuple[float, ...]
    velocity: tuple[float, ...] = ()


@dataclass(frozen=True)
class Robot:
    """Robot description."""

    dof: int
    links: tuple[Link, ...]
    limits: Limits
    gravity: float | None = None
    integrator: Literal["euler", "rk4"] = "rk4"
    enforce_limits: bool = True
    tags: list[str] = field(default_factory=list)
    gains: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    """Top-level experiment configuration."""

    robot: Robot
    demo: Path
    output_dir: Path
    seed: int = 0


VALID = """
seed = 7
demo = "demos/reach.sklog.npz"
output_dir = "/abs/out"

[robot]
dof = 2
integrator = "euler"
gravity = 9.81
tags = ["planar", "test"]

[robot.limits]
torque = [1.0, 2.0]

[robot.gains]
kp = 10.0
kd = 1.0

[[robot.links]]
length = 0.3
mass = 1.0

[[robot.links]]
length = 0.25
mass = 0.5
name = "forearm"
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write the valid document below a ``configs/`` directory and return its path."""
    path = tmp_path / "configs" / "exp.toml"
    path.parent.mkdir()
    path.write_text(VALID)
    return path


def test_valid_document_maps_to_typed_dataclasses(config_file: Path) -> None:
    """Scalars, nested tables, arrays of tables, dicts, literals, optionals, and defaults all load."""
    cfg = load_config(config_file, Experiment)
    assert cfg.seed == 7
    assert cfg.robot.dof == 2
    assert cfg.robot.integrator == "euler"
    assert cfg.robot.gravity == 9.81
    assert cfg.robot.enforce_limits is True
    assert cfg.robot.tags == ["planar", "test"]
    assert cfg.robot.gains == {"kp": 10.0, "kd": 1.0}
    assert cfg.robot.limits == Limits(torque=(1.0, 2.0), velocity=())
    assert cfg.robot.links == (Link(0.3, 1.0), Link(0.25, 0.5, "forearm"))


def test_relative_paths_resolve_against_config_directory(config_file: Path) -> None:
    """Relative Path fields resolve relative to the file; absolute paths are kept."""
    cfg = load_config(config_file, Experiment)
    assert cfg.demo == (config_file.parent / "demos" / "reach.sklog.npz").resolve()
    assert cfg.demo.is_absolute()
    assert cfg.output_dir == Path("/abs/out")


def test_relative_path_without_base_dir_is_rejected() -> None:
    """from_mapping without base_dir cannot resolve a relative path."""
    with pytest.raises(ConfigError, match=r"demo: relative path 'x\.npz' cannot be resolved"):
        from_mapping({"demo": "x.npz"}, _PathOnly)


@dataclass(frozen=True)
class _PathOnly:
    """Schema with a single Path field."""

    demo: Path


def test_missing_required_key_reports_location(config_file: Path) -> None:
    """A missing required field is reported with its dotted location and the file."""
    config_file.write_text(VALID.replace("dof = 2\n", ""))
    with pytest.raises(ConfigError, match=r"exp\.toml: robot\.dof: required key is missing"):
        load_config(config_file, Experiment)


@pytest.mark.parametrize(
    ("anchor", "snippet", "location", "unknown"),
    [
        ("seed = 7\n", "colour = 'red'\n", "<root>", "'colour'"),
        ("[robot]\n", "mas = 1\n", "robot", "'mas'"),
        ("[robot.limits]\n", "torqe = [1.0]\n", "robot.limits", "'torqe'"),
    ],
)
def test_unknown_keys_are_rejected(config_file: Path, anchor: str, snippet: str, location: str, unknown: str) -> None:
    """Unknown keys at any depth fail with the key name and location."""
    config_file.write_text(VALID.replace(anchor, anchor + snippet, 1))
    with pytest.raises(ConfigError, match=rf"{location}: unknown key\(s\) {unknown}"):
        load_config(config_file, Experiment)


def test_unknown_key_inside_array_of_tables(config_file: Path) -> None:
    """Element locations use bracket indices."""
    config_file.write_text(VALID + "\n[[robot.links]]\nlength = 0.1\nmass = 0.1\nmaterial = 'steel'\n")
    with pytest.raises(ConfigError, match=r"robot\.links\[2\]: unknown key\(s\) 'material'"):
        load_config(config_file, Experiment)


@pytest.mark.parametrize(
    ("old", "new", "pattern"),
    [
        ("seed = 7", "seed = 7.0", r"seed: expected integer, got float"),
        ("seed = 7", "seed = true", r"seed: expected integer, got boolean"),
        ("seed = 7", 'seed = "7"', r"seed: expected integer, got string"),
        ("gravity = 9.81", "gravity = 9", r"robot\.gravity: expected float, got integer \(write 1\.0, not 1\)"),
        ("gravity = 9.81", "gravity = nan", r"robot\.gravity: expected a finite float, got nan"),
        ("gravity = 9.81", "gravity = inf", r"robot\.gravity: expected a finite float, got inf"),
        ("dof = 2", "dof = [2]", r"robot\.dof: expected integer, got array"),
        (
            'integrator = "euler"',
            'integrator = "verlet"',
            r"robot\.integrator: expected one of 'euler', 'rk4', got 'verlet",
        ),
        ("torque = [1.0, 2.0]", "torque = [1.0, 2]", r"robot\.limits\.torque\[1\]: expected float, got integer"),
        ("torque = [1.0, 2.0]", "torque = 1.0", r"robot\.limits\.torque: expected array, got float"),
        ('tags = ["planar", "test"]', "tags = [1]", r"robot\.tags\[0\]: expected string, got integer"),
        ("kd = 1.0", "kd = false", r"robot\.gains\.kd: expected float, got boolean"),
        ('demo = "demos/reach.sklog.npz"', "demo = 3", r"demo: expected path string, got integer"),
        ('demo = "demos/reach.sklog.npz"', 'demo = ""', r"demo: expected a non-empty path"),
    ],
)
def test_type_invalid_values_are_rejected(config_file: Path, old: str, new: str, pattern: str) -> None:
    """Exact types are required; no coercion between int/float/bool/str or scalar/table/array."""
    config_file.write_text(VALID.replace(old, new, 1))
    with pytest.raises(ConfigError, match=pattern):
        load_config(config_file, Experiment)


def test_table_given_for_scalar_and_scalar_for_table() -> None:
    """Structural mismatches are reported as such."""
    robot: dict[str, object] = {"dof": 1, "links": [], "limits": {"torque": []}}
    with pytest.raises(ConfigError, match=r"robot: expected table, got integer"):
        from_mapping({"robot": 1, "demo": "/d", "output_dir": "/o"}, Experiment)
    with pytest.raises(ConfigError, match=r"seed: expected integer, got table"):
        from_mapping({"seed": {}, "robot": robot, "demo": "/d", "output_dir": "/o"}, Experiment)
    with pytest.raises(ConfigError, match=r"robot\.limits: expected table, got array"):
        from_mapping({"robot": {**robot, "limits": [1.0]}, "demo": "/d", "output_dir": "/o"}, Experiment)


def test_invalid_toml_and_missing_file(tmp_path: Path) -> None:
    """Syntax errors are ConfigErrors naming the file; missing files raise FileNotFoundError."""
    bad = tmp_path / "bad.toml"
    bad.write_text("seed = \n")
    with pytest.raises(ConfigError, match=r"bad\.toml: <root>: invalid TOML"):
        load_config(bad, Experiment)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.toml", Experiment)


def test_to_mapping_round_trips_resolved_config(config_file: Path) -> None:
    """The resolved mapping is plain data and maps back to an equal configuration."""
    cfg = load_config(config_file, Experiment)
    mapping = to_mapping(cfg)
    assert mapping["demo"] == cfg.demo.as_posix()
    assert mapping["robot"] == {
        "dof": 2,
        "links": [
            {"length": 0.3, "mass": 1.0, "name": "link"},
            {"length": 0.25, "mass": 0.5, "name": "forearm"},
        ],
        "limits": {"torque": [1.0, 2.0], "velocity": []},
        "gravity": 9.81,
        "integrator": "euler",
        "enforce_limits": True,
        "tags": ["planar", "test"],
        "gains": {"kp": 10.0, "kd": 1.0},
    }
    assert from_mapping(mapping, Experiment) == cfg


def test_schema_errors_are_type_errors() -> None:
    """Unsupported annotations and non-dataclass schemas are programming errors, not ConfigErrors."""

    @dataclass(frozen=True)
    class BadSet:
        values: set[int]

    @dataclass(frozen=True)
    class BadUnion:
        value: int | str

    @dataclass(frozen=True)
    class BadTuple:
        value: tuple[int, str]

    @dataclass(frozen=True)
    class BadDict:
        value: dict[int, str]

    with pytest.raises(TypeError, match="unsupported configuration annotation at values"):
        from_mapping({"values": [1]}, BadSet)
    with pytest.raises(TypeError, match="only `T \\| None` unions are supported at value"):
        from_mapping({"value": 1}, BadUnion)
    with pytest.raises(TypeError, match="only homogeneous `tuple\\[T, \\.\\.\\.\\]` is supported at value"):
        from_mapping({"value": [1, "a"]}, BadTuple)
    with pytest.raises(TypeError, match="only `dict\\[str, T\\]` tables are supported at value"):
        from_mapping({"value": {}}, BadDict)
    with pytest.raises(TypeError, match="schema must be a dataclass type"):
        from_mapping({}, int)
    with pytest.raises(TypeError, match="expected a dataclass instance"):
        to_mapping(Experiment)


def test_optional_field_accepts_missing_and_typed_value() -> None:
    """``T | None`` fields default to None when absent and validate the value when present."""
    base: dict[str, object] = {"dof": 1, "links": [], "limits": {"torque": []}}
    assert from_mapping(base, Robot).gravity is None
    assert from_mapping({**base, "gravity": 1.5}, Robot).gravity == 1.5
    with pytest.raises(ConfigError, match=r"gravity: expected float, got string"):
        from_mapping({**base, "gravity": "1.5"}, Robot)
