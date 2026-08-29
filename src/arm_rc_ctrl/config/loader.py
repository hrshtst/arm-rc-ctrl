# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Map TOML documents onto frozen dataclasses with strict validation.

Rules
-----
- Every key must correspond to a dataclass field; unknown keys are errors.
- Values must have exactly the annotated type. There is no implicit coercion:
  an ``int`` is not accepted for a ``float`` field, a ``bool`` is not accepted
  for an ``int`` field, and non-finite floats are rejected.
- Fields without a default are required.
- ``Path`` fields are read as strings and resolved relative to the directory of
  the configuration file that declared them.
- Supported annotations: ``str``, ``int``, ``float``, ``bool``, ``Path``,
  ``Literal[...]`` of strings/ints, nested dataclasses, ``list[T]``,
  ``tuple[T, ...]``, ``dict[str, T]``, and ``T | None``.

Errors in the *document* raise :class:`ConfigError` with a dotted location such
as ``robot.links[1].mass``. Errors in the *schema* (unsupported annotations,
non-dataclass types) raise :class:`TypeError`, since they are programming
mistakes rather than user input problems.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
import types
import typing
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = ["ConfigError", "from_mapping", "load_config", "to_mapping"]

_ROOT = ""

_SCALARS: dict[object, str] = {str: "string", int: "integer", float: "float", bool: "boolean"}


class ConfigError(ValueError):
    """The configuration document is invalid at ``location``."""

    def __init__(self, location: str, message: str, source: Path | None = None) -> None:
        self.location = location
        self.message = message
        self.source = source
        prefix = f"{source}: " if source is not None else ""
        where = location or "<root>"
        super().__init__(f"{prefix}{where}: {message}")


def load_config[T](path: Path, schema: type[T]) -> T:
    """Read a TOML file and map it onto ``schema``.

    Relative ``Path`` values are resolved against ``path.parent``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ConfigError
        If the document is not valid TOML or does not satisfy ``schema``.
    """
    with path.open("rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            msg = f"invalid TOML: {exc}"
            raise ConfigError(_ROOT, msg, source=path) from exc
    try:
        return from_mapping(data, schema, base_dir=path.parent)
    except ConfigError as exc:
        raise ConfigError(exc.location, exc.message, source=path) from None


def from_mapping[T](data: Mapping[str, object], schema: type[T], *, base_dir: Path | None = None) -> T:
    """Map an already-parsed TOML mapping onto the frozen dataclass ``schema``.

    Parameters
    ----------
    data : Mapping[str, object]
        Parsed document (as returned by :func:`tomllib.load`).
    schema : type[T]
        A dataclass type describing the expected structure.
    base_dir : Path | None, optional
        Directory used to resolve relative ``Path`` values. Without it, a
        relative path is an error.
    """
    if not dataclasses.is_dataclass(schema):
        msg = f"schema must be a dataclass type, got {schema!r}"
        raise TypeError(msg)
    return _convert(data, schema, _ROOT, base_dir)


def to_mapping(config: object) -> dict[str, object]:
    """Convert a dataclass configuration into a plain, serializable mapping.

    ``Path`` values become POSIX strings, tuples become lists, and nested
    dataclasses become nested mappings. The result is suitable for JSON/TOML
    dumps and digests of the resolved configuration.
    """
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        msg = f"expected a dataclass instance, got {config!r}"
        raise TypeError(msg)
    return {field.name: _plain(getattr(config, field.name)) for field in dataclasses.fields(config)}


def _plain(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_mapping(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in cast("list[object] | tuple[object, ...]", value)]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in cast("dict[object, object]", value).items()}
    return value


def _join(location: str, key: str) -> str:
    return f"{location}.{key}" if location else key


def _describe(value: object) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "table"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _convert(value: object, annotation: object, location: str, base_dir: Path | None) -> typing.Any:  # noqa: ANN401
    """Validate ``value`` against ``annotation`` and return the typed result."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if annotation in _SCALARS:
        return _scalar(value, cast("type[object]", annotation), location)
    if annotation is Path:
        return _path(value, location, base_dir)
    if origin is Literal:
        if value in args and not isinstance(value, bool):
            return value
        choices = ", ".join(repr(a) for a in args)
        raise ConfigError(location, f"expected one of {choices}, got {value!r}")
    if origin in (types.UnionType, typing.Union):
        return _optional(value, args, location, base_dir)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _dataclass(value, annotation, location, base_dir)
    if origin is list or origin is tuple:
        return _sequence(value, origin, args, location, base_dir)
    if origin is dict:
        return _table(value, args, location, base_dir)
    msg = f"unsupported configuration annotation at {location or '<root>'}: {annotation!r}"
    raise TypeError(msg)


def _scalar(value: object, expected: type[object], location: str) -> object:
    # bool is a subclass of int; keep the two strictly separate.
    if isinstance(value, bool) and expected is not bool:
        raise ConfigError(location, f"expected {_SCALARS[expected]}, got boolean")
    if not isinstance(value, expected):
        hint = " (write 1.0, not 1)" if expected is float and isinstance(value, int) else ""
        raise ConfigError(location, f"expected {_SCALARS[expected]}, got {_describe(value)}{hint}")
    if expected is float and not math.isfinite(cast("float", value)):
        raise ConfigError(location, f"expected a finite float, got {value!r}")
    return value


def _path(value: object, location: str, base_dir: Path | None) -> Path:
    if not isinstance(value, str):
        raise ConfigError(location, f"expected path string, got {_describe(value)}")
    if not value:
        raise ConfigError(location, "expected a non-empty path")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if base_dir is None:
        raise ConfigError(location, f"relative path {value!r} cannot be resolved without a base directory")
    return (base_dir / path).resolve()


def _optional(value: object, args: tuple[object, ...], location: str, base_dir: Path | None) -> object:
    members = [a for a in args if a is not type(None)]
    if len(members) != len(args) - 1 or len(members) != 1:
        msg = f"only `T | None` unions are supported at {location or '<root>'}, got {args!r}"
        raise TypeError(msg)
    if value is None:
        return None
    return _convert(value, members[0], location, base_dir)


def _dataclass(value: object, schema: type[DataclassInstance], location: str, base_dir: Path | None) -> object:
    if not isinstance(value, Mapping):
        raise ConfigError(location, f"expected table, got {_describe(value)}")
    data = cast("Mapping[str, object]", value)
    hints = cast("dict[str, object]", typing.get_type_hints(schema))
    fields = {f.name: f for f in dataclasses.fields(schema)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        names = ", ".join(repr(k) for k in unknown)
        raise ConfigError(location, f"unknown key(s) {names}; allowed: {', '.join(sorted(fields))}")
    kwargs: dict[str, object] = {}
    for name, field in fields.items():
        here = _join(location, name)
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], here, base_dir)
        elif field.default is not dataclasses.MISSING:
            kwargs[name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs[name] = field.default_factory()
        else:
            raise ConfigError(here, "required key is missing")
    return cast("Callable[..., object]", schema)(**kwargs)


def _sequence(
    value: object, origin: object, args: tuple[object, ...], location: str, base_dir: Path | None
) -> list[object] | tuple[object, ...]:
    if origin is tuple:
        try:
            element, ellipsis = args
        except ValueError:
            ellipsis = None
            element = None
        if ellipsis is not Ellipsis:
            msg = f"only homogeneous `tuple[T, ...]` is supported at {location or '<root>'}"
            raise TypeError(msg)
    else:
        try:
            (element,) = args
        except ValueError:
            msg = f"`list` requires exactly one type argument at {location or '<root>'}"
            raise TypeError(msg) from None
    if not isinstance(value, list):
        raise ConfigError(location, f"expected array, got {_describe(value)}")
    elements = cast("list[object]", value)
    items = [_convert(item, element, f"{location}[{i}]", base_dir) for i, item in enumerate(elements)]
    return tuple(items) if origin is tuple else items


def _table(value: object, args: tuple[object, ...], location: str, base_dir: Path | None) -> dict[str, object]:
    try:
        key_type, value_type = args
    except ValueError:
        key_type = value_type = None
    if key_type is not str:
        msg = f"only `dict[str, T]` tables are supported at {location or '<root>'}"
        raise TypeError(msg)
    if not isinstance(value, Mapping):
        raise ConfigError(location, f"expected table, got {_describe(value)}")
    data = cast("Mapping[str, object]", value)
    return {key: _convert(item, value_type, _join(location, key), base_dir) for key, item in data.items()}
