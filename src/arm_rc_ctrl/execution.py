# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Canonical execution environment: CPU affinity, numerical threading, and a versioned execution record (M3REP-009).

Results on a hybrid CPU depend on the core type a process starts on
(repetition plan section 12, clarification C10; the retained probe is
``docs/experiments/task_1a_repeated_demonstration/execution_environment_probe_v1.md``).
This module makes the environment explicit instead of implicit:

- a launcher that resolves an affinity policy from ``sysfs`` at run time (the
  CPU numbers are never hard-coded), pins the process, sets the single-thread
  variables, and ``exec``s the target command, so the target interpreter starts
  pinned and every child inherits the restriction;
- a guard that new commands and their workers call before importing numerical
  libraries, which fails unless the declared policy, the effective affinity,
  and the thread variables agree;
- a versioned :class:`ExecutionRecord` of the actual environment (requested and
  effective affinity, CPU model and core-type map, numerical-library versions,
  the loaded BLAS build and kernel name, effective thread counts, thread
  variables, launch command) whose :attr:`ExecutionRecord.identity` digest
  binds into new evidence and cache identities. Legacy provenance records,
  recipe hashes, and frozen evidence are untouched.

Command line::

    python -m arm_rc_ctrl.execution run --policy p-cores -- uv run --locked nox
    python -m arm_rc_ctrl.execution run --cpus 0-15 -- python -m arm_rc_ctrl.experiments.<runner> ...
    python -m arm_rc_ctrl.execution record --output <file>.json [--markdown <file>.md] [--allow-unmanaged]
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Protocol, cast

from arm_rc_ctrl import __version__, dependencies
from arm_rc_ctrl.config import from_mapping, to_mapping
from arm_rc_ctrl.provenance import canonical_json, command_line, sha256_bytes
from arm_rc_ctrl.validation import validate_utc_timestamp

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping, Sequence

__all__ = [
    "CPUS_VARIABLE",
    "EXECUTION_SCHEMA_VERSION",
    "POLICIES",
    "POLICY_VARIABLE",
    "THREAD_VARIABLES",
    "AffinityRequest",
    "BlasInfo",
    "CoreTopology",
    "ExecutionEnvironmentError",
    "ExecutionRecord",
    "OpenMpInfo",
    "Probes",
    "apply_affinity",
    "check_same_environment",
    "collect_execution",
    "execution_to_json",
    "format_cpu_list",
    "launch",
    "load_execution",
    "main",
    "parse_cpu_list",
    "probe_blas",
    "probe_openmp",
    "render_execution_markdown",
    "require_canonical",
]

EXECUTION_SCHEMA_VERSION: Final = 1
THREAD_VARIABLES: Final = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""Set to ``1`` before any numerical library is imported (C10, condition 2)."""
POLICY_VARIABLE: Final = "ARM_RC_CTRL_EXECUTION_POLICY"
CPUS_VARIABLE: Final = "ARM_RC_CTRL_EXECUTION_CPUS"
SYSFS: Final = Path("/sys")
CPUINFO: Final = Path("/proc/cpuinfo")
MAPS: Final = Path("/proc/self/maps")
type AffinityPolicy = Literal["p-cores", "all", "explicit"]
LAUNCH_POLICIES: Final = ("p-cores", "all", "explicit")
"""Policies a launcher can declare; ``p-cores`` resolves from sysfs on a hybrid CPU and refuses elsewhere."""
UNMANAGED: Final = "unmanaged"
"""Record policy of a process that was not launched under a declared affinity (never canonical)."""
POLICIES: Final = (*LAUNCH_POLICIES, UNMANAGED)
ROLES: Final = ("main", "worker")
_SHORT: Final = 12
_SYMBOL_PREFIXES: Final = ("", "scipy_")
_SYMBOL_SUFFIXES: Final = ("", "64_", "_64")


class ExecutionEnvironmentError(RuntimeError):
    """The process does not run in the declared canonical execution environment."""


class _CFunction(Protocol):
    """A foreign function whose return type is set before it is called."""

    restype: object

    def __call__(self) -> object: ...


def parse_cpu_list(text: str) -> tuple[int, ...]:
    """Parse a kernel-style CPU list (``0-15,20``) into sorted distinct CPU numbers."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        low, dash, high = part.partition("-")
        try:
            start = int(low)
            end = int(high) if dash else start
        except ValueError:
            msg = f"invalid CPU list {text!r}"
            raise ValueError(msg) from None
        if start < 0 or end < start:
            msg = f"invalid CPU range {part!r} in {text!r}"
            raise ValueError(msg)
        cpus.update(range(start, end + 1))
    if not cpus:
        msg = f"CPU list {text!r} names no CPU"
        raise ValueError(msg)
    return tuple(sorted(cpus))


def format_cpu_list(cpus: Sequence[int]) -> str:
    """Render sorted CPU numbers as a kernel-style list (``0-15,20``)."""
    ordered = sorted(set(cpus))
    if not ordered:
        msg = "an empty CPU set cannot be formatted"
        raise ValueError(msg)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = cpu
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def _sorted_distinct(name: str, cpus: Sequence[int]) -> None:
    if not cpus or list(cpus) != sorted(set(cpus)) or cpus[0] < 0:
        msg = f"{name} must be non-empty, sorted, distinct, non-negative CPU numbers, got {list(cpus)}"
        raise ValueError(msg)


@dataclass(frozen=True)
class CoreTopology:
    """The machine's logical CPUs, their hardware core types, and the caches that distinguish them."""

    model: str
    online: tuple[int, ...]
    core_types: dict[str, tuple[int, ...]]
    """Logical CPUs per core type as sysfs names them (``core`` and ``atom`` on Intel hybrid parts); empty
    when the kernel exposes no core types."""
    core_ids: dict[str, int]
    """Physical core id of each online logical CPU whose topology is exposed (keys are CPU numbers as strings)."""
    l2_cache_kib: dict[str, int]
    """Level-2 cache size per core type (``all`` without core types), read from the first CPU of the type."""

    def __post_init__(self) -> None:
        """Every core type is a non-empty, disjoint subset of the online CPUs."""
        if not self.model.strip():
            msg = "topology.model must not be empty"
            raise ValueError(msg)
        _sorted_distinct("topology.online", self.online)
        seen: set[int] = set()
        online = set(self.online)
        for name, cpus in self.core_types.items():
            _sorted_distinct(f"topology.core_types[{name}]", cpus)
            if not set(cpus) <= online or seen & set(cpus):
                msg = f"core type {name!r} must be a disjoint subset of the online CPUs"
                raise ValueError(msg)
            seen |= set(cpus)
        unknown = sorted(set(self.core_ids) - {str(cpu) for cpu in self.online})
        if unknown:
            msg = f"core_ids name CPUs that are not online: {unknown}"
            raise ValueError(msg)
        allowed = set(self.core_types) or {"all"}
        if not set(self.l2_cache_kib) <= allowed:
            msg = f"l2_cache_kib keys must be core types ({sorted(allowed)}), got {sorted(self.l2_cache_kib)}"
            raise ValueError(msg)

    @property
    def hybrid(self) -> bool:
        """Whether the kernel exposes more than one core type."""
        return len(self.core_types) > 1

    def performance_cores(self) -> tuple[int, ...]:
        """The performance cores of a hybrid CPU (sysfs ``cpu_core``); refuses machines without core types."""
        cpus = self.core_types.get("core")
        if not self.hybrid or cpus is None:
            msg = (
                "this machine exposes no hybrid core types in sysfs, so 'p-cores' is undefined here; "
                "use --policy all or an explicit --cpus list"
            )
            raise ExecutionEnvironmentError(msg)
        return cpus

    def core_type_of(self, cpu: int) -> str:
        """The core type holding ``cpu`` (``all`` without core types)."""
        for name, cpus in self.core_types.items():
            if cpu in cpus:
                return name
        return "all"

    @classmethod
    def from_sysfs(cls, sysfs: Path = SYSFS, cpuinfo: Path = CPUINFO) -> CoreTopology:
        """Read the topology from ``sysfs`` and ``/proc/cpuinfo`` (Linux)."""
        cpu_root = sysfs / "devices" / "system" / "cpu"
        online = parse_cpu_list(_read(cpu_root / "online"))
        core_types: dict[str, tuple[int, ...]] = {}
        for entry in sorted((sysfs / "devices").glob("cpu_*")):
            cpus_file = entry / "cpus"
            if cpus_file.is_file():
                text = cpus_file.read_text(encoding="utf-8").strip()
                if text:
                    core_types[entry.name.removeprefix("cpu_")] = parse_cpu_list(text)
        core_ids: dict[str, int] = {}
        for cpu in online:
            core_id = cpu_root / f"cpu{cpu}" / "topology" / "core_id"
            if core_id.is_file():
                core_ids[str(cpu)] = int(_read(core_id))
        groups = core_types or {"all": online}
        l2: dict[str, int] = {}
        for name, cpus in groups.items():
            size = cpu_root / f"cpu{cpus[0]}" / "cache" / "index2" / "size"
            if size.is_file():
                l2[name] = _kib(_read(size))
        return cls(model=_cpu_model(cpuinfo), online=online, core_types=core_types, core_ids=core_ids, l2_cache_kib=l2)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _kib(text: str) -> int:
    units = {"K": 1, "M": 1024}
    value, unit = text[:-1], text[-1].upper()
    if unit not in units:
        msg = f"unrecognized cache size {text!r}"
        raise ValueError(msg)
    return int(value) * units[unit]


def _cpu_model(cpuinfo: Path) -> str:
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "model name":
                return value.strip()
    return platform.processor() or "unknown"


@dataclass(frozen=True)
class AffinityRequest:
    """A launcher's affinity policy and the CPU set it resolved to on this machine."""

    policy: AffinityPolicy
    cpus: tuple[int, ...]

    def __post_init__(self) -> None:
        """The policy is a launch policy and the CPU set is well-formed."""
        if self.policy not in LAUNCH_POLICIES:
            msg = f"policy must be one of {LAUNCH_POLICIES}, got {self.policy!r}"
            raise ValueError(msg)
        _sorted_distinct("affinity.cpus", self.cpus)

    @classmethod
    def resolve(
        cls, policy: AffinityPolicy, topology: CoreTopology, *, explicit: Sequence[int] | None = None
    ) -> AffinityRequest:
        """Resolve a policy against the machine (``p-cores`` needs a hybrid CPU; ``explicit`` needs online CPUs)."""
        if policy == "p-cores":
            return cls(policy, topology.performance_cores())
        if policy == "all":
            return cls(policy, topology.online)
        if explicit is None:
            msg = "the explicit policy needs a CPU list"
            raise ValueError(msg)
        cpus = tuple(sorted(set(explicit)))
        offline = sorted(set(cpus) - set(topology.online))
        if offline:
            msg = f"CPUs {offline} are not online"
            raise ExecutionEnvironmentError(msg)
        return cls(policy, cpus)

    def environment(self) -> dict[str, str]:
        """The variables the launcher exports: the declared policy and CPU set, and single-thread settings."""
        return {
            POLICY_VARIABLE: self.policy,
            CPUS_VARIABLE: format_cpu_list(self.cpus),
            **dict.fromkeys(THREAD_VARIABLES, "1"),
        }

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> AffinityRequest | None:
        """The request a launcher declared in ``env``, or ``None`` when the process is unmanaged."""
        policy = env.get(POLICY_VARIABLE)
        cpus = env.get(CPUS_VARIABLE)
        if policy is None and cpus is None:
            return None
        if policy is None or cpus is None:
            msg = f"{POLICY_VARIABLE} and {CPUS_VARIABLE} must be declared together"
            raise ExecutionEnvironmentError(msg)
        if policy not in LAUNCH_POLICIES:
            msg = f"{POLICY_VARIABLE}={policy!r} is not one of {LAUNCH_POLICIES}"
            raise ExecutionEnvironmentError(msg)
        return cls(policy, parse_cpu_list(cpus))


def apply_affinity(request: AffinityRequest, env: MutableMapping[str, str] | None = None) -> None:
    """Pin the current process to the request and export its variables; a conflicting thread setting is an error."""
    env = os.environ if env is None else env
    for name in THREAD_VARIABLES:
        current = env.get(name)
        if current is not None and current != "1":
            msg = f"{name}={current!r} conflicts with the canonical single-thread setting; unset it or set it to 1"
            raise ExecutionEnvironmentError(msg)
    os.sched_setaffinity(0, request.cpus)
    effective = os.sched_getaffinity(0)
    if effective != set(request.cpus):  # pragma: no cover - sched_setaffinity either applies the set or raises
        msg = (
            f"affinity {format_cpu_list(sorted(effective))} in force differs from the requested "
            f"{format_cpu_list(request.cpus)}"
        )
        raise ExecutionEnvironmentError(msg)
    env.update(request.environment())


def launch(argv: Sequence[str], request: AffinityRequest) -> NoReturn:
    """Pin this process, export the environment, and replace it by ``argv`` (children inherit both)."""
    if not argv:
        msg = "launch needs a command"
        raise ValueError(msg)
    apply_affinity(request)
    os.execvp(argv[0], list(argv))  # noqa: S606 - the launcher exists to exec the operator's command pinned


def require_canonical(
    env: Mapping[str, str] | None = None, *, effective_cpus: Sequence[int] | None = None
) -> AffinityRequest:
    """Fail unless a launcher declared this process's affinity, it is in force, and the thread variables are 1.

    Commands and their workers call this before importing numerical libraries.
    """
    env = os.environ if env is None else env
    request = AffinityRequest.from_environment(env)
    if request is None:
        msg = (
            f"no execution policy declared ({POLICY_VARIABLE} unset): launch through "
            "`python -m arm_rc_ctrl.execution run --policy p-cores -- <command>`"
        )
        raise ExecutionEnvironmentError(msg)
    effective = set(os.sched_getaffinity(0)) if effective_cpus is None else set(effective_cpus)
    if effective != set(request.cpus):
        msg = (
            f"effective CPU affinity {format_cpu_list(sorted(effective))} differs from the declared "
            f"{request.policy} set {format_cpu_list(request.cpus)}; the restriction was not inherited"
        )
        raise ExecutionEnvironmentError(msg)
    for name in THREAD_VARIABLES:
        if env.get(name) != "1":
            msg = f"{name} must be 1 in the canonical execution environment, got {env.get(name)!r}"
            raise ExecutionEnvironmentError(msg)
    return request


@dataclass(frozen=True)
class BlasInfo:
    """The BLAS build loaded by numpy, as far as it exposes itself (OpenBLAS symbols; ``None`` otherwise)."""

    library: str | None
    config: str | None
    corename: str | None
    """The kernel set OpenBLAS selected at load time (``openblas_get_corename``)."""
    num_threads: int | None
    parallel: int | None
    """OpenBLAS threading model (0 sequential, 1 pthreads, 2 OpenMP) when exposed."""


@dataclass(frozen=True)
class OpenMpInfo:
    """The OpenMP runtime loaded by the compiled extensions, when any."""

    library: str | None
    max_threads: int | None


@dataclass(frozen=True)
class Probes:
    """Injectable probes of the loaded numerical runtimes."""

    blas: Callable[[], BlasInfo]
    openmp: Callable[[], OpenMpInfo]


def _loaded_libraries(*fragments: str) -> list[str]:
    if not MAPS.is_file():
        return []
    found: list[str] = []
    for line in MAPS.read_text(encoding="utf-8", errors="replace").splitlines():
        path = line.split()[-1] if line.split() else ""
        name = Path(path).name.lower()
        if path.startswith("/") and any(fragment in name for fragment in fragments) and path not in found:
            found.append(path)
    return found


def _symbol(handle: ctypes.CDLL, base: str) -> _CFunction | None:
    for prefix in _SYMBOL_PREFIXES:
        for suffix in _SYMBOL_SUFFIXES:
            try:
                return cast("_CFunction", getattr(handle, f"{prefix}{base}{suffix}"))
            except AttributeError:
                continue
    return None


def _call_str(handle: ctypes.CDLL, base: str) -> str | None:
    function = _symbol(handle, base)
    if function is None:
        return None
    function.restype = ctypes.c_char_p
    value = cast("bytes | None", function())
    return None if value is None else value.decode("utf-8", errors="replace")


def _call_int(handle: ctypes.CDLL, base: str) -> int | None:
    function = _symbol(handle, base)
    if function is None:
        return None
    function.restype = ctypes.c_int
    return int(cast("int", function()))


def _vendored(path: str) -> bool:
    return Path(path).parent.name.endswith(".libs")


def probe_blas() -> BlasInfo:
    """Load numpy and interrogate its own BLAS through the OpenBLAS query symbols it exports.

    Other packages (scipy, for one) map their own OpenBLAS copies; the library
    numpy vendors under ``numpy.libs`` is preferred so the record does not
    depend on import order.
    """
    numpy = importlib.import_module("numpy")
    module_file = cast("str | None", getattr(numpy, "__file__", None))
    libraries = _loaded_libraries("openblas", "mkl_rt")
    if module_file is not None:
        vendored = Path(module_file).resolve().parent.parent / "numpy.libs"
        libraries = [p for p in libraries if Path(p).resolve().parent == vendored] or libraries
    if not libraries:
        return BlasInfo(None, None, None, None, None)
    library = libraries[0]
    handle = ctypes.CDLL(library)
    return BlasInfo(
        library=Path(library).name,
        config=_call_str(handle, "openblas_get_config"),
        corename=_call_str(handle, "openblas_get_corename"),
        num_threads=_call_int(handle, "openblas_get_num_threads"),
        parallel=_call_int(handle, "openblas_get_parallel"),
    )


def probe_openmp() -> OpenMpInfo:
    """Interrogate the OpenMP runtime the compiled extensions loaded, when any.

    The system runtime the ``rclib`` extension links is preferred over copies
    other packages vendor, so the record does not depend on import order.
    """
    libraries = _loaded_libraries("libgomp", "libomp", "libiomp")
    libraries = [p for p in libraries if not _vendored(p)] or libraries
    if not libraries:
        return OpenMpInfo(None, None)
    handle = ctypes.CDLL(libraries[0])
    return OpenMpInfo(library=Path(libraries[0]).name, max_threads=_call_int(handle, "omp_get_max_threads"))


DEFAULT_PROBES: Final = Probes(blas=probe_blas, openmp=probe_openmp)


def _threads_ok(thread_environment: Mapping[str, str], blas: BlasInfo, openmp: OpenMpInfo) -> bool:
    variables = all(thread_environment.get(name) == "1" for name in THREAD_VARIABLES)
    return variables and blas.num_threads in (None, 1) and openmp.max_threads in (None, 1)


@dataclass(frozen=True)
class ExecutionRecord:
    """The actual execution environment of one process (C10, condition 3)."""

    role: str
    policy: str
    requested_cpus: tuple[int, ...]
    effective_cpus: tuple[int, ...]
    affinity_verified: bool
    """Whether the effective affinity equals the declared set (always false for an unmanaged process)."""
    thread_environment: dict[str, str]
    threads_verified: bool
    """Whether the three thread variables are 1 and every exposed runtime reports one thread."""
    topology: CoreTopology
    blas: BlasInfo
    openmp: OpenMpInfo
    packages: dict[str, str]
    python: str
    system: str
    release: str
    machine: str
    command: str
    created_at: str
    schema_version: int = field(default=EXECUTION_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        """Both verification flags re-derive from the recorded figures."""
        if self.schema_version != EXECUTION_SCHEMA_VERSION:
            msg = f"unsupported execution schema_version {self.schema_version}"
            raise ValueError(msg)
        if self.role not in ROLES or self.policy not in POLICIES:
            msg = f"role must be one of {ROLES} and policy one of {POLICIES}, got {self.role!r}, {self.policy!r}"
            raise ValueError(msg)
        _sorted_distinct("requested_cpus", self.requested_cpus)
        _sorted_distinct("effective_cpus", self.effective_cpus)
        expected_affinity = self.policy != UNMANAGED and self.requested_cpus == self.effective_cpus
        if self.affinity_verified != expected_affinity:
            msg = f"affinity_verified={self.affinity_verified} contradicts the recorded CPU sets"
            raise ValueError(msg)
        if set(self.thread_environment) - set(THREAD_VARIABLES):
            msg = f"thread_environment may only hold {THREAD_VARIABLES}"
            raise ValueError(msg)
        if self.threads_verified != _threads_ok(self.thread_environment, self.blas, self.openmp):
            msg = f"threads_verified={self.threads_verified} contradicts the recorded thread settings"
            raise ValueError(msg)
        if not self.packages or not self.python.strip() or not self.command.strip():
            msg = "an execution record needs its packages, interpreter version, and command"
            raise ValueError(msg)
        validate_utc_timestamp(self.created_at)

    @property
    def canonical(self) -> bool:
        """Whether both the affinity and the threading were verified."""
        return self.affinity_verified and self.threads_verified

    @property
    def identity(self) -> str:
        """SHA-256 of the environment-defining fields (the command, role, and timestamp are excluded).

        Bind this digest into new run evidence and execution-cache identities; two
        processes with equal identities ran in the same environment.
        """
        mapping = to_mapping(self)
        for name in ("role", "command", "created_at", "affinity_verified", "threads_verified"):
            del mapping[name]
        return sha256_bytes(canonical_json(mapping).encode("utf-8"))

    def check_canonical(self) -> None:
        """Fail unless the record describes the canonical environment."""
        if not self.canonical:
            problems = [
                *(
                    []
                    if self.affinity_verified
                    else [f"affinity: policy {self.policy!r}, effective {format_cpu_list(self.effective_cpus)}"]
                ),
                *(
                    []
                    if self.threads_verified
                    else [
                        (
                            f"threads: {self.thread_environment}, blas {self.blas.num_threads}, "
                            f"openmp {self.openmp.max_threads}"
                        )
                    ]
                ),
            ]
            msg = "not the canonical execution environment: " + "; ".join(problems)
            raise ExecutionEnvironmentError(msg)


def _packages() -> dict[str, str]:
    packages = {"arm-rc-ctrl": __version__, **dependencies.installed_versions()}
    for name in ("numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - only without numpy or scipy
            continue
    return dict(sorted(packages.items()))


def collect_execution(
    *,
    command: str,
    role: str = "main",
    env: Mapping[str, str] | None = None,
    topology: CoreTopology | None = None,
    effective_cpus: Sequence[int] | None = None,
    probes: Probes = DEFAULT_PROBES,
    now: datetime | None = None,
) -> ExecutionRecord:
    """Record the environment of this process; an unmanaged process is recorded as such, never as canonical."""
    env = os.environ if env is None else env
    topology = CoreTopology.from_sysfs() if topology is None else topology
    effective = tuple(sorted(os.sched_getaffinity(0) if effective_cpus is None else set(effective_cpus)))
    request = AffinityRequest.from_environment(env)
    policy = UNMANAGED if request is None else request.policy
    requested = effective if request is None else request.cpus
    thread_environment = {name: env[name] for name in THREAD_VARIABLES if name in env}
    blas = probes.blas()
    openmp = probes.openmp()
    stamp = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return ExecutionRecord(
        role=role,
        policy=policy,
        requested_cpus=requested,
        effective_cpus=effective,
        affinity_verified=request is not None and requested == effective,
        thread_environment=thread_environment,
        threads_verified=_threads_ok(thread_environment, blas, openmp),
        topology=topology,
        blas=blas,
        openmp=openmp,
        packages=_packages(),
        python=platform.python_version(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        command=command,
        created_at=stamp.isoformat(timespec="seconds"),
    )


def check_same_environment(parent: ExecutionRecord, worker: ExecutionRecord) -> None:
    """Fail unless a worker's record has the parent's identity (C10, condition 1)."""
    if worker.identity != parent.identity:
        msg = (
            f"worker environment {worker.identity[:_SHORT]} differs from the parent's {parent.identity[:_SHORT]} "
            f"(policy {worker.policy!r}, effective {format_cpu_list(worker.effective_cpus)})"
        )
        raise ExecutionEnvironmentError(msg)


def execution_to_json(record: ExecutionRecord) -> str:
    """Canonical JSON of the record."""
    return canonical_json(to_mapping(record))


def load_execution(path: Path) -> ExecutionRecord:
    """Strictly rebuild a record from JSON (re-deriving its verification flags)."""
    return from_mapping(cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8"))), ExecutionRecord)


def _fmt(value: object) -> str:
    return "n/a" if value is None else str(value)


def render_execution_markdown(record: ExecutionRecord) -> str:
    """A Markdown rendering of the record (the JSON stays the exact record)."""
    topology = record.topology
    core_types = (
        ", ".join(f"{name} = {format_cpu_list(cpus)}" for name, cpus in topology.core_types.items()) or "none exposed"
    )
    caches = ", ".join(f"{name} {size} KiB" for name, size in topology.l2_cache_kib.items()) or "n/a"
    lines = [
        "# Execution record",
        "",
        (
            f"- Identity `{record.identity[:_SHORT]}` (schema {record.schema_version}); role `{record.role}`; "
            f"{'canonical' if record.canonical else 'NOT canonical'}."
        ),
        f"- Command `{record.command}`; created {record.created_at}.",
        "",
        "## Affinity",
        "",
        (
            f"- Policy `{record.policy}`; requested CPUs `{format_cpu_list(record.requested_cpus)}`; effective CPUs "
            f"`{format_cpu_list(record.effective_cpus)}`; verified: {record.affinity_verified}."
        ),
        f"- CPU `{topology.model}`; online `{format_cpu_list(topology.online)}`; core types {core_types}; L2 {caches}.",
        "",
        "## Threading",
        "",
        "- Variables: "
        + (", ".join(f"`{k}={v}`" for k, v in sorted(record.thread_environment.items())) or "none")
        + ".",
        (
            f"- BLAS `{_fmt(record.blas.library)}`: config `{_fmt(record.blas.config)}`; kernel "
            f"`{_fmt(record.blas.corename)}`; threads {_fmt(record.blas.num_threads)}; parallel model "
            f"{_fmt(record.blas.parallel)}."
        ),
        (
            f"- OpenMP `{_fmt(record.openmp.library)}`: max threads {_fmt(record.openmp.max_threads)}; verified: "
            f"{record.threads_verified}."
        ),
        "",
        "## Software",
        "",
        f"- Python {record.python} on {record.system} {record.release} ({record.machine}).",
        "- Packages: " + ", ".join(f"{name} {version}" for name, version in sorted(record.packages.items())) + ".",
    ]
    return "\n".join(lines) + "\n"


def _write_record(record: ExecutionRecord, output: Path, markdown: Path | None) -> None:
    for target in (output, markdown):
        if target is not None and target.exists():
            msg = f"refusing to overwrite {target}"
            raise FileExistsError(msg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(execution_to_json(record) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown.write_text(render_execution_markdown(record), encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    command = cast("list[str]", args.command)
    if not command:
        msg = "run needs a command after --"
        raise ValueError(msg)
    topology = CoreTopology.from_sysfs()
    cpus = cast("str | None", args.cpus)
    if cpus is not None:
        request = AffinityRequest.resolve("explicit", topology, explicit=parse_cpu_list(cpus))
    else:
        request = AffinityRequest.resolve(cast("AffinityPolicy", args.policy), topology)
    launch(command, request)


def _record(args: argparse.Namespace) -> int:
    if not bool(args.allow_unmanaged):
        require_canonical()
    for name in ("numpy", "rclib"):  # load the project's numerical runtimes so the probes see them
        importlib.import_module(name)
    record = collect_execution(
        command=command_line("arm_rc_ctrl.execution", cast("list[str]", args.argv)),
        role=cast("str", args.role),
        now=datetime.now(tz=UTC),
    )
    _write_record(
        record, Path(cast("str", args.output)), None if args.markdown is None else Path(cast("str", args.markdown))
    )
    print(
        json.dumps({"identity": record.identity, "canonical": record.canonical, "output": str(args.output)}, indent=2)
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Canonical execution environment: pinned launch and execution records."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    run = subparsers.add_parser("run", help="pin this machine's CPUs, set single-thread variables, and exec a command")
    run.add_argument(
        "--policy", choices=["p-cores", "all"], default="p-cores", help="affinity policy resolved from sysfs"
    )
    run.add_argument(
        "--cpus", type=str, default=None, help="explicit CPU list instead of a policy (recorded as explicit)"
    )
    run.add_argument("command", nargs=argparse.REMAINDER, help="command to exec after `--`")
    record = subparsers.add_parser("record", help="write the execution record of this process")
    record.add_argument("--output", type=str, required=True, help="record JSON to write (must not exist)")
    record.add_argument("--markdown", type=str, default=None, help="optional Markdown rendering (must not exist)")
    record.add_argument("--role", choices=list(ROLES), default="main", help="process role")
    record.add_argument(
        "--allow-unmanaged", action="store_true", help="record a process that was not launched pinned (never canonical)"
    )
    args = parser.parse_args(argv)
    args.argv = argv
    if args.subcommand == "run":
        args.command = (
            [arg for arg in cast("list[str]", args.command) if arg != "--"]
            if args.command[:1] == ["--"]
            else cast("list[str]", args.command)
        )
        return _run(args)
    return _record(args)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
