# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""M3REP-009: affinity policies resolve from sysfs, the guard and record verify the environment, launches pin."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import arm_rc_ctrl.execution as execution_module
from arm_rc_ctrl.execution import (
    CPUS_VARIABLE,
    POLICY_VARIABLE,
    THREAD_VARIABLES,
    AffinityRequest,
    BlasInfo,
    CoreTopology,
    ExecutionEnvironmentError,
    OpenMpInfo,
    Probes,
    apply_affinity,
    check_same_environment,
    collect_execution,
    execution_to_json,
    format_cpu_list,
    launch,
    load_execution,
    main,
    parse_cpu_list,
    probe_blas,
    probe_openmp,
    render_execution_markdown,
    require_canonical,
)

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
FAKE_BLAS = BlasInfo(
    library="libfake_openblas.so", config="OpenBLAS test", corename="Haswell", num_threads=1, parallel=1
)
FAKE_OPENMP = OpenMpInfo(library="libgomp.so.1", max_threads=1)
PROBES = Probes(blas=lambda: FAKE_BLAS, openmp=lambda: FAKE_OPENMP)


def _sysfs(tmp_path: Path, *, hybrid: bool = True, online: str = "0-7") -> tuple[Path, Path]:
    """A fake sysfs/cpuinfo pair: eight CPUs, optionally split into four performance and four efficient cores."""
    sysfs = tmp_path / "sys"
    cpu_root = sysfs / "devices" / "system" / "cpu"
    cpu_root.mkdir(parents=True)
    (cpu_root / "online").write_text(online + "\n", encoding="utf-8")
    for cpu in parse_cpu_list(online):
        topology = cpu_root / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "core_id").write_text(f"{cpu // 2 if cpu < 4 else cpu + 10}\n", encoding="utf-8")
        cache = cpu_root / f"cpu{cpu}" / "cache" / "index2"
        cache.mkdir(parents=True)
        (cache / "size").write_text("2048K\n" if cpu < 4 else "4M\n", encoding="utf-8")
    if hybrid:
        for name, cpus in (("core", "0-3"), ("atom", "4-7")):
            (sysfs / "devices" / f"cpu_{name}").mkdir(parents=True)
            (sysfs / "devices" / f"cpu_{name}" / "cpus").write_text(cpus + "\n", encoding="utf-8")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Fake Hybrid CPU\n", encoding="utf-8")
    return sysfs, cpuinfo


def _env(request: AffinityRequest, **overrides: str) -> dict[str, str]:
    return {**request.environment(), **overrides}


def test_cpu_lists_roundtrip_and_reject_nonsense() -> None:
    """Kernel-style lists parse to sorted distinct CPUs and format back compactly."""
    assert parse_cpu_list("0-3,7,5-6") == (0, 1, 2, 3, 5, 6, 7)
    assert format_cpu_list((7, 0, 1, 2, 3, 5, 6)) == "0-3,5-7"
    assert format_cpu_list([4]) == "4"
    for bad in ("", "a", "3-1", "-1"):
        with pytest.raises(ValueError, match="CPU"):
            parse_cpu_list(bad)
    with pytest.raises(ValueError, match="empty"):
        format_cpu_list([])


def test_topology_reads_sysfs_and_resolves_policies(tmp_path: Path) -> None:
    """Core types, core ids, and L2 sizes come from sysfs; p-cores means the ``core`` type, never fixed numbers."""
    sysfs, cpuinfo = _sysfs(tmp_path)
    topology = CoreTopology.from_sysfs(sysfs, cpuinfo)
    assert topology.model == "Fake Hybrid CPU"
    assert topology.online == (0, 1, 2, 3, 4, 5, 6, 7)
    assert topology.core_types == {"atom": (4, 5, 6, 7), "core": (0, 1, 2, 3)}
    assert topology.core_ids["0"] == 0
    assert topology.core_ids["7"] == 17
    assert topology.l2_cache_kib == {"atom": 4096, "core": 2048}
    assert topology.hybrid
    assert topology.core_type_of(5) == "atom"
    assert AffinityRequest.resolve("p-cores", topology) == AffinityRequest("p-cores", (0, 1, 2, 3))
    assert AffinityRequest.resolve("all", topology).cpus == topology.online
    assert AffinityRequest.resolve("explicit", topology, explicit=[6, 4]).cpus == (4, 6)
    with pytest.raises(ExecutionEnvironmentError, match="not online"):
        AffinityRequest.resolve("explicit", topology, explicit=[9])
    with pytest.raises(ValueError, match="needs a CPU list"):
        AffinityRequest.resolve("explicit", topology)
    flat_sysfs, flat_cpuinfo = _sysfs(tmp_path / "flat", hybrid=False)
    flat = CoreTopology.from_sysfs(flat_sysfs, flat_cpuinfo)
    assert not flat.hybrid
    assert flat.core_types == {}
    assert flat.l2_cache_kib == {"all": 2048}
    assert flat.core_type_of(3) == "all"
    with pytest.raises(ExecutionEnvironmentError, match="p-cores"):
        AffinityRequest.resolve("p-cores", flat)


def test_topology_and_request_invariants() -> None:
    """Overlapping core types, offline core ids, unknown cache keys, and malformed CPU sets are rejected."""
    with pytest.raises(ValueError, match="disjoint"):
        CoreTopology("m", (0, 1), {"core": (0, 1), "atom": (1,)}, {}, {})
    with pytest.raises(ValueError, match="not online"):
        CoreTopology("m", (0, 1), {}, {"5": 0}, {})
    with pytest.raises(ValueError, match="l2_cache_kib"):
        CoreTopology("m", (0, 1), {}, {}, {"core": 1})
    with pytest.raises(ValueError, match="sorted"):
        AffinityRequest("all", (1, 0))
    with pytest.raises(ValueError, match="policy"):
        AffinityRequest("unmanaged", (0,))  # type: ignore[arg-type]
    with pytest.raises(ExecutionEnvironmentError, match="together"):
        AffinityRequest.from_environment({POLICY_VARIABLE: "all"})
    with pytest.raises(ExecutionEnvironmentError, match="not one of"):
        AffinityRequest.from_environment({POLICY_VARIABLE: "fast", CPUS_VARIABLE: "0"})
    assert AffinityRequest.from_environment({}) is None


def test_guard_requires_declared_and_inherited_affinity() -> None:
    """The guard fails without a declaration, on a lost restriction, and on a foreign thread setting."""
    request = AffinityRequest("explicit", (0, 1))
    assert require_canonical(_env(request), effective_cpus=[1, 0]) == request
    with pytest.raises(ExecutionEnvironmentError, match="no execution policy"):
        require_canonical({}, effective_cpus=[0, 1])
    with pytest.raises(ExecutionEnvironmentError, match="not inherited"):
        require_canonical(_env(request), effective_cpus=[0, 1, 2])
    with pytest.raises(ExecutionEnvironmentError, match="OPENBLAS_NUM_THREADS must be 1"):
        require_canonical(_env(request, OPENBLAS_NUM_THREADS="4"), effective_cpus=[0, 1])


def test_apply_affinity_pins_this_process_and_refuses_conflicts() -> None:
    """The current process is pinned to a subset of its mask and the variables are exported; conflicts fail first."""
    original = sorted(os.sched_getaffinity(0))
    request = AffinityRequest("explicit", (original[-1],))
    env: dict[str, str] = {"OMP_NUM_THREADS": "1"}
    try:
        apply_affinity(request, env)
        assert sorted(os.sched_getaffinity(0)) == [original[-1]]
        assert env[POLICY_VARIABLE] == "explicit"
        assert env[CPUS_VARIABLE] == str(original[-1])
        assert all(env[name] == "1" for name in THREAD_VARIABLES)
    finally:
        os.sched_setaffinity(0, original)
    with pytest.raises(ExecutionEnvironmentError, match="conflicts"):
        apply_affinity(request, {"MKL_NUM_THREADS": "2"})


def test_launch_execs_the_command_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """``launch`` pins, exports, and hands the process to the command through exec."""
    original = sorted(os.sched_getaffinity(0))
    captured: dict[str, object] = {}

    def fake_execvp(file: str, argv: list[str]) -> None:
        captured["file"] = file
        captured["argv"] = list(argv)
        captured["affinity"] = sorted(os.sched_getaffinity(0))
        captured["policy"] = os.environ.get(POLICY_VARIABLE)

    monkeypatch.setattr(os, "execvp", fake_execvp)
    for name in (POLICY_VARIABLE, CPUS_VARIABLE, *THREAD_VARIABLES):
        monkeypatch.delenv(name, raising=False)
    try:
        launch(["true", "--flag"], AffinityRequest("explicit", (original[0],)))
    finally:
        os.sched_setaffinity(0, original)
    assert captured == {"file": "true", "argv": ["true", "--flag"], "affinity": [original[0]], "policy": "explicit"}
    with pytest.raises(ValueError, match="needs a command"):
        launch([], AffinityRequest("explicit", (original[0],)))


def test_launcher_subprocess_pins_the_target_and_its_children() -> None:
    """The real launcher: the target interpreter starts pinned, exports the variables, and its child inherits them."""
    original = sorted(os.sched_getaffinity(0))
    target = [original[0]]
    child = (
        "import json, os, subprocess, sys; "
        "inner = subprocess.run([sys.executable, '-c', 'import os; print(sorted(os.sched_getaffinity(0)))'], "
        "capture_output=True, text=True, check=True).stdout.strip(); "
        "print(json.dumps({'self': sorted(os.sched_getaffinity(0)), 'child': inner, "
        f"'env': {{k: os.environ.get(k) for k in {[POLICY_VARIABLE, CPUS_VARIABLE, *THREAD_VARIABLES]!r}}}}}))"
    )
    env = {k: v for k, v in os.environ.items() if k not in (POLICY_VARIABLE, CPUS_VARIABLE, *THREAD_VARIABLES)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arm_rc_ctrl.execution",
            "run",
            "--cpus",
            format_cpu_list(target),
            "--",
            sys.executable,
            "-c",
            child,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert payload["self"] == target
    assert json.loads(payload["child"]) == target
    assert payload["env"][POLICY_VARIABLE] == "explicit"
    assert payload["env"][CPUS_VARIABLE] == format_cpu_list(target)
    assert all(payload["env"][name] == "1" for name in THREAD_VARIABLES)


def test_record_collects_verifies_and_roundtrips(tmp_path: Path) -> None:
    """A managed process yields a canonical record whose identity ignores the command and timestamp."""
    sysfs, cpuinfo = _sysfs(tmp_path)
    topology = CoreTopology.from_sysfs(sysfs, cpuinfo)
    request = AffinityRequest.resolve("p-cores", topology)
    record = collect_execution(
        command="python -m x", env=_env(request), topology=topology, effective_cpus=[3, 2, 1, 0], probes=PROBES, now=NOW
    )
    assert record.canonical
    assert record.policy == "p-cores"
    assert record.requested_cpus == record.effective_cpus == (0, 1, 2, 3)
    assert record.thread_environment == dict.fromkeys(THREAD_VARIABLES, "1")
    assert record.blas == FAKE_BLAS
    assert record.created_at == "2026-09-09T12:00:00+00:00"
    assert "numpy" in record.packages
    later = replace(record, command="python -m y", created_at="2026-09-10T00:00:00+00:00", role="worker")
    assert later.identity == record.identity
    other_cpus = collect_execution(
        command="python -m x", env=_env(request), topology=topology, effective_cpus=[0, 1], probes=PROBES, now=NOW
    )
    assert not other_cpus.affinity_verified
    assert other_cpus.identity != record.identity
    with pytest.raises(ExecutionEnvironmentError, match="differs from the parent"):
        check_same_environment(record, other_cpus)
    check_same_environment(record, later)
    file = tmp_path / "execution.json"
    file.write_text(execution_to_json(record) + "\n", encoding="utf-8")
    assert load_execution(file) == record
    tampered = json.loads(file.read_text(encoding="utf-8"))
    tampered["effective_cpus"] = [0, 1]
    file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="affinity_verified"):
        load_execution(file)


def test_unmanaged_and_multithreaded_records_are_never_canonical(tmp_path: Path) -> None:
    """Without a declared policy or with a runtime reporting several threads the record says so and the check fails."""
    sysfs, cpuinfo = _sysfs(tmp_path)
    topology = CoreTopology.from_sysfs(sysfs, cpuinfo)
    unmanaged = collect_execution(
        command="python -m x",
        env={"OMP_NUM_THREADS": "1"},
        topology=topology,
        effective_cpus=[0, 1],
        probes=PROBES,
        now=NOW,
    )
    assert unmanaged.policy == "unmanaged"
    assert unmanaged.requested_cpus == unmanaged.effective_cpus == (0, 1)
    assert not unmanaged.affinity_verified
    assert not unmanaged.threads_verified
    with pytest.raises(ExecutionEnvironmentError, match=r"affinity.*threads"):
        unmanaged.check_canonical()
    request = AffinityRequest.resolve("all", topology)
    busy = Probes(blas=lambda: replace(FAKE_BLAS, num_threads=8), openmp=lambda: FAKE_OPENMP)
    multithreaded = collect_execution(
        command="python -m x",
        env=_env(request),
        topology=topology,
        effective_cpus=topology.online,
        probes=busy,
        now=NOW,
    )
    assert multithreaded.affinity_verified
    assert not multithreaded.threads_verified
    with pytest.raises(ExecutionEnvironmentError, match="blas 8"):
        multithreaded.check_canonical()
    with pytest.raises(ValueError, match="threads_verified"):
        replace(multithreaded, threads_verified=True)
    with pytest.raises(ValueError, match="role"):
        replace(multithreaded, role="observer")
    with pytest.raises(ValueError, match="schema_version"):
        replace(multithreaded, schema_version=2)
    with pytest.raises(ValueError, match="thread_environment"):
        replace(multithreaded, thread_environment={"GOTO_NUM_THREADS": "1"})


def test_real_probes_expose_the_loaded_runtimes() -> None:
    """The BLAS numpy vendors is interrogated even after scipy mapped its copy; OpenMP reports its threads."""
    importlib.import_module("scipy.linalg")  # maps scipy's own OpenBLAS copy and Cython BLAS modules first
    numpy = importlib.import_module("numpy")
    blas = probe_blas()
    if blas.library is not None and blas.corename is not None:
        assert blas.config is not None
        assert blas.num_threads is not None
        assert blas.num_threads >= 1
        vendored = Path(str(numpy.__file__)).resolve().parent.parent / "numpy.libs"
        if vendored.is_dir():
            assert (vendored / blas.library).is_file()
    importlib.import_module("rclib")  # loads the OpenMP runtime the compiled extension links, when it has one
    openmp = probe_openmp()
    if openmp.library is not None:
        assert openmp.max_threads is not None
        assert openmp.max_threads >= 1


def test_markdown_states_canonicality_affinity_threading_and_software(tmp_path: Path) -> None:
    """The rendering names the identity, verdict, affinity, threading, and package versions."""
    sysfs, cpuinfo = _sysfs(tmp_path)
    topology = CoreTopology.from_sysfs(sysfs, cpuinfo)
    request = AffinityRequest.resolve("p-cores", topology)
    record = collect_execution(
        command="python -m x", env=_env(request), topology=topology, effective_cpus=[0, 1, 2, 3], probes=PROBES, now=NOW
    )
    text = render_execution_markdown(record)
    for required in (
        "# Execution record",
        "canonical",
        "## Affinity",
        "p-cores",
        "core = 0-3",
        "2048 KiB",
        "## Threading",
        "Haswell",
        "## Software",
        "numpy",
    ):
        assert required in text
    assert "NOT canonical" not in text
    assert "NOT canonical" in render_execution_markdown(replace(record, policy="unmanaged", affinity_verified=False))


def test_main_records_managed_and_unmanaged_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``record`` writes JSON and Markdown once for a declared environment; unmanaged needs --allow-unmanaged."""
    current = sorted(os.sched_getaffinity(0))
    request = AffinityRequest("explicit", tuple(current))
    for name, value in request.environment().items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "execution.json"
    markdown = tmp_path / "execution.md"
    assert main(["record", "--output", str(output), "--markdown", str(markdown)]) == 0
    record = load_execution(output)
    assert record.policy == "explicit"
    assert record.affinity_verified
    assert record.command.startswith("python -m arm_rc_ctrl.execution record")
    assert render_execution_markdown(record) == markdown.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing"):
        main(["record", "--output", str(output)])
    for name in (POLICY_VARIABLE, CPUS_VARIABLE):
        monkeypatch.delenv(name)
    with pytest.raises(ExecutionEnvironmentError, match="no execution policy"):
        main(["record", "--output", str(tmp_path / "other.json")])
    assert main(["record", "--output", str(tmp_path / "other.json"), "--allow-unmanaged"]) == 0
    assert load_execution(tmp_path / "other.json").policy == "unmanaged"


def test_main_run_resolves_the_policy_and_execs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run`` resolves ``--cpus`` or a policy against the live topology and execs the command pinned."""
    original = sorted(os.sched_getaffinity(0))
    captured: dict[str, object] = {}

    def fake_execvp(_file: str, argv: list[str]) -> None:
        captured["argv"] = list(argv)
        captured["affinity"] = sorted(os.sched_getaffinity(0))

    monkeypatch.setattr(os, "execvp", fake_execvp)
    for name in (POLICY_VARIABLE, CPUS_VARIABLE, *THREAD_VARIABLES):
        monkeypatch.delenv(name, raising=False)
    try:
        assert main(["run", "--cpus", str(original[0]), "--", "true", "x"]) is None
        assert captured == {"argv": ["true", "x"], "affinity": [original[0]]}
        assert os.environ[POLICY_VARIABLE] == "explicit"
        main(["run", "--policy", "all", "true"])
        assert captured["affinity"] == list(CoreTopology.from_sysfs().online)  # all online CPUs, not the test's mask
    finally:
        os.sched_setaffinity(0, original)
    with pytest.raises(ValueError, match="needs a command"):
        main(["run", "--policy", "all"])


def test_topology_edge_cases_and_probe_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty models, unknown cache units, missing cpuinfo, missing maps, and absent symbols are all handled."""
    with pytest.raises(ValueError, match="model"):
        CoreTopology(" ", (0,), {}, {}, {})
    sysfs, cpuinfo = _sysfs(tmp_path)
    (sysfs / "devices" / "system" / "cpu" / "cpu0" / "cache" / "index2" / "size").write_text("2X\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache size"):
        CoreTopology.from_sysfs(sysfs, cpuinfo)
    (sysfs / "devices" / "system" / "cpu" / "cpu0" / "cache" / "index2" / "size").write_text(
        "2048K\n", encoding="utf-8"
    )
    assert CoreTopology.from_sysfs(sysfs, tmp_path / "no-cpuinfo").model
    monkeypatch.setattr(execution_module, "MAPS", tmp_path / "no-maps")
    assert probe_blas() == BlasInfo(None, None, None, None, None)
    assert probe_openmp() == OpenMpInfo(None, None)
    monkeypatch.undo()
    monkeypatch.setattr(execution_module, "_SYMBOL_PREFIXES", ("no_such_prefix_",))
    blas = probe_blas()
    if blas.library is not None:
        assert (blas.config, blas.corename, blas.num_threads, blas.parallel) == (None, None, None, None)
    monkeypatch.undo()
    request = AffinityRequest.resolve("all", CoreTopology.from_sysfs(sysfs, cpuinfo))
    record = collect_execution(
        command="python -m x",
        env=_env(request),
        topology=CoreTopology.from_sysfs(sysfs, cpuinfo),
        effective_cpus=request.cpus,
        probes=PROBES,
        now=NOW,
    )
    with pytest.raises(ValueError, match="needs its packages"):
        replace(record, command=" ")
