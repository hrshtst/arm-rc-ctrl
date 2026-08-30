# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Import a recorded ``skelarm`` demonstration as an immutable raw artifact (M1-013).

The native ``.sklog.npz`` is copied unchanged to a staging directory under the
``raw`` bucket, digested, and moved to ``armrc://raw/<artifact-id>/demo.sklog.npz``;
the record is then verified with the raw loader before it is written to
``data/records/raw/`` and appended to the catalog. Intervals are either given
explicitly (scripted teachers) or proposed from the speed profile and confirmed
after visual review of the saved plot (human teachers).

Command line::

    python -m arm_rc_ctrl.data.import_demo --log demo.sklog.npz --scenario configs/tasks/task_1a.toml
        --session scripted-minjerk-01 --license CC0-1.0 --access public --clock simulated
        (--intervals 0,1,4,5 | --propose-intervals [--confirm]) [--plot review.png] [--notes ...] [--exploratory]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
from skelarm import StateLog

from arm_rc_ctrl.config import to_mapping
from arm_rc_ctrl.data.raw import load_raw_demonstration, read_log_schema_version
from arm_rc_ctrl.data.records import (
    RAW_PAYLOAD_FORMAT,
    RAW_PAYLOAD_NAME,
    AccessClass,
    ArtifactRecord,
    Intervals,
    Origin,
    Payload,
    RawDemonstrationRecord,
    Sampling,
    Scenario,
    catalog_path,
    load_catalog,
    load_record,
    make_artifact_id,
    write_catalog,
    write_record,
)
from arm_rc_ctrl.data.review import IntervalProposal, plot_intervals, propose_intervals
from arm_rc_ctrl.provenance import collect_provenance, require_clean_for_confirmatory, sha256_file
from arm_rc_ctrl.repo import repository_root
from arm_rc_ctrl.scenario import ScenarioConfig, load_scenario
from arm_rc_ctrl.storage import ArtifactUri, StorageRoot, open_storage

__all__ = ["ImportResult", "import_demonstration", "main", "sampling_from_log"]

type Clock = Literal["simulated", "wall"]


@dataclass(frozen=True)
class ImportResult:
    """What an import produced."""

    record: RawDemonstrationRecord
    record_file: Path
    payload_file: Path
    resumed: bool


def sampling_from_log(log: StateLog, clock: Clock, units_override: dict[str, str] | None = None) -> Sampling:
    """Derive the sampling section from the log: median period and per-channel units from the metadata."""
    times = log.times
    if times.shape[0] < 2:  # noqa: PLR2004
        msg = "the log needs at least two frames"
        raise ValueError(msg)
    period = float(np.median(np.diff(times)))
    units: dict[str, str] = {"t": "s"}
    for name in log.channel_names:
        unit = cast("dict[str, object]", log.channel_meta.get(name, {})).get("unit")
        if isinstance(unit, str):
            units[name] = unit
    units.update(units_override or {})
    missing = sorted(set(log.channel_names) - set(units))
    if missing:
        msg = f"the log declares no unit for channel(s) {missing}; pass --units name=unit for them"
        raise ValueError(msg)
    return Sampling(period_s=period, clock=clock, units=units)


def _scenario_section(scenario_file: Path, scenario: ScenarioConfig, records_root: Path) -> Scenario:
    try:
        relative = scenario_file.resolve().relative_to(records_root.resolve()).as_posix()
    except ValueError as exc:
        msg = f"scenario file {scenario_file} must live inside the repository {records_root}"
        raise ValueError(msg) from exc
    return Scenario(
        config_path=relative,
        config_sha256=sha256_file(scenario_file),
        robot=scenario.robot.name,
        task=scenario.name,
        dof=scenario.dof,
        initial_q=scenario.task.initial_q,
        target=scenario.task.target,
    )


def _identical_payload_present(final_dir: Path, digest: str, artifact_id: str) -> bool:
    """``True`` if the content-addressed directory already holds this exact payload; error if it holds another."""
    if not final_dir.exists():
        return False
    existing = final_dir / RAW_PAYLOAD_NAME
    if not existing.is_file() or sha256_file(existing) != digest:
        msg = f"{artifact_id} exists under {final_dir.parent} with a different payload; inspect it manually"
        raise FileExistsError(msg)
    return True


def import_demonstration(
    log_file: Path,
    scenario_file: Path,
    *,
    store: StorageRoot,
    records_root: Path,
    session: str,
    license_label: str,
    access: AccessClass,
    intervals: Intervals,
    clock: Clock,
    exploratory: bool,
    units_override: dict[str, str] | None = None,
    notes: str = "",
    now: datetime | None = None,
    command: str = "python -m arm_rc_ctrl.data.import_demo",
) -> ImportResult:
    """Copy the log into the store transactionally, verify it against its record, and register the record."""
    scenario = load_scenario(scenario_file)
    log = StateLog.load(log_file)
    payload_schema = read_log_schema_version(log_file)
    sampling = sampling_from_log(log, clock, units_override)
    duration = float(log.times[-1])
    if intervals.duration_s > duration + 0.5 * sampling.period_s:
        msg = f"intervals end at {intervals.duration_s} s but the recording ends at {duration:.4f} s"
        raise ValueError(msg)
    resolved = {
        "scenario": to_mapping(scenario),
        "import": {"session": session, "clock": clock, "intervals": to_mapping(intervals), "log": log_file.name},
    }
    provenance = collect_provenance(resolved, seeds={}, exploratory=exploratory, now=now)
    require_clean_for_confirmatory(provenance)

    staging = store.root / "raw" / f"staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    created = False
    final_dir: Path | None = None
    try:
        payload = staging / RAW_PAYLOAD_NAME
        shutil.copyfile(log_file, payload)
        digest = sha256_file(payload)
        artifact_id = make_artifact_id("raw", provenance.created_at, digest)
        final_dir = store.path(ArtifactUri("raw", (artifact_id,)), mode="write")
        record = RawDemonstrationRecord(
            artifact=ArtifactRecord(
                artifact_id=artifact_id,
                kind="raw",
                created_at=provenance.created_at,
                license=license_label,
                access=access,
                payload=Payload(
                    uri=f"armrc://raw/{artifact_id}/{RAW_PAYLOAD_NAME}",
                    sha256=digest,
                    size=payload.stat().st_size,
                    format=RAW_PAYLOAD_FORMAT,
                    schema_version=payload_schema,
                ),
                origin=Origin.from_provenance(provenance, command=command),
                notes=notes,
            ),
            scenario=_scenario_section(scenario_file, scenario, records_root),
            sampling=sampling,
            session=session,
            intervals=intervals,
            duration_s=intervals.duration_s,
        )
        resumed = _identical_payload_present(final_dir, digest, artifact_id)
        if resumed:
            shutil.rmtree(staging, ignore_errors=True)
        else:
            staging.rename(final_dir)
            created = True
        load_raw_demonstration(store, record)  # the payload must satisfy its own record before it is registered
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if created and final_dir is not None:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise

    record_file = records_root / "data" / "records" / "raw" / f"{artifact_id}.toml"
    if record_file.exists():
        existing_record = load_record(record_file, RawDemonstrationRecord)
        if not resumed or existing_record.artifact.payload != record.artifact.payload:
            msg = f"{record_file} already exists and does not describe this payload; records are immutable"
            raise FileExistsError(msg)
        record = existing_record
    else:
        record_file.parent.mkdir(parents=True, exist_ok=True)
        write_record(record_file, record)
    catalog_file = catalog_path(records_root)
    catalog = load_catalog(catalog_file)
    if catalog.find(artifact_id) is None:
        write_catalog(
            catalog_file, catalog.with_record(record.artifact, record_file.relative_to(records_root).as_posix())
        )
    return ImportResult(record, record_file, final_dir / RAW_PAYLOAD_NAME, resumed)


def _parse_intervals(text: str) -> Intervals:
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:  # noqa: PLR2004
        msg = "--intervals expects four boundaries: prime_start,move_start,dwell_start,end"
        raise ValueError(msg)
    a, b, c, d = values
    return Intervals(prime=(a, b), move=(b, c), dwell=(c, d))


def _parse_units(items: list[str]) -> dict[str, str]:
    units: dict[str, str] = {}
    for item in items:
        name, sep, unit = item.partition("=")
        if not sep or not name or not unit:
            msg = f"--units expects name=unit, got {item!r}"
            raise ValueError(msg)
        units[name] = unit
    return units


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; a thin wrapper around :func:`import_demonstration`."""
    parser = argparse.ArgumentParser(description="Import a recorded skelarm demonstration as a raw artifact.")
    parser.add_argument("--log", type=Path, required=True, help="recorded .sklog.npz")
    parser.add_argument(
        "--scenario", type=Path, required=True, help="scenario TOML the demonstration was recorded under"
    )
    parser.add_argument("--session", required=True, help="pseudonymous teacher/session ID (e.g. teacher-01)")
    parser.add_argument("--license", required=True, help="SPDX identifier or LicenseRef-... for the data")
    parser.add_argument("--access", required=True, choices=("private", "internal", "public"))
    parser.add_argument("--clock", default="wall", choices=("simulated", "wall"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--intervals", help="explicit boundaries prime_start,move_start,dwell_start,end (s)")
    group.add_argument("--propose-intervals", action="store_true", help="derive boundaries from the speed profile")
    parser.add_argument("--speed-threshold", type=float, default=None, help="rad/s, for --propose-intervals")
    parser.add_argument("--confirm", action="store_true", help="import with the proposed intervals after review")
    parser.add_argument("--plot", type=Path, default=None, help="write the review plot (PNG) here")
    parser.add_argument("--units", nargs="*", default=[], help="name=unit for channels without unit metadata")
    parser.add_argument("--notes", default="")
    parser.add_argument("--exploratory", action="store_true", help="allow a dirty worktree")
    args = parser.parse_args(argv)

    log_file = Path(args.log)
    log = StateLog.load(log_file)
    proposal: IntervalProposal | None = None
    if args.propose_intervals:
        q = log.channel("q")
        dq = log.channel("dq") if "dq" in log.channel_names else None
        kwargs = {} if args.speed_threshold is None else {"speed_threshold": float(args.speed_threshold)}
        proposal = propose_intervals(log.times, q, dq=dq, **kwargs)
        intervals = proposal.intervals
        print("proposed intervals:", json.dumps(to_mapping(intervals)))
    else:
        intervals = _parse_intervals(args.intervals)
    if args.plot is not None:
        q = log.channel("q")
        dq = log.channel("dq") if "dq" in log.channel_names else None
        speed = (
            proposal.speed
            if proposal is not None
            else np.max(np.abs(dq if dq is not None else np.gradient(q, log.times, axis=0)), axis=1)
        )
        plot_intervals(
            log.times,
            q,
            speed,
            intervals,
            Path(args.plot),
            speed_threshold=proposal.speed_threshold if proposal else None,
            title=f"{log_file.name}: {args.session}",
        )
        print(f"review plot written to {args.plot}")
    if args.propose_intervals and not args.confirm:
        print("re-run with --confirm (and the same --speed-threshold) to import with these intervals")
        return 0

    root = repository_root()
    result = import_demonstration(
        log_file,
        Path(args.scenario),
        store=open_storage(),
        records_root=root,
        session=args.session,
        license_label=args.license,
        access=cast("AccessClass", args.access),
        intervals=intervals,
        clock=cast("Clock", args.clock),
        exploratory=args.exploratory,
        units_override=_parse_units(args.units),
        notes=args.notes,
        now=datetime.now(UTC),
        command=" ".join(
            ["python", "-m", "arm_rc_ctrl.data.import_demo", *(argv if argv is not None else sys.argv[1:])]
        ),
    )
    print(
        json.dumps(
            {
                "artifact_id": result.record.artifact.artifact_id,
                "uri": result.record.artifact.payload.uri,
                "record": result.record_file.relative_to(root).as_posix(),
                "frames": len(log),
                "resumed": result.resumed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
