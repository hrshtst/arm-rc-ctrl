# Copyright (c) 2026 Hiroshi Atsuta
# SPDX-License-Identifier: GPL-3.0-only

"""Export the verified expert report's eight animations and matching plots, without the evidence store."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

import matplotlib as mpl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arm_rc_ctrl.config import from_mapping
from arm_rc_ctrl.experiments.manual_expert_report import CASES, Illustration, verify_report
from arm_rc_ctrl.experiments.manual_report import DOCUMENT_LOCATION
from arm_rc_ctrl.provenance import sha256_file, worktree_state
from arm_rc_ctrl.repo import repository_root

WIDTH, HEIGHT = 1500, 450
PANEL = WIDTH // 5
FRAME_MS = 50
PULSE_BOUNDS = 2
SPEED = 2.0
FONT = Path(mpl.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
DARK = "#172c39"
GREY = "#a1a9ae"


@dataclass(frozen=True)
class AnimationRun:
    """The exact decimated display samples used by the browser player."""

    role: str
    color: str
    success: bool
    reason: str | None
    end: float
    t: tuple[float, ...]
    q: tuple[tuple[float | None, ...], ...]
    ref: tuple[tuple[float | None, ...], ...]
    pulse: tuple[float, ...]
    final_error_mm: float
    dwell_task_s: float | None
    final_dwell_s: float
    sha256: str
    arrays_sha256: str
    uri: str
    samples: int
    termination: str


@dataclass(frozen=True)
class Animation:
    """One synchronized five-arm illustration."""

    slug: str
    title: str
    lengths: tuple[float, ...]
    target: tuple[float, ...]
    radius: float
    activation: float
    runs: tuple[AnimationRun, ...]


def load_animation(path: Path) -> Animation:
    """Parse the browser's data wrapper as JSON, never execute JavaScript."""
    text = path.read_text(encoding="utf-8").strip()
    prefix, suffix = "window.manualCases.push(", ");"
    if not text.startswith(prefix) or not text.endswith(suffix):
        msg = "Unexpected animation data wrapper"
        raise ValueError(msg)
    return from_mapping(cast("dict[str, object]", json.loads(text[len(prefix) : -len(suffix)])), Animation)


def sample_index(times: tuple[float, ...], t: float) -> int:
    """Hold the preceding sample, including the last recorded state after an abort."""
    return max(0, min(len(times) - 1, bisect_right(times, t) - 1))


def _points(q: tuple[float | None, ...], lengths: tuple[float, ...], panel: int) -> list[tuple[float, float]]:
    x = y = angle = 0.0
    points = [(panel * PANEL + PANEL / 2, 245.0)]
    for value, length in zip(q, lengths, strict=True):
        if value is None or not math.isfinite(value):
            return []
        angle += value
        x += length * math.cos(angle)
        y += length * math.sin(angle)
        points.append((panel * PANEL + PANEL / 2 + x * 240, 245 - y * 240))
    return points


def _dashed(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str) -> None:
    for a, b in pairwise(points):
        length = math.dist(a, b)
        if not length:
            continue
        for start in np.arange(0, length, 10):
            end = min(start + 5, length)
            draw.line(
                [(a[0] + (b[0] - a[0]) * s / length, a[1] + (b[1] - a[1]) * s / length) for s in (start, end)],
                fill=color,
                width=2,
            )


def render_frame(data: Animation, t: float) -> Image.Image:
    """A five-panel task-space frame with actual/reference poses, traces and recorded pulse times."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 16)
    small = ImageFont.truetype(str(FONT), 13)
    heading = ImageFont.truetype(str(FONT), 22)
    draw.text((18, 10), data.title, font=heading, fill=DARK)
    draw.text(
        (18, 42),
        "Solid: measured arm + tip trail    Dashed: desired arm    Ring: 10 mm target    PASS/FAIL: whole-run verdict",
        font=small,
        fill=DARK,
    )
    draw.text((1150, 12), f"t = {t:6.2f} s    Playback: 2x", font=font, fill=DARK)
    for panel, run in enumerate(data.runs):
        left = panel * PANEL
        i = sample_index(run.t, t)
        ended = t > run.end + 0.005
        color = GREY if ended else run.color
        draw.text((left + 14, 73), run.role, font=heading, fill=run.color)
        draw.text(
            (left + PANEL - 73, 77),
            "PASS" if run.success else "FAIL",
            font=font,
            fill="#277047" if run.success else "#a84322",
        )
        for v in (-0.5, -0.25, 0.0, 0.25, 0.5):
            x, y = left + PANEL / 2 + v * 240, 245 - v * 240
            draw.line((x, 113, x, 377), fill="#e7edf0")
            draw.line((left + 18, y, left + PANEL - 18, y), fill="#e7edf0")
        tx, ty = left + PANEL / 2 + data.target[0] * 240, 245 - data.target[1] * 240
        radius = data.radius * 240
        draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), outline=DARK, width=2)
        trail = [pts[-1] for q in run.q[: i + 1] if (pts := _points(q, data.lengths, panel))]
        if len(trail) > 1:
            draw.line(trail, fill=color, width=1)
        _dashed(draw, _points(run.ref[i], data.lengths, panel), color)
        actual = _points(run.q[i], data.lengths, panel)
        if actual:
            draw.line(actual, fill=color, width=4)
            for x, y in actual:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        pulse = len(run.pulse) == PULSE_BOUNDS and run.pulse[0] <= t <= run.pulse[1]
        state = (
            f"Ended at {run.end:.2f} s; last state"
            if ended
            else "Warm-up"
            if t < 0
            else "12 N FORCE PULSE"
            if pulse
            else "Active motion / hold"
        )
        draw.text((left + 14, 405), state, font=small, fill="#a84322" if pulse else DARK)
        draw.text((left + 14, 425), f"Recorded sample: {run.t[i]:.2f} s", font=small, fill=DARK)
        for x, label in ((left + 20, "-0.5"), (left + 145, "0"), (left + 255, "0.5 m")):
            draw.text((x, 382), label, font=small, fill=DARK)
    return image.quantize(colors=128, method=Image.Quantize.MEDIANCUT)


def export_case(report: Path, output: Path, spec: Illustration, *, stop_s: float = 30.0) -> dict[str, object]:
    """Export one named case; refuse to overwrite any earlier export."""
    data = load_animation(report / "assets" / f"{spec.slug}.js")
    output.mkdir(parents=True, exist_ok=False)
    times = np.unique(np.append(np.arange(-data.activation, stop_s, FRAME_MS / 1000 * SPEED), stop_s))
    # Decimal rounding avoids selecting the sample before an exactly represented event boundary.
    frames = [render_frame(data, round(float(t), 6)) for t in times]
    frames[0].save(
        output / "animation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[max(10, round(float(dt) / SPEED * 100) * 10) for dt in np.diff(times)] + [1000],
        loop=0,
        optimize=False,
        disposal=1,
    )
    for source, target in (("space", "task-space.png"), ("time", "time-series.png"), ("joints", "joint-angles.png")):
        shutil.copyfile(report / "assets" / f"{spec.slug}-{source}.png", output / target)
    info: dict[str, object] = {
        "slug": spec.slug,
        "case_id": spec.case_id,
        "title": spec.title,
        "caption": spec.caption,
        "playback_speed": SPEED,
        "frame_duration_ms": FRAME_MS,
        "final_hold_ms": 1000,
        "start_s": -data.activation,
        "stop_s": stop_s,
        "frames": len(frames),
        "source_animation_sha256": sha256_file(report / "assets" / f"{spec.slug}.js"),
        "runs": [
            {"role": r.role, "success": r.success, "reason": r.reason, "end_s": r.end, "pulse_s": list(r.pulse)}
            for r in data.runs
        ],
        "outputs": {p.name: sha256_file(p) for p in sorted(output.iterdir())},
    }
    (output / "case.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# {spec.title}\n\nCase: `{spec.case_id}`\n\n{spec.caption}\n\n"
        "animation.gif: all five arms, synchronized at 2x speed. PASS/FAIL is the final run verdict.\n"
        "task-space.png: measured and desired endpoint paths.\n"
        "time-series.png: error, speed, tracking and effort diagnostics.\n"
        "joint-angles.png: joint trajectories and endpoint error.\n\n"
        "Aborted runs stay grey at their last recorded state. Time is relative to activation.\n"
        "The final frame is held for one playback second before looping.\n"
    )
    return info


def export_bundle(report: Path, output: Path) -> None:
    """Verify the report, export eight case folders and three overview plots, and create a ZIP."""
    archive = Path(str(output) + ".zip")
    if archive.exists():
        raise FileExistsError(archive)
    verify_report(report)
    output.mkdir(parents=True, exist_ok=False)
    rows: list[str] = []
    for index, spec in enumerate(CASES, 1):
        name = f"{index:02d}_{spec.slug}"
        print(f"Exporting {name}", flush=True)
        export_case(report, output / name, spec)
        rows.append(f"| [{name}]({name}/README.md) | {spec.title} | `{spec.case_id}` |")
    overview = output / "00_overview"
    overview.mkdir()
    for name in ("demonstrations.png", "success_counts.png", "class_comparisons.png"):
        shutil.copyfile(report / "assets" / name, overview / name)
    (output / "README.md").write_text(
        "# Expert-report media\n\n"
        "Each numbered case contains `animation.gif`, `task-space.png`, `time-series.png`, "
        "`joint-angles.png`, a caption and source metadata. GIFs are 1500 x 450 pixels, 20 fps, "
        "labelled 2x playback; the 30 s active horizon and warm-up are retained.\n\n"
        "| Folder | Illustration | Source case |\n| --- | --- | --- |\n" + "\n".join(rows) + "\n\n"
        "Overview plots are in `00_overview/`. Matching PNGs are unchanged copies from the report.\n"
        "GIFs use its verified display samples (no new simulation or interpolation). "
        "Pulse times are per run; aborted trajectories stop. All examples use parent D01.\n\n"
        "Reproduce from the repository root:\n\n```sh\nuv run --locked python "
        "scripts/export_manual_expert_media.py --output /tmp/manual-expert-media\n```\n"
    )
    commit, dirty = worktree_state(repository_root())
    manifest = {
        "project_commit": commit,
        "project_dirty": dirty,
        "report_manifest_sha256": sha256_file(report / "manifest.json"),
        "renderer_sha256": sha256_file(Path(__file__)),
        "font_sha256": sha256_file(FONT),
        "outputs": {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(archive, mode="x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=path.relative_to(output.parent))
    print(f"Exported {output}\nArchive: {archive}")


def main() -> None:
    """Export the local verified presentation without requiring the external evidence store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=repository_root() / DOCUMENT_LOCATION / "expert_report")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_bundle(cast("Path", args.report), cast("Path", args.output))


if __name__ == "__main__":
    main()
