# Repository Guidelines

## Project Structure & Module Organization

Start with `docs/PLAN.md` (short roadmap) and `docs/TASKS.md` (work-queue entry
point). Then read only the chosen task's linked experiment plan and relevant
sections of `docs/design/`; do not load every specification or archive by default.

`docs/tasks/README.md` locates older task IDs. Active tasks have their canonical
row in `docs/TASKS.md`; future gated epics live in `docs/tasks/backlog/`, and
completed or historically deferred records in `docs/tasks/archive/`. Each task
has one authoritative status/acceptance row. Search the exact row with `rg`
when prior evidence is needed. Archived BLOCKED/TODO records do not authorize
execution, and moving a row must not change its status or evidence.

Keep `PLAN.md` below 8 KiB and `TASKS.md` below 15 KiB. Keep active rows to a few
sentences plus evidence links; put detailed review rounds beside the experiment.
On completion move the full row to its milestone archive and leave a short
closure link in the queue. If reopened, move the row back rather than copying
its status. See `docs/design/workflow.md` for the full workflow and definition
of done. Preserve task IDs and update relative links when moving documentation.

Follow this layout when adding files. `scripts/` holds thin reproducibility
entry points only (business logic lives in `src/`); `data/` holds Git-tracked
pointer records only (payloads use external storage).

## Build, Test, and Development Commands

Setup from a clean checkout and the nox quality gate (`uv run --locked nox`;
sessions `deps`, `lint`, `type_check`, `tests`, `cpp`, `pre_commit`) are
documented in `README.md`, together with the external storage root and the
smoke experiment. After advancing a submodule pin run
`uv run python -m arm_rc_ctrl.dependencies rebuild` again.

## Coding Style & Naming Conventions

Python 3.12+, type annotations checked by basedpyright in strict mode, NumPy
`float64`, and Ruff. C++ targets C++17 and follows `rtctrl`/`rclib`. TOML uses
lowercase `snake_case`. Reject invalid data and unknown keys instead of silently
correcting them.

## Licensing

Original work is `GPL-3.0-only`; add SPDX headers. Preserve dependency notices
and review `THIRD_PARTY_NOTICES.md` before redistribution.

## Testing Guidelines

Develop test-first. Name Python tests `test_<behavior>.py` and C++ tests
`<behavior>_test.cpp`. Unit-test math and validation; integration-test complete
data/control paths; use deterministic regression fixtures. Bug fixes require
reproducing tests. Hardware tests are supervised and must first pass
simulation/emulator gates. Justify numerical tolerances; retain failed runs.

## Commits & Pull Requests

Use the established Conventional Commit form, for example
`docs: add implementation plan and task ledger` or `feat(rc): add ESN priming`.
Keep commits reviewable and reference stable task IDs in a `Tasks:` trailer.
Update task status and evidence in the same commit as implementation.

PRs should explain scope, linked task, test results, config/schema changes,
artifacts, and limitations. Include plots for result changes.
Make generic dependency fixes in the owning project, then update its pin
separately.

## Safety & Reproducibility

Never bypass `rtctrl` limits, watchdogs, or abort paths. Do not operate hardware
without the approved checklist and a human-accessible power cutoff. Results must
record Git/submodule revisions, resolved config, DVC hashes, seeds, environment,
raw metrics, and dirty-worktree state. Never commit experimental payloads or
absolute storage paths.
