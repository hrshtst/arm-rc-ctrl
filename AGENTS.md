# Repository Guidelines

## Project Structure & Module Organization

The roadmap is `docs/PLAN.md`; `docs/TASKS.md` is the authoritative work queue.
Follow them when adding to the layout. `scripts/` holds thin reproducibility
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
Keep commits reviewable and reference `docs/TASKS.md` IDs in the body. Update task
status and evidence in the same commit as implementation.

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
