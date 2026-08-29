# Repository Guidelines

## Project Structure & Module Organization

The roadmap is `docs/PLAN.md`; `docs/TASKS.md` is the authoritative work queue.
Follow them when adding the planned layout:

- `src/arm_rc_ctrl/`: Python data, RC, control, metrics, and experiments.
- `tests/{unit,integration,regression}/`: tests and deterministic fixtures.
- `cpp/{include,src,apps,tests}/`: C++ inference and `rtctrl` integration.
- `configs/`: versioned robot, task, controller, study, and evaluation TOML.
- `scripts/`: thin reproducibility entry points; keep business logic in `src/`.
- `data/`: Git-tracked pointer records only; payloads use external storage.
- `third_party/`: pinned recursive submodules for `rclib`, `skelarm`, and `rtctrl`.

## Build, Test, and Development Commands

```bash
git submodule update --init third_party/skelarm third_party/rtctrl  # top-level pins
git submodule update --init --recursive third_party/rclib            # rclib needs its nested submodules
uv sync                                  # locked Python environment (builds rclib/skelarm)
uv run python -m arm_rc_ctrl.dependencies rebuild  # reinstall from submodules + record build identity
uv run nox                               # full gate: deps, lint, type_check, tests, cpp
uv run nox -s lint                       # ruff check + format check
uv run nox -s type_check                 # basedpyright (strict)
uv run nox -s tests                      # pytest with coverage
uv run nox -s cpp                        # cmake configure/build + ctest (-Werror)
uv run nox -s pre_commit                 # all pre-commit hooks
```

See `README.md` for the equivalent raw commands, the external storage root,
and the smoke experiment. After advancing a submodule pin run
`uv run python -m arm_rc_ctrl.dependencies rebuild` again.

## Coding Style & Naming Conventions

Target Python 3.12, four-space indentation, type annotations, NumPy `float64`,
and Ruff. Use `snake_case` for modules/functions, `PascalCase` for types, and
`UPPER_SNAKE_CASE` for constants. C++ targets C++17 and follows `rtctrl`/`rclib`.
TOML uses lowercase `snake_case`. Reject invalid data and unknown keys instead of
silently correcting them.

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
