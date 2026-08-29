# Repository Guidelines

## Project Structure & Module Organization

The roadmap is `docs/PLAN.md`; `docs/TASKS.md` is the authoritative work queue.
Follow them when adding the planned layout:

- `src/arm_rc_ctrl/`: Python data, RC, control, metrics, and experiments.
- `tests/{unit,integration,regression}/`: tests and deterministic fixtures.
- `cpp/{include,src,apps,tests}/`: C++ inference and `rtctrl` integration.
- `configs/`: versioned robot, task, controller, study, and evaluation TOML.
- `scripts/`: thin reproducibility entry points; keep business logic in `src/`.
- `data/`: DVC-managed inputs; generated runs belong in ignored `artifacts/`.
- `third_party/`: pinned recursive submodules for `rclib`, `skelarm`, and `rtctrl`.

## Build, Test, and Development Commands

Tooling is introduced by milestone M0. Once its configuration exists, use:

```bash
git submodule update --init --recursive  # obtain pinned domain libraries
uv sync                                  # create the Python environment
uv run ruff check .                      # lint Python
uv run pytest                            # run Python tests
cmake -S cpp -B build && cmake --build build
ctest --test-dir build --output-on-failure
```

Before M0 lands, validate documentation with `git diff --check`.

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
data/control paths; use small deterministic regression fixtures. Every bug fix
requires a reproducing test. Hardware tests are supervised and must first pass
simulation/emulator gates. Justify numerical tolerances; retain failed runs.

## Commits & Pull Requests

Use the established Conventional Commit form, for example
`docs: add implementation plan and task ledger` or `feat(rc): add ESN priming`.
Keep commits reviewable and reference `docs/TASKS.md` IDs in the body. Update task
status and evidence in the same commit as implementation.

PRs should explain scope, linked task/issue, test results, config/schema changes,
reproducibility artifacts, and limitations. Include plots for result changes.
Make generic dependency fixes in the owning project, then update its pin
separately.

## Safety & Reproducibility

Never bypass `rtctrl` limits, watchdogs, or abort paths. Do not operate hardware
without the approved checklist and a human-accessible power cutoff. Results must
record Git/submodule revisions, resolved config, DVC hashes, seeds, environment,
raw metrics, and dirty-worktree state.
