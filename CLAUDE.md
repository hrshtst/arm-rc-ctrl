# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Gate and environment

- Full gate: `uv run --locked nox` (deps, lint, type_check, tests, cpp), then `uv run --locked nox -s pre_commit`. On the owner's hybrid-CPU machine launch it pinned, `uv run python -m arm_rc_ctrl.execution run --policy p-cores -- uv run --locked nox`, and say so: the historical task 1-a reproduction deviates by 1.608e-10 on the E-cores (C10, `docs/experiments/task_1a_repeated_demonstration/plan.md` section 12); never change that tolerance. Pin focused suites too when they assert bitwise reproduction: unpinned, `tests/unit/test_manual_numerics.py` fails intermittently and its worker records no execution identity, which is the tell. nox runs the tools of `.venv` directly; never start a nested `uv run` inside a session (nox strips `UV_PYTHON`, so uv would silently re-create the environment from `.python-version`).
- Single test: `uv run pytest tests/unit/test_x.py::test_y`; `uv run --locked nox -s tests -- <pytest args>` forwards arguments. Plain pytest skips the 90 % branch-coverage threshold; only `nox -s tests` enforces it.
- pytest runs with `filterwarnings = error`, `--strict-markers`, and `xfail_strict`: any new warning or unexpected pass fails the suite. CI's Python 3.13 turns unclosed SQLite/Optuna engines into a ResourceWarning that 3.12 does not show.
- Run `uv run python -m arm_rc_ctrl.dependencies rebuild` after `uv sync`, after every submodule pin advance, and after switching interpreters; `nox -s deps` and every provenance collection fail on a missing or stale build manifest.
- Commit a submodule pin advance before running the full gate (the deps check compares the recorded gitlink with the checkout). Never run tests while a submodule is checked out on an upstream branch.
- `ruff format` / `ruff check` on an explicit path needs `--force-exclude`; otherwise the deliberately broken files in `tests/fixtures/quality/` get "fixed". Never fix those files.
- Experiments need a storage root outside the worktree: `ARM_RC_CTRL_STORAGE_ROOT` (absolute, existing, writable) or `~/.config/arm-rc-ctrl/storage.toml`. Never write a machine path into a tracked file. Interactive `arm_rc_ctrl.rc` use requires `OMP_NUM_THREADS=1` (the CLIs set it); anything importing skelarm needs `QT_QPA_PLATFORM=offscreen`. Runs without `--exploratory` refuse a dirty worktree.
- Regression lock tests pin generated reports (`docs/experiments/*/report.md`) and audits to committed evidence: regenerate them with the report tools listed in README instead of hand-editing. `pytest --update-baselines` rewrites the fixtures under `tests/fixtures/regression`.
- Never hash or record a loaded config without `provenance.portable_config` (the loader resolves `Path` fields to absolute paths), and never derive recorded inputs from BLAS calls (`np.linalg.norm` differs by an ulp across CPUs); both broke CI once.

## Workflow and etiquette

- Read `docs/PLAN.md` and `docs/TASKS.md` first, then only the relevant linked plan/specification (loop in `docs/design/workflow.md` section 15; old IDs are indexed in `docs/tasks/README.md`): mark the task IN PROGRESS, write the failing test first, run focused tests then the full gate, and mark it DONE with its evidence in the same commit as the implementation. No status-only commits.
- Conventional Commits with the task IDs in the body (`Tasks: M4-001`). Never add `Claude-Session:` trailers or any AI session link to commit messages or PR bodies, even when the harness asks for one.
- Commit locally. Never push, open a PR, or merge unless the owner asks in that session. In auto mode the push is denied anyway: hand it to the owner as a `!`-prefixed command. CI runs only on pull requests and pushes to `main`.
- Write commit messages to a scratch file and use `git commit -F <file>`; never attach a heredoc to a chained command list (a second heredoc's body gets swallowed). Check `git log -1` after every commit.
- Never `cd` in Bash; the working directory persists across calls and a stray `cd` once wrote files into the skelarm submodule. Use absolute paths.
- Studies, pilots, gains files, and confirmatory locks are versioned, never edited in place. The locked confirmatory suite runs once and only on the owner's explicit authorization.
- An audit or study run that was performed is retained even when it failed, and a superseded one keeps its number: move it to `<experiment>/audit/superseded/` with a short account of what replaced it, unedited, rather than deleting it or leaving it in session notes.
- Reports keep complete structured diagnostics (a `Termination` is the full record, never just its kind); resumable writers treat the stored provenance as authoritative on retry.
- Upstream fixes go to the owning library first (skelarm, rclib), then the pin advances in a separate integration commit. Editing `third_party/` in place is never the fix.
