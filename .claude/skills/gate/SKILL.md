---
name: gate
description: Run this repository's full quality gate (nox deps, lint, type_check, tests, cpp, then pre-commit) and triage each failure by class. Use before marking a docs/TASKS.md task DONE, before any commit touching src/, tests/, cpp/, or a submodule pin, or when the user says "run the gate", "run nox", or "will CI pass".
---

# Full quality gate

Everything runs from the repository root with absolute paths (never `cd`) through
the locked environment. Read CLAUDE.md's "Gate and environment" section first.

## 1. Preconditions

- `.venv` exists. If not: `uv sync --locked`, then
  `uv run --locked python -m arm_rc_ctrl.dependencies rebuild`.
- Submodules are clean and on their recorded pins: `git submodule status` shows no
  `+` or `-` prefix and `git status --porcelain third_party/` is empty. A `+`
  means the checkout differs from the committed gitlink: commit the pin advance
  first, never run the gate on top of it.
- No submodule is checked out on an upstream branch.

## 2. Run

Capture exit codes explicitly; a pipe hides them.

```bash
set -o pipefail
uv run --locked nox > "$SCRATCH/gate.log" 2>&1; echo "nox exit=$?"
tail -n 80 "$SCRATCH/gate.log"
uv run --locked nox -s pre_commit > "$SCRATCH/precommit.log" 2>&1; echo "pre-commit exit=$?"
tail -n 40 "$SCRATCH/precommit.log"
```

Use the session scratchpad directory for the logs. To run one session:
`uv run --locked nox -s tests -- <pytest args>` (arguments are forwarded).
To match CI's second interpreter on request:

```bash
UV_PYTHON=3.13 uv sync --locked
UV_PYTHON=3.13 uv run --locked python -m arm_rc_ctrl.dependencies rebuild
UV_PYTHON=3.13 ARM_RC_CTRL_EXPECTED_PYTHON=3.13 uv run --locked nox
```

then the same three commands with `3.12` to switch back (the switch replaces
`.venv` and its build manifest).

## 3. Triage by failure class

| Symptom | Cause | Fix |
|---|---|---|
| `deps` fails with `BuildIdentityError` (manifest missing, pin moved, dirty submodule, file differs) | installed rclib/skelarm no longer match the pins | commit any pin advance, ensure the submodule is clean, then `uv run --locked python -m arm_rc_ctrl.dependencies rebuild`; rerun |
| "nox must run from the locked project environment" | nox started outside `.venv` or via a nested `uv run` | invoke as `uv run --locked nox`; never nest `uv run` inside a session |
| `ARM_RC_CTRL_EXPECTED_PYTHON=X but this environment is Python Y` | interpreter mismatch | the three-command switch above |
| `lint` fails | ruff check/format | `uv run ruff check --fix --force-exclude <paths>` and `uv run ruff format --force-exclude <paths>`; never touch `tests/fixtures/quality/` |
| `type_check` fails | basedpyright strict | fix the types; a `# pyright: ignore[<rule>]` with a comment is allowed only for third-party typing gaps; remove stale ignores (`reportUnnecessaryTypeIgnoreComment` is an error) |
| `tests` fails on a warning | `filterwarnings = error` | fix the warning's source; a narrowly scoped, commented `filterwarnings` ignore in pyproject only for third-party warnings that cannot be fixed |
| `tests` fails on an `XPASS` | `xfail_strict` | the xfail is stale: remove the marker or restore the condition |
| coverage below 90 % | missing tests | add tests; never lower the threshold or add `omit` entries |
| `test_*_lock.py` regression failure | a generated report or audit was edited by hand or its inputs changed | regenerate with its tool (see README) and re-run; `pytest --update-baselines` only for an intended, reviewed change of committed expectations |
| `cpp` fails | `-Werror`, CMake >= 3.22 and a C++17 compiler required | fix the warning; do not disable `ARM_RC_CTRL_WERROR` |
| `pre_commit` rewrote files | ruff-check `--fix` / ruff-format / hygiene hooks | re-run once; then stage the rewritten files |

## 4. Report

State the exit code of every session and of pre-commit, what failed, and what
you changed. Do not describe the gate as green without the exit codes.
