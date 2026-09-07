---
name: pin-advance
description: Advance a third_party submodule pin (rclib, skelarm, rtctrl) to a new upstream commit with the commit ordering, build-identity rebuild, and gate this repository requires. User-invoked as /pin-advance <submodule> <commit>.
disable-model-invocation: true
---

# Submodule pin advance

Arguments: `$ARGUMENTS` = `<submodule> <commit>` where `<submodule>` is `rclib`,
`skelarm`, or `rtctrl` and `<commit>` is the upstream commit to pin (use the full
SHA in the commit message). Work from the repository root with absolute paths;
never `cd`.

## 0. Preconditions

- The upstream change is already merged in the owning library (docs/PLAN.md
  section 15: generic fixes go there first, this repo advances the pin in a
  separate integration commit). Editing `third_party/` in place is never the fix.
- `git status --porcelain` is empty. Do not mix the pin advance with task work.
- Note the current pin: `git -C third_party/<name> rev-parse HEAD`.

## 1. Check out the new pin

```bash
git -C third_party/<name> fetch origin
git -C third_party/<name> checkout <commit>
```

For `rclib` also refresh its nested submodules:
`git submodule update --init --recursive third_party/rclib`. If Eigen on
gitlab.com refuses the clone, point that one submodule at the mirror and retry:

```bash
git -C third_party/rclib config submodule.cpp_core/third_party/eigen.url \
  https://github.com/eigen-mirror/eigen.git
git submodule update --init --recursive third_party/rclib
```

Review the delta and record whether `src/` changed (it decides whether any
recorded result is affected): `git -C third_party/<name> log --oneline <old>..<new>`
and `git -C third_party/<name> diff --stat <old>..<new> -- src`.

## 2. Commit the gitlink BEFORE any gate

The deps check compares the committed gitlink with the checkout, so an
uncommitted pin fails the gate. Write the message to a scratch file, then:

```bash
git add third_party/<name>
git commit -F "$SCRATCH/pin.msg"
git log -1 --stat
```

Message form: `build(deps): advance <name> pin to <short sha>` with a body naming
the upstream change (PR number or commit), whether `src/` changed, and a
`Tasks:` line with the ledger ID. No session links.

## 3. Rebuild and gate

```bash
uv run --locked python -m arm_rc_ctrl.dependencies rebuild
```

Then run the full gate as in the `gate` skill (`uv run --locked nox`, then
`uv run --locked nox -s pre_commit`), capturing exit codes. `nox -s deps` must
pass against the new pin.

## 4. Consequences in this repository

- If the gate needs adaptations here, make them in a separate commit after the
  pin commit; never amend behavior changes into the pin commit.
- `scripts/reproduce_1a.py` refuses to run on pins that differ from the recorded
  evidence; that is correct. If `src/` changed upstream, list the recorded results
  that could be affected and ask the owner before regenerating any evidence
  (`--from-evidence` reproduces at the evidence's own commits).
- Update the ledger task's status and evidence in the same commit as the work it
  documents.

## 5. Never

- Leave a submodule on an upstream branch, or run tests while one is.
- Push. Report the commits and let the owner decide.

## Upstream note

`skelarm`'s origin is HTTPS. Pushing an upstream branch from the submodule uses an
inline SSH URL (`ssh://git@github.com/hrshtst/skelarm.git`), which has no tracking
ref, so `--force-with-lease` needs an explicit expected SHA.
