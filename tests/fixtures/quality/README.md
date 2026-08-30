# Deliberately invalid quality fixtures

These files intentionally violate the repository's lint, format, type, and
test rules. `tests/unit/test_quality_tools.py` runs each tool on them and
asserts that the tool reports the planted problem, proving the checks are
actually executed. They are excluded from Ruff, basedpyright, pre-commit, and
normal pytest collection. Do not "fix" them.
