<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Reproducing the human-readable report's figures

The [report](report.md) is authored interpretation. Its figures are generated
from the existing audited study; none of these commands trains or simulates
anything, and none writes into the canonical store.

From the repository root, in the existing locked environment:

```bash
uv run --locked python scripts/render_manual_report.py \
  --output /tmp/manual-report-render --trajectories
```

Use a **new output directory** for each render. The command refuses an existing
directory so an earlier export is not silently replaced. The destination is
not recorded as an absolute machine path. The default `--docs` points at this
experiment. External payloads resolve through the project's configured storage
root, as described in the [evidence usage guide](../results/usage_v1.md).

Omit `--trajectories` to reproduce only the two summary figures without the
external store:

```bash
uv run --locked python scripts/render_manual_report.py \
  --output /tmp/manual-report-summaries
```

The thin [entry point](../../../../scripts/render_manual_report.py) calls
[`manual_report.py`](../../../../src/arm_rc_ctrl/experiments/manual_report.py).
The loader checks the study, result index and plotted summary tables against
the fingerprints in audit v4. Teacher payloads and illustrated runs use the
existing verified readers. The renderer's `render_manifest.json` records the
input hashes, its source hash, Git revision/dirty state, case choices and every
output's SHA-256/size. Presentation was generated in an uncommitted reporting
worktree; this is explicitly recorded and does not change the clean producing
revisions bound into the simulation and audit evidence. Plotting introduces no
random seed, refit, resampling of evaluation cases or new experiment identity.

| File | Source and interpretation |
| --- | --- |
| `success_counts.png` | `arm_summary_v1.csv`, class `all`; ten parent points for S/C10/replay and one M10 point. Configurations and trackers remain separate. R10 is omitted visually because it has S's counts. |
| `class_comparisons.png` | `contrast_summary_v1.csv`, M10−S and C10−R10; each cell displays the median of ten parent differences. Color scales by that class's case count, never by a pooled denominator. |
| `demonstrations.png` | All ten processed datasets named by the frozen study, with each take on its own time axis; no time warping. |
| `ten_helps.png` | Frozen case `feasible-best__pd_v2__nominal`, all four learned arms and D01 replay. |
| `ten_hurts.png` | Frozen case `feasible-middle__pd_v2__posture-small-20261201-01`, the same five roles. |
| `ten_hurts.gif` | M10 in the preceding case; one frame per 100 control samples, 1 frame/s. With a 100 Hz grid this plays at nominal elapsed-time speed; this zero-warm-up case includes the 30 s last sample. |

The two case plots and animation call the existing
[`manual_figures.py`](../../../../src/arm_rc_ctrl/experiments/manual_figures.py)
renderer rather than reimplementing run loading or geometry. To render other
cases from the already frozen selection, use the existing entry point:

```bash
uv run --locked python scripts/render_manual_case.py cases \
  --inputs docs/experiments/task_1a_manual_demonstration/results/figure_inputs_v1.json

uv run --locked python scripts/render_manual_case.py animate \
  --inputs docs/experiments/task_1a_manual_demonstration/results/figure_inputs_v1.json \
  --case feasible-middle__pd_v2__posture-small-20261201-01 \
  --role S --out /tmp/manual-singleton.gif --fps 1 --stride 100
```

The successful A/M10 animation linked in the report is an existing M3MAN-010
validation asset. Its retained source command uses the same case renderer,
`--case feasible-best__pd_v2__nominal --role M10 --fps 2 --stride 50`.
No animation is evidence for a new run.

## Numerical claims and their sources

- The main report table reads `M10-S`, class `all`, from
  [contrast summaries](../results/contrast_summary_v1.csv), with singleton
  medians/ranges calculated from `per_model` in
  [arm summaries](../results/arm_summary_v1.csv).
- The all-ten versus copies, synthetic versus copies and all-ten versus
  synthetic contrasts use `M10-R10`, `C10-R10` and `M10-C10` respectively.
  All ten parent differences remain available in the `differences` column.
- The per-scenario improved/worsened/tied examples come from the stored
  `ManualContrastRow` table named in [results v1](../results/results_v1.json).
  They include gains and losses separately; a zero net difference need not
  mean unchanged scenario outcomes.
- Example endpoint errors, final-dwell times and failure reasons come from
  the stored `ManualRunRow` table in that same index, matched by
  `(model_label, scenario_id, tracker)`. Metres are converted to millimetres
  only for prose. The `source`, `arm_kind`, `parent` and `fit_identity` fields
  prevent mixing model runs with replay or counting M10 repeatedly.
- The 7,800 singleton/copy matches refer to **statuses for the same
  configuration, parent, scenario and tracker**, not array digests. These
  arrays need not be bitwise equal.
- Both tables are external payloads with committed SHA-256/size references.
  The [usage guide](../results/usage_v1.md#reading-the-csv-tables) gives a
  loader that verifies the payload before parsing it.

These are descriptive comparisons of this fixed panel. There is no statistical
resampling, population uncertainty estimate or post-run redefinition of success.
The report's six-versus-six median statement concerns the 12 configuration /
tracker combinations, not twelve independent experiments.

## Validation

`tests/regression/test_manual_report.py` checks that the M10 point represents
one model, that all ten singleton points remain, that changed summary bytes
and figure inputs are refused, that both summary figures render, and that the command retains
input/output bindings and refuses to overwrite an export. The figures and
animation were also visually inspected. The earlier independent review and
[audit v4](../audit/reproduction_audit_v4.md) verify the underlying experimental
evidence. Regenerating presentation assets does not require repeating the
31,980-run study or its 300-run re-simulation audit.
