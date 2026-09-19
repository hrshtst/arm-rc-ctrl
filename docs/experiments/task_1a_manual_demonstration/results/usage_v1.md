# Task 1-a manual-demonstration derived evidence: usage (v1)

How to read, verify and reproduce the machine-readable evidence of M3MAN-010
(plan section 7.1). Field definitions, units and denominators are in
[result schema v3](../result_schema_v3.md); this page explains how the pieces
fit together. [`results_v1.md`](results_v1.md) is the generated index: the
bound inputs, every output with its digest, and the totals.

This is the developer's evidence handoff, not an interpretation. Every count is
descriptive: copies, synthetic episodes, repeated deterministic runs and
overlapping training sets are not independent demonstrations, so no confidence
interval is attached to anything here (plan section 6).

## What is where

| output | record | where | holds |
| --- | --- | --- | --- |
| `results_v1.json` | `ManualResults` | committed | the index: inputs by digest, every output by digest and size, totals, the generating command and its provenance |
| `accounting_v1.json` | `StudyAccounting` | committed | one line per study model (186) and replay bank (60), read from the verified manifests |
| `arm_summary_v1.csv` | `ManualArmSummary` | committed | each arm's successes per configuration, tracker and class, with every model's own count |
| `contrast_summary_v1.csv` | `ManualContrastSummary` | committed | each contrast across the ten parents: all ten differences, their median and range, parents improved, worsened and tied |
| `selections_v1.json` | `ManualSelections` | committed | the frozen representative rule applied to each configuration under each tracker |
| `figure_inputs_v1.json` | `ManualFigureInputs` | committed | for each selected case, the runs and the demonstration a figure draws, bound by digest |
| `runs_v1-<digest>.csv` | `ManualRunRow` | store | one row per run the protocol names: 31,980 |
| `contrasts_v1-<digest>.csv` | `ManualContrastRow` | store | one row per contrast, parent, class, configuration and tracker |
| `figures/` | — | committed | one plot and one animation rendered from the inputs to validate the tools |

The two stored tables are named in `results_v1.json` under `tables`, each with
its `armrc://` location, SHA-256 and size. Resolve the location against the
configured storage root (`ARM_RC_CTRL_STORAGE_ROOT` or
`~/.config/arm-rc-ctrl/storage.toml`) and check the digest before reading:

```python
from pathlib import Path

from arm_rc_ctrl.experiments.manual_contrasts import ManualContrastRow
from arm_rc_ctrl.experiments.manual_results import ManualRunRow, load_results, table_from_csv
from arm_rc_ctrl.provenance import verify_artifact
from arm_rc_ctrl.storage import open_storage

results = load_results(Path("docs/experiments/task_1a_manual_demonstration/results/results_v1.json"))
store = open_storage()
runs = next(t for t in results.tables if t.record == "ManualRunRow")
rows = table_from_csv(verify_artifact(store, runs.payload).read_text(encoding="utf-8"), ManualRunRow)
```

`verify_artifact` refuses a file whose size or digest differs from the index,
and `table_from_csv` refuses a header that is not the record's fields.

## Reading the CSV tables

- The header is the record's fields in declaration order; the schema defines each.
- An empty cell is an absent value. The schema says what absence means for each
  optional field; an empty string is never written, so an empty cell is never a
  present value.
- Booleans are `true` and `false`. Floats are written in Python's shortest
  round-trip form, so they read back bit for bit.
- Tuples are JSON arrays, and an absent element is `null`. For example
  `differences` in the contrast summary is ten values in parent order, D01 to
  D10, and `per_model` in the arm summary is ten counts in parent order, or one
  for the all-ten arm.

## Runs

`status` separates what ran from what did not. `completed` and `infeasible` are
simulated runs with a verdict (`success`, `reason`) and a payload (`run_uri`,
`run_sha256`, `arrays_sha256`). `unexecuted` would be a pair the sweep recorded as
not run, and `unavailable` a run whose model has no evidence. Neither is ever
counted as a failure, and the complete study contains neither.

Two clocks appear. The run clock starts at the run's first sample, so
`activation_s`, `pulse_start_s` and `dwell_earliest_start_s` are on it. The task
clock is zero at activation, so `time_to_final_dwell_s` and
`departure_latency_s` are on it, and the warm-up is negative task time.

The diagnostic metrics are read from each run's own stored trajectories, over
the active segment from activation to the last sample. That is the segment the
sweep judged its dwell over, so the warm-up never contributes a peak or an
error. The verdicts are the sweep's own; the metrics describe runs and decide
nothing. `departure_latency_s` uses the task's 1 cm dwell radius as the
distance that counts as leaving the start (`departure_radius_m` in the index).

## Comparisons

Every comparison is made within one configuration and one tracker, over the
scenarios of one class (`nominal`, `posture_small`, `posture_large`, `force`,
`combined`) or of all 65 together (`all`), and never pooled across
configurations or trackers. The contrasts are the plan's four arm contrasts,
`M10-S`, `M10-R10`, `C10-R10` and `M10-C10`, then each arm against its parent's
replay: `S-replay`, `R10-replay`, `C10-replay` and `M10-replay`. Each is made per
parent. A parented arm is the one of that parent; the all-ten arm is the same
model in all ten comparisons.

A `ManualContrastRow` gives both measures the owner asked for:

- the paired success-count difference, `successes_a - successes_b`, over the
  `n_shared` scenarios both arms were run on, out of the class's `n_scenarios`;
- the per-scenario tally: `improved` (a succeeded, b failed), `worsened`
  (b succeeded, a failed), `tied_success` and `tied_failure`.

`difference` always equals `improved - worsened`. A row is `complete` when both
arms have every scenario of the class, `partial` when they share only some, and
`unavailable` when they share none.

A `ManualContrastSummary` lists the ten parents' differences over complete
comparisons, with their median, minimum and maximum and the number of parents
improved, worsened and tied. The median is a float, since ten values can have a
half-integer median.

## Class summaries

A `ManualArmSummary` totals one arm's successes over one class. The all-ten arm
is one model, so it is counted once (`n_models` 1), however many comparisons
reuse it. A parented arm has ten models, and each replay bank counts as a model
of the `replay` arm. `n_runs` is the denominator of `successes`. A model
missing any run of the class would be left out of its totals: its existing runs
are counted in `excluded_runs` and its missing ones in `unavailable_runs`. Both
are zero in the complete study.

## Illustrated cases

`selections_v1.json` applies the frozen
[representative-case rule](../representative_rule_v1.md) to each configuration
under each tracker, comparing `S/D01`, `M10`, `R10/D01` and `C10/D01`. It always
shows the nominal case, then the first scenario in frozen order where the
all-ten arm wins, the first where the singleton wins, and the first failure of
any arm. A category the rule found nowhere is listed in `absent`. These cases
illustrate; conclusions come from the whole study.

`figure_inputs_v1.json` binds, for each selected case, the five runs to draw
(the four arms and the replay of D01) and the D01 demonstration. The tools read
the stored payloads, verify them against those digests, and render:

```text
uv run python scripts/render_manual_case.py cases --inputs docs/experiments/task_1a_manual_demonstration/results/figure_inputs_v1.json
uv run python scripts/render_manual_case.py plot --inputs … --case <case id> --out case.png
uv run python scripts/render_manual_case.py animate --inputs … --case <case id> --role M10 --out case.gif [--fps 10] [--stride 10]
```

A plot draws endpoint paths, distance to the target and both joints. The
recorded demonstration is black and dotted, on its own recording clock. The
commanded reference (the RC readout's generated reference, or the replayed
recording) is dashed, and the actual motion is solid. The warm-up is shaded
grey and pulse windows are shaded in each run's colour. An animation shows one
run's actual arm solid and its commanded arm translucent, over the
demonstration's endpoint path, with the task time and phase printed.
`figures/` holds one plot and one animation rendered this way to validate the
tools: `feasible-best__pd_v2__nominal.png` and
`feasible-best__pd_v2__nominal__M10.gif`, the first case the rule chose as
`all_ten_wins`. The animation shows one frame per 0.5 s (`--stride 50 --fps 2`),
and exported animations use a 32-colour palette, so the committed file stays
small. In that case the R10/D01 trajectory stays within 1.4e-11 rad of S/D01's,
and the legend says so, since the two curves cannot be told apart by eye.
Presentation figures belong to the reporting task (M3MAN-012).

## Reproducing the outputs

From a clean checkout with the store configured, pinned to the canonical
environment like every other evidence step:

```text
uv run python -m arm_rc_ctrl.execution run --policy p-cores -- \
  uv run python scripts/derive_manual_results.py derive \
    --study docs/experiments/task_1a_manual_demonstration/study_manifest_v1.json \
    --evaluation configs/evaluations/task_1a_manual_dev_v1.toml \
    --evidence-dir docs/experiments/task_1a_manual_demonstration/evidence \
    --run-ordering docs/experiments/task_1a_manual_demonstration/run_ordering_v1.json \
    --representative-rule docs/experiments/task_1a_manual_demonstration/representative_rule_v1.json \
    --result-schema docs/experiments/task_1a_manual_demonstration/result_schema_v3.json \
    --output <new directory> --workers 8
```

The command starts the way the sweep does: in the canonical pinned environment,
from the frozen study with its demonstrations verified, and refusing a dirty
worktree unless `--exploratory` is given. Every model and bank manifest is
verified by digest and then checked against the study's trusted inputs with the
same functions a resume uses. A model's fit binding is compared whole with the
fit recorded under the study's own fit identity, whose recipe and weights
digests are verified and whose recipe construction is checked against the study
entry; the derivation does not refit, which the sweep did. Each manifest's
conditions are compared whole with those the evaluation configuration produces
for its configuration, and its runs' training sources with the parent's
demonstrations. A bank is accepted only under the identity its trusted
conditions and parent produce. Every run's summary and arrays are verified
against their recorded digests before anything is measured, and no output is
ever overwritten. As of the second review of M3MAN-010, the committed v1 outputs
are reproduced exactly by the checked derivation, except for the provenance of
the invocation. The stored tables are content-addressed, so re-deriving the same
evidence writes the same bytes to the same locations. The exact command that
produced these outputs is recorded in `results_v1.json`.

Recomputing the metrics independently and re-simulating the frozen
300-run subset are the clean-checkout audit of M3MAN-011, not part of this
derivation.
