# Task 1-a manual-demonstration representative-case rule (v1)

Frozen before execution. It chooses which cases the report ILLUSTRATES; aggregate conclusions come from the whole study, never from these cases. The selections are derived after execution.
Record definitions are not restated here: see result schema v2, bound by digest below (`Selection`, `IllustrationCase`).

| bound input | sha256 |
| --- | --- |
| study manifest | `b07b824c362a3513c03c9c34fa0437b390f47d148ac5a288231fba218aaa36b7` |
| evaluation configuration | `8780be5bac7f4ccb99d6636c947f78901646b886b5f768846b6d9ba5d48acb2b` |
| result schema v2 | `dfd3e29b2f7629253cdf511d1690e53d0b0cf82ba1013628e386ca6a12867b9e` |
| run ordering (tie-breaking order) | `0305be9ead9dcb51d58abdb9951626883a329033fcfb131f91d4df8c6c7bdf4f` |

## Scope

- Comparison arms: `S/D01`, `M10`, `R10/D01`, `C10/D01` (kinds `S`, `M10`, `R10`, `C10`, parent `D01`).
- Configurations: `feasible-best`, `feasible-middle`, `feasible-worst`, `failure-actual-dwell`, `failure-joint-velocity`, `failure-generated-dwell`.
- Trackers: `pd_v2`, `computed_torque`.

## Categories (in declared order)

| category | selects |
| --- | --- |
| `nominal` | the scenario `nominal`, always, so every configuration and tracker shows its nominal comparison |
| `all_ten_wins` | the first scenario in frozen order where the all-ten arm (M10) succeeds and the singleton (S) fails |
| `singleton_wins` | the first scenario in frozen order where the singleton (S) succeeds and the all-ten arm (M10) fails |
| `first_failure` | the first scenario in frozen order where any of the four comparison arms fails |

## Application

Applied separately to each configuration under each tracker. One application's verdicts cover one configuration under one tracker, and each scenario must carry every comparison arm exactly once: verdicts spanning trackers, a repeated arm, a missing arm, or a scenario outside the frozen order are refused rather than merged.

## Tie-breaking

The frozen scenario order of the run ordering this rule binds by digest. A category selects the FIRST qualifying scenario in that order, so it selects at most one.

## Deduplication

A scenario chosen by more than one category is shown once, in frozen scenario order, recording every category that chose it in the declared category order.

## Absent categories

A category that matches no scenario is listed in the selection's `absent`, never silently omitted, so a reader sees that the rule looked and found none.

## Replay baseline

Each selected case shows all four comparison arms beside the replay baseline of the parent under that case's configuration: the bank keyed (configuration, parent) in the run ordering.
