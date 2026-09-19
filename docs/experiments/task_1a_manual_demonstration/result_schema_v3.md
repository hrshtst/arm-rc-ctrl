# Task 1-a manual-demonstration result schema (v3)

Field definitions and units of the machine-readable evidence `task_1a_manual_v1` writes (plan section 7.1). Generated from the records themselves; a changed field fails the regression lock rather than refreshing this document, so changing one is a version decision.

## `ManualRunConditions`

Everything the revised protocol decides about a run; the identity of these keys every cache.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `evaluation_name` | `str` | — | — | Name the evaluation configuration declares for this protocol. | — |
| `evaluation_file` | `str` | — | — | Path of the evaluation configuration, relative to the repository root. | — |
| `evaluation_sha256` | `str` | — | — | SHA-256 of the evaluation configuration as read. | — |
| `development_file` | `str` | — | — | Path of the locked development-robustness draws. | — |
| `development_sha256` | `str` | — | — | SHA-256 of the locked draws: the scenarios are only these if this file is. | — |
| `scenario_file` | `str` | — | — | Path of the manual task configuration. | — |
| `scenario_sha256` | `str` | — | — | SHA-256 of the task configuration as read. | — |
| `horizon_s` | `float` | s | one run | Evaluation horizon measured from activation; completion is judged against it and never against a demonstration's length. | — |
| `trigger_hold_s` | `float` | s | one run | Continuous target dwell the arm must hold before the force pulse is armed. | — |
| `trigger_duration_s` | `float` | s | one run | Length of the force pulse once it fires. | — |
| `trigger_magnitude_n` | `float` | N | one run | Magnitude of the force pulse. | — |
| `dwell_min_duration_s` | `float` | s | one run | Uninterrupted final dwell a run must achieve to count as completed. | — |
| `dwell_tolerance_m` | `float` | m | one dwell sample | Radius of the target region every dwell sample must lie inside. | — |
| `dwell_max_velocity_rad_s` | `float` | rad/s | one dwell sample, maximum over joints | Maximum absolute joint speed allowed at every dwell sample; the acquisition rule, carried explicitly so a run records the rule it was judged by. | — |
| `replay_velocity_cutoff_hz` | `float` | Hz | one run | Velocity cutoff of the causal derivative estimator replay is driven through: its paired configuration's own, so both arms filter alike. | — |
| `replay_acceleration_cutoff_hz` | `float` | Hz | one run | Acceleration cutoff of that same estimator. | — |
| `velocity_abort` | `tuple[float, ...]` | rad/s | per joint, one sample | Per-joint speed at which a run is aborted as unsafe. | — |
| `trackers` | `dict[str, str]` | — | — | SHA-256 of each frozen tracker's gains, by name. | — |
| `tracker_order` | `tuple[str, ...]` | — | — | The trackers in evaluation order; the JSON form sorts mapping keys, so the order is explicit here. | — |
| `scenario_ids` | `tuple[str, ...]` | — | — | Identifiers of the development cases evaluated, in evaluation order. | — |
| `warmup_s` | `float` | s | one run | Hold at the reset posture before activation; the reference begins after it. | — |
| `execution_identity` | `str` | — | — | Identity of the canonical execution environment every run under these conditions was keyed in. | — |

## `ManualEvidencePointer`

The repository's pointer to one stored manifest: what it is, where it lives, and its digest.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `schema` | `str` | — | — | Schema tag of the pointer record. | — |
| `experiment` | `str` | — | — | Experiment label the evidence belongs to. | — |
| `kind` | `str` | — | — | Either `model` or `replay`: which kind of manifest this points at. | — |
| `identity` | `str` | — | — | Identity of the manifest pointed at. | — |
| `label` | `str` | — | — | Name of the model or replay bank, as the repository files it. | — |
| `status` | `str` | — | — | `feasible` when every pair completed, `infeasible` otherwise. | — |
| `payload` | `ArtifactReference` | — | — | Store reference and digest of the manifest itself. | — |
| `n_pairs` | `int` | pairs | one manifest | Pairs the manifest records. | — |
| `n_completed` | `int` | pairs | one manifest; of n_pairs | Pairs whose run reached the horizon and met every criterion. | — |
| `n_infeasible` | `int` | pairs | one manifest; of n_pairs | Pairs that failed at least one criterion. | — |

## `ManualModelEvidence`

Everything one model produced under one protocol, with its counts re-derived from its runs.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `identity` | `str` | — | — | Key this evidence is stored and served under: its fit and its conditions together. | — |
| `label` | `str` | — | — | `<configuration>/<arm>`, the model this evidence is of. | — |
| `conditions` | `ManualRunConditions` | — | — | The protocol conditions every pair here was run under. | — |
| `fit` | `ManualFitBinding \| None` | — | — | The fit these runs were produced from, bound by identity and digests. | a record naming no fit; every evidence the sweep writes binds one. |
| `assignment` | `str \| None` | — | — | The parent this model is paired against. | the all-ten arm, which trains on the whole bank and has no single parent. |
| `replay_bank` | `str \| None` | — | — | Identity of the replay bank this model's pairs are compared against. | the all-ten arm, which is paired against no single bank. |
| `status` | `str` | — | — | `feasible` when every pair completed, `infeasible` otherwise. | — |
| `pairs` | `tuple[ManualPairRecord, ...]` | — | — | Every scenario and tracker pair of this model, in evaluation order. | — |
| `n_pairs` | `int` | pairs | one model | Pairs recorded: scenarios times trackers. | — |
| `n_completed` | `int` | pairs | one model; of n_pairs | Pairs that met every criterion. | — |
| `n_infeasible` | `int` | pairs | one model; of n_pairs | Pairs that failed at least one criterion. | — |
| `n_unexecuted` | `int` | pairs | one model; of n_pairs | Pairs recorded as not executed. A completed sweep emits zero: every scenario is attempted from a fresh reset (D6), so nothing is skipped because an earlier scenario failed. An interrupted sweep's missing runs are simply absent from the record and are never turned into unexecuted pairs. | — |
| `schema_version` | `int` | version | one record | Version of the evidence schema this record follows. | — |

## `ManualReplayBank`

Every direct-replay baseline of one demonstration under one set of conditions.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `conditions` | `ManualRunConditions` | — | — | The protocol conditions every baseline here was run under. | — |
| `assignment` | `str` | — | — | The demonstration these baselines replay. | — |
| `pairs` | `tuple[ManualPairRecord, ...]` | — | — | Every scenario and tracker pair of this bank, in evaluation order. | — |

## `ManualPairRecord`

One (scenario, tracker) run of one arm, with the posture it started from and how it was judged.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `index` | `int` | index | one model's scenario list | Position of this scenario in the evaluation order. | — |
| `scenario_id` | `str` | — | — | Identifier of the development case. | — |
| `kind` | `str` | — | — | Perturbation family of the case, such as `nominal` or `posture_small`. | — |
| `tracker` | `str` | — | — | Which frozen tracker ran the pair. | — |
| `arm` | `str` | — | — | `rc` for the generated reference, `replay` for the recorded one. | — |
| `status` | `str` | — | — | `completed` when the run met every criterion, `infeasible` otherwise. | — |
| `initial_q` | `tuple[float, ...]` | rad | per joint | Joint angles the run started from, after the case's perturbation. | — |
| `outcome` | `ManualRunOutcome \| None` | — | — | What the run achieved and why it did not succeed when it did not. | a pair that was not simulated; a simulated pair always carries one. |
| `run` | `ManualRunArtifact \| None` | — | — | Where the stored run lives and what it contains. | a pair that was not simulated; a simulated pair always carries one. |
| `pulse_start_s` | `float \| None` | s | one run clock | When the pulse actually fired on the run clock, rather than when the levels prescribed it. | no pulse fired in this run. |

## `ManualRunOutcome`

What one run achieved, and why it did not succeed when it did not.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `completed` | `bool` | — | — | The run reached the configured horizon measured from activation. | — |
| `dwell` | `ManualDwellReport` | — | — | The dwell over the whole active segment: earliest, longest, final, and departures after holds. | — |
| `post_pulse_dwell` | `ManualDwellReport \| None` | — | — | The dwell measured from the pulse end, which is what a force case must satisfy. | no pulse fired, so the whole active segment is judged instead. |
| `generated` | `GeneratedReferenceReport \| None` | — | — | The generated reference judged by the same rules as the actual motion. | a replay run, which carries no readout to judge. |
| `saturation_fraction` | `float` | fraction in [0, 1] | active segment; denominator is its sample count | Share of active-segment samples in which any joint's requested torque reached its limit. | — |
| `torque_rms` | `float \| None` | N m | active segment, over joints and samples | Root mean square of the applied torque over the active segment. | the run aborted before its active segment began, so there is nothing to average. |
| `trigger` | `TriggerOutcome \| None` | — | — | Whether the pulse fired, and early enough to leave room for the dwell it disturbs. | a case that prescribes no pulse. |
| `success` | `bool` | — | — | Every criterion this sweep judges was met. | — |
| `reason` | `str \| None` | — | — | Why the run did not succeed, naming the cause rather than a consequence. | the run succeeded. |

## `ManualRunArtifact`

Where one persisted run lives and what it contains.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `artifact_id` | `str` | — | — | Identity of the stored run. | — |
| `uri` | `str` | — | — | Store location of the run summary. | — |
| `sha256` | `str` | — | — | SHA-256 of the run summary as written. | — |
| `size` | `int` | bytes | one run summary | Size of the run summary. | — |
| `arrays_sha256` | `str` | — | — | SHA-256 of the run's arrays, as the summary records it. | — |
| `sources` | `tuple[str, ...]` | — | — | The demonstrations this run's model trained on: all ten for the all-ten arm, one otherwise. | — |

## `ManualFitBinding`

The fit a model's runs were produced from, bound by the study's own identity and digests.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `identity` | `str` | — | — | Identity of the fit these runs were produced from. | — |
| `configuration` | `str` | — | — | Inherited configuration the fit belongs to. | — |
| `arm` | `str` | — | — | Training arm of the fit. | — |
| `solver_alpha` | `float` | ridge alpha | one readout solve | Ridge parameter the readout was solved at: the configuration's base alpha times the episode count. | — |
| `recipe_sha256` | `str` | — | — | SHA-256 of the recipe as written. | — |
| `weights_sha256` | `str` | — | — | Digest of the fitted readout weights, over dtype, shape and bytes. | — |

## `ManualDwellReport`

What the continuous dwell rule measured on one run (plan section 6's four reported quantities).

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `ok` | `bool` | — | — | The run ending at the last sample satisfies the dwell rule for its full required duration. | — |
| `final_samples` | `int` | samples | the final dwell | Samples of the dwell that ends at the last sample. | — |
| `final_duration_s` | `float` | s | the final dwell | Duration of that final dwell. | — |
| `longest_samples` | `int` | samples | the longest qualifying dwell | Samples of the longest qualifying dwell anywhere in the active segment. | — |
| `longest_duration_s` | `float` | s | the longest qualifying dwell | Duration of that longest dwell. | — |
| `earliest_start_s` | `float \| None` | s | run clock | Start of the first dwell that reached the required length. | no dwell ever reached the required length. |
| `departures_after_hold` | `int` | count | active segment | Qualifying holds that ended before the last sample: the arm reached the target and then left it. | — |

## `GeneratedReferenceReport`

The generated reference judged by the same rules as the actual motion (plan section 6).

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `within_position_limits` | `bool` | — | — | Every generated joint command lay inside the joint limits. | — |
| `within_speed_limits` | `bool` | — | — | Every generated joint speed lay inside the speed limits. | — |
| `within_workspace` | `bool \| None` | — | — | The generated endpoint stayed inside the reachable workspace. | the generated command left the joint limits, so forward kinematics would have answered for a clamped trajectory rather than the commanded one. |
| `dwell` | `ManualDwellReport \| None` | — | — | The dwell the generated reference itself achieves, judged by the rule the actual motion is judged by. | the generated command was not evaluable; see within_position_limits. |

## `TriggerOutcome`

Whether the dwell-triggered pulse fired, and early enough to leave room for the dwell it disturbs.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `ok` | `bool` | — | — | The pulse fired and left room for the dwell it disturbs. | — |
| `triggered` | `bool` | — | — | A pulse fired at all. | — |
| `pulse_start_s` | `float \| None` | s | one run clock | When the pulse began on the run clock. | no pulse fired. |
| `pulse_end_s` | `float \| None` | s | one run clock | When the pulse ended on the run clock. | no pulse fired. |
| `reason` | `str \| None` | — | — | Why the case cannot count as a disturbance-recovery test. | the timing rule was satisfied. |

## `ModelAccount`

One study model's line: what its evidence says, or that there is none.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `label` | `str` | — | — | Model label `<configuration>/<arm>`, as the frozen manifest names it. | — |
| `configuration` | `str` | — | — | Configuration the model belongs to, one of the six frozen. | — |
| `arm` | `str` | — | — | Arm within the configuration, such as `S/D01`, `M10`, `R10/D01` or `C10/D01`. | — |
| `present` | `bool` | — | — | Whether the study holds evidence for this model. False is a recorded absence, not an error. | — |
| `identity` | `str \| None` | — | — | Evidence identity the manifest carries: the digest over the fit identity and the conditions. | The model has no evidence yet; `present` is false and every other stated fact is absent with it. |
| `fit_identity` | `str \| None` | — | — | Cache identity of the fit this model's evidence was produced from. | No evidence exists for this model. |
| `assignment` | `str \| None` | — | — | Parent demonstration the model is paired against, such as `D01`. | Either the model has no evidence, or it is the all-ten arm, which is paired against no single parent. |
| `status` | `str \| None` | — | — | The manifest's verdict for the model: `feasible` when every pair completed, else `infeasible`. | No evidence exists for this model. |
| `n_pairs` | `int` | pairs | one model's line | Pairs the model's evidence records. The denominator of the three counts below, which sum to it. | — |
| `n_completed` | `int` | pairs | one model's line, of `n_pairs` | Pairs that completed and were judged successful. | — |
| `n_infeasible` | `int` | pairs | one model's line, of `n_pairs` | Pairs that ran and were judged infeasible, by the horizon, dwell, abort or saturation rules. | — |
| `n_unexecuted` | `int` | pairs | one model's line, of `n_pairs` | Pairs the protocol names but the evidence does not record as run. A completed sweep emits zero, because every scenario is attempted from a fresh reset and none is skipped after a failure; an interrupted sweep's missing runs are NOT counted here, because they were never recorded as pairs. | — |
| `execution_identity` | `str \| None` | — | — | Execution environment the model's runs were keyed in. | No evidence exists for this model. |
| `payload` | `ArtifactReference \| None` | — | — | Store reference and digest of the manifest this line was read from. | No evidence exists for this model. |

## `BankAccount`

One replay bank's line, keyed by the parent and the derivative policy it belongs to.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `identity` | `str` | — | — | Bank identity: the digest over the conditions and the parent. | — |
| `assignment` | `str` | — | — | Parent demonstration whose direct replay this bank holds. | — |
| `warmup_s` | `float` | s | one replay bank's line | Warm-up the bank was built under; banks of one parent differ by it. | — |
| `velocity_cutoff_hz` | `float` | Hz | one replay bank's line | Causal velocity cutoff replay was driven through, from the paired configuration's estimator. | — |
| `acceleration_cutoff_hz` | `float` | Hz | one replay bank's line | Causal acceleration cutoff replay was driven through, from the same estimator. | — |
| `n_pairs` | `int` | pairs | one replay bank's line | Replay pairs the bank records. The denominator of the two counts below. | — |
| `n_completed` | `int` | pairs | one replay bank's line, of `n_pairs` | Replay pairs judged successful. | — |
| `n_infeasible` | `int` | pairs | one replay bank's line, of `n_pairs` | Replay pairs judged infeasible. | — |
| `execution_identity` | `str` | — | — | Execution environment the bank's runs were keyed in. | — |
| `payload` | `ArtifactReference` | — | — | Store reference and digest of the bank manifest this line was read from. | — |

## `StudyAccounting`

The accounting of what the study has executed so far, and what it has not.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `experiment` | `str` | — | — | Experiment label the accounting belongs to. | — |
| `canonical_execution_identity` | `str` | — | — | The execution identity every admissible run must be keyed in, for `all_bind_canonical_execution`. | — |
| `models` | `tuple[ModelAccount, ...]` | — | — | One line per study model, present or absent, in the frozen manifest order. | — |
| `banks` | `tuple[BankAccount, ...]` | — | — | One line per replay bank the study produced. | — |
| `n_models` | `int` | models | the study's 186 models | Model lines the accounting carries: the study's full count, not only those with evidence. | — |
| `n_present` | `int` | models | one accounting document, of `n_models` | Models whose evidence exists. | — |
| `n_missing` | `int` | models | one accounting document, of `n_models` | Models with no evidence. Stated rather than omitted, so an incomplete study is legible as incomplete. | — |
| `missing` | `tuple[str, ...]` | — | — | Labels of the models with no evidence, so the gap is named and not merely counted. | — |
| `statuses` | `dict[str, int]` | — | — | Count of present models by their recorded status, such as `feasible` or `infeasible`. | — |
| `n_rc_runs` | `int` | pairs | one accounting document, over present models | Model pairs summed over every present model. | — |
| `n_replay_runs` | `int` | pairs | one accounting document, over banks | Replay pairs summed over every bank. | — |
| `all_bind_canonical_execution` | `bool` | — | — | Whether every present model and bank is keyed in the canonical execution identity. | — |
| `complete` | `bool` | — | — | Whether every study model has evidence. False while any line is absent. | — |
| `provenance` | `ProvenanceRecord` | — | — | Reproducibility record of the invocation that produced the accounting. | — |
| `schema_version` | `int` | version | one accounting document | Version of the accounting record itself; a changed field is a version decision. | — |

## `IllustrationCase`

One scenario the report illustrates, with the categories that chose it.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `scenario_id` | `str` | — | — | Identifier of the scenario the report illustrates. | — |
| `categories` | `tuple[str, ...]` | — | — | Every category that selected this case, in the rule's declared order; a case may satisfy several. | — |
| `arms` | `dict[str, bool]` | — | — | Each comparison arm's verdict on this scenario, so the figure shows all four beside the replay baseline. | — |

## `Selection`

The cases the frozen rule chose, and the categories that found nothing.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `cases` | `tuple[IllustrationCase, ...]` | — | — | The cases the frozen rule chose, deduplicated, in the rule's declared order. | — |
| `categories` | `dict[str, tuple[str, ...]]` | — | — | Scenario ids each category selected, before deduplication across categories. | — |
| `absent` | `tuple[str, ...]` | — | — | Categories that occurred nowhere in the results, stated so a reader sees the rule looked and found none. | — |

## `ResimulationSubset`

The runs a clean-checkout audit re-simulates, named before any of them exist.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `models` | `tuple[StudyModel, ...]` | — | — | The models the audit re-simulates, in the frozen manifest order. | — |
| `scenarios` | `tuple[RobustnessScenario, ...]` | — | — | One locked scenario from each perturbation class, in the frozen class order. | — |
| `trackers` | `tuple[str, ...]` | — | — | The trackers each model and scenario is run under, in evaluation order. | — |
| `n_banks` | `int` | banks | the re-simulation sample | Replay banks the chosen models share: one per configuration at the fixed parent. | — |

## `ManualRunRow`

One run of the study: what ran, how it was judged, where it stopped, what it measured, where it lives.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `source` | `str` | — | — | `rc` for a model's run, `replay` for a direct replay of a demonstration. | — |
| `configuration` | `str` | — | — | Inherited configuration the run belongs to, one of the six frozen. | — |
| `arm` | `str` | — | — | Arm as the study names it: `M10`, `S/D01`, `R10/D01`, `C10/D01`, or `replay/D01`. | — |
| `arm_kind` | `str` | — | — | Arm without its parent: `S`, `M10`, `R10`, `C10` or `replay`. | — |
| `parent` | `str \| None` | — | — | Demonstration the arm belongs to. | the all-ten arm, which trains on the whole bank and has no single parent. |
| `model_label` | `str` | — | — | `<configuration>/<arm>`, the model or replay bank the run is of. | — |
| `evidence_identity` | `str \| None` | — | — | Identity of the model evidence or replay bank manifest the run was read from. | the model has no evidence, so no manifest describes the run. |
| `tracker` | `str` | — | — | Frozen tracker that ran the pair. | — |
| `scenario_id` | `str` | — | — | Identifier of the development case. | — |
| `scenario_class` | `str` | — | — | Perturbation class of the case: `nominal`, `posture_small`, `posture_large`, `force` or `combined`. | — |
| `scenario_index` | `int` | index | the 65 locked cases | Position of the case in the frozen scenario order. | — |
| `status` | `str` | — | — | `completed` (met every criterion), `infeasible` (failed at least one), `unexecuted` (recorded as not run) or `unavailable` (the model has no evidence). Only the first two are runs. | — |
| `success` | `bool \| None` | — | — | The run met every criterion the sweep judges. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `reason` | `str \| None` | — | — | Why the run did not succeed, naming the cause rather than a consequence. | the run succeeded, or it was not simulated. |
| `initial_q` | `tuple[float, ...] \| None` | — | — | Joint angles the run started from, after the case's perturbation, one per joint (rad). | the model has no evidence, so no pair records the start. |
| `horizon_completed` | `bool \| None` | — | — | The run reached the configured horizon measured from activation. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `termination_kind` | `str \| None` | — | — | How the run ended, from the run's own termination record. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `termination_time_s` | `float \| None` | s | one run's clock, zero at its first sample | When the run ended. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `termination_limit` | `str \| None` | — | — | The limit an abort tripped. | the run did not abort on a limit, or it was not simulated. |
| `termination_joint` | `int \| None` | index | joints of the arm, from 0 | The joint whose limit tripped. | no joint limit tripped, or the run was not simulated. |
| `termination_value` | `float \| None` | the limit's unit: rad, rad/s or N m | the tripping sample | The value that crossed the limit. | no limit tripped, or the run was not simulated. |
| `termination_bound` | `float \| None` | the limit's unit: rad, rad/s or N m | the tripped limit | The bound the value crossed. | no limit tripped, or the run was not simulated. |
| `termination_failure` | `str \| None` | — | — | The failure kind the termination recorded. | the run ended without a failure, or it was not simulated. |
| `termination_detail` | `str \| None` | — | — | The termination record's free-text detail. | the record carries no detail, or the run was not simulated. |
| `dwell_ok` | `bool \| None` | — | — | The run ends with a qualifying dwell of the required length. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `dwell_final_s` | `float \| None` | s | the final dwell | Duration of the dwell that ends at the last sample. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `dwell_longest_s` | `float \| None` | s | one run's active segment, activation to last sample | Duration of the longest qualifying dwell. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `dwell_earliest_start_s` | `float \| None` | s | one run's clock, zero at its first sample | Start of the first dwell that reached the required length. | no dwell reached the required length, or the run was not simulated. |
| `departures_after_hold` | `int \| None` | count | one run's active segment, activation to last sample | Qualifying holds that ended before the last sample: the arm reached the target and left it. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `post_pulse_dwell_ok` | `bool \| None` | — | — | The dwell measured from the pulse end satisfies the rule; what a force case must meet. | no pulse fired, so the whole active segment is judged, or the run was not simulated. |
| `post_pulse_dwell_final_s` | `float \| None` | s | the final dwell after the pulse | Duration of the final dwell measured from the pulse end. | no pulse fired, or the run was not simulated. |
| `trigger_ok` | `bool \| None` | — | — | The pulse fired and left room for the dwell it disturbs. | the case prescribes no pulse, or the run was not simulated. |
| `pulse_triggered` | `bool \| None` | — | — | A pulse fired at all. | the case prescribes no pulse, or the run was not simulated. |
| `pulse_start_s` | `float \| None` | s | one run's clock, zero at its first sample | When the pulse began. | no pulse fired, or the run was not simulated. |
| `pulse_end_s` | `float \| None` | s | one run's clock, zero at its first sample | When the pulse ended. | no pulse fired, or the run was not simulated. |
| `trigger_reason` | `str \| None` | — | — | Why the case cannot count as a disturbance-recovery test. | the timing rule was met, the case prescribes no pulse, or the run was not simulated. |
| `generated_within_position_limits` | `bool \| None` | — | — | Every generated joint command lay inside the joint limits. | a replay run, which carries no readout, or the run was not simulated. |
| `generated_within_speed_limits` | `bool \| None` | — | — | Every generated joint speed lay inside the speed limits. | a replay run, or the run was not simulated. |
| `generated_within_workspace` | `bool \| None` | — | — | The generated endpoint stayed inside the reachable workspace. | a replay run, a generated command outside the joint limits, or a run that was not simulated. |
| `generated_dwell_ok` | `bool \| None` | — | — | The generated reference itself ends with a qualifying dwell. | a replay run, a generated command that was not evaluable, or a run that was not simulated. |
| `generated_dwell_final_s` | `float \| None` | s | the generated reference's final dwell | Duration of the generated reference's final dwell. | a replay run, a generated command that was not evaluable, or a run that was not simulated. |
| `saturation_fraction` | `float \| None` | fraction in [0, 1] | one run's active segment, activation to last sample; denominator is its sample count | Share of active samples in which any joint's requested torque reached its limit. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `torque_rms_nm` | `float \| None` | N m | one run's active segment, activation to last sample | Root mean square over active samples of the applied torque vector's Euclidean norm, joints combined, as the sweep recorded it; it can exceed any single joint's limit. | the run was not simulated, or it aborted before activation and has no active segment. |
| `activation_s` | `float \| None` | s | one run's clock, zero at its first sample | When the task activated: the end of the warm-up. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `duration_s` | `float \| None` | s | one run's clock, zero at its first sample | Length of the run as recorded. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `n_samples` | `int \| None` | samples | one run | Samples the run recorded, warm-up included. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `n_active_samples` | `int \| None` | samples | one run's active segment, activation to last sample | Samples from activation to the last sample. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `final_endpoint_error_m` | `float \| None` | m | the last sample | Distance from the endpoint to the target at the last sample. | the run was not simulated, or it aborted before activation and has no active segment. |
| `time_to_final_dwell_s` | `float \| None` | s | one run's task clock, zero at activation | Time from activation to the start of the dwell that ends the run, when that dwell is long enough. | the run's dwell failed, or the run was not simulated. |
| `departure_latency_s` | `float \| None` | s | one run's task clock, zero at activation | Time from activation to the first sample whose endpoint lies farther than the departure radius (the task's 1 cm dwell radius) from its position at activation. | the endpoint never left the departure radius, or the run has no active segment. |
| `peak_speed_rad_s` | `float \| None` | rad/s | one run's active segment, activation to last sample, over joints | Largest measured joint speed magnitude. | the run was not simulated, or it aborted before activation and has no active segment. |
| `peak_acceleration_rad_s2` | `float \| None` | rad/s^2 | one run's active segment, activation to last sample, over joints | Largest measured joint acceleration magnitude: the finite difference of measured velocity between consecutive active samples. | the run was not simulated, or it aborted before activation and has no active segment. |
| `peak_reference_speed_rad_s` | `float \| None` | rad/s | one run's active segment, activation to last sample, over joints | Largest commanded joint speed magnitude: the generated reference for an RC run, the replayed recording through the causal estimator for a replay run. | the run was not simulated, or it aborted before activation and has no active segment. |
| `peak_reference_acceleration_rad_s2` | `float \| None` | rad/s^2 | one run's active segment, activation to last sample, over joints | Largest commanded joint acceleration magnitude, from the same reference. | the run was not simulated, or it aborted before activation and has no active segment. |
| `activation_jump_rad` | `float \| None` | rad | the activation sample | Euclidean norm of the first active command minus the measured posture at activation. | the run was not simulated, or it aborted before activation and has no active segment. |
| `tracking_error_rms_rad` | `float \| None` | rad | one run's active segment, activation to last sample | Root mean square of the stored tracking error, over joints and samples. | the run was not simulated, or it aborted before activation and has no active segment. |
| `tracking_error_max_rad` | `float \| None` | rad | one run's active segment, activation to last sample, over joints | Largest stored tracking-error magnitude. | the run was not simulated, or it aborted before activation and has no active segment. |
| `torque_peak_nm` | `float \| None` | N m | one run's active segment, activation to last sample, over joints | Largest applied joint torque magnitude. | the run was not simulated, or it aborted before activation and has no active segment. |
| `run_artifact_id` | `str \| None` | — | — | Identity of the stored run. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `run_uri` | `str \| None` | — | — | Store location of the run summary. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `run_sha256` | `str \| None` | — | — | SHA-256 of the run summary as written. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `run_size` | `int \| None` | bytes | one run summary | Size of the run summary. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `arrays_sha256` | `str \| None` | — | — | SHA-256 of the run's arrays file, as the summary records it. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `sources` | `tuple[str, ...] \| None` | — | — | The demonstrations the run's model trained on (the replayed one for a replay run), by artifact ID. | the run was not simulated: its model has no evidence, or the pair was not executed. |
| `fit_identity` | `str \| None` | — | — | Identity of the fit the run's model was produced from. | a replay run, which has no fit, or a model with no evidence. |

## `ManualContrastRow`

One paired comparison: arm a against arm b for one parent, over one class of scenarios.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `configuration` | `str` | — | — | Configuration both arms belong to; comparisons never pool configurations. | — |
| `tracker` | `str` | — | — | Tracker both arms ran under; comparisons never pool trackers. | — |
| `scenario_class` | `str` | — | — | Class of scenarios compared, or `all` for the 65 together. | — |
| `contrast` | `str` | — | — | `M10-S`, `M10-R10`, `C10-R10`, `M10-C10` (plan section 6), or an arm against its parent's replay: `S-replay`, `R10-replay`, `C10-replay`, `M10-replay`. | — |
| `parent` | `str` | — | — | The parent whose arms are compared; the all-ten arm is the same model for every parent. | — |
| `arm_a` | `str` | — | — | First arm, as the study names it; the difference is a minus b. | — |
| `arm_b` | `str` | — | — | Second arm, as the study names it. | — |
| `status` | `str` | — | — | `complete` when both arms have every scenario of the class, `partial` when they share only some, `unavailable` when they share none. | — |
| `n_scenarios` | `int` | scenarios | one comparison row | Scenarios the class holds. | — |
| `n_shared` | `int` | scenarios | one comparison row, of `n_scenarios` | Scenarios both arms were run on: the denominator of every count below. | — |
| `successes_a` | `int` | scenarios | one comparison row, of `n_shared` | Shared scenarios arm a succeeded on. | — |
| `successes_b` | `int` | scenarios | one comparison row, of `n_shared` | Shared scenarios arm b succeeded on. | — |
| `difference` | `int` | scenarios | one comparison row, of `n_shared` | Paired success-count difference, `successes_a - successes_b`, equal to `improved - worsened`. | — |
| `improved` | `int` | scenarios | one comparison row, of `n_shared` | Shared scenarios arm a succeeded on and arm b failed. | — |
| `worsened` | `int` | scenarios | one comparison row, of `n_shared` | Shared scenarios arm b succeeded on and arm a failed. | — |
| `tied_success` | `int` | scenarios | one comparison row, of `n_shared` | Shared scenarios both arms succeeded on. | — |
| `tied_failure` | `int` | scenarios | one comparison row, of `n_shared` | Shared scenarios both arms failed. | — |

## `ManualContrastSummary`

One contrast across the ten parents: every difference, their median and range, and their signs.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `configuration` | `str` | — | — | Configuration of the ten comparisons. | — |
| `tracker` | `str` | — | — | Tracker of the ten comparisons. | — |
| `scenario_class` | `str` | — | — | Class of scenarios compared, or `all`. | — |
| `contrast` | `str` | — | — | The contrast summarised, as in `ManualContrastRow`. | — |
| `arm_a` | `str` | — | — | Kind of the first arm. | — |
| `arm_b` | `str` | — | — | Kind of the second arm. | — |
| `n_scenarios` | `int` | scenarios | one parent's comparison | Scenarios each parent's comparison is over. | — |
| `n_parents` | `int` | parents | the ten parents | Parents whose comparison is complete: the denominator of the figures below. | — |
| `differences` | `tuple[int \| None, ...]` | — | — | Every parent's paired success-count difference in parent order D01 to D10, absent (null) where that parent's comparison is not complete. | never absent as a whole; an element is null where that parent's comparison is not complete. |
| `median` | `float \| None` | scenarios | complete parents, of `n_parents` | Median of the complete differences. | no parent's comparison is complete. |
| `minimum` | `int \| None` | scenarios | complete parents, of `n_parents` | Smallest complete difference. | no parent's comparison is complete. |
| `maximum` | `int \| None` | scenarios | complete parents, of `n_parents` | Largest complete difference. | no parent's comparison is complete. |
| `parents_improved` | `int` | parents | of `n_parents` | Parents whose difference is positive. | — |
| `parents_worsened` | `int` | parents | of `n_parents` | Parents whose difference is negative. | — |
| `parents_tied` | `int` | parents | of `n_parents` | Parents whose difference is zero. | — |

## `ManualArmSummary`

One arm's successes over one class of scenarios, with the all-ten arm counted as the single model it is.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `configuration` | `str` | — | — | Configuration of the arm. | — |
| `tracker` | `str` | — | — | Tracker the runs ran under. | — |
| `scenario_class` | `str` | — | — | Class of scenarios summed over, or `all`. | — |
| `arm_kind` | `str` | — | — | `S`, `M10`, `R10`, `C10` or `replay`; the all-ten arm is one model and counted once. | — |
| `n_scenarios` | `int` | scenarios | one configuration, tracker and class | Scenarios the class holds. | — |
| `n_models` | `int` | models | one configuration, tracker and class | Models, or replay banks, with a verdict on every scenario of the class: ten for a parented arm, one for the all-ten arm. | — |
| `n_runs` | `int` | runs | one configuration, tracker and class | Runs in the total: `n_models` times `n_scenarios`. The denominator of `successes`. | — |
| `excluded_runs` | `int` | runs | one configuration, tracker and class | Existing runs of models left out of the total because another of their runs in the class is missing. | — |
| `unavailable_runs` | `int` | runs | one configuration, tracker and class | Runs the protocol names that do not exist. | — |
| `successes` | `int` | runs | one configuration, tracker and class, of `n_runs` | Runs in the total that met every criterion. | — |
| `per_model` | `tuple[int \| None, ...]` | — | — | Each model's successes over the class, in parent order (one entry for the all-ten arm), absent (null) for a model left out. | never absent as a whole; an element is null for a model left out of the total. |
| `median` | `float \| None` | runs | models in the total | Median of the per-model successes. | no model is in the total. |
| `minimum` | `int \| None` | runs | models in the total | Smallest per-model success count. | no model is in the total. |
| `maximum` | `int \| None` | runs | models in the total | Largest per-model success count. | no model is in the total. |

## `ManualSelection`

The frozen representative rule applied to one configuration under one tracker.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `configuration` | `str` | — | — | Configuration the frozen rule was applied to. | — |
| `tracker` | `str` | — | — | Tracker the frozen rule was applied under. | — |
| `arms` | `tuple[str, ...]` | — | — | The compared arm labels, as the rule was frozen with them. | — |
| `omitted` | `tuple[str, ...]` | — | — | Scenarios left out because one of the compared arms has no verdict there; empty in a complete study. | — |
| `selection` | `Selection` | — | — | The rule's selection for this configuration and tracker. | — |

## `ManualSelections`

Every application of the frozen representative rule, in the rule's configuration and tracker order.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `experiment` | `str` | — | — | Experiment label the selections belong to. | — |
| `rule_sha256` | `str` | — | — | SHA-256 of the frozen representative rule applied. | — |
| `ordering_sha256` | `str` | — | — | SHA-256 of the run ordering whose scenario order broke ties. | — |
| `applications` | `tuple[ManualSelection, ...]` | — | — | One application per configuration and tracker, in the rule's order. | — |
| `schema_version` | `int` | version | one selections document | Version of the selections record. | — |

## `ManualFigureRun`

One stored run a figure draws, with its role in the comparison and its verdict.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `role` | `str` | — | — | `S`, `M10`, `R10`, `C10` or `replay`: the run's place in the comparison. | — |
| `label` | `str` | — | — | Model or replay-bank label of the run. | — |
| `success` | `bool` | — | — | The run's verdict. | — |
| `pulse_start_s` | `float \| None` | s | one run's clock, zero at its first sample | When the run's pulse began. | no pulse fired in this run. |
| `pulse_end_s` | `float \| None` | s | one run's clock, zero at its first sample | When the run's pulse ended. | no pulse fired in this run. |
| `run` | `ManualRunArtifact` | — | — | The stored run, bound by the digests of its summary and arrays. | — |

## `ManualFigureCase`

One illustrated case: the runs to draw beside the demonstration they descend from.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `case_id` | `str` | — | — | `<configuration>__<tracker>__<scenario>`, how a rendering command names the case. | — |
| `configuration` | `str` | — | — | Configuration of the case. | — |
| `tracker` | `str` | — | — | Tracker of the case. | — |
| `scenario_id` | `str` | — | — | Scenario of the case. | — |
| `scenario_class` | `str` | — | — | Perturbation class of the scenario. | — |
| `categories` | `tuple[str, ...]` | — | — | Every rule category that chose the case, in the rule's order. | — |
| `activation_s` | `float` | s | one run's clock, zero at its first sample | Activation time shared by the case's runs: task time zero. | — |
| `teacher_assignment` | `str` | — | — | Parent whose recorded demonstration the figure draws. | — |
| `teacher` | `DatasetSource` | — | — | The recorded demonstration, bound by its record and payload digest. | — |
| `runs` | `tuple[ManualFigureRun, ...]` | — | — | The runs drawn: the rule's four arms and the parent's replay, where they exist. | — |
| `missing` | `tuple[str, ...]` | — | — | Roles the case names whose runs do not exist; empty in a complete study. | — |

## `ManualFigureInputs`

Every case the frozen rule chose, with the task configuration its kinematics come from.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `experiment` | `str` | — | — | Experiment label the inputs belong to. | — |
| `scenario_file` | `str` | — | — | Repository-relative path of the manual task configuration used for kinematics. | — |
| `scenario_sha256` | `str` | — | — | SHA-256 of that task configuration. | — |
| `cases` | `tuple[ManualFigureCase, ...]` | — | — | Every case the frozen rule chose, in selection order. | — |
| `schema_version` | `int` | version | one figure-inputs document | Version of the figure-inputs record. | — |

## `ManualResultInputs`

The frozen inputs the derivation read, bound by digest.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `study_manifest_sha256` | `str` | — | — | SHA-256 of the frozen study manifest. | — |
| `evaluation_sha256` | `str` | — | — | SHA-256 of the evaluation configuration every run was keyed under. | — |
| `run_ordering_sha256` | `str` | — | — | SHA-256 of the frozen run ordering. | — |
| `representative_rule_sha256` | `str` | — | — | SHA-256 of the frozen representative-case rule. | — |
| `result_schema_sha256` | `str` | — | — | SHA-256 of the result schema these outputs follow. | — |
| `evidence_sha256` | `str` | — | — | SHA-256 over every evidence pointer's file name and file digest, in name order. | — |
| `n_pointers` | `int` | pointers | the evidence directory | Evidence pointers read. | — |

## `ManualResultTable`

One table kept in the external store, cited by its digest and size.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | — | — | Name of the table, with its version. | — |
| `record` | `str` | — | — | Result-schema record each row follows. | — |
| `payload` | `ArtifactReference` | — | — | Store location, SHA-256 and size of the CSV file. | — |
| `n_rows` | `int` | rows | one stored table | Rows the table holds, header excluded. | — |
| `columns` | `tuple[str, ...]` | — | — | The header, in order: the record's fields. | — |

## `ManualResultDocument`

One committed output beside the results manifest, cited by its digest and size.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `name` | `str` | — | — | File name, beside the results index. | — |
| `record` | `str` | — | — | Result-schema record the document holds. | — |
| `sha256` | `str` | — | — | SHA-256 of the file as written. | — |
| `size` | `int` | bytes | one document | Size of the file. | — |
| `n_entries` | `int` | entries | one document | Rows of a CSV document, or the lines, applications or cases of a JSON one. | — |

## `ManualResults`

The index of the derived evidence: what was read, what was written, and the totals it all rests on.

| field | type | unit | scope | meaning | absent |
| --- | --- | --- | --- | --- | --- |
| `experiment` | `str` | — | — | Experiment label the derived evidence belongs to. | — |
| `version` | `int` | version | one results index | Version of the derived outputs; a changed output is a new version beside this one. | — |
| `result_schema_version` | `int` | version | one results index | Result schema version whose records the outputs follow. | — |
| `inputs` | `ManualResultInputs` | — | — | The frozen inputs read, bound by digest. | — |
| `tables` | `tuple[ManualResultTable, ...]` | — | — | The tables kept in the external store. | — |
| `documents` | `tuple[ManualResultDocument, ...]` | — | — | The committed documents beside this index. | — |
| `n_models` | `int` | models | the study's 186 models | Study models with evidence. | — |
| `n_replay_banks` | `int` | banks | the study's 60 banks | Replay banks keyed to a configuration and parent. | — |
| `n_rc_runs` | `int` | runs | one results index, over all runs | RC runs simulated. | — |
| `n_replay_runs` | `int` | runs | one results index, over all runs | Replay runs simulated. | — |
| `n_unavailable_runs` | `int` | runs | one results index, over all runs | Rows naming a run that was not simulated. | — |
| `n_rc_successes` | `int` | runs | one results index, over all runs, of `n_rc_runs` | RC runs that met every criterion. | — |
| `n_replay_successes` | `int` | runs | one results index, over all runs, of `n_replay_runs` | Replay runs that met every criterion. | — |
| `departure_radius_m` | `float` | m | every run | Radius used for `departure_latency_s`: the dwell radius every run was judged under. | — |
| `command` | `str` | — | — | The command line that derived the outputs. | — |
| `provenance` | `ProvenanceRecord` | — | — | Reproducibility record of that invocation. | — |
| `schema_version` | `int` | version | one results index | Version of the results index record itself. | — |
