# Task 1-a manual-demonstration result schema (v1)

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
