# Task 1-a repetition panel manifest (v1)

Experiment `task_1a_repetition_v1`: the six fixed source configurations of the timing-only recovery search (repetition plan section 5.1, decision D3), frozen before any pilot arm is trained.

## Sources

- Study `recovery-search-1a-no-augmentation-v1` via pointer `recovery_search_no_augmentation_v1.toml` (sha256 `293816654a26`); report payload `armrc://reports/task_1a_state_conditioned_recovery/recovery_search_no_augmentation_v1-0eef1efe875d.json` (sha256 `0eef1efe875d`, 39506133 bytes); 500 trials stored, 134 feasible.
- Protocol `configs/studies/recovery_search_1a_no_augmentation_v1.toml` (file sha256 `51fbc2d67fe2`, portable digest `d3d3d087187d`).
- Ablation `development_ablation_v2.json` (sha256 `ff1e945287e8`): 134 of 134 candidates belong to the source study.
- Model `configs/models/esn_task_1a_v4.toml` (`72a55213c7d5`), scenario `configs/tasks/task_1a.toml` (`23c89a7f8a15`), development levels `configs/evaluations/task_1a_recovery_dev_v1.toml` (`5d84d049ff40`).
- Dataset `processed-20260903-ce343c8ce6a5`: record `data/records/processed/processed-20260903-ce343c8ce6a5.toml` (`97f2e5cf1f67`), payload sha256 `ce343c8ce6a5`.
- Frozen trackers: `computed_torque` (`0ac3dff977cd`), `pd_v2` (`45f6e7a31490`).

## Selection rule

- Feasible ranking: Feasible trials of the source study sorted by (objective, trial_number) ascending over the development ablation's candidates; rank r takes the r-th trial (plan section 5.1).
- Failure categories: Lowest trial number among the study's infeasible trials whose top-level first-failure reason head is the category; the head is the gate before the first colon, keeping the limit name of a violation, so every dwell:* head (dwell_stationary, dwell_in_tolerance, or both) is the actual-motion dwell category (plan sections 5.1 and 12).
- Approved identities (D3): feasible-best = trial 17, feasible-middle = trial 136, feasible-worst = trial 53, failure-actual-dwell = trial 1, failure-joint-velocity = trial 0, failure-generated-dwell = trial 28. A resolution that differs fails; no configuration is replaced because its repeated or augmented variant performs poorly.

## Panel

| label | trial | role | warm-up (s) | objective | first failure |
| --- | ---: | --- | ---: | ---: | --- |
| feasible-best | 17 | Rank 1 of the 134 feasible trials by early-gap objective | 0.25 | 0.6704 |  |
| feasible-middle | 136 | Rank 67 of the 134 feasible trials by early-gap objective | 0 | 0.9602 |  |
| feasible-worst | 53 | Rank 134 of the 134 feasible trials by early-gap objective | 1 | 1.286 |  |
| failure-actual-dwell | 1 | Lowest-numbered trial whose first failure is dwell | 0.25 | 10 | scenario 0 [pd_v2]: dwell:dwell_stationary |
| failure-joint-velocity | 0 | Lowest-numbered trial whose first failure is limit_violation:joint_velocity; also the historical anchor 'anchor-v4-tw1' | 1 | 10 | scenario 23 [pd_v2]: limit_violation:joint_velocity |
| failure-generated-dwell | 28 | Lowest-numbered trial whose first failure is generated_dwell | 0 | 10 | scenario 0 [pd_v2]: generated_dwell:generated_dwell_stationary |

## Parameters

Reservoir/readout parameters and the warm-up are recipe inputs; the estimator cutoffs are evaluation-side settings, never recipe fields; `alpha` is the source ridge parameter alpha_0 that the pilot's R-scaled and S-effective arms scale.

| label | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | alpha | velocity_cutoff_hz | acceleration_cutoff_hz | max_dt_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feasible-best | 100 | 1.263266875517793 | 0.9086681297381796 | 0.06544592224060722 | 0.4162819932167906 | 965 | 0.1191718250387115 | 6.6694585741072885 | 5.590310542267199 | 3.0 |
| feasible-middle | 400 | 1.299786136219192 | 0.6405712425492811 | 0.03857698270429774 | 0.2758773237273159 | 922 | 0.003097590569817245 | 21.853688084124755 | 9.331655895077768 | 3.0 |
| feasible-worst | 300 | 1.1483563502746237 | 0.8674824469976874 | 0.013523859112187989 | 0.022281641437870696 | 938 | 0.004935486520748805 | 6.7310817038954465 | 5.1302452892157655 | 3.0 |
| failure-actual-dwell | 100 | 1.1621910014621486 | 0.6097988097560109 | 0.014664019100094813 | 0.2114098851877163 | 340 | 0.6191588897437532 | 28.457478731077835 | 5.730433263875364 | 3.0 |
| failure-joint-velocity | 250 | 1.2944675208876626 | 0.9791076284866893 | 0.04089985548951509 | 0.021176881502572638 | 896 | 0.002849478837743603 | 29.980411525699598 | 10.938122239871603 | 3.0 |
| failure-generated-dwell | 300 | 1.118277115171816 | 0.7819437758127186 | 0.12632472327232913 | 0.044104986024179345 | 966 | 0.08365993429880747 | 7.555972499674083 | 20.38331154329677 | 3.0 |

## Provenance

- Source study: commit `b7d29fe891cd`, created 2026-09-03T15:50:59+00:00, Python 3.12.11, lock `ac9811f8142e`; submodules rclib `a015aca1ec9e`, rtctrl `c601076ee60e`, skelarm `6ccc1eba8ff5`; builds rclib 0.1.0@a015aca1ec9eaabb9ad4e384bf33e2e76018bf8b, skelarm 0.4.1@6ccc1eba8ff57178ab8bf456e6ff9a2ec988cc80.
- This manifest: commit `3625969ac24f`, created 2026-09-09T06:29:55+00:00, Python 3.12.11, lock `ac9811f8142e`.

The two revisions are recorded separately (clarification C1): the parameters come from the historical study; the manifest comes from the pilot implementation. No new reservoir seed is introduced by this pilot.
