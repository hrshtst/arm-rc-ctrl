# ESN search `esn-search-1a-v2`

- Protocol: `configs/studies/esn_search_1a_v2.toml` (digest `7834cfda9710`), dataset `processed-20260830-feaf73e6663c`,
  tracker `pd_v2` (digest `45f6e7a31490`).
- Budget 1000; stored 1000 trials (1000 complete, 0 pruned); 902 feasible; this invocation ran 1000.
- Provenance: commit `6247473171a0`, sampler seed 20260922.

## Selection

Trial 843 with objective 0.006577 rad (median movement-window joint RMSE).

| parameter | value |
| --- | --- |
| n_neurons | 250 |
| spectral_radius | 1.2944675208876626 |
| sparsity | 0.9791076284866893 |
| leak_rate | 0.04089985548951509 |
| input_scaling | 0.021176881502572638 |
| seed | 896 |
| alpha | 0.002849478837743603 |
| velocity_cutoff_hz | 29.980411525699598 |
| acceleration_cutoff_hz | 10.938122239871603 |

Development metrics of the selected trial:

| scenario | kind | movement RMSE (rad) | saturation | criteria |
| --- | --- | --- | --- | --- |
| 0 | posture | 0.002599 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 1 | posture | 0.004772 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 2 | posture | 0.005201 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 3 | posture | 0.00354 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 4 | posture | 0.004006 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 5 | posture | 0.004601 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 6 | posture | 0.005249 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 7 | posture | 0.003963 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 8 | posture | 0.004049 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 9 | posture | 0.007855 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 10 | posture | 0.008369 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 11 | posture | 0.005808 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 12 | posture | 0.006323 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 13 | posture | 0.007512 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 14 | posture | 0.008138 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 15 | posture | 0.006577 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 16 | posture | 0.006534 | 0.001996 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 17 | force | 0.009345 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 18 | force | 0.009078 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 19 | force | 0.01025 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 20 | force | 0.008721 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 21 | force | 0.01349 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 22 | force | 0.01264 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 23 | force | 0.01532 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |
| 24 | force | 0.01357 | 0 | completed=ok, dwell_in_tolerance=ok, dwell_stationary=ok |

## Comparison points

| label | trial | objective (rad) | feasible | reason |
| --- | --- | --- | --- | --- |
| anchor-alpha-1e-2 | 0 | 10 | False | scenario 2: dwell:dwell_in_tolerance |
| anchor-alpha-3e-2 | 1 | 10 | False | scenario 2: dwell:dwell_in_tolerance |
| anchor-alpha-1e-1 | 2 | 10 | False | scenario 2: dwell:dwell_in_tolerance |
| anchor-alpha-3e-1 | 3 | 10 | False | scenario 2: dwell:dwell_in_tolerance |
| v1-trial-13 | 4 | 0.02165 | True |  |

## Best feasible trials

| trial | objective (rad) | n_neurons | spectral_radius | sparsity | leak_rate | input_scaling | seed | alpha | velocity_cutoff_hz | acceleration_cutoff_hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 843 | 0.006577 | 250 | 1.294 | 0.9791 | 0.0409 | 0.02118 | 896 | 0.002849 | 29.98 | 10.94 |
| 854 | 0.006591 | 350 | 1.299 | 0.9724 | 0.0472 | 0.02321 | 921 | 0.003268 | 28.49 | 11.73 |
| 887 | 0.006595 | 250 | 1.284 | 0.9642 | 0.0417 | 0.02296 | 940 | 0.003065 | 27.3 | 10.7 |
| 840 | 0.006645 | 250 | 1.299 | 0.9768 | 0.03424 | 0.02259 | 895 | 0.002273 | 29.91 | 10.96 |
| 836 | 0.006646 | 250 | 1.299 | 0.9798 | 0.02806 | 0.02501 | 899 | 0.002424 | 29.04 | 9.307 |
| 841 | 0.006699 | 250 | 1.3 | 0.9777 | 0.03924 | 0.0214 | 913 | 0.002757 | 28.24 | 10.35 |
| 844 | 0.006716 | 250 | 1.289 | 0.9762 | 0.043 | 0.02134 | 918 | 0.003313 | 29.91 | 11.74 |
| 853 | 0.006722 | 250 | 1.291 | 0.98 | 0.04316 | 0.02286 | 920 | 0.002964 | 28.11 | 11.81 |
| 895 | 0.006727 | 250 | 1.291 | 0.963 | 0.04056 | 0.02258 | 885 | 0.002377 | 26.61 | 12.49 |
| 320 | 0.006736 | 250 | 1.24 | 0.9569 | 0.0283 | 0.02536 | 607 | 0.001384 | 21.87 | 7.746 |

## Infeasible and pruned trials by reason

| reason | trials |
| --- | --- |
| dwell:dwell_in_tolerance,dwell_stationary | 44 |
| dwell:dwell_stationary | 38 |
| dwell:dwell_in_tolerance | 15 |
| limit_violation:joint_velocity | 1 |
