# Task 1-a manual-demonstration run ordering (v1)

Frozen before execution. The runs are the stated nesting over the lists below; nothing here is a result.
Record definitions are not restated here: see result schema v2, bound by digest below (`StudyAccounting`, `ModelAccount`, `BankAccount`).

| bound input | sha256 |
| --- | --- |
| study manifest | `b07b824c362a3513c03c9c34fa0437b390f47d148ac5a288231fba218aaa36b7` |
| evaluation configuration | `8780be5bac7f4ccb99d6636c947f78901646b886b5f768846b6d9ba5d48acb2b` |
| result schema v2 | `dfd3e29b2f7629253cdf511d1690e53d0b0cf82ba1013628e386ca6a12867b9e` |

## Expected accounting

| figure | value |
| --- | ---: |
| models | 186 |
| replay banks | 60 |
| pairs per model or bank | 130 |
| RC runs | 24,180 |
| replay runs | 7,800 |
| runs in total | 31,980 |

## Expansion

RC runs are model-major: for each model in `models`, each scenario in `scenarios`, each tracker in `trackers`. Replay runs are bank-major: for each bank in `replay_banks`, each scenario, each tracker. A model with an assignment is paired against the bank of its own configuration and assignment; the all-ten arm is paired against none.

## Models (186, manifest order)

- `feasible-best`: `S/D01`, `S/D02`, `S/D03`, `S/D04`, `S/D05`, `S/D06`, `S/D07`, `S/D08`, `S/D09`, `S/D10`, `M10`, `R10/D01`, `R10/D02`, `R10/D03`, `R10/D04`, `R10/D05`, `R10/D06`, `R10/D07`, `R10/D08`, `R10/D09`, `R10/D10`, `C10/D01`, `C10/D02`, `C10/D03`, `C10/D04`, `C10/D05`, `C10/D06`, `C10/D07`, `C10/D08`, `C10/D09`, `C10/D10`
- `feasible-middle`: `S/D01`, `S/D02`, `S/D03`, `S/D04`, `S/D05`, `S/D06`, `S/D07`, `S/D08`, `S/D09`, `S/D10`, `M10`, `R10/D01`, `R10/D02`, `R10/D03`, `R10/D04`, `R10/D05`, `R10/D06`, `R10/D07`, `R10/D08`, `R10/D09`, `R10/D10`, `C10/D01`, `C10/D02`, `C10/D03`, `C10/D04`, `C10/D05`, `C10/D06`, `C10/D07`, `C10/D08`, `C10/D09`, `C10/D10`
- `feasible-worst`: `S/D01`, `S/D02`, `S/D03`, `S/D04`, `S/D05`, `S/D06`, `S/D07`, `S/D08`, `S/D09`, `S/D10`, `M10`, `R10/D01`, `R10/D02`, `R10/D03`, `R10/D04`, `R10/D05`, `R10/D06`, `R10/D07`, `R10/D08`, `R10/D09`, `R10/D10`, `C10/D01`, `C10/D02`, `C10/D03`, `C10/D04`, `C10/D05`, `C10/D06`, `C10/D07`, `C10/D08`, `C10/D09`, `C10/D10`
- `failure-actual-dwell`: `S/D01`, `S/D02`, `S/D03`, `S/D04`, `S/D05`, `S/D06`, `S/D07`, `S/D08`, `S/D09`, `S/D10`, `M10`, `R10/D01`, `R10/D02`, `R10/D03`, `R10/D04`, `R10/D05`, `R10/D06`, `R10/D07`, `R10/D08`, `R10/D09`, `R10/D10`, `C10/D01`, `C10/D02`, `C10/D03`, `C10/D04`, `C10/D05`, `C10/D06`, `C10/D07`, `C10/D08`, `C10/D09`, `C10/D10`
- `failure-joint-velocity`: `S/D01`, `S/D02`, `S/D03`, `S/D04`, `S/D05`, `S/D06`, `S/D07`, `S/D08`, `S/D09`, `S/D10`, `M10`, `R10/D01`, `R10/D02`, `R10/D03`, `R10/D04`, `R10/D05`, `R10/D06`, `R10/D07`, `R10/D08`, `R10/D09`, `R10/D10`, `C10/D01`, `C10/D02`, `C10/D03`, `C10/D04`, `C10/D05`, `C10/D06`, `C10/D07`, `C10/D08`, `C10/D09`, `C10/D10`
- `failure-generated-dwell`: `S/D01`, `S/D02`, `S/D03`, `S/D04`, `S/D05`, `S/D06`, `S/D07`, `S/D08`, `S/D09`, `S/D10`, `M10`, `R10/D01`, `R10/D02`, `R10/D03`, `R10/D04`, `R10/D05`, `R10/D06`, `R10/D07`, `R10/D08`, `R10/D09`, `R10/D10`, `C10/D01`, `C10/D02`, `C10/D03`, `C10/D04`, `C10/D05`, `C10/D06`, `C10/D07`, `C10/D08`, `C10/D09`, `C10/D10`

## Scenarios (65, locked order)

1. `nominal`
2. `posture-small-20261201-00`
3. `posture-small-20261201-01`
4. `posture-small-20261201-02`
5. `posture-small-20261201-03`
6. `posture-small-20261202-00`
7. `posture-small-20261202-01`
8. `posture-small-20261202-02`
9. `posture-small-20261202-03`
10. `posture-small-20261203-00`
11. `posture-small-20261203-01`
12. `posture-small-20261203-02`
13. `posture-small-20261203-03`
14. `posture-small-20261204-00`
15. `posture-small-20261204-01`
16. `posture-small-20261204-02`
17. `posture-small-20261204-03`
18. `posture-small-20261205-00`
19. `posture-small-20261205-01`
20. `posture-small-20261205-02`
21. `posture-small-20261205-03`
22. `posture-large-20261201-00`
23. `posture-large-20261201-01`
24. `posture-large-20261201-02`
25. `posture-large-20261201-03`
26. `posture-large-20261202-00`
27. `posture-large-20261202-01`
28. `posture-large-20261202-02`
29. `posture-large-20261202-03`
30. `posture-large-20261203-00`
31. `posture-large-20261203-01`
32. `posture-large-20261203-02`
33. `posture-large-20261203-03`
34. `posture-large-20261204-00`
35. `posture-large-20261204-01`
36. `posture-large-20261204-02`
37. `posture-large-20261204-03`
38. `posture-large-20261205-00`
39. `posture-large-20261205-01`
40. `posture-large-20261205-02`
41. `posture-large-20261205-03`
42. `force-12N-000deg`
43. `force-12N-090deg`
44. `force-12N-180deg`
45. `force-12N-270deg`
46. `combined-20261201-00-000deg`
47. `combined-20261201-01-090deg`
48. `combined-20261201-02-180deg`
49. `combined-20261201-03-270deg`
50. `combined-20261202-00-000deg`
51. `combined-20261202-01-090deg`
52. `combined-20261202-02-180deg`
53. `combined-20261202-03-270deg`
54. `combined-20261203-00-000deg`
55. `combined-20261203-01-090deg`
56. `combined-20261203-02-180deg`
57. `combined-20261203-03-270deg`
58. `combined-20261204-00-000deg`
59. `combined-20261204-01-090deg`
60. `combined-20261204-02-180deg`
61. `combined-20261204-03-270deg`
62. `combined-20261205-00-000deg`
63. `combined-20261205-01-090deg`
64. `combined-20261205-02-180deg`
65. `combined-20261205-03-270deg`

## Trackers

1. `pd_v2`
2. `computed_torque`

## Replay banks (60, configuration x assignment)

| configuration | assignments | warm-up (s) | velocity cutoff (Hz) | acceleration cutoff (Hz) |
| --- | --- | ---: | ---: | ---: |
| `feasible-best` | D01, D02, D03, D04, D05, D06, D07, D08, D09, D10 | 0.25 | 6.669459 | 5.590311 |
| `feasible-middle` | D01, D02, D03, D04, D05, D06, D07, D08, D09, D10 | 0 | 21.853688 | 9.331656 |
| `feasible-worst` | D01, D02, D03, D04, D05, D06, D07, D08, D09, D10 | 1 | 6.731082 | 5.130245 |
| `failure-actual-dwell` | D01, D02, D03, D04, D05, D06, D07, D08, D09, D10 | 0.25 | 28.457479 | 5.730433 |
| `failure-joint-velocity` | D01, D02, D03, D04, D05, D06, D07, D08, D09, D10 | 1 | 29.980412 | 10.938122 |
| `failure-generated-dwell` | D01, D02, D03, D04, D05, D06, D07, D08, D09, D10 | 0 | 7.555972 | 20.383312 |
