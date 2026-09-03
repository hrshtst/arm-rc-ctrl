# Augmentation validation

- Dataset: `processed-20260903-ce343c8ce6a5` (payload `ce343c8ce6a5`)
- Scenario: task-1a-reach; derivative policy: central-difference
- Seed namespace: `[20260903, seed_bank, attempt]` (no evaluation seed consumed)
- Locked taper: smoothstep over 0.2 s, exact zero from 0.1 s before dwell onset (dwell starts at 2.999999999999999 s)
- Outcome: **PASS**

## Global checks

| check | outcome | detail |
| --- | --- | --- |
| source-binding | PASS | payload matches processed-20260903-ce343c8ce6a5 |
| bank-separation | PASS | banks 1 and 2 differ |

## Configurations

| N_aug | sigma (rad) | phi | gamma | attempts | rejected | determinism | bounds | smoothness | correlation | envelope | dwell-collapse | episode-separation | rejection-accounting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.01 | 0.98 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.98 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.98 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.99 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.99 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.99 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.995 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.995 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.995 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.98 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.98 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.98 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.99 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.99 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.99 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.995 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.995 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.995 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.98 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.98 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.98 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.99 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.99 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.99 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.995 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.995 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.995 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.98 | 0.5 | 17 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.98 | 1.0 | 17 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.98 | 2.0 | 17 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.99 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.99 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.99 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.995 | 0.5 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.995 | 1.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.995 | 2.0 | 16 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.98 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.98 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.98 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.99 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.99 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.99 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.995 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.995 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.995 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.98 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.98 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.98 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.99 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.99 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.99 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.995 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.995 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.995 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.98 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.98 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.98 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.99 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.99 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.99 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.995 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.995 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.995 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.98 | 0.5 | 34 | 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.98 | 1.0 | 34 | 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.98 | 2.0 | 34 | 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.99 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.99 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.99 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.995 | 0.5 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.995 | 1.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.995 | 2.0 | 32 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.98 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.98 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.98 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.99 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.99 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.99 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.995 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.995 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.995 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.98 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.98 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.98 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.99 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.99 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.99 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.995 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.995 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.995 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.98 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.98 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.98 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.99 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.99 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.99 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.995 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.995 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.995 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.98 | 0.5 | 70 | 11 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.98 | 1.0 | 70 | 11 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.98 | 2.0 | 70 | 11 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.99 | 0.5 | 65 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.99 | 1.0 | 65 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.99 | 2.0 | 65 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.995 | 0.5 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.995 | 1.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.995 | 2.0 | 64 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Realized amplitudes (per family, rad)

| N_aug | sigma | phi | gamma | family | rms median | rms max | peak median | peak max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.01 | 0.98 | 0.5 | non_decaying | 0.00822 | 0.01045 | 0.02672 | 0.03825 |
| 16 | 0.01 | 0.98 | 0.5 | contractive | 0.00598 | 0.00808 | 0.02193 | 0.02935 |
| 16 | 0.01 | 0.98 | 1.0 | non_decaying | 0.00822 | 0.01045 | 0.02672 | 0.03825 |
| 16 | 0.01 | 0.98 | 1.0 | contractive | 0.00532 | 0.00732 | 0.02169 | 0.02818 |
| 16 | 0.01 | 0.98 | 2.0 | non_decaying | 0.00822 | 0.01045 | 0.02672 | 0.03825 |
| 16 | 0.01 | 0.98 | 2.0 | contractive | 0.00473 | 0.00655 | 0.02104 | 0.02598 |
| 16 | 0.01 | 0.99 | 0.5 | non_decaying | 0.00789 | 0.01132 | 0.02375 | 0.03592 |
| 16 | 0.01 | 0.99 | 0.5 | contractive | 0.00591 | 0.00857 | 0.01980 | 0.02499 |
| 16 | 0.01 | 0.99 | 1.0 | non_decaying | 0.00789 | 0.01132 | 0.02375 | 0.03592 |
| 16 | 0.01 | 0.99 | 1.0 | contractive | 0.00519 | 0.00763 | 0.01881 | 0.02400 |
| 16 | 0.01 | 0.99 | 2.0 | non_decaying | 0.00789 | 0.01132 | 0.02375 | 0.03592 |
| 16 | 0.01 | 0.99 | 2.0 | contractive | 0.00460 | 0.00668 | 0.01817 | 0.02213 |
| 16 | 0.01 | 0.995 | 0.5 | non_decaying | 0.00756 | 0.01161 | 0.02142 | 0.03046 |
| 16 | 0.01 | 0.995 | 0.5 | contractive | 0.00546 | 0.00851 | 0.01752 | 0.02368 |
| 16 | 0.01 | 0.995 | 1.0 | non_decaying | 0.00756 | 0.01161 | 0.02142 | 0.03046 |
| 16 | 0.01 | 0.995 | 1.0 | contractive | 0.00491 | 0.00758 | 0.01646 | 0.02266 |
| 16 | 0.01 | 0.995 | 2.0 | non_decaying | 0.00756 | 0.01161 | 0.02142 | 0.03046 |
| 16 | 0.01 | 0.995 | 2.0 | contractive | 0.00424 | 0.00698 | 0.01589 | 0.02132 |
| 16 | 0.025 | 0.98 | 0.5 | non_decaying | 0.02055 | 0.02611 | 0.06680 | 0.09564 |
| 16 | 0.025 | 0.98 | 0.5 | contractive | 0.01494 | 0.02020 | 0.05482 | 0.07338 |
| 16 | 0.025 | 0.98 | 1.0 | non_decaying | 0.02055 | 0.02611 | 0.06680 | 0.09564 |
| 16 | 0.025 | 0.98 | 1.0 | contractive | 0.01330 | 0.01831 | 0.05421 | 0.07046 |
| 16 | 0.025 | 0.98 | 2.0 | non_decaying | 0.02055 | 0.02611 | 0.06680 | 0.09564 |
| 16 | 0.025 | 0.98 | 2.0 | contractive | 0.01183 | 0.01637 | 0.05260 | 0.06496 |
| 16 | 0.025 | 0.99 | 0.5 | non_decaying | 0.01973 | 0.02829 | 0.05938 | 0.08979 |
| 16 | 0.025 | 0.99 | 0.5 | contractive | 0.01477 | 0.02143 | 0.04949 | 0.06248 |
| 16 | 0.025 | 0.99 | 1.0 | non_decaying | 0.01973 | 0.02829 | 0.05938 | 0.08979 |
| 16 | 0.025 | 0.99 | 1.0 | contractive | 0.01297 | 0.01906 | 0.04701 | 0.05999 |
| 16 | 0.025 | 0.99 | 2.0 | non_decaying | 0.01973 | 0.02829 | 0.05938 | 0.08979 |
| 16 | 0.025 | 0.99 | 2.0 | contractive | 0.01150 | 0.01670 | 0.04542 | 0.05531 |
| 16 | 0.025 | 0.995 | 0.5 | non_decaying | 0.01889 | 0.02901 | 0.05355 | 0.07616 |
| 16 | 0.025 | 0.995 | 0.5 | contractive | 0.01364 | 0.02127 | 0.04381 | 0.05920 |
| 16 | 0.025 | 0.995 | 1.0 | non_decaying | 0.01889 | 0.02901 | 0.05355 | 0.07616 |
| 16 | 0.025 | 0.995 | 1.0 | contractive | 0.01227 | 0.01895 | 0.04115 | 0.05666 |
| 16 | 0.025 | 0.995 | 2.0 | non_decaying | 0.01889 | 0.02901 | 0.05355 | 0.07616 |
| 16 | 0.025 | 0.995 | 2.0 | contractive | 0.01059 | 0.01745 | 0.03971 | 0.05331 |
| 16 | 0.05 | 0.98 | 0.5 | non_decaying | 0.04110 | 0.05223 | 0.13360 | 0.19127 |
| 16 | 0.05 | 0.98 | 0.5 | contractive | 0.02989 | 0.04040 | 0.10965 | 0.14676 |
| 16 | 0.05 | 0.98 | 1.0 | non_decaying | 0.04110 | 0.05223 | 0.13360 | 0.19127 |
| 16 | 0.05 | 0.98 | 1.0 | contractive | 0.02659 | 0.03661 | 0.10843 | 0.14092 |
| 16 | 0.05 | 0.98 | 2.0 | non_decaying | 0.04110 | 0.05223 | 0.13360 | 0.19127 |
| 16 | 0.05 | 0.98 | 2.0 | contractive | 0.02365 | 0.03274 | 0.10519 | 0.12992 |
| 16 | 0.05 | 0.99 | 0.5 | non_decaying | 0.03947 | 0.05659 | 0.11876 | 0.17958 |
| 16 | 0.05 | 0.99 | 0.5 | contractive | 0.02954 | 0.04287 | 0.09898 | 0.12496 |
| 16 | 0.05 | 0.99 | 1.0 | non_decaying | 0.03947 | 0.05659 | 0.11876 | 0.17958 |
| 16 | 0.05 | 0.99 | 1.0 | contractive | 0.02594 | 0.03813 | 0.09403 | 0.11999 |
| 16 | 0.05 | 0.99 | 2.0 | non_decaying | 0.03947 | 0.05659 | 0.11876 | 0.17958 |
| 16 | 0.05 | 0.99 | 2.0 | contractive | 0.02300 | 0.03340 | 0.09084 | 0.11063 |
| 16 | 0.05 | 0.995 | 0.5 | non_decaying | 0.03778 | 0.05803 | 0.10709 | 0.15232 |
| 16 | 0.05 | 0.995 | 0.5 | contractive | 0.02728 | 0.04253 | 0.08761 | 0.11840 |
| 16 | 0.05 | 0.995 | 1.0 | non_decaying | 0.03778 | 0.05803 | 0.10709 | 0.15232 |
| 16 | 0.05 | 0.995 | 1.0 | contractive | 0.02454 | 0.03790 | 0.08230 | 0.11332 |
| 16 | 0.05 | 0.995 | 2.0 | non_decaying | 0.03778 | 0.05803 | 0.10709 | 0.15232 |
| 16 | 0.05 | 0.995 | 2.0 | contractive | 0.02119 | 0.03490 | 0.07943 | 0.10662 |
| 16 | 0.1 | 0.98 | 0.5 | non_decaying | 0.08245 | 0.10446 | 0.26719 | 0.38255 |
| 16 | 0.1 | 0.98 | 0.5 | contractive | 0.05978 | 0.08080 | 0.21447 | 0.29351 |
| 16 | 0.1 | 0.98 | 1.0 | non_decaying | 0.08245 | 0.10446 | 0.26719 | 0.38255 |
| 16 | 0.1 | 0.98 | 1.0 | contractive | 0.05286 | 0.07322 | 0.21062 | 0.28183 |
| 16 | 0.1 | 0.98 | 2.0 | non_decaying | 0.08245 | 0.10446 | 0.26719 | 0.38255 |
| 16 | 0.1 | 0.98 | 2.0 | contractive | 0.04606 | 0.06549 | 0.20332 | 0.25984 |
| 16 | 0.1 | 0.99 | 0.5 | non_decaying | 0.07893 | 0.11317 | 0.23751 | 0.35917 |
| 16 | 0.1 | 0.99 | 0.5 | contractive | 0.05907 | 0.08574 | 0.19797 | 0.24992 |
| 16 | 0.1 | 0.99 | 1.0 | non_decaying | 0.07893 | 0.11317 | 0.23751 | 0.35917 |
| 16 | 0.1 | 0.99 | 1.0 | contractive | 0.05188 | 0.07625 | 0.18806 | 0.23997 |
| 16 | 0.1 | 0.99 | 2.0 | non_decaying | 0.07893 | 0.11317 | 0.23751 | 0.35917 |
| 16 | 0.1 | 0.99 | 2.0 | contractive | 0.04600 | 0.06680 | 0.18168 | 0.22125 |
| 16 | 0.1 | 0.995 | 0.5 | non_decaying | 0.07556 | 0.11605 | 0.21418 | 0.30463 |
| 16 | 0.1 | 0.995 | 0.5 | contractive | 0.05456 | 0.08507 | 0.17523 | 0.23681 |
| 16 | 0.1 | 0.995 | 1.0 | non_decaying | 0.07556 | 0.11605 | 0.21418 | 0.30463 |
| 16 | 0.1 | 0.995 | 1.0 | contractive | 0.04908 | 0.07580 | 0.16459 | 0.22663 |
| 16 | 0.1 | 0.995 | 2.0 | non_decaying | 0.07556 | 0.11605 | 0.21418 | 0.30463 |
| 16 | 0.1 | 0.995 | 2.0 | contractive | 0.04237 | 0.06981 | 0.15885 | 0.21323 |
| 32 | 0.01 | 0.98 | 0.5 | non_decaying | 0.00810 | 0.01045 | 0.02482 | 0.03825 |
| 32 | 0.01 | 0.98 | 0.5 | contractive | 0.00572 | 0.00808 | 0.02215 | 0.02935 |
| 32 | 0.01 | 0.98 | 1.0 | non_decaying | 0.00810 | 0.01045 | 0.02482 | 0.03825 |
| 32 | 0.01 | 0.98 | 1.0 | contractive | 0.00518 | 0.00732 | 0.02113 | 0.02818 |
| 32 | 0.01 | 0.98 | 2.0 | non_decaying | 0.00810 | 0.01045 | 0.02482 | 0.03825 |
| 32 | 0.01 | 0.98 | 2.0 | contractive | 0.00455 | 0.00655 | 0.02062 | 0.02694 |
| 32 | 0.01 | 0.99 | 0.5 | non_decaying | 0.00758 | 0.01184 | 0.02274 | 0.03592 |
| 32 | 0.01 | 0.99 | 0.5 | contractive | 0.00585 | 0.00857 | 0.02008 | 0.02675 |
| 32 | 0.01 | 0.99 | 1.0 | non_decaying | 0.00758 | 0.01184 | 0.02274 | 0.03592 |
| 32 | 0.01 | 0.99 | 1.0 | contractive | 0.00527 | 0.00763 | 0.01906 | 0.02674 |
| 32 | 0.01 | 0.99 | 2.0 | non_decaying | 0.00758 | 0.01184 | 0.02274 | 0.03592 |
| 32 | 0.01 | 0.99 | 2.0 | contractive | 0.00462 | 0.00668 | 0.01812 | 0.02673 |
| 32 | 0.01 | 0.995 | 0.5 | non_decaying | 0.00746 | 0.01259 | 0.02080 | 0.03046 |
| 32 | 0.01 | 0.995 | 0.5 | contractive | 0.00572 | 0.00904 | 0.01759 | 0.02668 |
| 32 | 0.01 | 0.995 | 1.0 | non_decaying | 0.00746 | 0.01259 | 0.02080 | 0.03046 |
| 32 | 0.01 | 0.995 | 1.0 | contractive | 0.00519 | 0.00779 | 0.01653 | 0.02668 |
| 32 | 0.01 | 0.995 | 2.0 | non_decaying | 0.00746 | 0.01259 | 0.02080 | 0.03046 |
| 32 | 0.01 | 0.995 | 2.0 | contractive | 0.00451 | 0.00698 | 0.01603 | 0.02667 |
| 32 | 0.025 | 0.98 | 0.5 | non_decaying | 0.02025 | 0.02611 | 0.06204 | 0.09564 |
| 32 | 0.025 | 0.98 | 0.5 | contractive | 0.01430 | 0.02020 | 0.05536 | 0.07338 |
| 32 | 0.025 | 0.98 | 1.0 | non_decaying | 0.02025 | 0.02611 | 0.06204 | 0.09564 |
| 32 | 0.025 | 0.98 | 1.0 | contractive | 0.01295 | 0.01831 | 0.05282 | 0.07046 |
| 32 | 0.025 | 0.98 | 2.0 | non_decaying | 0.02025 | 0.02611 | 0.06204 | 0.09564 |
| 32 | 0.025 | 0.98 | 2.0 | contractive | 0.01137 | 0.01637 | 0.05156 | 0.06736 |
| 32 | 0.025 | 0.99 | 0.5 | non_decaying | 0.01895 | 0.02961 | 0.05685 | 0.08979 |
| 32 | 0.025 | 0.99 | 0.5 | contractive | 0.01462 | 0.02143 | 0.05020 | 0.06687 |
| 32 | 0.025 | 0.99 | 1.0 | non_decaying | 0.01895 | 0.02961 | 0.05685 | 0.08979 |
| 32 | 0.025 | 0.99 | 1.0 | contractive | 0.01316 | 0.01906 | 0.04765 | 0.06685 |
| 32 | 0.025 | 0.99 | 2.0 | non_decaying | 0.01895 | 0.02961 | 0.05685 | 0.08979 |
| 32 | 0.025 | 0.99 | 2.0 | contractive | 0.01156 | 0.01670 | 0.04530 | 0.06683 |
| 32 | 0.025 | 0.995 | 0.5 | non_decaying | 0.01864 | 0.03147 | 0.05199 | 0.07616 |
| 32 | 0.025 | 0.995 | 0.5 | contractive | 0.01430 | 0.02261 | 0.04396 | 0.06671 |
| 32 | 0.025 | 0.995 | 1.0 | non_decaying | 0.01864 | 0.03147 | 0.05199 | 0.07616 |
| 32 | 0.025 | 0.995 | 1.0 | contractive | 0.01297 | 0.01946 | 0.04133 | 0.06670 |
| 32 | 0.025 | 0.995 | 2.0 | non_decaying | 0.01864 | 0.03147 | 0.05199 | 0.07616 |
| 32 | 0.025 | 0.995 | 2.0 | contractive | 0.01128 | 0.01745 | 0.04008 | 0.06667 |
| 32 | 0.05 | 0.98 | 0.5 | non_decaying | 0.04051 | 0.05223 | 0.12408 | 0.19127 |
| 32 | 0.05 | 0.98 | 0.5 | contractive | 0.02860 | 0.04040 | 0.11073 | 0.14676 |
| 32 | 0.05 | 0.98 | 1.0 | non_decaying | 0.04051 | 0.05223 | 0.12408 | 0.19127 |
| 32 | 0.05 | 0.98 | 1.0 | contractive | 0.02590 | 0.03661 | 0.10564 | 0.14092 |
| 32 | 0.05 | 0.98 | 2.0 | non_decaying | 0.04051 | 0.05223 | 0.12408 | 0.19127 |
| 32 | 0.05 | 0.98 | 2.0 | contractive | 0.02273 | 0.03274 | 0.10312 | 0.13472 |
| 32 | 0.05 | 0.99 | 0.5 | non_decaying | 0.03789 | 0.05921 | 0.11370 | 0.17958 |
| 32 | 0.05 | 0.99 | 0.5 | contractive | 0.02924 | 0.04287 | 0.10040 | 0.13373 |
| 32 | 0.05 | 0.99 | 1.0 | non_decaying | 0.03789 | 0.05921 | 0.11370 | 0.17958 |
| 32 | 0.05 | 0.99 | 1.0 | contractive | 0.02633 | 0.03813 | 0.09530 | 0.13371 |
| 32 | 0.05 | 0.99 | 2.0 | non_decaying | 0.03789 | 0.05921 | 0.11370 | 0.17958 |
| 32 | 0.05 | 0.99 | 2.0 | contractive | 0.02312 | 0.03340 | 0.09059 | 0.13366 |
| 32 | 0.05 | 0.995 | 0.5 | non_decaying | 0.03728 | 0.06294 | 0.10398 | 0.15232 |
| 32 | 0.05 | 0.995 | 0.5 | contractive | 0.02860 | 0.04522 | 0.08793 | 0.13342 |
| 32 | 0.05 | 0.995 | 1.0 | non_decaying | 0.03728 | 0.06294 | 0.10398 | 0.15232 |
| 32 | 0.05 | 0.995 | 1.0 | contractive | 0.02594 | 0.03893 | 0.08267 | 0.13340 |
| 32 | 0.05 | 0.995 | 2.0 | non_decaying | 0.03728 | 0.06294 | 0.10398 | 0.15232 |
| 32 | 0.05 | 0.995 | 2.0 | contractive | 0.02255 | 0.03490 | 0.08017 | 0.13335 |
| 32 | 0.1 | 0.98 | 0.5 | non_decaying | 0.08101 | 0.10446 | 0.24834 | 0.38255 |
| 32 | 0.1 | 0.98 | 0.5 | contractive | 0.05719 | 0.08080 | 0.21586 | 0.29351 |
| 32 | 0.1 | 0.98 | 1.0 | non_decaying | 0.08101 | 0.10446 | 0.24834 | 0.38255 |
| 32 | 0.1 | 0.98 | 1.0 | contractive | 0.05181 | 0.07322 | 0.20906 | 0.28183 |
| 32 | 0.1 | 0.98 | 2.0 | non_decaying | 0.08101 | 0.10446 | 0.24834 | 0.38255 |
| 32 | 0.1 | 0.98 | 2.0 | contractive | 0.04546 | 0.06549 | 0.20062 | 0.26945 |
| 32 | 0.1 | 0.99 | 0.5 | non_decaying | 0.07578 | 0.11843 | 0.22740 | 0.35917 |
| 32 | 0.1 | 0.99 | 0.5 | contractive | 0.05848 | 0.08574 | 0.20081 | 0.26747 |
| 32 | 0.1 | 0.99 | 1.0 | non_decaying | 0.07578 | 0.11843 | 0.22740 | 0.35917 |
| 32 | 0.1 | 0.99 | 1.0 | contractive | 0.05265 | 0.07625 | 0.19059 | 0.26742 |
| 32 | 0.1 | 0.99 | 2.0 | non_decaying | 0.07578 | 0.11843 | 0.22740 | 0.35917 |
| 32 | 0.1 | 0.99 | 2.0 | contractive | 0.04623 | 0.06680 | 0.18118 | 0.26732 |
| 32 | 0.1 | 0.995 | 0.5 | non_decaying | 0.07456 | 0.12589 | 0.20795 | 0.30463 |
| 32 | 0.1 | 0.995 | 0.5 | contractive | 0.05720 | 0.09044 | 0.17586 | 0.26684 |
| 32 | 0.1 | 0.995 | 1.0 | non_decaying | 0.07456 | 0.12589 | 0.20795 | 0.30463 |
| 32 | 0.1 | 0.995 | 1.0 | contractive | 0.05187 | 0.07785 | 0.16533 | 0.26679 |
| 32 | 0.1 | 0.995 | 2.0 | non_decaying | 0.07456 | 0.12589 | 0.20795 | 0.30463 |
| 32 | 0.1 | 0.995 | 2.0 | contractive | 0.04511 | 0.06981 | 0.16033 | 0.26669 |
| 64 | 0.01 | 0.98 | 0.5 | non_decaying | 0.00818 | 0.01045 | 0.02495 | 0.03825 |
| 64 | 0.01 | 0.98 | 0.5 | contractive | 0.00588 | 0.00808 | 0.02155 | 0.03053 |
| 64 | 0.01 | 0.98 | 1.0 | non_decaying | 0.00818 | 0.01045 | 0.02495 | 0.03825 |
| 64 | 0.01 | 0.98 | 1.0 | contractive | 0.00518 | 0.00732 | 0.02102 | 0.02862 |
| 64 | 0.01 | 0.98 | 2.0 | non_decaying | 0.00818 | 0.01045 | 0.02495 | 0.03825 |
| 64 | 0.01 | 0.98 | 2.0 | contractive | 0.00457 | 0.00655 | 0.02041 | 0.02741 |
| 64 | 0.01 | 0.99 | 0.5 | non_decaying | 0.00780 | 0.01184 | 0.02274 | 0.03592 |
| 64 | 0.01 | 0.99 | 0.5 | contractive | 0.00585 | 0.00857 | 0.01965 | 0.02984 |
| 64 | 0.01 | 0.99 | 1.0 | non_decaying | 0.00780 | 0.01184 | 0.02274 | 0.03592 |
| 64 | 0.01 | 0.99 | 1.0 | contractive | 0.00523 | 0.00763 | 0.01881 | 0.02686 |
| 64 | 0.01 | 0.99 | 2.0 | non_decaying | 0.00780 | 0.01184 | 0.02274 | 0.03592 |
| 64 | 0.01 | 0.99 | 2.0 | contractive | 0.00458 | 0.00668 | 0.01840 | 0.02673 |
| 64 | 0.01 | 0.995 | 0.5 | non_decaying | 0.00747 | 0.01259 | 0.02013 | 0.03121 |
| 64 | 0.01 | 0.995 | 0.5 | contractive | 0.00562 | 0.00904 | 0.01698 | 0.02809 |
| 64 | 0.01 | 0.995 | 1.0 | non_decaying | 0.00747 | 0.01259 | 0.02013 | 0.03121 |
| 64 | 0.01 | 0.995 | 1.0 | contractive | 0.00511 | 0.00790 | 0.01625 | 0.02668 |
| 64 | 0.01 | 0.995 | 2.0 | non_decaying | 0.00747 | 0.01259 | 0.02013 | 0.03121 |
| 64 | 0.01 | 0.995 | 2.0 | contractive | 0.00437 | 0.00713 | 0.01590 | 0.02667 |
| 64 | 0.025 | 0.98 | 0.5 | non_decaying | 0.02046 | 0.02611 | 0.06238 | 0.09564 |
| 64 | 0.025 | 0.98 | 0.5 | contractive | 0.01470 | 0.02020 | 0.05387 | 0.07631 |
| 64 | 0.025 | 0.98 | 1.0 | non_decaying | 0.02046 | 0.02611 | 0.06238 | 0.09564 |
| 64 | 0.025 | 0.98 | 1.0 | contractive | 0.01294 | 0.01831 | 0.05254 | 0.07156 |
| 64 | 0.025 | 0.98 | 2.0 | non_decaying | 0.02046 | 0.02611 | 0.06238 | 0.09564 |
| 64 | 0.025 | 0.98 | 2.0 | contractive | 0.01142 | 0.01637 | 0.05103 | 0.06852 |
| 64 | 0.025 | 0.99 | 0.5 | non_decaying | 0.01949 | 0.02961 | 0.05685 | 0.08979 |
| 64 | 0.025 | 0.99 | 0.5 | contractive | 0.01462 | 0.02143 | 0.04913 | 0.07461 |
| 64 | 0.025 | 0.99 | 1.0 | non_decaying | 0.01949 | 0.02961 | 0.05685 | 0.08979 |
| 64 | 0.025 | 0.99 | 1.0 | contractive | 0.01308 | 0.01906 | 0.04703 | 0.06714 |
| 64 | 0.025 | 0.99 | 2.0 | non_decaying | 0.01949 | 0.02961 | 0.05685 | 0.08979 |
| 64 | 0.025 | 0.99 | 2.0 | contractive | 0.01145 | 0.01670 | 0.04599 | 0.06683 |
| 64 | 0.025 | 0.995 | 0.5 | non_decaying | 0.01869 | 0.03147 | 0.05032 | 0.07804 |
| 64 | 0.025 | 0.995 | 0.5 | contractive | 0.01405 | 0.02261 | 0.04246 | 0.07023 |
| 64 | 0.025 | 0.995 | 1.0 | non_decaying | 0.01869 | 0.03147 | 0.05032 | 0.07804 |
| 64 | 0.025 | 0.995 | 1.0 | contractive | 0.01277 | 0.01974 | 0.04063 | 0.06670 |
| 64 | 0.025 | 0.995 | 2.0 | non_decaying | 0.01869 | 0.03147 | 0.05032 | 0.07804 |
| 64 | 0.025 | 0.995 | 2.0 | contractive | 0.01094 | 0.01783 | 0.03974 | 0.06667 |
| 64 | 0.05 | 0.98 | 0.5 | non_decaying | 0.04092 | 0.05223 | 0.12476 | 0.19127 |
| 64 | 0.05 | 0.98 | 0.5 | contractive | 0.02940 | 0.04040 | 0.10774 | 0.15263 |
| 64 | 0.05 | 0.98 | 1.0 | non_decaying | 0.04092 | 0.05223 | 0.12476 | 0.19127 |
| 64 | 0.05 | 0.98 | 1.0 | contractive | 0.02588 | 0.03661 | 0.10508 | 0.14312 |
| 64 | 0.05 | 0.98 | 2.0 | non_decaying | 0.04092 | 0.05223 | 0.12476 | 0.19127 |
| 64 | 0.05 | 0.98 | 2.0 | contractive | 0.02284 | 0.03274 | 0.10206 | 0.13705 |
| 64 | 0.05 | 0.99 | 0.5 | non_decaying | 0.03898 | 0.05921 | 0.11370 | 0.17958 |
| 64 | 0.05 | 0.99 | 0.5 | contractive | 0.02923 | 0.04287 | 0.09826 | 0.14921 |
| 64 | 0.05 | 0.99 | 1.0 | non_decaying | 0.03898 | 0.05921 | 0.11370 | 0.17958 |
| 64 | 0.05 | 0.99 | 1.0 | contractive | 0.02616 | 0.03813 | 0.09405 | 0.13429 |
| 64 | 0.05 | 0.99 | 2.0 | non_decaying | 0.03898 | 0.05921 | 0.11370 | 0.17958 |
| 64 | 0.05 | 0.99 | 2.0 | contractive | 0.02289 | 0.03340 | 0.09198 | 0.13366 |
| 64 | 0.05 | 0.995 | 0.5 | non_decaying | 0.03737 | 0.06294 | 0.10064 | 0.15607 |
| 64 | 0.05 | 0.995 | 0.5 | contractive | 0.02810 | 0.04522 | 0.08492 | 0.14046 |
| 64 | 0.05 | 0.995 | 1.0 | non_decaying | 0.03737 | 0.06294 | 0.10064 | 0.15607 |
| 64 | 0.05 | 0.995 | 1.0 | contractive | 0.02554 | 0.03949 | 0.08126 | 0.13340 |
| 64 | 0.05 | 0.995 | 2.0 | non_decaying | 0.03737 | 0.06294 | 0.10064 | 0.15607 |
| 64 | 0.05 | 0.995 | 2.0 | contractive | 0.02187 | 0.03565 | 0.07948 | 0.13335 |
| 64 | 0.1 | 0.98 | 0.5 | non_decaying | 0.08195 | 0.12011 | 0.25670 | 0.38255 |
| 64 | 0.1 | 0.98 | 0.5 | contractive | 0.05879 | 0.08552 | 0.21609 | 0.30525 |
| 64 | 0.1 | 0.98 | 1.0 | non_decaying | 0.08195 | 0.12011 | 0.25670 | 0.38255 |
| 64 | 0.1 | 0.98 | 1.0 | contractive | 0.05227 | 0.07322 | 0.21102 | 0.28624 |
| 64 | 0.1 | 0.98 | 2.0 | non_decaying | 0.08195 | 0.12011 | 0.25670 | 0.38255 |
| 64 | 0.1 | 0.98 | 2.0 | contractive | 0.04581 | 0.06549 | 0.20284 | 0.27410 |
| 64 | 0.1 | 0.99 | 0.5 | non_decaying | 0.07893 | 0.11843 | 0.22891 | 0.35917 |
| 64 | 0.1 | 0.99 | 0.5 | contractive | 0.05851 | 0.08574 | 0.19735 | 0.29842 |
| 64 | 0.1 | 0.99 | 1.0 | non_decaying | 0.07893 | 0.11843 | 0.22891 | 0.35917 |
| 64 | 0.1 | 0.99 | 1.0 | contractive | 0.05242 | 0.07625 | 0.18992 | 0.26858 |
| 64 | 0.1 | 0.99 | 2.0 | non_decaying | 0.07893 | 0.11843 | 0.22891 | 0.35917 |
| 64 | 0.1 | 0.99 | 2.0 | contractive | 0.04619 | 0.06720 | 0.18445 | 0.26732 |
| 64 | 0.1 | 0.995 | 0.5 | non_decaying | 0.07475 | 0.12589 | 0.20128 | 0.31214 |
| 64 | 0.1 | 0.995 | 0.5 | contractive | 0.05619 | 0.09044 | 0.16984 | 0.28093 |
| 64 | 0.1 | 0.995 | 1.0 | non_decaying | 0.07475 | 0.12589 | 0.20128 | 0.31214 |
| 64 | 0.1 | 0.995 | 1.0 | contractive | 0.05108 | 0.07898 | 0.16252 | 0.26679 |
| 64 | 0.1 | 0.995 | 2.0 | non_decaying | 0.07475 | 0.12589 | 0.20128 | 0.31214 |
| 64 | 0.1 | 0.995 | 2.0 | contractive | 0.04375 | 0.07131 | 0.15896 | 0.26669 |
