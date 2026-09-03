# Augmentation validation

- Dataset: `processed-20260903-ce343c8ce6a5` (payload `ce343c8ce6a5`)
- Scenario: task-1a-reach; derivative policy: central-difference
- Seed namespace: `[415926535, seed_bank, attempt]` (no evaluation seed consumed)
- Locked taper: smoothstep over 0.2 s, exact zero from 0.1 s before dwell onset (dwell starts at 2.999999999999999 s)
- Outcome: **PASS**

## Global checks

| check | outcome | detail |
| --- | --- | --- |
| source-binding | PASS | payload matches processed-20260903-ce343c8ce6a5 |
| bank-separation | PASS | banks 1 and 2 differ |

## Configurations

| N_aug | sigma (rad) | phi | gamma | attempts | rejected attempts | family rejections | determinism | bounds | smoothness | correlation | envelope | dwell-collapse | episode-separation | rejection-accounting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.01 | 0.98 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.98 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.98 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.99 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.99 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.99 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.995 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.995 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.01 | 0.995 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.98 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.98 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.98 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.99 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.99 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.99 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.995 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.995 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.025 | 0.995 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.98 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.98 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.98 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.99 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.99 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.99 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.995 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.995 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.05 | 0.995 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.98 | 0.5 | 19 | 3 | 6 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.98 | 1.0 | 19 | 3 | 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.98 | 2.0 | 19 | 3 | 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.99 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.99 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.99 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.995 | 0.5 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.995 | 1.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 16 | 0.1 | 0.995 | 2.0 | 16 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.98 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.98 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.98 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.99 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.99 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.99 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.995 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.995 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.01 | 0.995 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.98 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.98 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.98 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.99 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.99 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.99 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.995 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.995 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.025 | 0.995 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.98 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.98 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.98 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.99 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.99 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.99 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.995 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.995 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.05 | 0.995 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.98 | 0.5 | 35 | 3 | 6 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.98 | 1.0 | 35 | 3 | 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.98 | 2.0 | 35 | 3 | 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.99 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.99 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.99 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.995 | 0.5 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.995 | 1.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 32 | 0.1 | 0.995 | 2.0 | 32 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.98 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.98 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.98 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.99 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.99 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.99 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.995 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.995 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.01 | 0.995 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.98 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.98 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.98 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.99 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.99 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.99 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.995 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.995 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.025 | 0.995 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.98 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.98 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.98 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.99 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.99 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.99 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.995 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.995 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.05 | 0.995 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.98 | 0.5 | 74 | 10 | 20 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.98 | 1.0 | 74 | 10 | 19 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.98 | 2.0 | 74 | 10 | 19 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.99 | 0.5 | 65 | 1 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.99 | 1.0 | 65 | 1 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.99 | 2.0 | 65 | 1 | 2 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.995 | 0.5 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.995 | 1.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| 64 | 0.1 | 0.995 | 2.0 | 64 | 0 | 0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Rejected attempts

Rejections are expected under the bounded resampling protocol; they are recorded, never hidden.

- (16, 0.1, 0.98, 0.5) attempt 4, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 0.5) attempt 4, contractive: velocity limit violated
- (16, 0.1, 0.98, 0.5) attempt 10, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 0.5) attempt 10, contractive: velocity limit violated
- (16, 0.1, 0.98, 0.5) attempt 12, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 0.5) attempt 12, contractive: velocity limit violated
- (16, 0.1, 0.98, 1.0) attempt 4, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 1.0) attempt 4, contractive: velocity limit violated
- (16, 0.1, 0.98, 1.0) attempt 10, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 1.0) attempt 10, contractive: velocity limit violated
- (16, 0.1, 0.98, 1.0) attempt 12, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 2.0) attempt 4, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 2.0) attempt 4, contractive: velocity limit violated
- (16, 0.1, 0.98, 2.0) attempt 10, non_decaying: velocity limit violated
- (16, 0.1, 0.98, 2.0) attempt 10, contractive: velocity limit violated
- (16, 0.1, 0.98, 2.0) attempt 12, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 0.5) attempt 4, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 0.5) attempt 4, contractive: velocity limit violated
- (32, 0.1, 0.98, 0.5) attempt 10, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 0.5) attempt 10, contractive: velocity limit violated
- (32, 0.1, 0.98, 0.5) attempt 12, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 0.5) attempt 12, contractive: velocity limit violated
- (32, 0.1, 0.98, 1.0) attempt 4, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 1.0) attempt 4, contractive: velocity limit violated
- (32, 0.1, 0.98, 1.0) attempt 10, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 1.0) attempt 10, contractive: velocity limit violated
- (32, 0.1, 0.98, 1.0) attempt 12, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 2.0) attempt 4, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 2.0) attempt 4, contractive: velocity limit violated
- (32, 0.1, 0.98, 2.0) attempt 10, non_decaying: velocity limit violated
- (32, 0.1, 0.98, 2.0) attempt 10, contractive: velocity limit violated
- (32, 0.1, 0.98, 2.0) attempt 12, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 4, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 4, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 10, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 10, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 12, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 12, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 36, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 36, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 39, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 39, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 45, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 45, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 50, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 50, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 52, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 52, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 65, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 65, contractive: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 71, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 0.5) attempt 71, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 4, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 4, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 10, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 10, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 12, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 36, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 36, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 39, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 39, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 45, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 45, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 50, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 50, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 52, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 52, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 65, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 65, contractive: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 71, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 1.0) attempt 71, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 4, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 4, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 10, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 10, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 12, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 36, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 36, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 39, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 39, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 45, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 45, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 50, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 50, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 52, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 52, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 65, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 65, contractive: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 71, non_decaying: velocity limit violated
- (64, 0.1, 0.98, 2.0) attempt 71, contractive: velocity limit violated
- (64, 0.1, 0.99, 0.5) attempt 36, non_decaying: velocity limit violated
- (64, 0.1, 0.99, 0.5) attempt 36, contractive: velocity limit violated
- (64, 0.1, 0.99, 1.0) attempt 36, non_decaying: velocity limit violated
- (64, 0.1, 0.99, 1.0) attempt 36, contractive: velocity limit violated
- (64, 0.1, 0.99, 2.0) attempt 36, non_decaying: velocity limit violated
- (64, 0.1, 0.99, 2.0) attempt 36, contractive: velocity limit violated

## Realized amplitudes (per family, rad)

| N_aug | sigma | phi | gamma | family | rms median | rms max | peak median | peak max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.01 | 0.98 | 0.5 | non_decaying | 0.00756 | 0.01303 | 0.02611 | 0.04057 |
| 16 | 0.01 | 0.98 | 0.5 | contractive | 0.00568 | 0.00988 | 0.02041 | 0.03555 |
| 16 | 0.01 | 0.98 | 1.0 | non_decaying | 0.00756 | 0.01303 | 0.02611 | 0.04057 |
| 16 | 0.01 | 0.98 | 1.0 | contractive | 0.00477 | 0.00840 | 0.01966 | 0.03152 |
| 16 | 0.01 | 0.98 | 2.0 | non_decaying | 0.00756 | 0.01303 | 0.02611 | 0.04057 |
| 16 | 0.01 | 0.98 | 2.0 | contractive | 0.00418 | 0.00732 | 0.01961 | 0.03053 |
| 16 | 0.01 | 0.99 | 0.5 | non_decaying | 0.00718 | 0.01430 | 0.02445 | 0.03725 |
| 16 | 0.01 | 0.99 | 0.5 | contractive | 0.00551 | 0.00967 | 0.01887 | 0.03092 |
| 16 | 0.01 | 0.99 | 1.0 | non_decaying | 0.00718 | 0.01430 | 0.02445 | 0.03725 |
| 16 | 0.01 | 0.99 | 1.0 | contractive | 0.00467 | 0.00835 | 0.01887 | 0.02685 |
| 16 | 0.01 | 0.99 | 2.0 | non_decaying | 0.00718 | 0.01430 | 0.02445 | 0.03725 |
| 16 | 0.01 | 0.99 | 2.0 | contractive | 0.00421 | 0.00698 | 0.01883 | 0.02557 |
| 16 | 0.01 | 0.995 | 0.5 | non_decaying | 0.00727 | 0.01386 | 0.02037 | 0.03527 |
| 16 | 0.01 | 0.995 | 0.5 | contractive | 0.00538 | 0.00941 | 0.01860 | 0.02700 |
| 16 | 0.01 | 0.995 | 1.0 | non_decaying | 0.00727 | 0.01386 | 0.02037 | 0.03527 |
| 16 | 0.01 | 0.995 | 1.0 | contractive | 0.00498 | 0.00810 | 0.01849 | 0.02500 |
| 16 | 0.01 | 0.995 | 2.0 | non_decaying | 0.00727 | 0.01386 | 0.02037 | 0.03527 |
| 16 | 0.01 | 0.995 | 2.0 | contractive | 0.00434 | 0.00675 | 0.01789 | 0.02184 |
| 16 | 0.025 | 0.98 | 0.5 | non_decaying | 0.01891 | 0.03256 | 0.06526 | 0.10142 |
| 16 | 0.025 | 0.98 | 0.5 | contractive | 0.01420 | 0.02471 | 0.05102 | 0.08888 |
| 16 | 0.025 | 0.98 | 1.0 | non_decaying | 0.01891 | 0.03256 | 0.06526 | 0.10142 |
| 16 | 0.025 | 0.98 | 1.0 | contractive | 0.01194 | 0.02099 | 0.04914 | 0.07879 |
| 16 | 0.025 | 0.98 | 2.0 | non_decaying | 0.01891 | 0.03256 | 0.06526 | 0.10142 |
| 16 | 0.025 | 0.98 | 2.0 | contractive | 0.01045 | 0.01831 | 0.04903 | 0.07632 |
| 16 | 0.025 | 0.99 | 0.5 | non_decaying | 0.01795 | 0.03574 | 0.06112 | 0.09313 |
| 16 | 0.025 | 0.99 | 0.5 | contractive | 0.01378 | 0.02418 | 0.04717 | 0.07731 |
| 16 | 0.025 | 0.99 | 1.0 | non_decaying | 0.01795 | 0.03574 | 0.06112 | 0.09313 |
| 16 | 0.025 | 0.99 | 1.0 | contractive | 0.01168 | 0.02088 | 0.04717 | 0.06714 |
| 16 | 0.025 | 0.99 | 2.0 | non_decaying | 0.01795 | 0.03574 | 0.06112 | 0.09313 |
| 16 | 0.025 | 0.99 | 2.0 | contractive | 0.01053 | 0.01745 | 0.04707 | 0.06392 |
| 16 | 0.025 | 0.995 | 0.5 | non_decaying | 0.01817 | 0.03465 | 0.05092 | 0.08818 |
| 16 | 0.025 | 0.995 | 0.5 | contractive | 0.01345 | 0.02353 | 0.04650 | 0.06750 |
| 16 | 0.025 | 0.995 | 1.0 | non_decaying | 0.01817 | 0.03465 | 0.05092 | 0.08818 |
| 16 | 0.025 | 0.995 | 1.0 | contractive | 0.01245 | 0.02026 | 0.04622 | 0.06249 |
| 16 | 0.025 | 0.995 | 2.0 | non_decaying | 0.01817 | 0.03465 | 0.05092 | 0.08818 |
| 16 | 0.025 | 0.995 | 2.0 | contractive | 0.01085 | 0.01687 | 0.04472 | 0.05460 |
| 16 | 0.05 | 0.98 | 0.5 | non_decaying | 0.03781 | 0.06513 | 0.13053 | 0.20283 |
| 16 | 0.05 | 0.98 | 0.5 | contractive | 0.02840 | 0.04942 | 0.10204 | 0.17776 |
| 16 | 0.05 | 0.98 | 1.0 | non_decaying | 0.03781 | 0.06513 | 0.13053 | 0.20283 |
| 16 | 0.05 | 0.98 | 1.0 | contractive | 0.02387 | 0.04198 | 0.09829 | 0.15758 |
| 16 | 0.05 | 0.98 | 2.0 | non_decaying | 0.03781 | 0.06513 | 0.13053 | 0.20283 |
| 16 | 0.05 | 0.98 | 2.0 | contractive | 0.02090 | 0.03662 | 0.09806 | 0.15264 |
| 16 | 0.05 | 0.99 | 0.5 | non_decaying | 0.03591 | 0.07148 | 0.12225 | 0.18627 |
| 16 | 0.05 | 0.99 | 0.5 | contractive | 0.02756 | 0.04837 | 0.09434 | 0.15462 |
| 16 | 0.05 | 0.99 | 1.0 | non_decaying | 0.03591 | 0.07148 | 0.12225 | 0.18627 |
| 16 | 0.05 | 0.99 | 1.0 | contractive | 0.02336 | 0.04176 | 0.09434 | 0.13427 |
| 16 | 0.05 | 0.99 | 2.0 | non_decaying | 0.03591 | 0.07148 | 0.12225 | 0.18627 |
| 16 | 0.05 | 0.99 | 2.0 | contractive | 0.02107 | 0.03491 | 0.09414 | 0.12784 |
| 16 | 0.05 | 0.995 | 0.5 | non_decaying | 0.03635 | 0.06931 | 0.10183 | 0.17636 |
| 16 | 0.05 | 0.995 | 0.5 | contractive | 0.02689 | 0.04706 | 0.09299 | 0.13501 |
| 16 | 0.05 | 0.995 | 1.0 | non_decaying | 0.03635 | 0.06931 | 0.10183 | 0.17636 |
| 16 | 0.05 | 0.995 | 1.0 | contractive | 0.02489 | 0.04051 | 0.09244 | 0.12498 |
| 16 | 0.05 | 0.995 | 2.0 | non_decaying | 0.03635 | 0.06931 | 0.10183 | 0.17636 |
| 16 | 0.05 | 0.995 | 2.0 | contractive | 0.02171 | 0.03373 | 0.08945 | 0.10920 |
| 16 | 0.1 | 0.98 | 0.5 | non_decaying | 0.07826 | 0.13026 | 0.26105 | 0.40567 |
| 16 | 0.1 | 0.98 | 0.5 | contractive | 0.05843 | 0.09884 | 0.20408 | 0.35551 |
| 16 | 0.1 | 0.98 | 1.0 | non_decaying | 0.07826 | 0.13026 | 0.26105 | 0.40567 |
| 16 | 0.1 | 0.98 | 1.0 | contractive | 0.04962 | 0.08397 | 0.19657 | 0.31516 |
| 16 | 0.1 | 0.98 | 2.0 | non_decaying | 0.07826 | 0.13026 | 0.26105 | 0.40567 |
| 16 | 0.1 | 0.98 | 2.0 | contractive | 0.04246 | 0.07324 | 0.19440 | 0.30527 |
| 16 | 0.1 | 0.99 | 0.5 | non_decaying | 0.07181 | 0.14295 | 0.24450 | 0.37253 |
| 16 | 0.1 | 0.99 | 0.5 | contractive | 0.05511 | 0.09673 | 0.18868 | 0.30924 |
| 16 | 0.1 | 0.99 | 1.0 | non_decaying | 0.07181 | 0.14295 | 0.24450 | 0.37253 |
| 16 | 0.1 | 0.99 | 1.0 | contractive | 0.04673 | 0.08352 | 0.18868 | 0.26854 |
| 16 | 0.1 | 0.99 | 2.0 | non_decaying | 0.07181 | 0.14295 | 0.24450 | 0.37253 |
| 16 | 0.1 | 0.99 | 2.0 | contractive | 0.04213 | 0.06981 | 0.18828 | 0.25569 |
| 16 | 0.1 | 0.995 | 0.5 | non_decaying | 0.07269 | 0.13862 | 0.20367 | 0.35272 |
| 16 | 0.1 | 0.995 | 0.5 | contractive | 0.05378 | 0.09412 | 0.18598 | 0.27001 |
| 16 | 0.1 | 0.995 | 1.0 | non_decaying | 0.07269 | 0.13862 | 0.20367 | 0.35272 |
| 16 | 0.1 | 0.995 | 1.0 | contractive | 0.04978 | 0.08103 | 0.18487 | 0.24996 |
| 16 | 0.1 | 0.995 | 2.0 | non_decaying | 0.07269 | 0.13862 | 0.20367 | 0.35272 |
| 16 | 0.1 | 0.995 | 2.0 | contractive | 0.04341 | 0.06746 | 0.17890 | 0.21840 |
| 32 | 0.01 | 0.98 | 0.5 | non_decaying | 0.00777 | 0.01303 | 0.02468 | 0.04057 |
| 32 | 0.01 | 0.98 | 0.5 | contractive | 0.00581 | 0.00988 | 0.02074 | 0.03555 |
| 32 | 0.01 | 0.98 | 1.0 | non_decaying | 0.00777 | 0.01303 | 0.02468 | 0.04057 |
| 32 | 0.01 | 0.98 | 1.0 | contractive | 0.00480 | 0.00840 | 0.01955 | 0.03152 |
| 32 | 0.01 | 0.98 | 2.0 | non_decaying | 0.00777 | 0.01303 | 0.02468 | 0.04057 |
| 32 | 0.01 | 0.98 | 2.0 | contractive | 0.00418 | 0.00732 | 0.01922 | 0.03053 |
| 32 | 0.01 | 0.99 | 0.5 | non_decaying | 0.00805 | 0.01430 | 0.02305 | 0.03725 |
| 32 | 0.01 | 0.99 | 0.5 | contractive | 0.00551 | 0.00967 | 0.01861 | 0.03092 |
| 32 | 0.01 | 0.99 | 1.0 | non_decaying | 0.00805 | 0.01430 | 0.02305 | 0.03725 |
| 32 | 0.01 | 0.99 | 1.0 | contractive | 0.00456 | 0.00835 | 0.01836 | 0.02685 |
| 32 | 0.01 | 0.99 | 2.0 | non_decaying | 0.00805 | 0.01430 | 0.02305 | 0.03725 |
| 32 | 0.01 | 0.99 | 2.0 | contractive | 0.00400 | 0.00698 | 0.01779 | 0.02557 |
| 32 | 0.01 | 0.995 | 0.5 | non_decaying | 0.00745 | 0.01386 | 0.02050 | 0.03527 |
| 32 | 0.01 | 0.995 | 0.5 | contractive | 0.00538 | 0.00941 | 0.01781 | 0.02700 |
| 32 | 0.01 | 0.995 | 1.0 | non_decaying | 0.00745 | 0.01386 | 0.02050 | 0.03527 |
| 32 | 0.01 | 0.995 | 1.0 | contractive | 0.00476 | 0.00810 | 0.01717 | 0.02580 |
| 32 | 0.01 | 0.995 | 2.0 | non_decaying | 0.00745 | 0.01386 | 0.02050 | 0.03527 |
| 32 | 0.01 | 0.995 | 2.0 | contractive | 0.00422 | 0.00675 | 0.01652 | 0.02565 |
| 32 | 0.025 | 0.98 | 0.5 | non_decaying | 0.01942 | 0.03256 | 0.06171 | 0.10142 |
| 32 | 0.025 | 0.98 | 0.5 | contractive | 0.01453 | 0.02471 | 0.05185 | 0.08888 |
| 32 | 0.025 | 0.98 | 1.0 | non_decaying | 0.01942 | 0.03256 | 0.06171 | 0.10142 |
| 32 | 0.025 | 0.98 | 1.0 | contractive | 0.01201 | 0.02099 | 0.04888 | 0.07879 |
| 32 | 0.025 | 0.98 | 2.0 | non_decaying | 0.01942 | 0.03256 | 0.06171 | 0.10142 |
| 32 | 0.025 | 0.98 | 2.0 | contractive | 0.01045 | 0.01831 | 0.04805 | 0.07632 |
| 32 | 0.025 | 0.99 | 0.5 | non_decaying | 0.02012 | 0.03574 | 0.05763 | 0.09313 |
| 32 | 0.025 | 0.99 | 0.5 | contractive | 0.01378 | 0.02418 | 0.04652 | 0.07731 |
| 32 | 0.025 | 0.99 | 1.0 | non_decaying | 0.02012 | 0.03574 | 0.05763 | 0.09313 |
| 32 | 0.025 | 0.99 | 1.0 | contractive | 0.01140 | 0.02088 | 0.04590 | 0.06714 |
| 32 | 0.025 | 0.99 | 2.0 | non_decaying | 0.02012 | 0.03574 | 0.05763 | 0.09313 |
| 32 | 0.025 | 0.99 | 2.0 | contractive | 0.01000 | 0.01745 | 0.04447 | 0.06392 |
| 32 | 0.025 | 0.995 | 0.5 | non_decaying | 0.01863 | 0.03465 | 0.05125 | 0.08818 |
| 32 | 0.025 | 0.995 | 0.5 | contractive | 0.01345 | 0.02353 | 0.04453 | 0.06750 |
| 32 | 0.025 | 0.995 | 1.0 | non_decaying | 0.01863 | 0.03465 | 0.05125 | 0.08818 |
| 32 | 0.025 | 0.995 | 1.0 | contractive | 0.01190 | 0.02026 | 0.04292 | 0.06449 |
| 32 | 0.025 | 0.995 | 2.0 | non_decaying | 0.01863 | 0.03465 | 0.05125 | 0.08818 |
| 32 | 0.025 | 0.995 | 2.0 | contractive | 0.01055 | 0.01687 | 0.04129 | 0.06413 |
| 32 | 0.05 | 0.98 | 0.5 | non_decaying | 0.03883 | 0.06513 | 0.12342 | 0.20283 |
| 32 | 0.05 | 0.98 | 0.5 | contractive | 0.02907 | 0.04942 | 0.10369 | 0.17776 |
| 32 | 0.05 | 0.98 | 1.0 | non_decaying | 0.03883 | 0.06513 | 0.12342 | 0.20283 |
| 32 | 0.05 | 0.98 | 1.0 | contractive | 0.02402 | 0.04198 | 0.09777 | 0.15758 |
| 32 | 0.05 | 0.98 | 2.0 | non_decaying | 0.03883 | 0.06513 | 0.12342 | 0.20283 |
| 32 | 0.05 | 0.98 | 2.0 | contractive | 0.02090 | 0.03662 | 0.09610 | 0.15264 |
| 32 | 0.05 | 0.99 | 0.5 | non_decaying | 0.04023 | 0.07148 | 0.11526 | 0.18627 |
| 32 | 0.05 | 0.99 | 0.5 | contractive | 0.02756 | 0.04837 | 0.09305 | 0.15462 |
| 32 | 0.05 | 0.99 | 1.0 | non_decaying | 0.04023 | 0.07148 | 0.11526 | 0.18627 |
| 32 | 0.05 | 0.99 | 1.0 | contractive | 0.02280 | 0.04176 | 0.09180 | 0.13427 |
| 32 | 0.05 | 0.99 | 2.0 | non_decaying | 0.04023 | 0.07148 | 0.11526 | 0.18627 |
| 32 | 0.05 | 0.99 | 2.0 | contractive | 0.02000 | 0.03491 | 0.08895 | 0.12784 |
| 32 | 0.05 | 0.995 | 0.5 | non_decaying | 0.03725 | 0.06931 | 0.10250 | 0.17636 |
| 32 | 0.05 | 0.995 | 0.5 | contractive | 0.02689 | 0.04706 | 0.08906 | 0.13501 |
| 32 | 0.05 | 0.995 | 1.0 | non_decaying | 0.03725 | 0.06931 | 0.10250 | 0.17636 |
| 32 | 0.05 | 0.995 | 1.0 | contractive | 0.02379 | 0.04051 | 0.08584 | 0.12899 |
| 32 | 0.05 | 0.995 | 2.0 | non_decaying | 0.03725 | 0.06931 | 0.10250 | 0.17636 |
| 32 | 0.05 | 0.995 | 2.0 | contractive | 0.02110 | 0.03373 | 0.08258 | 0.12826 |
| 32 | 0.1 | 0.98 | 0.5 | non_decaying | 0.07767 | 0.13026 | 0.24537 | 0.40567 |
| 32 | 0.1 | 0.98 | 0.5 | contractive | 0.05814 | 0.09884 | 0.20909 | 0.35551 |
| 32 | 0.1 | 0.98 | 1.0 | non_decaying | 0.07767 | 0.13026 | 0.24537 | 0.40567 |
| 32 | 0.1 | 0.98 | 1.0 | contractive | 0.04962 | 0.08488 | 0.19813 | 0.32371 |
| 32 | 0.1 | 0.98 | 2.0 | non_decaying | 0.07767 | 0.13026 | 0.24537 | 0.40567 |
| 32 | 0.1 | 0.98 | 2.0 | contractive | 0.04246 | 0.07583 | 0.19378 | 0.31632 |
| 32 | 0.1 | 0.99 | 0.5 | non_decaying | 0.08046 | 0.14295 | 0.23053 | 0.37253 |
| 32 | 0.1 | 0.99 | 0.5 | contractive | 0.05511 | 0.09673 | 0.18610 | 0.30924 |
| 32 | 0.1 | 0.99 | 1.0 | non_decaying | 0.08046 | 0.14295 | 0.23053 | 0.37253 |
| 32 | 0.1 | 0.99 | 1.0 | contractive | 0.04560 | 0.08352 | 0.18359 | 0.26854 |
| 32 | 0.1 | 0.99 | 2.0 | non_decaying | 0.08046 | 0.14295 | 0.23053 | 0.37253 |
| 32 | 0.1 | 0.99 | 2.0 | contractive | 0.03999 | 0.06981 | 0.17789 | 0.25569 |
| 32 | 0.1 | 0.995 | 0.5 | non_decaying | 0.07451 | 0.13862 | 0.20500 | 0.35272 |
| 32 | 0.1 | 0.995 | 0.5 | contractive | 0.05378 | 0.09412 | 0.17812 | 0.27001 |
| 32 | 0.1 | 0.995 | 1.0 | non_decaying | 0.07451 | 0.13862 | 0.20500 | 0.35272 |
| 32 | 0.1 | 0.995 | 1.0 | contractive | 0.04758 | 0.08103 | 0.17168 | 0.25797 |
| 32 | 0.1 | 0.995 | 2.0 | non_decaying | 0.07451 | 0.13862 | 0.20500 | 0.35272 |
| 32 | 0.1 | 0.995 | 2.0 | contractive | 0.04221 | 0.06746 | 0.16517 | 0.25652 |
| 64 | 0.01 | 0.98 | 0.5 | non_decaying | 0.00825 | 0.01303 | 0.02570 | 0.04057 |
| 64 | 0.01 | 0.98 | 0.5 | contractive | 0.00572 | 0.01016 | 0.02062 | 0.03700 |
| 64 | 0.01 | 0.98 | 1.0 | non_decaying | 0.00825 | 0.01303 | 0.02570 | 0.04057 |
| 64 | 0.01 | 0.98 | 1.0 | contractive | 0.00492 | 0.00972 | 0.01978 | 0.03657 |
| 64 | 0.01 | 0.98 | 2.0 | non_decaying | 0.00825 | 0.01303 | 0.02570 | 0.04057 |
| 64 | 0.01 | 0.98 | 2.0 | contractive | 0.00430 | 0.00923 | 0.01914 | 0.03574 |
| 64 | 0.01 | 0.99 | 0.5 | non_decaying | 0.00804 | 0.01430 | 0.02308 | 0.03829 |
| 64 | 0.01 | 0.99 | 0.5 | contractive | 0.00553 | 0.01152 | 0.01887 | 0.03785 |
| 64 | 0.01 | 0.99 | 1.0 | non_decaying | 0.00804 | 0.01430 | 0.02308 | 0.03829 |
| 64 | 0.01 | 0.99 | 1.0 | contractive | 0.00475 | 0.01093 | 0.01831 | 0.03741 |
| 64 | 0.01 | 0.99 | 2.0 | non_decaying | 0.00804 | 0.01430 | 0.02308 | 0.03829 |
| 64 | 0.01 | 0.99 | 2.0 | contractive | 0.00410 | 0.01018 | 0.01757 | 0.03656 |
| 64 | 0.01 | 0.995 | 0.5 | non_decaying | 0.00742 | 0.01447 | 0.02077 | 0.03670 |
| 64 | 0.01 | 0.995 | 0.5 | contractive | 0.00528 | 0.01244 | 0.01747 | 0.03628 |
| 64 | 0.01 | 0.995 | 1.0 | non_decaying | 0.00742 | 0.01447 | 0.02077 | 0.03670 |
| 64 | 0.01 | 0.995 | 1.0 | contractive | 0.00457 | 0.01155 | 0.01670 | 0.03586 |
| 64 | 0.01 | 0.995 | 2.0 | non_decaying | 0.00742 | 0.01447 | 0.02077 | 0.03670 |
| 64 | 0.01 | 0.995 | 2.0 | contractive | 0.00415 | 0.01050 | 0.01632 | 0.03504 |
| 64 | 0.025 | 0.98 | 0.5 | non_decaying | 0.02063 | 0.03256 | 0.06425 | 0.10142 |
| 64 | 0.025 | 0.98 | 0.5 | contractive | 0.01431 | 0.02540 | 0.05154 | 0.09250 |
| 64 | 0.025 | 0.98 | 1.0 | non_decaying | 0.02063 | 0.03256 | 0.06425 | 0.10142 |
| 64 | 0.025 | 0.98 | 1.0 | contractive | 0.01229 | 0.02429 | 0.04945 | 0.09143 |
| 64 | 0.025 | 0.98 | 2.0 | non_decaying | 0.02063 | 0.03256 | 0.06425 | 0.10142 |
| 64 | 0.025 | 0.98 | 2.0 | contractive | 0.01075 | 0.02309 | 0.04785 | 0.08934 |
| 64 | 0.025 | 0.99 | 0.5 | non_decaying | 0.02011 | 0.03574 | 0.05769 | 0.09572 |
| 64 | 0.025 | 0.99 | 0.5 | contractive | 0.01381 | 0.02880 | 0.04717 | 0.09462 |
| 64 | 0.025 | 0.99 | 1.0 | non_decaying | 0.02011 | 0.03574 | 0.05769 | 0.09572 |
| 64 | 0.025 | 0.99 | 1.0 | contractive | 0.01188 | 0.02732 | 0.04578 | 0.09354 |
| 64 | 0.025 | 0.99 | 2.0 | non_decaying | 0.02011 | 0.03574 | 0.05769 | 0.09572 |
| 64 | 0.025 | 0.99 | 2.0 | contractive | 0.01025 | 0.02545 | 0.04394 | 0.09140 |
| 64 | 0.025 | 0.995 | 0.5 | non_decaying | 0.01854 | 0.03619 | 0.05192 | 0.09175 |
| 64 | 0.025 | 0.995 | 0.5 | contractive | 0.01319 | 0.03111 | 0.04367 | 0.09069 |
| 64 | 0.025 | 0.995 | 1.0 | non_decaying | 0.01854 | 0.03619 | 0.05192 | 0.09175 |
| 64 | 0.025 | 0.995 | 1.0 | contractive | 0.01143 | 0.02888 | 0.04174 | 0.08965 |
| 64 | 0.025 | 0.995 | 2.0 | non_decaying | 0.01854 | 0.03619 | 0.05192 | 0.09175 |
| 64 | 0.025 | 0.995 | 2.0 | contractive | 0.01036 | 0.02625 | 0.04079 | 0.08760 |
| 64 | 0.05 | 0.98 | 0.5 | non_decaying | 0.04126 | 0.06513 | 0.12850 | 0.20283 |
| 64 | 0.05 | 0.98 | 0.5 | contractive | 0.02862 | 0.05081 | 0.10308 | 0.18499 |
| 64 | 0.05 | 0.98 | 1.0 | non_decaying | 0.04126 | 0.06513 | 0.12850 | 0.20283 |
| 64 | 0.05 | 0.98 | 1.0 | contractive | 0.02458 | 0.04858 | 0.09890 | 0.18287 |
| 64 | 0.05 | 0.98 | 2.0 | non_decaying | 0.04126 | 0.06513 | 0.12850 | 0.20283 |
| 64 | 0.05 | 0.98 | 2.0 | contractive | 0.02149 | 0.04617 | 0.09570 | 0.17869 |
| 64 | 0.05 | 0.99 | 0.5 | non_decaying | 0.04021 | 0.07148 | 0.11538 | 0.19145 |
| 64 | 0.05 | 0.99 | 0.5 | contractive | 0.02763 | 0.05760 | 0.09434 | 0.18925 |
| 64 | 0.05 | 0.99 | 1.0 | non_decaying | 0.04021 | 0.07148 | 0.11538 | 0.19145 |
| 64 | 0.05 | 0.99 | 1.0 | contractive | 0.02376 | 0.05463 | 0.09155 | 0.18707 |
| 64 | 0.05 | 0.99 | 2.0 | non_decaying | 0.04021 | 0.07148 | 0.11538 | 0.19145 |
| 64 | 0.05 | 0.99 | 2.0 | contractive | 0.02049 | 0.05091 | 0.08787 | 0.18280 |
| 64 | 0.05 | 0.995 | 0.5 | non_decaying | 0.03708 | 0.07237 | 0.10383 | 0.18350 |
| 64 | 0.05 | 0.995 | 0.5 | contractive | 0.02638 | 0.06222 | 0.08734 | 0.18139 |
| 64 | 0.05 | 0.995 | 1.0 | non_decaying | 0.03708 | 0.07237 | 0.10383 | 0.18350 |
| 64 | 0.05 | 0.995 | 1.0 | contractive | 0.02287 | 0.05777 | 0.08348 | 0.17930 |
| 64 | 0.05 | 0.995 | 2.0 | non_decaying | 0.03708 | 0.07237 | 0.10383 | 0.18350 |
| 64 | 0.05 | 0.995 | 2.0 | contractive | 0.02073 | 0.05251 | 0.08159 | 0.17521 |
| 64 | 0.1 | 0.98 | 0.5 | non_decaying | 0.08252 | 0.13026 | 0.25700 | 0.40567 |
| 64 | 0.1 | 0.98 | 0.5 | contractive | 0.05612 | 0.09884 | 0.20978 | 0.35551 |
| 64 | 0.1 | 0.98 | 1.0 | non_decaying | 0.08252 | 0.13026 | 0.25700 | 0.40567 |
| 64 | 0.1 | 0.98 | 1.0 | contractive | 0.04843 | 0.08488 | 0.20046 | 0.33432 |
| 64 | 0.1 | 0.98 | 2.0 | non_decaying | 0.08252 | 0.13026 | 0.25700 | 0.40567 |
| 64 | 0.1 | 0.98 | 2.0 | contractive | 0.04243 | 0.07590 | 0.19453 | 0.33432 |
| 64 | 0.1 | 0.99 | 0.5 | non_decaying | 0.08148 | 0.14295 | 0.22928 | 0.38289 |
| 64 | 0.1 | 0.99 | 0.5 | contractive | 0.05526 | 0.11520 | 0.18610 | 0.37849 |
| 64 | 0.1 | 0.99 | 1.0 | non_decaying | 0.08148 | 0.14295 | 0.22928 | 0.38289 |
| 64 | 0.1 | 0.99 | 1.0 | contractive | 0.04752 | 0.10927 | 0.18113 | 0.37414 |
| 64 | 0.1 | 0.99 | 2.0 | non_decaying | 0.08148 | 0.14295 | 0.22928 | 0.38289 |
| 64 | 0.1 | 0.99 | 2.0 | contractive | 0.04098 | 0.10182 | 0.17495 | 0.36559 |
| 64 | 0.1 | 0.995 | 0.5 | non_decaying | 0.07416 | 0.14474 | 0.20767 | 0.36700 |
| 64 | 0.1 | 0.995 | 0.5 | contractive | 0.05276 | 0.12444 | 0.17467 | 0.36278 |
| 64 | 0.1 | 0.995 | 1.0 | non_decaying | 0.07416 | 0.14474 | 0.20767 | 0.36700 |
| 64 | 0.1 | 0.995 | 1.0 | contractive | 0.04573 | 0.11553 | 0.16695 | 0.35861 |
| 64 | 0.1 | 0.995 | 2.0 | non_decaying | 0.07416 | 0.14474 | 0.20767 | 0.36700 |
| 64 | 0.1 | 0.995 | 2.0 | contractive | 0.04145 | 0.10501 | 0.16317 | 0.35041 |
