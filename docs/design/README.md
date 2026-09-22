<!-- Copyright (c) 2026 Hiroshi Atsuta -->
<!-- SPDX-License-Identifier: GPL-3.0-only -->

# Shared design references

Read [the roadmap](../PLAN.md) and [queue](../TASKS.md) first. Then open only
the relevant reference below. Original roadmap section numbers are retained;
legacy `PLAN.md section N` citations route through its short section map.

| Reference | Original sections |
| --- | --- |
| [Architecture](architecture.md) | 1–4: objectives, questions, ownership; 8: interfaces; 12: layout |
| [Controller](controller.md) | 5–6: ESN signals/training/history, derivatives and fair baselines |
| [Data](data.md) | 7: storage/data/run/visualization contracts; 11: experiment management |
| [Evaluation](evaluation.md) | 9–10: metrics, robustness and hyperparameter tuning |
| [Workflow](workflow.md) | 14–18: tests, reviews, reproduction, safety, assumptions; definition of done and document maintenance |

Historical phase gates are in the [roadmap archive](../tasks/archive/roadmap-2026-09-22.md).
Current experiment choices belong in `docs/experiments/<experiment>/plan.md`;
read those when they specialize a shared contract. Migration preserved the
numbered specification text and historical notes; it did not refreeze any
scientific artifact or modify a runtime configuration.
