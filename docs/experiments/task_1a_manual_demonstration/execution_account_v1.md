# Task 1-a manual-demonstration execution account (v1)

How the M3MAN-010 full evaluation was executed: one interrupted sweep, the
recovery of its store, and the resume that completed it. Times are JST
(UTC+9) as the system journal records them. Machine paths are omitted; "the
store" is the configured canonical storage root.

## Outcome

The study is complete and its totals equal the frozen expectation of
[`run_ordering_v1`](run_ordering_v1.md) exactly:

| quantity | executed | frozen expectation |
| --- | --- | --- |
| models | 186 | 186 |
| RC runs (model pairs) | 24,180 | 24,180 |
| replay banks | 60 | 60 |
| replay runs | 7,800 | 7,800 |
| runs in total | 31,980 | 31,980 |

Pair verdicts: 14,184 of 24,180 RC runs and 7,750 of 7,800 replay runs met
every criterion; 42 of 186 models and 40 of 60 replay banks met every
criterion on all 130 of their pairs. These are pooled counts, not the paired
comparisons the plan reports.

The 246 Git pointers the resume wrote are in [`evidence/`](evidence/)
(186 `model__*`, 60 `replay__*`); each cites its stored manifest by URI,
SHA-256 and size, and all 246 verified against the store after the run. Every
run executed in the canonical execution identity `a7f034c7aef4` (P-cores
0-15), non-exploratory, from the clean commit `0664a66`, under study manifest
`b07b824c362a`.

## First sweep: interrupted by a storage failure

- Started 2026-09-18 22:26 from clean `0664a66`: `manual_evaluation run`
  pinned with `arm_rc_ctrl.execution run --policy p-cores`, 12 workers.
- 2026-09-19 02:33:33: a write to the store's NVMe drive timed out; the kernel
  reset the controller at 02:34:03, the reset failed three times ("Device not
  ready; aborting reset, CSTS=0x1"), and at 02:36:06 the kernel disabled the
  device. In-flight writes failed and the ext4 filesystem shut down
  read-only.
- The sweep exited 1 at 02:36:20: the worker for `feasible-worst/C10/D09` hit
  an I/O error reading a cached recipe, and the parent refused to continue.
- The same drive had failed identically on 2025-12-05. After a reboot it
  mounted normally and its NVMe health log was clean (no media errors, no
  error-log entries, 100 % spare, 0 % wear); the fault is a controller hang
  the health log does not record. The study's store moved to a second disk.

## Store recovery

- Every payload committed pointers cite verified on the recovered drive:
  581 catalog payloads under `data/records/` and 169 pointers under `docs/`,
  including the ten locked demonstrations and their raw sources.
- The damage was confined to files written in the last 35 s before the hang
  (02:32:24-02:32:59): 144 run directories whose `run.json` and `arrays.npz`
  were empty, one empty model manifest and one empty progress file.
- The owner copied the whole store to a second disk; all 76,300 files matched
  the original by path, size and SHA-256. The original drive was left
  untouched as a second copy.
- A read-only dry run with the runner's own verification found 152 model
  manifests, 60 replay banks and 175 cached fits intact. Thirteen models that
  were mid-evaluation would stop a resume, because their progress records
  cite the emptied runs (a progress record is verified whole and is never
  edited):

| model | evidence identity | why a resume would stop |
| --- | --- | --- |
| `failure-joint-velocity/S/D01` | `303cde582f05` | progress.json: run run-20260918-61da41965d4a of posture-large-20261205-01 [pd_v2] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D02` | `65ab9de22047` | progress.json: run run-20260918-5b92fa1e1ab1 of force-12N-180deg [pd_v2] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D03` | `19fd3cb17f4f` | progress.json: run run-20260918-1b5527efd515 of posture-large-20261204-00 [pd_v2] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D04` | `6e3dd5c5cfb6` | progress.json: run run-20260918-f72f68d3a729 of posture-large-20261202-01 [computed_torque] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D05` | `c443b38a765b` | progress.json: run run-20260918-a9950e4b72ba of posture-large-20261205-01 [computed_torque] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D06` | `7197c99d62ea` | progress.json: run run-20260918-4ee9966da67f of combined-20261202-01-090deg [computed_torque] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D07` | `66b2fa0478b2` | progress.json: run run-20260918-2c1d91912ccd of posture-large-20261202-01 [computed_torque] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D08` | `754013278e64` | progress.json: run run-20260918-f0a8bfbc145d of posture-large-20261201-03 [pd_v2] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D09` | `651fc6317259` | progress.json: run run-20260918-bd9e6eea7356 of posture-small-20261203-01 [pd_v2] rc: run.json no longer matches its record |
| `failure-joint-velocity/S/D10` | `52fcf7de914f` | progress.json: empty file |
| `feasible-worst/C10/D06` | `74c5d6485732` | manifest-6f860d6a5cb1.json: empty file; progress.json: run run-20260918-417e63073e22 of combined-20261205-02-180deg [pd_v2] rc: run.json no longer matches its record |
| `feasible-worst/C10/D09` | `53eecca8d9ea` | progress.json: run run-20260918-88ae16581fc7 of combined-20261203-01-090deg [pd_v2] rc: run.json no longer matches its record |
| `feasible-worst/C10/D10` | `bcb39beb70ac` | progress.json: run run-20260918-c00560a7d5cf of combined-20261201-02-180deg [computed_torque] rc: run.json no longer matches its record |

- With the owner's approval, those 13 model directories, the 144 emptied run
  directories and one empty `runs/staging-*` directory left by the interrupted
  write were moved, by rename and without deletion, into
  `arm-rc-ctrl-quarantine-20260919/` beside the store root, keeping their
  relative paths; its `MOVED.txt` records each item and the verification
  failure that moved it. Store and quarantine together still hold all 76,300
  files. The 13 models were re-evaluated from scratch; the intact runs their
  quarantined progress records had cited remain in the store, unreferenced.

Quarantined runs (every file empty):

```text
    run-20260918-02f86b3e3e8d run-20260918-03c7e2e1e896 run-20260918-04b4c76c674b run-20260918-05edc218e39d
    run-20260918-06d7617aa119 run-20260918-0819d7105927 run-20260918-0bab20141a34 run-20260918-1008b3593e16
    run-20260918-13c0b7e64545 run-20260918-1a14d5628fc6 run-20260918-1aad8431a5cf run-20260918-1b5527efd515
    run-20260918-1c66429e9226 run-20260918-1d868c097270 run-20260918-1db464627ba0 run-20260918-1ead33a3bbe4
    run-20260918-1fd408d67406 run-20260918-22baccc1b1a2 run-20260918-275804388327 run-20260918-2a4e684969cb
    run-20260918-2ad10d6900ea run-20260918-2ae33a382e77 run-20260918-2c1d91912ccd run-20260918-31cfbe0991b3
    run-20260918-3713f68888d4 run-20260918-376f55135204 run-20260918-385b42700b90 run-20260918-3874d8b88d09
    run-20260918-3a4f2c11f555 run-20260918-401091540942 run-20260918-417e63073e22 run-20260918-46638dc936ff
    run-20260918-492ba0660a5b run-20260918-49c408dfb915 run-20260918-4ee9966da67f run-20260918-515e14a30451
    run-20260918-5263641ebf8e run-20260918-546a7fce007c run-20260918-551e9a222db5 run-20260918-58e37263ee6b
    run-20260918-592da6a630db run-20260918-5b0d12cb569d run-20260918-5b92fa1e1ab1 run-20260918-5c29d354c597
    run-20260918-5c2dd44f348e run-20260918-61cde7a2700c run-20260918-61da41965d4a run-20260918-63bd5280ae8b
    run-20260918-685ada6760af run-20260918-69386fbfbce1 run-20260918-6f20581f8966 run-20260918-6f27c098dfc0
    run-20260918-6f9fc9fd21c2 run-20260918-714bbb65c072 run-20260918-71dd60f8841f run-20260918-71f5d4647060
    run-20260918-7307948f3713 run-20260918-746ee586afe8 run-20260918-7dd2e2f8f7d7 run-20260918-7eb66a7fb1db
    run-20260918-7f6553eef992 run-20260918-7fe05a754303 run-20260918-81b0b223b16d run-20260918-83e640967b58
    run-20260918-84f6b7253dd7 run-20260918-88ae16581fc7 run-20260918-890c1275aaf6 run-20260918-8c0f95bea5b6
    run-20260918-8c18ca08b9d4 run-20260918-8c420da4b69d run-20260918-8cdc75f7b54e run-20260918-8d672c2effb1
    run-20260918-8f1a0666dc6a run-20260918-90866b008671 run-20260918-90973c9f1389 run-20260918-90ca7001c6ed
    run-20260918-92dcdc15263a run-20260918-930389c3e764 run-20260918-933f0bbd6d62 run-20260918-93656d1ed693
    run-20260918-94be525f2be3 run-20260918-9608a1e95319 run-20260918-987342ea5eb1 run-20260918-9a8b2faf2414
    run-20260918-9bed9c5d6450 run-20260918-9f6abc628b15 run-20260918-a25c10f9a871 run-20260918-a73caa4bde6c
    run-20260918-a994688824c1 run-20260918-a9950e4b72ba run-20260918-b0b397e21a0f run-20260918-b2210cda9c83
    run-20260918-b72e95dd3966 run-20260918-b7313fc6a4be run-20260918-b7611e94f448 run-20260918-b964e9d1589c
    run-20260918-b9acfdde18e8 run-20260918-bb3a6505fe5a run-20260918-bd9e6eea7356 run-20260918-c00560a7d5cf
    run-20260918-c0a5daf06ba5 run-20260918-c16749199c75 run-20260918-c2d28dee08e1 run-20260918-c31527072a2a
    run-20260918-ca22335bca0a run-20260918-cb48cd43dbd1 run-20260918-ce0e89c6f9d0 run-20260918-ce4abc0c749b
    run-20260918-cec3b08798f8 run-20260918-d0e39c9e58d8 run-20260918-d384cd2e1667 run-20260918-d42a9f59ec1a
    run-20260918-d4f2404f892c run-20260918-d7ebc56eaf5c run-20260918-d8b07c688e2c run-20260918-da9148ee1d1d
    run-20260918-dab0549f0d0f run-20260918-db8f85cdebd3 run-20260918-de1a79cee5a5 run-20260918-de4f71ea7433
    run-20260918-e0d835fe06f4 run-20260918-e11b0f74fc58 run-20260918-e1458bf76520 run-20260918-e53ebde0fba8
    run-20260918-e61764e19cf7 run-20260918-e8685f79429f run-20260918-e92db422eef4 run-20260918-eb96e672f600
    run-20260918-ed0877fed349 run-20260918-ef49200f5cc1 run-20260918-f0a8bfbc145d run-20260918-f5939d331ee1
    run-20260918-f72d8745aabb run-20260918-f72f68d3a729 run-20260918-f75c10052754 run-20260918-f82fe7cd2c77
    run-20260918-f9325447dc84 run-20260918-f9e23150ddb5 run-20260918-fa09666ca771 run-20260918-fa35e1f16b38
    run-20260918-fadd7dbc6ecb run-20260918-fcbc20126ba6 run-20260918-fdcaabb85b4e run-20260918-ff017a7ed7e0
```

## Resume

- Started 2026-09-19 12:15:43 from clean `0664a66` with the identical pinned
  12-worker command, the storage root reconfigured to the second disk.
- The runner re-verified every reused manifest, run, replay bank and fit
  against its recorded digests before serving it, re-evaluated the 13
  quarantined models and evaluated the 21 not yet started.
- Finished 12:58:12 with exit 0 and wrote the 246 pointers:

```json
{
  "models": 186,
  "pairs": 24180,
  "completed": 14184,
  "infeasible": 9996,
  "statuses": {"feasible": 42, "infeasible": 144},
  "pointers_written": 246,
  "workers": 12,
  "study_manifest": "b07b824c362a3513c03c9c34fa0437b390f47d148ac5a288231fba218aaa36b7",
  "execution_identity": "a7f034c7aef439a14a2c0fc2b9bae9febbe02468c2d19d5c7ac40345c40648e2",
  "exploratory": false
}
```

The owner's independent review on 2026-09-19 re-verified all 246 manifests
and all 31,980 run summaries and trajectory payloads against their recorded
hashes, the exact coverage of the frozen model, scenario, tracker and
replay-bank combinations, the totals above, canonical non-exploratory
provenance, the ten demonstrations and their raw sources, and all 186 recipes
and weight arrays against their fit bindings. Numerical metrics were not
recomputed from trajectories and no run was re-simulated; both belong to
M3MAN-011.
