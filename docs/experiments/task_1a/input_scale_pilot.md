# Input-scale pilot `input-scale-pilot-1a`

Dataset `processed-20260830-feaf73e6663c`, scenario `configs/tasks/task_1a.toml`, model config `configs/models/esn_task_1a.toml`, commit `8cd5920dbbb9`, trackers `computed_torque`, `pd_v2`.

Feasible fraction of variants per cell (rows: q scale in rad; columns: dq scale in rad/s); a cell is feasible at >= 0.75, marked with *.

| q \ dq | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| 0.1 | 1.00* (0.098) | 1.00* (0.1) | 1.00* (0.1) | 1.00* (0.1) | 1.00* (0.1) |
| 0.15 | 0.94* (0.13) | 1.00* (0.11) | 1.00* (0.11) | 1.00* (0.11) | 1.00* (0.1) |
| 0.2 | 0.81* (0.14) | 1.00* (0.11) | 1.00* (0.11) | 1.00* (0.11) | 1.00* (0.11) |
| 0.3 | 0.88* (0.12) | 1.00* (0.11) | 1.00* (0.1) | 1.00* (0.097) | 1.00* (0.095) |
| 0.5 | 0.56 (0.14) | 0.81* (0.12) | 0.88* (0.11) | 0.94* (0.1) | 1.00* (0.1) |

Cell entries: feasible fraction (median movement RMSE in rad over feasible variants).

## Selection

- anchor: q scale 0.3 rad, dq scale 12 rad/s (highest feasible fraction among cells whose grid neighbours are all feasible; ties broken by lower median RMSE)
- feasible region: 24 of 25 cells
