# Data and artifact conventions

Git never stores experimental payloads. Raw demonstrations, processed datasets,
run logs, trained models, MLflow state, and Optuna databases live below a
machine-local storage root that is resolved at run time (see
[`docs/PLAN.md`](../docs/PLAN.md), section 7.1):

1. `ARM_RC_CTRL_STORAGE_ROOT` environment variable;
2. `[storage].root` in `${XDG_CONFIG_HOME:-$HOME/.config}/arm-rc-ctrl/storage.toml`
   (template: [`configs/storage.example.toml`](../configs/storage.example.toml));
3. `/external/arm-rc-ctrl`.

Tools fail before producing data when the root is absent or not writable. They
never fall back to this repository.

## What Git tracks

| Path | Content | Tracked |
| --- | --- | --- |
| `data/catalog.toml` | Index of all artifact records (introduced by M1-001) | yes |
| `data/records/raw/` | One small TOML record per raw demonstration | yes |
| `data/records/processed/` | One record per canonical processed dataset | yes |
| `data/records/runs/` | One record per curated simulation/evaluation run | yes |
| `data/records/models/` | One record per frozen model recipe | yes |
| `*.dvc`, `dvc.yaml`, `dvc.lock` | Portable DVC metafiles | yes |
| `tests/fixtures/` | Tiny synthetic or sanitized fixtures for automated tests | yes |
| Any payload (`*.npz`, `*.sklog.npz`, `*.parquet`, `*.db`, ...) | External storage only | no |
| `.dvc/config.local`, `.dvc/cache/` | Machine-specific DVC cache/remote configuration | no |
| `storage.toml` | Machine-local storage configuration | no |

## Logical URIs

Records reference payloads by logical URI, never by absolute machine path:

```text
armrc://<bucket>/<relative/path>
```

`<bucket>` is one of `raw`, `processed`, `runs`, `models`, `mlflow`, `optuna`,
`dvc-cache`, or `dvc-store`, matching the external root layout:

```text
<storage-root>/
├── raw/
├── processed/
├── runs/
├── models/
├── mlflow/
├── optuna/
├── dvc-cache/
└── dvc-store/
```

## Immutability

Payload creation is transactional: write to an external temporary path,
validate, compute the SHA-256 digest, move atomically to the final URI, then
write the repository record. Raw recordings are never overwritten; corrections
create a new artifact ID that names the superseded one. Readers verify size and
digest and fail on missing or mismatched data. Removing a Git record never
deletes an external payload; garbage collection is a separate, explicit, audited
operation.

Each record declares its own license and access classification. A record
without that metadata describes a private, non-redistributable artifact.
