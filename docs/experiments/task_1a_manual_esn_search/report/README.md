# M3MS expert report

Open `index.html` in a browser. Keep the whole directory together: animations
use local JavaScript data files and require no server or external service.

Reproduce from the repository root, with the canonical evidence store configured:

```sh
uv run --locked python scripts/render_manual_search_report.py --output /tmp/m3ms-expert-report
uv run --locked python scripts/render_manual_search_report.py --output /tmp/m3ms-expert-report --verify
```

The output directory must not already exist. The renderer only reads the store;
it does not fit or simulate. Verification requires the checkout but not the store.

`assets/<case>-space.png`, `assets/<case>-time.png`,
`assets/<case>-joints.png` and `assets/<case>.js` correspond to the same
post-hoc case. The HTML supplies synchronized play, pause and seek controls.
`cases.json` binds all fifty illustrated runs to their original payloads.
`manifest.json` binds the complete presentation, sources and audited inputs.

The three summary plots use committed tables. Full-resolution plots use all
stored samples. Browser playback retains every tenth sample plus endpoints
and event samples and rounds displayed coordinates to six decimals; it never
extrapolates past an abort. Metrics are taken from the audited run table, not
recomputed from the display samples.

The source lives in `report_source/`,
`src/arm_rc_ctrl/experiments/manual_search_report.py` and
`manual_search_report_plots.py`. The plot/player code reuses the design of the
M3MAN expert report and the existing `manual_figures` renderer.

All ten illustrations were selected post-hoc for explanation. They must not
be described as a frozen representative sample or used to estimate prevalence.
The report interprets the evidence; the owner decides M3MS-GATE.
