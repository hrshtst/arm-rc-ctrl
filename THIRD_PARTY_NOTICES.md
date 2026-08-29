# Third-Party Notices

The root GPL-3.0-only license applies only to original work in this repository.
Third-party components retain their own copyrights and license conditions.

## Planned Direct Source Dependencies

| Component | License | Source |
| --- | --- | --- |
| rclib | Apache License 2.0 | <https://github.com/hrshtst/rclib> |
| skelarm | GNU GPL 3.0 only | <https://github.com/hrshtst/skelarm> |
| rtctrl | Apache License 2.0 | <https://github.com/hrshtst/rtctrl> |

These projects are fetched as pinned Git submodules. Their license files and
notices must remain with redistributed source or binaries. Apache-2.0 material
may be incorporated into this GPLv3 project, but it remains Apache-2.0 material.

## Transitive and Asset-Specific Terms

Each dependency has its own transitive dependencies. Audit the exact recursive
submodule and package lock revisions before redistribution. In particular,
`rtctrl` uses CRANE-X7 description/model material whose upstream license contains
noncommercial and other asset-specific conditions:
<https://github.com/rt-net/crane_x7_description/blob/master/LICENSE>.

Do not assume this project's GPL license grants permission to redistribute robot
descriptions, meshes, datasets, demonstrations, trained models, plots, or media.
Exclude them from releases unless their individual terms and notices have been
recorded and satisfied.

Update this inventory whenever dependency pins or build inputs change. A release
license audit must also cover Python and C++ packages resolved by `uv`, CMake,
and nested submodules.
