# Third-Party Notices

The root GPL-3.0-only license applies only to original work in this repository.
Third-party components retain their own copyrights and license conditions.

## Direct Source Dependencies (Git submodules)

| Component | License | Source | Path | Pinned commit |
| --- | --- | --- | --- | --- |
| rclib | Apache License 2.0 | <https://github.com/hrshtst/rclib> | `third_party/rclib` | `a015aca1ec9eaabb9ad4e384bf33e2e76018bf8b` |
| skelarm | GNU GPL 3.0 only | <https://github.com/hrshtst/skelarm> | `third_party/skelarm` | `11f563d37c23fcbbebddfbe6b6b23897204e38fb` |
| rtctrl | Apache License 2.0 | <https://github.com/hrshtst/rtctrl> | `third_party/rtctrl` | `c601076ee60ec712c1cb4a85756a186882df2e1b` |

These projects are pinned Git submodules with HTTPS URLs. The pinned commits
above are checked against the repository's recorded gitlinks by
`tests/unit/test_submodule_pins.py`; update this table in the same commit that
advances a pin. Their license files and notices must remain with redistributed
source or binaries. Apache-2.0 material may be incorporated into this GPLv3
project, but it remains Apache-2.0 material.

`rclib` is initialized recursively because its Python build needs the nested
Eigen, pybind11, and Catch2 submodules. `skelarm` has no nested submodules.
`rtctrl` is checked out top-level only until milestone M5; its nested
submodules (mi-lib, DynamixelSDK, CRANE-X7 description/ROS packages,
rt_manipulators_cpp) are then subject to their own audit.

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
