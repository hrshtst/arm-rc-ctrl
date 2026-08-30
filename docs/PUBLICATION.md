# Publication, release, and citation policy

**Owner:** Hiroshi Atsuta (<atsuta@ieee.org>)

**Repository:** <https://github.com/hrshtst/arm-rc-ctrl> (public)

## Development visibility

Development happens in the open on the public repository. Public visibility
does not make any revision a citable research release: until a milestone is
reproducible under the requirements of [`PLAN.md`](PLAN.md) section 16, the
code, configurations, and reports are working material.

## Archival releases

A stable research release is created only when a milestone gate in
[`TASKS.md`](TASKS.md) has been closed by review and its key result can be
reproduced from a clean checkout plus a configured external store. At that
point:

1. the version in `pyproject.toml`, `cpp/CMakeLists.txt`, and `CITATION.cff`
   is set to a release version and a `date-released` entry is added;
2. a signed Git tag is created;
3. the tagged source is archived on Zenodo and the resulting DOI is added to
   `CITATION.cff` (`identifiers`/`doi`) and to `README.md`.

Until then `CITATION.cff` carries a development version and no DOI.

## Preferred citation

Cite the archival release DOI once it exists. Before that, cite the
repository URL together with the exact commit hash that produced the result;
every run record stores that hash.

## Authorship

Authorship of the software is recorded in `CITATION.cff` and `pyproject.toml`.
Affiliation and ORCID are added when supplied by the author. Contributions
from other people are acknowledged in the same file when they occur.

## Data, models, and media

Software licensing (GPL-3.0-only) does not cover demonstrations, datasets,
trained models, plots, or media. Experimental data lives outside Git under the
external storage root and may remain private or embargoed. Each artifact
record declares its own license and access classification; records without
that metadata describe private artifacts. Third-party assets, in particular
CRANE-X7 descriptions and meshes, are excluded from releases unless their
terms are satisfied (see `THIRD_PARTY_NOTICES.md`).
