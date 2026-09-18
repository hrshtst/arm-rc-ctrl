# Task 1-a manual-demonstration re-simulation subset (v1)

Frozen before execution: the runs a clean-checkout audit re-simulates, each named before it exists. The identities are listed explicitly; the JSON beside this file is authoritative.
Record definitions are not restated here: see result schema v2, bound by digest below (`ResimulationSubset`).

| bound input | sha256 |
| --- | --- |
| study manifest | `b07b824c362a3513c03c9c34fa0437b390f47d148ac5a288231fba218aaa36b7` |
| evaluation configuration | `8780be5bac7f4ccb99d6636c947f78901646b886b5f768846b6d9ba5d48acb2b` |
| result schema v2 | `dfd3e29b2f7629253cdf511d1690e53d0b0cf82ba1013628e386ca6a12867b9e` |
| run ordering | `0305be9ead9dcb51d58abdb9951626883a329033fcfb131f91d4df8c6c7bdf4f` |

## Counts

| figure | value |
| --- | ---: |
| models | 24 |
| scenarios (one per class) | 5 |
| trackers | 2 |
| RC runs | 240 |
| replay runs | 60 |
| runs in total | 300 |

## Scenarios

| class | scenario |
| --- | --- |
| `nominal` | `nominal` |
| `posture_small` | `posture-small-20261201-00` |
| `posture_large` | `posture-large-20261201-00` |
| `force` | `force-12N-000deg` |
| `combined` | `combined-20261201-00-000deg` |

## RC runs (240, model x scenario x tracker)

| # | model | scenario | tracker |
| ---: | --- | --- | --- |
| 1 | `feasible-best/S/D01` | `nominal` | `pd_v2` |
| 2 | `feasible-best/S/D01` | `nominal` | `computed_torque` |
| 3 | `feasible-best/S/D01` | `posture-small-20261201-00` | `pd_v2` |
| 4 | `feasible-best/S/D01` | `posture-small-20261201-00` | `computed_torque` |
| 5 | `feasible-best/S/D01` | `posture-large-20261201-00` | `pd_v2` |
| 6 | `feasible-best/S/D01` | `posture-large-20261201-00` | `computed_torque` |
| 7 | `feasible-best/S/D01` | `force-12N-000deg` | `pd_v2` |
| 8 | `feasible-best/S/D01` | `force-12N-000deg` | `computed_torque` |
| 9 | `feasible-best/S/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 10 | `feasible-best/S/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 11 | `feasible-best/M10` | `nominal` | `pd_v2` |
| 12 | `feasible-best/M10` | `nominal` | `computed_torque` |
| 13 | `feasible-best/M10` | `posture-small-20261201-00` | `pd_v2` |
| 14 | `feasible-best/M10` | `posture-small-20261201-00` | `computed_torque` |
| 15 | `feasible-best/M10` | `posture-large-20261201-00` | `pd_v2` |
| 16 | `feasible-best/M10` | `posture-large-20261201-00` | `computed_torque` |
| 17 | `feasible-best/M10` | `force-12N-000deg` | `pd_v2` |
| 18 | `feasible-best/M10` | `force-12N-000deg` | `computed_torque` |
| 19 | `feasible-best/M10` | `combined-20261201-00-000deg` | `pd_v2` |
| 20 | `feasible-best/M10` | `combined-20261201-00-000deg` | `computed_torque` |
| 21 | `feasible-best/R10/D01` | `nominal` | `pd_v2` |
| 22 | `feasible-best/R10/D01` | `nominal` | `computed_torque` |
| 23 | `feasible-best/R10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 24 | `feasible-best/R10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 25 | `feasible-best/R10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 26 | `feasible-best/R10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 27 | `feasible-best/R10/D01` | `force-12N-000deg` | `pd_v2` |
| 28 | `feasible-best/R10/D01` | `force-12N-000deg` | `computed_torque` |
| 29 | `feasible-best/R10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 30 | `feasible-best/R10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 31 | `feasible-best/C10/D01` | `nominal` | `pd_v2` |
| 32 | `feasible-best/C10/D01` | `nominal` | `computed_torque` |
| 33 | `feasible-best/C10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 34 | `feasible-best/C10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 35 | `feasible-best/C10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 36 | `feasible-best/C10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 37 | `feasible-best/C10/D01` | `force-12N-000deg` | `pd_v2` |
| 38 | `feasible-best/C10/D01` | `force-12N-000deg` | `computed_torque` |
| 39 | `feasible-best/C10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 40 | `feasible-best/C10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 41 | `feasible-middle/S/D01` | `nominal` | `pd_v2` |
| 42 | `feasible-middle/S/D01` | `nominal` | `computed_torque` |
| 43 | `feasible-middle/S/D01` | `posture-small-20261201-00` | `pd_v2` |
| 44 | `feasible-middle/S/D01` | `posture-small-20261201-00` | `computed_torque` |
| 45 | `feasible-middle/S/D01` | `posture-large-20261201-00` | `pd_v2` |
| 46 | `feasible-middle/S/D01` | `posture-large-20261201-00` | `computed_torque` |
| 47 | `feasible-middle/S/D01` | `force-12N-000deg` | `pd_v2` |
| 48 | `feasible-middle/S/D01` | `force-12N-000deg` | `computed_torque` |
| 49 | `feasible-middle/S/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 50 | `feasible-middle/S/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 51 | `feasible-middle/M10` | `nominal` | `pd_v2` |
| 52 | `feasible-middle/M10` | `nominal` | `computed_torque` |
| 53 | `feasible-middle/M10` | `posture-small-20261201-00` | `pd_v2` |
| 54 | `feasible-middle/M10` | `posture-small-20261201-00` | `computed_torque` |
| 55 | `feasible-middle/M10` | `posture-large-20261201-00` | `pd_v2` |
| 56 | `feasible-middle/M10` | `posture-large-20261201-00` | `computed_torque` |
| 57 | `feasible-middle/M10` | `force-12N-000deg` | `pd_v2` |
| 58 | `feasible-middle/M10` | `force-12N-000deg` | `computed_torque` |
| 59 | `feasible-middle/M10` | `combined-20261201-00-000deg` | `pd_v2` |
| 60 | `feasible-middle/M10` | `combined-20261201-00-000deg` | `computed_torque` |
| 61 | `feasible-middle/R10/D01` | `nominal` | `pd_v2` |
| 62 | `feasible-middle/R10/D01` | `nominal` | `computed_torque` |
| 63 | `feasible-middle/R10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 64 | `feasible-middle/R10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 65 | `feasible-middle/R10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 66 | `feasible-middle/R10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 67 | `feasible-middle/R10/D01` | `force-12N-000deg` | `pd_v2` |
| 68 | `feasible-middle/R10/D01` | `force-12N-000deg` | `computed_torque` |
| 69 | `feasible-middle/R10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 70 | `feasible-middle/R10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 71 | `feasible-middle/C10/D01` | `nominal` | `pd_v2` |
| 72 | `feasible-middle/C10/D01` | `nominal` | `computed_torque` |
| 73 | `feasible-middle/C10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 74 | `feasible-middle/C10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 75 | `feasible-middle/C10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 76 | `feasible-middle/C10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 77 | `feasible-middle/C10/D01` | `force-12N-000deg` | `pd_v2` |
| 78 | `feasible-middle/C10/D01` | `force-12N-000deg` | `computed_torque` |
| 79 | `feasible-middle/C10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 80 | `feasible-middle/C10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 81 | `feasible-worst/S/D01` | `nominal` | `pd_v2` |
| 82 | `feasible-worst/S/D01` | `nominal` | `computed_torque` |
| 83 | `feasible-worst/S/D01` | `posture-small-20261201-00` | `pd_v2` |
| 84 | `feasible-worst/S/D01` | `posture-small-20261201-00` | `computed_torque` |
| 85 | `feasible-worst/S/D01` | `posture-large-20261201-00` | `pd_v2` |
| 86 | `feasible-worst/S/D01` | `posture-large-20261201-00` | `computed_torque` |
| 87 | `feasible-worst/S/D01` | `force-12N-000deg` | `pd_v2` |
| 88 | `feasible-worst/S/D01` | `force-12N-000deg` | `computed_torque` |
| 89 | `feasible-worst/S/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 90 | `feasible-worst/S/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 91 | `feasible-worst/M10` | `nominal` | `pd_v2` |
| 92 | `feasible-worst/M10` | `nominal` | `computed_torque` |
| 93 | `feasible-worst/M10` | `posture-small-20261201-00` | `pd_v2` |
| 94 | `feasible-worst/M10` | `posture-small-20261201-00` | `computed_torque` |
| 95 | `feasible-worst/M10` | `posture-large-20261201-00` | `pd_v2` |
| 96 | `feasible-worst/M10` | `posture-large-20261201-00` | `computed_torque` |
| 97 | `feasible-worst/M10` | `force-12N-000deg` | `pd_v2` |
| 98 | `feasible-worst/M10` | `force-12N-000deg` | `computed_torque` |
| 99 | `feasible-worst/M10` | `combined-20261201-00-000deg` | `pd_v2` |
| 100 | `feasible-worst/M10` | `combined-20261201-00-000deg` | `computed_torque` |
| 101 | `feasible-worst/R10/D01` | `nominal` | `pd_v2` |
| 102 | `feasible-worst/R10/D01` | `nominal` | `computed_torque` |
| 103 | `feasible-worst/R10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 104 | `feasible-worst/R10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 105 | `feasible-worst/R10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 106 | `feasible-worst/R10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 107 | `feasible-worst/R10/D01` | `force-12N-000deg` | `pd_v2` |
| 108 | `feasible-worst/R10/D01` | `force-12N-000deg` | `computed_torque` |
| 109 | `feasible-worst/R10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 110 | `feasible-worst/R10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 111 | `feasible-worst/C10/D01` | `nominal` | `pd_v2` |
| 112 | `feasible-worst/C10/D01` | `nominal` | `computed_torque` |
| 113 | `feasible-worst/C10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 114 | `feasible-worst/C10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 115 | `feasible-worst/C10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 116 | `feasible-worst/C10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 117 | `feasible-worst/C10/D01` | `force-12N-000deg` | `pd_v2` |
| 118 | `feasible-worst/C10/D01` | `force-12N-000deg` | `computed_torque` |
| 119 | `feasible-worst/C10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 120 | `feasible-worst/C10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 121 | `failure-actual-dwell/S/D01` | `nominal` | `pd_v2` |
| 122 | `failure-actual-dwell/S/D01` | `nominal` | `computed_torque` |
| 123 | `failure-actual-dwell/S/D01` | `posture-small-20261201-00` | `pd_v2` |
| 124 | `failure-actual-dwell/S/D01` | `posture-small-20261201-00` | `computed_torque` |
| 125 | `failure-actual-dwell/S/D01` | `posture-large-20261201-00` | `pd_v2` |
| 126 | `failure-actual-dwell/S/D01` | `posture-large-20261201-00` | `computed_torque` |
| 127 | `failure-actual-dwell/S/D01` | `force-12N-000deg` | `pd_v2` |
| 128 | `failure-actual-dwell/S/D01` | `force-12N-000deg` | `computed_torque` |
| 129 | `failure-actual-dwell/S/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 130 | `failure-actual-dwell/S/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 131 | `failure-actual-dwell/M10` | `nominal` | `pd_v2` |
| 132 | `failure-actual-dwell/M10` | `nominal` | `computed_torque` |
| 133 | `failure-actual-dwell/M10` | `posture-small-20261201-00` | `pd_v2` |
| 134 | `failure-actual-dwell/M10` | `posture-small-20261201-00` | `computed_torque` |
| 135 | `failure-actual-dwell/M10` | `posture-large-20261201-00` | `pd_v2` |
| 136 | `failure-actual-dwell/M10` | `posture-large-20261201-00` | `computed_torque` |
| 137 | `failure-actual-dwell/M10` | `force-12N-000deg` | `pd_v2` |
| 138 | `failure-actual-dwell/M10` | `force-12N-000deg` | `computed_torque` |
| 139 | `failure-actual-dwell/M10` | `combined-20261201-00-000deg` | `pd_v2` |
| 140 | `failure-actual-dwell/M10` | `combined-20261201-00-000deg` | `computed_torque` |
| 141 | `failure-actual-dwell/R10/D01` | `nominal` | `pd_v2` |
| 142 | `failure-actual-dwell/R10/D01` | `nominal` | `computed_torque` |
| 143 | `failure-actual-dwell/R10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 144 | `failure-actual-dwell/R10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 145 | `failure-actual-dwell/R10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 146 | `failure-actual-dwell/R10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 147 | `failure-actual-dwell/R10/D01` | `force-12N-000deg` | `pd_v2` |
| 148 | `failure-actual-dwell/R10/D01` | `force-12N-000deg` | `computed_torque` |
| 149 | `failure-actual-dwell/R10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 150 | `failure-actual-dwell/R10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 151 | `failure-actual-dwell/C10/D01` | `nominal` | `pd_v2` |
| 152 | `failure-actual-dwell/C10/D01` | `nominal` | `computed_torque` |
| 153 | `failure-actual-dwell/C10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 154 | `failure-actual-dwell/C10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 155 | `failure-actual-dwell/C10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 156 | `failure-actual-dwell/C10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 157 | `failure-actual-dwell/C10/D01` | `force-12N-000deg` | `pd_v2` |
| 158 | `failure-actual-dwell/C10/D01` | `force-12N-000deg` | `computed_torque` |
| 159 | `failure-actual-dwell/C10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 160 | `failure-actual-dwell/C10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 161 | `failure-joint-velocity/S/D01` | `nominal` | `pd_v2` |
| 162 | `failure-joint-velocity/S/D01` | `nominal` | `computed_torque` |
| 163 | `failure-joint-velocity/S/D01` | `posture-small-20261201-00` | `pd_v2` |
| 164 | `failure-joint-velocity/S/D01` | `posture-small-20261201-00` | `computed_torque` |
| 165 | `failure-joint-velocity/S/D01` | `posture-large-20261201-00` | `pd_v2` |
| 166 | `failure-joint-velocity/S/D01` | `posture-large-20261201-00` | `computed_torque` |
| 167 | `failure-joint-velocity/S/D01` | `force-12N-000deg` | `pd_v2` |
| 168 | `failure-joint-velocity/S/D01` | `force-12N-000deg` | `computed_torque` |
| 169 | `failure-joint-velocity/S/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 170 | `failure-joint-velocity/S/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 171 | `failure-joint-velocity/M10` | `nominal` | `pd_v2` |
| 172 | `failure-joint-velocity/M10` | `nominal` | `computed_torque` |
| 173 | `failure-joint-velocity/M10` | `posture-small-20261201-00` | `pd_v2` |
| 174 | `failure-joint-velocity/M10` | `posture-small-20261201-00` | `computed_torque` |
| 175 | `failure-joint-velocity/M10` | `posture-large-20261201-00` | `pd_v2` |
| 176 | `failure-joint-velocity/M10` | `posture-large-20261201-00` | `computed_torque` |
| 177 | `failure-joint-velocity/M10` | `force-12N-000deg` | `pd_v2` |
| 178 | `failure-joint-velocity/M10` | `force-12N-000deg` | `computed_torque` |
| 179 | `failure-joint-velocity/M10` | `combined-20261201-00-000deg` | `pd_v2` |
| 180 | `failure-joint-velocity/M10` | `combined-20261201-00-000deg` | `computed_torque` |
| 181 | `failure-joint-velocity/R10/D01` | `nominal` | `pd_v2` |
| 182 | `failure-joint-velocity/R10/D01` | `nominal` | `computed_torque` |
| 183 | `failure-joint-velocity/R10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 184 | `failure-joint-velocity/R10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 185 | `failure-joint-velocity/R10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 186 | `failure-joint-velocity/R10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 187 | `failure-joint-velocity/R10/D01` | `force-12N-000deg` | `pd_v2` |
| 188 | `failure-joint-velocity/R10/D01` | `force-12N-000deg` | `computed_torque` |
| 189 | `failure-joint-velocity/R10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 190 | `failure-joint-velocity/R10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 191 | `failure-joint-velocity/C10/D01` | `nominal` | `pd_v2` |
| 192 | `failure-joint-velocity/C10/D01` | `nominal` | `computed_torque` |
| 193 | `failure-joint-velocity/C10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 194 | `failure-joint-velocity/C10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 195 | `failure-joint-velocity/C10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 196 | `failure-joint-velocity/C10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 197 | `failure-joint-velocity/C10/D01` | `force-12N-000deg` | `pd_v2` |
| 198 | `failure-joint-velocity/C10/D01` | `force-12N-000deg` | `computed_torque` |
| 199 | `failure-joint-velocity/C10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 200 | `failure-joint-velocity/C10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 201 | `failure-generated-dwell/S/D01` | `nominal` | `pd_v2` |
| 202 | `failure-generated-dwell/S/D01` | `nominal` | `computed_torque` |
| 203 | `failure-generated-dwell/S/D01` | `posture-small-20261201-00` | `pd_v2` |
| 204 | `failure-generated-dwell/S/D01` | `posture-small-20261201-00` | `computed_torque` |
| 205 | `failure-generated-dwell/S/D01` | `posture-large-20261201-00` | `pd_v2` |
| 206 | `failure-generated-dwell/S/D01` | `posture-large-20261201-00` | `computed_torque` |
| 207 | `failure-generated-dwell/S/D01` | `force-12N-000deg` | `pd_v2` |
| 208 | `failure-generated-dwell/S/D01` | `force-12N-000deg` | `computed_torque` |
| 209 | `failure-generated-dwell/S/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 210 | `failure-generated-dwell/S/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 211 | `failure-generated-dwell/M10` | `nominal` | `pd_v2` |
| 212 | `failure-generated-dwell/M10` | `nominal` | `computed_torque` |
| 213 | `failure-generated-dwell/M10` | `posture-small-20261201-00` | `pd_v2` |
| 214 | `failure-generated-dwell/M10` | `posture-small-20261201-00` | `computed_torque` |
| 215 | `failure-generated-dwell/M10` | `posture-large-20261201-00` | `pd_v2` |
| 216 | `failure-generated-dwell/M10` | `posture-large-20261201-00` | `computed_torque` |
| 217 | `failure-generated-dwell/M10` | `force-12N-000deg` | `pd_v2` |
| 218 | `failure-generated-dwell/M10` | `force-12N-000deg` | `computed_torque` |
| 219 | `failure-generated-dwell/M10` | `combined-20261201-00-000deg` | `pd_v2` |
| 220 | `failure-generated-dwell/M10` | `combined-20261201-00-000deg` | `computed_torque` |
| 221 | `failure-generated-dwell/R10/D01` | `nominal` | `pd_v2` |
| 222 | `failure-generated-dwell/R10/D01` | `nominal` | `computed_torque` |
| 223 | `failure-generated-dwell/R10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 224 | `failure-generated-dwell/R10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 225 | `failure-generated-dwell/R10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 226 | `failure-generated-dwell/R10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 227 | `failure-generated-dwell/R10/D01` | `force-12N-000deg` | `pd_v2` |
| 228 | `failure-generated-dwell/R10/D01` | `force-12N-000deg` | `computed_torque` |
| 229 | `failure-generated-dwell/R10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 230 | `failure-generated-dwell/R10/D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 231 | `failure-generated-dwell/C10/D01` | `nominal` | `pd_v2` |
| 232 | `failure-generated-dwell/C10/D01` | `nominal` | `computed_torque` |
| 233 | `failure-generated-dwell/C10/D01` | `posture-small-20261201-00` | `pd_v2` |
| 234 | `failure-generated-dwell/C10/D01` | `posture-small-20261201-00` | `computed_torque` |
| 235 | `failure-generated-dwell/C10/D01` | `posture-large-20261201-00` | `pd_v2` |
| 236 | `failure-generated-dwell/C10/D01` | `posture-large-20261201-00` | `computed_torque` |
| 237 | `failure-generated-dwell/C10/D01` | `force-12N-000deg` | `pd_v2` |
| 238 | `failure-generated-dwell/C10/D01` | `force-12N-000deg` | `computed_torque` |
| 239 | `failure-generated-dwell/C10/D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 240 | `failure-generated-dwell/C10/D01` | `combined-20261201-00-000deg` | `computed_torque` |

## Replay runs (60, bank x scenario x tracker)

| # | configuration | assignment | scenario | tracker |
| ---: | --- | --- | --- | --- |
| 1 | `feasible-best` | `D01` | `nominal` | `pd_v2` |
| 2 | `feasible-best` | `D01` | `nominal` | `computed_torque` |
| 3 | `feasible-best` | `D01` | `posture-small-20261201-00` | `pd_v2` |
| 4 | `feasible-best` | `D01` | `posture-small-20261201-00` | `computed_torque` |
| 5 | `feasible-best` | `D01` | `posture-large-20261201-00` | `pd_v2` |
| 6 | `feasible-best` | `D01` | `posture-large-20261201-00` | `computed_torque` |
| 7 | `feasible-best` | `D01` | `force-12N-000deg` | `pd_v2` |
| 8 | `feasible-best` | `D01` | `force-12N-000deg` | `computed_torque` |
| 9 | `feasible-best` | `D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 10 | `feasible-best` | `D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 11 | `feasible-middle` | `D01` | `nominal` | `pd_v2` |
| 12 | `feasible-middle` | `D01` | `nominal` | `computed_torque` |
| 13 | `feasible-middle` | `D01` | `posture-small-20261201-00` | `pd_v2` |
| 14 | `feasible-middle` | `D01` | `posture-small-20261201-00` | `computed_torque` |
| 15 | `feasible-middle` | `D01` | `posture-large-20261201-00` | `pd_v2` |
| 16 | `feasible-middle` | `D01` | `posture-large-20261201-00` | `computed_torque` |
| 17 | `feasible-middle` | `D01` | `force-12N-000deg` | `pd_v2` |
| 18 | `feasible-middle` | `D01` | `force-12N-000deg` | `computed_torque` |
| 19 | `feasible-middle` | `D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 20 | `feasible-middle` | `D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 21 | `feasible-worst` | `D01` | `nominal` | `pd_v2` |
| 22 | `feasible-worst` | `D01` | `nominal` | `computed_torque` |
| 23 | `feasible-worst` | `D01` | `posture-small-20261201-00` | `pd_v2` |
| 24 | `feasible-worst` | `D01` | `posture-small-20261201-00` | `computed_torque` |
| 25 | `feasible-worst` | `D01` | `posture-large-20261201-00` | `pd_v2` |
| 26 | `feasible-worst` | `D01` | `posture-large-20261201-00` | `computed_torque` |
| 27 | `feasible-worst` | `D01` | `force-12N-000deg` | `pd_v2` |
| 28 | `feasible-worst` | `D01` | `force-12N-000deg` | `computed_torque` |
| 29 | `feasible-worst` | `D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 30 | `feasible-worst` | `D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 31 | `failure-actual-dwell` | `D01` | `nominal` | `pd_v2` |
| 32 | `failure-actual-dwell` | `D01` | `nominal` | `computed_torque` |
| 33 | `failure-actual-dwell` | `D01` | `posture-small-20261201-00` | `pd_v2` |
| 34 | `failure-actual-dwell` | `D01` | `posture-small-20261201-00` | `computed_torque` |
| 35 | `failure-actual-dwell` | `D01` | `posture-large-20261201-00` | `pd_v2` |
| 36 | `failure-actual-dwell` | `D01` | `posture-large-20261201-00` | `computed_torque` |
| 37 | `failure-actual-dwell` | `D01` | `force-12N-000deg` | `pd_v2` |
| 38 | `failure-actual-dwell` | `D01` | `force-12N-000deg` | `computed_torque` |
| 39 | `failure-actual-dwell` | `D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 40 | `failure-actual-dwell` | `D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 41 | `failure-joint-velocity` | `D01` | `nominal` | `pd_v2` |
| 42 | `failure-joint-velocity` | `D01` | `nominal` | `computed_torque` |
| 43 | `failure-joint-velocity` | `D01` | `posture-small-20261201-00` | `pd_v2` |
| 44 | `failure-joint-velocity` | `D01` | `posture-small-20261201-00` | `computed_torque` |
| 45 | `failure-joint-velocity` | `D01` | `posture-large-20261201-00` | `pd_v2` |
| 46 | `failure-joint-velocity` | `D01` | `posture-large-20261201-00` | `computed_torque` |
| 47 | `failure-joint-velocity` | `D01` | `force-12N-000deg` | `pd_v2` |
| 48 | `failure-joint-velocity` | `D01` | `force-12N-000deg` | `computed_torque` |
| 49 | `failure-joint-velocity` | `D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 50 | `failure-joint-velocity` | `D01` | `combined-20261201-00-000deg` | `computed_torque` |
| 51 | `failure-generated-dwell` | `D01` | `nominal` | `pd_v2` |
| 52 | `failure-generated-dwell` | `D01` | `nominal` | `computed_torque` |
| 53 | `failure-generated-dwell` | `D01` | `posture-small-20261201-00` | `pd_v2` |
| 54 | `failure-generated-dwell` | `D01` | `posture-small-20261201-00` | `computed_torque` |
| 55 | `failure-generated-dwell` | `D01` | `posture-large-20261201-00` | `pd_v2` |
| 56 | `failure-generated-dwell` | `D01` | `posture-large-20261201-00` | `computed_torque` |
| 57 | `failure-generated-dwell` | `D01` | `force-12N-000deg` | `pd_v2` |
| 58 | `failure-generated-dwell` | `D01` | `force-12N-000deg` | `computed_torque` |
| 59 | `failure-generated-dwell` | `D01` | `combined-20261201-00-000deg` | `pd_v2` |
| 60 | `failure-generated-dwell` | `D01` | `combined-20261201-00-000deg` | `computed_torque` |
