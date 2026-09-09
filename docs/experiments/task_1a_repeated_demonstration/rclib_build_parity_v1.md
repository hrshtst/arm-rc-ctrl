# rclib build parity (v1)

Old build: rclib 0.1.0 at `a015aca1ec9e`; new build: rclib 0.1.0 at `61a29f0ce6fa`; project commit `28d55f9daa8b`.

Every case refits one recipe under each build and compares the teacher-forced predictions of its first episode bitwise and its fit report exactly. Frozen recipes are refitted through the explicit identity override (they bind the commit they were made with); `exact` says whether each build's refit reproduced the recorded fit report bitwise, which a difference in the recording environment, not in the build, can already prevent.

- All predictions bitwise equal and all fit reports equal: **True**.

| case | predictions bitwise equal | max abs diff | fit report equal | rmse old | rmse new | exact old | exact new |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| fixture-absolute-R17 | True | 0.000e+00 | True | 0.00027982244863652034 | 0.00027982244863652034 | n/a | n/a |
| fixture-residual-R17 | True | 0.000e+00 | True | 7.947664161169315e-05 | 7.947664161169315e-05 | n/a | n/a |
| model-20260831-038a9b2c8432 | True | 0.000e+00 | True | 0.00021435326142743184 | 0.00021435326142743184 | True | True |
| model-20260831-1b9477aaa246 | True | 0.000e+00 | True | 0.0017356401024966018 | 0.0017356401024966018 | False | False |
| model-20260831-ea83321eeaa5 | True | 0.000e+00 | True | 0.0007853510857272852 | 0.0007853510857272852 | True | True |
