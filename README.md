# Preregistration (Study 4): stability of post-error slowing in Stroop and flanker, Hedge et al. (2019, Experiment 1)

- `PREREGISTRATION_study4_hedge2019.md`: the registration (version 1.0, written 2026-11-26 UTC).
- `code/study4_hedge2019.py`: registered analysis code; it calls `code/stability.py` (scoring, splits, coefficients, bootstrap; identical to the copies deposited with the earlier registrations, https://doi.org/10.5281/zenodo.24534671 and https://doi.org/10.5281/zenodo.24545639) and the modules it imports.
- `code/hedge2019_counts.py`, `hedge2019_counts.json`: counts that use no response time.
- `code/power_sim.py`, `power_sim.json`: precision simulation under the estimates of the earlier reanalysis (no data of this study used).
- `synthetic_test.json`: output of the registered code with every recorded RT replaced by a random draw (correctness codes unchanged; see Section 2 of the registration).
- `SHA256SUMS.txt`: checksums of all files.

Data: Hedge et al. (2019), https://osf.io/zag7c/ (`Data/Experiment1/Exp1S{1,2}{Stroop,Flanker}.mat`; checksums in the registration).

Run: `python3 code/study4_hedge2019.py <folder with the four .mat files> OUT.json` (Python 3, numpy, scipy; `requirements.txt`).
