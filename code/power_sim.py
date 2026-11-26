#!/usr/bin/env python3
"""Precision of rho in Study 4 under the Study 1 estimates (no data of Study 4 used).
Each task: n persons, true scores in two sessions with correlation rho, half scores with error such that the
Spearman-Brown split-half coefficient equals rel; rho estimated as r12 / sqrt(rel1 rel2); person bootstrap."""
import json, numpy as np
rng = np.random.default_rng(20261208)
def sim_task(n, rho, rel1, rel2):
    T = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    out = []
    for s, rel in ((0, rel1), (1, rel2)):
        rh = rel / (2 - rel); sd = np.sqrt(1 / rh - 1)
        a = T[:, s] + rng.normal(0, sd, n); b = T[:, s] + rng.normal(0, sd, n); out.append((a, b))
    return out
def rho_est(H, ii):
    (a1, b1), (a2, b2) = H
    sb = lambda r: 2 * r / (1 + r)
    r1 = sb(np.corrcoef(a1[ii], b1[ii])[0, 1]); r2 = sb(np.corrcoef(a2[ii], b2[ii])[0, 1])
    r12 = np.corrcoef((a1[ii] + b1[ii]) / 2, (a2[ii] + b2[ii]) / 2)[0, 1]
    return r12 / np.sqrt(r1 * r2) if r1 > 0 and r2 > 0 else np.nan
def power(scen, nsim=400, nb=600):
    hits = {"mean": 0, "Stroop": 0, "Flanker": 0}
    for _ in range(nsim):
        n = 43
        Hs = {t: sim_task(n, *scen[t]) for t in scen}
        B = rng.integers(0, n, (nb, n))
        bs = {t: np.array([rho_est(Hs[t], b) for b in B]) for t in scen}
        m = (bs["Stroop"] + bs["Flanker"]) / 2
        for lab, v in (("mean", m), ("Stroop", bs["Stroop"]), ("Flanker", bs["Flanker"])):
            if np.isnan(v).mean() <= .025 and np.nanquantile(v, .975) < 1: hits[lab] += 1
    return {k: v / nsim for k, v in hits.items()}
scen = {
 "study1_point": {"Stroop": (.66, .79, .76), "Flanker": (.50, .62, .83)},
 "higher_rel": {"Stroop": (.66, .85, .85), "Flanker": (.50, .80, .85)},
 "rho_.75": {"Stroop": (.75, .80, .80), "Flanker": (.75, .75, .80)},
 "rho_.85": {"Stroop": (.85, .80, .80), "Flanker": (.85, .75, .80)},
}
res = {k: power(v) for k, v in scen.items()}
json.dump(dict(n_per_task=43, scenarios=scen, power=res), open("power_sim.json", "w"), indent=1); print(json.dumps(res, indent=1))
