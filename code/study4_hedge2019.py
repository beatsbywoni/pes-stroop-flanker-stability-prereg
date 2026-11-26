#!/usr/bin/env python3
"""Study 4 (preregistered): stability across sessions of conventional post-error slowing in Stroop and flanker,
Hedge, Vivian-Griffiths, Powell, Bompas, and Sumner (2019), Experiment 1 (two sessions about four weeks apart).
The scores, splits, coefficients, and bootstrap are those of Study 1 (functions of stability.py, unchanged).

Usage: python3 study4_hedge2019.py <folder with Exp1S{1,2}{Stroop,Flanker}.mat> OUT.json [--synthetic]
With --synthetic every recorded RT is replaced by a random draw (used to test the code before registration)."""
import importlib.util, json, math, os, sys
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("stab", os.path.join(HERE, "stability.py")); ST = importlib.util.module_from_spec(spec); spec.loader.exec_module(ST)
TASKS = ("Stroop", "Flanker"); SEED = 20261207; NB = ST.NBOOT; ND_MAX = int(0.025 * NB)
EMPH = {1: "speed", 2: "speed", 3: "speed", 4: "accuracy", 5: "accuracy", 6: "accuracy", 7: "standard", 8: "standard", 9: "standard"}
r4, q, sb, cor = ST.r4, ST.q, ST.sb, ST.cor


def load(folder, task, s, synthetic=False):
    """participant -> rows (block, trial, cond, correct, rt_s, emphasis); file order is presentation order (12 blocks of 144).
    cond: "0" congruent, "1" neutral, "2" incongruent. correct: 1 correct, 0 incorrect response, 2 no response (RT missing)."""
    A = loadmat(os.path.join(folder, f"Exp1S{s}{task}.mat"))["AllData"]
    rng = np.random.default_rng(11 + s + (0 if task == "Stroop" else 10))
    out = {}
    for p in np.unique(A[:, 0]).astype(int):
        X = A[A[:, 0] == p]; rows = []
        assert len(X) == 1728
        for i, (pid, c, k, rt) in enumerate(X):
            c = int(c); k = int(k); miss = not np.isfinite(rt)
            if synthetic and not miss: rt = float(rng.lognormal(np.log(0.6), 0.25))
            corr = 2 if miss else k
            rows.append((i // 144 + 1, i % 144 + 1, str((c - 1) % 3), corr, -1.0 if miss else float(rt), EMPH[c]))
        assert all(len({r[5] for r in rows if r[0] == b}) == 1 for b in range(1, 13))
        out[int(p)] = rows
    return out


def accuracy(rows): return float(np.mean([r[3] == 1 for r in rows]))


def comps(rows, rng, emph=("standard",)):
    """components of one person-session: conventional PES, mean correct RT, error rate (trials of the given emphases)"""
    R = [r for r in rows if r[5] in emph]
    pe, pc = ST.pairs(R, "Stroop", "conv")            # same predicate for both tasks: every trial requires a response
    if len(emph) > 1:                                  # strata: emphasis x congruency of the scored trial
        pe = [(v, R[i + 1][5] + st_, b, i) for (v, st_, b, i) in pe]; pc = [(v, R[i + 1][5] + st_, b, i) for (v, st_, b, i) in pc]
    d = {"prim": {"ev": ST.Comp([t[0] for t in pe], [t[1] for t in pe], rng), "ctl": ST.Comp([t[0] for t in pc], [t[1] for t in pc], rng)}, "prim_raw": (pe, pc)}
    corr = [r for r in R if r[3] == 1 and ST.LO <= r[4] <= ST.HI]
    d["rt"] = {"x": ST.Comp([r[4] for r in corr], [r[2] for r in corr], rng)}
    d["er"] = {"x": ST.Comp([1.0 if r[3] != 1 else 0.0 for r in R], [r[2] for r in R], rng)}
    return d


def eligible(B, ps):
    """>= 10 post-error scored trials per session; >= 2 post-error and >= 2 control trials in each odd/even half"""
    ok = []
    for p in ps:
        good = True
        for s in (1, 2):
            pe, pc = B[s][p]["prim_raw"]
            if len(pe) < ST.MIN_EV: good = False; break
            for h in (0, 1):
                if sum(1 for k, _ in enumerate(pe) if k % 2 == h) < ST.MIN_HALF or sum(1 for k, _ in enumerate(pc) if k % 2 == h) < ST.MIN_HALF: good = False
        if good: ok.append(p)
    return ok


def with_nd(res, rho_b):
    und = int(np.isnan(rho_b).sum()); res = dict(res); res["rho_undefined"] = und
    if und > ND_MAX: res["rho_CI"] = [res["rho_CI"][0], "n.d."]
    return res


def pooled_mean(pools, labs, seed):
    """mean rho over the two tasks; joint bootstrap over the union of participants; undefined if any task undefined"""
    U = sorted(set().union(*[set(v["ps"]) for v in pools.values()]))
    pos = {t: {p: j for j, p in enumerate(v["ps"])} for t, v in pools.items()}
    def rho_from(t, lab, ii):
        H1, H2 = pools[t][lab]["H1"][:, ii], pools[t][lab]["H2"][:, ii]
        r1 = sb(ST.rcols(H1)).mean(); r2 = sb(ST.rcols(H2)).mean(); r = cor(pools[t][lab]["s1"][ii], pools[t][lab]["s2"][ii])
        return r / math.sqrt(r1 * r2) if r1 > 0 and r2 > 0 else np.nan
    pt = {lab: float(np.mean([rho_from(t, lab, np.arange(len(pools[t]["ps"]))) for t in pools])) for lab in labs}
    rng = np.random.default_rng(seed); bs = {lab: [] for lab in labs}
    for _ in range(NB):
        smp = rng.choice(U, len(U))
        for lab in labs:
            bs[lab].append(np.mean([rho_from(t, lab, np.array([pos[t][p] for p in smp if p in pos[t]])) for t in pools]))
    out = dict(n_unique=len(U))
    for lab in labs:
        b = np.array(bs[lab]); und = int(np.isnan(b).sum()); ci = q(b)
        out[lab] = dict(mean_rho=r4(pt[lab]), CI=[ci[0], "n.d." if und > ND_MAX else ci[1]], undefined=und)
    if "PES" in labs and "RT" in labs:
        d = np.array(bs["PES"]) - np.array(bs["RT"]); out["PES_minus_RT"] = dict(diff=r4(pt["PES"] - pt["RT"]), CI=q(d))
    return out


def run(folder, synthetic=False):
    res = dict(synthetic=synthetic, tasks={}); pools = {}; pools_all = {}
    for ti, task in enumerate(TASKS):
        S = {s: load(folder, task, s, synthetic) for s in (1, 2)}
        both = sorted(set(S[1]) & set(S[2]))
        acc_ok = [p for p in both if accuracy(S[1][p]) >= .60 and accuracy(S[2][p]) >= .60]
        rng = np.random.default_rng(SEED + 10 * ti)
        B = {s: {p: comps(S[s][p], rng) for p in acc_ok} for s in (1, 2)}
        ps = eligible(B, acc_ok); n = len(ps)
        idx = np.random.default_rng(SEED + 10 * ti + 1).integers(0, n, (NB, n))
        r = {}
        for key, kind, scale in (("prim", "diff", 1000.0), ("rt", "mean", 1000.0), ("er", "mean", 100.0)):
            c, extra = ST.coef(B, ps, key, kind, idx, scale)
            r[key] = with_nd(c, extra["rho_b"]); pools.setdefault(task, {"ps": ps})["PES" if key == "prim" else key.upper()] = extra
        # sensitivity: all blocks, splits stratified by emphasis and congruency
        rng2 = np.random.default_rng(SEED + 10 * ti + 5)
        BA = {s: {p: comps(S[s][p], rng2, emph=("speed", "accuracy", "standard")) for p in acc_ok} for s in (1, 2)}
        psa = eligible(BA, acc_ok); idxa = np.random.default_rng(SEED + 10 * ti + 6).integers(0, len(psa), (NB, len(psa)))
        ca, ea = ST.coef(BA, psa, "prim", "diff", idxa); r["prim_all_blocks"] = with_nd(ca, ea["rho_b"])
        pools_all[task] = {"ps": psa, "PES": ea}
        res["tasks"][task] = dict(n_both_sessions=len(both), n_accuracy_ok=len(acc_ok), n=n, n_all_blocks=len(psa),
                                  post_error_trials_median=[float(np.median([len(B[s][p]["prim_raw"][0]) for p in ps])) for s in (1, 2)], results=r, persons=ps)
    res["pooled"] = pooled_mean(pools, ("PES", "RT", "ER"), SEED + 99)
    res["pooled_all_blocks"] = pooled_mean(pools_all, ("PES",), SEED + 98)
    P = res["pooled"]["PES"]
    res["H11_supported"] = bool(P["CI"][1] != "n.d." and P["CI"][1] < 1)
    for h, task in (("H12_supported", "Stroop"), ("H13_supported", "Flanker")):
        x = res["tasks"][task]["results"]["prim"]; res[h] = bool(x["rho_CI"][1] != "n.d." and x["rho_CI"][1] < 1)
    return res


if __name__ == "__main__":
    folder, outp = sys.argv[1], sys.argv[2]; syn = "--synthetic" in sys.argv
    r = run(folder, syn)
    json.dump(r, open(outp, "w"), indent=1)
    print("STUDY4", "synthetic" if syn else "REAL", json.dumps({k: v for k, v in r.items() if k.startswith("H")}),
          {t: (r["tasks"][t]["n"], r["tasks"][t]["n_all_blocks"]) for t in TASKS})
