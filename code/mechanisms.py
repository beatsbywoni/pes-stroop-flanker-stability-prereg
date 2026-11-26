#!/usr/bin/env python3
"""What the occasion-specific part of post-error slowing (PES) reflects: variance components, a decomposition of PES
into the post-error adjustment and the pre-error deviation, error expectancy, and overall speed.
Plan: plans/ANALYSIS_PLAN_mechanisms.md (written before any of these quantities was computed; exploratory).

Usage: HEDGE_RAW=<.../RawData> python3 mechanisms.py OUT.json
"""
import importlib.util, json, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cp", os.path.join(HERE, "conv_pes.py"))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
L, ad = C.L, C.ad
NBOOT, NRND, SEED = 2000, 1000, 20261125
cor, sb, q = C.cor, C.sb, C.q
r4 = lambda x: round(float(x), 4)


def select_persons(task):
    """the person sets of conv_pes.run(): common (primary) and common_rob (robust PES); same code and order"""
    C1, C2 = C.load(task, 1), C.load(task, 2)
    D1, D2 = C.derive_session(task, 1), C.derive_session(task, 2)
    X1 = {p: C.episodes(d) for p, d in D1.items()}; X2 = {p: C.episodes(d) for p, d in D2.items()}

    def halves5(d, reg, E, assign):
        v = [[], []]; first = {}
        for k, (i, b, w) in enumerate(E):
            h = 0 if b in assign[0] else 1
            for _, j in w:
                x = L.dev(d["rows"], j, reg, "medium")
                if x is None: continue
                if j not in first: first[j] = (h, x)
        for j, (h, x) in first.items(): v[h].append(x)
        return v

    def blk_ok(p):
        return all(all(halves5(D1[p], *X1[p], a)[h] for h in (0, 1)) for a in L.A32)
    keep1 = [p for p in sorted(X1) if len(X1[p][1]) >= 20 and blk_ok(p)]
    ps5 = [p for p in keep1 if p in X2 and len(X2[p][1]) >= 20]

    def conv_ok(p, which="PES"):
        if p not in C1 or p not in C2: return False
        pe1, pc1, rb1 = C1[p]; pe2, pc2, rb2 = C2[p]
        if which == "PES":
            if len(pe1) < C.MIN_PE_SESSION or len(pe2) < C.MIN_PE_SESSION: return False
            hs = [C.oe_halves(pe1, pc1)] + [C.blk_halves(pe1, pc1, a) for a in L.A32]
            return all(len(h[x][0]) >= C.MIN_HALF and len(h[x][1]) >= C.MIN_HALF for h in hs for x in (0, 1))
        if len(rb1) < C.MIN_PE_SESSION or len(rb2) < C.MIN_PE_SESSION: return False
        hs = [C.rob_halves_oe(rb1)] + [C.rob_halves_blk(rb1, a) for a in L.A32]
        return all(len(h[x]) >= C.MIN_HALF for h in hs for x in (0, 1))
    elig = sorted(p for p in C1 if conv_ok(p))
    common = [p for p in ps5 if p in set(elig)]
    elig_rob = sorted(p for p in C1 if conv_ok(p, "ROB"))
    common_rob = [p for p in common if p in set(elig_rob)]
    return C1, C2, common, common_rob


def error_rates(task, sess, ps):
    out = {}; pset = set(ps)
    for (study, pid, s), path in sorted(ad.person_files(task).items()):
        key = f"{study}_P{pid}"
        if s != sess or key not in pset or not ad.in_scope("confirmatory", study, pid, 1): continue
        rows = ad.load_session(path, task)
        out[key] = sum(1 for r in rows if r[3] == 0) / len(rows)
    return np.array([out[p] for p in ps])


def keyf(task):
    return (lambda t: t[1]) if task in ("Stroop", "Flanker") else (lambda t: "x")


def split_halves(task, Cs, ps, rng, kind):
    """[NRND, n, 2] half-scores of one session; each half from its own trials
    kind: 'PES' conventional; 'P' proportional (log ratio); 'A','B','AB' robust decomposition"""
    n = len(ps); H = np.empty((NRND, n, 2)); kf = keyf(task)
    for k, p in enumerate(ps):
        pe, pc, rb = Cs[p]
        Apc = C.strat_assign([kf(t) for t in pc], rng, NRND); rpc = np.array([t[2] for t in pc])
        mpc = [((Apc == x) * rpc).sum(1) / (Apc == x).sum(1) for x in (0, 1)]
        if kind in ("PES", "P"):
            Ape = C.strat_assign([kf(t) for t in pe], rng, NRND); rpe = np.array([t[2] for t in pe])
            for x in (0, 1):
                mpe = ((Ape == x) * rpe).sum(1) / (Ape == x).sum(1)
                H[:, k, x] = mpe - mpc[x] if kind == "PES" else np.log(mpe) - np.log(mpc[x])
        else:
            Arb = C.strat_assign([kf(t) for t in rb], rng, NRND)
            a = np.array([t[2] for t in rb])                       # RT(e+1) - RT(e-1)
            pre = np.array([pc_rt for pc_rt in pre_rts(task, p, rb)])  # RT(e-1)
            for x in (0, 1):
                m = (Arb == x); nn = np.maximum(m.sum(1), 1)
                A = (m * a).sum(1) / nn; B = (m * pre).sum(1) / nn - mpc[x]
                H[:, k, x] = {"A": A, "B": B, "AB": A + B}[kind]
    return H


PRE = {}


def pre_rts(task, p, rb):
    return PRE[(task, p, id(rb))]


def attach_pre(task, sess, Cs, ps):
    """RT of trial e-1 for every robust sequence (rob tuples carry the trial index e-1 in position 4)"""
    rows_of = {}
    for (study, pid, s), path in sorted(ad.person_files(task).items()):
        key = f"{study}_P{pid}"
        if s != sess or key not in ps: continue
        rows_of[key] = ad.load_session(path, task)
    for p in ps:
        rb = Cs[p][2]; rows = rows_of[p]
        PRE[(task, p, id(rb))] = [rows[t[4]][4] for t in rb]
        # consistency check: RT(e+1) - RT(e-1) recomputed from rows equals the stored value
        assert all(abs((rows[t[5]][4] - rows[t[4]][4]) - t[2]) < 1e-12 for t in rb)


def session_score(task, Cs, p, kind):
    pe, pc, rb = Cs[p]; mpc = np.mean([t[2] for t in pc])
    if kind == "PES": return np.mean([t[2] for t in pe]) - mpc
    if kind == "P": return np.log(np.mean([t[2] for t in pe])) - np.log(mpc)
    a = np.mean([t[2] for t in rb]); b = np.mean(pre_rts(task, p, rb)) - mpc
    return {"A": a, "B": b, "AB": a + b}[kind]


def gap_block(task, C1, C2, ps, rng, idx, kind):
    s1 = np.array([session_score(task, C1, p, kind) for p in ps]); s2 = np.array([session_score(task, C2, p, kind) for p in ps])
    rt_pt = cor(s1, s2); rt_b = C.boot_r(s1, s2, idx)
    H = split_halves(task, C1, ps, rng, kind)
    r_pt = np.array([cor(H[s, :, 0], H[s, :, 1]) for s in range(NRND)])
    rb = C.boot_r_splits(H, idx); sbb = sb(rb).mean(1)
    res = dict(n=len(ps), retest_r=r4(rt_pt), retest_r_CI=q(rt_b), retest_spearman=r4(spearmanr(s1, s2)[0]),
               r_mean=r4(r_pt.mean()), SB_mean=r4(sb(r_pt).mean()), SB_CI=q(sbb),
               SB_minus_retest=r4(sb(r_pt).mean() - rt_pt), SB_minus_retest_CI=q(sbb - rt_b),
               mean_ms=[round(1000 * float(s1.mean()), 1), round(1000 * float(s2.mean()), 1)] if kind != "P" else
               [round(float(s1.mean()), 4), round(float(s2.mean()), 4)])
    return res, s1, s2, H, sbb - rt_b


def var_components(H1, H2):
    """H1, H2: [NRND, n, 2]; p x (h:s) design; returns components averaged over splits (per split arrays too)"""
    Y = np.stack([H1, H2], axis=2)  # [NRND, n, s, h]
    m = Y.mean((1, 2, 3), keepdims=True)
    mp = Y.mean((2, 3), keepdims=True); ms = Y.mean((1, 3), keepdims=True)
    mps = Y.mean(3, keepdims=True); msh = Y.mean(1, keepdims=True)
    n = Y.shape[1]
    MSp = 4 * ((mp - m) ** 2).sum((1, 2, 3)) / (n - 1)
    MSps = 2 * ((mps - mp - ms + m) ** 2).sum((1, 2, 3)) / (n - 1)
    MSr = ((Y - mps - msh + ms) ** 2).sum((1, 2, 3)) / (2 * (n - 1))
    e = MSr; ps_ = (MSps - MSr) / 2; p_ = (MSp - MSps) / 4
    return p_, ps_, e


def vc_summary(p_, ps_, e):
    P, PS, E = p_.mean(), ps_.mean(), e.mean()
    return dict(sigma2_p=P, sigma2_ps=PS, sigma2_e=E,
                share_occasion_specific=PS / (P + PS),
                within_session_coef=(P + PS) / (P + PS + E / 2),
                between_session_coef=P / (P + PS + E / 2))


def run(task, ti):
    rng = np.random.default_rng(SEED + ti); rngb = np.random.default_rng(SEED)
    C1, C2, ps, psr = select_persons(task)
    n = len(ps); idx = rngb.integers(0, n, (NBOOT, n))
    out = dict(task=task, n=n, n_rob=len(psr))
    # ---- M0 and M3(c): conventional and proportional PES, both sessions split
    conv, c1, c2, H1, gconv = gap_block(task, C1, C2, ps, rng, idx, "PES")
    H2 = split_halves(task, C2, ps, rng, "PES")
    p_, ps_, e = var_components(H1, H2)
    vc = vc_summary(p_, ps_, e)
    # bootstrap of the components: recompute on resampled persons for every split (in chunks)
    bP, bPS, bE = [], [], []
    for b in range(NBOOT):
        pp, pps, ee = var_components(H1[:, idx[b]], H2[:, idx[b]])
        bP.append(pp.mean()); bPS.append(pps.mean()); bE.append(ee.mean())
    bP, bPS, bE = map(np.array, (bP, bPS, bE))
    out["M0_variance_components"] = dict(
        sigma2_p_ms2=round(1e6 * vc["sigma2_p"], 1), sigma2_p_CI=[round(1e6 * x, 1) for x in q(bP)],
        sigma2_ps_ms2=round(1e6 * vc["sigma2_ps"], 1), sigma2_ps_CI=[round(1e6 * x, 1) for x in q(bPS)],
        sigma2_e_ms2=round(1e6 * vc["sigma2_e"], 1), sigma2_e_CI=[round(1e6 * x, 1) for x in q(bE)],
        share_occasion_specific=r4(vc["share_occasion_specific"]), share_occasion_specific_CI=q(bPS / (bP + bPS)),
        within_session_coef=r4(vc["within_session_coef"]), within_session_coef_CI=q((bP + bPS) / (bP + bPS + bE / 2)),
        between_session_coef=r4(vc["between_session_coef"]), between_session_coef_CI=q(bP / (bP + bPS + bE / 2)),
        n_splits_negative_sigma2_ps=int((ps_ < 0).sum()), n_splits_negative_sigma2_p=int((p_ < 0).sum()))
    out["conventional"] = conv
    prop, pr1, pr2, _, gprop = gap_block(task, C1, C2, ps, rng, idx, "P")
    out["M3_proportional"] = dict(**prop, gap_P_minus_gap_conv=r4(prop["SB_minus_retest"] - conv["SB_minus_retest"]),
                                  gap_P_minus_gap_conv_CI=q(gprop - gconv))
    # ---- M3(a,b): overall speed
    m1 = np.array([np.mean([t[2] for t in C1[p][1]]) for p in ps]); m2 = np.array([np.mean([t[2] for t in C2[p][1]]) for p in ps])
    dM, dP = m2 - m1, c2 - c1
    bm = lambda x: q(x[idx].mean(1))
    out["M3_speed"] = dict(mean_pc_rt_ms=[round(1000 * m1.mean(), 1), round(1000 * m2.mean(), 1)],
                           change_pc_rt_ms=round(1000 * dM.mean(), 1), change_pc_rt_CI=[round(1000 * x, 1) for x in bm(dM)],
                           change_PES_ms=round(1000 * dP.mean(), 1), change_PES_CI=[round(1000 * x, 1) for x in bm(dP)],
                           r_change_pcRT_change_PES=r4(cor(dM, dP)), r_change_pcRT_change_PES_CI=q(C.boot_r(dM, dP, idx)),
                           spearman_change=r4(spearmanr(dM, dP)[0]),
                           r_pcRT_PES_s1=r4(cor(m1, c1)), r_pcRT_PES_s1_CI=q(C.boot_r(m1, c1, idx)),
                           r_pcRT_PES_s2=r4(cor(m2, c2)), r_pcRT_PES_s2_CI=q(C.boot_r(m2, c2, idx)))
    # ---- M2: error expectancy
    e1, e2 = error_rates(task, 1, ps), error_rates(task, 2, ps)
    dE = e2 - e1

    def resid(y, x):
        X = np.column_stack([np.ones_like(x), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b
    res1, res2 = resid(c1, e1), resid(c2, e2)
    rr_b = []
    for b in range(NBOOT):
        ib = idx[b]; rr_b.append(cor(resid(c1[ib], e1[ib]), resid(c2[ib], e2[ib])))
    rr_b = np.array(rr_b); raw_b = C.boot_r(c1, c2, idx)
    out["M2_error_expectancy"] = dict(
        mean_ER=[r4(e1.mean()), r4(e2.mean())], change_ER=r4(dE.mean()), change_ER_CI=bm(dE),
        r_ER_PES_s1=r4(cor(e1, c1)), r_ER_PES_s1_CI=q(C.boot_r(e1, c1, idx)), spearman_ER_PES_s1=r4(spearmanr(e1, c1)[0]),
        r_ER_PES_s2=r4(cor(e2, c2)), r_ER_PES_s2_CI=q(C.boot_r(e2, c2, idx)), spearman_ER_PES_s2=r4(spearmanr(e2, c2)[0]),
        r_change_ER_change_PES=r4(cor(dE, dP)), r_change_ER_change_PES_CI=q(C.boot_r(dE, dP, idx)),
        spearman_change=r4(spearmanr(dE, dP)[0]),
        retest_ER=r4(cor(e1, e2)), retest_ER_CI=q(C.boot_r(e1, e2, idx)),
        retest_PES_raw=r4(cor(c1, c2)), retest_PES_resid_ER=r4(cor(res1, res2)), retest_PES_resid_ER_CI=q(rr_b),
        resid_minus_raw=r4(cor(res1, res2) - cor(c1, c2)), resid_minus_raw_CI=q(rr_b - raw_b))
    # ---- M1: decomposition on robust sequences
    attach_pre(task, 1, C1, psr); attach_pre(task, 2, C2, psr)
    nr = len(psr); idxr = rngb.integers(0, nr, (NBOOT, nr))
    dec = {}
    scores = {}
    for kind in ("A", "B", "AB"):
        res, s1, s2, _, _ = gap_block(task, C1, C2, psr, rng, idxr, kind)
        dec[kind] = res; scores[kind] = (s1, s2)
    a1, a2 = scores["A"]; b1, b2 = scores["B"]
    dec["r_A_B_s1"] = r4(cor(a1, b1)); dec["r_A_B_s1_CI"] = q(C.boot_r(a1, b1, idxr))
    dec["r_A_B_s2"] = r4(cor(a2, b2)); dec["r_A_B_s2_CI"] = q(C.boot_r(a2, b2, idxr))
    out["M1_decomposition"] = dec
    return out


if __name__ == "__main__":
    OUT = sys.argv[1]; res = {}
    for ti, task in enumerate(L.TASKS):
        res[task] = run(task, ti)
        print(task, json.dumps({k: res[task][k] for k in ("n", "n_rob")}), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
