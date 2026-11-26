#!/usr/bin/env python3
"""Third-round analyses (plan: plans/ANALYSIS_PLAN_revision.md; exploratory).
Primary estimand: the disattenuated between-session correlation rho = r12 / sqrt(rel1 * rel2), where rel_s is the
stratified random-split Spearman-Brown coefficient of session s. Event-matched PES for stop-signal (failed vs.
successful stops) and go/no-go (commissions vs. correct rejections); conventional PES for Stroop and flanker.
Robust variants, local-baseline scores, post-signal slowing, benchmarks (mean RT, error rate, interference effect,
SSRT, commission rate), post-event accuracy, error expectancy, block course, and a pooled summary.

Usage: HEDGE_RAW=<.../RawData> python3 stability.py OUT.json
"""
import csv, importlib.util, json, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mech", os.path.join(HERE, "mechanisms.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
C = M.C; L = C.L; ad = C.ad
LO, HI = C.LO, C.HI
NBOOT, NRND, SEED = int(os.environ.get("NBOOT", 2000)), int(os.environ.get("NRND", 1000)), 20261126
MIN_EV, MIN_HALF = 10, 2
TASKS = ("Stroop", "Flanker", "Stopsignal", "Gonogo")
sb = lambda r: 2 * r / (1 + r)
q = lambda x: [round(float(np.nanquantile(x, .025)), 4), round(float(np.nanquantile(x, .975)), 4)]
r4 = lambda x: round(float(x), 4)
cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- data
def load6(path, task):
    """rows (block, trial, cond, correct, rt_s, ssd_s), same filtering and go/no-go block recovery as ad.load_session"""
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 6 or not r[0].strip(): continue
            try:
                rt = float(r[5]); ssd = 0.0
                if task == "Stopsignal":
                    rt /= 1000.0; ssd = float(r[2]) / 1000.0
                rows.append((int(r[0]), int(float(r[1])), r[3].strip(), int(float(r[4])), rt, ssd))
            except ValueError:
                continue
    if task == "Gonogo" and rows and len({r[0] for r in rows}) == 1:
        blk, prev, out = 1, None, []
        for (b, tr, cond, cr, rt, ssd) in rows:
            if prev is not None and tr < prev: blk += 1
            out.append((blk, tr, cond, cr, rt, ssd)); prev = tr
        rows = out
    ref = ad.load_session(path, task)
    nn = lambda t: tuple(-1.0 if (isinstance(v, float) and v != v) else v for v in t)
    assert [nn(x[:5]) for x in rows] == [nn(tuple(y)) for y in ref], (path, "loader mismatch")
    return rows


def load_all(task):
    S = {1: {}, 2: {}}
    for (study, pid, s), path in sorted(ad.person_files(task).items()):
        S[s][f"{study}_P{pid}"] = load6(path, task)
    return S


def needs_resp(task, cond): return True if task in ("Stroop", "Flanker") else cond == "0"


PRED = {  # (event predicate, control predicate) on the preceding trial
    "conv": (lambda t, p: p[3] == 0, lambda t, p: p[3] == 1 and needs_resp(t, p[2])),
    "ss_em": (lambda t, p: p[2] == "1" and p[3] == 0, lambda t, p: p[2] == "1" and p[3] == 1),
    "gng_em": (lambda t, p: p[2] == "2" and p[3] == 0, lambda t, p: p[2] == "2" and p[3] == 1),
    "postsig_ss": (lambda t, p: p[2] == "1" and p[3] == 1, lambda t, p: p[2] == "0" and p[3] == 1),
    "postsig_gng": (lambda t, p: p[2] == "2" and p[3] == 1, lambda t, p: p[2] == "0" and p[3] == 1),
}
PRIMARY = {"Stroop": "conv", "Flanker": "conv", "Stopsignal": "ss_em", "Gonogo": "gng_em"}
STRAT = lambda task, cond: cond if task in ("Stroop", "Flanker") else "x"


def pairs(rows, task, kind, value="rt"):
    """post-event and control scored trials: lists of (value, stratum, block, event index)"""
    pe, pc = [], []
    fe, fc = PRED[kind]
    for i in range(1, len(rows)):
        cur, prv = rows[i], rows[i - 1]
        if prv[0] != cur[0] or not needs_resp(task, cur[2]): continue
        if value == "rt":
            if not (cur[3] == 1 and LO <= cur[4] <= HI): continue
            v = cur[4]
        else:
            v = float(cur[3])
        item = (v, STRAT(task, cur[2]), cur[0], i - 1)
        if fe(task, prv): pe.append(item)
        elif fc(task, prv): pc.append(item)
    return pe, pc


def e4_sequences(rows, task):
    """errors with correct, valid, response-requiring trials e-4, e-1, e+1 in the error's block; non-overlapping windows"""
    out, last = [], -10
    ok = lambda j, b: 0 <= j < len(rows) and rows[j][0] == b and rows[j][3] == 1 and needs_resp(task, rows[j][2]) and LO <= rows[j][4] <= HI
    for e in range(4, len(rows) - 1):
        if rows[e][3] != 0: continue
        b = rows[e][0]
        if not (ok(e - 4, b) and ok(e - 1, b) and ok(e + 1, b)): continue
        if e - 4 <= last: continue
        out.append((rows[e + 1][4] - rows[e - 4][4], rows[e - 4][4] - rows[e - 1][4], STRAT(task, rows[e + 1][2]), b))
        last = e + 1
    return out


# ---------------------------------------------------------------- splits
def strat_assign(keys, rng, nsplit):
    keys = np.asarray(keys); A = np.zeros((nsplit, len(keys)), dtype=np.int8)
    for kk in np.unique(keys):
        idx = np.where(keys == kk)[0]; m = len(idx)
        perm = np.argsort(rng.random((nsplit, m)), axis=1)
        flip = rng.integers(0, 2, nsplit).astype(np.int8)
        val = (np.arange(m) % 2).astype(np.int8)[None, :] ^ flip[:, None]
        sub = np.empty((nsplit, m), dtype=np.int8); sub[np.arange(nsplit)[:, None], perm] = val
        A[:, idx] = sub
    return A


def hm(vals, mask):  # half means
    k = mask.sum(1); return (mask * vals).sum(1) / np.where(k > 0, k, np.nan)


def hmed(vals, mask):
    return np.nanmedian(np.where(mask, vals[None, :], np.nan), axis=1)


def hbal(vals, strata, mask):
    tot = np.zeros(mask.shape[0]); cnt = np.zeros(mask.shape[0])
    for u in np.unique(strata):
        sel = strata == u; m = mask[:, sel]; k = m.sum(1)
        mu = (m * vals[sel]).sum(1) / np.maximum(k, 1)
        tot += np.where(k > 0, mu, 0.0); cnt += (k > 0)
    return tot / cnt


class Comp:
    """one component (trial set) of a person-session with its random-split assignment"""
    def __init__(self, vals, strata, rng):
        self.v = np.asarray(vals, float); self.s = np.asarray(strata)
        self.A = strat_assign(self.s, rng, NRND) if len(self.v) else np.zeros((NRND, 0), np.int8)

    def mask(self, x): return self.A == x


def score_half(kind, comps, x):
    if kind == "diff": return hm(comps["ev"].v, comps["ev"].mask(x)) - hm(comps["ctl"].v, comps["ctl"].mask(x))
    if kind == "median": return hmed(comps["ev"].v, comps["ev"].mask(x)) - hmed(comps["ctl"].v, comps["ctl"].mask(x))
    if kind == "log": return np.log(hm(comps["ev"].v, comps["ev"].mask(x))) - np.log(hm(comps["ctl"].v, comps["ctl"].mask(x)))
    if kind == "bal": return hbal(comps["ev"].v, comps["ev"].s, comps["ev"].mask(x)) - hbal(comps["ctl"].v, comps["ctl"].s, comps["ctl"].mask(x))
    if kind == "mean": return hm(comps["x"].v, comps["x"].mask(x))
    if kind == "interf": return hm(comps["inc"].v, comps["inc"].mask(x)) - hm(comps["con"].v, comps["con"].mask(x))
    if kind == "ssrt":
        g, st = comps["go"], comps["stop"]
        mg = g.mask(x); ms_ = st.mask(x)
        p = (ms_ * (1 - st.v)).sum(1) / ms_.sum(1)                  # P(respond | signal); st.v = correct (1 = stopped)
        ssd = (ms_ * comps["ssd"].v).sum(1) / ms_.sum(1)
        srt = np.sort(np.where(mg, g.v[None, :], np.inf), axis=1); ng = mg.sum(1)
        k = np.clip(np.ceil(p * ng).astype(int) - 1, 0, ng - 1)
        return srt[np.arange(NRND), k] - ssd
    raise ValueError(kind)


def score_full(kind, comps):
    ones = {k: c for k, c in comps.items()}
    saveA = {k: c.A for k, c in comps.items()}
    for c in comps.values(): c.A = np.zeros((1, len(c.v)), np.int8)
    try:
        if kind == "ssrt":
            g, st = comps["go"], comps["stop"]
            p = (1 - st.v).mean(); ssd = comps["ssd"].v.mean(); srt = np.sort(g.v)
            k = int(np.clip(np.ceil(p * len(srt)) - 1, 0, len(srt) - 1)); out = srt[k] - ssd
        else:
            out = score_half(kind, comps, 0)[0]
    finally:
        for k, c in comps.items(): c.A = saveA[k]
    return float(out)


# ---------------------------------------------------------------- components per task, session, person
def build(task, S, rng):
    """returns dict sess -> pid -> dict of component sets"""
    out = {1: {}, 2: {}}
    pk = PRIMARY[task]
    for s in (1, 2):
        for p, rows in S[s].items():
            d = {}
            pe, pc = pairs(rows, task, pk)
            d["prim"] = {"ev": Comp([t[0] for t in pe], [t[1] for t in pe], rng), "ctl": Comp([t[0] for t in pc], [t[1] for t in pc], rng)}
            d["prim_raw"] = (pe, pc)
            ae, ac = pairs(rows, task, pk, value="acc")
            d["pea"] = {"ev": Comp([t[0] for t in ae], [t[1] for t in ae], rng), "ctl": Comp([t[0] for t in ac], [t[1] for t in ac], rng)}
            if task in ("Stopsignal", "Gonogo"):
                ce, cc = pairs(rows, task, "conv")
                d["conv"] = {"ev": Comp([t[0] for t in ce], [t[1] for t in ce], rng), "ctl": Comp([t[0] for t in cc], [t[1] for t in cc], rng)}
                d["conv_raw"] = (ce, cc)
                se, sc = pairs(rows, task, "postsig_ss" if task == "Stopsignal" else "postsig_gng")
                d["postsig"] = {"ev": Comp([t[0] for t in se], [t[1] for t in se], rng), "ctl": Comp([t[0] for t in sc], [t[1] for t in sc], rng)}
            else:
                seq = e4_sequences(rows, task)
                d["e4"] = {"x": Comp([t[0] for t in seq], [t[2] for t in seq], rng)}
                d["speed"] = {"x": Comp([t[1] for t in seq], [t[2] for t in seq], rng)}
                d["e4_raw"] = seq
                d["interf"] = {"inc": Comp([r[4] for r in rows if r[2] == "2" and r[3] == 1 and LO <= r[4] <= HI], ["x"] * sum(1 for r in rows if r[2] == "2" and r[3] == 1 and LO <= r[4] <= HI), rng),
                               "con": Comp([r[4] for r in rows if r[2] == "0" and r[3] == 1 and LO <= r[4] <= HI], ["x"] * sum(1 for r in rows if r[2] == "0" and r[3] == 1 and LO <= r[4] <= HI), rng)}
            corr = [r for r in rows if r[3] == 1 and needs_resp(task, r[2]) and LO <= r[4] <= HI]
            d["rt"] = {"x": Comp([r[4] for r in corr], [STRAT(task, r[2]) for r in corr], rng)}
            d["er"] = {"x": Comp([1.0 - r[3] for r in rows], [r[2] for r in rows], rng)}
            if task == "Stopsignal":
                go = [r for r in rows if r[2] == "0"]; stp = [r for r in rows if r[2] == "1"]
                gorts = [r[4] if r[4] > 0.05 else np.nan for r in go]
                mx = np.nanmax(gorts); gorts = [x if x == x else mx for x in gorts]   # omissions -> max go RT
                stopc = Comp([r[3] for r in stp], ["x"] * len(stp), rng)
                d["ssrt"] = {"go": Comp(gorts, ["x"] * len(go), rng), "stop": stopc,
                             "ssd": Comp([r[5] for r in stp], ["x"] * len(stp), rng)}
                d["ssrt"]["ssd"].A = stopc.A   # SSD follows the stop-trial assignment
            if task == "Gonogo":
                ng = [r for r in rows if r[2] == "2"]
                d["comm"] = {"x": Comp([1.0 - r[3] for r in ng], ["x"] * len(ng), rng)}
            out[s][p] = d
    return out


def eligible(task, B, raw_key):
    """persons with >= MIN_EV post-event trials per session and >= MIN_HALF per half under odd/even and block splits"""
    ok = []
    for p in sorted(set(B[1]) & set(B[2])):
        good = True
        for s in (1, 2):
            pe, pc = B[s][p][raw_key]
            if len(pe) < MIN_EV: good = False; break
            halves = [([t for k, t in enumerate(pe) if k % 2 == h], [t for k, t in enumerate(pc) if k % 2 == h]) for h in (0, 1)]
            if s == 1:
                for a in L.A32:
                    halves += [([t for t in pe if (t[2] in a[0]) == (h == 0)], [t for t in pc if (t[2] in a[0]) == (h == 0)]) for h in (0, 1)]
            if any(len(x) < MIN_HALF or len(y) < MIN_HALF for x, y in halves): good = False; break
        if good: ok.append(p)
    return ok


def eligible_seq(B, ps):
    ok = []
    for p in ps:
        good = True
        for s in (1, 2):
            seq = B[s][p]["e4_raw"]
            if len(seq) < MIN_EV: good = False; break
            hs = [[t for k, t in enumerate(seq) if k % 2 == h] for h in (0, 1)]
            if s == 1:
                for a in L.A32: hs += [[t for t in seq if (t[3] in a[0]) == (h == 0)] for h in (0, 1)]
            if any(len(h) < MIN_HALF for h in hs): good = False; break
        if good: ok.append(p)
    return ok


# ---------------------------------------------------------------- coefficients
def halves(B, s, ps, key, kind):
    H = np.empty((NRND, len(ps), 2))
    for j, p in enumerate(ps):
        for x in (0, 1): H[:, j, x] = score_half(kind, B[s][p][key], x)
    return H


def rcols(H):
    A = H[:, :, 0] - H[:, :, 0].mean(1, keepdims=True); Bm = H[:, :, 1] - H[:, :, 1].mean(1, keepdims=True)
    return (A * Bm).sum(1) / np.sqrt((A * A).sum(1) * (Bm * Bm).sum(1))


def coef(B, ps, key, kind, idx, scale=1000.0):
    H1, H2 = halves(B, 1, ps, key, kind), halves(B, 2, ps, key, kind)
    s1 = np.array([score_full(kind, B[1][p][key]) for p in ps]); s2 = np.array([score_full(kind, B[2][p][key]) for p in ps])
    rel1 = float(sb(rcols(H1)).mean()); rel2 = float(sb(rcols(H2)).mean()); r12 = cor(s1, s2)
    rb1 = sb(C.boot_r_splits(H1, idx)).mean(1); rb2 = sb(C.boot_r_splits(H2, idx)).mean(1); rtb = C.boot_r(s1, s2, idx)
    with np.errstate(invalid="ignore"):
        rho_b = np.where((rb1 > 0) & (rb2 > 0), rtb / np.sqrt(np.clip(rb1, 1e-12, None) * np.clip(rb2, 1e-12, None)), np.nan)
        gap_b = np.sqrt(np.clip(rb1 * rb2, 0, None)) - rtb
    rho = r12 / np.sqrt(rel1 * rel2) if rel1 > 0 and rel2 > 0 else float("nan")
    sd1, sd2 = s1.std(ddof=1), s2.std(ddof=1)
    rel_change = (sd1 ** 2 * rel1 + sd2 ** 2 * rel2 - 2 * sd1 * sd2 * r12) / (sd1 ** 2 + sd2 ** 2 - 2 * sd1 * sd2 * r12)
    d = s2 - s1
    res = dict(n=len(ps), rel1=r4(rel1), rel1_CI=q(rb1), rel2=r4(rel2), rel2_CI=q(rb2), r12=r4(r12), r12_CI=q(rtb),
               r12_spearman=r4(spearmanr(s1, s2)[0]), rho=r4(rho), rho_CI=q(rho_b), rho_undefined=int(np.isnan(rho_b).sum()),
               gap=r4(np.sqrt(max(rel1 * rel2, 0)) - r12), gap_CI=q(gap_b),
               mean=[r4(scale * s1.mean()), r4(scale * s2.mean())], sd=[r4(scale * sd1), r4(scale * sd2)],
               change=r4(scale * d.mean()), change_CI=[r4(scale * x) for x in q(d[idx].mean(1))], rel_change=r4(rel_change))
    return res, dict(H1=H1, H2=H2, s1=s1, s2=s2, rho_b=rho_b, rb1=rb1, rb2=rb2, rtb=rtb)


def oe_blk(B, ps, key, idx, rt_b, rt_pt):
    """session-1 odd/even and block splits of a diff score (post-event and control trials alternated by order)"""
    out = {}
    a = []
    for p in ps:
        pe, pc = B[1][p][key + "_raw"]
        a.append([np.mean([t[0] for k, t in enumerate(pe) if k % 2 == h]) - np.mean([t[0] for k, t in enumerate(pc) if k % 2 == h]) for h in (0, 1)])
    a = np.array(a); r = cor(a[:, 0], a[:, 1]); rb = C.boot_r(a[:, 0], a[:, 1], idx)
    out["OE"] = dict(SB=r4(sb(r)), SB_CI=q(sb(rb)))
    rs, bs = [], []
    for asg in L.A32:
        hh = []
        for p in ps:
            pe, pc = B[1][p][key + "_raw"]
            hh.append([np.mean([t[0] for t in pe if (t[2] in asg[0]) == (h == 0)]) - np.mean([t[0] for t in pc if (t[2] in asg[0]) == (h == 0)]) for h in (0, 1)])
        hh = np.array(hh); rs.append(cor(hh[:, 0], hh[:, 1])); bs.append(C.boot_r(hh[:, 0], hh[:, 1], idx))
    out["BLK"] = dict(SB=r4(sb(np.mean(rs))), SB_CI=q(sb(np.mean(bs, 0))))
    return out


def paired(bA, bB):
    d = bA - bB; return q(d)


# ---------------------------------------------------------------- error expectancy, PEA, block course
def rates(task, rows):
    out = dict(ER=np.mean([1 - r[3] for r in rows]))
    if task == "Stopsignal":
        st = [r for r in rows if r[2] == "1"]; go = [r for r in rows if r[2] == "0"]
        out.update(failed_stop=np.mean([1 - r[3] for r in st]), go_error=np.mean([1 - r[3] for r in go]))
    if task == "Gonogo":
        ng = [r for r in rows if r[2] == "2"]; go = [r for r in rows if r[2] == "0"]
        out.update(commission=np.mean([1 - r[3] for r in ng]), omission=np.mean([1 - r[3] for r in go]))
    return out


def local_expectancy(task, S, B, ps, s):
    diffs = []
    for p in ps:
        rows = S[s][p]; pe, _ = B[s][p]["prim_raw"]
        if not pe: diffs.append(np.nan); continue
        cnt = np.array([sum(1 for r in rows[max(0, t[3] - 20):t[3]] if r[3] == 0) for t in pe]); rt = np.array([t[0] for t in pe])
        med = np.median(cnt); rare, freq = rt[cnt <= med], rt[cnt > med]
        diffs.append(rare.mean() - freq.mean() if len(rare) >= 3 and len(freq) >= 3 else np.nan)
    return np.array(diffs)


def block_course(B, ps, s):
    out = np.full((len(ps), 5), np.nan)
    for j, p in enumerate(ps):
        pe, pc = B[s][p]["prim_raw"]
        for b in range(1, 6):
            e = [t[0] for t in pe if t[2] == b]; c = [t[0] for t in pc if t[2] == b]
            if len(e) >= 2 and len(c) >= 2: out[j, b - 1] = np.mean(e) - np.mean(c)
    return out


# ---------------------------------------------------------------- run
def run(task, ti, common_set):
    rng = np.random.default_rng(SEED + 10 * ti)
    S = load_all(task); B = build(task, S, rng)
    ps = eligible(task, B, "prim_raw"); n = len(ps)
    idx = np.random.default_rng(SEED + 10 * ti + 1).integers(0, n, (NBOOT, n))
    res = dict(task=task, n=n, persons=ps)
    out, boots = {}, {}
    out["PES"], boots["PES"] = coef(B, ps, "prim", "diff", idx)
    out["PES"]["post_event_trials_median"] = [float(np.median([len(B[s][p]["prim_raw"][0]) for p in ps])) for s in (1, 2)]
    out["PES"]["control_trials_median"] = [float(np.median([len(B[s][p]["prim_raw"][1]) for p in ps])) for s in (1, 2)]
    out["PES"].update(oe_blk(B, ps, "prim", idx, None, None))
    p0, ps_, e = M.var_components(boots["PES"]["H1"], boots["PES"]["H2"])
    out["PES"]["share_occasion_specific_VC"] = r4(ps_.mean() / (p0.mean() + ps_.mean()))
    for lab, key, kind in (("PES_median", "prim", "median"), ("PES_log", "prim", "log"), ("PEA", "pea", "diff"),
                           ("RT", "rt", "mean"), ("ER", "er", "mean")):
        out[lab], boots[lab] = coef(B, ps, key, kind, idx, scale=1.0 if lab in ("PES_log", "PEA", "ER") else 1000.0)
    if task in ("Stroop", "Flanker"):
        out["PES_bal"], boots["PES_bal"] = coef(B, ps, "prim", "bal", idx)
        out["INTERF"], boots["INTERF"] = coef(B, ps, "interf", "interf", idx)
        pse = eligible_seq(B, ps); idx_e = np.random.default_rng(SEED + 10 * ti + 2).integers(0, len(pse), (NBOOT, len(pse)))
        out["E4"], _ = coef(B, pse, "e4", "mean", idx_e); out["SPEED"], _ = coef(B, pse, "speed", "mean", idx_e)
        out["PES_on_E4_persons"], _ = coef(B, pse, "prim", "diff", idx_e)
    else:
        out["POSTSIG"], boots["POSTSIG"] = coef(B, ps, "postsig", "diff", idx)
        psc = [p for p in eligible(task, B, "conv_raw") if p in set(ps)]
        idx_c = np.random.default_rng(SEED + 10 * ti + 3).integers(0, len(psc), (NBOOT, len(psc)))
        out["PES_conv"], _ = coef(B, psc, "conv", "diff", idx_c)
        if task == "Stopsignal": out["SSRT"], boots["SSRT"] = coef(B, ps, "ssrt", "ssrt", idx)
        else: out["COMM"], boots["COMM"] = coef(B, ps, "comm", "mean", idx, scale=1.0)
    # paired differences in rho: PES minus each benchmark (and minus post-signal slowing)
    out["rho_PES_minus"] = {lab: dict(diff=r4(out["PES"]["rho"] - out[lab]["rho"]), CI=paired(boots["PES"]["rho_b"], boots[lab]["rho_b"]))
                            for lab in boots if lab not in ("PES", "PES_median", "PES_log", "PES_bal", "PEA")}
    out["rho_robust_minus_PES"] = {lab: dict(diff=r4(out[lab]["rho"] - out["PES"]["rho"]), CI=paired(boots[lab]["rho_b"], boots["PES"]["rho_b"]))
                                   for lab in ("PES_median", "PES_log", "PES_bal") if lab in boots}
    # outlier-robust: exclude persons > 3.5 robust SD in either session, recompute everything for PES
    s1, s2 = boots["PES"]["s1"], boots["PES"]["s2"]
    z = lambda x: (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
    keep = [p for p, a, b in zip(ps, np.abs(z(s1)) <= 3.5, np.abs(z(s2)) <= 3.5) if a and b]
    idx_o = np.random.default_rng(SEED + 10 * ti + 4).integers(0, len(keep), (NBOOT, len(keep)))
    out["PES_no_outliers"], _ = coef(B, keep, "prim", "diff", idx_o)
    out["PES_no_outliers"]["excluded"] = [p for p in ps if p not in set(keep)]
    # comparison sample of the earlier analyses: conventional PES, persons of conv_pes.py
    cs = [p for p in common_set if p in B[1] and p in B[2]]
    idx_cs = np.random.default_rng(SEED + 10 * ti + 5).integers(0, len(cs), (NBOOT, len(cs)))
    key_c = "prim" if task in ("Stroop", "Flanker") else "conv"
    out["PES_conv_comparison_sample"], _ = coef(B, cs, key_c, "diff", idx_cs)
    # error expectancy and speed
    R1 = {p: rates(task, S[1][p]) for p in ps}; R2 = {p: rates(task, S[2][p]) for p in ps}
    ee = {}
    for k in R1[ps[0]]:
        e1 = np.array([R1[p][k] for p in ps]); e2 = np.array([R2[p][k] for p in ps])
        ee[k] = dict(mean=[r4(e1.mean()), r4(e2.mean())], change=r4((e2 - e1).mean()), change_CI=q((e2 - e1)[idx].mean(1)),
                     r_with_PES_s1=r4(cor(e1, s1)), r_with_PES_s1_CI=q(C.boot_r(e1, s1, idx)),
                     r_with_PES_s2=r4(cor(e2, s2)), r_with_PES_s2_CI=q(C.boot_r(e2, s2, idx)),
                     r_change=r4(cor(e2 - e1, s2 - s1)), r_change_CI=q(C.boot_r(e2 - e1, s2 - s1, idx)),
                     spearman_change=r4(spearmanr(e2 - e1, s2 - s1)[0]))
    out["error_rates"] = ee
    loc = {}
    for s in (1, 2):
        d = local_expectancy(task, S, B, ps, s); ok = ~np.isnan(d); dd = d[ok]
        bi = np.random.default_rng(SEED + 10 * ti + 6 + s).integers(0, len(dd), (NBOOT, len(dd)))
        loc[f"s{s}"] = dict(n=int(ok.sum()), mean_ms=r4(1000 * dd.mean()), CI=[r4(1000 * x) for x in q(dd[bi].mean(1))])
    out["local_expectancy_rare_minus_frequent"] = loc
    m1, m2 = boots["RT"]["s1"], boots["RT"]["s2"]
    out["speed_change"] = dict(r_change_RT_change_PES=r4(cor(m2 - m1, s2 - s1)), CI=q(C.boot_r(m2 - m1, s2 - s1, idx)))
    a1, a2 = boots["PEA"]["s1"], boots["PEA"]["s2"]
    out["PEA_with_PES"] = dict(r_s1=r4(cor(a1, s1)), r_s1_CI=q(C.boot_r(a1, s1, idx)), r_s2=r4(cor(a2, s2)), r_s2_CI=q(C.boot_r(a2, s2, idx)),
                               r_change=r4(cor(a2 - a1, s2 - s1)), r_change_CI=q(C.boot_r(a2 - a1, s2 - s1, idx)))
    bc = {}
    for s in (1, 2):
        X = block_course(B, ps, s); mu = np.nanmean(X, 0)
        bi = np.random.default_rng(SEED + 10 * ti + 8 + s).integers(0, n, (NBOOT, n))
        bm = np.array([np.nanmean(X[b], 0) for b in bi])
        bc[f"s{s}"] = dict(mean_ms=[r4(1000 * x) for x in mu], CI_ms=[[r4(1000 * np.nanquantile(bm[:, k], .025)), r4(1000 * np.nanquantile(bm[:, k], .975))] for k in range(5)],
                           n_per_block=[int((~np.isnan(X[:, k])).sum()) for k in range(5)])
    out["block_course"] = bc
    res["results"] = out
    pool = dict(ps=ps, PES=boots["PES"], RT=boots["RT"])
    return res, pool


def pooled(pools):
    """mean rho over tasks with a joint bootstrap over unique participants"""
    U = sorted(set().union(*[set(p["ps"]) for p in pools.values()]))
    pos = {t: {p: j for j, p in enumerate(v["ps"])} for t, v in pools.items()}
    def rho_from(t, lab, ii):
        H1, H2 = pools[t][lab]["H1"][:, ii], pools[t][lab]["H2"][:, ii]
        r1 = sb(rcols(H1)).mean(); r2 = sb(rcols(H2)).mean(); r = cor(pools[t][lab]["s1"][ii], pools[t][lab]["s2"][ii])
        return r / np.sqrt(r1 * r2) if r1 > 0 and r2 > 0 else np.nan
    pt = {lab: np.mean([rho_from(t, lab, np.arange(len(pools[t]["ps"]))) for t in pools]) for lab in ("PES", "RT")}
    rng = np.random.default_rng(SEED + 99); bs = {"PES": [], "RT": []}
    for b in range(NBOOT):
        smp = rng.choice(U, len(U))
        for lab in ("PES", "RT"):
            v = []
            for t in pools:
                ii = np.array([pos[t][p] for p in smp if p in pos[t]])
                v.append(rho_from(t, lab, ii))
            bs[lab].append(np.mean(v))
    bP, bR = np.array(bs["PES"]), np.array(bs["RT"])
    return dict(n_unique=len(U), n_in_all_four=int(sum(all(p in pos[t] for t in pools) for p in U)),
                mean_rho_PES=r4(pt["PES"]), mean_rho_PES_CI=q(bP), mean_rho_RT=r4(pt["RT"]), mean_rho_RT_CI=q(bR),
                diff_PES_minus_RT=r4(pt["PES"] - pt["RT"]), diff_CI=q(bP - bR))


if __name__ == "__main__":
    OUT = sys.argv[1]; res, pools = {}, {}
    for ti, task in enumerate(TASKS):
        _, _, common, _ = M.select_persons(task)
        res[task], pools[task] = run(task, ti, common)
        print(task, res[task]["n"], json.dumps({k: res[task]["results"]["PES"][k] for k in ("rel1", "rel2", "r12", "rho", "rho_CI")}), flush=True)
    res["pooled"] = pooled(pools)
    print("pooled", json.dumps(res["pooled"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
