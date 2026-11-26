#!/usr/bin/env python3
"""Laboratory post-error tasks (Hedge et al., 2018; session 1; confirmatory scope of the preregistered secondary analysis):
partitions, half sizes, block-assignment sensitivity, and the shared baseline.

The post-error deviations are re-derived from the raw trials with the registered rules (adapter
copy adapter_hedge_gng_fix.py: valid RT .15-3 s, K = 5 trials within the block, baseline = correct
post-correct trials outside an 8-trial guard after any error, cells with >= 5 trials, z = (RT - m)/SD).
The re-derivation reproduces the registered derived tables exactly (checked below).

Measure: person-level mean deviation over the five post-error trials. Persons with >= 20 episodes.

Partitions (same episodes and persons throughout)
  A  odd/even episode split, all model rows (a trial in two windows of one half counts twice);
  B  odd/even episode split, each trial once (in the half of its earliest episode);
  C  block split, each trial once; windows never cross blocks. Summarized as the mean correlation over
     all 10 assignments of three versus two blocks (each assignment once), with the odd/even
     assignment (1, 3, 5 vs 2, 4) and the 15 two-versus-two assignments reported for reference.
Baselines
  REG  registered: condition x block-pair cells ({1,2}, {3,4}, {5}) estimated from the whole session.
       Under a block split both halves use the same baseline estimates.
  OWN  each half's cells estimated only from that half's blocks, at the registered resolution where the
       half contains both blocks of a pair and at block resolution otherwise. Nothing is shared.
  FINE condition x block cells from the whole session (a preregistered multiverse alternative; under a
       block split nothing is shared). Small cells can give very small SDs and extreme z values.
  HALF condition cells estimated within each half (nothing shared; coarser in time).
Equal-precision control under REG (C, odd/even and all 10 assignments): the baseline trials of every
cell are split at random into two halves; SH = both partition halves use the same random half, DJ =
each partition half uses its own. 50 random baseline splits; both orientations averaged.
Uncertainty: 2,000 bootstrap resamples of persons, the same resamples for every coefficient, giving
intervals for coefficients and for differences between them.

Usage: HEDGE_RAW=<.../RawData> python3 lab_partition_audit.py OUT.json [REGISTERED_DIR GNG_FIXED_DIR]
"""
import csv, gzip, importlib.util, itertools, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("adapter", os.environ.get("ADAPTER", os.path.join(HERE, "adapter_hedge_gng_fix.py")))
ad = importlib.util.module_from_spec(spec); spec.loader.exec_module(ad)
CFG, K, GUARD = ad.CFG, 5, 8
LO, HI = CFG["rt_valid_range_s"]
TASKS = ("Stroop", "Flanker", "Stopsignal", "Gonogo")
NBOOT, NSPLIT = 2000, 50
BLOCKS = (1, 2, 3, 4, 5)
A32 = [(frozenset(c), frozenset(set(BLOCKS) - set(c))) for c in itertools.combinations(BLOCKS, 3)]
A22 = []
for drop in BLOCKS:
    rem = [b for b in BLOCKS if b != drop]
    for c in rem[1:]:
        A22.append((frozenset({rem[0], c}), frozenset(set(rem) - {rem[0], c})))
ODDEVEN = (frozenset({1, 3, 5}), frozenset({2, 4}))
sb = lambda r: 2 * r / (1 + r)
q = lambda x: [round(float(np.quantile(x, .025)), 4), round(float(np.quantile(x, .975)), 4)]


def derive(task):
    persons = {}
    for (study, pid, sess), path in sorted(ad.person_files(task).items()):
        if sess != 1 or not ad.in_scope("confirmatory", study, pid, sess):
            continue
        rows = ad.load_session(path, task); n = len(rows)
        err = {i for i, r in enumerate(rows) if r[3] == 0}
        cont = set()
        for i in err: cont.update(range(i, min(n, i + GUARD + 1)))
        base = [i for i, (blk, tr, cond, cor, rt) in enumerate(rows)
                if cor == 1 and i not in cont and i >= 1 and rows[i - 1][3] == 1 and LO <= rt <= HI]
        epis = [(i, [(d, i + d) for d in range(1, K + 1) if i + d < n and rows[i + d][0] == rows[i][0]]) for i in sorted(err)]
        persons[f"{study}_P{pid}"] = dict(rows=rows, base=base, epis=epis)
    return persons


def cells(rows, idx, res):
    v = defaultdict(list)
    for i in idx:
        blk, tr, cond, cor, rt = rows[i]; v[ad.cell_key(cond, blk, res)].append(rt)
    out = {}
    for k, x in v.items():
        if len(x) >= 5:
            x = np.array(x); sd = x.std(ddof=1)
            if sd > 0: out[k] = (x.mean(), sd)
    return out


def dev(rows, j, cl, res):
    blk, tr, cond, cor, rt = rows[j]
    if not (LO <= rt <= HI): return None
    c = cl.get(ad.cell_key(cond, blk, res))
    return None if c is None else (rt - c[0]) / c[1]


def boot_r(a, b, idx):
    A = a[idx]; B = b[idx]
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))


def task_audit(task, reg_dir=None, gng_dir=None):
    P = derive(task)
    reg = {p: cells(d["rows"], d["base"], "medium") for p, d in P.items()}
    E = {}
    for p, d in P.items():
        E[p] = [(i, d["rows"][i][0], win) for i, win in d["epis"]
                if any(dev(d["rows"], j, reg[p], "medium") is not None for _, j in win)]
    persons = sorted(p for p in E if len(E[p]) >= 20)
    out = {"task": task, "n_persons_floor": len(persons), "n_episodes_all": sum(len(v) for v in E.values())}
    # S over all rows with a valid registered deviation (all persons), odd/even episode split
    tot = sh = 0
    for p in E:
        half = {i: k % 2 for k, (i, b, w) in enumerate(E[p])}
        seen = defaultdict(set); rws = []
        for i, b, w in E[p]:
            for d, j in w:
                if dev(P[p]["rows"], j, reg[p], "medium") is not None:
                    seen[j].add(half[i]); rws.append(j)
        tot += len(rws); sh += sum(len(seen[j]) > 1 for j in rws)
    out["S_episode"] = round(sh / tot, 4); out["n_rows_all"] = tot
    if reg_dir:
        d_ = gng_dir if task == "Gonogo" else reg_dir
        epi = {r["eid"]: r for r in csv.DictReader(gzip.open(f"{d_}/episodes.csv.gz", "rt")) if r["task"] == task and r["session"] == "1"}
        ref = {}
        for r in csv.DictReader(gzip.open(f"{d_}/frames_long.csv.gz", "rt")):
            e = epi.get(r["eid"])
            if e is not None and r["dev_rt"] not in ("", None):
                ref[(e["phash"], int(e["block"]), int(e["trial"]), int(r["d"]))] = float(r["dev_rt"])
        mine = {}
        for p, L in E.items():
            rows = P[p]["rows"]
            for i, b, w in L:
                for d, j in w:
                    v = dev(rows, j, reg[p], "medium")
                    if v is not None: mine[(p, b, rows[i][1], d)] = round(v, 5)
        com = set(ref) & set(mine)
        out["validation"] = dict(n_registered=len(ref), n_rederived=len(mine), n_common=len(com),
                                 max_abs_diff=max(abs(mine[k] - ref[k]) for k in com))

    def scores(part, scheme, assign=None, bsets=None):
        res = {}
        for p in persons:
            rows, base = P[p]["rows"], P[p]["base"]
            if scheme == "REG": cl = (reg[p], reg[p]); rs = "medium"
            elif scheme == "FINE": f = cells(rows, base, "fine"); cl = (f, f); rs = "fine"
            elif scheme == "OWN": cl = tuple(cells(rows, [i for i in base if rows[i][0] in assign[h]], "medium") for h in (0, 1)); rs = "medium"
            elif scheme == "HALF": cl = tuple(cells(rows, [i for i in base if rows[i][0] in assign[h]], "coarse") for h in (0, 1)); rs = "coarse"
            else: cl = bsets[p]; rs = "medium"
            vals = [[], []]; ne = [0, 0]; first = {}
            for k, (i, b, w) in enumerate(E[p]):
                if part == "C":
                    if b in assign[0]: h = 0
                    elif b in assign[1]: h = 1
                    else: continue
                else:
                    h = k % 2
                ne[h] += 1
                for d, j in w:
                    v = dev(rows, j, cl[h], rs)
                    if v is None: continue
                    if part == "A": vals[h].append(v)
                    elif j not in first: first[j] = (h, v)
            if part != "A":
                for j, (h, v) in first.items(): vals[h].append(v)
            if vals[0] and vals[1]:
                res[p] = (np.mean(vals[0]), np.mean(vals[1]), len(vals[0]), len(vals[1]), ne[0], ne[1])
        return res

    def info(hs, ps):
        n0 = np.array([hs[p][2] for p in ps]); n1 = np.array([hs[p][3] for p in ps])
        a = np.array([hs[p][0] for p in ps]); b = np.array([hs[p][1] for p in ps])
        return dict(obs_half_total=[int(n0.sum()), int(n1.sum())],
                    episodes_half_total=[int(sum(hs[p][4] for p in ps)), int(sum(hs[p][5] for p in ps))],
                    median_small_large_ratio=round(float(np.median(np.minimum(n0, n1) / np.maximum(n0, n1))), 3),
                    var_half=[round(float(a.var(ddof=1)), 4), round(float(b.var(ddof=1)), 4)])

    rng = np.random.default_rng(20261017)
    out["schemes"] = {}
    for scheme in ("REG", "FINE", "OWN", "HALF"):
        S = {}
        if scheme in ("REG", "FINE"):
            S["A"] = scores("A", scheme); S["B"] = scores("B", scheme)
        for g in A32 + A22:
            S[("C",) + tuple(sorted(g[0]))] = scores("C", scheme, g)
        common = sorted(set.intersection(*[set(v) for v in S.values()]))
        idx = rng.integers(0, len(common), size=(NBOOT, len(common)))
        arr = lambda hs: (np.array([hs[p][0] for p in common]), np.array([hs[p][1] for p in common]))
        blk = {"n": len(common)}
        boots = {}
        for k in ("A", "B"):
            if k in S:
                a, b = arr(S[k]); r = float(np.corrcoef(a, b)[0, 1]); rb = boot_r(a, b, idx); boots[k] = rb
                blk[k] = dict(r=round(r, 4), r_CI=q(rb), SB=round(sb(r), 4), SB_CI=q(sb(rb)), **info(S[k], common))
        r32 = []; b32 = []
        for g in A32:
            a, b = arr(S[("C",) + tuple(sorted(g[0]))]); r32.append(float(np.corrcoef(a, b)[0, 1])); b32.append(boot_r(a, b, idx))
        rb32 = np.mean(b32, axis=0); boots["C32"] = rb32
        blk["C_3v2_mean"] = dict(r=round(float(np.mean(r32)), 4), r_CI=q(rb32), r_min=round(min(r32), 4), r_max=round(max(r32), 4),
                                 SB=round(float(np.mean([sb(x) for x in r32])), 4), SB_CI=q(np.mean([sb(x) for x in b32], axis=0)))
        a, b = arr(S[("C", 1, 3, 5)]); r = float(np.corrcoef(a, b)[0, 1]); rb = boot_r(a, b, idx)
        blk["C_oddeven"] = dict(r=round(r, 4), r_CI=q(rb), SB=round(sb(r), 4), SB_CI=q(sb(rb)), **info(S[("C", 1, 3, 5)], common))
        r22 = [float(np.corrcoef(*arr(S[("C",) + tuple(sorted(g[0]))]))[0, 1]) for g in A22]
        blk["C_2v2"] = dict(k=len(r22), r_mean=round(float(np.mean(r22)), 4), r_min=round(min(r22), 4), r_max=round(max(r22), 4))
        if "A" in boots:
            blk["diff_r"] = {"A-B": q(boots["A"] - boots["B"]), "B-C32": q(boots["B"] - boots["C32"]), "A-C32": q(boots["A"] - boots["C32"])}
        blk["_S"] = S; blk["_common"] = common
        out["schemes"][scheme] = blk
    # Paired comparisons of the block split between baseline schemes: common persons, one index matrix
    shares = lambda g: any((a in g[0]) != (b in g[0]) for a, b in ((1, 2), (3, 4)))   # a block pair divided?
    out["assignments_sharing_registered_cells"] = dict(k_3v2=sum(shares(g) for g in A32), k_2v2=sum(shares(g) for g in A22),
                                                      oddeven_shares_all_pairs=True)
    sch = out["schemes"]
    cmn = sorted(set.intersection(*[set(sch[k]["_common"]) for k in ("REG", "OWN", "FINE", "HALF")]))
    idx = rng.integers(0, len(cmn), size=(NBOOT, len(cmn)))
    def c32(S, which=None):
        gs = [g for g in A32 if which is None or shares(g) == which]
        rs, bs = [], []
        for g in gs:
            hs = S[("C",) + tuple(sorted(g[0]))]
            a = np.array([hs[p][0] for p in cmn]); b = np.array([hs[p][1] for p in cmn])
            rs.append(float(np.corrcoef(a, b)[0, 1])); bs.append(boot_r(a, b, idx))
        return float(np.mean(rs)), np.mean(bs, axis=0)
    pc = {"n": len(cmn)}
    for k in ("REG", "OWN", "FINE", "HALF"):
        r_all, b_all = c32(sch[k]["_S"]); r_sh, _ = c32(sch[k]["_S"], True); r_ns, _ = c32(sch[k]["_S"], False)
        hs = sch[k]["_S"][("C", 1, 3, 5)]
        a = np.array([hs[p][0] for p in cmn]); b = np.array([hs[p][1] for p in cmn])
        pc[k] = dict(r_3v2_mean=round(r_all, 4), r_3v2_CI=q(b_all), r_mean_sharing_assignments=round(r_sh, 4),
                     r_mean_nonsharing_assignments=round(r_ns, 4), r_oddeven=round(float(np.corrcoef(a, b)[0, 1]), 4))
        pc[k]["_b"] = b_all; pc[k]["_boe"] = boot_r(a, b, idx)
    for k in ("OWN", "FINE", "HALF"):
        pc[f"REG_minus_{k}"] = dict(r_3v2_mean=round(pc["REG"]["r_3v2_mean"] - pc[k]["r_3v2_mean"], 4),
                                    r_3v2_CI=q(pc["REG"]["_b"] - pc[k]["_b"]),
                                    r_oddeven=round(pc["REG"]["r_oddeven"] - pc[k]["r_oddeven"], 4),
                                    r_oddeven_CI=q(pc["REG"]["_boe"] - pc[k]["_boe"]))
    for k in ("REG", "OWN", "FINE", "HALF"):
        pc[k].pop("_b"); pc[k].pop("_boe")
    out["paired_scheme_comparison_C"] = pc
    # baseline cell sizes (odd/even assignment for the within-half schemes)
    def cellsizes(rows, idx_, res):
        v = defaultdict(int)
        for i in idx_:
            blk, tr, cond, cor, rt = rows[i]; v[ad.cell_key(cond, blk, res)] += 1
        return list(v.values())
    cs = {}
    for k in ("REG", "FINE", "OWN", "HALF"):
        sizes = []
        for p in persons:
            rows, base = P[p]["rows"], P[p]["base"]
            if k == "REG": sizes += cellsizes(rows, base, "medium")
            elif k == "FINE": sizes += cellsizes(rows, base, "fine")
            else:
                for h in (0, 1):
                    sizes += cellsizes(rows, [i for i in base if rows[i][0] in ODDEVEN[h]], "medium" if k == "OWN" else "coarse")
        sizes = np.array(sizes)
        cs[k] = dict(cells=int(len(sizes)), median_trials=float(np.median(sizes)), p10_trials=float(np.percentile(sizes, 10)),
                     share_cells_below_5=round(float((sizes < 5).mean()), 4))
    out["baseline_cell_sizes"] = cs
    for v in out["schemes"].values():
        v.pop("_S"); v.pop("_common")
    # equal-precision control
    arrs = {"SH": [], "DJ": []}
    for s in range(NSPLIT):
        r2 = np.random.default_rng(1000 + s)
        S1, S2 = {}, {}
        for p in persons:
            rows = P[p]["rows"]; byc = defaultdict(list)
            for i in P[p]["base"]:
                blk, tr, cond, cor, rt = rows[i]; byc[ad.cell_key(cond, blk, "medium")].append(i)
            s1, s2 = [], []
            for k in sorted(byc):
                L = list(byc[k]); r2.shuffle(L); s1 += L[0::2]; s2 += L[1::2]
            c1, c2 = cells(rows, s1, "medium"), cells(rows, s2, "medium")
            S1[p] = {k: v for k, v in c1.items() if k in c2}; S2[p] = {k: v for k, v in c2.items() if k in c1}
        for g in A32:  # includes the odd/even assignment
            pair = {}
            for lab, bs in (("SH1", {p: (S1[p], S1[p]) for p in persons}), ("SH2", {p: (S2[p], S2[p]) for p in persons}),
                            ("DJ12", {p: (S1[p], S2[p]) for p in persons}), ("DJ21", {p: (S2[p], S1[p]) for p in persons})):
                pair[lab] = scores("C", "CTRL", g, bs)
            arrs["SH"].append((s, tuple(sorted(g[0])), pair["SH1"], pair["SH2"]))
            arrs["DJ"].append((s, tuple(sorted(g[0])), pair["DJ12"], pair["DJ21"]))
    common = sorted(set.intersection(*[set(x[2]) & set(x[3]) for x in arrs["SH"] + arrs["DJ"]]))
    idx = rng.integers(0, len(common), size=(NBOOT, len(common)))
    def rr(hs):
        a = np.array([hs[p][0] for p in common]); b = np.array([hs[p][1] for p in common])
        return float(np.corrcoef(a, b)[0, 1]), boot_r(a, b, idx)
    res = {}
    for which, keyset in (("oddeven", [(1, 3, 5)]), ("3v2_mean", [tuple(sorted(g[0])) for g in A32])):
        sh_pt, dj_pt, sh_b, dj_b, diffs_by_split = [], [], [], [], []
        for s in range(NSPLIT):
            for key in keyset:
                (_, _, h1, h2), = [x for x in arrs["SH"] if x[0] == s and x[1] == key]
                (_, _, d1, d2), = [x for x in arrs["DJ"] if x[0] == s and x[1] == key]
                a1, b1 = rr(h1); a2, b2 = rr(h2); c1, e1 = rr(d1); c2, e2 = rr(d2)
                sh_pt.append((a1 + a2) / 2); dj_pt.append((c1 + c2) / 2)
                sh_b.append((b1 + b2) / 2); dj_b.append((e1 + e2) / 2)
        sh_pt, dj_pt = np.array(sh_pt), np.array(dj_pt)
        res[which] = dict(n=len(common), r_shared=round(float(sh_pt.mean()), 4), r_separate=round(float(dj_pt.mean()), 4),
                          diff=round(float((sh_pt - dj_pt).mean()), 4),
                          diff_range_over_splits=[round(float((sh_pt - dj_pt).min()), 4), round(float((sh_pt - dj_pt).max()), 4)],
                          share_of_splits_positive=round(float(((sh_pt - dj_pt) > 0).mean()), 3),
                          diff_CI=q(np.mean(sh_b, axis=0) - np.mean(dj_b, axis=0)))
    out["equal_precision_control"] = res
    return out


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "lab_partition_audit.json"
    reg_dir = sys.argv[2] if len(sys.argv) > 2 else None
    gng_dir = sys.argv[3] if len(sys.argv) > 3 else None
    allres = []
    for task in TASKS:
        r = task_audit(task, reg_dir, gng_dir); allres.append(r)
        print(json.dumps({"task": task, "n": r["n_persons_floor"], "S": r["S_episode"], "validation": r.get("validation")}), flush=True)
    json.dump(allres, open(out_path, "w"), indent=1)
