#!/usr/bin/env python3
"""Conventional post-error slowing (PES) and the five-trial score in the Hedge et al. (2018) data:
session-1 split halves (whole score recomputed within each half) versus the correlation between sessions,
for the same persons. Plan: ANALYSIS_PLAN_conventional_PES.md (written before any coefficient of the
conventional score was computed; exploratory).

Scores
  PES   traditional: mean RT of correct post-error trials minus mean RT of correct post-correct trials
        (trial n + 1 correct, response required, valid RT, same block as trial n).
  ROB   robust (Dutilh et al., 2012): mean over errors of RT(n + 1) - RT(n - 1), both correct, response
        required, valid, in the error's block.
  FIVE  mean standardized RT over the five trials after each error (registered rules), all rows or each
        trial once.
Partitions of session 1
  OE    odd/even (errors by order; post-correct trials by their own order), each half its own control mean
  OEsh  odd/even, both halves subtract the full-session post-correct mean (shared control)
  EQP   equal-precision control for sharing: post-correct trials split at random into P1, P2; shared = both
        halves use P1 (P2); separate = one half P1, the other P2; 50 random splits, both orientations
  RND   stratified random splits (1,000; seed 20261029); for PES, post-error and post-correct trials permuted
        separately within the condition of trial n + 1 (Stroop, flanker); for FIVE, episodes permuted
  BLK   three versus two blocks, mean r over the 10 assignments, then Spearman-Brown
Between sessions: Pearson r of full-session scores (each session scored from its own trials).
Uncertainty: 2,000 bootstrap resamples of persons, the same resamples for every coefficient of a task.

Usage: HEDGE_RAW=<.../RawData> python3 conv_pes.py OUT.json [--counts]
"""
import importlib.util, json, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lpa", os.path.join(HERE, "lab_partition_audit.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
ad = L.ad
LO, HI = L.LO, L.HI
NBOOT, NRND, NEQP = 2000, 1000, 50
MIN_PE_SESSION, MIN_HALF = 10, 2
sb = lambda r: 2 * r / (1 + r)
q = lambda x: [round(float(np.quantile(x, .025)), 4), round(float(np.quantile(x, .975)), 4)]
cor = lambda a, b: float(np.corrcoef(a, b)[0, 1])


def resp(task, cond):
    return True if task in ("Stroop", "Flanker") else cond == "0"


# ---------------------------------------------------------------- five-trial score (as lab_retest.py)
def derive_session(task, sess_want):
    persons = {}
    for (study, pid, sess), path in sorted(ad.person_files(task).items()):
        if sess != sess_want or not ad.in_scope("confirmatory", study, pid, 1):
            continue
        rows = ad.load_session(path, task); n = len(rows)
        err = {i for i, r in enumerate(rows) if r[3] == 0}
        cont = set()
        for i in err: cont.update(range(i, min(n, i + L.GUARD + 1)))
        base = [i for i, (blk, tr, cond, cor_, rt) in enumerate(rows)
                if cor_ == 1 and i not in cont and i >= 1 and rows[i - 1][3] == 1 and LO <= rt <= HI]
        epis = [(i, [(d, i + d) for d in range(1, L.K + 1) if i + d < n and rows[i + d][0] == rows[i][0]]) for i in sorted(err)]
        persons[f"{study}_P{pid}"] = dict(rows=rows, base=base, epis=epis)
    return persons


def episodes(d):
    reg = L.cells(d["rows"], d["base"], "medium")
    E = [(i, d["rows"][i][0], w) for i, w in d["epis"] if any(L.dev(d["rows"], j, reg, "medium") is not None for _, j in w)]
    return reg, E


def five_rows(d, reg, E):
    """rows of the five-trial score: (episode index, block, trial index j, value, first occurrence?)"""
    out, seen = [], set()
    for k, (i, b, w) in enumerate(E):
        for _, j in w:
            x = L.dev(d["rows"], j, reg, "medium")
            if x is None: continue
            out.append((k, b, j, x, j not in seen)); seen.add(j)
    return out


def five_halves(R, h_of_episode, once):
    v = [[], []]
    for k, b, j, x, first in R:
        if once and not first: continue
        v[h_of_episode[k]].append(x)
    return v


# ---------------------------------------------------------------- conventional scores
def conv_trials(rows, task):
    n = len(rows); pe, pc, rob = [], [], []
    for i in range(1, n):
        blk, tr, cond, c, rt = rows[i]
        if rows[i - 1][0] != blk or not (c == 1 and resp(task, cond) and LO <= rt <= HI):
            continue
        p = rows[i - 1]
        if p[3] == 0: pe.append((i, cond, rt, blk))
        elif p[3] == 1 and resp(task, p[2]): pc.append((i, cond, rt, blk))
    for e in range(1, n - 1):
        if rows[e][3] != 0: continue
        a, b = rows[e - 1], rows[e + 1]
        if a[0] != rows[e][0] or b[0] != rows[e][0]: continue
        if (a[3] == 1 and resp(task, a[2]) and LO <= a[4] <= HI and b[3] == 1 and resp(task, b[2]) and LO <= b[4] <= HI):
            rob.append((e, b[2], b[4] - a[4], rows[e][0], e - 1, e + 1))
    return pe, pc, rob


def load(task, sess):
    out = {}
    for (study, pid, s), path in sorted(ad.person_files(task).items()):
        if s != sess or not ad.in_scope("confirmatory", study, pid, 1):
            continue
        rows = ad.load_session(path, task)
        out[f"{study}_P{pid}"] = conv_trials(rows, task)
    return out


def oe_halves(pe, pc):
    h = [[[], []], [[], []]]
    for k, t in enumerate(pe): h[k % 2][0].append(t[2])
    for k, t in enumerate(pc): h[k % 2][1].append(t[2])
    return h  # h[half] = [pe RTs, pc RTs]


def blk_halves(pe, pc, assign):
    h = [[[], []], [[], []]]
    for t in pe: h[0 if t[3] in assign[0] else 1][0].append(t[2])
    for t in pc: h[0 if t[3] in assign[0] else 1][1].append(t[2])
    return h


def rob_halves_oe(rob):
    h = [[], []]
    for k, t in enumerate(rob): h[k % 2].append(t[2])
    return h


def rob_halves_blk(rob, assign):
    h = [[], []]
    for t in rob: h[0 if t[3] in assign[0] else 1].append(t[2])
    return h


def strat_assign(keys, rng, nsplit):
    """random halves of equal size within strata; returns [nsplit, len(keys)] of 0/1"""
    keys = np.asarray(keys); A = np.zeros((nsplit, len(keys)), dtype=np.int8)
    for kk in np.unique(keys):
        idx = np.where(keys == kk)[0]; m = len(idx)
        base = np.arange(m) % 2
        for s in range(nsplit):
            perm = rng.permutation(m); flip = rng.integers(0, 2)
            A[s, idx[perm]] = base ^ flip
    return A


def boot_r(a, b, idx):
    A = a[idx]; B = b[idx]
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))


def boot_r_splits(H, idx):
    """H: [nsplit, n, 2] half scores; returns [nboot, nsplit] correlations"""
    out = np.empty((idx.shape[0], H.shape[0]))
    for b in range(idx.shape[0]):
        A = H[:, idx[b], 0]; B = H[:, idx[b], 1]
        A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
        out[b] = (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1))
    return out


def op(r, n):
    """Olkin-Pratt (1958) approximately unbiased correlation, as used by Kahveci et al. (2025) before aggregation"""
    return r * (1 + (1 - r ** 2) / (2 * (n - 3)))


def summarize_split(r_pt, r_boot, retest_boot, retest_pt, n=None):
    """r_pt: [nsplit] point correlations; r_boot: [nboot, nsplit]"""
    sbs = sb(r_pt); sbb = sb(r_boot).mean(1)
    agg = {}
    if n is not None:  # aggregate the correlations first, then Spearman-Brown (all aggregated values are positive here)
        a_pt = sb(op(r_pt, n).mean()); a_b = sb(op(r_boot, n).mean(1))
        agg = dict(SB_of_aggregated_r=round(float(a_pt), 4), SB_of_aggregated_r_CI=q(a_b),
                   SB_of_aggregated_r_minus_retest=round(float(a_pt - retest_pt), 4), SB_of_aggregated_r_minus_retest_CI=q(a_b - retest_boot))
    return dict(**agg, r_mean=round(float(r_pt.mean()), 4), SB_mean=round(float(sbs.mean()), 4), SB_CI=q(sbb),
                SB_split_range=[round(float(np.quantile(sbs, .025)), 4), round(float(np.quantile(sbs, .975)), 4)],
                SB_MCSE=round(float(sbs.std(ddof=1) / np.sqrt(len(sbs))), 4),
                SB_minus_retest=round(float(sbs.mean() - retest_pt), 4), SB_minus_retest_CI=q(sbb - retest_boot))


def single(r_pt, r_boot, retest_boot, retest_pt):
    return dict(r=round(r_pt, 4), SB=round(sb(r_pt), 4), SB_CI=q(sb(r_boot)),
                SB_minus_retest=round(sb(r_pt) - retest_pt, 4), SB_minus_retest_CI=q(sb(r_boot) - retest_boot))


def run(task, counts_only=False):
    rng = np.random.default_rng(20261029)
    C1, C2 = load(task, 1), load(task, 2)
    D1, D2 = derive_session(task, 1), derive_session(task, 2)
    X1 = {p: episodes(d) for p, d in D1.items()}; X2 = {p: episodes(d) for p, d in D2.items()}
    # five-trial person set of the session comparison (lab_retest.py)
    def blk_ok(p):
        return all(all(halves5(D1[p], *X1[p], a)[h] for h in (0, 1)) for a in L.A32)
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
    keep1 = [p for p in sorted(X1) if len(X1[p][1]) >= 20 and blk_ok(p)]
    ps5 = [p for p in keep1 if p in X2 and len(X2[p][1]) >= 20]

    def conv_ok(p, which="PES"):
        if p not in C1 or p not in C2: return False
        pe1, pc1, rb1 = C1[p]; pe2, pc2, rb2 = C2[p]
        if which == "PES":
            if len(pe1) < MIN_PE_SESSION or len(pe2) < MIN_PE_SESSION: return False
            hs = [oe_halves(pe1, pc1)] + [blk_halves(pe1, pc1, a) for a in L.A32]
            return all(len(h[x][0]) >= MIN_HALF and len(h[x][1]) >= MIN_HALF for h in hs for x in (0, 1))
        if len(rb1) < MIN_PE_SESSION or len(rb2) < MIN_PE_SESSION: return False
        hs = [rob_halves_oe(rb1)] + [rob_halves_blk(rb1, a) for a in L.A32]
        return all(len(h[x]) >= MIN_HALF for h in hs for x in (0, 1))

    elig = sorted(p for p in C1 if conv_ok(p))
    common = [p for p in ps5 if p in set(elig)]
    elig_rob = sorted(p for p in C1 if conv_ok(p, "ROB"))
    common_rob = [p for p in common if p in set(elig_rob)]
    info = dict(n_five_session_set=len(ps5), n_conv_eligible=len(elig), n_common=len(common),
                n_rob_eligible=len(elig_rob), n_common_rob=len(common_rob),
                median_pe_trials_s1=float(np.median([len(C1[p][0]) for p in common])) if common else None,
                median_pe_trials_s2=float(np.median([len(C2[p][0]) for p in common])) if common else None,
                median_pc_trials_s1=float(np.median([len(C1[p][1]) for p in common])) if common else None,
                median_errors_s1=float(np.median([len(X1[p][1]) for p in common])) if common else None,
                median_rob_s1=float(np.median([len(C1[p][2]) for p in common_rob])) if common_rob else None,
                five_excluded_by_conv_floor=[p for p in ps5 if p not in set(elig)])
    if counts_only:
        return info

    out = {"task": task, "counts": info}

    def pes_block(ps, label):
        n = len(ps); idx = rng.integers(0, n, (NBOOT, n))
        s1 = np.array([np.mean([t[2] for t in C1[p][0]]) - np.mean([t[2] for t in C1[p][1]]) for p in ps])
        s2 = np.array([np.mean([t[2] for t in C2[p][0]]) - np.mean([t[2] for t in C2[p][1]]) for p in ps])
        rt_pt = cor(s1, s2); rt_b = boot_r(s1, s2, idx)
        z = lambda x: (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))))
        ok = (np.abs(z(s1)) <= 3.5) & (np.abs(z(s2)) <= 3.5)
        res = dict(n=n, retest_r=round(rt_pt, 4), retest_r_CI=q(rt_b), retest_spearman=round(float(spearmanr(s1, s2)[0]), 4),
                   retest_r_without_outliers=round(cor(s1[ok], s2[ok]), 4), n_without_outliers=int(ok.sum()),
                   mean_PES_ms=[round(1000 * float(s1.mean()), 1), round(1000 * float(s2.mean()), 1)],
                   sd_PES_ms=[round(1000 * float(s1.std(ddof=1)), 1), round(1000 * float(s2.std(ddof=1)), 1)])
        # OE separate and shared control
        oe = [oe_halves(*C1[p][:2]) for p in ps]
        a = np.array([[np.mean(h[x][0]) - np.mean(h[x][1]) for x in (0, 1)] for h in oe])
        pcfull = np.array([np.mean([t[2] for t in C1[p][1]]) for p in ps])
        ash = np.array([[np.mean(h[x][0]) for x in (0, 1)] for h in oe]) - pcfull[:, None]
        res["OE"] = single(cor(a[:, 0], a[:, 1]), boot_r(a[:, 0], a[:, 1], idx), rt_b, rt_pt)
        res["OE_shared_control"] = single(cor(ash[:, 0], ash[:, 1]), boot_r(ash[:, 0], ash[:, 1], idx), rt_b, rt_pt)
        res["OE_shared_minus_separate_r"] = dict(diff=round(cor(ash[:, 0], ash[:, 1]) - cor(a[:, 0], a[:, 1]), 4),
                                                 CI=q(boot_r(ash[:, 0], ash[:, 1], idx) - boot_r(a[:, 0], a[:, 1], idx)))
        # equal-precision control
        r_eq = np.random.default_rng(777)
        sh_pt, dj_pt, sh_b, dj_b = [], [], [], []
        for s in range(NEQP):
            P1, P2 = [], []
            for p in ps:
                pcs = np.array([t[2] for t in C1[p][1]]); perm = r_eq.permutation(len(pcs))
                P1.append(pcs[perm[0::2]].mean()); P2.append(pcs[perm[1::2]].mean())
            P1, P2 = np.array(P1), np.array(P2)
            pem = np.array([[np.mean(h[x][0]) for x in (0, 1)] for h in oe])
            for (u, v) in ((P1, P2), (P2, P1)):
                shA = pem - u[:, None]; dj = np.column_stack([pem[:, 0] - u, pem[:, 1] - v])
                sh_pt.append(cor(shA[:, 0], shA[:, 1])); dj_pt.append(cor(dj[:, 0], dj[:, 1]))
                sh_b.append(boot_r(shA[:, 0], shA[:, 1], idx)); dj_b.append(boot_r(dj[:, 0], dj[:, 1], idx))
        sh_pt, dj_pt = np.array(sh_pt), np.array(dj_pt)
        res["EQP"] = dict(r_shared=round(float(sh_pt.mean()), 4), r_separate=round(float(dj_pt.mean()), 4),
                          diff=round(float((sh_pt - dj_pt).mean()), 4), diff_CI=q(np.mean(sh_b, 0) - np.mean(dj_b, 0)),
                          share_positive=round(float(((sh_pt - dj_pt) > 0).mean()), 3))
        # stratified random splits
        H = np.empty((NRND, n, 2))
        for k, p in enumerate(ps):
            pe, pc = C1[p][0], C1[p][1]
            kpe = [t[1] if task in ("Stroop", "Flanker") else "x" for t in pe]
            kpc = [t[1] if task in ("Stroop", "Flanker") else "x" for t in pc]
            Ape = strat_assign(kpe, rng, NRND); Apc = strat_assign(kpc, rng, NRND)
            rpe = np.array([t[2] for t in pe]); rpc = np.array([t[2] for t in pc])
            for x in (0, 1):
                mpe = ((Ape == x) * rpe).sum(1) / (Ape == x).sum(1)
                mpc = ((Apc == x) * rpc).sum(1) / (Apc == x).sum(1)
                H[:, k, x] = mpe - mpc
        r_pt = np.array([cor(H[s, :, 0], H[s, :, 1]) for s in range(NRND)])
        res["RND"] = summarize_split(r_pt, boot_r_splits(H, idx), rt_b, rt_pt, n)
        # block split
        rs, bs = [], []
        for asg in L.A32:
            hh = np.array([[np.mean(h[x][0]) - np.mean(h[x][1]) for x in (0, 1)] for h in (blk_halves(*C1[p][:2], asg) for p in ps)])
            rs.append(cor(hh[:, 0], hh[:, 1])); bs.append(boot_r(hh[:, 0], hh[:, 1], idx))
        rm = float(np.mean(rs)); bm = np.mean(bs, 0)
        res["BLK"] = dict(r_mean=round(rm, 4), r_range=[round(min(rs), 4), round(max(rs), 4)], **single(rm, bm, rt_b, rt_pt))
        res["_boot"] = dict(rt=rt_b, OE=sb(boot_r(a[:, 0], a[:, 1], idx)), BLK=sb(bm))
        res["_idx"] = idx
        return res

    def rob_block(ps):
        n = len(ps); idx = rng.integers(0, n, (NBOOT, n))
        s1 = np.array([np.mean([t[2] for t in C1[p][2]]) for p in ps]); s2 = np.array([np.mean([t[2] for t in C2[p][2]]) for p in ps])
        rt_pt = cor(s1, s2); rt_b = boot_r(s1, s2, idx)
        res = dict(n=n, retest_r=round(rt_pt, 4), retest_r_CI=q(rt_b), retest_spearman=round(float(spearmanr(s1, s2)[0]), 4),
                   mean_ms=[round(1000 * float(s1.mean()), 1), round(1000 * float(s2.mean()), 1)])
        a = np.array([[np.mean(h[x]) for x in (0, 1)] for h in (rob_halves_oe(C1[p][2]) for p in ps)])
        res["OE"] = single(cor(a[:, 0], a[:, 1]), boot_r(a[:, 0], a[:, 1], idx), rt_b, rt_pt)
        H = np.empty((NRND, n, 2))
        for k, p in enumerate(ps):
            rb = C1[p][2]; A = strat_assign([t[1] if task in ("Stroop", "Flanker") else "x" for t in rb], rng, NRND)
            v = np.array([t[2] for t in rb])
            for x in (0, 1):
                H[:, k, x] = ((A == x) * v).sum(1) / np.maximum((A == x).sum(1), 1)
        r_pt = np.array([cor(H[s, :, 0], H[s, :, 1]) for s in range(NRND)])
        res["RND"] = summarize_split(r_pt, boot_r_splits(H, idx), rt_b, rt_pt, n)
        rs, bs = [], []
        for asg in L.A32:
            hh = np.array([[np.mean(h[x]) for x in (0, 1)] for h in (rob_halves_blk(C1[p][2], asg) for p in ps)])
            rs.append(cor(hh[:, 0], hh[:, 1])); bs.append(boot_r(hh[:, 0], hh[:, 1], idx))
        res["BLK"] = single(float(np.mean(rs)), np.mean(bs, 0), rt_b, rt_pt)
        # reuse of raw trials between sequences (C E C E C: the middle trial is post-error of one error
        # and pre-error of the next); audited under the odd/even split and the random splits
        def shared_trials(rb):
            use = {}
            for k, t in enumerate(rb):
                for j in (t[4], t[5]): use.setdefault(j, []).append(k)
            return {j: ks for j, ks in use.items() if len(ks) > 1}
        n_seq = sum(len(C1[p][2]) for p in ps); n_shared_seq = 0; rows_cross_oe = 0; rows_all = 0; persons_cross_oe = 0
        cross_rnd = []
        for k, p in enumerate(ps):
            rb = C1[p][2]; sh = shared_trials(rb)
            n_shared_seq += len({kk for ks in sh.values() for kk in ks})
            rows_all += 2 * len(rb)
            cr = sum(len(ks) for j, ks in sh.items() if len({kk % 2 for kk in ks}) > 1)
            rows_cross_oe += cr; persons_cross_oe += cr > 0
        res["reuse"] = dict(sequences=n_seq, sequences_sharing_a_trial=n_shared_seq,
                            share_sequences_sharing=round(n_shared_seq / n_seq, 4),
                            S_oddeven=round(rows_cross_oe / rows_all, 4), persons_with_cross_half_reuse_oddeven=int(persons_cross_oe))
        # sensitivity: keep the earlier of two sequences that share a trial (non-overlapping sequences), both sessions
        def nonover(rb):
            used, keep = set(), []
            for t in rb:
                if t[4] in used or t[5] in used: continue
                keep.append(t); used.update((t[4], t[5]))
            return keep
        N1 = {p: nonover(C1[p][2]) for p in ps}; N2 = {p: nonover(C2[p][2]) for p in ps}
        ok = [p for p in ps if len(N1[p]) >= MIN_PE_SESSION and len(N2[p]) >= MIN_PE_SESSION and
              all(len(h[x]) >= MIN_HALF for h in [rob_halves_oe(N1[p])] + [rob_halves_blk(N1[p], a_) for a_ in L.A32] for x in (0, 1))]
        m = len(ok); idx2 = rng.integers(0, m, (NBOOT, m))
        u1 = np.array([np.mean([t[2] for t in N1[p]]) for p in ok]); u2 = np.array([np.mean([t[2] for t in N2[p]]) for p in ok])
        rt2 = cor(u1, u2); rt2_b = boot_r(u1, u2, idx2)
        H2 = np.empty((NRND, m, 2))
        for k, p in enumerate(ok):
            rb = N1[p]; A = strat_assign([t[1] if task in ("Stroop", "Flanker") else "x" for t in rb], rng, NRND)
            v = np.array([t[2] for t in rb])
            for x in (0, 1):
                H2[:, k, x] = ((A == x) * v).sum(1) / np.maximum((A == x).sum(1), 1)
        r2 = np.array([cor(H2[s_, :, 0], H2[s_, :, 1]) for s_ in range(NRND)])
        res["nonoverlapping"] = dict(n=m, retest_r=round(rt2, 4), retest_r_CI=q(rt2_b),
                                     RND=summarize_split(r2, boot_r_splits(H2, idx2), rt2_b, rt2, m))
        return res

    def five_block(ps, idx):
        n = len(ps)
        R1 = {p: five_rows(D1[p], *X1[p]) for p in ps}
        def sess_score(D, X, p, once):
            R = five_rows(D[p], *X[p]); return np.mean([x for k, b, j, x, f in R if f or not once])
        res = {}
        for once in (False, True):
            lab = "once" if once else "allrows"
            s1 = np.array([sess_score(D1, X1, p, once) for p in ps]); s2 = np.array([sess_score(D2, X2, p, once) for p in ps])
            rt_pt = cor(s1, s2); rt_b = boot_r(s1, s2, idx)
            # odd/even episodes
            a = np.array([[np.mean(v) for v in five_halves(R1[p], [k % 2 for k in range(len(X1[p][1]))], once)] for p in ps])
            blk = dict(retest_r=round(rt_pt, 4), retest_r_CI=q(rt_b),
                       OE=single(cor(a[:, 0], a[:, 1]), boot_r(a[:, 0], a[:, 1], idx), rt_b, rt_pt))
            # random episode splits
            H = np.empty((NRND, n, 2))
            for k, p in enumerate(ps):
                ne = len(X1[p][1]); A = strat_assign(["x"] * ne, rng, NRND)
                R = R1[p]; ep = np.array([t[0] for t in R]); val = np.array([t[3] for t in R]); fst = np.array([t[4] for t in R])
                keep = fst if once else np.ones(len(R), bool)
                hv = A[:, ep]  # [nsplit, rows]
                for x in (0, 1):
                    m = (hv == x) & keep[None, :]
                    H[:, k, x] = (m * val).sum(1) / np.maximum(m.sum(1), 1)
            r_pt = np.array([cor(H[s, :, 0], H[s, :, 1]) for s in range(NRND)])
            blk["RND"] = summarize_split(r_pt, boot_r_splits(H, idx), rt_b, rt_pt, n)
            if once:  # block split (each trial once, registered baselines), as lab_retest.py
                rs, bs = [], []
                for asg in L.A32:
                    hh = np.array([[np.mean(v) for v in five_halves(R1[p], [0 if X1[p][1][k][1] in asg[0] else 1 for k in range(len(X1[p][1]))], True)] for p in ps])
                    rs.append(cor(hh[:, 0], hh[:, 1])); bs.append(boot_r(hh[:, 0], hh[:, 1], idx))
                blk["BLK"] = dict(r_range=[round(min(rs), 4), round(max(rs), 4)], **single(float(np.mean(rs)), np.mean(bs, 0), rt_b, rt_pt))
                blk["_boot"] = dict(rt=rt_b, BLK=sb(np.mean(bs, 0)), OE=sb(boot_r(a[:, 0], a[:, 1], idx)))
            res[lab] = blk
        return res

    prim = pes_block(common, "common")
    five = five_block(common, prim["_idx"])
    # gap comparison on the same resamples: (SB_block - retest) five-trial (once) minus conventional
    g5 = five["once"]["_boot"]["BLK"] - five["once"]["_boot"]["rt"]; gc = prim["_boot"]["BLK"] - prim["_boot"]["rt"]
    g5o = five["once"]["_boot"]["OE"] - five["once"]["_boot"]["rt"]; gco = prim["_boot"]["OE"] - prim["_boot"]["rt"]
    out["gap_five_minus_conv"] = dict(
        OE=dict(diff=round((five["once"]["OE"]["SB_minus_retest"]) - prim["OE"]["SB_minus_retest"], 4), CI=q(g5o - gco)),
        BLK=dict(diff=round(five["once"]["BLK"]["SB_minus_retest"] - prim["BLK"]["SB_minus_retest"], 4), CI=q(g5 - gc)))
    for d in (prim, five["once"]):
        d.pop("_boot", None)
    prim.pop("_idx", None)
    out["PES_common"] = prim
    out["FIVE_common"] = five
    sens = pes_block(elig, "eligible"); sens.pop("_boot", None); sens.pop("_idx", None)
    out["PES_all_eligible"] = sens
    out["ROB_common"] = rob_block(common_rob) if len(common_rob) >= 30 else dict(n=len(common_rob), note="fewer than 30 persons; not estimated")
    return out


if __name__ == "__main__":
    OUT = sys.argv[1]; counts = "--counts" in sys.argv
    res = {}
    for task in L.TASKS:
        res[task] = run(task, counts)
        print(task, json.dumps(res[task] if counts else res[task]["counts"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
