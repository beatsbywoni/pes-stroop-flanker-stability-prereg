#!/usr/bin/env python3
"""Hedge et al. (2019), Experiment 1: counts that use no response time (the RT column, column 4, is discarded on loading).
Usage: python3 -I hedge2019_counts.py <data folder> OUT.json"""
import json, sys, os
import numpy as np
from scipy.io import loadmat
folder, outp = sys.argv[1], sys.argv[2]
EMPH = {1: "speed", 2: "speed", 3: "speed", 4: "accuracy", 5: "accuracy", 6: "accuracy", 7: "standard", 8: "standard", 9: "standard"}
q = lambda v: [float(x) for x in np.quantile(v, [0, .25, .5, .75, 1])]
out = {"note": "counts only; the RT column is discarded on loading"}
per = {}
for task in ("Stroop", "Flanker"):
    for s in (1, 2):
        A = loadmat(os.path.join(folder, f"Exp1S{s}{task}.mat"))["AllData"][:, :3]   # drop RT column
        pid, cond, corr = A[:, 0].astype(int), A[:, 1].astype(int), A[:, 2].astype(int)
        d = {}
        for p in np.unique(pid):
            m = pid == p; c = cond[m]; k = corr[m]
            runs = 1 + int(np.sum(np.diff([EMPH[x] != EMPH[c[0]] for x in c]) != 0)) if len(c) else 0
            seg = [EMPH[c[0]]]
            for x in c[1:]:
                if EMPH[x] != seg[-1]: seg.append(EMPH[x])
            err = 1 - k
            # errors followed by a trial of the same emphasis (scorable as post-error)
            nxt_same = np.r_[[EMPH[a] == EMPH[b] for a, b in zip(c[:-1], c[1:])], False]
            d[int(p)] = dict(trials=int(m.sum()), errors=int(err.sum()), errors_scorable=int((err.astype(bool) & nxt_same).sum()),
                             by_emph={e: dict(trials=int(sum(EMPH[x] == e for x in c)), errors=int(sum(err[i] for i in range(len(c)) if EMPH[c[i]] == e))) for e in ("speed", "accuracy", "standard")},
                             emphasis_sequence=seg)
        per[(task, s)] = d
        out[f"{task}_S{s}"] = dict(n=len(d), trials=q([v["trials"] for v in d.values()]), errors=q([v["errors"] for v in d.values()]),
                                  errors_speed=q([v["by_emph"]["speed"]["errors"] for v in d.values()]), errors_accuracy=q([v["by_emph"]["accuracy"]["errors"] for v in d.values()]),
                                  errors_standard=q([v["by_emph"]["standard"]["errors"] for v in d.values()]),
                                  trials_per_emphasis=q([v["by_emph"]["standard"]["trials"] for v in d.values()]),
                                  sequences=sorted({"-".join(v["emphasis_sequence"]) for v in d.values()})[:6], n_sequences=len({"-".join(v["emphasis_sequence"]) for v in d.values()}))
    both = sorted(set(per[(task, 1)]) & set(per[(task, 2)]))
    for thr in (10, 20):
        out[f"{task}_both_sessions_ge{thr}_errors_all"] = sum(per[(task, 1)][p]["errors_scorable"] >= thr and per[(task, 2)][p]["errors_scorable"] >= thr for p in both)
        out[f"{task}_both_sessions_ge{thr}_errors_standard"] = sum(per[(task, 1)][p]["by_emph"]["standard"]["errors"] >= thr and per[(task, 2)][p]["by_emph"]["standard"]["errors"] >= thr for p in both)
    out[f"{task}_n_both_sessions"] = len(both)
# order checks (condition column only): 12 consecutive runs of 144 trials, one emphasis and 48 trials of each congruency per run
for task in ("Stroop", "Flanker"):
    for s in (1, 2):
        A = loadmat(os.path.join(folder, f"Exp1S{s}{task}.mat"))["AllData"][:, :3]
        pid, cond, corr = A[:, 0].astype(int), A[:, 1].astype(int), A[:, 2].astype(int)
        single, balanced, changes, low_acc = 0, 0, [], []
        for p in np.unique(pid):
            c = cond[pid == p]; k = corr[pid == p]
            single += all(len({EMPH[x] for x in c[i:i + 144]}) == 1 for i in range(0, len(c), 144))
            balanced += all(sorted(np.bincount((c[i:i + 144] - 1) % 3, minlength=3).tolist()) == [48, 48, 48] for i in range(0, len(c), 144))
            changes += [int(np.sum(np.diff(c[i:i + 144]) != 0)) for i in range(0, len(c), 144)]
            if k.mean() < .60: low_acc.append(int(p))
        out[f"{task}_S{s}"].update(runs_single_emphasis=int(single), runs_balanced=int(balanced), condition_changes_per_run=q(changes), accuracy_below_60=low_acc)
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps(out, indent=1))
