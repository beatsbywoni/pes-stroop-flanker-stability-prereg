#!/usr/bin/env python3
"""
adapter_hedge_gng_fix.py (copy of adapter_hedge.py with go/no-go block reconstruction; set FIX_GNG_BLOCKS=0 to reproduce the original)
adapter_hedge.py: thin adapter for Study 2 (Hedge et al., 2018, to the input contract of the existing estimation core).

Pipeline design carried over: domain knowledge lives in config/hedge2018.yaml;
this adapter only converts raw data to episodes.csv.gz + frames_long.csv.gz (same output
contract as 05_estimation/prep_data.py). Standard library only, deterministic.

Usage:
  python3 adapter_hedge.py --scope exploration   # Study 1 x session 1 x odd pid (exploration subset)
  python3 adapter_hedge.py --scope confirmatory  # all remaining data (only after the registration is frozen)
  python3 adapter_hedge.py --scope all --tasks Stroop Flanker

Output: analysis/_derived/<scope>/episodes.csv.gz, frames_long.csv.gz, adapter_report.json
frames_long contract: [eid, phash, d, dev_rt, dev_acc] (laboratory counterparts of dev_gold/cs/xp)
"""
import os, re, csv, json, gzip, math, hashlib, argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("HEDGE_RAW", os.path.abspath(os.path.join(HERE, "..", "raw", "hedge2018", "RawData")))  # deposit copy: path
FIX_BLOCKS = os.environ.get("FIX_GNG_BLOCKS", "1") == "1"

# ── config mirror (hedge2018.yaml FROZEN v1, as preregistered) ──
FROZEN_YAML_SHA256 = "8b3917b74b8d985000eaf93c6080c9ba18b9a9f66ed41273f0ab6362f1c0732e"
OSF_REGISTRATION = "(preregistration link withheld for anonymous review)"  # anonymized copy: metadata only
CFG = {
    "rt_valid_range_s": (0.15, 3.0),      # FROZEN
    "K": 5,                                # FROZEN (sensitivity K=8)
    "baseline_design": "disjoint",        # FROZEN: structurally disjoint (heldout = sensitivity)
    "baseline_resolution": "medium",      # FROZEN: condition x block pair (fine/coarse = multiverse)
    "min_episodes": 20,
    "split_salt": "v6-heldout::",
    "readme_excluded_all": {"Study2": {25, 34, 38}},
    "readme_excluded_task": {("Study1", "Stopsignal"): {26, 38},
                             ("Study2", "Stroop"): {54},
                             ("Study2", "Stopsignal"): {3, 18, 32}},
}
RT_MS_TASKS = {"Stopsignal"}

def load_session(path, task):
    """One raw-data file to [(block, trial, cond, correct, rt_s), ...] in trial order."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 6 or not r[0].strip():
                continue
            try:
                rt = float(r[5])
                if task in RT_MS_TASKS:
                    rt /= 1000.0
                rows.append((int(r[0]), int(float(r[1])), r[3].strip(), int(float(r[4])), rt))
            except ValueError:
                continue
    # Go/no-go raw files code every block as 1 although each session has five blocks of 120
    # trials (Hedge et al., 2018); block boundaries are recovered from trial-counter resets.
    if FIX_BLOCKS and task == "Gonogo" and rows and len({r[0] for r in rows}) == 1:
        blk, prev, out = 1, None, []
        for (b, tr, cond, cor, rt) in rows:
            if prev is not None and tr < prev:
                blk += 1
            out.append((blk, tr, cond, cor, rt)); prev = tr
        rows = out
    return rows

def heldout(pid, task, sess, trial_idx):
    h = hashlib.sha256(f"{CFG['split_salt']}{pid}|{task}|{sess}|{trial_idx}".encode()).hexdigest()
    return "A" if int(h[-1], 16) % 2 == 0 else "B"   # A = baseline, B = event window

# folder suffix (task) to file-name token (absorbs inconsistent naming in the Hedge raw data)
FILE_TOKEN = {"Stroop": "Stroop", "Flanker": "Flanker", "Stopsignal": "Stop", "Gonogo": "gng"}

def person_files(task):
    folder_by = {}
    tok = FILE_TOKEN.get(task, task)
    for d in sorted(os.listdir(RAW)):
        if not d.endswith("-" + task):
            continue
        study = d.split("-")[0]
        if study == "Study3":
            continue
        for fn in sorted(os.listdir(os.path.join(RAW, d))):
            m = re.match(rf"(Study\d)_P(\d+){tok}([12])\.csv$", fn, re.IGNORECASE)
            if not m:
                continue
            study, pid, sess = m.group(1), int(m.group(2)), int(m.group(3))
            if pid in CFG["readme_excluded_all"].get(study, set()):
                continue
            if pid in CFG["readme_excluded_task"].get((study, task), set()):
                continue
            folder_by[(study, pid, sess)] = os.path.join(RAW, d, fn)
    return folder_by

def in_scope(scope, study, pid, sess):
    if scope == "exploration":
        return study == "Study1" and sess == 1 and pid % 2 == 1
    if scope == "confirmatory":
        return not (study == "Study1" and sess == 1 and pid % 2 == 1)
    return True

def cell_key(cond, blk, res=None):
    res = res or CFG["baseline_resolution"]
    if res == "fine":   return (cond, blk)
    if res == "medium": return (cond, (blk + 1) // 2)   # block pairs {1,2},{3,4},{5,..}
    return (cond,)                                       # coarse

def process(scope, tasks, outdir, design=None, res=None, K=None, burst=False):
    """FROZEN v1 defaults: design=disjoint, res=medium, K=5. Arguments override them for sensitivity runs."""
    design = design or CFG["baseline_design"]; res = res or CFG["baseline_resolution"]
    os.makedirs(outdir, exist_ok=True)
    K = K or CFG["K"]; lo, hi = CFG["rt_valid_range_s"]
    GUARD = 8   # FROZEN: baseline contamination guard, excludes 8 trials after any error
    fe = gzip.open(os.path.join(outdir, "episodes.csv.gz"), "wt", newline="")
    fl = gzip.open(os.path.join(outdir, "frames_long.csv.gz"), "wt", newline="")
    we, wl = csv.writer(fe), csv.writer(fl)
    # episodes: laboratory counterpart of the prep_data.py contract (no respawn/gdiff/pos/patch; task context instead)
    we.writerow(["eid", "phash", "study", "task", "session", "block", "trial",
                 "cond_of_error", "err_rt", "overlap", "K", "n_valid_d"])
    wl.writerow(["eid", "phash", "d", "dev_rt", "dev_acc"])
    eid = 0
    stats = defaultdict(int)
    persons = defaultdict(int)
    for task in tasks:
        files = person_files(task)
        for (study, pid, sess), path in sorted(files.items()):
            if not in_scope(scope, study, pid, sess):
                continue
            phash = f"{study}_P{pid}"
            rows = load_session(path, task)
            n = len(rows)
            err_idx = {i for i, r in enumerate(rows) if r[3] == 0}
            # baseline: post-correct (previous trial correct), outside the GUARD trials after an error, valid RT
            #         (disjoint design: disjoint from the event stream by definition; FROZEN v1 primary)
            contaminated = set()
            for i in err_idx:
                for j in range(i, min(n, i + GUARD + 1)):
                    contaminated.add(j)
            base = defaultdict(list)
            for i, (blk, tr, cond, cor, rt) in enumerate(rows):
                if (cor == 1 and i not in contaminated and i - 1 >= 0 and rows[i-1][3] == 1
                        and lo <= rt <= hi
                        and (design != "heldout" or heldout(pid, task, sess, i) == "A")):
                    base[cell_key(cond, blk, res)].append(rt)
            cell = {}
            for k, v in base.items():
                if len(v) >= 5:
                    m = sum(v) / len(v)
                    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
                    if sd > 0:
                        cell[k] = (m, sd)
            stats[f"{task}_baseline_cells"] += len(cell)
            # burst collapsing (sensitivity): only the first of consecutive errors is an event (skip if the previous trial was an error)
            event_idx = sorted(err_idx)
            if burst:
                event_idx = [i for i in event_idx if (i - 1) not in err_idx]
            # episodes: all error trials (disjoint) or split B (heldout sensitivity), same block, k=1..K
            for i in event_idx:
                if design == "heldout" and heldout(pid, task, sess, i) != "B":
                    continue
                blk_e = rows[i][0]
                devs = []
                for d in range(1, K + 1):
                    j = i + d
                    if j >= n or rows[j][0] != blk_e:
                        devs.append((d, None, None)); continue
                    blk, tr, cond, cor, rt = rows[j]
                    key = cell_key(cond, blk, res)
                    if key not in cell or not (lo <= rt <= hi):
                        devs.append((d, None, cor)); continue
                    m, sd = cell[key]
                    devs.append((d, (rt - m) / sd, cor))
                n_valid = sum(1 for _, z, _ in devs if z is not None)
                if n_valid == 0:
                    stats[f"{task}_dropped_no_valid_frames"] += 1; continue
                # overlap: whether another error occurs within the K-window (covariate not in the registration)
                overlap = int(any((i + d) in err_idx for d in range(1, K + 1)))
                eid += 1
                we.writerow([eid, phash, study, task, sess, rows[i][0], rows[i][1],
                             rows[i][2], round(rows[i][4], 5), overlap, K, n_valid])
                for d, z, cor in devs:
                    if z is None and cor is None:
                        continue
                    wl.writerow([eid, phash, d,
                                 round(z, 5) if z is not None else "",
                                 cor if cor is not None else ""])
                persons[(task, phash, sess)] += 1
    fe.close(); fl.close()
    floor_pass = sum(1 for v in persons.values() if v >= CFG["min_episodes"])
    rep = {"scope": scope, "tasks": tasks, "config_mirror": {k: str(v) for k, v in CFG.items()},
           "n_episodes": eid, "n_person_task_sessions": len(persons),
           "n_pass_floor": floor_pass,
           "pct_pass_floor": round(100 * floor_pass / len(persons), 1) if persons else None,
           "stats": dict(stats)}
    json.dump(rep, open(os.path.join(outdir, "adapter_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="exploration",
                    choices=["exploration", "confirmatory", "all"])
    ap.add_argument("--tasks", nargs="+", default=["Stroop", "Flanker"])
    ap.add_argument("--design", default=None, choices=[None, "disjoint", "heldout"])
    ap.add_argument("--res", default=None, choices=[None, "fine", "medium", "coarse"])
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--burst", action="store_true")
    ap.add_argument("--outdir", default=None, help="output folder for derived files (separate per sensitivity variant)")
    a = ap.parse_args()
    if a.scope != "exploration":
        print("Warning: run the confirmatory/all scopes only after the preregistration is frozen")
    outdir = a.outdir or os.path.join(HERE, "_derived", a.scope)
    process(a.scope, a.tasks, outdir, design=a.design, res=a.res, K=a.K, burst=a.burst)
