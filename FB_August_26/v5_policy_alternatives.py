"""
v5_policy_alternatives.py

The two policy experiments of Section 5.3 (Table: Policy Alternatives), one
command, both panels, resumable.

Panel 1, changing the objective (Appendix G.2.1, objective (19) with
constraints (20)): Absolute Hard maximizing the number of students co-assigned
with a sibling, printed as the Co-assignments row.

Panel 2, removing the initial tie breakers (Appendix G.2.2, formulation (21)):
Absolute Hard with the group order only, under the Rank and the Co-assignments
objectives. The lottery plays no role in the model, so the draws differ only
through alternative optima.

    cd v5
    nohup python v5_policy_alternatives.py > policy.log 2>&1 &

Defaults: Magallanes, 100 draws, 20 cores, 3600 s per solve. Output:
results/v5/policy_alternatives/ with rows.jsonl and table_policy.tex in the
paper's exact two panel layout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import v5_exact as exact
import v5_heuristics as heuristics
import v5_metrics as metrics
from v5_inputs import load_region, resample_lottery

_G: dict = {}

# Row specs: priority, enforcement, objective, tie_breakers, warm source.
# warm="stable" uses the per-draw cached stable matching; warm="rankcache"
# uses the Panel 2 rank optimum, feasible for every no-lottery absolute row.
PANEL1_ALL = [
    ("Co-assignments", dict(priority="absolute", enforcement="hard",
                            objective="together_members", warm="stable")),
]
PANEL1_DEFAULT = "Co-assignments"

PANEL2_ALL = [
    ("Rank", dict(priority="absolute", enforcement="hard",
                  objective="rank", tie_breakers=False, warm=None)),
    ("Co-assignments", dict(priority="absolute", enforcement="hard",
                            objective="together_members", tie_breakers=False,
                            warm="rankcache")),
]
PANEL2_DEFAULT = "Rank,Co-assignments"



def _discover_warm(results_root, region):
    cands = [d for d in glob.glob(os.path.join(results_root, f"{region}_*",
                                               "warm")) if os.path.isdir(d)]
    return max(cands, key=os.path.getmtime) if cands else None


def _init(cfg):
    _G["cfg"] = cfg
    _G["base"] = load_region(cfg["region"], cfg["year"], cfg["r_root"],
                             tie_break=cfg["tie_break"], seed=cfg["seed"],
                             verbose=False)


def _panel2_rank_start(inst, draw, cfg):
    """A feasible point for the no-tie-breaker model, namely the optimum of the
    Rank objective, which has exactly the same feasible set. Removing the
    lotteries leaves the model highly symmetric and the co-assignments
    objective has a far weaker relaxation than rank, so from a cold start
    Gurobi can return no incumbent at all within the time limit. Cached per draw, so the Rank solve
    is paid once whichever order the pool happens to run the tasks in."""
    path = os.path.join(cfg["rank_dir"], f"{draw}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f).get("matching")
        except Exception:
            pass
    mu, _ = exact.solve(inst, "absolute", "hard", objective="rank",
                        tie_breakers=False, time_limit=cfg["tl"],
                        mip_gap=cfg["gap"], threads=1)
    if mu is not None:
        with open(path, "w") as f:
            json.dump({"matching": {k: v for k, v in mu.items()}}, f)
    return mu


def _run_task(task):
    panel, label, draw = task
    cfg = _G["cfg"]
    inst = resample_lottery(_G["base"], seed=cfg["seed"] + draw,
                            tie_break=cfg["tie_break"])
    w = None
    if panel == 1 and cfg["warm_dir"]:
        p = os.path.join(cfg["warm_dir"], f"{draw}.json")
        if os.path.exists(p):
            with open(p) as f:
                w = json.load(f).get("matching")
    kw = dict(PANEL1_ALL if panel == 1 else PANEL2_ALL)[label]
    if kw.get("warm") == "rankcache":
        w = _panel2_rank_start(inst, draw, cfg)
    elif kw.get("warm") != "stable":
        w = None
    t0 = time.perf_counter()
    try:
        mu, ip = exact.solve(inst, kw["priority"], kw["enforcement"],
                             zeta_min=kw.get("zeta_min"),
                             objective=kw["objective"],
                             tie_breakers=kw.get("tie_breakers", True),
                             time_limit=cfg["tl"], mip_gap=cfg["gap"],
                             threads=1, warm_start=w,
                             no_rel_heur_time=cfg.get("norel", 0.0))
    except Exception as e:
        return {"panel": panel, "method": label, "draw": draw, "solved": 0,
                "status": f"error: {e}",
                "runtime": round(time.perf_counter() - t0, 1)}
    if panel == 2 and label == "Rank" and mu is not None:
        path = os.path.join(cfg["rank_dir"], f"{draw}.json")
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({"matching": {k: v for k, v in mu.items()}}, f)
    row = {"panel": panel, "method": label, "draw": draw,
           "solved": int(mu is not None),
           "status": (ip or {}).get("status_str"),
           "sol_count": (ip or {}).get("sol_count"),
           "obj_bound": (ip or {}).get("obj_bound"),
           "runtime": round(time.perf_counter() - t0, 1)}
    if mu is not None:
        row.update(metrics.evaluate(inst, mu, ip))
    return row


def _load(ckpt):
    rows, done = [], set()
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        rows.append(r)
                        done.add((r["panel"], r["method"], r["draw"]))
                    except json.JSONDecodeError:
                        continue
    return rows, done


def _panel_rows(rows, panel, labels):
    out = []
    for label in labels:
        rs = [r for r in rows if r["panel"] == panel and r["method"] == label]
        ok = [r for r in rs if r.get("solved")]
        agg = metrics.aggregate_by_method(ok) if ok else {}
        a = agg.get(label) or (next(iter(agg.values())) if agg else {})
        out.append((label, len(ok), a))
    return out


def _ms(a, key, d=2):
    m, e = a.get(key + "_mean"), a.get(key + "_se")
    f = lambda v: "--" if v is None else f"{v:.{d}f}"
    return f(m), f(e)


def report(rows, out_dir, p1_labels, p2_labels):
    """Two tables. table_policy.tex uses the exact column layout of
    table_main_abs.tex, with the two panels as bands. table_policy_separated
    .tex carries the student-level separated partition for the same rows."""
    panels = ((1, "Panel 1: Changing Objective", p1_labels),
              (2, "Panel 2: Removing Initial Lotteries", p2_labels))

    main = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Policy alternatives, Magallanes 2023, 100 draws, "
            r"MTB-F. Columns as in the main results table. In Panel 2 the "
            r"tie-breakers do not enter the formulation, so all draws "
            r"coincide and the standard errors are zero by construction.}",
            r"\label{tab: policy alternatives}",
            r"\begin{tabular}{l r rr rr rr rr rr}", r"\toprule",
            r" & & \multicolumn{2}{c}{Avg.\ Pref.} & "
            r"\multicolumn{2}{c}{Top Pref.} & \multicolumn{2}{c}{Top-3} & "
            r"\multicolumn{2}{c}{Unassigned} & "
            r"\multicolumn{2}{c}{Together} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
            r"\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
            r" & Solved & Mean & SE & Mean & SE & Mean & SE & Mean & SE & "
            r"Mean & SE \\"]
    sep = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Policy alternatives, separated students. Together "
           r"plus the three columns equals the students applying with a "
           r"sibling on every row.}",
           r"\label{tab: policy alternatives separated}",
           r"\begin{tabular}{l rr rr rr rr}", r"\toprule",
           r" & \multicolumn{2}{c}{Together} & \multicolumn{2}{c}{None} & "
           r"\multicolumn{2}{c}{One} & "
           r"\multicolumn{2}{c}{At least two} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
           r"\cmidrule(lr){8-9}",
           r" & Mean & SE & Mean & SE & Mean & SE & Mean & SE \\"]
    for panel, title, labels in panels:
        main += [r"\midrule",
                 rf"\multicolumn{{12}}{{c}}{{{title}}} \\", r"\midrule"]
        sep += [r"\midrule",
                rf"\multicolumn{{9}}{{c}}{{{title}}} \\", r"\midrule"]
        for label, n, a in _panel_rows(rows, panel, labels):
            ap = _ms(a, "avg_pref", 3)
            cells = [label, str(n), ap[0], ap[1]]
            for k in ("top_pref", "top3", "unassigned", "together"):
                cells += list(_ms(a, k))
            main.append(" & ".join(cells) + r" \\")
            scells = [label]
            for k in ("together", "stu_none", "stu_one", "stu_both"):
                scells += list(_ms(a, k))
            sep.append(" & ".join(scells) + r" \\")
    main += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    sep += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out_dir, "table_policy.tex"), "w") as f:
        f.write("\n".join(main) + "\n")
    with open(os.path.join(out_dir, "table_policy_separated.tex"), "w") as f:
        f.write("\n".join(sep) + "\n")
    print(f"[policy] wrote table_policy.tex and table_policy_separated.tex "
          f"to {out_dir}")
    for panel, title, labels in panels:
        print(f"[policy] {title}")
        for label, n, a in _panel_rows(rows, panel, labels):
            tg = a.get("together_mean")
            g = a.get("mip_gap_mean")
            tgs = f"together {tg:.2f}" if tg is not None else ""
            gs = f"  gap {g:.2e}" if g is not None else ""
            print(f"[policy]   {label:>10}  solved {n:>3}  {tgs}{gs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Magallanes")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--r-root", default="../R")
    ap.add_argument("--results-root",
                    default=os.path.join("..", "results", "v5"))
    ap.add_argument("--tie-break", default="mtbf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--cores", type=int, default=20)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--mip-gap", type=float, default=1e-3)
    ap.add_argument("--panels", default="1,2")
    ap.add_argument("--panel1", default=PANEL1_DEFAULT,
                    help="comma list of Panel 1 rows; available: "
                         f"{', '.join(l for l, _ in PANEL1_ALL)}")
    ap.add_argument("--panel2", default=PANEL2_DEFAULT,
                    help="comma list of Panel 2 rows; available: "
                         f"{', '.join(l for l, _ in PANEL2_ALL)}")
    ap.add_argument("--no-rel-heur", type=float, default=0.0,
                    help="seconds of Gurobi's NoRel heuristic before the "
                         "branch and bound starts. It searches for feasible "
                         "solutions without solving the relaxation, which the "
                         "Panel 2 co-assignments objective can need: with the "
                         "lotteries removed the model is highly symmetric and "
                         "a cold solve may return no incumbent inside an hour.")
    ap.add_argument("--warm-dir", default=None)
    a = ap.parse_args()

    out_dir = os.path.join(a.results_root, "policy_alternatives")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "rows.jsonl")

    warm_dir = a.warm_dir or _discover_warm(a.results_root, a.region)
    print(f"[policy] region {a.region}, warm cache "
          f"{warm_dir if warm_dir else 'none, DA-free start'}")

    panels = [int(p) for p in a.panels.split(",") if p.strip()]
    def _pick(arg, catalogue, which):
        chosen = [m.strip() for m in arg.split(",") if m.strip()]
        known = [l for l, _ in catalogue]
        unknown = [m for m in chosen if m not in known]
        if unknown:
            raise SystemExit(f"unknown Panel {which} row(s) {unknown}; "
                             f"choose from {known}")
        return [(l, kw) for l, kw in catalogue if l in chosen]
    p1 = _pick(a.panel1, PANEL1_ALL, 1)
    p2 = _pick(a.panel2, PANEL2_ALL, 2)
    print(f"[policy] Panel 1 rows: {[l for l, _ in p1]}")
    print(f"[policy] Panel 2 rows: {[l for l, _ in p2]}")
    tasks = []
    rows, done = _load(ckpt)
    for panel in panels:
        for label, _ in (p1 if panel == 1 else p2):
            for d in range(a.draws):
                if (panel, label, d) not in done:
                    tasks.append((panel, label, d))
    print(f"[policy] {len(done)} rows on disk, {len(tasks)} to run; "
          f"{a.cores} workers, {a.time_limit:.0f}s per solve")

    cfg = dict(region=a.region, year=a.year, r_root=a.r_root,
               tie_break=a.tie_break, seed=a.seed, tl=a.time_limit,
               gap=a.mip_gap, warm_dir=warm_dir, norel=a.no_rel_heur,
               rank_dir=os.path.join(out_dir, "panel2_rank"))
    os.makedirs(cfg["rank_dir"], exist_ok=True)
    t0 = time.perf_counter()
    if tasks:
        with Pool(a.cores, initializer=_init, initargs=(cfg,)) as pool, \
                open(ckpt, "a") as f:
            for i, row in enumerate(pool.imap_unordered(_run_task, tasks), 1):
                f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(tasks):
                    print(f"[policy] {i}/{len(tasks)}  "
                          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    rows, _ = _load(ckpt)
    report(rows, out_dir, [l for l, _ in p1], [l for l, _ in p2])


if __name__ == "__main__":
    main()
