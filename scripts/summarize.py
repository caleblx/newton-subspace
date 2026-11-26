# headline table from results/resolution.json, written to results/summary.{md,json}

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORCE = ("diag_load", "side_heavy")


def mean_err(e):
    return float(np.mean([e[k] for k in FORCE]))


def fmt_pct(x):
    p = 100 * x
    return f"{p:.2f}%" if p < 1 else f"{p:.1f}%" if p < 10 else f"{p:.0f}%"


def matched_vbd(vbd, target):
    # VBD iterations and ms/frame at error `target`, log-log interpolated
    pts = sorted(vbd, key=lambda p: p["iterations"])
    err = [mean_err(p["errors"]) for p in pts]
    for a, b, ea, eb in zip(pts, pts[1:], err, err[1:]):
        if ea > target >= eb:
            t = (np.log(target) - np.log(ea)) / (np.log(eb) - np.log(ea))
            it = np.exp(np.log(a["iterations"]) + t * (np.log(b["iterations"]) - np.log(a["iterations"])))
            ms = np.exp(np.log(a["ms_per_frame"]) + t * (np.log(b["ms_per_frame"]) - np.log(a["ms_per_frame"])))
            return float(it), float(ms)
    if err[0] <= target:  # even the cheapest setting is accurate enough
        return float(pts[0]["iterations"]), float(pts[0]["ms_per_frame"])
    return None, None  # needs more iterations than measured (the reference itself)


d = json.loads((ROOT / "results" / "resolution.json").read_text())
lines = [
    "| res | particles | tets | full DOFs | VBD 10 it: ms / err | reduced (POD, cubature): ms / err (mean, max) "
    "| VBD matched: iters / ms | speedup vs VBD 10 it | speedup at matched accuracy |",
    "|---|---|---|---|---|---|---|---|---|",
]
rows = []
for row in d["rows"]:
    v10 = next(v for v in row["vbd"] if v["iterations"] == 10)
    red = row["reduced"]["pod"]["cubature"]
    e_red = mean_err(red["errors"])
    e_max = max(red["errors"][k] for k in FORCE)
    it_m, ms_m = matched_vbd(row["vbd"], e_red)
    if it_m is None:
        it_m, ms_m = row["ref_iterations"], row["vbd_ref_ms_per_frame"]
    s10 = v10["ms_per_frame"] / red["ms_per_frame"]
    sm = ms_m / red["ms_per_frame"]
    rows.append({"res": row["res"], "dofs": row["n_free_dofs"], "speedup_default": s10, "speedup_matched": sm,
                 "err_mean": e_red, "err_max": e_max, "ms": red["ms_per_frame"], "vbd_matched_iters": it_m,
                 "vbd_matched_ms": ms_m})
    lines.append(
        f"| {row['res']} | {row['n_particles']:,} | {row['n_tets']:,} | {row['n_free_dofs']:,} "
        f"| {v10['ms_per_frame']:.1f} / {fmt_pct(mean_err(v10['errors']))} "
        f"| {red['ms_per_frame']:.2f} / {100 * e_red:.2f}%, {100 * e_max:.2f}% "
        f"| {it_m:.0f} / {ms_m:.0f} | {s10:.0f}× | {sm:.0f}× |"
    )
md = "\n".join(lines)
print(md)
(ROOT / "results" / "summary.md").write_text(md + "\n")
(ROOT / "results" / "summary.json").write_text(json.dumps(rows, indent=1))
