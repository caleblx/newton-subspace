# figures (png + svg) from results/*.json into docs/figures

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#ffffff"  # page background
TEXT = "#1f2328"
TEXT2 = "#59636e"
GRID = "#e6e8eb"
# one color per basis, same in every figure
SLOT = ["#d1462f", "#2266aa", "#8a9099", "#2b2f36"]

BASIS_STYLE = {  # basis -> (label, color, marker)
    "pod": ("POD (randomized SVD)", SLOT[0], "o"),
    "modal+MD": ("Modes + modal derivatives", SLOT[1], "s"),
    "modal": ("Linear modes", SLOT[2], "^"),
    "modal+grav": ("Linear modes + gravity", SLOT[3], "D"),
}
FORCE_TESTS = ("diag_load", "side_heavy")

# Inter to match the page, if it's installed; Nimbus Sans covers the Greek
for f in sorted((Path.home() / ".local/share/fonts/inter").glob("*.ttf")):
    font_manager.fontManager.addfont(str(f))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "medium", "axes.titlepad": 10, "font.family": ["Inter", "Nimbus Sans", "DejaVu Sans"],
    "legend.fontsize": 9.5, "axes.labelsize": 11, "svg.fonttype": "path", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})  # fmt: skip


def load(name):
    return json.loads((RES / name).read_text())


def mean_force_err(errors):
    return 100 * float(np.mean([errors[k] for k in FORCE_TESTS]))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def top_legend(fig, handles, labels):
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.06, 1.0), ncol=len(labels), handlelength=2.2,
               columnspacing=1.6, borderaxespad=0)


def line(ax, x, y, kind):
    lab, col, mk = BASIS_STYLE[kind]
    ax.plot(x, y, color=col, marker=mk, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)


def fig_basis_size():
    d = load("basis_size.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (title, keys) in zip(axes, [("push loads", FORCE_TESTS), ("twist", ("twist",))]):
        for kind in ("modal", "modal+MD", "pod"):  # modal+grav coincides with modal
            rows = d["basis"][kind]
            r = [row["r"] for row in rows]
            e = [100 * np.mean([row["errors"][k] for k in keys]) for row in rows]
            line(ax, r, e, kind)
        rows = d["basis"]["pod"]
        ax.plot([row["r"] for row in rows], [100 * np.mean([row["projection_errors"][k] for k in keys]) for row in rows],
                color=SLOT[0], linewidth=1, alpha=0.6, label="POD, projection of the reference")
        ax.set_yscale("log")
        ax.set_xlabel("basis size r")
        ax.set_title(title)
        ax.axvline(30, color=GRID, linewidth=1)
    axes[0].set_ylabel("error on unseen loads (%)")
    fig.tight_layout()
    top_legend(fig, *axes[0].get_legend_handles_labels())
    save(fig, "basis_size")


def fig_stiffness():
    d = load("stiffness.json")
    proj = load("stiffness_projection.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ks = [row["k"] for row in d["rows"]]
    for ax, (title, keys) in zip(axes, [("push loads", FORCE_TESTS), ("twist", ("twist",))]):
        for kind in ("modal", "modal+MD", "pod"):  # modal+grav coincides with modal
            e = [100 * np.mean([row["basis"][kind]["errors"][k] for k in keys]) for row in d["rows"]]
            line(ax, ks, e, kind)
            pe = [100 * np.mean([row["basis"][kind][k] for k in keys]) for row in proj["rows"]]
            ax.plot(ks, pe, color=BASIS_STYLE[kind][1], linewidth=1, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Lamé parameters μ = λ (Pa)")
        ax.set_title(title)
    axes[0].set_ylabel("error on unseen loads (%)")
    h, _ = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=TEXT2, linewidth=1, alpha=0.7))
    labs = [BASIS_STYLE[k][0] for k in ("modal", "modal+MD", "pod")] + ["projection of the reference"]
    # tip displacement on a secondary axis
    disp = [max(row["max_disp"][k] for k in FORCE_TESTS) for row in d["rows"]]
    for ax in axes:
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(ks, [f"{v:.2f} m" for v in disp], fontsize=8.5, color=TEXT2)
        sec.tick_params(length=0)
        sec.set_xlabel("largest tip displacement", fontsize=9, color=TEXT2, labelpad=6)
    fig.tight_layout()
    top_legend(fig, h, labs)
    save(fig, "stiffness")


def fig_cubature():
    d = load("cubature_res4.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for kind in ("modal+MD", "pod"):
        rows = [row for row in d["basis"][kind] if not row["exact"]]
        exact = [row for row in d["basis"][kind] if row["exact"]][0]
        n = [row["points"] for row in rows]
        line(axes[0], n, [mean_force_err(row["errors"]) for row in rows], kind)
        axes[0].axhline(mean_force_err(exact["errors"]), color=BASIS_STYLE[kind][1], linewidth=1, alpha=0.6)
        line(axes[1], n, [row["ms_per_frame"] for row in rows], kind)
        axes[1].axhline(exact["ms_per_frame"], color=BASIS_STYLE[kind][1], linewidth=1, alpha=0.6)
    exact_ms = [row for row in d["basis"]["pod"] if row["exact"]][0]["ms_per_frame"]
    axes[1].annotate(f"all {d['n_tets']:,} tets: {exact_ms:.1f} ms", (30, exact_ms),
                     xytext=(0, -14), textcoords="offset points", fontsize=9, color=TEXT2)
    h, labs = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=TEXT2, linewidth=1, alpha=0.6))
    axes[0].legend(h, labs + ["all tets"], loc="upper right")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
        ax.minorticks_off()
        ax.set_xlabel("cubature elements")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("error on unseen loads (%)")
    axes[0].set_title("error")
    axes[1].set_ylabel("ms per frame")
    axes[1].set_title("cost")
    axes[1].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "cubature")


def fig_resolution():
    d = load("resolution.json")
    rows = d["rows"]
    dofs = [row["n_free_dofs"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    series = {
        "Newton VBD, 10 iterations (default)": ([next(v["ms_per_frame"] for v in row["vbd"] if v["iterations"] == 10)
                                                 for row in rows], SLOT[2], "^"),
        "Newton VBD, converged reference": ([row["vbd_ref_ms_per_frame"] for row in rows], SLOT[3], "D"),
        "Reduced (POD), all tets": ([row["reduced"]["pod"]["exact"]["ms_per_frame"] for row in rows], SLOT[1], "s"),
        "Reduced (POD), cubature": ([row["reduced"]["pod"]["cubature"]["ms_per_frame"] for row in rows], SLOT[0], "o"),
    }
    for lab, (y, col, mk) in series.items():
        ax.plot(dofs, y, color=col, marker=mk, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("full-order DOFs")
    ax.set_ylabel("ms per frame")
    ax.set_title("cost")
    ax = axes[1]
    ax.plot(dofs, [100 * np.mean([v["errors"][k] for k in FORCE_TESTS])
                   for row in rows for v in row["vbd"] if v["iterations"] == 10],
            color=SLOT[2], marker="^", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="Newton VBD, 10 iterations")
    for lab, key, col, mk in (("Reduced (POD), all tets", "exact", SLOT[1], "s"),
                                ("Reduced (POD), cubature", "cubature", SLOT[0], "o")):
        ax.plot(dofs, [mean_force_err(row["reduced"]["pod"][key]["errors"]) for row in rows], color=col, marker=mk,
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("full-order DOFs")
    ax.set_ylabel("error on unseen loads (%)")
    ax.set_title("error")
    fig.tight_layout()
    h, labs = axes[0].get_legend_handles_labels()
    fig.legend(h, labs, loc="lower left", bbox_to_anchor=(0.06, 1.0), ncol=2, columnspacing=1.6, borderaxespad=0)
    save(fig, "resolution")


def fig_pareto():
    # accuracy vs cost at the finest resolution
    row = load("resolution.json")["rows"][-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    v = row["vbd"]
    x = [p["ms_per_frame"] for p in v]
    y = [mean_force_err(p["errors"]) for p in v]
    ax.plot(x, y, color=SLOT[2], marker="^", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="Newton VBD, 5 to 300 iterations")
    for p, xi, yi in zip(v, x, y):
        ax.annotate(f"{p['iterations']} it", (xi, yi), xytext=(6, -14) if p["iterations"] == 30 else (6, 4),
                    textcoords="offset points", fontsize=8.5, color=TEXT2)
    for kind in ("pod", "modal+MD"):
        for key, mk, name in (("exact", "s", "all tets"), ("cubature", "o", "cubature")):
            ev = row["reduced"][kind][key]
            lab, col, _ = BASIS_STYLE[kind]
            ax.plot([ev["ms_per_frame"]], [mean_force_err(ev["errors"])], color=col, marker=mk, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none",
                    label=f"{lab}, {name}")
    srow = next(r for r in load("summary.json") if r["res"] == row["res"])
    y0 = 100 * srow["err_mean"]
    ax.annotate("", xy=(srow["vbd_matched_ms"], y0), xytext=(srow["ms"] * 1.15, y0),
                arrowprops={"arrowstyle": "<->", "color": TEXT2, "linewidth": 1})
    ax.annotate(f"{srow['speedup_matched']:.0f}x at equal error\n"
                f"(VBD needs ~{srow['vbd_matched_iters']:.0f} iterations)",
                (np.sqrt(srow["ms"] * srow["vbd_matched_ms"]), y0), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9.5, color=TEXT)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ms per frame")
    ax.set_ylabel("error on unseen loads (%)")
    ax.set_title(f"{row['n_tets']:,} tets, {row['n_free_dofs']:,} DOFs")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    save(fig, "pareto")


def fig_solver_settings():
    d = load("solver_settings_res2.json")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    iters = sorted({row["newton_iterations"] for row in d["rows"]})
    for i, it in enumerate(iters):
        rows = sorted([row for row in d["rows"] if row["newton_iterations"] == it], key=lambda r: r["substeps"])
        ax.plot([row["ms_per_frame"] for row in rows], [mean_force_err(row["errors"]) for row in rows], color=SLOT[i],
                marker="osD^"[i], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{it}")
        for row in rows:
            ax.annotate(f"{row['substeps']}", (row["ms_per_frame"], mean_force_err(row["errors"])), xytext=(6, 3),
                        textcoords="offset points", fontsize=8.5, color=TEXT2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ms per frame")
    ax.set_ylabel("error on unseen loads (%)")
    ax.set_title("numbers at points: substeps per frame", fontsize=10, color=TEXT2)
    ax.legend(title="Newton iterations\nper substep", title_fontsize=9.5, alignment="left")
    fig.tight_layout()
    save(fig, "solver_settings")


if __name__ == "__main__":
    fig_basis_size()
    fig_stiffness()
    fig_cubature()
    fig_resolution()
    fig_pareto()
    fig_solver_settings()
