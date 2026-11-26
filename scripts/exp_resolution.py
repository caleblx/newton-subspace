# E3: speed and accuracy vs mesh resolution, reduced (r = 30) vs Newton SolverVBD
# usage: python scripts/exp_resolution.py [res ...]

import sys
import time

import _common  # noqa: F401
import numpy as np

from newton_subspace.experiment import (FRAMES, REF_ITERATIONS, SUBSTEPS, build_basis, cubature_for, evaluate,
                                        make_model, reference, save_json, training_snapshots)
from newton_subspace.fem import TetMesh
from newton_subspace.sim import TEST_SCENARIOS, FullOrderRunner, rel_error

RESOLUTIONS = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]
K, R, CUB_POINTS = 1e6, 30, 400
VBD_ITERS = [5, 10, 20, 30, 50, 100, 200, 300, 400, 600]

all_rows = []
for res in RESOLUTIONS:
    model = make_model(res, K)
    mesh = TetMesh.from_model(model)
    print(f"=== res {res}: particles {mesh.n_particles}, tets {mesh.n_tets}, free DOFs {mesh.n_free_dofs}", flush=True)
    snaps = training_snapshots(model, res, K)
    refs = {sc.name: reference(model, res, K, sc) for sc in TEST_SCENARIOS}
    row = {"res": res, "n_particles": mesh.n_particles, "n_tets": mesh.n_tets, "n_free_dofs": mesh.n_free_dofs,
           "ref_iterations": REF_ITERATIONS[res], "vbd": [], "reduced": {}}
    for it in [i for i in VBD_ITERS if i < REF_ITERATIONS[res]]:
        errs = {}
        for sc in TEST_SCENARIOS:
            runner = FullOrderRunner(model, sc, substeps=SUBSTEPS, iterations=it)
            errs[sc.name] = rel_error(runner.run(FRAMES), refs[sc.name], mesh)
        ms = 1e3 * runner.time_frames(10 if it >= 100 else 30, repeats=2)
        row["vbd"].append({"iterations": it, "errors": errs, "ms_per_frame": ms})
        print(f"  VBD it={it}: {ms:.2f} ms/frame, err {np.mean(list(errs.values())):.2%}", flush=True)
    if res <= 3:
        # check the VBD reference against full-order Newton-PCG
        pcg = reference(model, res, K, TEST_SCENARIOS[0], solver="pcg")
        row["ref_vs_pcg"] = rel_error(refs[TEST_SCENARIOS[0].name], pcg, mesh)
        print(f"  VBD reference vs Newton-PCG: {row['ref_vs_pcg']:.3%}", flush=True)
    runner = FullOrderRunner(model, TEST_SCENARIOS[0], substeps=SUBSTEPS, iterations=REF_ITERATIONS[res])
    row["vbd_ref_ms_per_frame"] = 1e3 * runner.time_frames(5, repeats=1)
    del runner
    for kind in ("pod", "modal+MD"):
        U, t_basis = build_basis(kind, mesh, R, snaps)
        # 1 newton iteration is as accurate as 3 at this step size (exp_solver_settings.py)
        ex = evaluate(model, mesh, U, refs, newton_iterations=1)
        t0 = time.perf_counter()
        cub, info = cubature_for(mesh, U, snaps, CUB_POINTS)
        t_fit = time.perf_counter() - t0
        cu = evaluate(model, mesh, U, refs, cubature=cub, newton_iterations=1)
        cu3 = evaluate(model, mesh, U, refs, cubature=cub, newton_iterations=3)
        meta = {"points": int(len(cub[0])), "fit_time_s": t_fit, "fit_residual": info["residual"]}
        row["reduced"][kind] = {"basis_time_s": t_basis, "exact": ex, "cubature": {**cu, **meta},
                                "cubature_3it": {**cu3, **meta}}
        for lab, ev in (("exact", ex), ("cubature", cu), ("cubature 3 it", cu3)):
            print(f"  {kind} {lab}: {ev['ms_per_frame']:.3f} ms/frame, errs",
                  {n: f"{v:.2%}" for n, v in ev["errors"].items()}, flush=True)
    all_rows.append(row)
    save_json({"k": K, "r": R, "rows": all_rows}, "resolution.json")
