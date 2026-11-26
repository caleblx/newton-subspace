# E4: cubature size vs accuracy and speed, POD and modal+MD at r = 30

import sys

import _common  # noqa: F401

from newton_subspace.experiment import (build_basis, cubature_for, evaluate, make_model, reference, save_json,
                                        training_snapshots)
from newton_subspace.fem import TetMesh
from newton_subspace.sim import TEST_SCENARIOS

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
K, R = 1e6, 30
POINTS = [25, 50, 100, 200, 400, 800]

model = make_model(RES, K)
mesh = TetMesh.from_model(model)
snaps = training_snapshots(model, RES, K)
refs = {sc.name: reference(model, RES, K, sc) for sc in TEST_SCENARIOS}
results = {"res": RES, "k": K, "r": R, "n_tets": mesh.n_tets, "basis": {}}
for kind in ("pod", "modal+MD"):
    U, _ = build_basis(kind, mesh, R, snaps)
    exact = evaluate(model, mesh, U, refs)
    rows = [{"points": mesh.n_tets, "exact": True, **exact}]
    print(kind, "exact", {n: f"{v:.2%}" for n, v in exact["errors"].items()}, f"{exact['ms_per_frame']:.3f} ms", flush=True)
    for n in POINTS:
        cub, info = cubature_for(mesh, U, snaps, n)
        ev = evaluate(model, mesh, U, refs, cubature=cub)
        rows.append({"points": int(len(cub[0])), "exact": False, "fit_residual": info["residual"],
                     "fit_time_s": info["time"], **ev})
        print(kind, len(cub[0]), f"fit res {info['residual']:.4f} ({info['time']:.1f}s)",
              {k: f"{v:.2%}" for k, v in ev["errors"].items()}, f"{ev['ms_per_frame']:.3f} ms", flush=True)
    results["basis"][kind] = rows
save_json(results, f"cubature_res{RES}.json")
