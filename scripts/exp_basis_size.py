# E1: accuracy vs basis size per basis family, exact integration (no cubature)

import _common  # noqa: F401
import numpy as np

from newton_subspace.experiment import (BASIS_TYPES, build_basis, evaluate, make_model, project, reference,
                                        save_json, training_snapshots)
from newton_subspace.fem import TetMesh
from newton_subspace.sim import TEST_SCENARIOS, rel_error

RES, K = 2, 1e6
SIZES = [5, 10, 15, 20, 30, 45, 60]

model = make_model(RES, K)
mesh = TetMesh.from_model(model)
snaps = training_snapshots(model, RES, K)
refs = {sc.name: reference(model, RES, K, sc) for sc in TEST_SCENARIOS}
results = {"res": RES, "k": K, "n_free_dofs": mesh.n_free_dofs, "sizes": SIZES, "basis": {}}
for kind in BASIS_TYPES:
    rows = []
    for r in SIZES:
        U, t_basis = build_basis(kind, mesh, r, snaps)
        ev = evaluate(model, mesh, U, refs)
        # best achievable error in the subspace: M-orthogonal projection of the reference
        proj = {n: rel_error(mesh.rest + (project(mesh, U, ref) @ U.T).reshape(ref.shape), ref, mesh)
                for n, ref in refs.items()}
        rows.append({"r": U.shape[1], "basis_time_s": t_basis, "projection_errors": proj, **ev})
        print(kind, r, {k: f"{v:.2%}" for k, v in ev["errors"].items()}, f"{ev['ms_per_frame']:.2f} ms", flush=True)
    results["basis"][kind] = rows
save_json(results, "basis_size.json")
