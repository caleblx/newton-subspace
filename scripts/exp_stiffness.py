# E2: accuracy vs stiffness at r = 30. references use full-order Newton-PCG since VBD
# needs thousands of iterations to converge at the stiff end.

import _common  # noqa: F401
import numpy as np

from newton_subspace.experiment import (BASIS_TYPES, build_basis, evaluate, make_model, reference, save_json,
                                        training_snapshots)
from newton_subspace.fem import TetMesh
from newton_subspace.sim import TEST_SCENARIOS

RES, R = 2, 30
STIFFNESS = [1e5, 3e5, 1e6, 3e6, 1e7]
results = {"res": RES, "r": R, "stiffness": STIFFNESS, "rows": []}
for k in STIFFNESS:
    model = make_model(RES, k)
    mesh = TetMesh.from_model(model)
    snaps = training_snapshots(model, RES, k, solver="pcg")
    refs = {sc.name: reference(model, RES, k, sc, solver="pcg") for sc in TEST_SCENARIOS}
    max_disp = {n: float(np.linalg.norm(ref - mesh.rest, axis=-1).max()) for n, ref in refs.items()}
    row = {"k": k, "max_disp": max_disp, "basis": {}}
    for kind in BASIS_TYPES:
        U, _ = build_basis(kind, mesh, R, snaps)
        row["basis"][kind] = evaluate(model, mesh, U, refs, time_it=False)
        print(f"k={k:.0e}", kind, {n: f"{v:.2%}" for n, v in row["basis"][kind]["errors"].items()}, flush=True)
    results["rows"].append(row)
save_json(results, "stiffness.json")
