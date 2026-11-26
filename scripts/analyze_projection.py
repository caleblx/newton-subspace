# projection error of each basis vs stiffness (best case, no reduced dynamics). cpu only.
# reads results/stiffness.json, writes results/stiffness_projection.json

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import _common  # noqa: F401
import numpy as np

from newton_subspace.experiment import BASIS_TYPES, ROOT, build_basis, make_model, project, reference, save_json
from newton_subspace.fem import TetMesh
from newton_subspace.sim import TEST_SCENARIOS, TRAIN_SCENARIOS, rel_error

d = json.loads((ROOT / "results" / "stiffness.json").read_text())
out = {"res": d["res"], "r": d["r"], "rows": []}
for k in d["stiffness"]:
    model = make_model(d["res"], k)
    mesh = TetMesh.from_model(model)
    snaps = np.concatenate([reference(model, d["res"], k, sc, solver="pcg") for sc in TRAIN_SCENARIOS])
    refs = {sc.name: reference(model, d["res"], k, sc, solver="pcg") for sc in TEST_SCENARIOS}
    row = {"k": k, "basis": {}}
    for kind in BASIS_TYPES:
        U, _ = build_basis(kind, mesh, d["r"], snaps)
        row["basis"][kind] = {n: rel_error(mesh.rest + (project(mesh, U, ref) @ U.T).reshape(ref.shape), ref, mesh)
                              for n, ref in refs.items()}
    print(f"k={k:.0e}", {kind: {n: f"{v:.2%}" for n, v in e.items()} for kind, e in row["basis"].items()}, flush=True)
    out["rows"].append(row)
save_json(out, "stiffness_projection.json")
