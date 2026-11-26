# E5: newton iterations x substeps per frame vs accuracy and speed

import sys

import _common  # noqa: F401
import numpy as np

from newton_subspace.experiment import (FRAMES, build_basis, cubature_for, make_model, reference, save_json,
                                        training_snapshots)
from newton_subspace.fem import TetMesh
from newton_subspace.sim import TEST_SCENARIOS, ReducedRunner, rel_error

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 2
K, R = 1e6, 30
model = make_model(RES, K)
mesh = TetMesh.from_model(model)
snaps = training_snapshots(model, RES, K)
refs = {sc.name: reference(model, RES, K, sc) for sc in TEST_SCENARIOS}
U, _ = build_basis("pod", mesh, R, snaps)
cub, _ = cubature_for(mesh, U, snaps, 400)
rows = []
for substeps in (1, 2, 5, 10):
    for iters in (1, 2, 3, 5):
        errs = {}
        for sc in TEST_SCENARIOS:
            rr = ReducedRunner(model, sc, U, substeps=substeps, newton_iterations=iters, cubature=cub)
            tr = rr.run(FRAMES)
            errs[sc.name] = rel_error(tr, refs[sc.name], mesh) if np.all(np.isfinite(tr)) else float("nan")
        ms = 1e3 * rr.time_frames(60)
        rows.append({"substeps": substeps, "newton_iterations": iters, "errors": errs, "ms_per_frame": ms})
        print(f"substeps={substeps} iters={iters}: {ms:.3f} ms/frame", {k: f"{v:.2%}" for k, v in errs.items()},
              flush=True)
save_json({"res": RES, "k": K, "r": R, "cubature_points": int(len(cub[0])), "rows": rows},
          f"solver_settings_res{RES}.json")
