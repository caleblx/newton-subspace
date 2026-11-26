# viewer data in docs/data: VBD reference + both r = 30 reduced runs, surface only, uint16
# usage: python scripts/export_playback.py [res]

import json
import sys

import _common  # noqa: F401
import numpy as np

from newton_subspace.experiment import ROOT, make_model, reduced_trajectory, reference
from newton_subspace.fem import TetMesh
from newton_subspace.sim import FPS, TEST_SCENARIOS, rel_error

RES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
K = 1e6
STRIDE = 2  # keep every other frame; the viewer interpolates
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def boundary_faces(tets):
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]])
    _, inv, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv.ravel()] == 1]


model = make_model(RES, K)
mesh = TetMesh.from_model(model)
faces = boundary_faces(mesh.tets)
verts, faces = np.unique(faces, return_inverse=True)
faces = faces.reshape(-1, 3).astype(np.uint32)
faces.tofile(OUT / f"faces_res{RES}.bin")
mesh.rest[verts].astype(np.float32).tofile(OUT / f"rest_res{RES}.bin")

meta = {"res": RES, "n_tets": mesh.n_tets, "n_dofs": mesh.n_free_dofs, "n_verts": len(verts), "n_faces": len(faces),
        "fps": FPS / STRIDE, "runs": ["vbd", "pod", "modal+MD"], "scenarios": {}}
for sc in TEST_SCENARIOS:
    ref = reference(model, RES, K, sc)
    runs = [ref] + [reduced_trajectory(model, RES, K, kind, sc) for kind in ("pod", "modal+MD")]
    errs = [rel_error(t, ref, mesh) for t in runs[1:]]
    print(sc.name, [f"{e:.2%}" for e in errs], flush=True)
    X = np.stack([t[::STRIDE, verts] for t in runs])  # (run, frame, vert, 3)
    lo, hi = X.min(axis=(0, 1, 2)), X.max(axis=(0, 1, 2))
    Q = np.round((X - lo) / (hi - lo) * 65535).astype(np.uint16)
    Q.tofile(OUT / f"{sc.name}_res{RES}.bin")
    meta["scenarios"][sc.name] = {"frames": X.shape[1], "lo": lo.tolist(), "hi": hi.tolist(),
                                  "load_time": sc.load_time, "errors": dict(zip(meta["runs"][1:], errs))}
(OUT / f"playback_res{RES}.json").write_text(json.dumps(meta, indent=1))
print("wrote", OUT)
