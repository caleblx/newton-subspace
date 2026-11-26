# Shared experiment plumbing: cached references, basis construction, evaluation.

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .basis import modal_basis, pod_basis
from .cubature import fit_cubature
from .fem import TetMesh
from .scenes import cantilever
from .sim import TEST_SCENARIOS, TRAIN_SCENARIOS, FullOrderRunner, PCGRunner, ReducedRunner, rel_error

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
FRAMES = 120  # 2 s at 60 Hz
SUBSTEPS = 10

# VBD iterations for a converged reference (2x iterations changes it by < 0.5%)
REF_ITERATIONS = {1: 50, 2: 100, 3: 200, 4: 300, 5: 400, 6: 600}

BASIS_TYPES = ("modal", "modal+grav", "modal+MD", "pod")


def scene_key(res: int, k: float) -> str:
    return f"res{res}_k{k:.0e}"


def make_model(res: int, k: float):
    return cantilever(res=res, k_mu=k, k_lambda=k)


# converged full-order trajectory, cached. solver="pcg" (FullNewtonPCG) is for stiff materials
# where VBD converges too slowly.
def reference(model, res: int, k: float, scenario, solver: str = "vbd") -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    if solver == "pcg":
        path = CACHE / f"{scene_key(res, k)}_{scenario.name}_pcg.npy"
    else:
        it = REF_ITERATIONS[res]
        path = CACHE / f"{scene_key(res, k)}_{scenario.name}_vbd{it}.npy"
    if path.exists():
        return np.load(path).astype(np.float64)
    t0 = time.perf_counter()
    if solver == "pcg":
        traj = PCGRunner(model, scenario, substeps=SUBSTEPS).run(FRAMES)
    else:
        traj = FullOrderRunner(model, scenario, substeps=SUBSTEPS, iterations=it).run(FRAMES)
    np.save(path, traj.astype(np.float32))
    print(f"    [ref] {path.name} in {time.perf_counter() - t0:.1f}s", flush=True)
    return traj.astype(np.float64)


def training_snapshots(model, res, k, solver: str = "vbd") -> np.ndarray:
    return np.concatenate([reference(model, res, k, sc, solver=solver) for sc in TRAIN_SCENARIOS])


def build_basis(kind: str, mesh: TetMesh, r: int, snapshots: np.ndarray | None = None):
    t0 = time.perf_counter()
    if kind == "modal":
        U = modal_basis(mesh, r)
    elif kind == "modal+grav":
        U = modal_basis(mesh, r, gravity_mode=True)
    elif kind == "modal+MD":
        U = modal_basis(mesh, r, gravity_mode=True, modal_derivatives=True)
    elif kind == "pod":
        U = pod_basis(mesh, snapshots, r)
    else:
        raise ValueError(kind)
    return U, time.perf_counter() - t0


def project(mesh: TetMesh, U: np.ndarray, traj: np.ndarray) -> np.ndarray:
    m = mesh.mass_diag()
    D = (traj - mesh.rest).reshape(len(traj), -1)
    return (D * m) @ U


def cubature_for(mesh, U, snapshots, n_points):
    Q = project(mesh, U, snapshots)
    idx = np.random.default_rng(0).choice(len(Q), size=min(120, len(Q)), replace=False)
    ids, w, info = fit_cubature(mesh, U, Q[idx], max_points=n_points, tol=1e-4)
    return (ids, w), info


def evaluate(model, mesh, U, refs: dict, *, cubature=None, newton_iterations=3, time_it=True):
    out = {"errors": {}, "ms_per_frame": None}
    for sc in TEST_SCENARIOS:
        runner = ReducedRunner(model, sc, U, substeps=SUBSTEPS, newton_iterations=newton_iterations, cubature=cubature)
        traj = runner.run(FRAMES)
        out["errors"][sc.name] = rel_error(traj, refs[sc.name], mesh) if np.all(np.isfinite(traj)) else float("nan")
    if time_it:
        out["ms_per_frame"] = 1e3 * runner.time_frames(60)
    return out


def save_json(obj, name):
    path = ROOT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=1, default=conv))
    print(f"wrote {path}")


# cached cubature roll-out for the videos and the web viewer
def reduced_trajectory(model, res, k, kind, scenario, *, r=30, n_points=400):
    path = CACHE / f"{scene_key(res, k)}_{scenario.name}_reduced_{kind}_r{r}_c{n_points}.npy"
    if path.exists():
        return np.load(path).astype(np.float64)
    mesh = TetMesh.from_model(model)
    snaps = training_snapshots(model, res, k)
    U, _ = build_basis(kind, mesh, r, snaps)
    cub_path = CACHE / f"{scene_key(res, k)}_cubature_{kind}_r{r}_c{n_points}.npz"
    if cub_path.exists():
        z = np.load(cub_path)
        cub = (z["ids"], z["w"])
    else:
        cub, _ = cubature_for(mesh, U, snaps, n_points)
        np.savez(cub_path, ids=cub[0], w=cub[1])
    traj = ReducedRunner(model, scenario, U, substeps=SUBSTEPS, newton_iterations=1, cubature=cub).run(FRAMES)
    np.save(path, traj.astype(np.float32))
    return traj.astype(np.float64)
