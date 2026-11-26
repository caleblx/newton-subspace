# data for the live demo (docs/demo.js): voxelized newton asset, modes + modal derivatives basis,
# cubature fit to poses from the reduced solver itself. cpu, ~10 min.
# usage: python scripts/build_demo.py [bunny|bear]

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from newton_subspace.basis import modal_basis  # noqa: E402
from newton_subspace.cubature import fit_cubature  # noqa: E402
from newton_subspace.fem import TetMesh  # noqa: E402

HEIGHT = 0.4  # m
DENSITY = 1000.0
POISSON = 0.4
TARGET_HZ = 2.0  # lowest vibration mode
DAMP_BETA = 0.004  # stiffness-proportional damping, kd = beta * mu
R = 30
H_STEP = 1.0 / 120.0  # 2 substeps per 60 Hz frame, what the page runs
SPRING_K = 3000.0  # N/m, mouse drag spring
OUT = ROOT / "docs" / "data"


# rotation into z-up facing -y (the page camera), voxels along the height, and render-mesh
# 1:4 subdivisions. the bear's triangles are big and flat, so they need one split to bend visibly.
MODELS = {
    "bunny": {"file": "bunny.usd", "rot": [[1, 0, 0], [0, 0, -1], [0, 1, 0]], "cells": 24, "subdiv": 0},
    "bear": {"file": "bear.usd", "rot": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "cells": 18, "subdiv": 1},
}


def subdivide(P, F):
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    mid = len(P) + inv.reshape(3, -1).T  # (T, 3): midpoints of edges 01, 12, 20
    P = np.concatenate([P, 0.5 * (P[uniq[:, 0]] + P[uniq[:, 1]])])
    a, b, c = F.T
    ab, bc, ca = mid.T
    F = np.concatenate([np.stack(t, 1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return P, F


def load_model(cfg):
    from pxr import Usd, UsdGeom

    import newton

    path = Path(newton.__file__).parent / "examples" / "assets" / cfg["file"]
    stage = Usd.Stage.Open(str(path))
    mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    P = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    F = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    P = P @ np.array(cfg["rot"], dtype=np.float64).T
    for _ in range(cfg["subdiv"]):
        P, F = subdivide(P, F)
    # three.js lighting and the ball contacts need ccw winding from outside; the usd is cw
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    if np.einsum("ij,ij->", a, np.cross(b, c)) < 0:
        F = F[:, [0, 2, 1]]
    P *= HEIGHT / (P[:, 2].max() - P[:, 2].min())
    P -= [0.5 * (P[:, 0].min() + P[:, 0].max()), 0.5 * (P[:, 1].min() + P[:, 1].max()), P[:, 2].min()]
    return P, F


def inside_columns(P, F, xs, ys):
    # sorted z of the surface crossings along a vertical ray through each (x, y)
    hits = [[[] for _ in ys] for _ in xs]
    for t in P[F]:
        lo, hi = t[:, :2].min(0), t[:, :2].max(0)
        ix = np.flatnonzero((xs >= lo[0]) & (xs <= hi[0]))
        iy = np.flatnonzero((ys >= lo[1]) & (ys <= hi[1]))
        if ix.size == 0 or iy.size == 0:
            continue
        a, b, c = t
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(det) < 1e-14:
            continue
        X, Y = np.meshgrid(xs[ix], ys[iy], indexing="ij")
        l1 = ((X - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (Y - a[1])) / det
        l2 = ((b[0] - a[0]) * (Y - a[1]) - (X - a[0]) * (b[1] - a[1])) / det
        l0 = 1 - l1 - l2
        ok = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        z = l0 * a[2] + l1 * b[2] + l2 * c[2]
        for i, j in zip(*np.nonzero(ok)):
            hits[ix[i]][iy[j]].append(z[i, j])
    return [[np.sort(np.array(h)) for h in row] for row in hits]


# Kuhn subdivision of a cube into 6 tets; corners indexed by bits (x, y, z)
KUHN = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]


def voxelize(P, F, cells_high):
    h = HEIGHT / cells_high
    lo = P.min(0) - h
    n = np.ceil((P.max(0) + h - lo) / h).astype(int)
    centers = [lo[a] + h * (np.arange(n[a]) + 0.5) for a in range(3)]
    cols = inside_columns(P, F, centers[0], centers[1])
    occ = np.zeros(n, dtype=bool)
    for i in range(n[0]):
        for j in range(n[1]):
            zs = cols[i][j]
            if zs.size:
                occ[i, j] = (np.searchsorted(zs, centers[2]) % 2) == 1
    # every surface vertex has to land in a cell so it can be embedded
    cell = np.floor((P - lo) / h).astype(int)
    occ[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    cells = np.argwhere(occ)

    corner = np.array([[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)])
    gid = {}
    tets = []
    for c in cells:
        ids = []
        for k in range(8):
            key = tuple(c + corner[k])
            if key not in gid:
                gid[key] = len(gid)
            ids.append(gid[key])
        tets.extend([[ids[a] for a in t] for t in KUHN])
    keys = np.array(list(gid.keys()))
    X = lo + h * keys
    tets = np.array(tets, dtype=np.int64)
    d = np.einsum("ei,ei->e", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]),
                  X[tets[:, 3]] - X[tets[:, 0]])
    tets[d < 0] = tets[d < 0][:, [0, 2, 1, 3]]
    return X, tets, h, lo, cells


def make_mesh(X, tets, h, mu):
    D = np.stack([X[tets[:, k]] - X[tets[:, 0]] for k in (1, 2, 3)], axis=2)
    vol = np.linalg.det(D) / 6.0
    mass = np.zeros(len(X))
    np.add.at(mass, tets.ravel(), np.repeat(DENSITY * vol / 4.0, 4))
    lam = 2 * mu * POISSON / (1 - 2 * POISSON)
    E = len(tets)
    free = X[:, 2] > X[:, 2].min() + 0.5 * h  # glue the bottom layer of nodes to the floor
    return TetMesh(rest=X, tets=tets, Dm_inv=np.linalg.inv(D), volume=vol, mu=np.full(E, mu),
                   lam=np.full(E, lam), kd=np.full(E, DAMP_BETA * mu), mass=mass, free=free,
                   gravity=np.array([0.0, 0.0, -9.81]))


def embed(P, X, tets, cells, h, lo):
    # containing tet and barycentric weights of each render vertex
    cell_of = {tuple(c): i for i, c in enumerate(cells)}
    cid = np.floor((P - lo) / h).astype(int)
    tid = np.empty(len(P), dtype=np.int64)
    bary = np.empty((len(P), 4))
    for v, c in enumerate(cid):
        base = 6 * cell_of[tuple(c)]
        best = None
        for t in range(base, base + 6):
            T = X[tets[t]]
            lam = np.linalg.solve(np.stack([T[1] - T[0], T[2] - T[0], T[3] - T[0]], 1), P[v] - T[0])
            b = np.array([1 - lam.sum(), *lam])
            if best is None or b.min() > best[1].min():
                best = (t, b)
        tid[v], bary[v] = best
    return tid, bary


# numpy port of the reduced solver, same energy as newton_subspace/solver.py
def cof(F):
    return np.stack([np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]),
                     np.cross(F[:, :, 0], F[:, :, 1])], axis=2)


def green2(H):
    Ht = H.transpose(0, 2, 1)
    return H + Ht + Ht @ H


def nl_terms(H):
    trH = np.trace(H, axis1=1, axis2=2)
    i2 = 0.5 * (trH**2 - np.einsum("eij,eji->e", H, H))
    return trH, i2 + np.linalg.det(H)


def energy(H, Gp, mu, lam, kd, inv_h):
    trH, nl = nl_terms(H)
    d = trH + nl
    E = green2(H) - Gp
    return (0.5 * mu * np.einsum("eij,eij->e", H, H) + 0.5 * lam * d * d - mu * nl
            + 0.5 * kd * inv_h * np.einsum("eij,eij->e", E, E))


def stress(H, Gp, mu, lam, kd, inv_h):
    I = np.eye(3)
    trH, nl = nl_terms(H)
    d = trH + nl
    cH = cof(H)
    cF = (1 + trH)[:, None, None] * I - H.transpose(0, 2, 1) + cH
    P = mu[:, None, None] * (H + H.transpose(0, 2, 1) - trH[:, None, None] * I - cH) + (lam * d)[:, None, None] * cF
    return P + (2 * kd * inv_h)[:, None, None] * ((I + H) @ (green2(H) - Gp))


def hess_apply(H, dF, mu, lam, kd, inv_h):
    # dF is (E, r, 3, 3)
    F = np.eye(3) + H
    trH, nl = nl_terms(H)
    s = lam * (trH + nl) - mu
    c = cof(F)
    f = [F[:, None, :, k] for k in range(3)]
    dd = [dF[:, :, :, k] for k in range(3)]
    dc = np.stack([np.cross(dd[1], f[2]) + np.cross(f[1], dd[2]),
                   np.cross(dd[2], f[0]) + np.cross(f[2], dd[0]),
                   np.cross(dd[0], f[1]) + np.cross(f[0], dd[1])], axis=3)
    out = mu[:, None, None, None] * dF + (lam[:, None] * np.einsum("eij,erij->er", c, dF))[:, :, None, None] * c[:, None]
    out += s[:, None, None, None] * dc
    Fx = F[:, None]
    out += (2 * kd * inv_h)[:, None, None, None] * (Fx @ (dF.transpose(0, 1, 3, 2) @ Fx + Fx.transpose(0, 1, 3, 2) @ dF))
    return out


UNIT9 = np.zeros((9, 3, 3))
for _k in range(9):
    UNIT9[_k, _k % 3, _k // 3] = 1.0  # column-major vec index 3 j + i


def to_mat(v):
    return v.reshape(-1, 3, 3).transpose(0, 2, 1)


def to_vec(M):
    return M.transpose(0, 2, 1).reshape(len(M), 9)


class Reduced:
    # backward euler in the subspace, one newton iteration per step plus a line search
    def __init__(self, mesh, U, ids=None, w=None):
        E = mesh.n_tets
        ids = np.arange(E) if ids is None else ids
        G = mesh.element_G()[ids]
        dofs = (3 * mesh.tets[ids][:, :, None] + np.arange(3)).reshape(len(ids), 12)
        self.B = np.einsum("eab,ebj->eaj", G, U[dofs])  # (e, 9, r), column-major vec(F - I) = B q
        self.B2 = self.B.reshape(-1, U.shape[1])
        self.wv = mesh.volume[ids] * (np.ones(len(ids)) if w is None else w)
        self.mu, self.lam, self.kd = mesh.mu[ids], mesh.lam[ids] + mesh.mu[ids], mesh.kd[ids]
        self.fg = U.T @ (np.repeat(mesh.mass, 3) * np.tile(mesh.gravity, mesh.n_particles))
        self.r = U.shape[1]
        self.q = np.zeros(self.r)
        self.qd = np.zeros(self.r)

    def Hmat(self, q):
        return to_mat(self.B2 @ q)

    def step(self, h, spring=None):
        # spring: (Uv (3, r), Xv (3,), target (3,), k) or None
        inv_h, inv_h2 = 1 / h, 1 / h**2
        E = len(self.wv)
        y = self.q + h * self.qd + h * h * self.fg
        H = self.Hmat(self.q)
        Gp = green2(H)
        args = (self.mu, self.lam, self.kd, np.full(E, inv_h))
        P = to_vec(stress(H, Gp, *args)) * self.wv[:, None]
        g = self.B2.T @ P.reshape(-1)
        # 9x9 element Hessians, then one big product for the reduced matrix
        Hu = hess_apply(H, np.broadcast_to(UNIT9, (E, 9, 3, 3)), *args)  # (E, 9, 3, 3)
        He = Hu.transpose(0, 1, 3, 2).reshape(E, 9, 9) * self.wv[:, None, None]
        C = np.einsum("eab,ebr->ear", He, self.B).reshape(-1, self.r)
        A = self.B2.T @ C + inv_h2 * np.eye(self.r)
        if spring is not None:
            Uv, Xv, tgt, k = spring
            A = A + k * Uv.T @ Uv
            g = g + k * Uv.T @ (Uv @ self.q + Xv - tgt)
        d = np.linalg.solve(A, g + inv_h2 * (self.q - y))
        Hd = self.Hmat(d)
        best = None
        for a in (1.0, 0.5, 0.25, 0.125, 0.0):
            qa = self.q - a * d
            e = np.sum(self.wv * energy(H - a * Hd, Gp, *args)) + 0.5 * inv_h2 * np.sum((qa - y) ** 2)
            if spring is not None:
                e += 0.5 * k * np.sum((Uv @ qa + Xv - tgt) ** 2)
            if best is None or e < best[0]:
                best = (e, qa)
        q_new = best[1]
        self.qd = (q_new - self.q) * inv_h
        self.q = q_new


def drag_script(U, mesh, rng, n_frames=120):
    # grab a random upper node, pull it somewhere, let go
    top = np.flatnonzero(mesh.free & (mesh.rest[:, 2] > 0.3 * HEIGHT))
    v = int(rng.choice(top))
    Uv = U[3 * v:3 * v + 3]
    pull = rng.normal(size=3)
    pull *= rng.uniform(0.08, 0.2) / np.linalg.norm(pull)
    hold = int(rng.integers(30, 80))
    return v, Uv, pull, hold, n_frames


def run(red, mesh, U, script, substeps=4, record_every=4):
    v, Uv, pull, hold, n_frames = script
    Xv = mesh.rest[v]
    out = []
    for f in range(n_frames):
        ramp = min(1.0, f / 20)
        spring = (Uv, Xv, Xv + ramp * pull, SPRING_K) if f < hold else None
        for _ in range(substeps):
            red.step(1.0 / (60 * substeps), spring)
        if f % record_every == 0:
            out.append(red.q.copy())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="bunny", choices=sorted(MODELS))
    name = ap.parse_args().model
    cfg = MODELS[name]
    t0 = time.perf_counter()
    P, F = load_model(cfg)
    X, tets, h, lo, cells = voxelize(P, F, cfg["cells"])
    print(f"surface {len(P)} verts, {len(F)} tris; tet mesh {len(X)} nodes, {len(tets)} tets ({time.perf_counter() - t0:.1f} s)")

    # pick mu so the lowest mode sits at TARGET_HZ (frequencies scale with sqrt(mu))
    mesh = make_mesh(X, tets, h, mu=1.0)
    _, info = modal_basis(mesh, R, n_modes=1, return_info=True)
    mu = (TARGET_HZ / info["frequencies_hz"][0]) ** 2
    mesh = make_mesh(X, tets, h, mu=mu)
    U, info = modal_basis(mesh, R, gravity_mode=True, modal_derivatives=True, return_info=True)
    print(f"mu {mu:.0f} Pa, {mesh.n_free_dofs} free DOFs, modes (Hz) {np.round(info['frequencies_hz'], 2)}, "
          f"basis {info['time_total']:.1f} s")

    # training poses from the reduced solver with every element. 2 substeps instead of 4 gives the
    # same poses for half the cost
    rng = np.random.default_rng(0)
    full = Reduced(mesh, U)
    Q = []
    for _ in range(4):
        full.q[:], full.qd[:] = 0, 0
        Q.append(run(full, mesh, U, drag_script(U, mesh, rng), substeps=2))
    Q = np.concatenate(Q)
    print(f"{len(Q)} training poses ({time.perf_counter() - t0:.1f} s)")
    ids, w, cinfo = fit_cubature(mesh, U, Q, max_points=300, tol=0.01, verbose=True)
    print(f"cubature {len(ids)} tets, residual {cinfo['residual']:.4f}")

    # cubature check on unseen yanks
    errs = []
    for i in range(2):
        s = drag_script(U, mesh, np.random.default_rng(100 + i))
        a, b = Reduced(mesh, U), Reduced(mesh, U, ids, w)
        qa, qb = run(a, mesh, U, s, substeps=2), run(b, mesh, U, s, substeps=2)
        errs.append(np.linalg.norm(qa - qb) / np.linalg.norm(qa))
    print("cubature trajectory error on unseen yanks:", np.round(errs, 4))

    tid, bary = embed(P, X, tets, cells, h, lo)
    Ur = np.einsum("vk,vkdr->vdr", bary, U.reshape(-1, 3, R)[tets[tid]])  # (V, 3, r)
    red = Reduced(mesh, U, ids, w)
    # page starts from rest under gravity, 10 s of settling
    for _ in range(600 * 2):
        red.step(H_STEP)
    q0 = red.q.copy()
    OUT.mkdir(parents=True, exist_ok=True)
    blobs = {
        "rest": P.astype(np.float32),
        "faces": F.astype(np.uint32),
        "U": Ur.astype(np.float32),
        "B": red.B.astype(np.float32),  # (c, 9, r) column-major vec(F - I) = B q
        "cub_rest": X[tets[ids]].astype(np.float32),  # (c, 4, 3)
        "cub_U": U.reshape(-1, 3, R)[tets[ids]].astype(np.float32),  # (c, 4, 3, r)
        "tet_rest": X.astype(np.float32),  # the whole tet mesh, for the wireframe
    }
    edges = set()
    for t in tets:
        for a in range(4):
            for b in range(a + 1, 4):
                edges.add((min(t[a], t[b]), max(t[a], t[b])))
    blobs["tet_edges"] = np.array(sorted(edges), dtype=np.uint32)
    blobs["tet_U"] = U.reshape(-1, 3, R).astype(np.float32)
    offsets, buf, pos = {}, [], 0
    for k, a in blobs.items():
        b = np.ascontiguousarray(a).tobytes()
        offsets[k] = {"offset": pos, "shape": list(a.shape), "dtype": str(a.dtype)}
        buf.append(b)
        pos += len(b)
        pad = (-pos) % 8
        buf.append(b"\0" * pad)
        pos += pad
    (OUT / f"{name}.bin").write_bytes(b"".join(buf))
    meta = {
        "arrays": offsets, "r": R, "h": H_STEP, "substeps": 2, "spring_k": SPRING_K,
        "weight": red.wv.tolist(), "mu": float(mu), "lam_nh": float(red.lam[0]), "kd": float(red.kd[0]),
        "fg": red.fg.tolist(), "n_tets": int(mesh.n_tets), "n_nodes": int(mesh.n_particles),
        "n_dofs": int(mesh.n_free_dofs), "n_cubature": int(len(ids)), "height": HEIGHT, "name": name,
        "modes_hz": info["frequencies_hz"].tolist(), "cubature_error": [float(e) for e in errs], "q0": q0.tolist(),
        # reduced force of a unit acceleration along x, y, z, for the shake table
        "fa": [np.einsum("n,nr->r", mesh.mass, U.reshape(-1, 3, R)[:, c]).tolist() for c in range(3)],
    }
    (OUT / f"{name}.json").write_text(json.dumps(meta))
    print(f"wrote {OUT / name}.bin ({pos / 1e6:.1f} MB) in {time.perf_counter() - t0:.0f} s")

    # reference trajectory for scripts/check_demo_js.mjs
    s = drag_script(U, mesh, np.random.default_rng(7), n_frames=90)
    v, _, pull, hold, _ = s
    qs = run(Reduced(mesh, U, ids, w), mesh, U, s, record_every=1)
    (ROOT / "results" / "cache" / f"{name}_check.json").write_text(json.dumps(
        {"v": int(v), "pull": pull.tolist(), "hold": int(hold), "q": qs.tolist(), "h": 1 / 240, "substeps": 4}))


if __name__ == "__main__":
    main()
