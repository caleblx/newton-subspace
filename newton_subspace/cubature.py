# Optimized cubature (An, Kim & James 2008): the reduced elastic force sum_e g_e(q) is replaced
# by a sparse non-negative sum_{e in S} w_e g_e(q), fitted by greedy selection + NNLS.

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import nnls

from .fem import TetMesh


def _stable_nh_stress(F: np.ndarray, mu: np.ndarray, lam_nh: np.ndarray) -> np.ndarray:
    alpha = 1.0 + mu / lam_nh
    J = np.linalg.det(F)
    cof = np.stack(
        [np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]), np.cross(F[:, :, 0], F[:, :, 1])],
        axis=2,
    )
    return mu[:, None, None] * F + (lam_nh * (J - alpha))[:, None, None] * cof


# g_e(q_t) = V_e B_e^T P(F_e(q_t)) for every pose t and element e, shape (T, r, E)
def element_reduced_forces(mesh: TetMesh, U: np.ndarray, Q: np.ndarray) -> np.ndarray:
    E, r = mesh.n_tets, U.shape[1]
    G = mesh.element_G()
    dofs = (3 * mesh.tets[:, :, None] + np.arange(3)).reshape(E, 12)
    B = np.einsum("eab,ebj->eaj", G, U[dofs])  # (E, 9, r)
    F0 = np.einsum("eab,eb->ea", G, mesh.rest.reshape(-1)[dofs])  # (E, 9)
    lam_nh = mesh.lam + mesh.mu
    out = np.empty((Q.shape[0], r, E))
    for t, q in enumerate(Q):
        F = (F0 + B @ q).reshape(E, 3, 3).transpose(0, 2, 1)  # column-major vec(F)
        P = _stable_nh_stress(F, mesh.mu, lam_nh)
        p = P.transpose(0, 2, 1).reshape(E, 9) * mesh.volume[:, None]
        out[t] = np.einsum("ear,ea->re", B, p)
    return out


def fit_cubature(mesh: TetMesh, U: np.ndarray, Q: np.ndarray, *, max_points: int = 200, tol: float = 0.01,
                 verbose: bool = False):
    """Greedy NNLS cubature on (T, r) training poses Q. Weights multiply the rest volume."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    A3 = element_reduced_forces(mesh, U, Q)
    # normalize each pose so large and small deformations count equally
    b3 = A3.sum(axis=2, keepdims=True)
    scale = 1.0 / np.maximum(np.linalg.norm(b3, axis=1, keepdims=True), 1e-30)
    A = (A3 * scale).reshape(-1, mesh.n_tets)
    b = (b3 * scale).reshape(-1)
    col_norm = np.linalg.norm(A, axis=0) + 1e-30
    bnorm = np.linalg.norm(b)

    selected: list[int] = []
    w = np.zeros(0)
    res = b.copy()
    history = []
    while len(selected) < max_points:
        # score a random pool of 2000 candidates per step, not all elements
        pool = np.setdiff1d(rng.choice(mesh.n_tets, size=min(2000, mesh.n_tets), replace=False), selected)
        if pool.size == 0:
            break
        score = (A[:, pool].T @ res) / col_norm[pool]
        selected.append(int(pool[np.argmax(score)]))
        w, _ = nnls(A[:, selected], b, maxiter=50 * len(selected))
        res = b - A[:, selected] @ w
        rel = np.linalg.norm(res) / bnorm
        history.append(rel)
        if verbose and len(selected) % 25 == 0:
            print(f"  cubature {len(selected)} points, rel residual {rel:.4f}")
        if rel < tol:
            break
    keep = w > 0
    info = {"residual": history[-1] if history else 1.0, "history": history, "time": time.perf_counter() - t0}
    return np.array(selected)[keep], w[keep], info
