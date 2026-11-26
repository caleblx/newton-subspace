# Subspace bases on the full (3N,) coordinate vector. All are mass-orthonormal (U^T M U = I)
# with zero rows on clamped particles.

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .fem import TetMesh


def mass_orthonormalize(U: np.ndarray, m: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    # QR in the sqrt(M)-scaled space, dropping dependent columns
    s = np.sqrt(m)[:, None]
    Q, R = np.linalg.qr(s * U)
    keep = np.abs(np.diag(R)) > tol * np.abs(np.diag(R)).max()
    Q = Q[:, keep]
    out = np.zeros_like(Q)
    nz = m > 0
    out[nz] = Q[nz] / s[nz]
    return out


def modal_basis(
    mesh: TetMesh,
    r: int,
    *,
    n_modes: int | None = None,
    gravity_mode: bool = False,
    modal_derivatives: bool = False,
    return_info: bool = False,
):
    """Lowest vibration modes K u = w^2 M u, optionally with gravity sag and modal derivatives."""
    t0 = time.perf_counter()
    fd = mesh.free_dofs
    Kff = mesh.rest_stiffness()[fd][:, fd].tocsc()
    m = mesh.mass_diag()
    mf = m[fd]

    n_extra = int(gravity_mode)
    if n_modes is None:
        if modal_derivatives:
            # smallest k with k + k(k+1)/2 (+ gravity) >= r
            k = 1
            while k + k * (k + 1) // 2 + n_extra < r:
                k += 1
            n_modes = k
        else:
            n_modes = r - n_extra

    # shift-invert Lanczos. a small negative shift keeps K - sigma M positive definite.
    # The square beam has repeated eigenvalues and ARPACK's random start vector picks a different
    # rotation inside each pair every run (and so different modal derivatives), so fix v0.
    v0 = np.random.default_rng(0).standard_normal(Kff.shape[0])
    lam, phi = spla.eigsh(Kff, k=n_modes, M=sp.diags(mf).tocsc(), sigma=-1.0, which="LM", v0=v0)
    order = np.argsort(lam)
    lam, phi = lam[order], phi[:, order]
    t_eig = time.perf_counter() - t0

    cols = []
    if gravity_mode:
        g = np.tile(mesh.gravity, mesh.n_particles)[fd] * mf
        cols.append(spla.spsolve(Kff, g)[:, None])
    cols.append(phi)
    if modal_derivatives:
        cols.append(_modal_derivatives(mesh, Kff, phi))

    U = np.zeros((3 * mesh.n_particles, sum(c.shape[1] for c in cols)))
    U[fd] = np.concatenate(cols, axis=1)
    U = mass_orthonormalize(U, m)[:, :r]
    info = {
        "eigenvalues": lam,
        "frequencies_hz": np.sqrt(np.maximum(lam, 0.0)) / (2 * np.pi),
        "time_eigsh": t_eig,
        "time_total": time.perf_counter() - t0,
    }
    return (U, info) if return_info else U


# Barbic & James 2005: Psi_ij = -K^{-1} (H : phi_i) phi_j. The third derivative of the energy at
# F = I is taken by central differences of the stress differential, good enough for a basis.
def _modal_derivatives(mesh: TetMesh, Kff, phi: np.ndarray) -> np.ndarray:
    fd = mesh.free_dofs
    G = mesh.element_G()
    E = mesh.n_tets
    dofs = (3 * mesh.tets[:, :, None] + np.arange(3)[None, None, :]).reshape(E, 12)
    k = phi.shape[1]
    Phi = np.zeros((3 * mesh.n_particles, k))
    Phi[fd] = phi
    dF = np.einsum("eab,ebk->eak", G, Phi[dofs])  # (E, 9, k)

    def dP(F, dFv):
        eps = 1e-6
        return (_pk1(mesh, F + eps * dFv) - _pk1(mesh, F - eps * dFv)) / (2 * eps)

    I9 = np.tile(np.eye(3).reshape(-1, order="F"), (E, 1))
    lu = spla.splu(Kff)
    h = 1e-4
    out = []
    for i in range(k):
        for j in range(i, k):
            d = (dP(I9 + h * dF[:, :, i], dF[:, :, j]) - dP(I9 - h * dF[:, :, i], dF[:, :, j])) / (2 * h)
            fe = np.einsum("eab,ea->eb", G, d * mesh.volume[:, None])  # (E, 12)
            f = np.zeros(3 * mesh.n_particles)
            np.add.at(f, dofs.ravel(), fe.ravel())
            out.append(-lu.solve(f[fd]))
    return np.stack(out, axis=1)


# stable NH PK1, column-major vec(F) in, vec(P) out
def _pk1(mesh: TetMesh, f: np.ndarray) -> np.ndarray:
    F = f.reshape(-1, 3, 3).transpose(0, 2, 1)
    J = np.linalg.det(F)
    lam_nh = mesh.lam + mesh.mu
    alpha = 1.0 + mesh.mu / lam_nh
    cof = np.stack(
        [np.cross(F[:, :, 1], F[:, :, 2]), np.cross(F[:, :, 2], F[:, :, 0]), np.cross(F[:, :, 0], F[:, :, 1])],
        axis=2,
    )
    P = mesh.mu[:, None, None] * F + (lam_nh * (J - alpha))[:, None, None] * cof
    return P.transpose(0, 2, 1).reshape(-1, 9)


# Halko, Martinsson & Tropp 2011, Alg. 4.4 (power iterations)
def randomized_svd(A: np.ndarray, k: int, *, oversample: int = 10, n_iter: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    m, n = A.shape
    p = min(n, k + oversample)
    Y = A @ rng.standard_normal((n, p))
    Q, _ = np.linalg.qr(Y)
    for _ in range(n_iter):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], S[:k], Vt[:k]


def pod_basis(mesh: TetMesh, snapshots: np.ndarray, r: int) -> np.ndarray:
    """POD of (S, N, 3) position snapshots, minimizing the mass-weighted projection error."""
    D = (snapshots - mesh.rest[None]).reshape(snapshots.shape[0], -1).T  # (3N, S)
    m = mesh.mass_diag()
    s = np.sqrt(m)[:, None]
    Ul, _, _ = randomized_svd(s * D, r)
    U = np.zeros_like(Ul)
    nz = m > 0
    U[nz] = Ul[nz] / s[nz]
    return U
