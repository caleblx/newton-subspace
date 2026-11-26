# Host-side (float64) FEM data from a finalized newton.Model. Same energy as SolverVBD: stable
# Neo-Hookean (Smith et al. 2018) with mu_nh = mu, lambda_nh = lambda + mu, plus VBD damping.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class TetMesh:
    rest: np.ndarray  # (N, 3)
    tets: np.ndarray  # (E, 4)
    Dm_inv: np.ndarray  # (E, 3, 3)
    volume: np.ndarray  # (E,)
    mu: np.ndarray  # (E,)
    lam: np.ndarray  # (E,)
    kd: np.ndarray  # (E,)
    mass: np.ndarray  # (N,) lumped
    free: np.ndarray  # (N,) bool, false where inv_mass == 0
    gravity: np.ndarray  # (3,)

    @property
    def n_particles(self) -> int:
        return self.rest.shape[0]

    @property
    def n_tets(self) -> int:
        return self.tets.shape[0]

    @property
    def free_dofs(self) -> np.ndarray:
        return np.flatnonzero(np.repeat(self.free, 3))

    @property
    def n_free_dofs(self) -> int:
        return 3 * int(self.free.sum())

    @staticmethod
    def from_model(model) -> TetMesh:
        tet_materials = model.tet_materials.numpy().astype(np.float64)
        Dm_inv = model.tet_poses.numpy().astype(np.float64)
        return TetMesh(
            rest=model.particle_q.numpy().astype(np.float64),
            tets=model.tet_indices.numpy().astype(np.int64),
            Dm_inv=Dm_inv,
            volume=1.0 / (6.0 * np.linalg.det(Dm_inv)),
            mu=tet_materials[:, 0],
            lam=tet_materials[:, 1],
            kd=tet_materials[:, 2],
            mass=model.particle_mass.numpy().astype(np.float64),
            free=model.particle_inv_mass.numpy() > 0.0,
            gravity=model.gravity.numpy()[0].astype(np.float64),
        )

    # a[e, v, j]: weight of vertex v in column j of F, F[:, j] = sum_v a[e, v, j] x_v
    def vertex_coeffs(self) -> np.ndarray:
        a = np.empty((self.n_tets, 4, 3))
        a[:, 1:, :] = self.Dm_inv
        a[:, 0, :] = -self.Dm_inv.sum(axis=1)
        return a

    # G[e] is 9 x 12 with vec(F) = G[e] @ x_e, column-major vec (index 3*j + i)
    def element_G(self) -> np.ndarray:
        a = self.vertex_coeffs()
        G = np.zeros((self.n_tets, 9, 12))
        for v in range(4):
            for j in range(3):
                for i in range(3):
                    G[:, 3 * j + i, 3 * v + i] = a[:, v, j]
        return G

    def mass_diag(self) -> np.ndarray:
        return np.repeat(self.mass, 3)

    # at F = I the stable NH Hessian is linear elasticity: dP = mu (dF + dF^T) + lambda tr(dF) I
    def rest_stiffness(self) -> sp.csr_matrix:
        E = self.n_tets
        T = np.zeros((9, 9))  # vec(F^T) = T vec(F)
        for i in range(3):
            for j in range(3):
                T[3 * j + i, 3 * i + j] = 1.0
        vI = np.eye(3).reshape(-1, order="F")
        C = (
            self.mu[:, None, None] * (np.eye(9) + T)[None]
            + self.lam[:, None, None] * np.outer(vI, vI)[None]
        ) * self.volume[:, None, None]
        G = self.element_G()
        Ke = np.einsum("eai,eab,ebj->eij", G, C, G)  # (E, 12, 12)
        dofs = (3 * self.tets[:, :, None] + np.arange(3)[None, None, :]).reshape(E, 12)
        rows = np.repeat(dofs, 12, axis=1).ravel()
        cols = np.tile(dofs, (1, 12)).ravel()
        return sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(3 * self.n_particles,) * 2).tocsr()
