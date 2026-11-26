# Reduced-order backward Euler FEM for Newton soft bodies. x = X + U q with U (3N, r) mass-orthonormal;
# each step minimizes |q - y|^2 / (2 h^2) + sum_e w_e Psi_e(F_e(q)) with Newton + a small line search.
# vec(F_e - I) = B_e q with B_e = G_e U_e precomputed, so an iteration never touches the full mesh.

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase

from .fem import TetMesh

wp.set_module_options({"enable_backward": False})

LINE_SEARCH_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0)  # 0 rejects the step
N_ALPHA = len(LINE_SEARCH_ALPHAS)


# column-major vec(F) of element e as a matrix
@wp.func
def load_vec9(f: wp.array[float], e: int) -> wp.mat33:
    b = 9 * e
    return wp.mat33(
        f[b + 0], f[b + 3], f[b + 6],
        f[b + 1], f[b + 4], f[b + 7],
        f[b + 2], f[b + 5], f[b + 8],
    )  # fmt: skip


# column j of B_e as a 3x3
@wp.func
def load_dF(B: wp.array2d[float], e: int, j: int) -> wp.mat33:
    b = 9 * e
    return wp.mat33(
        B[b + 0, j], B[b + 3, j], B[b + 6, j],
        B[b + 1, j], B[b + 4, j], B[b + 7, j],
        B[b + 2, j], B[b + 5, j], B[b + 8, j],
    )  # fmt: skip


@wp.func
def cofactor(F: wp.mat33) -> wp.mat33:
    f0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    return wp.matrix_from_cols(wp.cross(f1, f2), wp.cross(f2, f0), wp.cross(f0, f1))


# 2x Green strain F^T F - I, written in H to avoid cancellation at small strain
@wp.func
def green2(H: wp.mat33) -> wp.mat33:
    return H + wp.transpose(H) + wp.transpose(H) * H


# energy, stress and Hessian match newton/_src/solvers/vbd/particle_vbd_kernels.py.
# Everything is in H = F - I so each term is O(|H|^2). with d = J - 1 = tr H + i2(H) + det H,
#   psi = mu/2 |H|^2 + lam/2 d^2 - mu (i2(H) + det H)
# equals mu/2 (tr C - 3) + lam/2 ((J - alpha)^2 - (1 - alpha)^2) but keeps its float32 digits at
# small strain. damping potential is kd/(2h) |F^T F - Fp^T Fp|^2.
@wp.func
def element_energy(H: wp.mat33, Gp: wp.mat33, mu: float, lam_nh: float, kd: float, inv_h: float) -> float:
    trH = wp.trace(H)
    i2 = 0.5 * (trH * trH - wp.trace(H * H))
    nl = i2 + wp.determinant(H)
    d = trH + nl
    E = green2(H) - Gp
    return 0.5 * mu * wp.ddot(H, H) + 0.5 * lam_nh * d * d - mu * nl + 0.5 * kd * inv_h * wp.ddot(E, E)


# P = mu F + lam (J - alpha) cof(F) + (2 kd / h) F E, in H:
#   mu (H + H^T - tr(H) I - cof(H)) + lam d cof(F) + (2 kd / h) F E
@wp.func
def element_stress(H: wp.mat33, Gp: wp.mat33, mu: float, lam_nh: float, kd: float, inv_h: float) -> wp.mat33:
    I = wp.identity(n=3, dtype=float)
    F = I + H
    trH = wp.trace(H)
    cofH = cofactor(H)
    d = trH + 0.5 * (trH * trH - wp.trace(H * H)) + wp.determinant(H)
    cofF = (1.0 + trH) * I - wp.transpose(H) + cofH
    P = mu * (H + wp.transpose(H) - trH * I - cofH) + (lam_nh * d) * cofF
    return P + (2.0 * kd * inv_h) * (F * (green2(H) - Gp))


# exact stable NH Hessian (plus VBD's Gauss-Newton damping term) applied to dF:
#   mu dF + lam (cof : dF) cof + lam (J - alpha) dcof[dF] + (2 kd / h) F (dF^T F + F^T dF)
@wp.func
def hessian_apply(H: wp.mat33, dF: wp.mat33, mu: float, lam_nh: float, kd: float, inv_h: float) -> wp.mat33:
    F = wp.identity(n=3, dtype=float) + H
    trH = wp.trace(H)
    s = lam_nh * (trH + 0.5 * (trH * trH - wp.trace(H * H)) + wp.determinant(H)) - mu  # lam (J - alpha)
    cof = cofactor(F)
    f0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    d0 = wp.vec3(dF[0, 0], dF[1, 0], dF[2, 0])
    d1 = wp.vec3(dF[0, 1], dF[1, 1], dF[2, 1])
    d2 = wp.vec3(dF[0, 2], dF[1, 2], dF[2, 2])
    dcof = wp.matrix_from_cols(
        wp.cross(d1, f2) + wp.cross(f1, d2),
        wp.cross(d2, f0) + wp.cross(f2, d0),
        wp.cross(d0, f1) + wp.cross(f0, d1),
    )
    out = mu * dF + (lam_nh * wp.ddot(cof, dF)) * cof + s * dcof
    if kd > 0.0:
        out += (2.0 * kd * inv_h) * (F * (wp.transpose(dF) * F + wp.transpose(F) * dF))
    return out


# one thread per row of the stacked (9E, r) B
@wp.kernel
def gemv_kernel(B: wp.array2d[float], x: wp.array[float], r: int, out: wp.array[float]):
    i = wp.tid()
    acc = float(0.0)
    for j in range(r):
        acc += B[i, j] * x[j]
    out[i] = acc


@wp.kernel
def eval_elements_kernel(
    Hvec: wp.array[float],  # vec(F - I) = B q
    weight: wp.array[float],  # cubature weight * rest volume
    mu: wp.array[float],
    lam_nh: wp.array[float],
    kd: wp.array[float],
    inv_h: float,
    first_iteration: int,
    Gprev: wp.array[wp.mat33],
    H_out: wp.array[wp.mat33],
    P_out: wp.array2d[float],  # (9E, 1)
):
    e = wp.tid()
    H = load_vec9(Hvec, e)
    if first_iteration != 0:
        # the first Newton iterate is q_n, so this is the start-of-step strain used by the damping term
        Gprev[e] = green2(H)
    P = weight[e] * element_stress(H, Gprev[e], mu[e], lam_nh[e], kd[e], inv_h)
    H_out[e] = H
    for a in range(9):
        P_out[9 * e + a, 0] = P[a % 3, a // 3]


# C_e[:, j] = w_e Hess_e B_e[:, j]
@wp.kernel
def hessian_times_basis_kernel(
    Hd: wp.array[wp.mat33],
    B: wp.array2d[float],
    weight: wp.array[float],
    mu: wp.array[float],
    lam_nh: wp.array[float],
    kd: wp.array[float],
    inv_h: float,
    C_out: wp.array2d[float],
):
    e, j = wp.tid()
    HdF = weight[e] * hessian_apply(Hd[e], load_dF(B, e, j), mu[e], lam_nh[e], kd[e], inv_h)
    b = 9 * e
    for c in range(3):
        for i in range(3):
            C_out[b + 3 * c + i, j] = HdF[i, c]


vec_alpha_d = wp.types.vector(length=N_ALPHA, dtype=wp.float64)


# elastic energy at every line-search candidate, block-reduced with tiles in float64
@wp.kernel
def line_search_energy_kernel(
    alphas: wp.array[float],
    Hd: wp.array[wp.mat33],
    Fd: wp.array[float],  # B d, stacked
    Gprev: wp.array[wp.mat33],
    weight: wp.array[float],
    mu: wp.array[float],
    lam_nh: wp.array[float],
    kd: wp.array[float],
    inv_h: float,
    n_elems: int,
    energy: wp.array[wp.float64],
):
    blk, lane = wp.tid()
    e = blk * wp.block_dim() + lane
    psi = vec_alpha_d()
    if e < n_elems:
        dF = load_vec9(Fd, e)
        He = Hd[e]
        w = weight[e]
        for k in range(wp.static(N_ALPHA)):
            psi[k] = wp.float64(w * element_energy(He - alphas[k] * dF, Gprev[e], mu[e], lam_nh[e], kd[e], inv_h))
    wp.tile_atomic_add(energy, wp.tile_sum(wp.tile(psi), axis=1))


@wp.kernel
def begin_step_kernel(
    q: wp.array[float], qd: wp.array[float], f_r: wp.array[float], h: float,
    q_prev: wp.array[float], y: wp.array[float],
):
    j = wp.tid()
    q_prev[j] = q[j]
    y[j] = q[j] + h * qd[j] + h * h * f_r[j]


@wp.kernel
def end_step_kernel(q: wp.array[float], q_prev: wp.array[float], inv_h: float, qd: wp.array[float]):
    j = wp.tid()
    qd[j] = (q[j] - q_prev[j]) * inv_h


@wp.kernel
def expand_kernel(
    q: wp.array[float], qd: wp.array[float], U: wp.array2d[float], X: wp.array[wp.vec3], r: int,
    x_out: wp.array[wp.vec3], v_out: wp.array[wp.vec3],
):
    p = wp.tid()
    x = X[p]
    v = wp.vec3(0.0)
    for j in range(r):
        u = wp.vec3(U[3 * p + 0, j], U[3 * p + 1, j], U[3 * p + 2, j])
        x += u * q[j]
        v += u * qd[j]
    x_out[p] = x
    v_out[p] = v


@wp.kernel
def project_force_kernel(f: wp.array[wp.vec3], U: wp.array2d[float], f_r: wp.array[float]):
    p, j = wp.tid()
    fp = f[p]
    val = U[3 * p + 0, j] * fp[0] + U[3 * p + 1, j] * fp[1] + U[3 * p + 2, j] * fp[2]
    if val != 0.0:
        wp.atomic_add(f_r, j, val)


_REDUCE_BLOCKS = 132  # one per H100 SM
_TILED_KERNELS: dict = {}


# tile shapes must be compile-time constants, so these get built per (r, tile_rows)
def _tiled_kernels(r: int, tile_rows: int):
    key = (r, tile_rows)
    if key in _TILED_KERNELS:
        return _TILED_KERNELS[key]
    R = r
    TR = tile_rows

    @wp.kernel(module="unique", enable_backward=False)
    def reduce_kernel(B: wp.array2d[float], C: wp.array2d[float], P: wp.array2d[float], n_chunks: int,
                      n_blocks: int, H_r: wp.array2d[float], g_r: wp.array2d[float]):
        blk, _lane = wp.tid()
        H_acc = wp.tile_zeros(shape=(R, R), dtype=float)
        g_acc = wp.tile_zeros(shape=(R, 1), dtype=float)
        c = blk
        while c < n_chunks:
            tB = wp.tile_load(B, shape=(TR, R), offset=(c * TR, 0))
            tC = wp.tile_load(C, shape=(TR, R), offset=(c * TR, 0))
            tP = wp.tile_load(P, shape=(TR, 1), offset=(c * TR, 0))
            tBt = wp.tile_transpose(tB)
            wp.tile_matmul(tBt, tC, H_acc)
            wp.tile_matmul(tBt, tP, g_acc)
            c += n_blocks
        wp.tile_atomic_add(H_r, H_acc)
        wp.tile_atomic_add(g_r, g_acc)

    @wp.kernel(module="unique", enable_backward=False)
    def solve_kernel(H_r: wp.array2d[float], g_r: wp.array2d[float], q: wp.array[float], y: wp.array[float],
                     inv_h2: float, d: wp.array[float]):
        tH = wp.tile_load(H_r, shape=(R, R), storage="shared")
        tA = wp.tile_diag_add(tH, wp.tile_full(shape=R, value=inv_h2, dtype=float))
        tg = wp.tile_load(g_r, shape=(R, 1))
        tq = wp.tile_load(q, shape=R)
        ty = wp.tile_load(y, shape=R)
        grad = wp.tile_reshape(tg, shape=(R,)) + (tq - ty) * inv_h2
        # zero the accumulators for the next reduction
        wp.tile_store(H_r, wp.tile_zeros(shape=(R, R), dtype=float))
        wp.tile_store(g_r, wp.tile_zeros(shape=(R, 1), dtype=float))
        L = wp.tile_cholesky(tA)
        wp.tile_store(d, wp.tile_cholesky_solve(L, grad))

    @wp.kernel(module="unique", enable_backward=False)
    def select_kernel(q: wp.array[float], d: wp.array[float], y: wp.array[float], alphas: wp.array[float],
                      energy: wp.array[wp.float64], inv_h2: float, stats: wp.array[float]):
        _blk, lane = wp.tid()
        tq = wp.tile_load(q, shape=R)
        td = wp.tile_load(d, shape=R)
        ty = wp.tile_load(y, shape=R)
        r0 = tq - ty
        best_a = float(0.0)
        best_e = wp.float64(0.0)
        for k in range(wp.static(N_ALPHA)):
            a = alphas[k]
            diff = r0 - a * td
            inertia = 0.5 * inv_h2 * wp.tile_sum(wp.tile_map(wp.mul, diff, diff))[0]
            etot = wp.float64(inertia) + energy[k]
            # strict '<' keeps the earliest (largest) step on ties, i.e. the full Newton step
            if k == 0 or etot < best_e:
                best_a = a
                best_e = etot
        wp.tile_store(q, tq - best_a * td)
        wp.tile_store(energy, wp.tile_zeros(shape=N_ALPHA, dtype=wp.float64))
        dd = wp.tile_sum(wp.tile_map(wp.mul, td, td))[0]
        if lane == 0:
            stats[0] = float(best_e)
            stats[1] = best_a
            stats[2] = wp.sqrt(dd)

    _TILED_KERNELS[key] = (reduce_kernel, solve_kernel, select_kernel)
    return _TILED_KERNELS[key]


# Keeps its own reduced state (q, qdot). step() expands into state_out, pass None to skip the O(N r)
# expansion on substeps. cubature is (element_ids, weights), weights multiply the rest volume; None
# integrates every element. apply_particle_forces projects state_in.particle_f each step (O(N r)).
# one Newton iteration is as accurate as three at 10 substeps per frame.
class SolverReducedFEM(SolverBase):

    def __init__(
        self,
        model: newton.Model,
        basis: np.ndarray,
        *,
        newton_iterations: int = 1,
        cubature: tuple[np.ndarray, np.ndarray] | None = None,
        apply_particle_forces: bool = False,
        tile_rows: int = 72,
        block_dim: int = 128,
    ):
        super().__init__(model)
        self.mesh = mesh = TetMesh.from_model(model)
        U = np.asarray(basis, dtype=np.float64)
        self.r = r = U.shape[1]
        self.newton_iterations = newton_iterations
        self.apply_particle_forces = apply_particle_forces
        # warp runs CPU tile kernels with one thread per block, and the line-search grid depends on it
        self.block_dim = block_dim if model.device.is_cuda else 1
        m = mesh.mass_diag()
        if not np.allclose(U.T @ (m[:, None] * U), np.eye(r), atol=1e-6):
            raise ValueError("basis must be mass-orthonormal (U^T M U = I)")
        elems, w = cubature if cubature is not None else (np.arange(mesh.n_tets), np.ones(mesh.n_tets))
        self.n_elems = ne = len(elems)

        # B_e = G_e U_e in float64, stored float32, rows padded to a multiple of tile_rows
        G = mesh.element_G()[elems]  # (ne, 9, 12)
        dofs = (3 * mesh.tets[elems][:, :, None] + np.arange(3)).reshape(ne, 12)
        n_rows = 9 * ne
        self.n_chunks = n_chunks = (n_rows + tile_rows - 1) // tile_rows
        n_rows_pad = n_chunks * tile_rows
        Bflat = np.zeros((n_rows_pad, r))
        Bflat[:n_rows] = np.einsum("eab,ebj->eaj", G, U[dofs]).reshape(n_rows, r)
        # the rest configuration defines Dm_inv, so F(X) = I and vec(F - I) = B_e q exactly
        F0 = np.einsum("eab,eb->ea", G, mesh.rest.reshape(-1)[dofs])
        assert np.abs(F0 - np.eye(3).reshape(-1)).max() < 1e-6, "rest deformation gradient must be identity"
        self.n_rows = n_rows
        dev = model.device
        f32 = wp.float32
        self.B = wp.array(Bflat, dtype=f32, device=dev)
        self.weight = wp.array(w * mesh.volume[elems], dtype=f32, device=dev)
        self.mu = wp.array(mesh.mu[elems], dtype=f32, device=dev)
        self.lam_nh = wp.array(mesh.lam[elems] + mesh.mu[elems], dtype=f32, device=dev)
        self.kd = wp.array(mesh.kd[elems], dtype=f32, device=dev)
        self.Gprev = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.Hdisp = wp.zeros(ne, dtype=wp.mat33, device=dev)
        self.Hvec = wp.zeros(n_rows_pad, dtype=f32, device=dev)
        self.Fd = wp.zeros(n_rows_pad, dtype=f32, device=dev)
        self.P = wp.zeros((n_rows_pad, 1), dtype=f32, device=dev)
        self.C = wp.zeros((n_rows_pad, r), dtype=f32, device=dev)
        # accumulators; kept zero between iterations by the kernels that consume them
        self.H_r = wp.zeros((r, r), dtype=f32, device=dev)
        self.g_r = wp.zeros((r, 1), dtype=f32, device=dev)
        self.energy = wp.zeros(N_ALPHA, dtype=wp.float64, device=dev)
        self.alphas = wp.array(LINE_SEARCH_ALPHAS, dtype=f32, device=dev)
        self.stats = wp.zeros(3, dtype=f32, device=dev)  # energy, alpha, |d| of the last iteration
        self.U = wp.array(U, dtype=f32, device=dev)
        self.X = wp.array(mesh.rest, dtype=wp.vec3, device=dev)
        g_r = U.T @ (m * np.tile(mesh.gravity, mesh.n_particles))
        self.f_gravity = wp.array(g_r, dtype=f32, device=dev)
        self.f_r = wp.array(g_r, dtype=f32, device=dev)
        self.q = wp.zeros(r, dtype=f32, device=dev)
        self.qd = wp.zeros(r, dtype=f32, device=dev)
        self.q_prev = wp.zeros(r, dtype=f32, device=dev)
        self.y = wp.zeros(r, dtype=f32, device=dev)
        self.d = wp.zeros(r, dtype=f32, device=dev)
        self._reduce_kernel, self._solve_kernel, self._select_kernel = _tiled_kernels(r, tile_rows)
        self._reduce_blocks = min(_REDUCE_BLOCKS, n_chunks)

    def reset(self, q0: np.ndarray | None = None, qd0: np.ndarray | None = None):
        self.q.assign(np.zeros(self.r) if q0 is None else q0)
        self.qd.assign(np.zeros(self.r) if qd0 is None else qd0)

    # M-orthogonal projection of a full state
    def project_state(self, state: newton.State):
        m = self.mesh.mass_diag()
        U = self.U.numpy().astype(np.float64)
        dx = (state.particle_q.numpy() - self.mesh.rest).reshape(-1)
        v = state.particle_qd.numpy().reshape(-1)
        self.reset(U.T @ (m * dx), U.T @ (m * v))

    def newton_iteration(self, h: float, first: bool):
        dev = self.model.device
        r, ne = self.r, self.n_elems
        inv_h = 1.0 / h
        mat = [self.weight, self.mu, self.lam_nh, self.kd, inv_h]
        wp.launch(gemv_kernel, dim=self.n_rows, inputs=[self.B, self.q, r], outputs=[self.Hvec], device=dev)
        wp.launch(eval_elements_kernel, dim=ne, inputs=[self.Hvec, *mat, int(first)],
                  outputs=[self.Gprev, self.Hdisp, self.P], device=dev)
        wp.launch(hessian_times_basis_kernel, dim=(ne, r), inputs=[self.Hdisp, self.B, *mat], outputs=[self.C],
                  device=dev)
        wp.launch_tiled(
            self._reduce_kernel, dim=[self._reduce_blocks], block_dim=self.block_dim, device=dev,
            inputs=[self.B, self.C, self.P, self.n_chunks, self._reduce_blocks], outputs=[self.H_r, self.g_r],
        )  # fmt: skip
        wp.launch_tiled(
            self._solve_kernel, dim=[1], block_dim=self.block_dim, device=dev,
            inputs=[self.H_r, self.g_r, self.q, self.y, inv_h * inv_h], outputs=[self.d],
        )  # fmt: skip
        wp.launch(gemv_kernel, dim=self.n_rows, inputs=[self.B, self.d, r], outputs=[self.Fd], device=dev)
        wp.launch_tiled(
            line_search_energy_kernel, dim=[(ne + self.block_dim - 1) // self.block_dim], block_dim=self.block_dim,
            inputs=[self.alphas, self.Hdisp, self.Fd, self.Gprev, *mat, ne], outputs=[self.energy], device=dev,
        )  # fmt: skip
        wp.launch_tiled(
            self._select_kernel, dim=[1], block_dim=64, device=dev,
            inputs=[self.q, self.d, self.y, self.alphas, self.energy, inv_h * inv_h], outputs=[self.stats],
        )  # fmt: skip

    def step(self, state_in, state_out, control, contacts, dt: float, **kwargs):
        dev = self.model.device
        r = self.r
        if self.apply_particle_forces and state_in is not None and state_in.particle_f is not None:
            wp.copy(self.f_r, self.f_gravity)
            wp.launch(project_force_kernel, dim=(self.mesh.n_particles, r),
                      inputs=[state_in.particle_f, self.U], outputs=[self.f_r], device=dev)
        wp.launch(begin_step_kernel, dim=r, inputs=[self.q, self.qd, self.f_r, dt], outputs=[self.q_prev, self.y],
                  device=dev)
        for it in range(self.newton_iterations):
            self.newton_iteration(dt, first=(it == 0))
        wp.launch(end_step_kernel, dim=r, inputs=[self.q, self.q_prev, 1.0 / dt], outputs=[self.qd], device=dev)
        if state_out is not None:
            self.write_state(state_out)

    def write_state(self, state):
        wp.launch(
            expand_kernel, dim=self.mesh.n_particles, device=self.model.device,
            inputs=[self.q, self.qd, self.U, self.X, self.r], outputs=[state.particle_q, state.particle_qd],
        )  # fmt: skip
