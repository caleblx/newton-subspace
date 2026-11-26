# Full-order backward Euler, Newton + Jacobi-PCG, same energy as SolverVBD. Slow; used to check
# the VBD references and to replace them on stiff materials where Gauss-Seidel VBD stalls.

from __future__ import annotations

import numpy as np
import warp as wp

from .fem import TetMesh
from .solver import element_energy, element_stress, green2, hessian_apply

wp.set_module_options({"enable_backward": False})


@wp.func
def tet_H(e: int, x: wp.array[wp.vec3], X: wp.array[wp.vec3], tets: wp.array2d[int], a: wp.array3d[float]) -> wp.mat33:
    H = wp.mat33()
    for v in range(4):
        p = tets[e, v]
        u = x[p] - X[p]
        av = wp.vec3(a[e, v, 0], a[e, v, 1], a[e, v, 2])
        H += wp.outer(u, av)
    return H


@wp.kernel
def begin_kernel(Gp: wp.array[wp.mat33], x: wp.array[wp.vec3], X: wp.array[wp.vec3], tets: wp.array2d[int],
                 a: wp.array3d[float]):
    e = wp.tid()
    Gp[e] = green2(tet_H(e, x, X, tets, a))


@wp.kernel
def gradient_kernel(x: wp.array[wp.vec3], X: wp.array[wp.vec3], tets: wp.array2d[int], a: wp.array3d[float],
                    vol: wp.array[float], mu: wp.array[float], lam: wp.array[float], kd: wp.array[float],
                    Gp: wp.array[wp.mat33], inv_h: float, Hout: wp.array[wp.mat33], g: wp.array[wp.vec3]):
    e = wp.tid()
    H = tet_H(e, x, X, tets, a)
    Hout[e] = H
    P = vol[e] * element_stress(H, Gp[e], mu[e], lam[e], kd[e], inv_h)
    for v in range(4):
        av = wp.vec3(a[e, v, 0], a[e, v, 1], a[e, v, 2])
        wp.atomic_add(g, tets[e, v], P * av)


@wp.kernel
def diag_kernel(Hd: wp.array[wp.mat33], tets: wp.array2d[int], a: wp.array3d[float], vol: wp.array[float],
                mu: wp.array[float], lam: wp.array[float], kd: wp.array[float], inv_h: float,
                diag: wp.array[wp.vec3]):
    e = wp.tid()
    for v in range(4):
        av = wp.vec3(a[e, v, 0], a[e, v, 1], a[e, v, 2])
        dvec = wp.vec3()
        for i in range(3):
            ei = wp.vec3()
            ei[i] = 1.0
            dF = wp.outer(ei, av)
            dvec[i] = vol[e] * wp.ddot(dF, hessian_apply(Hd[e], dF, mu[e], lam[e], kd[e], inv_h))
        wp.atomic_add(diag, tets[e, v], dvec)


@wp.kernel
def hessvec_kernel(Hd: wp.array[wp.mat33], tets: wp.array2d[int], a: wp.array3d[float], vol: wp.array[float],
                   mu: wp.array[float], lam: wp.array[float], kd: wp.array[float], inv_h: float,
                   p: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    e = wp.tid()
    dF = wp.mat33()
    for v in range(4):
        av = wp.vec3(a[e, v, 0], a[e, v, 1], a[e, v, 2])
        dF += wp.outer(p[tets[e, v]], av)
    HdF = vol[e] * hessian_apply(Hd[e], dF, mu[e], lam[e], kd[e], inv_h)
    for v in range(4):
        av = wp.vec3(a[e, v, 0], a[e, v, 1], a[e, v, 2])
        wp.atomic_add(out, tets[e, v], HdF * av)


@wp.kernel
def energy_kernel(x: wp.array[wp.vec3], d: wp.array[wp.vec3], alpha: float, X: wp.array[wp.vec3],
                  tets: wp.array2d[int], a: wp.array3d[float], vol: wp.array[float], mu: wp.array[float],
                  lam: wp.array[float], kd: wp.array[float], Gp: wp.array[wp.mat33], inv_h: float,
                  out: wp.array[wp.float64]):
    e = wp.tid()
    H = wp.mat33()
    for v in range(4):
        p = tets[e, v]
        av = wp.vec3(a[e, v, 0], a[e, v, 1], a[e, v, 2])
        H += wp.outer(x[p] - alpha * d[p] - X[p], av)
    wp.atomic_add(out, 0, wp.float64(vol[e] * element_energy(H, Gp[e], mu[e], lam[e], kd[e], inv_h)))


# state stays in float64 on the host between Newton iterations
class FullNewtonPCG:
    newton_tol = 1e-4
    step_tol = 1e-7  # [m], about float32 resolution on the positions
    max_newton = 12
    cg_tol = 1e-5
    max_cg = 3000

    def __init__(self, model):
        self.mesh = mesh = TetMesh.from_model(model)
        self.dev = model.device
        f32 = wp.float32
        self.X = wp.array(mesh.rest, dtype=wp.vec3, device=self.dev)
        self.tets = wp.array(mesh.tets.astype(np.int32), dtype=int, device=self.dev)
        self.a = wp.array(mesh.vertex_coeffs(), dtype=f32, device=self.dev)
        self.vol = wp.array(mesh.volume, dtype=f32, device=self.dev)
        self.mu = wp.array(mesh.mu, dtype=f32, device=self.dev)
        self.lam = wp.array(mesh.lam + mesh.mu, dtype=f32, device=self.dev)
        self.kd = wp.array(mesh.kd, dtype=f32, device=self.dev)
        E, N = mesh.n_tets, mesh.n_particles
        self.Gp = wp.zeros(E, dtype=wp.mat33, device=self.dev)
        self.Hd = wp.zeros(E, dtype=wp.mat33, device=self.dev)
        self.buf = wp.zeros(N, dtype=wp.vec3, device=self.dev)
        self.buf2 = wp.zeros(N, dtype=wp.vec3, device=self.dev)
        self.en = wp.zeros(1, dtype=wp.float64, device=self.dev)
        self.m = mesh.mass[:, None].astype(np.float64)
        self.free = mesh.free[:, None].astype(np.float64)
        self.x = mesh.rest.copy()
        self.v = np.zeros_like(self.x)

    def _launch(self, k, dim, inputs):
        wp.launch(k, dim=dim, inputs=inputs, device=self.dev)

    def _elastic_grad(self, x, inv_h):
        self.buf.zero_()
        xw = wp.array(x, dtype=wp.vec3, device=self.dev)
        self._launch(gradient_kernel, self.mesh.n_tets, [xw, self.X, self.tets, self.a, self.vol, self.mu, self.lam,
                                                         self.kd, self.Gp, inv_h, self.Hd, self.buf])
        return self.buf.numpy().astype(np.float64)

    def _energy(self, x, d, alpha, y, inv_h):
        self.en.zero_()
        self._launch(energy_kernel, self.mesh.n_tets,
                     [wp.array(x, dtype=wp.vec3, device=self.dev), wp.array(d, dtype=wp.vec3, device=self.dev), alpha,
                      self.X, self.tets, self.a, self.vol, self.mu, self.lam, self.kd, self.Gp, inv_h, self.en])
        xn = x - alpha * d
        return 0.5 * inv_h * inv_h * float(np.sum(self.m * (xn - y) ** 2)) + float(self.en.numpy()[0])

    def _hessvec(self, p, inv_h):
        self.buf2.zero_()
        self._launch(hessvec_kernel, self.mesh.n_tets, [self.Hd, self.tets, self.a, self.vol, self.mu, self.lam,
                                                        self.kd, inv_h, wp.array(p, dtype=wp.vec3, device=self.dev),
                                                        self.buf2])
        return (self.buf2.numpy().astype(np.float64) + self.m * inv_h * inv_h * p) * self.free

    def step(self, h, f_ext):
        inv_h = 1.0 / h
        g_acc = self.mesh.gravity[None, :]
        y = self.x + h * self.v + h * h * (g_acc + f_ext / np.maximum(self.m, 1e-30)) * self.free
        x_prev = self.x.copy()
        self._launch(begin_kernel, self.mesh.n_tets,
                     [self.Gp, wp.array(x_prev, dtype=wp.vec3, device=self.dev), self.X, self.tets, self.a])
        x = x_prev.copy()
        for it in range(self.max_newton):
            g = (self._elastic_grad(x, inv_h) + self.m * inv_h * inv_h * (x - y)) * self.free
            gn = np.sqrt(np.sum(g**2))
            if it == 0:
                g0 = max(gn, 1e-30)
            if gn < self.newton_tol * g0 or gn < 1e-12:
                break
            # Jacobi preconditioner from the element Hessian diagonals
            self.buf.zero_()
            self._launch(diag_kernel, self.mesh.n_tets, [self.Hd, self.tets, self.a, self.vol, self.mu, self.lam,
                                                         self.kd, inv_h, self.buf])
            Dinv = self.free / (self.buf.numpy().astype(np.float64) + self.m * inv_h * inv_h + 1e-30)
            # PCG on (M/h^2 + K) d = g
            d = np.zeros_like(x)
            r = g.copy()
            z = Dinv * r
            p = z.copy()
            rz = np.sum(r * z)
            for _ in range(self.max_cg):
                Ap = self._hessvec(p, inv_h)
                pAp = np.sum(p * Ap)
                if pAp <= 0:
                    break
                al = rz / pAp
                d += al * p
                r -= al * Ap
                if np.sqrt(np.sum(r**2)) < self.cg_tol * gn:
                    break
                z = Dinv * r
                rz_new = np.sum(r * z)
                p = z + (rz_new / rz) * p
                rz = rz_new
            # backtracking line search on the incremental potential
            e0 = self._energy(x, d, 0.0, y, inv_h)
            alpha = 1.0
            while alpha > 1e-4 and self._energy(x, d, alpha, y, inv_h) > e0:
                alpha *= 0.5
            x = x - alpha * d
            if alpha * np.abs(d).max() < self.step_tol:
                break
        self.v = (x - x_prev) * inv_h
        self.x = x
        return x
