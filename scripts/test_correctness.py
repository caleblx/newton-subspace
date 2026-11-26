# with a full basis (r = all free DOFs) the reduced solver is plain full-order Newton,
# so it should match SolverVBD run to convergence

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp

import newton
from newton_subspace import SolverReducedFEM, TetMesh, mass_orthonormalize
from newton_subspace.scenes import cantilever

wp.init()


def run_vbd(model, steps, dt, iterations):
    solver = newton.solvers.SolverVBD(model, iterations=iterations, particle_enable_self_contact=False,
                                      particle_enable_tile_solve=False)
    s0, s1, c = model.state(), model.state(), model.control()
    traj = []
    for _ in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, c, None, dt)
        s0, s1 = s1, s0
        traj.append(s0.particle_q.numpy().copy())
    return np.array(traj)


def run_reduced(model, U, steps, dt, iters):
    solver = SolverReducedFEM(model, U, newton_iterations=iters)
    s = model.state()
    traj, stats = [], []
    for _ in range(steps):
        solver.step(s, s, None, None, dt)
        traj.append(s.particle_q.numpy().copy())
        stats.append(solver.stats.numpy().copy())
    return np.array(traj), np.array(stats)


def test_full_basis(k_damp):
    # small, stiffish beam so VBD converges
    model = cantilever(dims=(4, 2), k_mu=2e5, k_lambda=2e5, k_damp=k_damp)
    mesh = TetMesh.from_model(model)
    m = mesh.mass_diag()
    fd = mesh.free_dofs
    print(f"k_damp={k_damp} particles={mesh.n_particles} tets={mesh.n_tets} free dofs={len(fd)}")
    dt, steps = 1.0 / 600.0, 240
    # identity on the free dofs, scaled to be M-orthonormal
    U = np.zeros((3 * mesh.n_particles, len(fd)))
    U[fd, np.arange(len(fd))] = 1.0 / np.sqrt(m[fd])
    U = mass_orthonormalize(U, m)
    red, stats = run_reduced(model, U, steps, dt, iters=4)
    print("reduced line-search alphas (first steps):", stats[:5, 1], "final |d|:", stats[-1, 2])
    disp_scale = np.sqrt(np.mean(np.sum((red - mesh.rest) ** 2, axis=-1)))
    for iters in (10, 200):
        ref = run_vbd(model, steps, dt, iters)
        err = np.sqrt(np.mean(np.sum((red - ref) ** 2, axis=-1)))
        print(f"VBD iters={iters:4d}: rms |x_full_newton - x_vbd| = {err:.3e}  (rms disp {disp_scale:.3e}, rel {err/disp_scale:.2%})")


if __name__ == "__main__":
    test_full_basis(0.0)
    test_full_basis(1.0e2)
    test_full_basis(1.0e3)
