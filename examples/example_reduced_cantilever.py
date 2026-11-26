# Neo-Hookean cantilever simulated in a 30-dim subspace with SolverReducedFEM, next to the
# full-order SolverVBD beam (offset along +y). Default basis is modes + modal derivatives;
# --basis pod builds one from VBD snapshots, --cubature N fits an N-element cubature.
# Viewer picking forces are projected into the subspace, so the reduced beam can be dragged.
#
# python examples/example_reduced_cantilever.py [--viewer gl|null|usd] [--basis pod] [--cubature 200]

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp

import newton
import newton.examples
from newton_subspace import SolverReducedFEM, TetMesh, modal_basis, pod_basis
from newton_subspace.cubature import fit_cubature
from newton_subspace.sim import TRAIN_SCENARIOS, FullOrderRunner, ReducedRunner


def build(res, k, offset_y=0.0):
    builder = newton.ModelBuilder()
    dim_x, dim_yz = 12 * res, 4 * res
    builder.add_soft_grid(
        pos=wp.vec3(0.0, offset_y, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0),
        dim_x=dim_x, dim_y=dim_yz, dim_z=dim_yz,
        cell_x=1.2 / dim_x, cell_y=0.4 / dim_yz, cell_z=0.4 / dim_yz,
        density=1.0e3, k_mu=k, k_lambda=k, k_damp=1.0e1, fix_left=True,
    )  # fmt: skip
    builder.color()
    return builder.finalize()


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.model = build(args.res, args.stiffness)
        mesh = TetMesh.from_model(self.model)
        print(f"full model: {mesh.n_particles} particles, {mesh.n_tets} tets, {mesh.n_free_dofs} free DOFs")

        if args.basis == "pod":
            snaps = np.concatenate([FullOrderRunner(self.model, sc, iterations=100).run(120) for sc in TRAIN_SCENARIOS])
            U = pod_basis(mesh, snaps, args.r)
        else:
            U = modal_basis(mesh, args.r, gravity_mode=True, modal_derivatives=True)
            snaps = None
        print(f"basis: {args.basis}, r = {U.shape[1]}")

        cubature = None
        if args.cubature > 0:
            if snaps is None:
                # no snapshots for modal bases, so train on an exact reduced roll-out
                snaps = np.concatenate([ReducedRunner(self.model, sc, U).run(120) for sc in TRAIN_SCENARIOS])
            m = mesh.mass_diag()
            Q = ((snaps - mesh.rest).reshape(len(snaps), -1) * m) @ U
            ids, w, info = fit_cubature(mesh, U, Q[:: max(1, len(Q) // 120)], max_points=args.cubature, tol=1e-4)
            cubature = (ids, w)
            print(f"cubature: {len(ids)} of {mesh.n_tets} elements, fit residual {info['residual']:.4f}")

        self.solver = SolverReducedFEM(self.model, U, cubature=cubature, apply_particle_forces=True)
        self.state_0 = self.model.state()

        self.show_reference = args.reference
        if self.show_reference:
            self.ref_model = build(args.res, args.stiffness, offset_y=0.8)
            self.ref_solver = newton.solvers.SolverVBD(
                self.ref_model, iterations=args.vbd_iterations, particle_enable_self_contact=False,
                particle_enable_tile_solve=False,
            )  # fmt: skip
            self.ref_0, self.ref_1 = self.ref_model.state(), self.ref_model.state()
            self.ref_control = self.ref_model.control()

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, None, None, None, self.sim_dt)
        self.solver.write_state(self.state_0)

    def step(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        if self.show_reference:
            for _ in range(self.sim_substeps):
                self.ref_0.clear_forces()
                self.ref_solver.step(self.ref_0, self.ref_1, self.ref_control, None, self.sim_dt)
                self.ref_0, self.ref_1 = self.ref_1, self.ref_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.show_reference:
            self.viewer.log_points("vbd_reference", self.ref_0.particle_q, radii=0.01)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(q)), "non-finite particle positions"
        assert q[:, 2].min() > -1.0 and q[:, 0].max() < 2.0, "beam left the expected volume"
        if self.show_reference:
            ref = self.ref_0.particle_q.numpy() - np.array([0.0, 0.8, 0.0])
            err = np.linalg.norm(q - ref) / max(np.linalg.norm(ref - self.solver.mesh.rest), 1e-12)
            print(f"test_final: reduced vs full-order VBD at t = {self.sim_time:.2f} s: relative error {err:.2%}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--res", type=int, default=2, help="grid resolution multiplier (12r x 4r x 4r cells)")
        parser.add_argument("--stiffness", type=float, default=1.0e6, help="Lame parameters mu = lambda [Pa]")
        parser.add_argument("--basis", choices=["modal-md", "pod"], default="modal-md")
        parser.add_argument("--r", type=int, default=30, help="basis size")
        parser.add_argument("--cubature", type=int, default=0, help="cubature elements (0 = exact integration)")
        parser.add_argument("--reference", action=argparse.BooleanOptionalAction, default=True,
                            help="also run the full-order VBD beam for comparison")
        parser.add_argument("--vbd-iterations", type=int, default=100)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
