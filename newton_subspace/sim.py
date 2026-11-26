# Load scenarios and full / reduced simulation drivers with CUDA-graph timing.

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .fem import TetMesh
from .solver import SolverReducedFEM

FPS = 60


# Gravity plus a load on the free-end particles for load_time seconds, then release.
# kind "force" pushes along direction, "torque" twists the tip about the beam axis.
# magnitude is the total load in units of the beam weight.
@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str = "none"
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnitude: float = 0.0
    load_time: float = 0.5

    def particle_forces(self, mesh: TetMesh) -> np.ndarray:
        f = np.zeros((mesh.n_particles, 3))
        if self.kind == "none":
            return f
        x = mesh.rest
        tip = np.isclose(x[:, 0], x[:, 0].max()) & mesh.free
        weight = mesh.mass.sum() * np.linalg.norm(mesh.gravity)
        if self.kind == "force":
            d = np.asarray(self.direction, dtype=np.float64)
            d = d / np.linalg.norm(d)
            f[tip] = self.magnitude * weight * d / tip.sum()
        elif self.kind == "torque":
            c = x[tip].mean(axis=0)
            rel = x[tip] - c
            t = np.cross(np.array([1.0, 0.0, 0.0]), rel)  # tangential, grows with radius
            norm = np.sum(np.linalg.norm(t, axis=1))
            f[tip] = self.magnitude * weight * t / norm
        return f


TRAIN_SCENARIOS = [
    Scenario("drop"),
    Scenario("push_up", "force", (0.0, 0.0, 1.0), 1.5),
    Scenario("push_side", "force", (0.0, 1.0, 0.0), 1.0),
    Scenario("push_down", "force", (0.0, 0.0, -1.0), 0.5),
]

TEST_SCENARIOS = [
    Scenario("diag_load", "force", (0.0, 0.7, 0.7), 1.2, 0.4),
    Scenario("side_heavy", "force", (0.0, -1.0, 0.2), 1.6, 0.3),
    Scenario("twist", "torque", magnitude=2.0, load_time=0.5),
]


# mass-weighted relative displacement error over the whole trajectory
def rel_error(traj: np.ndarray, ref: np.ndarray, mesh: TetMesh) -> float:
    m = mesh.mass[None, :, None]
    num = np.sum(m * (traj - ref) ** 2)
    den = np.sum(m * (ref - mesh.rest[None]) ** 2)
    return float(np.sqrt(num / den))


# per-frame load switching, substeps captured in a CUDA graph
class _Runner:
    def __init__(self, model: newton.Model, scenario: Scenario, substeps: int):
        self.model = model
        self.mesh = TetMesh.from_model(model)
        self.scenario = scenario
        self.substeps = substeps
        self.dt = 1.0 / (FPS * substeps)
        self.graph = None

    def _capture(self):
        if not self.model.device.is_cuda:
            return
        with wp.ScopedCapture(device=self.model.device) as cap:
            self._frame()
        self.graph = cap.graph

    def frame(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()

    def run(self, frames: int):
        traj = []
        for i in range(frames):
            self.set_load(i / FPS < self.scenario.load_time)
            self.frame()
            traj.append(self.positions())
        return np.array(traj)

    # median seconds per frame, no host readback
    def time_frames(self, frames: int, repeats: int = 3) -> float:
        best = []
        for _ in range(repeats):
            self.reset()
            wp.synchronize_device(self.model.device)
            t0 = time.perf_counter()
            for i in range(frames):
                self.set_load(i / FPS < self.scenario.load_time)
                self.frame()
            wp.synchronize_device(self.model.device)
            best.append((time.perf_counter() - t0) / frames)
        return float(np.median(best))


class FullOrderRunner(_Runner):
    def __init__(self, model, scenario, substeps=10, iterations=10):
        super().__init__(model, scenario, substeps)
        self.solver = newton.solvers.SolverVBD(
            model, iterations=iterations, particle_enable_self_contact=False, particle_enable_tile_solve=False
        )
        self.s0, self.s1 = model.state(), model.state()
        self.control = model.control()
        self.f_on = wp.array(scenario.particle_forces(self.mesh), dtype=wp.vec3, device=model.device)
        self.f_off = wp.zeros_like(self.f_on)
        self.reset()
        self._capture()

    def reset(self):
        q0 = self.model.particle_q
        for s in (self.s0, self.s1):
            wp.copy(s.particle_q, q0)
            s.particle_qd.zero_()

    def set_load(self, on: bool):
        src = self.f_on if on else self.f_off
        wp.copy(self.s0.particle_f, src)
        wp.copy(self.s1.particle_f, src)

    def _frame(self):
        # even number of substeps keeps s0 as the output after the frame
        for _ in range(self.substeps):
            self.solver.step(self.s0, self.s1, self.control, None, self.dt)
            self.s0, self.s1 = self.s1, self.s0

    def positions(self):
        return self.s0.particle_q.numpy().copy()


# host-driven, not timed
class PCGRunner:
    def __init__(self, model, scenario, substeps=10):
        from .full_newton import FullNewtonPCG  # noqa: PLC0415

        self.solver = FullNewtonPCG(model)
        self.scenario = scenario
        self.substeps = substeps
        self.f = scenario.particle_forces(self.solver.mesh)

    def run(self, frames: int):
        h = 1.0 / (FPS * self.substeps)
        traj = []
        for i in range(frames):
            f = self.f if i / FPS < self.scenario.load_time else 0.0 * self.f
            for _ in range(self.substeps):
                self.solver.step(h, f)
            traj.append(self.solver.x.copy())
        return np.array(traj)


class ReducedRunner(_Runner):
    def __init__(self, model, scenario, U, substeps=10, newton_iterations=3, cubature=None):
        super().__init__(model, scenario, substeps)
        self.solver = SolverReducedFEM(model, U, newton_iterations=newton_iterations, cubature=cubature)
        self.state = model.state()
        m = self.mesh.mass_diag()
        f = scenario.particle_forces(self.mesh).reshape(-1)
        g = U.T @ (m * np.tile(self.mesh.gravity, self.mesh.n_particles))
        self.fr_on = wp.array(g + U.T @ f, dtype=wp.float32, device=model.device)
        self.fr_off = wp.array(g, dtype=wp.float32, device=model.device)
        self.reset()
        self._capture()

    def reset(self):
        self.solver.reset()

    def set_load(self, on: bool):
        wp.copy(self.solver.f_r, self.fr_on if on else self.fr_off)

    def _frame(self):
        for _ in range(self.substeps):
            self.solver.step(None, None, None, None, self.dt)
        self.solver.write_state(self.state)  # expand x = X + U q once per frame

    def positions(self):
        return self.state.particle_q.numpy().copy()
