// One body: the reduced solver plus external forcing (mouse spring, floor acceleration, balls).
// Used by demo_worker.js and scripts/check_demo_scenes.mjs.
//
// Balls are rigid spheres colliding with every third surface vertex, using rest-shape normals.
// A hit is an impulse j along n: ball velocity += j n / m, reduced velocity -= j U_s^T n (the
// basis is mass-orthonormal), so |U_s^T n|^2 is the body's inverse mass at that point.
import { ReducedSolver } from "./reduced_solver.js";

// solid steel at this radius
const BALL_R = 0.035, BALL_M = 1.4, BALL_SUB = 4;
const E_BUNNY = 0.35, E_FLOOR = 0.55, G = 9.81;
const MAX_BALLS = 12, BALL_LIFE = 8; // s

export class Body {
  constructor(meta, A, scale = 1) {
    const r = (this.r = meta.r);
    this.solver = new ReducedSolver(meta, A.B, scale);
    this.solver.settle();
    this.h = meta.h;
    this.substeps = meta.substeps;
    this.fa = meta.fa.map((f) => Float64Array.from(f));
    const nv = meta.arrays.rest.shape[0];
    const ids = [];
    for (let v = 0; v < nv; v += 3) ids.push(v);
    this.ns = ids.length;
    this.sRest = new Float64Array(3 * this.ns);
    this.sU = new Float64Array(3 * r * this.ns);
    ids.forEach((v, i) => {
      for (let c = 0; c < 3; c++) {
        this.sRest[3 * i + c] = A.rest[3 * v + c];
        for (let j = 0; j < r; j++) this.sU[(3 * i + c) * r + j] = A.U[(3 * v + c) * r + j];
      }
    });
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let i = 0; i < this.ns; i++) for (let c = 0; c < 3; c++) {
      lo[c] = Math.min(lo[c], this.sRest[3 * i + c]);
      hi[c] = Math.max(hi[c], this.sRest[3 * i + c]);
    }
    this.center = lo.map((l, c) => 0.5 * (l + hi[c]));
    this.radius = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    // area-weighted outward normals of the rest shape
    const vn = new Float64Array(3 * nv), P = A.rest, F = A.faces;
    for (let t = 0; t < F.length; t += 3) {
      const a = 3 * F[t], b = 3 * F[t + 1], c = 3 * F[t + 2];
      const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
      const e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      for (const v of [a, b, c]) for (let k = 0; k < 3; k++) vn[v + k] += n[k];
    }
    this.sN = new Float64Array(3 * this.ns);
    ids.forEach((v, i) => {
      const l = Math.hypot(vn[3 * v], vn[3 * v + 1], vn[3 * v + 2]) || 1;
      for (let k = 0; k < 3; k++) this.sN[3 * i + k] = vn[3 * v + k] / l;
    });
    this.sPos = new Float64Array(3 * this.ns);
    this.sFresh = false;
    this.balls = [];
    this.hits = [];
    this.offset = [0, 0]; // where this body stands; balls live in world coordinates
    this._u = new Float64Array(r);
  }

  reset() {
    this.solver.reset();
    this.balls = [];
    this.hits = [];
  }

  // one 60 Hz frame. accel: floor acceleration per substep, [[ax, ay], ...] or null
  frame({ pull = null, accel = null, throws = null } = {}) {
    const s = this.solver, r = this.r;
    if (throws) {
      for (const t of throws) this.balls.push({ p: t.p.slice(), v: t.v.slice(), age: 0 });
      while (this.balls.length > MAX_BALLS) this.balls.shift();
    }
    for (let k = 0; k < this.substeps; k++) {
      s.fext.fill(0);
      if (accel) {
        // in the floor's frame a floor acceleration a shows up as a body force -m a
        const [ax, ay] = accel[k];
        for (let j = 0; j < r; j++) s.fext[j] = -(ax * this.fa[0][j] + ay * this.fa[1][j]);
      }
      s.step(this.h, pull);
      this.sFresh = false;
      if (this.balls.length) this.stepBalls(this.h);
    }
    this.balls = this.balls.filter((b) => b.age < BALL_LIFE && b.p[2] > -1);
  }

  samplePositions() {
    if (this.sFresh) return;
    const { r, ns, sRest, sU, sPos } = this, q = this.solver.q;
    for (let i = 0; i < 3 * ns; i++) {
      let x = sRest[i];
      const o = i * r;
      for (let j = 0; j < r; j++) x += sU[o + j] * q[j];
      sPos[i] = x;
    }
    this.sFresh = true;
  }

  stepBalls(h) {
    const dt = h / BALL_SUB, R = BALL_R, R2 = R * R;
    const { r, sU, sPos, sN, center, radius, offset } = this, qd = this.solver.qd, u = this._u;
    for (const b of this.balls) {
      b.age += h;
      for (let k = 0; k < BALL_SUB; k++) {
        const p = b.p, v = b.v;
        v[2] -= G * dt;
        for (let c = 0; c < 3; c++) p[c] += v[c] * dt;
        if (p[2] < R) {
          p[2] = R;
          if (v[2] < 0) v[2] = -E_FLOOR * v[2];
          const f = Math.exp(-3 * dt); // rolling drag
          v[0] *= f;
          v[1] *= f;
        }
        // body, in its own coordinates
        const lx = p[0] - offset[0], ly = p[1] - offset[1], lz = p[2];
        if (Math.hypot(lx - center[0], ly - center[1], lz - center[2]) > radius + R + 0.05) continue;
        this.samplePositions();
        // nearest sample within reach, then how far the ball is past the plane through it
        let best = -1, bd = 4 * R2;
        for (let i = 0; i < this.ns; i++) {
          const dx = lx - sPos[3 * i], dy = ly - sPos[3 * i + 1], dz = lz - sPos[3 * i + 2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < bd) { bd = d2; best = i; }
        }
        if (best < 0) continue;
        const n = [sN[3 * best], sN[3 * best + 1], sN[3 * best + 2]];
        const depth = R - ((lx - sPos[3 * best]) * n[0] + (ly - sPos[3 * best + 1]) * n[1] + (lz - sPos[3 * best + 2]) * n[2]);
        if (depth <= 0) continue;
        for (let c = 0; c < 3; c++) p[c] += n[c] * depth;
        // u = U_s^T n, and the surface point's velocity along n is u . qd
        const o = 3 * best * r;
        let w = 0, vs = 0;
        for (let j = 0; j < r; j++) {
          u[j] = sU[o + j] * n[0] + sU[o + r + j] * n[1] + sU[o + 2 * r + j] * n[2];
          w += u[j] * u[j];
          vs += u[j] * qd[j];
        }
        const vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2] - vs;
        if (vn >= 0) continue;
        const j = (-(1 + E_BUNNY) * vn) / (1 / BALL_M + w);
        for (let c = 0; c < 3; c++) v[c] += (j / BALL_M) * n[c];
        for (let i = 0; i < r; i++) qd[i] -= j * u[i];
        // for the page: hit point (world), normal, impulse
        if (j > 0.3) this.hits.push([p[0] - R * n[0], p[1] - R * n[1], p[2] - R * n[2], n[0], n[1], n[2], j]);
      }
    }
  }

  takeHits() {
    const h = this.hits;
    this.hits = [];
    return h;
  }

  ballPositions() {
    const out = new Float32Array(3 * this.balls.length);
    this.balls.forEach((b, i) => out.set(b.p, 3 * i));
    return out;
  }
}

export const BALL = { radius: BALL_R, mass: BALL_M };
