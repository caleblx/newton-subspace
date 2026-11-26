// Port of the reduced solver in newton_subspace/solver.py for the live demo. Same energy (stable
// Neo-Hookean in H = F - I plus VBD's damping), backward Euler, one Newton iteration per step and
// a line search over a few step sizes. Data from scripts/build_demo.py; checked against Python by
// scripts/check_demo_js.mjs.
//
// scale multiplies mu, lambda and the damping together. The basis stays valid: scaling K by a
// constant leaves its eigenvectors, the modal derivatives and the gravity direction where they
// were, so one basis serves a soft copy and a stiff one.
//
// hessianEvery > 1 reuses the reduced Hessian and its factor for that many steps. off by default:
// on the bunny, rebuilding once per frame instead of every step changed the trajectory by ~3%,
// since the modal-derivative directions are stiff.
const ALPHAS = [1, 0.5, 0.25, 0.125, 0];

export class ReducedSolver {
  constructor(meta, B, scale = 1) {
    this.r = meta.r;
    this.nc = meta.n_cubature;
    this.B = Float64Array.from(B); // (c, 9, r), column-major vec(F - I) = B q
    this.w = Float64Array.from(meta.weight);
    this.mu = scale * meta.mu;
    this.lam = scale * meta.lam_nh;
    this.kd = scale * meta.kd;
    this.fg = Float64Array.from(meta.fg);
    this.hessianEvery = 1;
    const r = this.r, nc = this.nc;
    this.fext = new Float64Array(r); // extra reduced force per unit mass, e.g. from a shaking floor
    this.q0 = Float64Array.from(meta.q0); // resting under gravity
    if (scale !== 1) for (let j = 0; j < r; j++) this.q0[j] /= scale; // first guess, see settle()
    this.q = Float64Array.from(this.q0);
    this.qd = new Float64Array(r);
    this.y = new Float64Array(r);
    this.d = new Float64Array(r);
    this.g = new Float64Array(r);
    this.grad = new Float64Array(r);
    this.K = new Float64Array(r * r); // reduced elastic Hessian
    this.A = new Float64Array(r * r);
    this.L = new Float64Array(r * r);
    this.Hv = new Float64Array(9 * nc); // vec(F - I) per element
    this.Gp = new Float64Array(9 * nc); // start-of-step 2 x Green strain, row-major
    this.Dv = new Float64Array(9 * nc); // B d
    this.C = new Float64Array(9 * r);
    this.count = 0;
    this.factored = null; // what the current factor was built for
  }

  // let it sag to rest under gravity with a few big, heavily damped steps
  settle(steps = 80, h = 1 / 20) {
    this.fext.fill(0);
    for (let i = 0; i < steps; i++) {
      this.step(h, null);
      this.qd.fill(0);
    }
    this.q0.set(this.q);
    this.reset();
  }

  reset() {
    this.q.set(this.q0);
    this.qd.fill(0);
    this.count = 0;
    this.factored = null;
  }

  // spring: { U: Float64Array(3 r), X: [3], target: [3], k } or null
  step(h, spring) {
    const { r, nc, B, w, q, qd, y, g, C, K } = this;
    const inv_h = 1 / h, inv_h2 = inv_h * inv_h;
    const rebuild = this.count % this.hessianEvery === 0;
    this.count++;
    for (let j = 0; j < r; j++) y[j] = q[j] + h * qd[j] + h * h * (this.fg[j] + this.fext[j]);
    this.gemv(q, this.Hv);
    g.fill(0);
    if (rebuild) K.fill(0);
    for (let e = 0; e < nc; e++) {
      vecToMat(this.Hv, 9 * e, _H);
      green2(_H, this.Gp, 9 * e);
      stress(_H, this.Gp, 9 * e, this.mu, this.lam, this.kd, inv_h, _P);
      const be = 9 * r * e, we = w[e];
      // g += B_e^T vec(P)
      for (let a = 0; a < 9; a++) {
        const pa = we * _P[(a % 3) * 3 + ((a / 3) | 0)];
        const row = be + a * r;
        for (let j = 0; j < r; j++) g[j] += B[row + j] * pa;
      }
      if (!rebuild) continue;
      // 9 x 9 element Hessian from the 9 unit directions
      for (let k = 0; k < 9; k++) {
        _dF.fill(0);
        _dF[(k % 3) * 3 + ((k / 3) | 0)] = 1;
        hessApply(_H, _dF, this.mu, this.lam, this.kd, inv_h, _out);
        for (let a = 0; a < 9; a++) _He[a * 9 + k] = we * _out[(a % 3) * 3 + ((a / 3) | 0)];
      }
      // C = He B_e, then K += B_e^T C (upper triangle)
      C.fill(0);
      for (let a = 0; a < 9; a++) {
        const ca = a * r;
        for (let b = 0; b < 9; b++) {
          const hab = _He[a * 9 + b];
          if (hab === 0) continue;
          const row = be + b * r;
          for (let j = 0; j < r; j++) C[ca + j] += hab * B[row + j];
        }
      }
      for (let a = 0; a < 9; a++) {
        const row = be + a * r, ca = a * r;
        for (let i = 0; i < r; i++) {
          const bai = B[row + i];
          if (bai === 0) continue;
          const Ki = i * r;
          for (let j = i; j < r; j++) K[Ki + j] += bai * C[ca + j];
        }
      }
    }
    if (rebuild) for (let i = 0; i < r; i++) for (let j = 0; j < i; j++) K[i * r + j] = K[j * r + i];
    const grad = this.grad;
    for (let j = 0; j < r; j++) grad[j] = g[j] + inv_h2 * (q[j] - y[j]);
    let sres = null;
    if (spring) {
      const { U, X, target, k } = spring;
      sres = _sres;
      for (let c = 0; c < 3; c++) {
        let x = X[c] - target[c];
        for (let j = 0; j < r; j++) x += U[c * r + j] * q[j];
        sres[c] = x;
        for (let i = 0; i < r; i++) grad[i] += k * U[c * r + i] * x;
      }
    }
    // refactor when the Hessian, the step size or the grabbed point changed
    const key = spring ? spring.U : null;
    if (rebuild || !this.factored || this.factored.key !== key || this.factored.h !== h) {
      this.assemble(inv_h2, spring);
      // after a violent yank the Hessian can go indefinite: shift the diagonal until it factors,
      // which keeps d a descent direction for the line search
      let shift = 0;
      while (!this.cholesky()) {
        shift = shift ? 4 * shift : inv_h2;
        this.assemble(inv_h2 + shift, spring);
      }
      this.factored = { key, h };
    }
    this.cholSolve(grad, this.d);
    // line search on the total incremental potential
    this.gemv(this.d, this.Dv);
    let bestE = Infinity, bestA = 0;
    for (const a of ALPHAS) {
      let E = 0;
      for (let e = 0; e < nc; e++) {
        const o = 9 * e;
        for (let m = 0; m < 9; m++) _v[m] = this.Hv[o + m] - a * this.Dv[o + m];
        vecToMat(_v, 0, _H);
        E += w[e] * energy(_H, this.Gp, o, this.mu, this.lam, this.kd, inv_h);
      }
      for (let j = 0; j < r; j++) {
        const dj = q[j] - a * this.d[j] - y[j];
        E += 0.5 * inv_h2 * dj * dj;
      }
      if (spring) {
        const { U, k } = spring;
        for (let c = 0; c < 3; c++) {
          let x = sres[c];
          for (let j = 0; j < r; j++) x -= a * U[c * r + j] * this.d[j];
          E += 0.5 * k * x * x;
        }
      }
      if (E < bestE) { bestE = E; bestA = a; }
    }
    for (let j = 0; j < r; j++) {
      const qn = q[j] - bestA * this.d[j];
      qd[j] = (qn - q[j]) * inv_h;
      q[j] = qn;
    }
    return bestA;
  }

  assemble(inv_h2, spring) {
    const { r, K, A } = this;
    A.set(K);
    for (let i = 0; i < r; i++) A[i * r + i] += inv_h2;
    if (spring) {
      const { U, k } = spring;
      for (let c = 0; c < 3; c++) {
        for (let i = 0; i < r; i++) {
          const ui = k * U[c * r + i];
          for (let j = 0; j < r; j++) A[i * r + j] += ui * U[c * r + j];
        }
      }
    }
  }

  gemv(x, out) {
    const { r, B } = this;
    for (let i = 0; i < 9 * this.nc; i++) {
      let s = 0;
      const row = i * r;
      for (let j = 0; j < r; j++) s += B[row + j] * x[j];
      out[i] = s;
    }
  }

  cholesky() {
    const { r, A, L } = this;
    for (let i = 0; i < r; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * r + j];
        for (let k = 0; k < j; k++) s -= L[i * r + k] * L[j * r + k];
        if (i === j) {
          if (!(s > 0)) return false;
          L[i * r + i] = Math.sqrt(s);
        } else {
          L[i * r + j] = s / L[j * r + j];
        }
      }
    }
    return true;
  }

  cholSolve(b, x) {
    const { r, L } = this;
    for (let i = 0; i < r; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i * r + k] * x[k];
      x[i] = s / L[i * r + i];
    }
    for (let i = r - 1; i >= 0; i--) {
      let s = x[i];
      for (let k = i + 1; k < r; k++) s -= L[k * r + i] * x[k];
      x[i] = s / L[i * r + i];
    }
  }
}

// 3x3 matrices below are row-major Float64Array(9); vec(F) is column-major (index 3 j + i).
// shared scratch so the inner loops don't allocate
const _H = new Float64Array(9), _P = new Float64Array(9), _He = new Float64Array(81);
const _dF = new Float64Array(9), _out = new Float64Array(9), _v = new Float64Array(9);
const _E = new Float64Array(9), _cH = new Float64Array(9), _F = new Float64Array(9);
const _cF = new Float64Array(9), _dc = new Float64Array(9), _S = new Float64Array(9);
const _sres = new Float64Array(3);
let _tr = 0, _nl = 0;

function vecToMat(v, o, M) {
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[3 * i + j] = v[o + 3 * j + i];
}

function det3(M) {
  return M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

// out[:, jo] += A[:, ja] x B[:, jb]
function colCross(A, ja, B, jb, out, jo) {
  const ax = A[ja], ay = A[3 + ja], az = A[6 + ja];
  const bx = B[jb], by = B[3 + jb], bz = B[6 + jb];
  out[jo] += ay * bz - az * by;
  out[3 + jo] += az * bx - ax * bz;
  out[6 + jo] += ax * by - ay * bx;
}

// columns of cof(M) are cross products of the columns of M
function cof3(M, out) {
  out.fill(0);
  colCross(M, 1, M, 2, out, 0);
  colCross(M, 2, M, 0, out, 1);
  colCross(M, 0, M, 1, out, 2);
}

// 2 x Green strain, H + H^T + H^T H, written into G at offset o
function green2(H, G, o) {
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      G[o + 3 * i + j] = H[3 * i + j] + H[3 * j + i] + H[i] * H[j] + H[3 + i] * H[3 + j] + H[6 + i] * H[6 + j];
    }
  }
}

// tr H and the nonlinear part of J - 1, i2(H) + det H
function invariants(H) {
  const tr = H[0] + H[4] + H[8];
  let trHH = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) trHH += H[3 * i + j] * H[3 * j + i];
  _tr = tr;
  _nl = 0.5 * (tr * tr - trHH) + det3(H);
}

function energy(H, Gp, o, mu, lam, kd, inv_h) {
  invariants(H);
  const d = _tr + _nl;
  green2(H, _E, 0);
  let hh = 0, ee = 0;
  for (let m = 0; m < 9; m++) {
    hh += H[m] * H[m];
    const x = _E[m] - Gp[o + m];
    ee += x * x;
  }
  return 0.5 * mu * hh + 0.5 * lam * d * d - mu * _nl + 0.5 * kd * inv_h * ee;
}

function stress(H, Gp, o, mu, lam, kd, inv_h, P) {
  invariants(H);
  const tr = _tr, d = _tr + _nl;
  cof3(H, _cH);
  green2(H, _E, 0);
  for (let m = 0; m < 9; m++) _E[m] -= Gp[o + m];
  const c = 2 * kd * inv_h;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const I = i === j ? 1 : 0;
      const cF = (1 + tr) * I - H[3 * j + i] + _cH[3 * i + j];
      const p = mu * (H[3 * i + j] + H[3 * j + i] - tr * I - _cH[3 * i + j]) + lam * d * cF;
      // (2 kd / h) F E
      const fe = (H[3 * i] + (i === 0 ? 1 : 0)) * _E[j] + (H[3 * i + 1] + (i === 1 ? 1 : 0)) * _E[3 + j]
        + (H[3 * i + 2] + (i === 2 ? 1 : 0)) * _E[6 + j];
      P[3 * i + j] = p + c * fe;
    }
  }
}

function hessApply(H, dF, mu, lam, kd, inv_h, out) {
  for (let m = 0; m < 9; m++) _F[m] = H[m] + (m % 4 === 0 ? 1 : 0);
  invariants(H);
  const s = lam * (_tr + _nl) - mu;
  cof3(_F, _cF);
  let cd = 0;
  for (let m = 0; m < 9; m++) cd += _cF[m] * dF[m];
  // derivative of cof(F) along dF
  _dc.fill(0);
  colCross(dF, 1, _F, 2, _dc, 0); colCross(_F, 1, dF, 2, _dc, 0);
  colCross(dF, 2, _F, 0, _dc, 1); colCross(_F, 2, dF, 0, _dc, 1);
  colCross(dF, 0, _F, 1, _dc, 2); colCross(_F, 0, dF, 1, _dc, 2);
  for (let m = 0; m < 9; m++) out[m] = mu * dF[m] + lam * cd * _cF[m] + s * _dc[m];
  if (kd > 0) {
    // (2 kd / h) F (dF^T F + F^T dF)
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        _S[3 * i + j] = dF[i] * _F[j] + dF[3 + i] * _F[3 + j] + dF[6 + i] * _F[6 + j]
          + _F[i] * dF[j] + _F[3 + i] * dF[3 + j] + _F[6 + i] * dF[6 + j];
      }
    }
    const c = 2 * kd * inv_h;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        out[3 * i + j] += c * (_F[3 * i] * _S[j] + _F[3 * i + 1] * _S[3 + j] + _F[3 * i + 2] * _S[6 + j]);
      }
    }
  }
}
