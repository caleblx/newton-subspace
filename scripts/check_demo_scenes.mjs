// headless run of the page scenes (shake table at each stiffness, thrown balls): ear swing and
// frame cost, used to pick the shake amplitude and ball settings
// usage: node scripts/check_demo_scenes.mjs [bunny|bear]
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Body } from "../docs/demo_body.js";
import { SHAKE, shakeAccel } from "../docs/demo_shake.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const model = process.argv[2] || "bunny";
const meta = JSON.parse(fs.readFileSync(path.join(root, `docs/data/${model}.json`)));
const raw = fs.readFileSync(path.join(root, `docs/data/${model}.bin`));
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length);
const A = {};
for (const [k, a] of Object.entries(meta.arrays)) {
  const T = { float32: Float32Array, uint32: Uint32Array }[a.dtype];
  A[k] = new T(buf, a.offset, a.shape.reduce((x, y) => x * y, 1));
}
const r = meta.r, nv = meta.arrays.rest.shape[0];
let ear = 0;
for (let v = 0; v < nv; v++) if (A.rest[3 * v + 2] > A.rest[3 * ear + 2]) ear = v;
const earPos = (q) => [0, 1, 2].map((c) => {
  let x = A.rest[3 * ear + c];
  for (let j = 0; j < r; j++) x += A.U[(3 * ear + c) * r + j] * q[j];
  return x;
});

// shake table
for (const scale of SHAKE.scales) {
  let t0 = performance.now();
  const b = new Body(meta, A, scale);
  const tSettle = performance.now() - t0;
  const rest = earPos(b.solver.q);
  const restZ = A.rest[3 * ear + 2];
  const n = Math.round(SHAKE.duration * 60) + 120;
  let worst = 0, worstT = 0, cost = 0;
  const buckets = [];
  for (let f = 0; f < n; f++) {
    const t = f / 60;
    t0 = performance.now();
    b.frame({ accel: shakeAccel(t, meta.h, meta.substeps) });
    cost += performance.now() - t0;
    const p = earPos(b.solver.q);
    const d = Math.hypot(p[0] - rest[0], p[1] - rest[1], p[2] - rest[2]);
    if (d > worst) { worst = d; worstT = t; }
    const k = Math.floor(t);
    buckets[k] = Math.max(buckets[k] || 0, d);
  }
  console.log(`scale ${scale}: settle ${tSettle.toFixed(0)} ms, ear sags ${(100 * (restZ - rest[2])).toFixed(1)} cm, ` +
    `biggest swing ${(100 * worst).toFixed(1)} cm at t = ${worstT.toFixed(1)} s, ${(cost / n).toFixed(2)} ms/frame`);
  console.log("   per second (cm): " + buckets.map((x) => (100 * x).toFixed(0)).join(" "));
}

// balls
const b = new Body(meta, A, 1);
const rest = earPos(b.solver.q);
let worst = 0, cost = 0, hits = 0;
const n = 360; // six throws
for (let f = 0; f < n; f++) {
  let throws = null;
  if (f % 60 === 0) {
    // from the camera side like the page's opening throws, a bit higher each time
    const from = [0.9, -0.8, 0.45], to = [0, 0, 0.1 + 0.03 * (f / 60)];
    const T = 0.16; // flight time
    const v = [0, 1, 2].map((c) => (to[c] - from[c]) / T + (c === 2 ? 0.5 * 9.81 * T : 0));
    throws = [{ p: from, v }];
  }
  const t0 = performance.now();
  b.frame({ throws });
  cost += performance.now() - t0;
  // a ball bounced if it ends up moving back toward where it came from
  for (const x of b.balls) {
    x.v0 ||= x.v.slice();
    if (!x.back && x.v[0] * x.v0[0] + x.v[1] * x.v0[1] < 0) { x.back = true; hits++; }
  }
  const p = earPos(b.solver.q);
  worst = Math.max(worst, Math.hypot(p[0] - rest[0], p[1] - rest[1], p[2] - rest[2]));
}
const balls = b.ballPositions();
console.log(`balls: ${hits} of 6 bounced back off the model, ear swings up to ${(100 * worst).toFixed(1)} cm, ` +
  `${(cost / n).toFixed(2)} ms/frame, ${balls.length / 3} balls left, lowest z ${Math.min(...Array.from(balls).filter((_, i) => i % 3 === 2)).toFixed(3)}`);
