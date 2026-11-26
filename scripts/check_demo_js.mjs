// docs/reduced_solver.js vs the python trajectory saved by build_demo.py, plus per-frame cost
// usage: node scripts/check_demo_js.mjs [bunny|bear]
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { ReducedSolver } from "../docs/reduced_solver.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const model = process.argv[2] || "bunny";
const meta = JSON.parse(fs.readFileSync(path.join(root, `docs/data/${model}.json`)));
const raw = fs.readFileSync(path.join(root, `docs/data/${model}.bin`));
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length);
const arr = (k) => {
  const a = meta.arrays[k];
  return new Float32Array(buf, a.offset, a.shape.reduce((x, y) => x * y, 1));
};
const chk = JSON.parse(fs.readFileSync(path.join(root, `results/cache/${model}_check.json`)));

const r = meta.r;
const tetU = arr("tet_U"), tetX = arr("tet_rest");
const U = new Float64Array(3 * r);
for (let c = 0; c < 3; c++) for (let j = 0; j < r; j++) U[c * r + j] = tetU[(3 * chk.v + c) * r + j];
const X = [0, 1, 2].map((c) => tetX[3 * chk.v + c]);

function run(every) {
  const s = new ReducedSolver(meta, arr("B"));
  s.hessianEvery = every;
  s.q0.fill(0); // the Python run starts from the undeformed shape, not the resting one the page uses
  s.reset();
  let worst = 0, t = 0, num2 = 0, den2 = 0;
  for (let f = 0; f < chk.q.length; f++) {
    const ramp = Math.min(1, f / 20);
    const spring = f < chk.hold ? { U, X, target: X.map((x, c) => x + ramp * chk.pull[c]), k: meta.spring_k } : null;
    const t0 = performance.now();
    for (let k = 0; k < chk.substeps; k++) s.step(chk.h, spring);
    t += performance.now() - t0;
    let num = 0, den = 0;
    for (let j = 0; j < r; j++) {
      num += (s.q[j] - chk.q[f][j]) ** 2;
      den += chk.q[f][j] ** 2;
    }
    num2 += num;
    den2 += den;
    if (f >= 10) worst = Math.max(worst, Math.sqrt(num / Math.max(den, 1e-30)));
  }
  console.log(`Hessian every ${every} step(s): trajectory difference from Python ${Math.sqrt(num2 / den2).toExponential(2)} (worst frame after 10: ${worst.toExponential(2)}), ` +
    `${(t / chk.q.length).toFixed(2)} ms per 60 Hz frame (${chk.substeps} steps) in Node`);
}

run(1);
// timing at the page's step settings
const s = new ReducedSolver(meta, arr("B"));
const t0 = performance.now();
for (let f = 0; f < 120; f++) for (let k = 0; k < meta.substeps; k++) s.step(meta.h, null);
console.log(`page settings (${meta.substeps} steps of 1/${Math.round(1 / meta.h)} s): ${((performance.now() - t0) / 120).toFixed(2)} ms per frame in Node`);
