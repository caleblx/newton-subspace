// One Body (demo_body.js) off the main thread. The page sends a message per batch of 60 Hz frames
// with the inputs and gets back q, the ball positions, the hits and the time per frame.
import { Body } from "./demo_body.js";
let body = null;

onmessage = ({ data: m }) => {
  if (m.type === "init") {
    const { meta, buf, scale, offset } = m;
    const A = {};
    for (const [k, a] of Object.entries(meta.arrays)) {
      const T = { float32: Float32Array, uint32: Uint32Array }[a.dtype];
      A[k] = new T(buf, a.offset, a.shape.reduce((x, y) => x * y, 1));
    }
    body = new Body(meta, A, scale);
    body.offset = offset;
    postMessage({ type: "ready", q: body.solver.q });
    return;
  }
  if (m.type === "reset") {
    body.reset();
    return;
  }
  if (m.type === "frame") {
    const t0 = performance.now();
    for (let f = 0; f < m.frames; f++) {
      body.frame({ pull: m.pull, accel: m.accel ? m.accel[f] : null, throws: f === 0 ? m.throws : null });
    }
    postMessage({ type: "state", q: body.solver.q, balls: body.ballPositions(), hits: body.takeHits(), ms: (performance.now() - t0) / m.frames });
  }
};
