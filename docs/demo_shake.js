// Shake table: floor displacement along x with a frequency sweep from f0 to f1. A copy with
// stiffness scale s has its first mode at about 2 Hz * sqrt(s).
export const SHAKE = {
  scales: [0.35, 1, 3], // soft, the original, stiff
  f0: 0.7,
  f1: 4.4,
  duration: 12, // s
  amp: 0.009, // m at 1.2 Hz; grows with sqrt(f) so the stiff, better-damped copy still gets going
};

function phase(t) {
  const { f0, f1, duration } = SHAKE;
  return 2 * Math.PI * (f0 * t + (0.5 * (f1 - f0) * t * t) / duration);
}

function envelope(t) {
  const { duration } = SHAKE;
  if (t <= 0 || t >= duration) return 0;
  const ramp = 0.8;
  return Math.min(1, t / ramp, (duration - t) / ramp);
}

export function shakeDisp(t) {
  return SHAKE.amp * Math.sqrt(shakeFreq(t) / 1.2) * envelope(t) * Math.sin(phase(t));
}

export function shakeFreq(t) {
  const { f0, f1, duration } = SHAKE;
  return f0 + ((f1 - f0) * Math.min(Math.max(t, 0), duration)) / duration;
}

// floor acceleration at each substep of the 60 Hz frame starting at t, as [[ax, ay], ...]
export function shakeAccel(t, h, substeps) {
  const out = [], e = 1e-3;
  for (let k = 1; k <= substeps; k++) {
    const s = t + k * h;
    out.push([(shakeDisp(s + e) - 2 * shakeDisp(s) + shakeDisp(s - e)) / (e * e), 0]);
  }
  return out;
}
