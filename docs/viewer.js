// Playback of the beam runs in docs/data (written by scripts/export_playback.py). Each scenario
// file is uint16 positions, shape (run, frame, vertex, 3), runs vbd, pod, modal+MD.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const RES = 3;
const ERR_FULL = 0.05; // metres of vertex error at the dark end of the color ramp
const RAMP = ["#d7dadf", "#f0a58f", "#d1462f", "#7d1d0e"].map((c) => new THREE.Color(c));
const LOOP_PAUSE = 0.6; // s held on the last frame before looping

const root = document.getElementById("viewer");
const canvas = document.getElementById("stage");
const hudL = document.getElementById("hud-l");
const hudR = document.getElementById("hud-r");
const scrub = document.getElementById("scrub");
const clock = document.getElementById("clock");
const playBtn = document.getElementById("play");
const loadbar = document.getElementById("loadbar");

let meta, faces, ghostIndex;
let mesh, ghost, positions, colors, ghostPos, refNow;
const cache = new Map();
let scenario = "twist";
let run = "modal+MD";
let data = null; // { q, frames, lo, hi }
let t = 0;
let playing = !matchMedia("(prefers-reduced-motion: reduce)").matches;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 2, 0.05, 50);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas);
controls.enableZoom = false; // fixed zoom; the wheel always scrolls the page
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 1.2;
controls.maxDistance = 8;

scene.add(new THREE.HemisphereLight(0xffffff, 0x5a5e66, 1.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(1.5, -2.5, 4);
scene.add(sun);

// the clamp
const wall = new THREE.Mesh(
  new THREE.BoxGeometry(0.02, 0.9, 0.9),
  new THREE.MeshStandardMaterial({ color: 0x4a4e56, roughness: 0.9 })
);
wall.position.set(-0.01, 0.2, 1.2);
scene.add(wall);

function decode(k, f, out) {
  const { q, lo, hi } = data;
  const n = meta.n_verts;
  const base = ((k * data.frames + f) * n) * 3;
  const s = [(hi[0] - lo[0]) / 65535, (hi[1] - lo[1]) / 65535, (hi[2] - lo[2]) / 65535];
  for (let i = 0; i < n * 3; i++) out[i] = lo[i % 3] + q[base + i] * s[i % 3];
}

const bufA = { cur: null, nxt: null, ref: null, refN: null };

function frameAt(time) {
  const fpos = Math.min(time * meta.fps, data.frames - 1);
  const f0 = Math.floor(fpos);
  return [f0, Math.min(f0 + 1, data.frames - 1), fpos - f0];
}

function lerp(a, b, w, out) {
  for (let i = 0; i < out.length; i++) out[i] = a[i] + (b[i] - a[i]) * w;
}

function rampColor(x, c) {
  const u = Math.max(0, Math.min(1, x)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(u), RAMP.length - 2);
  c.copy(RAMP[i]).lerp(RAMP[i + 1], u - i);
}

const tmpColor = new THREE.Color();
function update() {
  const k = meta.runs.indexOf(run);
  const [f0, f1, w] = frameAt(t);
  decode(k, f0, bufA.cur);
  decode(k, f1, bufA.nxt);
  lerp(bufA.cur, bufA.nxt, w, positions);
  decode(0, f0, bufA.ref);
  decode(0, f1, bufA.refN);
  lerp(bufA.ref, bufA.refN, w, refNow);

  let maxErr = 0;
  for (let v = 0; v < meta.n_verts; v++) {
    const dx = positions[3 * v] - refNow[3 * v];
    const dy = positions[3 * v + 1] - refNow[3 * v + 1];
    const dz = positions[3 * v + 2] - refNow[3 * v + 2];
    const e = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (e > maxErr) maxErr = e;
    rampColor(run === "vbd" ? 0 : e / ERR_FULL, tmpColor);
    colors[3 * v] = tmpColor.r;
    colors[3 * v + 1] = tmpColor.g;
    colors[3 * v + 2] = tmpColor.b;
  }
  for (let i = 0; i < ghostIndex.length; i++) {
    const v = ghostIndex[i];
    ghostPos[3 * i] = refNow[3 * v];
    ghostPos[3 * i + 1] = refNow[3 * v + 1];
    ghostPos[3 * i + 2] = refNow[3 * v + 2];
  }
  const g = mesh.geometry;
  g.attributes.position.needsUpdate = true;
  g.attributes.color.needsUpdate = true;
  g.computeVertexNormals();
  ghost.geometry.attributes.position.needsUpdate = true;
  ghost.visible = run !== "vbd";

  const sc = meta.scenarios[scenario];
  const name = { vbd: "full-order VBD", pod: "POD, 30 DOFs", "modal+MD": "modes + derivatives, 30 DOFs" }[run];
  hudL.innerHTML = `<strong>${name}</strong>` +
    (run === "vbd" ? ` · ${meta.n_dofs.toLocaleString()} DOFs` : ` · trajectory error ${(100 * sc.errors[run]).toFixed(1)}%`);
  hudR.textContent = run === "vbd" ? "" : `max vertex error now ${(100 * maxErr).toFixed(1)} cm`;
  clock.textContent = `${t.toFixed(2)} s`;
  scrub.value = Math.round((1000 * t) / duration());
}

function duration() {
  return (data.frames - 1) / meta.fps;
}

// outline of the reference: box edges plus rings around the beam every 0.2 m
function buildGhost(rest) {
  const n = meta.n_verts;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < n; v++) for (let c = 0; c < 3; c++) {
    lo[c] = Math.min(lo[c], rest[3 * v + c]);
    hi[c] = Math.max(hi[c], rest[3 * v + c]);
  }
  const eps = 1e-4;
  const mask = new Uint8Array(n);
  for (let v = 0; v < n; v++) for (let c = 0; c < 3; c++) {
    if (Math.abs(rest[3 * v + c] - lo[c]) < eps) mask[v] |= 1 << (2 * c);
    if (Math.abs(rest[3 * v + c] - hi[c]) < eps) mask[v] |= 1 << (2 * c + 1);
  }
  const bits = (m) => { let k = 0; while (m) { k += m & 1; m >>= 1; } return k; };
  const seen = new Set();
  const idx = [];
  for (let f = 0; f < faces.length; f += 3) {
    for (let e = 0; e < 3; e++) {
      const a = faces[f + e], b = faces[f + ((e + 1) % 3)];
      const key = a < b ? a * n + b : b * n + a;
      if (seen.has(key)) continue;
      seen.add(key);
      const boxEdge = bits(mask[a] & mask[b]) >= 2;
      const xa = rest[3 * a], xb = rest[3 * b];
      const ring = Math.abs(xa - xb) < eps && Math.abs(xa / 0.2 - Math.round(xa / 0.2)) < 1e-3 && xa > eps;
      // ring segments run along a side face; skip the diagonals on the end cap
      const common = mask[a] & mask[b];
      if (boxEdge || (ring && common && !(common & 2))) idx.push(a, b);
    }
  }
  return new Uint32Array(idx);
}

async function loadScenario(name) {
  if (!cache.has(name)) {
    hudL.textContent = "loading…";
    const buf = await fetch(`data/${name}_res${RES}.bin`).then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.arrayBuffer();
    });
    const sc = meta.scenarios[name];
    cache.set(name, { q: new Uint16Array(buf), frames: sc.frames, lo: sc.lo, hi: sc.hi });
  }
  data = cache.get(name);
  const span = duration();
  loadbar.style.width = `${(100 * Math.min(meta.scenarios[name].load_time, span)) / span}%`;
  t = Math.min(t, span);
}

function setPressed(group, value) {
  for (const b of group.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.v === value));
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

async function init() {
  meta = await fetch(`data/playback_res${RES}.json`).then((r) => r.json());
  faces = new Uint32Array(await fetch(`data/faces_res${RES}.bin`).then((r) => r.arrayBuffer()));
  const rest = new Float32Array(await fetch(`data/rest_res${RES}.bin`).then((r) => r.arrayBuffer()));
  await loadScenario(scenario);

  const n = meta.n_verts;
  positions = new Float32Array(3 * n);
  colors = new Float32Array(3 * n);
  for (const k of Object.keys(bufA)) bufA[k] = new Float32Array(3 * n);
  refNow = new Float32Array(3 * n);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, metalness: 0 }));
  // the beam moves far from where it starts, so a bounding sphere taken from one frame can cull it
  mesh.frustumCulled = false;
  scene.add(mesh);

  ghostIndex = buildGhost(rest);
  ghostPos = new Float32Array(3 * ghostIndex.length);
  const gg = new THREE.BufferGeometry();
  gg.setAttribute("position", new THREE.BufferAttribute(ghostPos, 3));
  ghost = new THREE.LineSegments(gg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8, depthTest: false }));
  ghost.renderOrder = 1;
  ghost.frustumCulled = false;
  scene.add(ghost);

  camera.position.set(2.3, -2.6, 2.0);
  controls.target.set(0.6, 0.2, 1.0);
  controls.update();

  document.getElementById("opt-load").addEventListener("click", async (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    scenario = b.dataset.v;
    setPressed(e.currentTarget, scenario);
    t = 0;
    await loadScenario(scenario);
    update();
  });
  document.getElementById("opt-run").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    run = b.dataset.v;
    setPressed(e.currentTarget, run);
    update();
  });
  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "pause" : "play";
  });
  playBtn.textContent = playing ? "pause" : "play";
  scrub.addEventListener("input", () => {
    playing = false;
    playBtn.textContent = "play";
    t = (scrub.value / 1000) * duration();
    update();
  });

  let last = performance.now(), hold = 0, failed = false;
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    hudR.textContent = "WebGL context lost, reload the page";
  });
  renderer.setAnimationLoop((now) => {
    try {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      resize();
      if (playing) {
        if (t >= duration()) {
          hold += dt;
          if (hold > LOOP_PAUSE) { t = 0; hold = 0; }
        } else {
          t = Math.min(t + dt, duration());
        }
        update();
      }
      controls.update();
      renderer.render(scene, camera);
    } catch (err) {
      // keep the loop alive and show the error
      if (!failed) console.error(err);
      failed = true;
      hudR.textContent = `viewer error: ${err.message}`;
    }
  });
  update();
}

init().catch((err) => {
  console.error(err);
  root.classList.add("failed");
});
