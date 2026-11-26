// Live demo at the top of the page. Three scenes: point loads, balls thrown at the bunny, and a
// shake table with three bunnies of different stiffness. Each bunny steps in its own worker
// (demo_worker.js); this file only draws, expanding the 30 coordinates with rest + U q.
// ?model=bear loads the bear instead (scripts/build_demo.py bear).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { Reflector } from "three/addons/objects/Reflector.js";
import { SHAKE, shakeAccel, shakeDisp, shakeFreq } from "./demo_shake.js";
import { BALL } from "./demo_body.js";

const MAX_PULL = 0.18; // m, max distance of the grab target from the grabbed point's rest position
const TABLE_H = 0.03;
const BG = 0x16181c;
const COLORS = { 0.35: 0xf3b3a1, 1: 0xe0664a, 3: 0x8d98a5 };

const MODEL = new URLSearchParams(location.search).get("model") === "bear" ? "bear" : "bunny";
const NOUN = { bear: ["bear", "bears"], bunny: ["bunny", "bunnies"] }[MODEL];
const el = (id) => document.getElementById(id);
const root = el("live");
const canvas = el("live-stage");

async function load() {
  const [meta, buf] = await Promise.all([
    fetch(`data/${MODEL}.json`).then((r) => r.json()),
    fetch(`data/${MODEL}.bin`).then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }),
  ]);
  const A = {};
  for (const [k, a] of Object.entries(meta.arrays)) {
    const T = { float32: Float32Array, uint32: Uint32Array }[a.dtype];
    A[k] = new T(buf, a.offset, a.shape.reduce((x, y) => x * y, 1));
  }
  return { meta, buf, A };
}

function gridTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#24272d";
  g.fillRect(0, 0, 256, 256);
  g.strokeStyle = "#33373f";
  g.lineWidth = 2;
  g.strokeRect(0, 0, 256, 256);
  g.strokeStyle = "#2c3036";
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(128, 0); g.lineTo(128, 256);
  g.moveTo(0, 128); g.lineTo(256, 128);
  g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(40, 40); // 0.25 m cells on a 10 m floor
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function main({ meta, buf, A }) {
  const r = meta.r;
  const nv = meta.arrays.rest.shape[0];
  const rest = A.rest, Ur = A.U;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < nv; v++) for (let c = 0; c < 3; c++) {
    lo[c] = Math.min(lo[c], rest[3 * v + c]);
    hi[c] = Math.max(hi[c], rest[3 * v + c]);
  }
  const spacing = 1.35 * (hi[0] - lo[0]);
  const depth = hi[1] - lo[1];
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.Fog(BG, 2.5, 7);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  const camera = new THREE.PerspectiveCamera(30, 2, 0.02, 30);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableZoom = false; // fixed zoom; the wheel always scrolls the page
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.mouseButtons = { LEFT: -1, MIDDLE: -1, RIGHT: THREE.MOUSE.ROTATE }; // left is for grabbing and throwing
  controls.autoRotateSpeed = -0.5; // slow drift during autoplay
  controls.minDistance = 0.4;
  controls.maxDistance = 4;
  controls.maxPolarAngle = Math.PI * 0.48;

  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.position.set(0.8, -0.6, 1.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: 0.2, far: 4 });
  sun.shadow.radius = 5;
  sun.shadow.bias = -0.0005;
  scene.add(sun);

  // mirror under a mostly opaque grid, for a faint reflection
  const mirror = new Reflector(new THREE.PlaneGeometry(10, 10), {
    textureWidth: 1024, textureHeight: 1024, color: 0x6a6e76,
  });
  mirror.position.z = -0.0005;
  scene.add(mirror);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(10, 10),
    new THREE.MeshStandardMaterial({ map: gridTexture(), roughness: 0.75, transparent: true, opacity: 0.86 })
  );
  floor.receiveShadow = true;
  scene.add(floor);

  const table = new THREE.Mesh(
    new THREE.BoxGeometry(3 * spacing + 0.1, depth + 0.12, TABLE_H),
    new THREE.MeshStandardMaterial({ color: 0x3a3f47, roughness: 0.4, metalness: 0.3 })
  );
  table.position.z = TABLE_H / 2;
  table.castShadow = table.receiveShadow = true;
  table.visible = false;
  scene.add(table);

  const index = new THREE.BufferAttribute(A.faces, 1);
  const bodies = [];

  function makeBody(scale) {
    const pos = new Float32Array(rest);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(index);
    geo.computeVertexNormals();
    const mat = new THREE.MeshPhysicalMaterial({
      color: COLORS[scale], roughness: 0.42, clearcoat: 0.6, clearcoatRoughness: 0.3, sheen: 0.4, sheenColor: 0xffffff,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    mesh.visible = false;
    scene.add(mesh);
    const b = { scale, mesh, geo, pos, mat, q: null, ready: false, busy: false, ms: 0 };
    b.worker = new Worker(new URL("./demo_worker.js", import.meta.url), { type: "module" });
    b.worker.onmessage = ({ data: m }) => {
      if (m.type === "ready") { b.ready = true; b.q = m.q; b.q0 = m.q.slice(); b.dirty = true; return; }
      b.busy = false;
      b.q = m.q;
      b.balls = m.balls;
      if (m.hits.length) onHits(b, m.hits);
      b.ms = m.ms;
      b.dirty = true;
    };
    b.worker.onerror = (e) => fail(e.message || "worker failed");
    b.worker.postMessage({ type: "init", meta, buf: buf.slice(0), scale, offset: [0, 0] });
    bodies.push(b);
    return b;
  }
  const main0 = makeBody(1);
  let sideBodies = null; // made on first visit to the shake table

  function expand(restArr, U, n, out, q) {
    for (let v = 0; v < n; v++) {
      for (let c = 0; c < 3; c++) {
        const o = (3 * v + c) * r;
        let s = restArr[3 * v + c];
        for (let j = 0; j < r; j++) s += U[o + j] * q[j];
        out[3 * v + c] = s;
      }
    }
  }

  // full tet mesh as a wireframe, and the cubature tets the solver actually evaluates
  const nn = meta.arrays.tet_rest.shape[0];
  const tetPos = new Float32Array(3 * nn);
  const wgeo = new THREE.BufferGeometry();
  wgeo.setAttribute("position", new THREE.BufferAttribute(tetPos, 3));
  wgeo.setIndex(new THREE.BufferAttribute(A.tet_edges, 1));
  const wire = new THREE.LineSegments(wgeo, new THREE.LineBasicMaterial({ color: 0xaab3bd, transparent: true, opacity: 0.22 }));
  wire.frustumCulled = false;
  wire.visible = false;
  scene.add(wire);
  const nc = meta.n_cubature;
  const cubPos = new Float32Array(nc * 12);
  const cubIdx = new Uint32Array(nc * 12);
  for (let e = 0; e < nc; e++) {
    const b = 4 * e;
    cubIdx.set([b, b + 2, b + 1, b, b + 1, b + 3, b, b + 3, b + 2, b + 1, b + 2, b + 3], 12 * e);
  }
  const cgeo = new THREE.BufferGeometry();
  cgeo.setAttribute("position", new THREE.BufferAttribute(cubPos, 3));
  cgeo.setIndex(new THREE.BufferAttribute(cubIdx, 1));
  const cub = new THREE.Mesh(cgeo, new THREE.MeshBasicMaterial({ color: 0xff6a3d, side: THREE.DoubleSide }));
  cub.frustumCulled = false;
  cub.visible = false;
  scene.add(cub);

  const MAXB = 12;
  const balls = new THREE.InstancedMesh(
    new THREE.SphereGeometry(BALL.radius, 32, 16),
    new THREE.MeshStandardMaterial({ color: 0xe8e8e8, metalness: 1, roughness: 0.12 }),
    MAXB
  );
  balls.castShadow = true;
  balls.count = 0;
  balls.frustumCulled = false;
  scene.add(balls);
  const _m = new THREE.Matrix4();

  // rubber band from the grabbed point to the cursor
  const bandGeo = new THREE.BufferGeometry();
  bandGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
  const band = new THREE.Line(bandGeo, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
  band.renderOrder = 2;
  band.visible = false;
  scene.add(band);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.008, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
  knob.renderOrder = 2;
  knob.visible = false;
  scene.add(knob);

  const f1 = meta.modes_hz[0];
  const SCENES = [
    {
      title: "Point loads",
      text: `${meta.n_tets.toLocaleString()} tets, ${meta.n_dofs.toLocaleString()} DOFs, simulated in a ${r}-dimensional ` +
        `subspace with ${nc} cubature tets.`,
      cam: [[0.95, -0.85, 0.5], [0, 0, 0.17]],
      tools: ["live-tets", "live-cub"],
    },
    {
      title: "Impacts",
      text: `${BALL.mass} kg steel balls. Each hit is an impulse in the reduced coordinates.`,
      cam: [[1.05, -1.0, 0.55], [0, 0, 0.17]],
      tools: [],
    },
    {
      title: "Shake table",
      text: `The same basis at 0.35x, 1x and 3x stiffness. The table sweeps from ${SHAKE.f0} to ${SHAKE.f1} Hz.`,
      cam: [[0.0, -2.05, 0.7], [0, 0, 0.16]],
      shift: 0.6,
      tools: ["live-shake"],
    },
  ];
  const POINT = 0, IMPACTS = 1, TABLE = 2;
  // camera pose for a scene. With `keep` (the previous scene) the current viewing direction and
  // relative zoom are kept and only target and distance change. Pulled back on narrow screens;
  // on wide ones the subject is nudged right of the caption card.
  function sceneDist(S) {
    const w = canvas.clientWidth || 16, h = canvas.clientHeight || 9, aspect = w / h;
    const back = aspect < 1 ? 1.75 : aspect < 1.5 ? 1.45 : 1.25;
    return back * new THREE.Vector3(...S.cam[0]).distanceTo(new THREE.Vector3(...S.cam[1]));
  }
  function camPose(S, keep = null) {
    const tgt = new THREE.Vector3(...S.cam[1]);
    let dir, dist = sceneDist(S);
    if (keep) {
      dir = camera.position.clone().sub(controls.target).normalize();
      dist *= camera.position.distanceTo(controls.target) / sceneDist(keep);
    } else dir = new THREE.Vector3(...S.cam[0]).sub(tgt).normalize();
    const pos = tgt.clone().addScaledVector(dir, dist);
    const w = canvas.clientWidth || 16, h = canvas.clientHeight || 9;
    if (w / h >= 1.5 && w > 900) {
      const left = new THREE.Vector3(0, 0, 1).cross(dir.clone().negate()).normalize(); // screen left
      const shift = 0.1 * dist * (S.shift ?? 1);
      pos.addScaledVector(left, shift);
      tgt.addScaledVector(left, shift);
    }
    return [pos, tgt];
  }
  let cur = -1;
  let camFrom = null, camTo = null, camT = 1;
  let shakeT = -1; // seconds into the sweep, -1 when not shaking
  let userShake = false; // the running sweep was started with the button, not by autoplay
  const labels = el("live-labels");

  function active() {
    return cur === TABLE ? [sideBodies[0], main0, sideBodies[1]] : [main0];
  }
  function cardFor(i) {
    const S = SCENES[i];
    el("live-text").textContent = S.text;
    el("live-sweep").hidden = i !== TABLE;
    for (const id of ["live-tets", "live-cub", "live-shake"]) el(id).hidden = !S.tools.includes(id);
  }
  function setScene(i) {
    i = (i + SCENES.length) % SCENES.length;
    if (i === cur) return;
    yankT = -1;
    grab = null;
    controls.enabled = true;
    const prev = cur;
    cur = i;
    const S = SCENES[i];
    if (i === TABLE && !sideBodies) sideBodies = [makeBody(SHAKE.scales[0]), makeBody(SHAKE.scales[2])];
    for (const b of bodies) { b.mesh.visible = false; b.worker.postMessage({ type: "reset" }); b.balls = null; }
    const act = active();
    act.forEach((b, k) => {
      b.mesh.visible = true;
      b.x = act.length === 1 ? 0 : (k - 1) * spacing;
      b.z = i === TABLE ? TABLE_H : 0;
    });
    table.visible = i === TABLE;
    wire.visible = cub.visible = false;
    setXray();
    labels.hidden = i !== TABLE;
    shakeT = i === TABLE && auto ? -0.6 : -1;
    userShake = false;
    cue = 0;
    sceneClock = 0;
    cardFor(i);
    sceneBtns.forEach((b, k) => {
      b.setAttribute("aria-current", String(k === i));
      b.firstChild.style.width = "0";
    });
    canvas.style.cursor = i === IMPACTS ? "crosshair" : "";
    showHints(i);
    if (prev >= 0) {
      camFrom = [camera.position.clone(), controls.target.clone()];
      camTo = camPose(S, SCENES[prev]);
      camT = 0;
    }
  }

  // control hints in the corner, per scene and input type
  const TOUCH = matchMedia("(pointer: coarse)").matches;
  const mouseIcon = (btn) => {
    const fill = (on) => (on ? "currentColor" : "none");
    return `<svg width="14" height="20" viewBox="0 0 14 20" fill="none" stroke="currentColor" stroke-width="1.2">` +
      `<path d="M7 1.5v6M1.5 7.5h11" />` +
      `<path d="M7 1.5H6.5A5 5 0 0 0 1.5 6.5v1h5.5z" fill="${fill(btn === "L")}" />` +
      `<path d="M7 1.5h.5a5 5 0 0 1 5 5v1H7z" fill="${fill(btn === "R")}" />` +
      `<rect x="1.5" y="1.5" width="11" height="17" rx="5.5" /></svg>`;
  };
  const fingerIcon = `<svg width="16" height="20" viewBox="0 0 16 20" fill="none" stroke="currentColor" stroke-width="1.2" ` +
    `stroke-linecap="round" stroke-linejoin="round"><path d="M6 10V3a1.5 1.5 0 0 1 3 0v5l3.6.8a2 2 0 0 1 1.5 2.3L13.3 16` +
    `a2.5 2.5 0 0 1-2.5 2H8.3a2.5 2.5 0 0 1-2-1L3 13a1.4 1.4 0 0 1 2-2l1 1" /></svg>`;
  function showHints(i) {
    const rows = TOUCH
      ? (i === IMPACTS ? [[fingerIcon, "tap to throw"], [fingerIcon, "drag to rotate"]]
        : [[fingerIcon, `drag the ${NOUN[i === TABLE ? 1 : 0]} to pull`], [fingerIcon, "drag elsewhere to rotate"]])
      : [i === IMPACTS ? [mouseIcon("L"), "throw"] : [mouseIcon("L"), "pull"], [mouseIcon("R"), "rotate"],
        ["<kbd>&larr;</kbd><kbd>&rarr;</kbd>", "scene"]];
    el("live-hints").innerHTML = rows.map(([ic, t]) => `<span class="ic">${ic}</span><span>${t}</span>`).join("");
  }

  let grab = null; // { body, v, U, X, target, k } in the body's own coordinates
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const plane = new THREE.Plane();
  const hit = new THREE.Vector3();

  function setNdc(ev) {
    const b = canvas.getBoundingClientRect();
    ndc.set(((ev.clientX - b.left) / b.width) * 2 - 1, -((ev.clientY - b.top) / b.height) * 2 + 1);
  }

  function grabVertex(body, v, point) {
    const U = new Float64Array(3 * r);
    for (let c = 0; c < 3; c++) for (let j = 0; j < r; j++) U[c * r + j] = Ur[(3 * v + c) * r + j];
    const X = [rest[3 * v], rest[3 * v + 1], rest[3 * v + 2]];
    plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), point);
    grab = { body, v, U, X, k: meta.spring_k, target: null };
    moveTarget(point);
  }

  function moveTarget(p) {
    const b = grab.body;
    const d = [p.x - b.mesh.position.x - grab.X[0], p.y - b.mesh.position.y - grab.X[1], p.z - b.mesh.position.z - grab.X[2]];
    const len = Math.hypot(...d);
    const s = len > MAX_PULL ? MAX_PULL / len : 1;
    grab.target = [grab.X[0] + s * d[0], grab.X[1] + s * d[1], Math.max(0.005, grab.X[2] + s * d[2])];
  }

  // left button or a finger grabs (throws in the impacts scene); only the right button orbits
  let down = null;
  canvas.addEventListener("contextmenu", (ev) => ev.preventDefault());
  canvas.addEventListener("pointerdown", (ev) => {
    if (ev.pointerType === "mouse" && ev.button !== 0) return;
    down = { x: ev.clientX, y: ev.clientY, t: performance.now() };
    if (cur === IMPACTS) return;
    setNdc(ev);
    ray.setFromCamera(ndc, camera);
    const meshes = active().map((b) => b.mesh);
    for (const m of meshes) m.geometry.computeBoundingSphere();
    const h = ray.intersectObjects(meshes, false)[0];
    if (!h) return; // empty space: camera turns, autoplay carries on
    takeOver();
    const body = active().find((b) => b.mesh === h.object);
    const f = h.face, pos = body.pos, lp = h.object.worldToLocal(h.point.clone());
    let best = f.a, bd = Infinity;
    for (const v of [f.a, f.b, f.c]) {
      const dd = Math.hypot(pos[3 * v] - lp.x, pos[3 * v + 1] - lp.y, pos[3 * v + 2] - lp.z);
      if (dd < bd) { bd = dd; best = v; }
    }
    controls.enabled = false;
    canvas.setPointerCapture(ev.pointerId);
    grabVertex(body, best, h.point);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!grab || grab.auto) return;
    setNdc(ev);
    ray.setFromCamera(ndc, camera);
    if (ray.ray.intersectPlane(plane, hit)) moveTarget(hit);
  });
  const release = (ev) => {
    if (grab && !grab.auto) { grab = null; controls.enabled = true; stir(); }
    if (cur === IMPACTS && down && ev.type === "pointerup" &&
        Math.hypot(ev.clientX - down.x, ev.clientY - down.y) < 6 && performance.now() - down.t < 400) {
      takeOver();
      setNdc(ev);
      throwAt(ndc);
    }
    down = null;
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  // throw from just below the camera at whatever is under the cursor
  function throwAt(at) {
    ray.setFromCamera(at, camera);
    main0.geo.computeBoundingSphere();
    const h = ray.intersectObject(main0.mesh, false)[0];
    let target;
    if (h) target = h.point.clone();
    else {
      target = ray.ray.at(1.4, new THREE.Vector3());
      if (target.z < 0.05) target.z = 0.05;
    }
    const from = camera.position.clone().add(new THREE.Vector3(0, 0, -0.12));
    const T = from.distanceTo(target) / 6.5; // flight time at 6.5 m/s
    const v = target.clone().sub(from).divideScalar(T);
    v.z += 0.5 * 9.81 * T;
    pendingThrows.push({ p: [from.x, from.y, from.z], v: [v.x, v.y, v.z] });
  }
  let pendingThrows = [];

  const fx = el("live-fx");
  const _s = new THREE.Vector3();
  function toScreen(p) {
    _s.set(p[0], p[1], p[2]).project(camera);
    return [((_s.x + 1) / 2) * canvas.clientWidth, ((1 - _s.y) / 2) * canvas.clientHeight];
  }
  const world = (b, v) => [b.pos[3 * v] + b.mesh.position.x, b.pos[3 * v + 1] + b.mesh.position.y, b.pos[3 * v + 2] + b.mesh.position.z];

  // rest shape drawn faintly, so it only shows where the body has moved out of it
  const ghostPos = new Float32Array(rest);
  const ghostGeo = new THREE.BufferGeometry();
  ghostGeo.setAttribute("position", new THREE.BufferAttribute(ghostPos, 3));
  ghostGeo.setIndex(index);
  const ghost = new THREE.Mesh(ghostGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false }));
  ghost.renderOrder = 1;
  ghost.frustumCulled = false;
  ghost.visible = false;
  scene.add(ghost);
  let ghostReady = false, ghostFade = 0, stirT = 0;
  const stir = () => { stirT = 1.8; };

  const pullTag = document.createElement("div");
  pullTag.className = "pull";
  pullTag.hidden = true;
  fx.appendChild(pullTag);

  const ringGeo = new THREE.RingGeometry(0.72, 1, 40);
  const rings = [];
  let lastHit = -1;
  function onHits(b, hits) {
    if (b !== main0 || cur !== IMPACTS) return;
    // only real hits, not a ball rolling or resting against it
    const h = hits.reduce((a, c) => (c[6] > a[6] ? c : a));
    if (h[6] < 1.5 || Math.abs(sceneClock - lastHit) < 0.3) return;
    lastHit = sceneClock;
    const m = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, depthTest: false, side: THREE.DoubleSide }));
    m.renderOrder = 3;
    m.position.set(h[0], h[1], h[2]);
    m.lookAt(h[0] + h[3], h[1] + h[4], h[2] + h[5]);
    m.userData.t = 0;
    scene.add(m);
    rings.push(m);
    stir();
  }

  // Autoplay: timed cues per scene, next scene when the duration is up. Grabbing, throwing or
  // the x-ray/shake/reset buttons stop it (switching scenes doesn't). Resumes after IDLE seconds
  // unless the autoplay button turned it off.
  const IDLE = 20;
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let auto = !reduceMotion, autoOff = reduceMotion, lastInput = performance.now(), cue = 0;
  function extreme(dx, dy, dz) {
    let best = 0, bd = -Infinity;
    for (let v = 0; v < nv; v++) {
      const d = dx * rest[3 * v] + dy * rest[3 * v + 1] + dz * rest[3 * v + 2];
      if (d > bd) { bd = d; best = v; }
    }
    return best;
  }
  const top = extreme(0, 0, 1), front = extreme(0.3, -1, 0.6), side = extreme(1, 0, 0.6);

  // scripted grab: ramp the target out over 0.35 s, hold, let go
  let yankT = -1, yankD = null;
  function yank(v, d) {
    grabVertex(main0, v, new THREE.Vector3(...world(main0, v)));
    grab.auto = true;
    yankT = 0;
    yankD = d;
  }
  function tickYank(dt) {
    if (yankT < 0) return;
    if (!grab || !grab.auto) { yankT = -1; return; }
    yankT += dt;
    const u = Math.min(1, yankT / 0.35), e = u * u * (3 - 2 * u);
    grab.target = [grab.X[0] + e * yankD[0], grab.X[1] + e * yankD[1], Math.max(0.005, grab.X[2] + e * yankD[2])];
    if (yankT > 0.9) {
      grab = null;
      yankT = -1;
      stir();
    }
  }
  const yankCue = (v, d) => () => yank(v, d);
  const xrayCue = (w, c) => () => {
    wire.visible = w;
    cub.visible = c;
    setXray();
  };
  // aim at a random vertex on the upper part of the body
  function throwCue() {
    let v;
    do v = Math.floor(Math.random() * nv); while (rest[3 * v + 2] < 0.35 * meta.height);
    const p = toScreen(world(main0, v));
    throwAt(new THREE.Vector2((p[0] / canvas.clientWidth) * 2 - 1, 1 - (p[1] / canvas.clientHeight) * 2));
  }

  SCENES[POINT].dur = 10.5;
  SCENES[POINT].cues = [
    [0.4, yankCue(top, [0.12, -0.06, -0.02])],
    [2.3, yankCue(front, [0.1, 0.02, 0.05])],
    [4.0, xrayCue(true, false)],
    [4.9, yankCue(top, [-0.1, -0.06, 0])],
    [6.7, xrayCue(false, true)],
    [7.5, yankCue(side, [0, -0.11, 0.03])],
    [9.6, xrayCue(false, false)],
  ];
  SCENES[IMPACTS].dur = 8;
  SCENES[IMPACTS].cues = [0.4, 1.1, 1.8, 2.7, 3.3, 4.4, 5.0, 5.6, 6.6].map((t) => [t, throwCue]);
  SCENES[TABLE].dur = 0.6 + SHAKE.duration + 1.4;
  SCENES[TABLE].cues = [];

  function direct() {
    const S = SCENES[cur];
    if (!auto) {
      if (!autoOff && !grab && performance.now() - lastInput > IDLE * 1000) resume();
      return;
    }
    while (cue < S.cues.length && sceneClock >= S.cues[cue][0]) S.cues[cue++][1]();
    if (sceneClock >= S.dur && (cur !== TABLE || shakeT === -1)) setScene(cur + 1);
  }
  function resume() {
    auto = true;
    autoOff = false;
    if (cur === TABLE && shakeT === -1) { shakeT = -0.6; sceneClock = 0; }
    const S = SCENES[cur];
    cue = S.cues.findIndex((c) => c[0] > sceneClock);
    if (cue < 0) cue = S.cues.length;
    el("live-auto").setAttribute("aria-pressed", String(auto));
  }
  // called on any input. off=true is the autoplay button, which keeps it off until pressed again
  function takeOver(off = false) {
    lastInput = performance.now();
    if (off) autoOff = true;
    if (!auto) return;
    auto = false;
    if (cur === TABLE && !userShake) shakeT = -1;
    if (grab && grab.auto) grab = null;
    yankT = -1;
    el("live-auto").setAttribute("aria-pressed", String(auto));
  }

  // sweep gauge on the shake scene card
  const sweepPct = (f) => (100 * (f - SHAKE.f0)) / (SHAKE.f1 - SHAKE.f0);
  const sweepTicks = SHAKE.scales.map((s) => {
    const d = document.createElement("div");
    const f = f1 * Math.sqrt(s);
    d.className = "tick";
    d.style.left = `${sweepPct(f)}%`;
    d.innerHTML = `<span>${s === 1 ? "as built" : s < 1 ? "soft" : "stiff"}<br>${f.toFixed(1)} Hz</span>`;
    el("live-sweep").firstElementChild.appendChild(d);
    return d;
  });
  {
    const ends = document.createElement("div");
    ends.className = "ends";
    ends.innerHTML = `<span>${SHAKE.f0} Hz</span><span>table frequency</span><span>${SHAKE.f1} Hz</span>`;
    el("live-sweep").appendChild(ends);
  }
  function drawFx(dt) {
    if (!ghostReady && main0.q0) {
      expand(rest, Ur, nv, ghostPos, main0.q0);
      ghostGeo.attributes.position.needsUpdate = true;
      ghostReady = true;
    }
    stirT -= dt;
    const want = cur !== TABLE && ghostReady && !wire.visible && !cub.visible && ((grab && grab.body === main0) || stirT > 0) ? 1 : 0;
    ghostFade += (want - ghostFade) * (1 - Math.exp(-5 * dt));
    ghost.visible = ghostFade > 0.01;
    ghost.material.opacity = 0.14 * ghostFade;
    ghost.position.copy(main0.mesh.position);

    if (grab) {
      const v = grab.v, d = Math.hypot(grab.body.pos[3 * v] - ghostPos[3 * v], grab.body.pos[3 * v + 1] - ghostPos[3 * v + 1],
        grab.body.pos[3 * v + 2] - ghostPos[3 * v + 2]);
      const [x, y] = toScreen(knob.position.toArray());
      pullTag.textContent = `${(100 * d).toFixed(1)} cm`;
      pullTag.style.transform = `translate(${x + 14}px, ${y - 24}px)`;
    }
    pullTag.hidden = !grab || !ghostReady;

    for (let i = rings.length - 1; i >= 0; i--) {
      const m = rings[i];
      m.userData.t += dt;
      const u = m.userData.t / 0.55;
      m.scale.setScalar(0.012 + 0.07 * u);
      m.material.opacity = Math.max(0, 1 - u);
      if (u >= 1) {
        scene.remove(m);
        m.material.dispose();
        rings.splice(i, 1);
      }
    }

    if (cur === TABLE) {
      const f = shakeT > 0 ? shakeFreq(shakeT) : null;
      const at = el("live-sweep-at");
      at.style.left = `${sweepPct(f ?? SHAKE.f0)}%`;
      at.style.opacity = f === null ? 0.3 : 1;
      sweepTicks.forEach((d, i) => d.classList.toggle("on", f !== null && Math.abs(f - f1 * Math.sqrt(SHAKE.scales[i])) < 0.35));
    }
  }

  function setXray() {
    const on = wire.visible || cub.visible;
    main0.mat.transparent = on;
    main0.mat.opacity = on ? 0.16 : 1;
    main0.mat.depthWrite = !on;
    main0.mesh.castShadow = !on;
    el("live-tets").setAttribute("aria-pressed", String(wire.visible));
    el("live-cub").setAttribute("aria-pressed", String(cub.visible));
  }
  for (const id of ["live-tets", "live-cub", "live-reset"]) {
    el(id).addEventListener("click", () => takeOver());
  }
  // orbiting doesn't stop autoplay, just pauses the auto-rotate
  let orbiting = false;
  controls.addEventListener("start", () => { orbiting = true; });
  controls.addEventListener("end", () => { orbiting = false; });
  el("live-auto").addEventListener("click", () => (auto ? takeOver(true) : resume()));
  el("live-tets").addEventListener("click", () => { wire.visible = !wire.visible; setXray(); });
  el("live-cub").addEventListener("click", () => { cub.visible = !cub.visible; setXray(); });
  el("live-shake").addEventListener("click", () => {
    const running = shakeT > -1;
    takeOver();
    userShake = true;
    shakeT = running ? -1 : 0;
  });
  el("live-reset").addEventListener("click", () => {
    for (const b of active()) b.worker.postMessage({ type: "reset" });
    grab = null;
    controls.enabled = true;
    if (cur === TABLE) shakeT = -1;
  });
  document.addEventListener("keydown", (ev) => {
    if (!onScreen || ev.altKey || ev.ctrlKey || ev.metaKey || /INPUT|TEXTAREA|SELECT/.test(ev.target.tagName)) return;
    if (ev.key === "ArrowLeft") setScene(cur - 1);
    if (ev.key === "ArrowRight") setScene(cur + 1);
  });

  const labelEls = SHAKE.scales.map((s) => {
    const d = document.createElement("div");
    d.className = "tag";
    const name = s === 1 ? "as built" : s < 1 ? "soft" : "stiff";
    d.innerHTML = `${name}<br><span>${(f1 * Math.sqrt(s)).toFixed(1)} Hz</span>`;
    labels.appendChild(d);
    return d;
  });

  function fail(msg) {
    console.error(msg);
    root.classList.add("failed");
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      // keep the subject in frame on tall phone screens
      camera.fov = w / h < 1 ? 46 : 30;
      camera.updateProjectionMatrix();
    }
  }

  let onScreen = true;
  new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; }).observe(canvas);

  const FRAME = 1 / 60;
  let acc = 0, last = performance.now(), sceneClock = 0, shown = 0, simFrames = 0, simStart = performance.now();
  el("live-auto").setAttribute("aria-pressed", String(auto));
  // one button per scene; the current one fills up during autoplay, clicking one resumes it
  const sceneBtns = SCENES.map((S, k) => {
    const b = document.createElement("button");
    b.innerHTML = `<i></i>${S.title}`;
    b.addEventListener("click", () => {
      setScene(k);
      if (!auto) resume();
    });
    el("live-scenes").appendChild(b);
    return b;
  });
  el("live-prev").addEventListener("click", () => setScene(cur - 1));
  el("live-next").addEventListener("click", () => setScene(cur + 1));
  setScene(0);
  resize();
  const [p0, t0] = camPose(SCENES[0]);
  camera.position.copy(p0);
  controls.target.copy(t0);
  camT = 1;
  controls.update();

  const _p = new THREE.Vector3();
  renderer.setAnimationLoop((now) => {
    try {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      resize();
      if (!onScreen) return;
      direct();
      const act = active();
      const ready = act.every((b) => b.ready);
      if (!ready) el("live-hud-r").textContent = "starting the solver…";

      // bodies step in lockstep: the next frames go out once every worker has answered. up to
      // three frames per message, so a slow machine gets choppier rather than slow motion
      acc = Math.min(acc + dt, 3 * FRAME);
      if (ready && act.every((b) => !b.busy) && acc >= FRAME) {
        const frames = Math.floor(acc / FRAME + 1e-6);
        acc -= frames * FRAME;
        tickYank(frames * FRAME);
        sceneClock += frames * FRAME;
        let accel = null;
        if (cur === TABLE && shakeT > -1) {
          accel = [];
          for (let f = 0; f < frames; f++) {
            accel.push(shakeT >= 0 ? shakeAccel(shakeT, meta.h, meta.substeps) : null);
            shakeT += FRAME;
          }
          if (accel.every((a) => a === null)) accel = null;
          else accel = accel.map((a) => a || Array.from({ length: meta.substeps }, () => [0, 0]));
          if (shakeT > SHAKE.duration + 1.4) shakeT = -1;
        }
        let throws = null;
        if (cur === IMPACTS && pendingThrows.length) { throws = pendingThrows; pendingThrows = []; }
        for (const b of act) {
          const pull = grab && grab.body === b ? { U: grab.U, X: grab.X, target: grab.target, k: grab.k } : null;
          b.busy = true;
          b.worker.postMessage({ type: "frame", frames, pull, accel, throws: b === main0 ? throws : null });
        }
        simFrames += frames;
      }

      const disp = cur === TABLE && shakeT > 0 ? shakeDisp(shakeT) : 0;
      table.position.x = disp;
      for (const b of act) {
        b.mesh.position.set(b.x + disp, 0, b.z);
        if (b.dirty) {
          expand(rest, Ur, nv, b.pos, b.q);
          b.geo.attributes.position.needsUpdate = true;
          b.geo.computeVertexNormals();
          b.dirty = false;
          if (b === main0 && wire.visible) {
            expand(A.tet_rest, A.tet_U, nn, tetPos, b.q);
            wgeo.attributes.position.needsUpdate = true;
          }
          if (b === main0 && cub.visible) {
            expand(A.cub_rest, A.cub_U, 4 * nc, cubPos, b.q);
            cgeo.attributes.position.needsUpdate = true;
          }
        }
      }
      const bl = cur === IMPACTS && main0.balls ? main0.balls : null;
      balls.count = bl ? bl.length / 3 : 0;
      for (let i = 0; i < balls.count; i++) {
        _m.makeTranslation(bl[3 * i], bl[3 * i + 1], bl[3 * i + 2]);
        balls.setMatrixAt(i, _m);
      }
      balls.instanceMatrix.needsUpdate = true;

      if (grab) {
        const v = grab.v, b = grab.body, o = b.mesh.position;
        const p = bandGeo.attributes.position.array;
        p.set([b.pos[3 * v] + o.x, b.pos[3 * v + 1] + o.y, b.pos[3 * v + 2] + o.z,
          grab.target[0] + o.x, grab.target[1] + o.y, grab.target[2] + o.z]);
        bandGeo.attributes.position.needsUpdate = true;
        knob.position.set(grab.target[0] + o.x, grab.target[1] + o.y, grab.target[2] + o.z);
      }
      band.visible = knob.visible = !!grab;
      drawFx(dt);

      // labels over the shake-table copies light up while the sweep is near their mode
      if (cur === TABLE) {
        el("live-shake").textContent = shakeT > -1 ? "Stop" : "Shake";
        const w = canvas.clientWidth, h = canvas.clientHeight;
        const fNow = shakeT > 0 ? shakeFreq(shakeT) : null;
        act.forEach((b, k) => {
          _p.set(b.x, 0, b.z + hi[2] + 0.07).project(camera);
          const d = labelEls[k];
          d.style.transform = `translate(${((_p.x + 1) / 2) * w}px, ${((1 - _p.y) / 2) * h}px) translate(-50%, -100%)`;
          const fk = f1 * Math.sqrt(b.scale);
          d.classList.toggle("on", fNow !== null && Math.abs(fNow - fk) < 0.35);
        });
      }

      controls.autoRotate = auto && !orbiting && camT >= 1 && cur !== TABLE;
      const S0 = SCENES[cur];
      sceneBtns[cur].firstChild.style.width = auto ? `calc(${Math.min(1, sceneClock / S0.dur)} * (100% - 8px))` : "0";

      if (camT < 1) {
        camT = Math.min(1, camT + dt / 0.9);
        const e = camT * camT * (3 - 2 * camT);
        camera.position.lerpVectors(camFrom[0], camTo[0], e);
        controls.target.lerpVectors(camFrom[1], camTo[1], e);
      }

      if (now - shown > 500 && ready) {
        const ms = Math.max(...act.map((b) => b.ms));
        const rate = (simFrames / ((now - simStart) / 1000)) || 0;
        let s = `${ms.toFixed(1)} ms/frame`;
        if (rate < 50) s += ` (${Math.round((100 * rate) / 60)}% speed)`;
        if (cur === TABLE && shakeT > 0) s = `table ${shakeFreq(shakeT).toFixed(1)} Hz · ` + s;
        el("live-hud-r").textContent = s;
        simFrames = 0;
        simStart = now;
        shown = now;
      }
      controls.update();
      renderer.render(scene, camera);
    } catch (err) {
      console.error(err);
      el("live-hud-r").textContent = `error: ${err.message}`;
    }
  });

  el("live-hud-l").textContent = "";
}

if (typeof Worker === "undefined") root.classList.add("failed");
else load().then(main).catch((err) => {
  console.error(err);
  root.classList.add("failed");
});
