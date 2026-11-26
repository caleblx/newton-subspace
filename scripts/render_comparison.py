# side-by-side video of VBD and the two reduced runs colored by per-vertex error
# usage: python scripts/render_comparison.py <res> <scenario>

import subprocess
import sys
from pathlib import Path

import _common  # noqa: F401
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from newton_subspace.experiment import FRAMES, ROOT, make_model, reduced_trajectory, reference
from newton_subspace.fem import TetMesh
from newton_subspace.sim import TEST_SCENARIOS, rel_error

RES, SCENARIO = int(sys.argv[1]), sys.argv[2]
K = 1e6

model = make_model(RES, K)
mesh = TetMesh.from_model(model)
sc = next(s for s in TEST_SCENARIOS if s.name == SCENARIO)
ref = reference(model, RES, K, sc)
panels = [(f"Newton VBD, {mesh.n_free_dofs:,} DOFs", ref, None)]
for kind, lab in (("pod", "POD, 30 DOFs"), ("modal+MD", "modes + derivatives, 30 DOFs")):
    tr = reduced_trajectory(model, RES, K, kind, sc)
    panels.append((lab, tr, rel_error(tr, ref, mesh)))

# boundary triangles (faces of exactly one tet), oriented outward
faces = np.concatenate([mesh.tets[:, [0, 2, 1]], mesh.tets[:, [0, 1, 3]], mesh.tets[:, [0, 3, 2]],
                        mesh.tets[:, [1, 2, 3]]])
_, inv, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
surf = faces[cnt[inv.ravel()] == 1]

SURFACE = "#1c1e22"
for f in sorted((Path.home() / ".local/share/fonts/inter").glob("*.ttf")):
    font_manager.fontManager.addfont(str(f))
FONT = ["Inter", "Nimbus Sans", "DejaVu Sans"]
cmap = LinearSegmentedColormap.from_list("err", ["#d7dadf", "#f0a58f", "#d1462f", "#7d1d0e"])
scale = mesh.rest[:, 0].max()
err_max = 0.05 * scale  # color saturates at 5% of the beam length

media = ROOT / "docs" / "media"
frames_dir = media / f"frames_{SCENARIO}_res{RES}"
frames_dir.mkdir(parents=True, exist_ok=True)
lo = np.min([p[1].min(axis=(0, 1)) for p in panels], axis=0) - 0.05
hi = np.max([p[1].max(axis=(0, 1)) for p in panels], axis=0) + 0.05
light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])

for f in range(FRAMES):
    fig = plt.figure(figsize=(12, 3.6), facecolor=SURFACE)
    for i, (title, traj, err) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d", facecolor=SURFACE)
        tri = traj[f][surf]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
        if i == 0:
            base = np.tile(np.array([0.86, 0.87, 0.89]), (len(surf), 1))
        else:
            ve = np.linalg.norm(traj[f] - ref[f], axis=1)
            base = cmap(np.clip(ve[surf].mean(axis=1) / err_max, 0, 1))[:, :3]
        col = np.clip(base * shade[:, None], 0, 1)
        # edges drawn in the face color hide the antialiasing seams between triangles
        ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors=col, linewidths=0.3))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.2)
        # wall at the clamped end x = 0
        wy, wz = (lo[1], hi[1]), (lo[2], hi[2])
        wall = [[(0, wy[0], wz[0]), (0, wy[1], wz[0]), (0, wy[1], wz[1]), (0, wy[0], wz[1])]]
        ax.add_collection3d(Poly3DCollection(wall, facecolors=[(0.7, 0.72, 0.75, 0.16)], edgecolors="none"))
        ax.view_init(elev=18, azim=-60)
        ax.set_axis_off()
        sub = "reference" if err is None else f"error {100 * err:.1f}%"
        ax.set_title(f"{title}\n{sub}", fontsize=10, color="#e4e7eb", fontfamily=FONT, linespacing=1.35, pad=4)
    fig.text(0.012, 0.035, f"t = {f / 60:.2f} s", fontsize=9, color="#9aa1aa", fontfamily=FONT)
    fig.subplots_adjust(left=0, right=1, top=0.83, bottom=0.02, wspace=0)
    fig.savefig(frames_dir / f"{f:04d}.png", dpi=160, facecolor=SURFACE)
    if f == 45:
        fig.savefig(media / f"poster_{SCENARIO}_res{RES}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

out = media / f"compare_{SCENARIO}_res{RES}.mp4"
cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-threads", "1", "-filter_threads", "1",
       "-framerate", "60", "-i", str(frames_dir / "%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(out)]
subprocess.run(cmd, check=True)
print("wrote", out)
