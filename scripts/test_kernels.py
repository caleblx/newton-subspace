# constitutive kernels on warp's cpu device: energy vs textbook stable neo-hookean,
# stress and hessian-apply vs central differences

import _common  # noqa: F401
import numpy as np
import warp as wp

from newton_subspace.solver import element_energy, element_stress, hessian_apply

mat = wp.mat33


@wp.kernel
def eval_kernel(H: wp.array[mat], dF: wp.array[mat], Gp: wp.array[mat], mu: float, lam: float, kd: float,
                inv_h: float, psi: wp.array[float], P: wp.array[mat], HdF: wp.array[mat]):
    i = wp.tid()
    psi[i] = element_energy(H[i], Gp[i], mu, lam, kd, inv_h)
    P[i] = element_stress(H[i], Gp[i], mu, lam, kd, inv_h)
    HdF[i] = hessian_apply(H[i], dF[i], mu, lam, kd, inv_h)


def run(H, dF, Gp, mu, lam_nh, kd, inv_h):
    n = len(H)
    out = [wp.zeros(n, dtype=float, device="cpu"), wp.zeros(n, dtype=mat, device="cpu"),
           wp.zeros(n, dtype=mat, device="cpu")]
    wp.launch(eval_kernel, dim=n, device="cpu",
              inputs=[wp.array(H, dtype=mat, device="cpu"), wp.array(dF, dtype=mat, device="cpu"),
                      wp.array(Gp, dtype=mat, device="cpu"), mu, lam_nh, kd, inv_h], outputs=out)
    return [o.numpy().astype(np.float64) for o in out]


def textbook_energy(F, Fp, mu, lam_nh, kd, inv_h):
    alpha = 1 + mu / lam_nh
    J = np.linalg.det(F)
    C = np.einsum("nki,nkj->nij", F, F)
    E = C - np.einsum("nki,nkj->nij", Fp, Fp)
    psi = 0.5 * mu * (np.trace(C, axis1=1, axis2=2) - 3) + 0.5 * lam_nh * ((J - alpha) ** 2 - (1 - alpha) ** 2)
    return psi + 0.5 * kd * inv_h * np.sum(E * E, axis=(1, 2))


rng = np.random.default_rng(0)
mu, lam_nh, kd, inv_h = 1.0e6, 2.0e6, 1.0e2, 600.0
n = 64
I = np.eye(3)
for scale in (0.3, 1e-2, 1e-4):
    H = scale * rng.standard_normal((n, 3, 3))
    Hp = scale * rng.standard_normal((n, 3, 3))
    Gp = np.einsum("nki,nkj->nij", I + Hp, I + Hp) - I
    dF = rng.standard_normal((n, 3, 3))
    psi, P, HdF = run(H, dF, Gp, mu, lam_nh, kd, inv_h)
    ref = textbook_energy(I + H, I + Hp, mu, lam_nh, kd, inv_h)
    e_rel = np.abs(psi - ref).max() / np.abs(ref).max()
    # stress vs FD of the float64 textbook energy
    eps = 1e-6 * max(scale, 1e-2)
    P_fd = np.zeros_like(P)
    for a in range(3):
        for b in range(3):
            D = np.zeros((3, 3))
            D[a, b] = eps
            P_fd[:, a, b] = (textbook_energy(I + H + D, I + Hp, mu, lam_nh, kd, inv_h)
                             - textbook_energy(I + H - D, I + Hp, mu, lam_nh, kd, inv_h)) / (2 * eps)
    p_rel = np.abs(P - P_fd).max() / np.abs(P_fd).max()
    # hessian vs FD of the float32 stress along dF; kd = 0 since the Gauss-Newton damping term is off by O(E)
    _, P1, _ = run(H + 1e-3 * dF, dF, Gp, mu, lam_nh, 0.0, inv_h)
    _, P0, _ = run(H - 1e-3 * dF, dF, Gp, mu, lam_nh, 0.0, inv_h)
    _, _, Hk0 = run(H, dF, Gp, mu, lam_nh, 0.0, inv_h)
    h_fd = (P1 - P0) / 2e-3
    h_rel = np.abs(Hk0 - h_fd).max() / np.abs(h_fd).max()
    print(f"|H| ~ {scale:g}: energy rel err {e_rel:.1e}, stress vs FD {p_rel:.1e}, Hessian vs FD {h_rel:.1e}")
    assert e_rel < 1e-4 and p_rel < 1e-3 and h_rel < 1e-2
print("constitutive kernels OK")
