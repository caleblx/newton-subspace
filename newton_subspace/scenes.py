from __future__ import annotations

import warp as wp

import newton


def cantilever(
    res: int = 1,
    *,
    dims: tuple[int, int] | None = None,
    length: float = 1.2,
    width: float = 0.4,
    k_mu: float = 1.0e5,
    k_lambda: float = 1.0e5,
    k_damp: float = 1.0e1,
    density: float = 1.0e3,
) -> newton.Model:
    """Beam clamped at x = 0 with (12 res) x (4 res) x (4 res) cells, 5 tets per cell."""
    dim_x, dim_yz = dims if dims is not None else (12 * res, 4 * res)
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x,
        dim_y=dim_yz,
        dim_z=dim_yz,
        cell_x=length / dim_x,
        cell_y=width / dim_yz,
        cell_z=width / dim_yz,
        density=density,  # per unit volume, so total mass doesn't depend on res
        k_mu=k_mu,
        k_lambda=k_lambda,
        k_damp=k_damp,
        fix_left=True,
    )
    builder.color()
    return builder.finalize()
