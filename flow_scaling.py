from __future__ import annotations

"""Distributed physical scaling for the coupled [u,v,p] correction system."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class FlowScaling:
    """Two-sided field/equation scaling without replicated O(N) arrays.

    Physical correction and scaled unknown are related by ``delta_x = R y``.
    Momentum rows use ``momentum_equation_scale`` on the left; continuity rows
    use ``continuity_equation_scale``.  Velocity columns use
    ``velocity_scale`` on the right and pressure columns use
    ``pressure_scale``.
    """

    velocity_scale: float
    pressure_scale: float
    momentum_equation_scale: float
    continuity_equation_scale: float
    global_ndof: int

    def left_scale_for_row(self, global_row: int) -> float:
        return (
            self.continuity_equation_scale
            if int(global_row) % 3 == 2
            else self.momentum_equation_scale
        )

    def right_scale_for_col(self, global_col: int) -> float:
        return (
            self.pressure_scale
            if int(global_col) % 3 == 2
            else self.velocity_scale
        )

    def local_left(self, row_start: int, row_end: int) -> np.ndarray:
        rows = np.arange(int(row_start), int(row_end), dtype=np.int64)
        result = np.full(rows.size, self.momentum_equation_scale, dtype=float)
        result[rows % 3 == 2] = self.continuity_equation_scale
        return result

    def local_right(self, row_start: int, row_end: int) -> np.ndarray:
        rows = np.arange(int(row_start), int(row_end), dtype=np.int64)
        result = np.full(rows.size, self.velocity_scale, dtype=float)
        result[rows % 3 == 2] = self.pressure_scale
        return result

    def unscale_local_solution(
        self,
        scaled_solution,
        row_start: int,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        y = np.asarray(scaled_solution, dtype=float).reshape(-1)
        if out is None or out.size != y.size:
            out = np.empty_like(y)
        global_rows = int(row_start) + np.arange(y.size, dtype=np.int64)
        out[:] = y * self.velocity_scale
        pressure = global_rows % 3 == 2
        out[pressure] = y[pressure] * self.pressure_scale
        return out

    # Compatibility helpers for serial regression/debugging only.  Production
    # Phase F code does not access these because they allocate global arrays.
    @property
    def left(self) -> np.ndarray:
        return self.local_left(0, self.global_ndof)

    @property
    def right(self) -> np.ndarray:
        return self.local_right(0, self.global_ndof)

    def scale_matrix(self, matrix) -> sparse.csr_matrix:
        matrix_csr = sparse.csr_matrix(matrix)
        scaled = sparse.diags(self.left) @ matrix_csr @ sparse.diags(self.right)
        scaled = scaled.tocsr()
        scaled.sum_duplicates()
        scaled.sort_indices()
        return scaled

    def scale_rhs(self, rhs) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        if rhs_arr.size != self.global_ndof:
            raise ValueError("scale_rhs compatibility helper expects a global RHS.")
        return self.left * rhs_arr

    def unscale_solution(self, scaled_solution) -> np.ndarray:
        y = np.asarray(scaled_solution, dtype=float).reshape(-1)
        if y.size != self.global_ndof:
            raise ValueError(
                "unscale_solution compatibility helper expects the global solution; "
                "use unscale_local_solution in distributed production runs."
            )
        return self.right * y

    @property
    def pressure_left_scale(self) -> float:
        return float(self.continuity_equation_scale)

    @property
    def pressure_right_scale(self) -> float:
        return float(self.pressure_scale)


def _positive_or_auto(value: Any, fallback: float) -> float:
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return float(fallback)
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Scaling value must be finite and positive, got {value!r}.")
    return value


def infer_reference_velocity(
    ctx: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> float:
    user_value = cfg.get("velocity_scale", "auto")
    if not (
        user_value is None
        or (isinstance(user_value, str) and user_value.lower() == "auto")
    ):
        return _positive_or_auto(user_value, 1.0)

    candidates = [float(cfg.get("minimum_velocity_scale", 1.0e-3))]
    for side in ("west", "east", "south", "north"):
        bc = ctx["BC_flow"].get(side, {})
        candidates.extend(
            [abs(float(bc.get("u", 0.0))), abs(float(bc.get("v", 0.0)))]
        )

    domain = ctx.get("domain")
    if domain is not None:
        owned = domain.owned_slice
        mask = ctx["is_fluid"][owned, :]
        if np.any(mask):
            speed = np.hypot(fields["u"][owned, :], fields["v"][owned, :])
            local_max = float(np.max(speed[mask])) if np.any(mask) else 0.0
        else:
            local_max = 0.0
        candidates.append(domain.allreduce_max(local_max))
    else:
        is_fluid = ctx["is_fluid"]
        if np.any(is_fluid):
            speed = np.hypot(fields["u"][is_fluid], fields["v"][is_fluid])
            if speed.size:
                candidates.append(float(np.max(speed)))

    value = max(candidates)
    return max(value, float(cfg.get("minimum_velocity_scale", 1.0e-3)))


def build_flow_scaling(
    ctx: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    ndof: int,
    cfg: Dict[str, Any] | None,
) -> FlowScaling:
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", True))
    if ndof % 3 != 0:
        raise ValueError("The coupled flow system must contain three DOFs per fluid cell.")

    if not enabled:
        return FlowScaling(
            velocity_scale=1.0,
            pressure_scale=1.0,
            momentum_equation_scale=1.0,
            continuity_equation_scale=1.0,
            global_ndof=int(ndof),
        )

    # Static global medians are computed once from the original case during
    # distributed initialization and then retained as scalars only.
    rho_fallback = float(ctx.get("rho_reference_scale", 1.0))
    mu_fallback = float(ctx.get("mu_reference_scale", 1.0e-3))
    rho_ref = _positive_or_auto(cfg.get("rho_scale", "auto"), rho_fallback)
    mu_ref = _positive_or_auto(cfg.get("mu_scale", "auto"), mu_fallback)

    u_ref = infer_reference_velocity(ctx, fields, cfg)
    p_fallback = rho_ref * u_ref * u_ref
    p_ref = _positive_or_auto(
        cfg.get("pressure_scale", "auto"), max(p_fallback, 1.0e-12)
    )

    h_ref = max(float(np.sqrt(ctx["dx"] * ctx["dy"])), 1.0e-30)
    l_ref = max(
        float(ctx.get("Lx", ctx["dx"])),
        float(ctx.get("Ly", ctx["dy"])),
        h_ref,
    )
    momentum_rhs = max(
        rho_ref * u_ref * u_ref * h_ref,
        mu_ref * u_ref * h_ref / l_ref,
        float(cfg.get("minimum_momentum_scale", 1.0e-20)),
    )
    continuity_rhs = max(
        rho_ref * u_ref * h_ref,
        float(cfg.get("minimum_continuity_scale", 1.0e-20)),
    )

    return FlowScaling(
        velocity_scale=u_ref,
        pressure_scale=p_ref,
        momentum_equation_scale=1.0 / momentum_rhs,
        continuity_equation_scale=1.0 / continuity_rhs,
        global_ndof=int(ndof),
    )