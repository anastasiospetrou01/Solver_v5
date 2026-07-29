from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class FlowScaling:
    """Two-sided physical scaling for the coupled correction system.

    The original correction is ``delta_x`` and the scaled unknown is ``y``:

        delta_x = R y
        (L A R) y = L rhs

    ``left`` stores the diagonal of L and ``right`` stores the diagonal of R.
    """

    left: np.ndarray
    right: np.ndarray
    velocity_scale: float
    pressure_scale: float
    momentum_equation_scale: float
    continuity_equation_scale: float

    def scale_matrix(self, matrix) -> sparse.csr_matrix:
        matrix_csr = sparse.csr_matrix(matrix)
        scaled = sparse.diags(self.left) @ matrix_csr @ sparse.diags(self.right)
        scaled = scaled.tocsr()
        scaled.sum_duplicates()
        scaled.sort_indices()
        return scaled

    def scale_rhs(self, rhs) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float).reshape(-1)
        return self.left * rhs_arr

    def unscale_solution(self, scaled_solution) -> np.ndarray:
        y = np.asarray(scaled_solution, dtype=float).reshape(-1)
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


def infer_reference_velocity(ctx: Dict[str, Any], fields: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> float:
    """Infer a robust velocity scale from user input, BCs and the current state."""
    user_value = cfg.get("velocity_scale", "auto")
    if not (user_value is None or (isinstance(user_value, str) and user_value.lower() == "auto")):
        return _positive_or_auto(user_value, 1.0)

    candidates = [float(cfg.get("minimum_velocity_scale", 1.0e-3))]
    for side in ("west", "east", "south", "north"):
        bc = ctx["BC_flow"].get(side, {})
        candidates.extend([abs(float(bc.get("u", 0.0))), abs(float(bc.get("v", 0.0)))])

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
    """Build global physical field/equation scales for interleaved [u,v,p] DOFs."""
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", True))

    if ndof % 3 != 0:
        raise ValueError("The coupled flow system must contain three DOFs per fluid cell.")

    if not enabled:
        ones = np.ones(ndof, dtype=float)
        return FlowScaling(
            left=ones,
            right=ones,
            velocity_scale=1.0,
            pressure_scale=1.0,
            momentum_equation_scale=1.0,
            continuity_equation_scale=1.0,
        )

    is_fluid = ctx["is_fluid"]
    rho_values = np.asarray(ctx["rho"][is_fluid], dtype=float)
    mu_values = np.asarray(ctx["mu"][is_fluid], dtype=float)
    rho_ref = _positive_or_auto(cfg.get("rho_scale", "auto"), float(np.median(rho_values)))
    mu_ref = _positive_or_auto(cfg.get("mu_scale", "auto"), float(np.median(mu_values)))

    u_ref = infer_reference_velocity(ctx, fields, cfg)
    p_fallback = rho_ref * u_ref * u_ref
    p_ref = _positive_or_auto(cfg.get("pressure_scale", "auto"), max(p_fallback, 1.0e-12))

    # The equations are volume-integrated in a two-dimensional unit-depth domain.
    # h_ref gives grid-consistent scales for face-integrated convective and mass fluxes.
    h_ref = max(float(np.sqrt(ctx["dx"] * ctx["dy"])), 1.0e-30)
    l_ref = max(float(ctx.get("Lx", ctx["dx"])), float(ctx.get("Ly", ctx["dy"])), h_ref)

    momentum_rhs = max(
        rho_ref * u_ref * u_ref * h_ref,
        mu_ref * u_ref * h_ref / l_ref,
        float(cfg.get("minimum_momentum_scale", 1.0e-20)),
    )
    continuity_rhs = max(
        rho_ref * u_ref * h_ref,
        float(cfg.get("minimum_continuity_scale", 1.0e-20)),
    )

    left = np.empty(ndof, dtype=float)
    right = np.empty(ndof, dtype=float)
    left[0::3] = 1.0 / momentum_rhs
    left[1::3] = 1.0 / momentum_rhs
    left[2::3] = 1.0 / continuity_rhs
    right[0::3] = u_ref
    right[1::3] = u_ref
    right[2::3] = p_ref

    return FlowScaling(
        left=left,
        right=right,
        velocity_scale=u_ref,
        pressure_scale=p_ref,
        momentum_equation_scale=1.0 / momentum_rhs,
        continuity_equation_scale=1.0 / continuity_rhs,
    )
