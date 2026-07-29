from __future__ import annotations

"""Canonical direct-flow assembly for serial and MPI MUMPS solves.

The finite-volume equations are defined once by :func:`build_cell_rows`.
Serial runs convert those rows to a SciPy CSR matrix. MPI runs insert only the
rows owned by each PETSc rank directly into a distributed AIJ matrix.
"""

from dataclasses import dataclass
from typing import Any, Dict, MutableMapping, Optional, Tuple

import numpy as np
from scipy import sparse

from flow_scaling import FlowScaling, build_flow_scaling
from solver_utils import (
    body_force_y,
    cell_rho,
    cell_mu,
    cell_volume,
    compute_cell_gradients,
    compute_face_fluxes,
    compute_pressure_gradients,
    east_face_kind,
    face_mu_e,
    face_mu_n,
    face_mu_s,
    face_mu_w,
    face_rho_x,
    face_rho_y,
    flow_bc_p,
    flow_bc_type,
    flow_bc_u,
    flow_bc_v,
    fluid_at,
    north_face_kind,
    open_mdot_e,
    open_mdot_n,
    open_mdot_s,
    open_mdot_w,
    south_face_kind,
    sou_deferred_source,
    upwind_aE,
    upwind_aN,
    upwind_aS,
    upwind_aW,
    west_face_kind,
)

@dataclass
class MomentumPass:
    """Momentum coefficients computed before continuity assembly."""

    aE: np.ndarray
    aW: np.ndarray
    aN: np.ndarray
    aS: np.ndarray
    aPu: np.ndarray
    aPv: np.ndarray
    source_u: np.ndarray
    source_v: np.ndarray
    Fe: np.ndarray
    Fw: np.ndarray
    Fn: np.ndarray
    Fs: np.ndarray
    gradients: Dict[str, np.ndarray]
    fluxes: Dict[str, np.ndarray]


def _gidx_f(fid: int, variable: int) -> int:
    return 3 * fid + variable


def _gidx(ctx, i: int, j: int, variable: int) -> int:
    fid = int(ctx["cell_to_fid"][j, i])
    if fid < 0:
        raise ValueError(f"Cell ({i}, {j}) is not fluid.")
    return _gidx_f(fid, variable)


def _resolve_pressure_constraint(ctx: Dict[str, Any], settings: Dict[str, Any]) -> str:
    """Resolve the pressure gauge for a nonsingular direct factorization."""
    cfg = settings.get("pressure_reference", {})
    mode = str(cfg.get("mode", "auto")).lower().strip()
    if mode not in ("auto", "pin", "nullspace", "none"):
        raise ValueError(f"Unsupported pressure constraint mode: {mode!r}")

    has_pressure_boundary = any(
        flow_bc_type(ctx, side) == "open"
        for side in ("west", "east", "south", "north")
    )
    if mode == "auto":
        return "none" if has_pressure_boundary else "pin"
    if mode == "nullspace":
        # Sparse direct LU requires a nonsingular matrix.
        return "pin"
    return mode


def _pressure_reference_cell(ctx, settings):
    cfg = settings.get("pressure_reference", {})
    i = int(cfg.get("i", ctx.get("p_ref_i", 0)))
    j = int(cfg.get("j", ctx.get("p_ref_j", 0)))
    if fluid_at(ctx, i, j):
        return i, j
    if not ctx["fluid_cells"]:
        raise RuntimeError("No fluid cells are available for the pressure reference.")
    return ctx["fluid_cells"][0]


def _boundary_face_fluxes(ctx, fields, coeffs, gradients, i, j, kinds):
    """Return Fe, Fw, Fn, Fs using the existing sign convention."""
    kind_e, kind_w, kind_n, kind_s = kinds
    u = fields["u"]
    v = fields["v"]
    dy = float(ctx["dy"])
    dx = float(ctx["dx"])

    if kind_e == "fluid-fluid":
        Fe = np.nan
    elif kind_e == "fluid-solid":
        Fe = 0.0
    else:
        t = flow_bc_type(ctx, "east")
        rhoP = cell_rho(ctx, i, j)
        if t == "open":
            Fe = open_mdot_e(ctx, fields, coeffs, gradients, i, j)
        elif t == "outlet":
            Fe = rhoP * dy * u[j, i]
        elif t == "inlet":
            Fe = rhoP * dy * flow_bc_u(ctx, "east")
        else:
            Fe = 0.0

    if kind_w == "fluid-fluid":
        Fw = np.nan
    elif kind_w == "fluid-solid":
        Fw = 0.0
    else:
        t = flow_bc_type(ctx, "west")
        rhoP = cell_rho(ctx, i, j)
        if t == "open":
            Fw = open_mdot_w(ctx, fields, coeffs, gradients, i, j)
        elif t == "inlet":
            Fw = rhoP * dy * flow_bc_u(ctx, "west")
        elif t == "outlet":
            Fw = rhoP * dy * u[j, i]
        else:
            Fw = 0.0

    if kind_n == "fluid-fluid":
        Fn = np.nan
    elif kind_n == "fluid-solid":
        Fn = 0.0
    else:
        t = flow_bc_type(ctx, "north")
        rhoP = cell_rho(ctx, i, j)
        if t == "open":
            Fn = open_mdot_n(ctx, fields, coeffs, gradients, i, j)
        elif t == "inlet":
            Fn = rhoP * dx * flow_bc_v(ctx, "north")
        elif t == "outlet":
            Fn = rhoP * dx * v[j, i]
        else:
            Fn = 0.0

    if kind_s == "fluid-fluid":
        Fs = np.nan
    elif kind_s == "fluid-solid":
        Fs = 0.0
    else:
        t = flow_bc_type(ctx, "south")
        rhoP = cell_rho(ctx, i, j)
        if t == "open":
            Fs = open_mdot_s(ctx, fields, coeffs, gradients, i, j)
        elif t == "inlet":
            Fs = rhoP * dx * flow_bc_v(ctx, "south")
        elif t == "outlet":
            Fs = rhoP * dx * v[j, i]
        else:
            Fs = 0.0

    return Fe, Fw, Fn, Fs


def build_momentum_pass(ctx, settings, fields, lagged_coeffs) -> MomentumPass:
    """Pass 1: compute current momentum coefficients for every fluid cell."""
    nx = int(ctx["nx"])
    ny = int(ctx["ny"])
    dx = float(ctx["dx"])
    dy = float(ctx["dy"])
    is_fluid = ctx["is_fluid"]

    u_old = fields["u"]
    v_old = fields["v"]

    dpdx, dpdy = compute_pressure_gradients(ctx, fields["p"])
    gradients = {"dpdx": dpdx, "dpdy": dpdy}
    fluxes = compute_face_fluxes(ctx, settings, fields, lagged_coeffs, gradients)
    me, mn = fluxes["me"], fluxes["mn"]

    if settings["schemes"].get("momentum") == "sou":
        dudx, dudy = compute_cell_gradients(ctx, u_old, is_fluid)
        dvdx, dvdy = compute_cell_gradients(ctx, v_old, is_fluid)
    else:
        dudx = dudy = dvdx = dvdy = None

    shape = (ny, nx)
    aE = np.zeros(shape)
    aW = np.zeros(shape)
    aN = np.zeros(shape)
    aS = np.zeros(shape)
    aPu = np.ones(shape)
    aPv = np.ones(shape)
    source_u = np.zeros(shape)
    source_v = np.zeros(shape)
    Fe_field = np.zeros(shape)
    Fw_field = np.zeros(shape)
    Fn_field = np.zeros(shape)
    Fs_field = np.zeros(shape)

    for (i, j) in ctx["fluid_cells"]:
        kind_e = east_face_kind(ctx, i, j)
        kind_w = west_face_kind(ctx, i, j)
        kind_n = north_face_kind(ctx, i, j)
        kind_s = south_face_kind(ctx, i, j)
        kinds = (kind_e, kind_w, kind_n, kind_s)

        muP = cell_mu(ctx, i, j)
        De = (face_mu_e(ctx, i, j) if kind_e == "fluid-fluid" else muP) * dy / dx
        Dw = (face_mu_w(ctx, i, j) if kind_w == "fluid-fluid" else muP) * dy / dx
        Dn = (face_mu_n(ctx, i, j) if kind_n == "fluid-fluid" else muP) * dx / dy
        Ds = (face_mu_s(ctx, i, j) if kind_s == "fluid-fluid" else muP) * dx / dy

        boundary_Fe, boundary_Fw, boundary_Fn, boundary_Fs = _boundary_face_fluxes(
            ctx, fields, lagged_coeffs, gradients, i, j, kinds
        )
        Fe = float(me[j, i]) if kind_e == "fluid-fluid" else float(boundary_Fe)
        Fw = float(me[j, i - 1]) if kind_w == "fluid-fluid" else float(boundary_Fw)
        Fn = float(mn[j, i]) if kind_n == "fluid-fluid" else float(boundary_Fn)
        Fs = float(mn[j - 1, i]) if kind_s == "fluid-fluid" else float(boundary_Fs)

        Fe_field[j, i] = Fe
        Fw_field[j, i] = Fw
        Fn_field[j, i] = Fn
        Fs_field[j, i] = Fs

        ae = upwind_aE(Fe, De) if kind_e == "fluid-fluid" else 0.0
        aw = upwind_aW(Fw, Dw) if kind_w == "fluid-fluid" else 0.0
        an = upwind_aN(Fn, Dn) if kind_n == "fluid-fluid" else 0.0
        a_s = upwind_aS(Fs, Ds) if kind_s == "fluid-fluid" else 0.0

        Sp_u = 0.0
        Sp_v = 0.0
        Su_u = 0.0
        Su_v = 0.0

        if kind_w == "fluid-solid":
            coeff = (2.0 * muP * dy / dx) + max(Fw, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff
        if kind_e == "fluid-solid":
            coeff = (2.0 * muP * dy / dx) + max(-Fe, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff
        if kind_s == "fluid-solid":
            coeff = (2.0 * muP * dx / dy) + max(Fs, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff
        if kind_n == "fluid-solid":
            coeff = (2.0 * muP * dx / dy) + max(-Fn, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff

        if kind_w == "boundary-west":
            t = flow_bc_type(ctx, "west")
            if t in ("wall", "inlet"):
                coeff = (2.0 * muP * dy / dx) + max(Fw, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u(ctx, "west")
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v(ctx, "west")
            elif t == "symmetry":
                Sp_u -= (2.0 * muP * dy / dx) + max(Fw, 0.0)

        if kind_e == "boundary-east":
            t = flow_bc_type(ctx, "east")
            if t in ("wall", "inlet"):
                coeff = (2.0 * muP * dy / dx) + max(-Fe, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u(ctx, "east")
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v(ctx, "east")
            elif t == "symmetry":
                Sp_u -= (2.0 * muP * dy / dx) + max(-Fe, 0.0)

        if kind_s == "boundary-south":
            t = flow_bc_type(ctx, "south")
            if t in ("wall", "inlet"):
                coeff = (2.0 * muP * dx / dy) + max(Fs, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u(ctx, "south")
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v(ctx, "south")
            elif t == "symmetry":
                Sp_v -= (2.0 * muP * dx / dy) + max(Fs, 0.0)

        if kind_n == "boundary-north":
            t = flow_bc_type(ctx, "north")
            if t in ("wall", "inlet"):
                coeff = (2.0 * muP * dx / dy) + max(-Fn, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u(ctx, "north")
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v(ctx, "north")
            elif t == "symmetry":
                Sp_v -= (2.0 * muP * dx / dy) + max(-Fn, 0.0)

        ap_u = max(ae + aw + an + a_s + (Fe - Fw + Fn - Fs) - Sp_u, 1.0e-30)
        ap_v = max(ae + aw + an + a_s + (Fe - Fw + Fn - Fs) - Sp_v, 1.0e-30)

        if settings["schemes"].get("momentum") == "sou":
            Su_u += sou_deferred_source(
                ctx,
                u_old,
                dudx,
                dudy,
                i,
                j,
                Fe,
                Fw,
                Fn,
                Fs,
                kind_e,
                kind_w,
                kind_n,
                kind_s,
                settings["schemes"]["momentum_blend"],
            )
            Su_v += sou_deferred_source(
                ctx,
                v_old,
                dvdx,
                dvdy,
                i,
                j,
                Fe,
                Fw,
                Fn,
                Fs,
                kind_e,
                kind_w,
                kind_n,
                kind_s,
                settings["schemes"]["momentum_blend"],
            )

        aE[j, i] = ae
        aW[j, i] = aw
        aN[j, i] = an
        aS[j, i] = a_s
        aPu[j, i] = ap_u
        aPv[j, i] = ap_v
        source_u[j, i] = Su_u
        source_v[j, i] = Su_v

    aPu[ctx["is_solid"]] = 1.0
    aPv[ctx["is_solid"]] = 1.0

    return MomentumPass(
        aE=aE,
        aW=aW,
        aN=aN,
        aS=aS,
        aPu=aPu,
        aPv=aPv,
        source_u=source_u,
        source_v=source_v,
        Fe=Fe_field,
        Fw=Fw_field,
        Fn=Fn_field,
        Fs=Fs_field,
        gradients=gradients,
        fluxes=fluxes,
    )


def _add(row: MutableMapping[int, float], col: int, value: float) -> None:
    value = float(value)
    if value == 0.0:
        return
    col = int(col)
    row[col] = row.get(col, 0.0) + value


def build_cell_rows(
    ctx: Dict[str, Any],
    settings: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    momentum: MomentumPass,
    fid: int,
    *,
    pressure_constraint_mode: str,
    pressure_reference_row: Optional[int],
) -> Tuple[Tuple[Row, float], Tuple[Row, float], Tuple[Row, float]]:
    """Return the three absolute-system rows for one compact fluid cell."""
    i, j = ctx["fluid_cells"][fid]
    cell_to_fid = ctx["cell_to_fid"]
    dx = float(ctx["dx"])
    dy = float(ctx["dy"])
    V = float(ctx["V"])

    ru = _gidx_f(fid, 0)
    rv = _gidx_f(fid, 1)
    rp = _gidx_f(fid, 2)

    row_u: Row = {}
    row_v: Row = {}
    row_p: Row = {}
    rhs_u = 0.0
    rhs_v = 0.0
    rhs_p = 0.0

    kind_e = east_face_kind(ctx, i, j)
    kind_w = west_face_kind(ctx, i, j)
    kind_n = north_face_kind(ctx, i, j)
    kind_s = south_face_kind(ctx, i, j)

    ae = float(momentum.aE[j, i])
    aw = float(momentum.aW[j, i])
    an = float(momentum.aN[j, i])
    a_s = float(momentum.aS[j, i])
    ap_u = float(momentum.aPu[j, i])
    ap_v = float(momentum.aPv[j, i])

    _add(row_u, ru, ap_u)
    _add(row_v, rv, ap_v)

    px = V / dx
    py = V / dy

    if kind_e == "fluid-fluid":
        _add(row_u, _gidx(ctx, i + 1, j, 0), -ae)
        _add(row_v, _gidx(ctx, i + 1, j, 1), -ae)
        _add(row_u, rp, 0.5 * px)
        _add(row_u, _gidx(ctx, i + 1, j, 2), 0.5 * px)
    elif kind_e == "boundary-east" and flow_bc_type(ctx, "east") == "open":
        rhs_u -= px * flow_bc_p(ctx, "east")
    else:
        _add(row_u, rp, px)

    if kind_w == "fluid-fluid":
        _add(row_u, _gidx(ctx, i - 1, j, 0), -aw)
        _add(row_v, _gidx(ctx, i - 1, j, 1), -aw)
        _add(row_u, rp, -0.5 * px)
        _add(row_u, _gidx(ctx, i - 1, j, 2), -0.5 * px)
    elif kind_w == "boundary-west" and flow_bc_type(ctx, "west") == "open":
        rhs_u += px * flow_bc_p(ctx, "west")
    else:
        _add(row_u, rp, -px)

    if kind_n == "fluid-fluid":
        _add(row_u, _gidx(ctx, i, j + 1, 0), -an)
        _add(row_v, _gidx(ctx, i, j + 1, 1), -an)
        _add(row_v, rp, 0.5 * py)
        _add(row_v, _gidx(ctx, i, j + 1, 2), 0.5 * py)
    elif kind_n == "boundary-north" and flow_bc_type(ctx, "north") == "open":
        rhs_v -= py * flow_bc_p(ctx, "north")
    else:
        _add(row_v, rp, py)

    if kind_s == "fluid-fluid":
        _add(row_u, _gidx(ctx, i, j - 1, 0), -a_s)
        _add(row_v, _gidx(ctx, i, j - 1, 1), -a_s)
        _add(row_v, rp, -0.5 * py)
        _add(row_v, _gidx(ctx, i, j - 1, 2), -0.5 * py)
    elif kind_s == "boundary-south" and flow_bc_type(ctx, "south") == "open":
        rhs_v += py * flow_bc_p(ctx, "south")
    else:
        _add(row_v, rp, -py)

    rhs_u += float(momentum.source_u[j, i]) + float(ctx["sx"][j, i]) * cell_volume(ctx, i, j)
    rhs_v += (
        float(momentum.source_v[j, i])
        + body_force_y(ctx, settings, i, j, fields["T"][j, i]) * cell_volume(ctx, i, j)
        + float(ctx["sy"][j, i]) * cell_volume(ctx, i, j)
    )

    dpdx = momentum.gradients["dpdx"]
    dpdy = momentum.gradients["dpdy"]

    aEc = aWc = aNc = aSc = 0.0
    ucoef_e = ucoef_w = 0.0
    vcoef_n = vcoef_s = 0.0
    ucoef_p = vcoef_p = 0.0
    dpe_interp = dpw_interp = dpn_interp = dps_interp = 0.0
    byn_corr = bys_corr = 0.0

    if kind_e == "fluid-fluid":
        de = 0.5 * (
            V / max(momentum.aPu[j, i], 1.0e-30)
            + V / max(momentum.aPu[j, i + 1], 1.0e-30)
        )
        rho_e = face_rho_x(ctx, i, j)
        dpe_interp = rho_e * dy * de * 0.5 * (dpdx[j, i] + dpdx[j, i + 1])
        aEc = rho_e * dy * de / dx
        ucoef_e = rho_e * dy / 2.0
        ucoef_p += rho_e * dy / 2.0
    elif kind_e == "boundary-east":
        t = flow_bc_type(ctx, "east")
        rhoP = cell_rho(ctx, i, j)
        if t == "outlet":
            _add(row_p, _gidx(ctx, i, j, 0), rhoP * dy)
        elif t == "inlet":
            rhs_p += -rhoP * dy * flow_bc_u(ctx, "east")
        elif t == "open":
            d_open = V / max(momentum.aPu[j, i], 1.0e-30)
            a_open = rhoP * dy * d_open / dx
            _add(row_p, _gidx(ctx, i, j, 0), rhoP * dy)
            _add(row_p, rp, a_open)
            rhs_p += a_open * flow_bc_p(ctx, "east") - rhoP * dy * d_open * dpdx[j, i]

    if kind_w == "fluid-fluid":
        dw = 0.5 * (
            V / max(momentum.aPu[j, i - 1], 1.0e-30)
            + V / max(momentum.aPu[j, i], 1.0e-30)
        )
        rho_w = face_rho_x(ctx, i - 1, j)
        dpw_interp = rho_w * dy * dw * 0.5 * (dpdx[j, i] + dpdx[j, i - 1])
        aWc = rho_w * dy * dw / dx
        ucoef_w = rho_w * dy / 2.0
        ucoef_p -= rho_w * dy / 2.0
    elif kind_w == "boundary-west":
        t = flow_bc_type(ctx, "west")
        rhoP = cell_rho(ctx, i, j)
        if t == "inlet":
            rhs_p += rhoP * dy * flow_bc_u(ctx, "west")
        elif t == "outlet":
            _add(row_p, _gidx(ctx, i, j, 0), -rhoP * dy)
        elif t == "open":
            d_open = V / max(momentum.aPu[j, i], 1.0e-30)
            a_open = rhoP * dy * d_open / dx
            _add(row_p, _gidx(ctx, i, j, 0), -rhoP * dy)
            _add(row_p, rp, a_open)
            rhs_p += a_open * flow_bc_p(ctx, "west") + rhoP * dy * d_open * dpdx[j, i]

    if kind_n == "fluid-fluid":
        dP = V / max(momentum.aPv[j, i], 1.0e-30)
        dN = V / max(momentum.aPv[j + 1, i], 1.0e-30)
        dn = 0.5 * (dP + dN)
        rho_n = face_rho_y(ctx, i, j)
        aNc = rho_n * dx * dn / dy
        vcoef_n = rho_n * dx / 2.0
        vcoef_p += rho_n * dx / 2.0
        dpn_interp = rho_n * dx * dn * 0.5 * (dpdy[j, i] + dpdy[j + 1, i])
        if settings["physics"].get("buoyancy", False):
            ByP = body_force_y(ctx, settings, i, j, fields["T"][j, i])
            ByN = body_force_y(ctx, settings, i, j, fields["T"][j + 1, i])
            By_face = 0.5 * (ByP + ByN)
            byn_corr = rho_n * dx * (dn * By_face - 0.5 * (dP * ByP + dN * ByN))
    elif kind_n == "boundary-north":
        t = flow_bc_type(ctx, "north")
        rhoP = cell_rho(ctx, i, j)
        if t == "inlet":
            rhs_p += -rhoP * dx * flow_bc_v(ctx, "north")
        elif t == "outlet":
            _add(row_p, _gidx(ctx, i, j, 1), rhoP * dx)
        elif t == "open":
            d_open = V / max(momentum.aPv[j, i], 1.0e-30)
            a_open = rhoP * dx * d_open / dy
            _add(row_p, _gidx(ctx, i, j, 1), rhoP * dx)
            _add(row_p, rp, a_open)
            rhs_p += a_open * flow_bc_p(ctx, "north") - rhoP * dx * d_open * dpdy[j, i]

    if kind_s == "fluid-fluid":
        dS = V / max(momentum.aPv[j - 1, i], 1.0e-30)
        dP = V / max(momentum.aPv[j, i], 1.0e-30)
        ds = 0.5 * (dS + dP)
        rho_s = face_rho_y(ctx, i, j - 1)
        aSc = rho_s * dx * ds / dy
        vcoef_s = rho_s * dx / 2.0
        vcoef_p -= rho_s * dx / 2.0
        dps_interp = rho_s * dx * ds * 0.5 * (dpdy[j - 1, i] + dpdy[j, i])
        if settings["physics"].get("buoyancy", False):
            ByS = body_force_y(ctx, settings, i, j, fields["T"][j - 1, i])
            ByP = body_force_y(ctx, settings, i, j, fields["T"][j, i])
            By_face = 0.5 * (ByS + ByP)
            bys_corr = rho_s * dx * (ds * By_face - 0.5 * (dS * ByS + dP * ByP))
    elif kind_s == "boundary-south":
        t = flow_bc_type(ctx, "south")
        rhoP = cell_rho(ctx, i, j)
        if t == "inlet":
            rhs_p += rhoP * dx * flow_bc_v(ctx, "south")
        elif t == "outlet":
            _add(row_p, _gidx(ctx, i, j, 1), -rhoP * dx)
        elif t == "open":
            d_open = V / max(momentum.aPv[j, i], 1.0e-30)
            a_open = rhoP * dx * d_open / dy
            _add(row_p, _gidx(ctx, i, j, 1), -rhoP * dx)
            _add(row_p, rp, a_open)
            rhs_p += a_open * flow_bc_p(ctx, "south") + rhoP * dx * d_open * dpdy[j, i]

    _add(row_p, rp, aEc + aWc + aNc + aSc)
    _add(row_p, _gidx(ctx, i, j, 0), ucoef_p)
    _add(row_p, _gidx(ctx, i, j, 1), vcoef_p)

    if kind_e == "fluid-fluid":
        nb = int(cell_to_fid[j, i + 1])
        _add(row_p, _gidx_f(nb, 0), ucoef_e)
        _add(row_p, _gidx_f(nb, 2), -aEc)
    if kind_w == "fluid-fluid":
        nb = int(cell_to_fid[j, i - 1])
        _add(row_p, _gidx_f(nb, 0), -ucoef_w)
        _add(row_p, _gidx_f(nb, 2), -aWc)
    if kind_n == "fluid-fluid":
        nb = int(cell_to_fid[j + 1, i])
        _add(row_p, _gidx_f(nb, 1), vcoef_n)
        _add(row_p, _gidx_f(nb, 2), -aNc)
    if kind_s == "fluid-fluid":
        nb = int(cell_to_fid[j - 1, i])
        _add(row_p, _gidx_f(nb, 1), -vcoef_s)
        _add(row_p, _gidx_f(nb, 2), -aSc)

    rhs_p += -dpe_interp + dpw_interp - dpn_interp + dps_interp + byn_corr - bys_corr

    if pressure_constraint_mode == "pin" and pressure_reference_row == rp:
        row_p = {rp: 1.0}
        rhs_p = float(settings.get("pressure_reference", {}).get("value", 0.0))

    return (row_u, rhs_u), (row_v, rhs_v), (row_p, rhs_p)


@dataclass
class SerialFlowLinearSystem:
    """Fully assembled serial correction system."""

    matrix: sparse.csr_matrix
    rhs: np.ndarray
    scaled_matrix: sparse.csr_matrix
    scaled_rhs: np.ndarray
    old_state: np.ndarray
    scaling: FlowScaling
    metadata: Dict[str, Any]
    momentum: MomentumPass
    absolute_rhs: np.ndarray
    pressure_constraint_mode: str

    is_distributed_local: bool = False
    block_size: int = 3

    def recover_correction(self, scaled_solution: np.ndarray) -> np.ndarray:
        return self.scaling.unscale_solution(scaled_solution)


@dataclass
class DistributedFlowLinearSystem:
    """Descriptor that assembles owned coupled rows directly into PETSc."""

    ctx: Dict[str, Any]
    settings: Dict[str, Any]
    fields: Dict[str, np.ndarray]
    momentum: MomentumPass
    old_state: np.ndarray
    scaling: FlowScaling
    pressure_constraint_mode: str
    pressure_reference_row: Optional[int]
    preallocation_nnz: int = 20

    is_distributed_local: bool = True
    block_size: int = 3

    @property
    def global_size(self) -> int:
        return int(self.old_state.size)

    @property
    def pattern_key(self) -> Tuple[Any, ...]:
        return (
            "v5_direct_local_flow",
            int(self.ctx["nx"]),
            int(self.ctx["ny"]),
            int(self.ctx["Nf"]),
            int(self.pressure_reference_row if self.pressure_reference_row is not None else -1),
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "Nf": int(self.ctx["Nf"]),
            "block_size": 3,
            "scaling": self.scaling,
            "pressure_constraint_mode": self.pressure_constraint_mode,
            "distributed_local_assembly": True,
            "pattern_key": self.pattern_key,
        }

    def _owned_fid_range(self, mat) -> Tuple[int, int]:
        row_start, row_end = mat.getOwnershipRange()
        if row_start % 3 or row_end % 3:
            raise RuntimeError(
                "PETSc ownership must be aligned to complete [u,v,p] cell blocks."
            )
        return row_start // 3, row_end // 3

    def assemble_petsc(self, mat, rhs_vec) -> Dict[str, float]:
        """Insert only the rows owned by this MPI rank."""
        from petsc4py import PETSc

        mat.zeroEntries()
        rhs_vec.set(0.0)
        fid_start, fid_end = self._owned_fid_range(mat)

        for fid in range(fid_start, fid_end):
            equations = build_cell_rows(
                self.ctx,
                self.settings,
                self.fields,
                self.momentum,
                fid,
                pressure_constraint_mode=self.pressure_constraint_mode,
                pressure_reference_row=self.pressure_reference_row,
            )
            for variable, (row_values, rhs_absolute) in enumerate(equations):
                global_row = _gidx_f(fid, variable)
                correction_rhs = float(rhs_absolute)
                for column, value in row_values.items():
                    correction_rhs -= float(value) * float(self.old_state[column])

                row_index = np.asarray([global_row], dtype=PETSc.IntType)
                columns = np.fromiter(
                    row_values.keys(), dtype=PETSc.IntType, count=len(row_values)
                )
                values_unscaled = np.fromiter(
                    row_values.values(),
                    dtype=PETSc.ScalarType,
                    count=len(row_values),
                )
                values_scaled = np.asarray(
                    float(self.scaling.left[global_row])
                    * values_unscaled
                    * self.scaling.right[np.asarray(columns, dtype=np.intp)],
                    dtype=PETSc.ScalarType,
                )
                mat.setValues(
                    row_index,
                    columns,
                    values_scaled.reshape(1, -1),
                )
                rhs_vec.setValue(
                    int(global_row),
                    float(self.scaling.left[global_row]) * correction_rhs,
                )

        mat.assemble()
        rhs_vec.assemble()
        return {
            "local_fid_start": float(fid_start),
            "local_fid_end": float(fid_end),
            "local_cells": float(fid_end - fid_start),
        }

    def recover_correction(self, scaled_solution: np.ndarray) -> np.ndarray:
        return self.scaling.unscale_solution(scaled_solution)

    def distributed_residual_metrics(self, mat, rhs_vec, solution_vec) -> Dict[str, float]:
        """Compute collective scaled and physical infinity-norm residuals."""
        from petsc4py import PETSc

        residual = rhs_vec.duplicate()
        mat.mult(solution_vec, residual)
        residual.aypx(-1.0, rhs_vec)  # residual = rhs - A*x

        scaled_abs = float(residual.norm(PETSc.NormType.NORM_INFINITY))
        scaled_rhs = max(
            float(rhs_vec.norm(PETSc.NormType.NORM_INFINITY)), 1.0e-30
        )
        scaled_rel = scaled_abs / scaled_rhs

        row_start, row_end = mat.getOwnershipRange()
        local_left = np.asarray(self.scaling.left[row_start:row_end], dtype=float)

        physical_residual = residual.duplicate()
        physical_residual.getArray()[:] = (
            np.asarray(residual.getArray(readonly=True), dtype=float) / local_left
        )
        physical_residual.assemble()

        physical_rhs = rhs_vec.duplicate()
        physical_rhs.getArray()[:] = (
            np.asarray(rhs_vec.getArray(readonly=True), dtype=float) / local_left
        )
        physical_rhs.assemble()

        unscaled_abs = float(
            physical_residual.norm(PETSc.NormType.NORM_INFINITY)
        )
        unscaled_rhs = max(
            float(physical_rhs.norm(PETSc.NormType.NORM_INFINITY)), 1.0e-30
        )
        unscaled_rel = unscaled_abs / unscaled_rhs

        physical_rhs.destroy()
        physical_residual.destroy()
        residual.destroy()
        return {
            "scaled_true_rel_residual": scaled_rel,
            "scaled_true_abs_residual": scaled_abs,
            "unscaled_true_rel_residual": unscaled_rel,
            "unscaled_true_abs_residual": unscaled_abs,
            "true_rel_residual": max(scaled_rel, unscaled_rel),
            "true_abs_residual": max(scaled_abs, unscaled_abs),
        }


def build_state_vector(ctx, fields) -> np.ndarray:
    x = np.zeros(3 * int(ctx["Nf"]), dtype=float)
    for fid, (i, j) in enumerate(ctx["fluid_cells"]):
        x[_gidx_f(fid, 0)] = fields["u"][j, i]
        x[_gidx_f(fid, 1)] = fields["v"][j, i]
        x[_gidx_f(fid, 2)] = fields["p"][j, i]
    return x


def _pressure_reference_row(
    ctx: Dict[str, Any],
    settings: Dict[str, Any],
    pressure_constraint_mode: str,
) -> Optional[int]:
    if pressure_constraint_mode != "pin":
        return None
    i_ref, j_ref = _pressure_reference_cell(ctx, settings)
    return _gidx(ctx, i_ref, j_ref, 2)


def assemble_serial_absolute_system(
    ctx: Dict[str, Any],
    settings: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    momentum: MomentumPass,
    pressure_constraint_mode: str,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Build a global CSR matrix from the canonical cell-row equations."""
    ndof = 3 * int(ctx["Nf"])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.zeros(ndof, dtype=float)
    reference_row = _pressure_reference_row(ctx, settings, pressure_constraint_mode)

    for fid in range(int(ctx["Nf"])):
        equations = build_cell_rows(
            ctx,
            settings,
            fields,
            momentum,
            fid,
            pressure_constraint_mode=pressure_constraint_mode,
            pressure_reference_row=reference_row,
        )
        for variable, (row_values, rhs_value) in enumerate(equations):
            row = _gidx_f(fid, variable)
            rhs[row] = float(rhs_value)
            for column, value in row_values.items():
                rows.append(row)
                columns.append(int(column))
                values.append(float(value))

    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(ndof, ndof)
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, rhs


def assemble_flow_correction_system(
    ctx: Dict[str, Any],
    settings: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    lagged_coeffs: Dict[str, np.ndarray],
    *,
    distributed: bool,
):
    """Build the direct-LU correction system for one nonlinear iteration."""
    momentum = build_momentum_pass(ctx, settings, fields, lagged_coeffs)
    constraint_mode = _resolve_pressure_constraint(ctx, settings)
    old_state = build_state_vector(ctx, fields)

    scaling_cfg = (
        settings.get("linear_solver", {})
        .get("flow_coupled", {})
        .get("scaling", {})
    )
    scaling = build_flow_scaling(ctx, fields, old_state.size, scaling_cfg)
    reference_row = _pressure_reference_row(ctx, settings, constraint_mode)

    if distributed:
        return DistributedFlowLinearSystem(
            ctx=ctx,
            settings=settings,
            fields=fields,
            momentum=momentum,
            old_state=old_state,
            scaling=scaling,
            pressure_constraint_mode=constraint_mode,
            pressure_reference_row=reference_row,
            preallocation_nnz=int(
                settings.get("linear_solver", {})
                .get("flow_coupled", {})
                .get("preallocation_nnz", 20)
            ),
        )

    matrix, absolute_rhs = assemble_serial_absolute_system(
        ctx,
        settings,
        fields,
        momentum,
        pressure_constraint_mode=constraint_mode,
    )
    rhs = np.asarray(absolute_rhs - matrix @ old_state, dtype=float)
    scaled_matrix = scaling.scale_matrix(matrix)
    scaled_rhs = scaling.scale_rhs(rhs)
    metadata = {
        "Nf": int(ctx["Nf"]),
        "block_size": 3,
        "scaling": scaling,
        "pressure_constraint_mode": constraint_mode,
        "true_matrix": scaled_matrix,
        "true_rhs": scaled_rhs,
        "unscaled_true_matrix": matrix,
        "unscaled_true_rhs": rhs,
    }
    return SerialFlowLinearSystem(
        matrix=matrix,
        rhs=rhs,
        scaled_matrix=scaled_matrix,
        scaled_rhs=np.asarray(scaled_rhs, dtype=float),
        old_state=old_state,
        scaling=scaling,
        metadata=metadata,
        momentum=momentum,
        absolute_rhs=np.asarray(absolute_rhs, dtype=float),
        pressure_constraint_mode=constraint_mode,
    )


def reconstruct_serial_from_rows(
    ctx: Dict[str, Any],
    settings: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    momentum: MomentumPass,
    pressure_constraint_mode: str,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Regression helper retained for assembly tests."""
    return assemble_serial_absolute_system(
        ctx, settings, fields, momentum, pressure_constraint_mode
    )