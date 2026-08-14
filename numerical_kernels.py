from __future__ import annotations

"""Numba-compiled numerical kernels for Solver V5.

The kernels intentionally accept only NumPy arrays, scalars and integer codes.
No Python dictionaries, strings, sparse objects or PETSc objects enter JIT code.
This keeps the discretization numerically equivalent to the validated Python
implementation while removing Python loop overhead.
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on installations without Numba
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator


# Direction order must match geometry.py.
EAST = 0
WEST = 1
NORTH = 2
SOUTH = 3

FACE_FLUID_FLUID = 0
FACE_FLUID_SOLID = 1
FACE_BOUNDARY = 2

FLOW_BC_WALL = 0
FLOW_BC_INLET = 1
FLOW_BC_OUTLET = 2
FLOW_BC_OPEN = 3
FLOW_BC_SYMMETRY = 4
FLOW_BC_OTHER = 5

HEAT_BC_DIRICHLET = 0
HEAT_BC_NEUMANN = 1
HEAT_BC_ADIABATIC = 2
HEAT_BC_SYMMETRY = 3
HEAT_BC_OUTLET = 4
HEAT_BC_OPEN = 5
HEAT_BC_OTHER = 6

_EPS = 1.0e-30


@njit(cache=True, fastmath=False)
def _harmonic_mean(a: float, b: float) -> float:
    denom = a + b
    if denom == 0.0:
        return 0.0
    return 2.0 * a * b / denom


@njit(cache=True, fastmath=False)
def _body_force_y(
    rho: float,
    beta: float,
    temperature: float,
    t_ref: float,
    gy: float,
    buoyancy_enabled: bool,
) -> float:
    if not buoyancy_enabled:
        return 0.0
    return -rho * beta * (temperature - t_ref) * gy


@njit(cache=True, fastmath=False)
def pressure_gradients_kernel(
    p: np.ndarray,
    is_fluid: np.ndarray,
    fluid_i: np.ndarray,
    fluid_j: np.ndarray,
    dx: float,
    dy: float,
    dpdx: np.ndarray,
    dpdy: np.ndarray,
) -> None:
    dpdx.fill(0.0)
    dpdy.fill(0.0)
    ny, nx = p.shape

    for fid in range(fluid_i.size):
        i = int(fluid_i[fid])
        j = int(fluid_j[fid])

        west = i > 0 and is_fluid[j, i - 1]
        east = i < nx - 1 and is_fluid[j, i + 1]
        if west and east:
            dpdx[j, i] = (p[j, i + 1] - p[j, i - 1]) / (2.0 * dx)
        elif east:
            dpdx[j, i] = (p[j, i + 1] - p[j, i]) / dx
        elif west:
            dpdx[j, i] = (p[j, i] - p[j, i - 1]) / dx

        south = j > 0 and is_fluid[j - 1, i]
        north = j < ny - 1 and is_fluid[j + 1, i]
        if south and north:
            dpdy[j, i] = (p[j + 1, i] - p[j - 1, i]) / (2.0 * dy)
        elif north:
            dpdy[j, i] = (p[j + 1, i] - p[j, i]) / dy
        elif south:
            dpdy[j, i] = (p[j, i] - p[j - 1, i]) / dy


@njit(cache=True, fastmath=False)
def cell_gradients_kernel(
    phi: np.ndarray,
    active_mask: np.ndarray,
    dx: float,
    dy: float,
    gx: np.ndarray,
    gy: np.ndarray,
) -> None:
    gx.fill(0.0)
    gy.fill(0.0)
    ny, nx = phi.shape

    for j in range(ny):
        for i in range(nx):
            if not active_mask[j, i]:
                continue

            west = i > 0 and active_mask[j, i - 1]
            east = i < nx - 1 and active_mask[j, i + 1]
            if west and east:
                gx[j, i] = (phi[j, i + 1] - phi[j, i - 1]) / (2.0 * dx)
            elif east:
                gx[j, i] = (phi[j, i + 1] - phi[j, i]) / dx
            elif west:
                gx[j, i] = (phi[j, i] - phi[j, i - 1]) / dx

            south = j > 0 and active_mask[j - 1, i]
            north = j < ny - 1 and active_mask[j + 1, i]
            if south and north:
                gy[j, i] = (phi[j + 1, i] - phi[j - 1, i]) / (2.0 * dy)
            elif north:
                gy[j, i] = (phi[j + 1, i] - phi[j, i]) / dy
            elif south:
                gy[j, i] = (phi[j, i] - phi[j - 1, i]) / dy


@njit(cache=True, fastmath=False)
def face_fluxes_kernel(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    temperature: np.ndarray,
    is_fluid: np.ndarray,
    rho: np.ndarray,
    beta: np.ndarray,
    aPu: np.ndarray,
    aPv: np.ndarray,
    dpdx: np.ndarray,
    dpdy: np.ndarray,
    volume: float,
    dx: float,
    dy: float,
    t_ref: float,
    gy: float,
    buoyancy_enabled: bool,
    me: np.ndarray,
    mn: np.ndarray,
) -> None:
    me.fill(0.0)
    mn.fill(0.0)
    ny, nx = u.shape

    for j in range(ny):
        for i in range(nx - 1):
            if is_fluid[j, i] and is_fluid[j, i + 1]:
                dP = volume / max(aPu[j, i], _EPS)
                dE = volume / max(aPu[j, i + 1], _EPS)
                de = 0.5 * (dP + dE)
                ubar = 0.5 * (u[j, i] + u[j, i + 1])
                dp_face = (p[j, i + 1] - p[j, i]) / dx
                dp_interp = 0.5 * (dpdx[j, i] + dpdx[j, i + 1])
                ue = ubar - de * (dp_face - dp_interp)
                rho_e = 0.5 * (rho[j, i] + rho[j, i + 1])
                me[j, i] = rho_e * dy * ue

    for j in range(ny - 1):
        for i in range(nx):
            if is_fluid[j, i] and is_fluid[j + 1, i]:
                dP = volume / max(aPv[j, i], _EPS)
                dN = volume / max(aPv[j + 1, i], _EPS)
                dn = 0.5 * (dP + dN)
                vbar = 0.5 * (v[j, i] + v[j + 1, i])
                dp_face = (p[j + 1, i] - p[j, i]) / dy
                dp_interp = 0.5 * (dpdy[j, i] + dpdy[j + 1, i])

                if buoyancy_enabled:
                    byP = _body_force_y(
                        rho[j, i], beta[j, i], temperature[j, i],
                        t_ref, gy, True
                    )
                    byN = _body_force_y(
                        rho[j + 1, i], beta[j + 1, i], temperature[j + 1, i],
                        t_ref, gy, True
                    )
                    by_face = 0.5 * (byP + byN)
                    by_interp = 0.5 * (dP * byP + dN * byN)
                    vn = (
                        vbar
                        - dn * (dp_face - dp_interp)
                        + (dn * by_face - by_interp)
                    )
                else:
                    vn = vbar - dn * (dp_face - dp_interp)

                rho_n = 0.5 * (rho[j, i] + rho[j + 1, i])
                mn[j, i] = rho_n * dx * vn


@njit(cache=True, fastmath=False)
def _local_face_bounds(
    phi: np.ndarray,
    is_fluid: np.ndarray,
    i1: int,
    j1: int,
    i2: int,
    j2: int,
) -> tuple[float, float]:
    ny, nx = phi.shape
    lo = np.inf
    hi = -np.inf

    for which in range(2):
        if which == 0:
            i = i1
            j = j1
        else:
            i = i2
            j = j2

        if i < 0 or i >= nx or j < 0 or j >= ny or not is_fluid[j, i]:
            continue

        value = phi[j, i]
        if value < lo:
            lo = value
        if value > hi:
            hi = value

        if i > 0 and is_fluid[j, i - 1]:
            value = phi[j, i - 1]
            if value < lo:
                lo = value
            if value > hi:
                hi = value
        if i < nx - 1 and is_fluid[j, i + 1]:
            value = phi[j, i + 1]
            if value < lo:
                lo = value
            if value > hi:
                hi = value
        if j > 0 and is_fluid[j - 1, i]:
            value = phi[j - 1, i]
            if value < lo:
                lo = value
            if value > hi:
                hi = value
        if j < ny - 1 and is_fluid[j + 1, i]:
            value = phi[j + 1, i]
            if value < lo:
                lo = value
            if value > hi:
                hi = value

    if lo == np.inf:
        return -np.inf, np.inf
    return lo, hi


@njit(cache=True, fastmath=False)
def _sou_face_delta(
    phi: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    is_fluid: np.ndarray,
    direction: int,
    i: int,
    j: int,
    flux: float,
    dx: float,
    dy: float,
    limiter_enabled: bool,
) -> float:
    if direction == EAST:
        i1, j1 = i, j
        i2, j2 = i + 1, j
        if flux >= 0.0:
            iu, ju = i, j
            drx, dry = 0.5 * dx, 0.0
        else:
            iu, ju = i + 1, j
            drx, dry = -0.5 * dx, 0.0
    elif direction == WEST:
        i1, j1 = i - 1, j
        i2, j2 = i, j
        if flux >= 0.0:
            iu, ju = i - 1, j
            drx, dry = 0.5 * dx, 0.0
        else:
            iu, ju = i, j
            drx, dry = -0.5 * dx, 0.0
    elif direction == NORTH:
        i1, j1 = i, j
        i2, j2 = i, j + 1
        if flux >= 0.0:
            iu, ju = i, j
            drx, dry = 0.0, 0.5 * dy
        else:
            iu, ju = i, j + 1
            drx, dry = 0.0, -0.5 * dy
    else:
        i1, j1 = i, j - 1
        i2, j2 = i, j
        if flux >= 0.0:
            iu, ju = i, j - 1
            drx, dry = 0.0, 0.5 * dy
        else:
            iu, ju = i, j
            drx, dry = 0.0, -0.5 * dy

    phi_uds = phi[ju, iu]
    phi_sou = phi_uds + gx[ju, iu] * drx + gy[ju, iu] * dry
    if limiter_enabled:
        lo, hi = _local_face_bounds(phi, is_fluid, i1, j1, i2, j2)
        if phi_sou < lo:
            phi_sou = lo
        elif phi_sou > hi:
            phi_sou = hi
    return phi_sou - phi_uds


@njit(cache=True, fastmath=False)
def _sou_deferred_source(
    phi: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    is_fluid: np.ndarray,
    i: int,
    j: int,
    Fe: float,
    Fw: float,
    Fn: float,
    Fs: float,
    kind_e: int,
    kind_w: int,
    kind_n: int,
    kind_s: int,
    blend: float,
    dx: float,
    dy: float,
    limiter_enabled: bool,
) -> float:
    if blend == 0.0:
        return 0.0

    correction = 0.0
    if kind_e == FACE_FLUID_FLUID:
        correction += Fe * _sou_face_delta(
            phi, gx, gy, is_fluid, EAST, i, j, Fe, dx, dy, limiter_enabled
        )
    if kind_w == FACE_FLUID_FLUID:
        correction -= Fw * _sou_face_delta(
            phi, gx, gy, is_fluid, WEST, i, j, Fw, dx, dy, limiter_enabled
        )
    if kind_n == FACE_FLUID_FLUID:
        correction += Fn * _sou_face_delta(
            phi, gx, gy, is_fluid, NORTH, i, j, Fn, dx, dy, limiter_enabled
        )
    if kind_s == FACE_FLUID_FLUID:
        correction -= Fs * _sou_face_delta(
            phi, gx, gy, is_fluid, SOUTH, i, j, Fs, dx, dy, limiter_enabled
        )
    return -blend * correction


@njit(cache=True, fastmath=False)
def _open_mdot_e(
    i: int, j: int, u: np.ndarray, p: np.ndarray, dpdx: np.ndarray,
    aPu: np.ndarray, rho: np.ndarray, volume: float, dx: float, dy: float,
    p_bc: float,
) -> float:
    d = volume / max(aPu[j, i], _EPS)
    ue = u[j, i] - d * ((p_bc - p[j, i]) / dx - dpdx[j, i])
    return rho[j, i] * dy * ue


@njit(cache=True, fastmath=False)
def _open_mdot_w(
    i: int, j: int, u: np.ndarray, p: np.ndarray, dpdx: np.ndarray,
    aPu: np.ndarray, rho: np.ndarray, volume: float, dx: float, dy: float,
    p_bc: float,
) -> float:
    d = volume / max(aPu[j, i], _EPS)
    uw = u[j, i] - d * ((p[j, i] - p_bc) / dx - dpdx[j, i])
    return rho[j, i] * dy * uw


@njit(cache=True, fastmath=False)
def _open_mdot_n(
    i: int, j: int, v: np.ndarray, p: np.ndarray, dpdy: np.ndarray,
    aPv: np.ndarray, rho: np.ndarray, volume: float, dx: float, dy: float,
    p_bc: float,
) -> float:
    d = volume / max(aPv[j, i], _EPS)
    vn = v[j, i] - d * ((p_bc - p[j, i]) / dy - dpdy[j, i])
    return rho[j, i] * dx * vn


@njit(cache=True, fastmath=False)
def _open_mdot_s(
    i: int, j: int, v: np.ndarray, p: np.ndarray, dpdy: np.ndarray,
    aPv: np.ndarray, rho: np.ndarray, volume: float, dx: float, dy: float,
    p_bc: float,
) -> float:
    d = volume / max(aPv[j, i], _EPS)
    vs = v[j, i] - d * ((p[j, i] - p_bc) / dy - dpdy[j, i])
    return rho[j, i] * dx * vs


@njit(cache=True, fastmath=False)
def momentum_pass_kernel(
    fluid_i: np.ndarray,
    fluid_j: np.ndarray,
    face_kind: np.ndarray,
    flow_bc_code: np.ndarray,
    flow_bc_u: np.ndarray,
    flow_bc_v: np.ndarray,
    flow_bc_p: np.ndarray,
    is_fluid: np.ndarray,
    rho: np.ndarray,
    mu: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    lag_aPu: np.ndarray,
    lag_aPv: np.ndarray,
    dpdx: np.ndarray,
    dpdy: np.ndarray,
    me: np.ndarray,
    mn: np.ndarray,
    dudx: np.ndarray,
    dudy: np.ndarray,
    dvdx: np.ndarray,
    dvdy: np.ndarray,
    dx: float,
    dy: float,
    volume: float,
    sou_enabled: bool,
    sou_blend: float,
    limiter_enabled: bool,
    aE: np.ndarray,
    aW: np.ndarray,
    aN: np.ndarray,
    aS: np.ndarray,
    aPu: np.ndarray,
    aPv: np.ndarray,
    source_u: np.ndarray,
    source_v: np.ndarray,
    Fe_field: np.ndarray,
    Fw_field: np.ndarray,
    Fn_field: np.ndarray,
    Fs_field: np.ndarray,
) -> None:
    aE.fill(0.0)
    aW.fill(0.0)
    aN.fill(0.0)
    aS.fill(0.0)
    aPu.fill(1.0)
    aPv.fill(1.0)
    source_u.fill(0.0)
    source_v.fill(0.0)
    Fe_field.fill(0.0)
    Fw_field.fill(0.0)
    Fn_field.fill(0.0)
    Fs_field.fill(0.0)

    nf = fluid_i.size
    for fid in range(nf):
        i = int(fluid_i[fid])
        j = int(fluid_j[fid])
        kind_e = int(face_kind[fid, EAST])
        kind_w = int(face_kind[fid, WEST])
        kind_n = int(face_kind[fid, NORTH])
        kind_s = int(face_kind[fid, SOUTH])

        muP = mu[j, i]
        if kind_e == FACE_FLUID_FLUID:
            De = _harmonic_mean(muP, mu[j, i + 1]) * dy / dx
            Fe = me[j, i]
        else:
            De = muP * dy / dx
            if kind_e == FACE_FLUID_SOLID:
                Fe = 0.0
            else:
                code = int(flow_bc_code[EAST])
                if code == FLOW_BC_OPEN:
                    Fe = _open_mdot_e(
                        i, j, u, p, dpdx, lag_aPu, rho,
                        volume, dx, dy, flow_bc_p[EAST]
                    )
                elif code == FLOW_BC_OUTLET:
                    Fe = rho[j, i] * dy * u[j, i]
                elif code == FLOW_BC_INLET:
                    Fe = rho[j, i] * dy * flow_bc_u[EAST]
                else:
                    Fe = 0.0

        if kind_w == FACE_FLUID_FLUID:
            Dw = _harmonic_mean(mu[j, i - 1], muP) * dy / dx
            Fw = me[j, i - 1]
        else:
            Dw = muP * dy / dx
            if kind_w == FACE_FLUID_SOLID:
                Fw = 0.0
            else:
                code = int(flow_bc_code[WEST])
                if code == FLOW_BC_OPEN:
                    Fw = _open_mdot_w(
                        i, j, u, p, dpdx, lag_aPu, rho,
                        volume, dx, dy, flow_bc_p[WEST]
                    )
                elif code == FLOW_BC_INLET:
                    Fw = rho[j, i] * dy * flow_bc_u[WEST]
                elif code == FLOW_BC_OUTLET:
                    Fw = rho[j, i] * dy * u[j, i]
                else:
                    Fw = 0.0

        if kind_n == FACE_FLUID_FLUID:
            Dn = _harmonic_mean(muP, mu[j + 1, i]) * dx / dy
            Fn = mn[j, i]
        else:
            Dn = muP * dx / dy
            if kind_n == FACE_FLUID_SOLID:
                Fn = 0.0
            else:
                code = int(flow_bc_code[NORTH])
                if code == FLOW_BC_OPEN:
                    Fn = _open_mdot_n(
                        i, j, v, p, dpdy, lag_aPv, rho,
                        volume, dx, dy, flow_bc_p[NORTH]
                    )
                elif code == FLOW_BC_INLET:
                    Fn = rho[j, i] * dx * flow_bc_v[NORTH]
                elif code == FLOW_BC_OUTLET:
                    Fn = rho[j, i] * dx * v[j, i]
                else:
                    Fn = 0.0

        if kind_s == FACE_FLUID_FLUID:
            Ds = _harmonic_mean(mu[j - 1, i], muP) * dx / dy
            Fs = mn[j - 1, i]
        else:
            Ds = muP * dx / dy
            if kind_s == FACE_FLUID_SOLID:
                Fs = 0.0
            else:
                code = int(flow_bc_code[SOUTH])
                if code == FLOW_BC_OPEN:
                    Fs = _open_mdot_s(
                        i, j, v, p, dpdy, lag_aPv, rho,
                        volume, dx, dy, flow_bc_p[SOUTH]
                    )
                elif code == FLOW_BC_INLET:
                    Fs = rho[j, i] * dx * flow_bc_v[SOUTH]
                elif code == FLOW_BC_OUTLET:
                    Fs = rho[j, i] * dx * v[j, i]
                else:
                    Fs = 0.0

        Fe_field[j, i] = Fe
        Fw_field[j, i] = Fw
        Fn_field[j, i] = Fn
        Fs_field[j, i] = Fs

        ae = De + max(-Fe, 0.0) if kind_e == FACE_FLUID_FLUID else 0.0
        aw = Dw + max(Fw, 0.0) if kind_w == FACE_FLUID_FLUID else 0.0
        an = Dn + max(-Fn, 0.0) if kind_n == FACE_FLUID_FLUID else 0.0
        a_s = Ds + max(Fs, 0.0) if kind_s == FACE_FLUID_FLUID else 0.0

        Sp_u = 0.0
        Sp_v = 0.0
        Su_u = 0.0
        Su_v = 0.0

        if kind_w == FACE_FLUID_SOLID:
            coeff = 2.0 * muP * dy / dx + max(Fw, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff
        if kind_e == FACE_FLUID_SOLID:
            coeff = 2.0 * muP * dy / dx + max(-Fe, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff
        if kind_s == FACE_FLUID_SOLID:
            coeff = 2.0 * muP * dx / dy + max(Fs, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff
        if kind_n == FACE_FLUID_SOLID:
            coeff = 2.0 * muP * dx / dy + max(-Fn, 0.0)
            Sp_u -= coeff
            Sp_v -= coeff

        if kind_w == FACE_BOUNDARY:
            code = int(flow_bc_code[WEST])
            if code == FLOW_BC_WALL or code == FLOW_BC_INLET:
                coeff = 2.0 * muP * dy / dx + max(Fw, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u[WEST]
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v[WEST]
            elif code == FLOW_BC_SYMMETRY:
                Sp_u -= 2.0 * muP * dy / dx + max(Fw, 0.0)

        if kind_e == FACE_BOUNDARY:
            code = int(flow_bc_code[EAST])
            if code == FLOW_BC_WALL or code == FLOW_BC_INLET:
                coeff = 2.0 * muP * dy / dx + max(-Fe, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u[EAST]
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v[EAST]
            elif code == FLOW_BC_SYMMETRY:
                Sp_u -= 2.0 * muP * dy / dx + max(-Fe, 0.0)

        if kind_s == FACE_BOUNDARY:
            code = int(flow_bc_code[SOUTH])
            if code == FLOW_BC_WALL or code == FLOW_BC_INLET:
                coeff = 2.0 * muP * dx / dy + max(Fs, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u[SOUTH]
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v[SOUTH]
            elif code == FLOW_BC_SYMMETRY:
                Sp_v -= 2.0 * muP * dx / dy + max(Fs, 0.0)

        if kind_n == FACE_BOUNDARY:
            code = int(flow_bc_code[NORTH])
            if code == FLOW_BC_WALL or code == FLOW_BC_INLET:
                coeff = 2.0 * muP * dx / dy + max(-Fn, 0.0)
                Sp_u -= coeff
                Su_u += coeff * flow_bc_u[NORTH]
                Sp_v -= coeff
                Su_v += coeff * flow_bc_v[NORTH]
            elif code == FLOW_BC_SYMMETRY:
                Sp_v -= 2.0 * muP * dx / dy + max(-Fn, 0.0)

        ap_u = max(ae + aw + an + a_s + (Fe - Fw + Fn - Fs) - Sp_u, _EPS)
        ap_v = max(ae + aw + an + a_s + (Fe - Fw + Fn - Fs) - Sp_v, _EPS)

        if sou_enabled:
            Su_u += _sou_deferred_source(
                u, dudx, dudy, is_fluid, i, j,
                Fe, Fw, Fn, Fs, kind_e, kind_w, kind_n, kind_s,
                sou_blend, dx, dy, limiter_enabled
            )
            Su_v += _sou_deferred_source(
                v, dvdx, dvdy, is_fluid, i, j,
                Fe, Fw, Fn, Fs, kind_e, kind_w, kind_n, kind_s,
                sou_blend, dx, dy, limiter_enabled
            )

        aE[j, i] = ae
        aW[j, i] = aw
        aN[j, i] = an
        aS[j, i] = a_s
        aPu[j, i] = ap_u
        aPv[j, i] = ap_v
        source_u[j, i] = Su_u
        source_v[j, i] = Su_v


@njit(cache=True, fastmath=False)
def build_state_vector_kernel(
    fluid_i: np.ndarray,
    fluid_j: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    state: np.ndarray,
) -> None:
    for fid in range(fluid_i.size):
        i = int(fluid_i[fid])
        j = int(fluid_j[fid])
        base = 3 * fid
        state[base] = u[j, i]
        state[base + 1] = v[j, i]
        state[base + 2] = p[j, i]


@njit(cache=True, fastmath=False)
def _store_scaled_value(
    data: np.ndarray,
    pos: int,
    row: int,
    col: int,
    value: float,
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    data[pos] = left[row] * value * right[col]


@njit(cache=True, fastmath=False)
def fill_flow_local_coo_kernel(
    fid_start: int,
    fid_end: int,
    nz_start: int,
    indptr: np.ndarray,
    fluid_i: np.ndarray,
    fluid_j: np.ndarray,
    neighbor_fid: np.ndarray,
    face_kind: np.ndarray,
    flow_bc_code: np.ndarray,
    flow_bc_u: np.ndarray,
    flow_bc_v: np.ndarray,
    flow_bc_p: np.ndarray,
    rho: np.ndarray,
    beta: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    temperature: np.ndarray,
    aE: np.ndarray,
    aW: np.ndarray,
    aN: np.ndarray,
    aS: np.ndarray,
    aPu: np.ndarray,
    aPv: np.ndarray,
    source_u: np.ndarray,
    source_v: np.ndarray,
    dpdx: np.ndarray,
    dpdy: np.ndarray,
    old_state: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    dx: float,
    dy: float,
    volume: float,
    t_ref: float,
    gy: float,
    buoyancy_enabled: bool,
    pressure_reference_row: int,
    pressure_reference_value: float,
    local_data: np.ndarray,
    local_rhs: np.ndarray,
) -> None:
    local_data.fill(0.0)
    local_rhs.fill(0.0)

    for fid in range(fid_start, fid_end):
        local_fid = fid - fid_start
        i = int(fluid_i[fid])
        j = int(fluid_j[fid])
        nb_e = int(neighbor_fid[fid, EAST])
        nb_w = int(neighbor_fid[fid, WEST])
        nb_n = int(neighbor_fid[fid, NORTH])
        nb_s = int(neighbor_fid[fid, SOUTH])
        kind_e = int(face_kind[fid, EAST])
        kind_w = int(face_kind[fid, WEST])
        kind_n = int(face_kind[fid, NORTH])
        kind_s = int(face_kind[fid, SOUTH])

        ru = 3 * fid
        rv = ru + 1
        rp = ru + 2
        px = volume / dx
        py = volume / dy

        ae = aE[j, i]
        aw = aW[j, i]
        an = aN[j, i]
        a_s = aS[j, i]
        ap_u = aPu[j, i]
        ap_v = aPv[j, i]

        # ---------------- u momentum row ----------------
        pos = int(indptr[ru] - nz_start)
        ax_old = 0.0
        rhs_abs = source_u[j, i] + sx[j, i] * volume

        col = ru
        value = ap_u
        _store_scaled_value(local_data, pos, ru, col, value, left, right)
        ax_old += value * old_state[col]
        pos += 1

        if nb_e >= 0:
            col = 3 * nb_e
            value = -ae
            _store_scaled_value(local_data, pos, ru, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_w >= 0:
            col = 3 * nb_w
            value = -aw
            _store_scaled_value(local_data, pos, ru, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_n >= 0:
            col = 3 * nb_n
            value = -an
            _store_scaled_value(local_data, pos, ru, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_s >= 0:
            col = 3 * nb_s
            value = -a_s
            _store_scaled_value(local_data, pos, ru, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1

        self_p = 0.0
        if kind_e == FACE_FLUID_FLUID:
            self_p += 0.5 * px
        elif kind_e == FACE_BOUNDARY and int(flow_bc_code[EAST]) == FLOW_BC_OPEN:
            rhs_abs -= px * flow_bc_p[EAST]
        else:
            self_p += px

        if kind_w == FACE_FLUID_FLUID:
            self_p -= 0.5 * px
        elif kind_w == FACE_BOUNDARY and int(flow_bc_code[WEST]) == FLOW_BC_OPEN:
            rhs_abs += px * flow_bc_p[WEST]
        else:
            self_p -= px

        col = rp
        _store_scaled_value(local_data, pos, ru, col, self_p, left, right)
        ax_old += self_p * old_state[col]
        pos += 1

        if nb_e >= 0:
            col = 3 * nb_e + 2
            value = 0.5 * px
            _store_scaled_value(local_data, pos, ru, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_w >= 0:
            col = 3 * nb_w + 2
            value = -0.5 * px
            _store_scaled_value(local_data, pos, ru, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1

        local_rhs[3 * local_fid] = left[ru] * (rhs_abs - ax_old)

        # ---------------- v momentum row ----------------
        pos = int(indptr[rv] - nz_start)
        ax_old = 0.0
        rhs_abs = (
            source_v[j, i]
            + sy[j, i] * volume
            + _body_force_y(
                rho[j, i], beta[j, i], temperature[j, i],
                t_ref, gy, buoyancy_enabled
            ) * volume
        )

        col = rv
        value = ap_v
        _store_scaled_value(local_data, pos, rv, col, value, left, right)
        ax_old += value * old_state[col]
        pos += 1

        if nb_e >= 0:
            col = 3 * nb_e + 1
            value = -ae
            _store_scaled_value(local_data, pos, rv, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_w >= 0:
            col = 3 * nb_w + 1
            value = -aw
            _store_scaled_value(local_data, pos, rv, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_n >= 0:
            col = 3 * nb_n + 1
            value = -an
            _store_scaled_value(local_data, pos, rv, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_s >= 0:
            col = 3 * nb_s + 1
            value = -a_s
            _store_scaled_value(local_data, pos, rv, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1

        self_p = 0.0
        if kind_n == FACE_FLUID_FLUID:
            self_p += 0.5 * py
        elif kind_n == FACE_BOUNDARY and int(flow_bc_code[NORTH]) == FLOW_BC_OPEN:
            rhs_abs -= py * flow_bc_p[NORTH]
        else:
            self_p += py

        if kind_s == FACE_FLUID_FLUID:
            self_p -= 0.5 * py
        elif kind_s == FACE_BOUNDARY and int(flow_bc_code[SOUTH]) == FLOW_BC_OPEN:
            rhs_abs += py * flow_bc_p[SOUTH]
        else:
            self_p -= py

        col = rp
        _store_scaled_value(local_data, pos, rv, col, self_p, left, right)
        ax_old += self_p * old_state[col]
        pos += 1

        if nb_n >= 0:
            col = 3 * nb_n + 2
            value = 0.5 * py
            _store_scaled_value(local_data, pos, rv, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_s >= 0:
            col = 3 * nb_s + 2
            value = -0.5 * py
            _store_scaled_value(local_data, pos, rv, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1

        local_rhs[3 * local_fid + 1] = left[rv] * (rhs_abs - ax_old)

        # ---------------- continuity row ----------------
        pos = int(indptr[rp] - nz_start)
        if rp == pressure_reference_row:
            # Pattern remains fixed; non-reference entries are structural zeros.
            col = rp
            value = 1.0
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old = value * old_state[col]
            pos += 1
            # Zero remaining fixed-pattern entries in this row.
            row_end = int(indptr[rp + 1] - nz_start)
            while pos < row_end:
                local_data[pos] = 0.0
                pos += 1
            local_rhs[3 * local_fid + 2] = left[rp] * (
                pressure_reference_value - ax_old
            )
            continue

        self_p = 0.0
        self_u = 0.0
        self_v = 0.0
        e_u = e_p = 0.0
        w_u = w_p = 0.0
        n_v = n_p = 0.0
        s_v = s_p = 0.0
        rhs_abs = 0.0

        dpe_interp = 0.0
        dpw_interp = 0.0
        dpn_interp = 0.0
        dps_interp = 0.0
        byn_corr = 0.0
        bys_corr = 0.0

        if kind_e == FACE_FLUID_FLUID:
            dP = volume / max(aPu[j, i], _EPS)
            dE = volume / max(aPu[j, i + 1], _EPS)
            de = 0.5 * (dP + dE)
            rho_e = 0.5 * (rho[j, i] + rho[j, i + 1])
            aEc = rho_e * dy * de / dx
            dpe_interp = rho_e * dy * de * 0.5 * (dpdx[j, i] + dpdx[j, i + 1])
            self_p += aEc
            self_u += rho_e * dy / 2.0
            e_u = rho_e * dy / 2.0
            e_p = -aEc
        elif kind_e == FACE_BOUNDARY:
            code = int(flow_bc_code[EAST])
            rhoP = rho[j, i]
            if code == FLOW_BC_OUTLET:
                self_u += rhoP * dy
            elif code == FLOW_BC_INLET:
                rhs_abs += -rhoP * dy * flow_bc_u[EAST]
            elif code == FLOW_BC_OPEN:
                d_open = volume / max(aPu[j, i], _EPS)
                a_open = rhoP * dy * d_open / dx
                self_u += rhoP * dy
                self_p += a_open
                rhs_abs += (
                    a_open * flow_bc_p[EAST]
                    - rhoP * dy * d_open * dpdx[j, i]
                )

        if kind_w == FACE_FLUID_FLUID:
            dW = volume / max(aPu[j, i - 1], _EPS)
            dP = volume / max(aPu[j, i], _EPS)
            dw = 0.5 * (dW + dP)
            rho_w = 0.5 * (rho[j, i - 1] + rho[j, i])
            aWc = rho_w * dy * dw / dx
            dpw_interp = rho_w * dy * dw * 0.5 * (dpdx[j, i] + dpdx[j, i - 1])
            self_p += aWc
            self_u -= rho_w * dy / 2.0
            w_u = -rho_w * dy / 2.0
            w_p = -aWc
        elif kind_w == FACE_BOUNDARY:
            code = int(flow_bc_code[WEST])
            rhoP = rho[j, i]
            if code == FLOW_BC_INLET:
                rhs_abs += rhoP * dy * flow_bc_u[WEST]
            elif code == FLOW_BC_OUTLET:
                self_u -= rhoP * dy
            elif code == FLOW_BC_OPEN:
                d_open = volume / max(aPu[j, i], _EPS)
                a_open = rhoP * dy * d_open / dx
                self_u -= rhoP * dy
                self_p += a_open
                rhs_abs += (
                    a_open * flow_bc_p[WEST]
                    + rhoP * dy * d_open * dpdx[j, i]
                )

        if kind_n == FACE_FLUID_FLUID:
            dP = volume / max(aPv[j, i], _EPS)
            dN = volume / max(aPv[j + 1, i], _EPS)
            dn = 0.5 * (dP + dN)
            rho_n = 0.5 * (rho[j, i] + rho[j + 1, i])
            aNc = rho_n * dx * dn / dy
            dpn_interp = rho_n * dx * dn * 0.5 * (dpdy[j, i] + dpdy[j + 1, i])
            self_p += aNc
            self_v += rho_n * dx / 2.0
            n_v = rho_n * dx / 2.0
            n_p = -aNc
            if buoyancy_enabled:
                byP = _body_force_y(
                    rho[j, i], beta[j, i], temperature[j, i],
                    t_ref, gy, True
                )
                byN = _body_force_y(
                    rho[j + 1, i], beta[j + 1, i], temperature[j + 1, i],
                    t_ref, gy, True
                )
                by_face = 0.5 * (byP + byN)
                byn_corr = rho_n * dx * (
                    dn * by_face - 0.5 * (dP * byP + dN * byN)
                )
        elif kind_n == FACE_BOUNDARY:
            code = int(flow_bc_code[NORTH])
            rhoP = rho[j, i]
            if code == FLOW_BC_INLET:
                rhs_abs += -rhoP * dx * flow_bc_v[NORTH]
            elif code == FLOW_BC_OUTLET:
                self_v += rhoP * dx
            elif code == FLOW_BC_OPEN:
                d_open = volume / max(aPv[j, i], _EPS)
                a_open = rhoP * dx * d_open / dy
                self_v += rhoP * dx
                self_p += a_open
                rhs_abs += (
                    a_open * flow_bc_p[NORTH]
                    - rhoP * dx * d_open * dpdy[j, i]
                )

        if kind_s == FACE_FLUID_FLUID:
            dS = volume / max(aPv[j - 1, i], _EPS)
            dP = volume / max(aPv[j, i], _EPS)
            ds = 0.5 * (dS + dP)
            rho_s = 0.5 * (rho[j - 1, i] + rho[j, i])
            aSc = rho_s * dx * ds / dy
            dps_interp = rho_s * dx * ds * 0.5 * (dpdy[j - 1, i] + dpdy[j, i])
            self_p += aSc
            self_v -= rho_s * dx / 2.0
            s_v = -rho_s * dx / 2.0
            s_p = -aSc
            if buoyancy_enabled:
                byS = _body_force_y(
                    rho[j - 1, i], beta[j - 1, i], temperature[j - 1, i],
                    t_ref, gy, True
                )
                byP = _body_force_y(
                    rho[j, i], beta[j, i], temperature[j, i],
                    t_ref, gy, True
                )
                by_face = 0.5 * (byS + byP)
                bys_corr = rho_s * dx * (
                    ds * by_face - 0.5 * (dS * byS + dP * byP)
                )
        elif kind_s == FACE_BOUNDARY:
            code = int(flow_bc_code[SOUTH])
            rhoP = rho[j, i]
            if code == FLOW_BC_INLET:
                rhs_abs += rhoP * dx * flow_bc_v[SOUTH]
            elif code == FLOW_BC_OUTLET:
                self_v -= rhoP * dx
            elif code == FLOW_BC_OPEN:
                d_open = volume / max(aPv[j, i], _EPS)
                a_open = rhoP * dx * d_open / dy
                self_v -= rhoP * dx
                self_p += a_open
                rhs_abs += (
                    a_open * flow_bc_p[SOUTH]
                    + rhoP * dx * d_open * dpdy[j, i]
                )

        rhs_abs += (
            -dpe_interp + dpw_interp - dpn_interp + dps_interp
            + byn_corr - bys_corr
        )

        ax_old = 0.0
        col = rp
        value = self_p
        _store_scaled_value(local_data, pos, rp, col, value, left, right)
        ax_old += value * old_state[col]
        pos += 1

        col = ru
        value = self_u
        _store_scaled_value(local_data, pos, rp, col, value, left, right)
        ax_old += value * old_state[col]
        pos += 1

        col = rv
        value = self_v
        _store_scaled_value(local_data, pos, rp, col, value, left, right)
        ax_old += value * old_state[col]
        pos += 1

        if nb_e >= 0:
            col = 3 * nb_e
            value = e_u
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
            col = 3 * nb_e + 2
            value = e_p
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_w >= 0:
            col = 3 * nb_w
            value = w_u
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
            col = 3 * nb_w + 2
            value = w_p
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_n >= 0:
            col = 3 * nb_n + 1
            value = n_v
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
            col = 3 * nb_n + 2
            value = n_p
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
        if nb_s >= 0:
            col = 3 * nb_s + 1
            value = s_v
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1
            col = 3 * nb_s + 2
            value = s_p
            _store_scaled_value(local_data, pos, rp, col, value, left, right)
            ax_old += value * old_state[col]
            pos += 1

        local_rhs[3 * local_fid + 2] = left[rp] * (rhs_abs - ax_old)


@njit(cache=True, fastmath=False)
def update_flow_fields_kernel(
    fluid_i: np.ndarray,
    fluid_j: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    p_old: np.ndarray,
    correction: np.ndarray,
    alpha_u: float,
    alpha_v: float,
    alpha_p: float,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
) -> None:
    for fid in range(fluid_i.size):
        i = int(fluid_i[fid])
        j = int(fluid_j[fid])
        base = 3 * fid
        u[j, i] = u_old[j, i] + alpha_u * correction[base]
        v[j, i] = v_old[j, i] + alpha_v * correction[base + 1]
        p[j, i] = p_old[j, i] + alpha_p * correction[base + 2]


@njit(cache=True, fastmath=False)
def _boundary_flux_numeric(
    direction: int,
    i: int,
    j: int,
    flow_bc_code: np.ndarray,
    flow_bc_u: np.ndarray,
    flow_bc_v: np.ndarray,
    flow_bc_p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    dpdx: np.ndarray,
    dpdy: np.ndarray,
    aPu: np.ndarray,
    aPv: np.ndarray,
    rho: np.ndarray,
    volume: float,
    dx: float,
    dy: float,
) -> float:
    code = int(flow_bc_code[direction])
    rhoP = rho[j, i]
    if direction == EAST:
        if code == FLOW_BC_OPEN:
            return _open_mdot_e(i, j, u, p, dpdx, aPu, rho, volume, dx, dy, flow_bc_p[EAST])
        if code == FLOW_BC_OUTLET:
            return rhoP * dy * u[j, i]
        if code == FLOW_BC_INLET:
            return rhoP * dy * flow_bc_u[EAST]
        return 0.0
    if direction == WEST:
        if code == FLOW_BC_OPEN:
            return _open_mdot_w(i, j, u, p, dpdx, aPu, rho, volume, dx, dy, flow_bc_p[WEST])
        if code == FLOW_BC_INLET:
            return rhoP * dy * flow_bc_u[WEST]
        if code == FLOW_BC_OUTLET:
            return rhoP * dy * u[j, i]
        return 0.0
    if direction == NORTH:
        if code == FLOW_BC_OPEN:
            return _open_mdot_n(i, j, v, p, dpdy, aPv, rho, volume, dx, dy, flow_bc_p[NORTH])
        if code == FLOW_BC_INLET:
            return rhoP * dx * flow_bc_v[NORTH]
        if code == FLOW_BC_OUTLET:
            return rhoP * dx * v[j, i]
        return 0.0
    if code == FLOW_BC_OPEN:
        return _open_mdot_s(i, j, v, p, dpdy, aPv, rho, volume, dx, dy, flow_bc_p[SOUTH])
    if code == FLOW_BC_INLET:
        return rhoP * dx * flow_bc_v[SOUTH]
    if code == FLOW_BC_OUTLET:
        return rhoP * dx * v[j, i]
    return 0.0


@njit(cache=True, fastmath=False)
def mass_residual_kernel(
    fluid_i: np.ndarray,
    fluid_j: np.ndarray,
    face_kind: np.ndarray,
    flow_bc_code: np.ndarray,
    flow_bc_u: np.ndarray,
    flow_bc_v: np.ndarray,
    flow_bc_p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    dpdx: np.ndarray,
    dpdy: np.ndarray,
    aPu: np.ndarray,
    aPv: np.ndarray,
    rho: np.ndarray,
    me: np.ndarray,
    mn: np.ndarray,
    volume: float,
    dx: float,
    dy: float,
    use_pressure_reference: bool,
    p_ref_i: int,
    p_ref_j: int,
) -> float:
    result = 0.0
    for fid in range(fluid_i.size):
        i = int(fluid_i[fid])
        j = int(fluid_j[fid])
        if use_pressure_reference and i == p_ref_i and j == p_ref_j:
            continue

        kind_e = int(face_kind[fid, EAST])
        kind_w = int(face_kind[fid, WEST])
        kind_n = int(face_kind[fid, NORTH])
        kind_s = int(face_kind[fid, SOUTH])

        if kind_e == FACE_FLUID_FLUID:
            mdot_e = me[j, i]
        elif kind_e == FACE_BOUNDARY:
            mdot_e = _boundary_flux_numeric(
                EAST, i, j, flow_bc_code, flow_bc_u, flow_bc_v, flow_bc_p,
                u, v, p, dpdx, dpdy, aPu, aPv, rho, volume, dx, dy
            )
        else:
            mdot_e = 0.0

        if kind_w == FACE_FLUID_FLUID:
            mdot_w = me[j, i - 1]
        elif kind_w == FACE_BOUNDARY:
            mdot_w = _boundary_flux_numeric(
                WEST, i, j, flow_bc_code, flow_bc_u, flow_bc_v, flow_bc_p,
                u, v, p, dpdx, dpdy, aPu, aPv, rho, volume, dx, dy
            )
        else:
            mdot_w = 0.0

        if kind_n == FACE_FLUID_FLUID:
            mdot_n = mn[j, i]
        elif kind_n == FACE_BOUNDARY:
            mdot_n = _boundary_flux_numeric(
                NORTH, i, j, flow_bc_code, flow_bc_u, flow_bc_v, flow_bc_p,
                u, v, p, dpdx, dpdy, aPu, aPv, rho, volume, dx, dy
            )
        else:
            mdot_n = 0.0

        if kind_s == FACE_FLUID_FLUID:
            mdot_s = mn[j - 1, i]
        elif kind_s == FACE_BOUNDARY:
            mdot_s = _boundary_flux_numeric(
                SOUTH, i, j, flow_bc_code, flow_bc_u, flow_bc_v, flow_bc_p,
                u, v, p, dpdx, dpdy, aPu, aPv, rho, volume, dx, dy
            )
        else:
            mdot_s = 0.0

        imbalance = abs(mdot_e - mdot_w + mdot_n - mdot_s)
        if imbalance > result:
            result = imbalance
    return result


@njit(cache=True, fastmath=False)
def fill_energy_csr_kernel(
    nx: int,
    ny: int,
    indptr: np.ndarray,
    cell_face_kind: np.ndarray,
    cell_neighbor: np.ndarray,
    heat_bc_code: np.ndarray,
    heat_bc_T: np.ndarray,
    heat_bc_q: np.ndarray,
    is_fluid: np.ndarray,
    k: np.ndarray,
    qdot: np.ndarray,
    temperature: np.ndarray,
    me: np.ndarray,
    mn: np.ndarray,
    dTdx: np.ndarray,
    dTdy: np.ndarray,
    dx: float,
    dy: float,
    volume: float,
    sou_enabled: bool,
    sou_blend: float,
    limiter_enabled: bool,
    data: np.ndarray,
    rhs: np.ndarray,
) -> None:
    data.fill(0.0)
    rhs.fill(0.0)

    for j in range(ny):
        for i in range(nx):
            P = j * nx + i
            kP = k[j, i]
            kind_e = int(cell_face_kind[P, EAST])
            kind_w = int(cell_face_kind[P, WEST])
            kind_n = int(cell_face_kind[P, NORTH])
            kind_s = int(cell_face_kind[P, SOUTH])

            aE = 0.0
            aW = 0.0
            aN = 0.0
            aS = 0.0
            Su = 0.0
            Sp = 0.0

            if kind_e != FACE_BOUNDARY:
                kf = _harmonic_mean(kP, k[j, i + 1])
                De = kf * dy / dx
                if is_fluid[j, i] and kind_e == FACE_FLUID_FLUID:
                    Fe = me[j, i]
                    aE = De + max(-Fe, 0.0)
                else:
                    aE = De
            else:
                code = int(heat_bc_code[EAST])
                if code == HEAT_BC_DIRICHLET:
                    coeff = 2.0 * kP * dy / dx
                    Sp -= coeff
                    Su += coeff * heat_bc_T[EAST]
                elif code == HEAT_BC_NEUMANN:
                    Su += heat_bc_q[EAST] * dy

            if kind_w != FACE_BOUNDARY:
                kf = _harmonic_mean(kP, k[j, i - 1])
                Dw = kf * dy / dx
                if is_fluid[j, i] and kind_w == FACE_FLUID_FLUID:
                    Fw = me[j, i - 1]
                    aW = Dw + max(Fw, 0.0)
                else:
                    aW = Dw
            else:
                code = int(heat_bc_code[WEST])
                if code == HEAT_BC_DIRICHLET:
                    coeff = 2.0 * kP * dy / dx
                    Sp -= coeff
                    Su += coeff * heat_bc_T[WEST]
                elif code == HEAT_BC_NEUMANN:
                    Su += heat_bc_q[WEST] * dy

            if kind_n != FACE_BOUNDARY:
                kf = _harmonic_mean(kP, k[j + 1, i])
                Dn = kf * dx / dy
                if is_fluid[j, i] and kind_n == FACE_FLUID_FLUID:
                    Fn = mn[j, i]
                    aN = Dn + max(-Fn, 0.0)
                else:
                    aN = Dn
            else:
                code = int(heat_bc_code[NORTH])
                if code == HEAT_BC_DIRICHLET:
                    coeff = 2.0 * kP * dx / dy
                    Sp -= coeff
                    Su += coeff * heat_bc_T[NORTH]
                elif code == HEAT_BC_NEUMANN:
                    Su += heat_bc_q[NORTH] * dx

            if kind_s != FACE_BOUNDARY:
                kf = _harmonic_mean(kP, k[j - 1, i])
                Ds = kf * dx / dy
                if is_fluid[j, i] and kind_s == FACE_FLUID_FLUID:
                    Fs = mn[j - 1, i]
                    aS = Ds + max(Fs, 0.0)
                else:
                    aS = Ds
            else:
                code = int(heat_bc_code[SOUTH])
                if code == HEAT_BC_DIRICHLET:
                    coeff = 2.0 * kP * dx / dy
                    Sp -= coeff
                    Su += coeff * heat_bc_T[SOUTH]
                elif code == HEAT_BC_NEUMANN:
                    Su += heat_bc_q[SOUTH] * dx

            if sou_enabled and is_fluid[j, i]:
                Fe_corr = me[j, i] if kind_e == FACE_FLUID_FLUID else 0.0
                Fw_corr = me[j, i - 1] if kind_w == FACE_FLUID_FLUID else 0.0
                Fn_corr = mn[j, i] if kind_n == FACE_FLUID_FLUID else 0.0
                Fs_corr = mn[j - 1, i] if kind_s == FACE_FLUID_FLUID else 0.0
                Su += _sou_deferred_source(
                    temperature, dTdx, dTdy, is_fluid, i, j,
                    Fe_corr, Fw_corr, Fn_corr, Fs_corr,
                    kind_e, kind_w, kind_n, kind_s,
                    sou_blend, dx, dy, limiter_enabled
                )

            Su += qdot[j, i] * volume
            aP = max(aE + aW + aN + aS - Sp, _EPS)

            pos = int(indptr[P])
            data[pos] = aP
            pos += 1
            if int(cell_neighbor[P, EAST]) >= 0:
                data[pos] = -aE
                pos += 1
            if int(cell_neighbor[P, WEST]) >= 0:
                data[pos] = -aW
                pos += 1
            if int(cell_neighbor[P, NORTH]) >= 0:
                data[pos] = -aN
                pos += 1
            if int(cell_neighbor[P, SOUTH]) >= 0:
                data[pos] = -aS
                pos += 1
            rhs[P] = Su