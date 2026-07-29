import numpy as np


# ============================================================
# GENERIC SOLVER UTILITIES
# ------------------------------------------------------------
# Shared by steady and future transient solvers.
# Uses ctx/settings/fields/coeffs dictionaries instead of solver globals.
# ============================================================


# ------------------------------------------------------------
# Basic topology / masks
# ------------------------------------------------------------
def inside(ctx, i, j):
    return 0 <= i < ctx["nx"] and 0 <= j < ctx["ny"]


def fluid_at(ctx, i, j):
    return inside(ctx, i, j) and ctx["is_fluid"][j, i]


def solid_at(ctx, i, j):
    return inside(ctx, i, j) and ctx["is_solid"][j, i]


# ------------------------------------------------------------
# Boundary accessors
# ------------------------------------------------------------
def boundary_flow_type(ctx, side):
    return ctx["BC_flow"][side]["type"].lower()


def boundary_heat_type(ctx, side):
    return ctx["BC_heat"][side]["type"].lower()


# Backward-compatible names used by the solver
def flow_bc_type(ctx, side):
    return boundary_flow_type(ctx, side)


def flow_bc_u(ctx, side):
    return float(ctx["BC_flow"][side].get("u", 0.0))


def flow_bc_v(ctx, side):
    return float(ctx["BC_flow"][side].get("v", 0.0))


def flow_bc_p(ctx, side, default=0.0):
    return float(ctx["BC_flow"][side].get("p", default))


def heat_bc_type(ctx, side):
    return boundary_heat_type(ctx, side)


def heat_bc_T(ctx, side, default=0.0):
    return float(ctx["BC_heat"][side].get("T", default))


def heat_bc_q(ctx, side, default=0.0):
    return float(ctx["BC_heat"][side].get("q", default))


# ------------------------------------------------------------
# Uniform-grid geometry accessors
# ------------------------------------------------------------
# These return the current uniform-grid values. Later, only these
# functions need to change to read local non-uniform grid arrays.
def cell_volume(ctx, i, j):
    return ctx["V"]


def area_e(ctx, i, j):
    return ctx["dy"]


def area_w(ctx, i, j):
    return ctx["dy"]


def area_n(ctx, i, j):
    return ctx["dx"]


def area_s(ctx, i, j):
    return ctx["dx"]


def dist_e(ctx, i, j):
    return ctx["dx"]


def dist_w(ctx, i, j):
    return ctx["dx"]


def dist_n(ctx, i, j):
    return ctx["dy"]


def dist_s(ctx, i, j):
    return ctx["dy"]


# ------------------------------------------------------------
# Numerical coefficients / interpolation
# ------------------------------------------------------------
def upwind_aW(Fw, Dw):
    return Dw + max(Fw, 0.0)


def upwind_aE(Fe, De):
    return De + max(-Fe, 0.0)


def upwind_aS(Fs, Ds):
    return Ds + max(Fs, 0.0)


def upwind_aN(Fn, Dn):
    return Dn + max(-Fn, 0.0)


def harmonic_mean(a, b):
    if a + b == 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def face_rho_x(ctx, i, j):
    rho = ctx["rho"]
    return 0.5 * (rho[j, i] + rho[j, i + 1])


def face_rho_y(ctx, i, j):
    rho = ctx["rho"]
    return 0.5 * (rho[j, i] + rho[j + 1, i])


def face_mu_e(ctx, i, j):
    mu = ctx["mu"]
    return harmonic_mean(mu[j, i], mu[j, i + 1])


def face_mu_w(ctx, i, j):
    mu = ctx["mu"]
    return harmonic_mean(mu[j, i - 1], mu[j, i])


def face_mu_n(ctx, i, j):
    mu = ctx["mu"]
    return harmonic_mean(mu[j, i], mu[j + 1, i])


def face_mu_s(ctx, i, j):
    mu = ctx["mu"]
    return harmonic_mean(mu[j - 1, i], mu[j, i])


def cell_rho(ctx, i, j):
    return ctx["rho"][j, i]


def cell_mu(ctx, i, j):
    return ctx["mu"][j, i]


# ------------------------------------------------------------
# Face kind helpers
# ------------------------------------------------------------
def east_face_kind(ctx, i, j):
    if i == ctx["nx"] - 1:
        return "boundary-east"
    if fluid_at(ctx, i + 1, j):
        return "fluid-fluid"
    if solid_at(ctx, i + 1, j):
        return "fluid-solid"
    return "boundary-east"


def west_face_kind(ctx, i, j):
    if i == 0:
        return "boundary-west"
    if fluid_at(ctx, i - 1, j):
        return "fluid-fluid"
    if solid_at(ctx, i - 1, j):
        return "fluid-solid"
    return "boundary-west"


def north_face_kind(ctx, i, j):
    if j == ctx["ny"] - 1:
        return "boundary-north"
    if fluid_at(ctx, i, j + 1):
        return "fluid-fluid"
    if solid_at(ctx, i, j + 1):
        return "fluid-solid"
    return "boundary-north"


def south_face_kind(ctx, i, j):
    if j == 0:
        return "boundary-south"
    if fluid_at(ctx, i, j - 1):
        return "fluid-fluid"
    if solid_at(ctx, i, j - 1):
        return "fluid-solid"
    return "boundary-south"


# ------------------------------------------------------------
# Gradients
# ------------------------------------------------------------
def compute_pressure_gradients(ctx, p):
    nx = ctx["nx"]
    ny = ctx["ny"]
    dx = ctx["dx"]
    dy = ctx["dy"]
    fluid_cells = ctx["fluid_cells"]

    dpdx = np.zeros((ny, nx))
    dpdy = np.zeros((ny, nx))

    for (i, j) in fluid_cells:
        if fluid_at(ctx, i - 1, j) and fluid_at(ctx, i + 1, j):
            dpdx[j, i] = (p[j, i + 1] - p[j, i - 1]) / (2.0 * dx)
        elif fluid_at(ctx, i + 1, j):
            dpdx[j, i] = (p[j, i + 1] - p[j, i]) / dx
        elif fluid_at(ctx, i - 1, j):
            dpdx[j, i] = (p[j, i] - p[j, i - 1]) / dx
        else:
            dpdx[j, i] = 0.0

        if fluid_at(ctx, i, j - 1) and fluid_at(ctx, i, j + 1):
            dpdy[j, i] = (p[j + 1, i] - p[j - 1, i]) / (2.0 * dy)
        elif fluid_at(ctx, i, j + 1):
            dpdy[j, i] = (p[j + 1, i] - p[j, i]) / dy
        elif fluid_at(ctx, i, j - 1):
            dpdy[j, i] = (p[j, i] - p[j - 1, i]) / dy
        else:
            dpdy[j, i] = 0.0

    return dpdx, dpdy


def compute_cell_gradients(ctx, phi, active_mask):
    nx = ctx["nx"]
    ny = ctx["ny"]
    dx = ctx["dx"]
    dy = ctx["dy"]

    gx_phi = np.zeros((ny, nx))
    gy_phi = np.zeros((ny, nx))

    for j in range(ny):
        for i in range(nx):
            if not active_mask[j, i]:
                continue

            if i > 0 and active_mask[j, i - 1] and i < nx - 1 and active_mask[j, i + 1]:
                gx_phi[j, i] = (phi[j, i + 1] - phi[j, i - 1]) / (2.0 * dx)
            elif i < nx - 1 and active_mask[j, i + 1]:
                gx_phi[j, i] = (phi[j, i + 1] - phi[j, i]) / dx
            elif i > 0 and active_mask[j, i - 1]:
                gx_phi[j, i] = (phi[j, i] - phi[j, i - 1]) / dx
            else:
                gx_phi[j, i] = 0.0

            if j > 0 and active_mask[j - 1, i] and j < ny - 1 and active_mask[j + 1, i]:
                gy_phi[j, i] = (phi[j + 1, i] - phi[j - 1, i]) / (2.0 * dy)
            elif j < ny - 1 and active_mask[j + 1, i]:
                gy_phi[j, i] = (phi[j + 1, i] - phi[j, i]) / dy
            elif j > 0 and active_mask[j - 1, i]:
                gy_phi[j, i] = (phi[j, i] - phi[j - 1, i]) / dy
            else:
                gy_phi[j, i] = 0.0

    return gx_phi, gy_phi


# ------------------------------------------------------------
# Second-order upwind deferred correction
# ------------------------------------------------------------
def local_bounds_for_face(ctx, phi, cells):
    vals = []
    for (i, j) in cells:
        if fluid_at(ctx, i, j):
            vals.append(phi[j, i])
            for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if fluid_at(ctx, ii, jj):
                    vals.append(phi[jj, ii])

    if not vals:
        return -np.inf, np.inf
    return min(vals), max(vals)


def sou_face_value(ctx, phi, gx_phi, gy_phi, face, i, j, F):
    dx = ctx["dx"]
    dy = ctx["dy"]

    if face == "e":
        cells = [(i, j), (i + 1, j)]
        if F >= 0.0:
            iu, ju = i, j
            drx, dry = 0.5 * dx, 0.0
        else:
            iu, ju = i + 1, j
            drx, dry = -0.5 * dx, 0.0
    elif face == "w":
        cells = [(i - 1, j), (i, j)]
        if F >= 0.0:
            iu, ju = i - 1, j
            drx, dry = 0.5 * dx, 0.0
        else:
            iu, ju = i, j
            drx, dry = -0.5 * dx, 0.0
    elif face == "n":
        cells = [(i, j), (i, j + 1)]
        if F >= 0.0:
            iu, ju = i, j
            drx, dry = 0.0, 0.5 * dy
        else:
            iu, ju = i, j + 1
            drx, dry = 0.0, -0.5 * dy
    elif face == "s":
        cells = [(i, j - 1), (i, j)]
        if F >= 0.0:
            iu, ju = i, j - 1
            drx, dry = 0.0, 0.5 * dy
        else:
            iu, ju = i, j
            drx, dry = 0.0, -0.5 * dy
    else:
        raise ValueError(f"Unknown face: {face}")

    phi_uds = phi[ju, iu]
    phi_sou = phi_uds + gx_phi[ju, iu] * drx + gy_phi[ju, iu] * dry

    if ctx.get("enable_sou_limiter", True):
        lo, hi = local_bounds_for_face(ctx, phi, cells)
        phi_sou = min(max(phi_sou, lo), hi)

    return phi_sou, phi_uds


def sou_deferred_source(ctx, phi, gx_phi, gy_phi, i, j, Fe, Fw, Fn, Fs,
                        kind_e, kind_w, kind_n, kind_s, blend):
    if blend == 0.0:
        return 0.0

    corr = 0.0
    if kind_e == "fluid-fluid":
        phi_sou, phi_uds = sou_face_value(ctx, phi, gx_phi, gy_phi, "e", i, j, Fe)
        corr += Fe * (phi_sou - phi_uds)
    if kind_w == "fluid-fluid":
        phi_sou, phi_uds = sou_face_value(ctx, phi, gx_phi, gy_phi, "w", i, j, Fw)
        corr -= Fw * (phi_sou - phi_uds)
    if kind_n == "fluid-fluid":
        phi_sou, phi_uds = sou_face_value(ctx, phi, gx_phi, gy_phi, "n", i, j, Fn)
        corr += Fn * (phi_sou - phi_uds)
    if kind_s == "fluid-fluid":
        phi_sou, phi_uds = sou_face_value(ctx, phi, gx_phi, gy_phi, "s", i, j, Fs)
        corr -= Fs * (phi_sou - phi_uds)

    return -blend * corr


# ------------------------------------------------------------
# Physics helpers
# ------------------------------------------------------------
def body_force_y(ctx, settings, i, j, Tval):
    if not settings["physics"].get("buoyancy", False):
        return 0.0
    rho_loc = ctx["rho"][j, i]
    beta_loc = ctx["beta"][j, i]
    return -rho_loc * beta_loc * (Tval - ctx["T_ref"]) * ctx["gy"]


# ------------------------------------------------------------
# Boundary and open-boundary mass fluxes
# ------------------------------------------------------------
def open_mdot_e(ctx, fields, coeffs, gradients, i, j):
    u = fields["u"]
    p = fields["p"]
    dpdx = gradients["dpdx"]
    aPu = coeffs["aPu"]
    rhoP = cell_rho(ctx, i, j)
    d = cell_volume(ctx, i, j) / max(aPu[j, i], 1e-30)
    ue = u[j, i] - d * ((flow_bc_p(ctx, "east") - p[j, i]) / dist_e(ctx, i, j) - dpdx[j, i])
    return rhoP * area_e(ctx, i, j) * ue


def open_mdot_w(ctx, fields, coeffs, gradients, i, j):
    u = fields["u"]
    p = fields["p"]
    dpdx = gradients["dpdx"]
    aPu = coeffs["aPu"]
    rhoP = cell_rho(ctx, i, j)
    d = cell_volume(ctx, i, j) / max(aPu[j, i], 1e-30)
    uw = u[j, i] - d * ((p[j, i] - flow_bc_p(ctx, "west")) / dist_w(ctx, i, j) - dpdx[j, i])
    return rhoP * area_w(ctx, i, j) * uw


def open_mdot_n(ctx, fields, coeffs, gradients, i, j):
    v = fields["v"]
    p = fields["p"]
    dpdy = gradients["dpdy"]
    aPv = coeffs["aPv"]
    rhoP = cell_rho(ctx, i, j)
    d = cell_volume(ctx, i, j) / max(aPv[j, i], 1e-30)
    vn = v[j, i] - d * ((flow_bc_p(ctx, "north") - p[j, i]) / dist_n(ctx, i, j) - dpdy[j, i])
    return rhoP * area_n(ctx, i, j) * vn


def open_mdot_s(ctx, fields, coeffs, gradients, i, j):
    v = fields["v"]
    p = fields["p"]
    dpdy = gradients["dpdy"]
    aPv = coeffs["aPv"]
    rhoP = cell_rho(ctx, i, j)
    d = cell_volume(ctx, i, j) / max(aPv[j, i], 1e-30)
    vs = v[j, i] - d * ((p[j, i] - flow_bc_p(ctx, "south")) / dist_s(ctx, i, j) - dpdy[j, i])
    return rhoP * area_s(ctx, i, j) * vs


def boundary_mdot(ctx, side, i, j, fields, coeffs, gradients):
    """
    Boundary mass flux with the same sign convention as finite-volume faces:
    east/north positive leaves the cell; west/south positive enters the cell.
    """
    t = flow_bc_type(ctx, side)
    rhoP = cell_rho(ctx, i, j)

    if side == "east":
        if t == "open":
            return open_mdot_e(ctx, fields, coeffs, gradients, i, j)
        if t == "outlet":
            return rhoP * area_e(ctx, i, j) * fields["u"][j, i]
        if t == "inlet":
            return rhoP * area_e(ctx, i, j) * flow_bc_u(ctx, "east")
        return 0.0

    if side == "west":
        if t == "open":
            return open_mdot_w(ctx, fields, coeffs, gradients, i, j)
        if t == "inlet":
            return rhoP * area_w(ctx, i, j) * flow_bc_u(ctx, "west")
        if t == "outlet":
            return rhoP * area_w(ctx, i, j) * fields["u"][j, i]
        return 0.0

    if side == "north":
        if t == "open":
            return open_mdot_n(ctx, fields, coeffs, gradients, i, j)
        if t == "inlet":
            return rhoP * area_n(ctx, i, j) * flow_bc_v(ctx, "north")
        if t == "outlet":
            return rhoP * area_n(ctx, i, j) * fields["v"][j, i]
        return 0.0

    if side == "south":
        if t == "open":
            return open_mdot_s(ctx, fields, coeffs, gradients, i, j)
        if t == "inlet":
            return rhoP * area_s(ctx, i, j) * flow_bc_v(ctx, "south")
        if t == "outlet":
            return rhoP * area_s(ctx, i, j) * fields["v"][j, i]
        return 0.0

    raise ValueError(f"Unknown boundary side: {side}")


# ------------------------------------------------------------
# Rhie-Chow face fluxes
# ------------------------------------------------------------
def compute_face_fluxes(ctx, settings, fields, coeffs, gradients):
    nx = ctx["nx"]
    ny = ctx["ny"]
    me = np.zeros((ny, nx - 1))
    mn = np.zeros((ny - 1, nx))

    u = fields["u"]
    v = fields["v"]
    p = fields["p"]
    T = fields["T"]
    aPu = coeffs["aPu"]
    aPv = coeffs["aPv"]
    dpdx = gradients["dpdx"]
    dpdy = gradients["dpdy"]

    for j in range(ny):
        for i in range(nx - 1):
            if fluid_at(ctx, i, j) and fluid_at(ctx, i + 1, j):
                dP = cell_volume(ctx, i, j) / max(aPu[j, i], 1e-30)
                dE = cell_volume(ctx, i + 1, j) / max(aPu[j, i + 1], 1e-30)
                de = 0.5 * (dP + dE)
                ubar = 0.5 * (u[j, i] + u[j, i + 1])
                dp_face = (p[j, i + 1] - p[j, i]) / dist_e(ctx, i, j)
                dp_interp = 0.5 * (dpdx[j, i] + dpdx[j, i + 1])
                ue = ubar - de * (dp_face - dp_interp)
                rho_e = face_rho_x(ctx, i, j)
                me[j, i] = rho_e * area_e(ctx, i, j) * ue

    for j in range(ny - 1):
        for i in range(nx):
            if fluid_at(ctx, i, j) and fluid_at(ctx, i, j + 1):
                dP = cell_volume(ctx, i, j) / max(aPv[j, i], 1e-30)
                dN = cell_volume(ctx, i, j + 1) / max(aPv[j + 1, i], 1e-30)
                dn = 0.5 * (dP + dN)
                vbar = 0.5 * (v[j, i] + v[j + 1, i])
                dp_face = (p[j + 1, i] - p[j, i]) / dist_n(ctx, i, j)
                dp_interp = 0.5 * (dpdy[j, i] + dpdy[j + 1, i])

                if settings["physics"].get("buoyancy", False):
                    ByP = body_force_y(ctx, settings, i, j, T[j, i])
                    ByN = body_force_y(ctx, settings, i, j, T[j + 1, i])
                    By_face = 0.5 * (ByP + ByN)
                    by_interp_term = 0.5 * (dP * ByP + dN * ByN)
                    vn = vbar - dn * (dp_face - dp_interp) + (dn * By_face - by_interp_term)
                else:
                    vn = vbar - dn * (dp_face - dp_interp)

                rho_n = face_rho_y(ctx, i, j)
                mn[j, i] = rho_n * area_n(ctx, i, j) * vn

    return {"me": me, "mn": mn}


# ------------------------------------------------------------
# Residuals and balances
# ------------------------------------------------------------
def compute_mass_residual(ctx, settings, fields, coeffs, gradients):
    u = fields["u"]
    v = fields["v"]
    p = fields["p"]
    T = fields["T"]
    aPu = coeffs["aPu"]
    aPv = coeffs["aPv"]
    dpdx = gradients["dpdx"]
    dpdy = gradients["dpdy"]

    mass_res = 0.0

    for (i, j) in ctx["fluid_cells"]:
        if ctx.get("use_pressure_reference", False) and i == ctx.get("p_ref_i") and j == ctx.get("p_ref_j"):
            continue

        kind_e = east_face_kind(ctx, i, j)
        kind_w = west_face_kind(ctx, i, j)
        kind_n = north_face_kind(ctx, i, j)
        kind_s = south_face_kind(ctx, i, j)

        if kind_e == "fluid-fluid":
            dP = cell_volume(ctx, i, j) / max(aPu[j, i], 1e-30)
            dE = cell_volume(ctx, i + 1, j) / max(aPu[j, i + 1], 1e-30)
            de = 0.5 * (dP + dE)
            rc_grad_e = 0.5 * (dpdx[j, i] + dpdx[j, i + 1])
            rho_e = face_rho_x(ctx, i, j)
            mdot_e = rho_e * area_e(ctx, i, j) * (
                0.5 * (u[j, i] + u[j, i + 1]) - de * ((p[j, i + 1] - p[j, i]) / dist_e(ctx, i, j) - rc_grad_e)
            )
        elif kind_e == "boundary-east":
            mdot_e = boundary_mdot(ctx, "east", i, j, fields, coeffs, gradients)
        else:
            mdot_e = 0.0

        if kind_w == "fluid-fluid":
            dW = cell_volume(ctx, i - 1, j) / max(aPu[j, i - 1], 1e-30)
            dP = cell_volume(ctx, i, j) / max(aPu[j, i], 1e-30)
            dw = 0.5 * (dW + dP)
            rc_grad_w = 0.5 * (dpdx[j, i - 1] + dpdx[j, i])
            rho_w = face_rho_x(ctx, i - 1, j)
            mdot_w = rho_w * area_w(ctx, i, j) * (
                0.5 * (u[j, i - 1] + u[j, i]) - dw * ((p[j, i] - p[j, i - 1]) / dist_w(ctx, i, j) - rc_grad_w)
            )
        elif kind_w == "boundary-west":
            mdot_w = boundary_mdot(ctx, "west", i, j, fields, coeffs, gradients)
        else:
            mdot_w = 0.0

        if kind_n == "fluid-fluid":
            dP = cell_volume(ctx, i, j) / max(aPv[j, i], 1e-30)
            dN = cell_volume(ctx, i, j + 1) / max(aPv[j + 1, i], 1e-30)
            dn = 0.5 * (dP + dN)
            rc_grad_n = 0.5 * (dpdy[j, i] + dpdy[j + 1, i])
            rho_n = face_rho_y(ctx, i, j)

            if settings["physics"].get("buoyancy", False):
                ByP = body_force_y(ctx, settings, i, j, T[j, i])
                ByN = body_force_y(ctx, settings, i, j, T[j + 1, i])
                By_face = 0.5 * (ByP + ByN)
                by_interp_term = 0.5 * (dP * ByP + dN * ByN)
                mdot_n = rho_n * area_n(ctx, i, j) * (
                    0.5 * (v[j, i] + v[j + 1, i])
                    - dn * ((p[j + 1, i] - p[j, i]) / dist_n(ctx, i, j) - rc_grad_n)
                    + (dn * By_face - by_interp_term)
                )
            else:
                mdot_n = rho_n * area_n(ctx, i, j) * (
                    0.5 * (v[j, i] + v[j + 1, i])
                    - dn * ((p[j + 1, i] - p[j, i]) / dist_n(ctx, i, j) - rc_grad_n)
                )
        elif kind_n == "boundary-north":
            mdot_n = boundary_mdot(ctx, "north", i, j, fields, coeffs, gradients)
        else:
            mdot_n = 0.0

        if kind_s == "fluid-fluid":
            dS = cell_volume(ctx, i, j - 1) / max(aPv[j - 1, i], 1e-30)
            dP = cell_volume(ctx, i, j) / max(aPv[j, i], 1e-30)
            ds = 0.5 * (dS + dP)
            rc_grad_s = 0.5 * (dpdy[j - 1, i] + dpdy[j, i])
            rho_s = face_rho_y(ctx, i, j - 1)

            if settings["physics"].get("buoyancy", False):
                ByS = body_force_y(ctx, settings, i, j, T[j - 1, i])
                ByP = body_force_y(ctx, settings, i, j, T[j, i])
                By_face = 0.5 * (ByS + ByP)
                by_interp_term = 0.5 * (dS * ByS + dP * ByP)
                mdot_s = rho_s * area_s(ctx, i, j) * (
                    0.5 * (v[j - 1, i] + v[j, i])
                    - ds * ((p[j, i] - p[j - 1, i]) / dist_s(ctx, i, j) - rc_grad_s)
                    + (ds * By_face - by_interp_term)
                )
            else:
                mdot_s = rho_s * area_s(ctx, i, j) * (
                    0.5 * (v[j - 1, i] + v[j, i])
                    - ds * ((p[j, i] - p[j - 1, i]) / dist_s(ctx, i, j) - rc_grad_s)
                )
        elif kind_s == "boundary-south":
            mdot_s = boundary_mdot(ctx, "south", i, j, fields, coeffs, gradients)
        else:
            mdot_s = 0.0

        mass_res = max(mass_res, abs(mdot_e - mdot_w + mdot_n - mdot_s))

    return mass_res


def compute_global_mass_balance(ctx, fields, coeffs, gradients):
    west_in = west_out = 0.0
    east_in = east_out = 0.0
    south_in = south_out = 0.0
    north_in = north_out = 0.0

    nx = ctx["nx"]
    ny = ctx["ny"]

    for j in range(ny):
        mdot = boundary_mdot(ctx, "west", 0, j, fields, coeffs, gradients)
        if mdot >= 0.0:
            west_in += mdot
        else:
            west_out += -mdot

    for j in range(ny):
        mdot = boundary_mdot(ctx, "east", nx - 1, j, fields, coeffs, gradients)
        if mdot >= 0.0:
            east_out += mdot
        else:
            east_in += -mdot

    for i in range(nx):
        mdot = boundary_mdot(ctx, "south", i, 0, fields, coeffs, gradients)
        if mdot >= 0.0:
            south_in += mdot
        else:
            south_out += -mdot

    for i in range(nx):
        mdot = boundary_mdot(ctx, "north", i, ny - 1, fields, coeffs, gradients)
        if mdot >= 0.0:
            north_out += mdot
        else:
            north_in += -mdot

    total_in = west_in + east_in + south_in + north_in
    total_out = west_out + east_out + south_out + north_out
    net = total_out - total_in
    denom = max(total_in, total_out, 1e-30)

    return {
        "west_in": west_in, "west_out": west_out,
        "east_in": east_in, "east_out": east_out,
        "south_in": south_in, "south_out": south_out,
        "north_in": north_in, "north_out": north_out,
        "total_in": total_in,
        "total_out": total_out,
        "net": net,
        "rel_imbalance": abs(net) / denom,
    }
