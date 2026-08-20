import numpy as np

from numerical_kernels import (
    NUMBA_AVAILABLE,
    cell_gradients_kernel,
    face_fluxes_kernel,
    mass_residual_kernel,
    pressure_gradients_kernel,
)


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
# Reusable workspace and Numba-backed gradients
# ------------------------------------------------------------
def initialize_solver_workspace(ctx):
    """Allocate reusable full-field work arrays once per case."""
    ny, nx = ctx["is_fluid"].shape
    workspace = ctx.setdefault("workspace", {})

    def ensure(name, shape=(ny, nx), fill=0.0):
        array = workspace.get(name)
        if array is None or array.shape != shape:
            array = np.full(shape, fill, dtype=float)
            workspace[name] = array
        return array

    ensure("dpdx")
    ensure("dpdy")
    ensure("grad_x_1")
    ensure("grad_y_1")
    ensure("grad_x_2")
    ensure("grad_y_2")
    ensure("old_u")
    ensure("old_v")
    ensure("old_p")
    ensure("old_T")
    ensure("me", (ny, max(nx - 1, 0)))
    ensure("mn", (max(ny - 1, 0), nx))
    return workspace


def snapshot_fields(ctx, fields):
    """Copy the nonlinear state into persistent work arrays without allocation."""
    workspace = initialize_solver_workspace(ctx)
    np.copyto(workspace["old_u"], fields["u"])
    np.copyto(workspace["old_v"], fields["v"])
    np.copyto(workspace["old_p"], fields["p"])
    np.copyto(workspace["old_T"], fields["T"])
    return {
        "u": workspace["old_u"],
        "v": workspace["old_v"],
        "p": workspace["old_p"],
        "T": workspace["old_T"],
    }


def require_numba(ctx):
    if bool(ctx.get("use_numba", True)) and not NUMBA_AVAILABLE:
        raise ImportError(
            "RUN_SETTINGS requests Numba acceleration, but numba is not installed. "
            "Install it in the cfd-petsc environment before running Solver V5."
        )


def compute_pressure_gradients(ctx, p):
    """Pressure gradients on owned + ghost cells.

    In Phase F the ghost layer must also receive valid gradients because
    Rhie-Chow interpolation at partition interfaces reads neighbour gradients.
    """
    require_numba(ctx)
    workspace = initialize_solver_workspace(ctx)
    dpdx = workspace["dpdx"]
    dpdy = workspace["dpdy"]
    cell_gradients_kernel(
        np.asarray(p, dtype=float),
        np.asarray(ctx["is_fluid"], dtype=np.bool_),
        float(ctx["dx"]),
        float(ctx["dy"]),
        dpdx,
        dpdy,
    )
    return dpdx, dpdy


def compute_cell_gradients(ctx, phi, active_mask, workspace_slot=1):
    require_numba(ctx)
    workspace = initialize_solver_workspace(ctx)
    slot = 1 if int(workspace_slot) == 1 else 2
    gx = workspace[f"grad_x_{slot}"]
    gy = workspace[f"grad_y_{slot}"]
    cell_gradients_kernel(
        np.asarray(phi, dtype=float),
        np.asarray(active_mask, dtype=np.bool_),
        float(ctx["dx"]),
        float(ctx["dy"]),
        gx,
        gy,
    )
    return gx, gy


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
# Rhie-Chow face fluxes — Numba kernel with persistent arrays
# ------------------------------------------------------------
def compute_face_fluxes(ctx, settings, fields, coeffs, gradients):
    require_numba(ctx)
    workspace = initialize_solver_workspace(ctx)
    me = workspace["me"]
    mn = workspace["mn"]
    face_fluxes_kernel(
        fields["u"],
        fields["v"],
        fields["p"],
        fields["T"],
        ctx["is_fluid"],
        ctx["rho"],
        ctx["beta"],
        coeffs["aPu"],
        coeffs["aPv"],
        gradients["dpdx"],
        gradients["dpdy"],
        float(ctx["V"]),
        float(ctx["dx"]),
        float(ctx["dy"]),
        float(ctx["T_ref"]),
        float(ctx["gy"]),
        bool(settings["physics"].get("buoyancy", False)),
        me,
        mn,
    )
    return {"me": me, "mn": mn}


# ------------------------------------------------------------
# Residuals and balances
# ------------------------------------------------------------
def compute_mass_residual(ctx, settings, fields, coeffs, gradients, fluxes=None):
    """Global maximum continuity imbalance from local owned fluid cells."""
    require_numba(ctx)
    if fluxes is None:
        fluxes = compute_face_fluxes(ctx, settings, fields, coeffs, gradients)
    topology = ctx["topology"]

    p_ref_i = int(ctx.get("p_ref_i", -1))
    p_ref_j = int(ctx.get("p_ref_j", -1))
    domain = ctx.get("domain")
    if domain is not None:
        if domain.owns_global_j(p_ref_j):
            p_ref_j_kernel = domain.global_to_local_j(p_ref_j)
        else:
            p_ref_j_kernel = -10**9
    else:
        p_ref_j_kernel = p_ref_j

    local_value = float(
        mass_residual_kernel(
            topology["fluid_i"],
            topology["fluid_j"],
            topology["face_kind"],
            topology["flow_bc_code"],
            topology["flow_bc_u"],
            topology["flow_bc_v"],
            topology["flow_bc_p"],
            fields["u"],
            fields["v"],
            fields["p"],
            gradients["dpdx"],
            gradients["dpdy"],
            coeffs["aPu"],
            coeffs["aPv"],
            ctx["rho"],
            fluxes["me"],
            fluxes["mn"],
            float(ctx["V"]),
            float(ctx["dx"]),
            float(ctx["dy"]),
            bool(ctx.get("use_pressure_reference", False)),
            p_ref_i,
            int(p_ref_j_kernel),
        )
    )
    return domain.allreduce_max(local_value) if domain is not None else local_value


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