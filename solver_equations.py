# ============================================================
# COUPLED FLOW AND ENERGY EQUATIONS — DIRECT-LU PRODUCTION PATH
# ============================================================

from __future__ import annotations

import time

import numpy as np

from flow_assembly import assemble_flow_correction_system
from solver_utils import compute_face_fluxes, compute_pressure_gradients


def _component_indices(nf: int):
    cells = np.arange(nf, dtype=int)
    return 3 * cells, 3 * cells + 1, 3 * cells + 2


def solve_pressure_velocity(
    ctx,
    settings,
    fields,
    coeffs,
    transient=None,
    linear_solver=None,
):
    """Solve one nonlinear iteration of the fully coupled direct-LU system."""
    if transient is not None:
        raise NotImplementedError(
            "Transient pressure-velocity terms are reserved for the transient solver."
        )
    if linear_solver is None:
        raise ValueError(
            "A PetscDirectSolver instance is required for the flow solve."
        )

    timing_enabled = bool(settings.get("profiling", {}).get("enabled", False))
    timing = {}
    total_start = time.perf_counter()

    fields_old = {
        "u": fields["u"].copy(),
        "v": fields["v"].copy(),
        "p": fields["p"].copy(),
    }

    assembly_start = time.perf_counter()
    system = assemble_flow_correction_system(
        ctx,
        settings,
        fields,
        coeffs,
        distributed=bool(linear_solver.uses_local_flow_assembly()),
    )
    timing["flow_assembly_total"] = time.perf_counter() - assembly_start
    timing["flow_matrix_alloc"] = 0.0
    timing["flow_assembly_loop"] = timing["flow_assembly_total"]
    timing["flow_pressure_reference"] = 0.0
    timing["flow_initial_guess"] = 0.0
    timing["flow_pre_flux"] = 0.0

    linear_start = time.perf_counter()
    if system.is_distributed_local:
        scaled_correction = linear_solver.solve(
            system,
            None,
            system_type="flow_coupled",
            metadata=system.metadata,
        )
    else:
        scaled_correction = linear_solver.solve(
            system.scaled_matrix,
            system.scaled_rhs,
            system_type="flow_coupled",
            metadata=system.metadata,
        )
    timing["flow_linear_solve"] = time.perf_counter() - linear_start

    if not np.all(np.isfinite(scaled_correction)):
        raise RuntimeError("The direct linear solver returned non-finite corrections.")

    correction = system.recover_correction(scaled_correction)
    nf = int(ctx["Nf"])
    backend_info = linear_solver.last_info
    backend_extra = backend_info.extra if backend_info is not None else {}
    true_rel = float(
        backend_extra.get("unscaled_true_rel_residual", np.inf)
    )
    allowed = float(
        settings.get("linear_solver", {})
        .get("flow_coupled", {})
        .get("true_residual_tolerance", 1.0e-5)
    )
    if true_rel > allowed:
        raise RuntimeError(
            "The coupled correction does not satisfy the original unscaled system: "
            f"true relative infinity residual={true_rel:.6e}, "
            f"allowed={allowed:.6e}."
        )

    alpha_u = float(settings["relaxation"]["u"])
    alpha_v = float(settings["relaxation"]["v"])
    alpha_p = float(settings["relaxation"]["p"])

    update_start = time.perf_counter()
    for fid, (i, j) in enumerate(ctx["fluid_cells"]):
        fields["u"][j, i] = (
            fields_old["u"][j, i] + alpha_u * correction[3 * fid]
        )
        fields["v"][j, i] = (
            fields_old["v"][j, i] + alpha_v * correction[3 * fid + 1]
        )
        fields["p"][j, i] = (
            fields_old["p"][j, i] + alpha_p * correction[3 * fid + 2]
        )

    fields["u"][ctx["is_solid"]] = 0.0
    fields["v"][ctx["is_solid"]] = 0.0
    fields["p"][ctx["is_solid"]] = 0.0

    if system.pressure_constraint_mode == "pin":
        pressure_cfg = settings.get("pressure_reference", {})
        i_ref = int(pressure_cfg.get("i", ctx.get("p_ref_i", 0)))
        j_ref = int(pressure_cfg.get("j", ctx.get("p_ref_j", 0)))
        if (
            0 <= i_ref < ctx["nx"]
            and 0 <= j_ref < ctx["ny"]
            and ctx["is_fluid"][j_ref, i_ref]
        ):
            fields["p"][j_ref, i_ref] = float(
                pressure_cfg.get("value", 0.0)
            )
    timing["flow_field_update"] = time.perf_counter() - update_start

    coefficient_start = time.perf_counter()
    coeffs["aPu"] = system.momentum.aPu.copy()
    coeffs["aPv"] = system.momentum.aPv.copy()
    timing["flow_coeff_update"] = time.perf_counter() - coefficient_start

    flux_start = time.perf_counter()
    dpdx, dpdy = compute_pressure_gradients(ctx, fields["p"])
    gradients = {"dpdx": dpdx, "dpdy": dpdy}
    fluxes = compute_face_fluxes(ctx, settings, fields, coeffs, gradients)
    timing["flow_post_flux"] = time.perf_counter() - flux_start

    if not (
        np.all(np.isfinite(fields["u"]))
        and np.all(np.isfinite(fields["v"]))
        and np.all(np.isfinite(fields["p"]))
        and np.all(np.isfinite(fluxes["me"]))
        and np.all(np.isfinite(fluxes["mn"]))
    ):
        raise RuntimeError("Non-finite flow fields or face fluxes were generated.")

    fluxes["dpdx"] = dpdx
    fluxes["dpdy"] = dpdy
    fluxes["gradients"] = gradients
    fluxes["linear_true_relative_residual"] = true_rel
    fluxes["pressure_constraint_mode"] = system.pressure_constraint_mode
    fluxes["scaling"] = {
        "velocity": system.scaling.velocity_scale,
        "pressure": system.scaling.pressure_scale,
        "momentum_equation": system.scaling.momentum_equation_scale,
        "continuity_equation": system.scaling.continuity_equation_scale,
    }

    backend_timing = backend_extra.get("timing", {})
    if backend_timing:
        timing["backend"] = dict(backend_timing)

    timing["flow_total"] = time.perf_counter() - total_start
    if timing_enabled:
        fluxes["timing"] = timing

    return fields, coeffs, fluxes


# ============================================================
# ENERGY EQUATION SECTION
# ============================================================

from scipy.sparse import lil_matrix

from solver_utils import (
    fluid_at,
    heat_bc_type, heat_bc_T, heat_bc_q,
    upwind_aE, upwind_aW, upwind_aN, upwind_aS,
    harmonic_mean,
    east_face_kind, west_face_kind, north_face_kind, south_face_kind,
    compute_cell_gradients,
    sou_deferred_source,
    cell_volume,
    area_e, area_w, area_n, area_s,
    dist_e, dist_w, dist_n, dist_s,
)


# ============================================================
# ENERGY EQUATION UTILITIES
# ------------------------------------------------------------
# Kept separate from the main solver so steady/transient solvers
# can reuse the same energy assembly. Current implementation is
# steady; transient hook is reserved for later.
# ============================================================


def tidx(ctx, i, j):
    return j * ctx["nx"] + i


def east_conductance(ctx, i, j):
    k = ctx["k"]
    kP = k[j, i]
    kind = east_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        kE = k[j, i + 1]
        kf = harmonic_mean(kP, kE)
        return kf * area_e(ctx, i, j) / dist_e(ctx, i, j)
    if kind == "boundary-east" and heat_bc_type(ctx, "east") == "dirichlet":
        return 2.0 * kP * area_e(ctx, i, j) / dist_e(ctx, i, j)
    return 0.0


def west_conductance(ctx, i, j):
    k = ctx["k"]
    kP = k[j, i]
    kind = west_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        kW = k[j, i - 1]
        kf = harmonic_mean(kP, kW)
        return kf * area_w(ctx, i, j) / dist_w(ctx, i, j)
    if kind == "boundary-west" and heat_bc_type(ctx, "west") == "dirichlet":
        return 2.0 * kP * area_w(ctx, i, j) / dist_w(ctx, i, j)
    return 0.0


def north_conductance(ctx, i, j):
    k = ctx["k"]
    kP = k[j, i]
    kind = north_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        kN = k[j + 1, i]
        kf = harmonic_mean(kP, kN)
        return kf * area_n(ctx, i, j) / dist_n(ctx, i, j)
    if kind == "boundary-north" and heat_bc_type(ctx, "north") == "dirichlet":
        return 2.0 * kP * area_n(ctx, i, j) / dist_n(ctx, i, j)
    return 0.0


def south_conductance(ctx, i, j):
    k = ctx["k"]
    kP = k[j, i]
    kind = south_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        kS = k[j - 1, i]
        kf = harmonic_mean(kP, kS)
        return kf * area_s(ctx, i, j) / dist_s(ctx, i, j)
    if kind == "boundary-south" and heat_bc_type(ctx, "south") == "dirichlet":
        return 2.0 * kP * area_s(ctx, i, j) / dist_s(ctx, i, j)
    return 0.0


def solve_energy(ctx, settings, fields, fluxes, transient=None, linear_solver=None):
    """
    Solve the current steady energy equation.

    Parameters
    ----------
    ctx : dict
        Case/grid/material/boundary context.
    settings : dict
        Solver controls, physics flags, schemes, relaxation values.
    fields : dict
        Current solution fields. Must contain T.
    fluxes : dict
        Face mass fluxes: me and mn.
    transient : None or dict
        Reserved for future transient energy source terms.
    linear_solver : PetscDirectSolver
        Persistent PETSc/MUMPS backend used for the energy system.
    """
    if linear_solver is None:
        raise ValueError(
            "A PetscDirectSolver instance is required for the energy solve."
        )

    profiling_settings = settings.get("profiling", {})
    timing_enabled = bool(profiling_settings.get("enabled", False))
    timing = {}
    t_total = time.perf_counter()

    nx = ctx["nx"]
    ny = ctx["ny"]
    N = nx * ny
    is_fluid = ctx["is_fluid"]
    k = ctx["k"]
    qdot = ctx["qdot"]
    T_old = fields["T"]
    me = fluxes["me"]
    mn = fluxes["mn"]

    alpha_T = settings["relaxation"]["T"]
    energy_scheme = settings["schemes"].get("energy", "upwind").lower()
    energy_blend = float(settings["schemes"].get("energy_blend", 0.0))

    t_stage = time.perf_counter()
    A = lil_matrix((N, N))
    b = np.zeros(N)
    if timing_enabled:
        timing["energy_matrix_alloc"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    if energy_scheme == "sou" and energy_blend != 0.0:
        dTdx, dTdy = compute_cell_gradients(ctx, T_old, is_fluid)
    else:
        dTdx = dTdy = None
    if timing_enabled:
        timing["energy_gradient"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    for j in range(ny):
        for i in range(nx):
            P = tidx(ctx, i, j)
            kP = k[j, i]

            aE = aW = aN = aS = 0.0
            Su = 0.0
            Sp = 0.0

            kind_e = east_face_kind(ctx, i, j)
            kind_w = west_face_kind(ctx, i, j)
            kind_n = north_face_kind(ctx, i, j)
            kind_s = south_face_kind(ctx, i, j)

            if kind_e in ("fluid-fluid", "fluid-solid"):
                kf = harmonic_mean(kP, k[j, i + 1])
                De = kf * area_e(ctx, i, j) / dist_e(ctx, i, j)
                if is_fluid[j, i] and fluid_at(ctx, i + 1, j):
                    Fe = me[j, i]
                    aE = upwind_aE(Fe, De)
                else:
                    aE = De
            elif kind_e == "boundary-east":
                bct = heat_bc_type(ctx, "east")
                if bct == "dirichlet":
                    coeff = 2.0 * kP * area_e(ctx, i, j) / dist_e(ctx, i, j)
                    Sp -= coeff
                    Su += coeff * heat_bc_T(ctx, "east")
                elif bct == "neumann":
                    Su += heat_bc_q(ctx, "east") * area_e(ctx, i, j)
                elif bct in ("adiabatic", "symmetry", "outlet", "open"):
                    pass
                else:
                    raise ValueError(f"Unsupported east heat BC type: {bct}")

            if kind_w in ("fluid-fluid", "fluid-solid"):
                kf = harmonic_mean(kP, k[j, i - 1])
                Dw = kf * area_w(ctx, i, j) / dist_w(ctx, i, j)
                if is_fluid[j, i] and fluid_at(ctx, i - 1, j):
                    Fw = me[j, i - 1]
                    aW = upwind_aW(Fw, Dw)
                else:
                    aW = Dw
            elif kind_w == "boundary-west":
                bct = heat_bc_type(ctx, "west")
                if bct == "dirichlet":
                    coeff = 2.0 * kP * area_w(ctx, i, j) / dist_w(ctx, i, j)
                    Sp -= coeff
                    Su += coeff * heat_bc_T(ctx, "west")
                elif bct == "neumann":
                    Su += heat_bc_q(ctx, "west") * area_w(ctx, i, j)
                elif bct in ("adiabatic", "symmetry", "outlet", "open"):
                    pass
                else:
                    raise ValueError(f"Unsupported west heat BC type: {bct}")

            if kind_n in ("fluid-fluid", "fluid-solid"):
                kf = harmonic_mean(kP, k[j + 1, i])
                Dn = kf * area_n(ctx, i, j) / dist_n(ctx, i, j)
                if is_fluid[j, i] and fluid_at(ctx, i, j + 1):
                    Fn = mn[j, i]
                    aN = upwind_aN(Fn, Dn)
                else:
                    aN = Dn
            elif kind_n == "boundary-north":
                bct = heat_bc_type(ctx, "north")
                if bct == "dirichlet":
                    coeff = 2.0 * kP * area_n(ctx, i, j) / dist_n(ctx, i, j)
                    Sp -= coeff
                    Su += coeff * heat_bc_T(ctx, "north")
                elif bct == "neumann":
                    Su += heat_bc_q(ctx, "north") * area_n(ctx, i, j)
                elif bct in ("adiabatic", "symmetry", "outlet", "open"):
                    pass
                else:
                    raise ValueError(f"Unsupported north heat BC type: {bct}")

            if kind_s in ("fluid-fluid", "fluid-solid"):
                kf = harmonic_mean(kP, k[j - 1, i])
                Ds = kf * area_s(ctx, i, j) / dist_s(ctx, i, j)
                if is_fluid[j, i] and fluid_at(ctx, i, j - 1):
                    Fs = mn[j - 1, i]
                    aS = upwind_aS(Fs, Ds)
                else:
                    aS = Ds
            elif kind_s == "boundary-south":
                bct = heat_bc_type(ctx, "south")
                if bct == "dirichlet":
                    coeff = 2.0 * kP * area_s(ctx, i, j) / dist_s(ctx, i, j)
                    Sp -= coeff
                    Su += coeff * heat_bc_T(ctx, "south")
                elif bct == "neumann":
                    Su += heat_bc_q(ctx, "south") * area_s(ctx, i, j)
                elif bct in ("adiabatic", "symmetry", "outlet", "open"):
                    pass
                else:
                    raise ValueError(f"Unsupported south heat BC type: {bct}")

            if energy_scheme == "sou" and energy_blend != 0.0 and is_fluid[j, i]:
                Fe_corr = me[j, i] if kind_e == "fluid-fluid" else 0.0
                Fw_corr = me[j, i - 1] if kind_w == "fluid-fluid" else 0.0
                Fn_corr = mn[j, i] if kind_n == "fluid-fluid" else 0.0
                Fs_corr = mn[j - 1, i] if kind_s == "fluid-fluid" else 0.0

                Su += sou_deferred_source(
                    ctx, T_old, dTdx, dTdy, i, j, Fe_corr, Fw_corr, Fn_corr, Fs_corr,
                    kind_e, kind_w, kind_n, kind_s, energy_blend
                )

            # Steady volumetric heat source [W/m^3] * cell volume.
            Su += qdot[j, i] * cell_volume(ctx, i, j)

            # Reserved transient hook. Not active unless transient is supplied later.
            if transient is not None:
                raise NotImplementedError("Transient energy term is reserved for the next solver phase.")

            aP = max(aE + aW + aN + aS - Sp, 1e-30)
            A[P, P] = aP

            if i < nx - 1 and kind_e in ("fluid-fluid", "fluid-solid"):
                A[P, tidx(ctx, i + 1, j)] = -aE
            if i > 0 and kind_w in ("fluid-fluid", "fluid-solid"):
                A[P, tidx(ctx, i - 1, j)] = -aW
            if j < ny - 1 and kind_n in ("fluid-fluid", "fluid-solid"):
                A[P, tidx(ctx, i, j + 1)] = -aN
            if j > 0 and kind_s in ("fluid-fluid", "fluid-solid"):
                A[P, tidx(ctx, i, j - 1)] = -aS

            b[P] = Su

    if timing_enabled:
        timing["energy_assembly_loop"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    T_vec = linear_solver.solve(
        A,
        b,
        system_type="energy",
        x0=T_old.reshape(-1),
    )
    if timing_enabled:
        timing["energy_linear_solve"] = time.perf_counter() - t_stage
        backend_info = getattr(linear_solver, "last_info", None)
        backend_extra = getattr(backend_info, "extra", {}) if backend_info is not None else {}
        backend_timing = backend_extra.get("timing", {}) if isinstance(backend_extra, dict) else {}
        if backend_timing:
            timing["backend"] = dict(backend_timing)

    if not np.all(np.isfinite(T_vec)):
        raise RuntimeError("Energy linear solve produced non-finite values.")

    t_stage = time.perf_counter()
    T_new = T_vec.reshape((ny, nx))
    T_relaxed = alpha_T * T_new + (1.0 - alpha_T) * T_old
    if timing_enabled:
        timing["energy_field_update"] = time.perf_counter() - t_stage
        timing["energy_total"] = time.perf_counter() - t_total
        timing["energy_assembly_total"] = (
            timing.get("energy_matrix_alloc", 0.0)
            + timing.get("energy_gradient", 0.0)
            + timing.get("energy_assembly_loop", 0.0)
        )
        settings["_last_energy_timing"] = timing
    return T_relaxed