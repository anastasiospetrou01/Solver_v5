# ============================================================
# COUPLED FLOW AND ENERGY EQUATIONS — PHASE A/B/C OPTIMIZED
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

import numpy as np

from flow_assembly import (
    FixedCSRPattern,
    assemble_flow_correction_system,
)
from numerical_kernels import (
    fill_energy_csr_kernel,
    update_flow_fields_kernel,
)
from solver_utils import (
    compute_cell_gradients,
    compute_face_fluxes,
    compute_pressure_gradients,
    initialize_solver_workspace,
    snapshot_fields,
)


def solve_pressure_velocity(
    ctx,
    settings,
    fields,
    coeffs,
    transient=None,
    linear_solver=None,
    old_fields=None,
):
    """Solve one nonlinear iteration of the fully coupled direct-LU system."""
    if transient is not None:
        raise NotImplementedError(
            "Transient pressure-velocity terms are reserved for the transient solver."
        )
    if linear_solver is None:
        raise ValueError("A PetscDirectSolver instance is required for the flow solve.")

    timing_enabled = bool(settings.get("profiling", {}).get("enabled", False))
    total_start = time.perf_counter()

    if old_fields is None:
        old_fields = snapshot_fields(ctx, fields)

    system = assemble_flow_correction_system(
        ctx,
        settings,
        fields,
        coeffs,
        distributed=True,
    )
    timing = dict(system.assembly_timing)

    linear_start = time.perf_counter()
    scaled_correction = linear_solver.solve(
        system,
        None,
        system_type="flow_coupled",
        metadata=system.metadata,
    )
    timing["flow_linear_solve"] = time.perf_counter() - linear_start

    if not np.all(np.isfinite(scaled_correction)):
        raise RuntimeError("The direct linear solver returned non-finite corrections.")

    correction = system.recover_correction(scaled_correction)
    backend_info = linear_solver.last_info
    backend_extra = backend_info.extra if backend_info is not None else {}
    true_rel = float(backend_extra.get("unscaled_true_rel_residual", np.inf))
    allowed = float(
        settings.get("linear_solver", {})
        .get("flow_coupled", {})
        .get("true_residual_tolerance", 1.0e-5)
    )
    if true_rel > allowed:
        raise RuntimeError(
            "The coupled correction does not satisfy the original unscaled system: "
            f"true relative infinity residual={true_rel:.6e}, allowed={allowed:.6e}."
        )

    update_start = time.perf_counter()
    topology = ctx["topology"]
    update_flow_fields_kernel(
        topology["fluid_i"],
        topology["fluid_j"],
        old_fields["u"],
        old_fields["v"],
        old_fields["p"],
        correction,
        float(settings["relaxation"]["u"]),
        float(settings["relaxation"]["v"]),
        float(settings["relaxation"]["p"]),
        fields["u"],
        fields["v"],
        fields["p"],
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
            fields["p"][j_ref, i_ref] = float(pressure_cfg.get("value", 0.0))
    timing["flow_field_update"] = time.perf_counter() - update_start

    coefficient_start = time.perf_counter()
    # Keep lagged coefficient arrays persistent and overwrite them in-place.
    np.copyto(coeffs["aPu"], system.momentum.aPu)
    np.copyto(coeffs["aPv"], system.momentum.aPv)
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
    local_stats = backend_extra.get("local_assembly_stats", {})
    if isinstance(local_stats, dict):
        for key in ("coo_value_fill", "coo_matrix_update", "coo_rhs_update"):
            if key in local_stats:
                timing[f"flow_{key}"] = float(local_stats[key])

    timing["flow_total"] = time.perf_counter() - total_start
    if timing_enabled:
        fluxes["timing"] = timing
    return fields, coeffs, fluxes


# ============================================================
# ENERGY EQUATION — FIXED FIVE-POINT COO PATTERN
# ============================================================


def build_energy_pattern(ctx) -> FixedCSRPattern:
    existing = ctx.get("energy_pattern")
    if existing is not None:
        return existing

    nx = int(ctx["nx"])
    ny = int(ctx["ny"])
    neighbor = ctx["topology"]["cell_neighbor"]
    ncell = nx * ny
    indptr = [0]
    indices = []

    for cell in range(ncell):
        cols = [cell]
        for direction in range(4):
            nb = int(neighbor[cell, direction])
            if nb >= 0:
                cols.append(nb)
        indices.extend(cols)
        indptr.append(len(indices))

    pattern = FixedCSRPattern(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        pattern_key=(
            "v5_phase_abc_energy_coo",
            nx,
            ny,
            ncell,
            len(indices),
        ),
    )
    ctx["energy_pattern"] = pattern
    return pattern


@dataclass
class EnergyCOOLinearSystem:
    ctx: Dict[str, Any]
    pattern: FixedCSRPattern
    data: np.ndarray
    rhs_global: np.ndarray

    is_fixed_coo: bool = True
    is_distributed_local: bool = False
    block_size: int = 1

    @property
    def global_size(self) -> int:
        return int(self.rhs_global.size)

    @property
    def pattern_key(self):
        return self.pattern.pattern_key

    @property
    def metadata(self):
        return {
            "block_size": 1,
            "fixed_coo": True,
            "pattern_key": self.pattern_key,
        }

    def local_coo_pattern(self, row_start: int, row_end: int):
        rows, cols, _nz0, _nz1 = self.pattern.local_coo(row_start, row_end)
        return rows, cols

    def assemble_petsc(self, mat, rhs_vec, row_start: int, row_end: int):
        from petsc4py import PETSc

        _rows, _cols, nz_start, nz_end = self.pattern.local_coo(
            row_start, row_end
        )
        stage = time.perf_counter()
        mat.setValuesCOO(
            np.asarray(self.data[nz_start:nz_end], dtype=PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        coo_matrix = time.perf_counter() - stage

        stage = time.perf_counter()
        rhs_array = rhs_vec.getArray()
        rhs_array[:] = np.asarray(
            self.rhs_global[row_start:row_end], dtype=rhs_array.dtype
        )
        rhs_vec.assemble()
        rhs_update = time.perf_counter() - stage
        return {
            "local_rows": float(row_end - row_start),
            "coo_value_fill": 0.0,
            "coo_matrix_update": coo_matrix,
            "coo_rhs_update": rhs_update,
        }


def assemble_energy_fixed_system(ctx, settings, fields, fluxes):
    """Fill the fixed energy matrix numerical values without SciPy LIL/COO rebuilds."""
    timing: Dict[str, float] = {}
    pattern = build_energy_pattern(ctx)
    workspace = initialize_solver_workspace(ctx)

    data = workspace.get("energy_matrix_data")
    if data is None or data.size != pattern.nnz:
        data = np.empty(pattern.nnz, dtype=float)
        workspace["energy_matrix_data"] = data
    rhs = workspace.get("energy_rhs")
    ncell = int(ctx["nx"]) * int(ctx["ny"])
    if rhs is None or rhs.size != ncell:
        rhs = np.empty(ncell, dtype=float)
        workspace["energy_rhs"] = rhs

    energy_scheme = str(settings["schemes"].get("energy", "upwind")).lower()
    sou_enabled = (
        energy_scheme == "sou"
        and float(settings["schemes"].get("energy_blend", 0.0)) != 0.0
    )

    stage = time.perf_counter()
    if sou_enabled:
        dTdx, dTdy = compute_cell_gradients(
            ctx, fields["T"], ctx["is_fluid"], workspace_slot=1
        )
    else:
        dTdx = workspace["grad_x_1"]
        dTdy = workspace["grad_y_1"]
    timing["energy_gradient"] = time.perf_counter() - stage

    stage = time.perf_counter()
    fill_energy_csr_kernel(
        int(ctx["nx"]),
        int(ctx["ny"]),
        pattern.indptr,
        ctx["topology"]["cell_face_kind"],
        ctx["topology"]["cell_neighbor"],
        ctx["topology"]["heat_bc_code"],
        ctx["topology"]["heat_bc_T"],
        ctx["topology"]["heat_bc_q"],
        ctx["is_fluid"],
        ctx["k"],
        ctx["qdot"],
        fields["T"],
        fluxes["me"],
        fluxes["mn"],
        dTdx,
        dTdy,
        float(ctx["dx"]),
        float(ctx["dy"]),
        float(ctx["V"]),
        sou_enabled,
        float(settings["schemes"].get("energy_blend", 0.0)),
        bool(ctx.get("enable_sou_limiter", True)),
        data,
        rhs,
    )
    timing["energy_value_fill"] = time.perf_counter() - stage
    timing["energy_assembly_total"] = (
        timing["energy_gradient"] + timing["energy_value_fill"]
    )
    return EnergyCOOLinearSystem(ctx, pattern, data, rhs), timing


def solve_energy(ctx, settings, fields, fluxes, transient=None, linear_solver=None):
    if transient is not None:
        raise NotImplementedError(
            "Transient energy term is reserved for the transient solver."
        )
    if linear_solver is None:
        raise ValueError("A PetscDirectSolver instance is required for the energy solve.")

    timing_enabled = bool(settings.get("profiling", {}).get("enabled", False))
    timing: Dict[str, Any] = {}
    total_start = time.perf_counter()

    system, assembly_timing = assemble_energy_fixed_system(
        ctx, settings, fields, fluxes
    )
    timing.update(assembly_timing)

    stage = time.perf_counter()
    T_vec = linear_solver.solve(
        system,
        None,
        system_type="energy",
        x0=fields["T"].reshape(-1),
        metadata=system.metadata,
    )
    timing["energy_linear_solve"] = time.perf_counter() - stage

    if not np.all(np.isfinite(T_vec)):
        raise RuntimeError("Energy linear solve produced non-finite values.")

    stage = time.perf_counter()
    T_old = fields["T"]
    T_new = T_vec.reshape((int(ctx["ny"]), int(ctx["nx"])))
    alpha_T = float(settings["relaxation"]["T"])
    workspace = initialize_solver_workspace(ctx)
    relaxed = workspace.get("energy_relaxed")
    if relaxed is None or relaxed.shape != T_old.shape:
        relaxed = np.empty_like(T_old)
        workspace["energy_relaxed"] = relaxed
    np.add(
        alpha_T * T_new,
        (1.0 - alpha_T) * T_old,
        out=relaxed,
    )
    timing["energy_field_update"] = time.perf_counter() - stage

    backend_info = getattr(linear_solver, "last_info", None)
    backend_extra = (
        getattr(backend_info, "extra", {}) if backend_info is not None else {}
    )
    backend_timing = (
        backend_extra.get("timing", {})
        if isinstance(backend_extra, dict)
        else {}
    )
    if backend_timing:
        timing["backend"] = dict(backend_timing)
    local_stats = (
        backend_extra.get("local_assembly_stats", {})
        if isinstance(backend_extra, dict)
        else {}
    )
    if isinstance(local_stats, dict):
        for key in ("coo_value_fill", "coo_matrix_update", "coo_rhs_update"):
            if key in local_stats:
                timing[f"energy_{key}"] = float(local_stats[key])

    timing["energy_total"] = time.perf_counter() - total_start
    if timing_enabled:
        settings["_last_energy_timing"] = timing
    return relaxed


# ============================================================
# ENERGY BALANCE CONDUCTANCE HELPERS — retained for reporting
# ============================================================

from solver_utils import (
    area_e,
    area_n,
    area_s,
    area_w,
    dist_e,
    dist_n,
    dist_s,
    dist_w,
    east_face_kind,
    heat_bc_type,
    harmonic_mean,
    north_face_kind,
    south_face_kind,
    west_face_kind,
)


def east_conductance(ctx, i, j):
    kP = ctx["k"][j, i]
    kind = east_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        return harmonic_mean(kP, ctx["k"][j, i + 1]) * area_e(ctx, i, j) / dist_e(ctx, i, j)
    if kind == "boundary-east" and heat_bc_type(ctx, "east") == "dirichlet":
        return 2.0 * kP * area_e(ctx, i, j) / dist_e(ctx, i, j)
    return 0.0


def west_conductance(ctx, i, j):
    kP = ctx["k"][j, i]
    kind = west_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        return harmonic_mean(kP, ctx["k"][j, i - 1]) * area_w(ctx, i, j) / dist_w(ctx, i, j)
    if kind == "boundary-west" and heat_bc_type(ctx, "west") == "dirichlet":
        return 2.0 * kP * area_w(ctx, i, j) / dist_w(ctx, i, j)
    return 0.0


def north_conductance(ctx, i, j):
    kP = ctx["k"][j, i]
    kind = north_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        return harmonic_mean(kP, ctx["k"][j + 1, i]) * area_n(ctx, i, j) / dist_n(ctx, i, j)
    if kind == "boundary-north" and heat_bc_type(ctx, "north") == "dirichlet":
        return 2.0 * kP * area_n(ctx, i, j) / dist_n(ctx, i, j)
    return 0.0


def south_conductance(ctx, i, j):
    kP = ctx["k"][j, i]
    kind = south_face_kind(ctx, i, j)
    if kind in ("fluid-fluid", "fluid-solid"):
        return harmonic_mean(kP, ctx["k"][j - 1, i]) * area_s(ctx, i, j) / dist_s(ctx, i, j)
    if kind == "boundary-south" and heat_bc_type(ctx, "south") == "dirichlet":
        return 2.0 * kP * area_s(ctx, i, j) / dist_s(ctx, i, j)
    return 0.0