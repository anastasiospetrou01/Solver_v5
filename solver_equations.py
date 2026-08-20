from __future__ import annotations

"""Coupled flow and energy equations with distributed Phase D/F data ownership."""

from dataclasses import dataclass
import time
from typing import Any, Dict

import numpy as np

from flow_assembly import (
    LocalCOOPattern,
    assemble_flow_correction_system,
)
from numerical_kernels import (
    fill_energy_local_coo_kernel,
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
        old_fields=old_fields,
        distributed=True,
    )
    timing = dict(system.assembly_timing)

    linear_start = time.perf_counter()
    scaled_local_correction = linear_solver.solve(
        system,
        None,
        system_type="flow_coupled",
        metadata=system.metadata,
    )
    timing["flow_linear_solve"] = time.perf_counter() - linear_start

    if not np.all(np.isfinite(scaled_local_correction)):
        raise RuntimeError("The direct linear solver returned non-finite corrections.")
    correction = system.recover_correction(scaled_local_correction)

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

    stage = time.perf_counter()
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
    # Solids in owned and ghost rows remain exactly zero for flow variables.
    fields["u"][ctx["is_solid"]] = 0.0
    fields["v"][ctx["is_solid"]] = 0.0
    fields["p"][ctx["is_solid"]] = 0.0

    if system.pressure_constraint_mode == "pin":
        i_ref = int(ctx["p_ref_i"])
        j_ref = int(ctx["p_ref_j"])
        if ctx["domain"].owns_global_j(j_ref):
            jl = ctx["domain"].global_to_local_j(j_ref)
            if ctx["is_fluid"][jl, i_ref]:
                fields["p"][jl, i_ref] = float(
                    settings.get("pressure_reference", {}).get("value", 0.0)
                )
    timing["flow_field_update"] = time.perf_counter() - stage

    stage = time.perf_counter()
    ctx["domain"].exchange_many((fields["u"], fields["v"], fields["p"]))
    timing["flow_field_halo"] = time.perf_counter() - stage

    stage = time.perf_counter()
    np.copyto(coeffs["aPu"], system.momentum.aPu)
    np.copyto(coeffs["aPv"], system.momentum.aPv)
    # momentum.aPu/aPv already contain current neighbour halos after assembly.
    timing["flow_coeff_update"] = time.perf_counter() - stage

    stage = time.perf_counter()
    dpdx, dpdy = compute_pressure_gradients(ctx, fields["p"])
    gradients = {"dpdx": dpdx, "dpdy": dpdy}
    fluxes = compute_face_fluxes(ctx, settings, fields, coeffs, gradients)
    timing["flow_post_flux"] = time.perf_counter() - stage

    owned = ctx["domain"].owned_slice
    if not (
        np.all(np.isfinite(fields["u"][owned, :]))
        and np.all(np.isfinite(fields["v"][owned, :]))
        and np.all(np.isfinite(fields["p"][owned, :]))
    ):
        raise RuntimeError("Non-finite distributed flow fields were generated.")

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
# ENERGY — PETSc-OWNED ROWS ONLY (PHASE D)
# ============================================================


def build_energy_pattern(ctx) -> LocalCOOPattern:
    existing = ctx.get("energy_pattern")
    if existing is not None:
        return existing

    topology = ctx["topology"]
    neighbor = topology["energy_neighbor_gid"]
    gids = topology["energy_global_gid"]
    row_start = int(ctx["domain"].energy_row_start)
    row_end = int(ctx["domain"].energy_row_end)

    indptr = [0]
    cols = []
    for lc, gid in enumerate(gids):
        row_cols = [int(gid)]
        for direction in range(4):
            nb = int(neighbor[lc, direction])
            if nb >= 0:
                row_cols.append(nb)
        cols.extend(row_cols)
        indptr.append(len(cols))

    local_indptr = np.asarray(indptr, dtype=np.int64)
    cols_array = np.asarray(cols, dtype=np.int64)
    counts = np.diff(local_indptr)
    rows = np.repeat(np.arange(row_start, row_end, dtype=np.int64), counts)
    pattern = LocalCOOPattern(
        local_indptr=local_indptr,
        rows=rows,
        cols=cols_array,
        row_start=row_start,
        row_end=row_end,
        pattern_key=(
            "v5_phase_df_energy_local_coo",
            int(ctx["nx"]),
            int(ctx["ny"]),
            int(ctx["domain"].size),
            row_start,
            row_end,
            int(cols_array.size),
        ),
    )
    ctx["energy_pattern"] = pattern
    return pattern


@dataclass
class EnergyCOOLinearSystem:
    ctx: Dict[str, Any]
    pattern: LocalCOOPattern
    data: np.ndarray
    rhs_local: np.ndarray

    is_fixed_coo: bool = True
    is_distributed_local: bool = True
    returns_local_solution: bool = True
    block_size: int = 1

    @property
    def global_size(self) -> int:
        return int(self.ctx["nx"]) * int(self.ctx["ny"])

    @property
    def local_size(self) -> int:
        return int(self.pattern.row_end - self.pattern.row_start)

    @property
    def row_start(self) -> int:
        return int(self.pattern.row_start)

    @property
    def row_end(self) -> int:
        return int(self.pattern.row_end)

    @property
    def pattern_key(self):
        return self.pattern.pattern_key

    @property
    def metadata(self):
        return {
            "block_size": 1,
            "fixed_coo": True,
            "distributed_energy_assembly": True,
            "returns_local_solution": True,
            "pattern_key": self.pattern_key,
        }

    def local_coo_pattern(self, *_args):
        return self.pattern.rows, self.pattern.cols

    def assemble_petsc(self, mat, rhs_vec, row_start: int, row_end: int):
        from petsc4py import PETSc

        if int(row_start) != self.row_start or int(row_end) != self.row_end:
            raise RuntimeError(
                "PETSc energy ownership does not match the structured y-slab: "
                f"PETSc={row_start}:{row_end}, expected={self.row_start}:{self.row_end}."
            )
        stage = time.perf_counter()
        mat.setValuesCOO(
            np.asarray(self.data, dtype=PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        matrix_time = time.perf_counter() - stage

        stage = time.perf_counter()
        rhs_array = rhs_vec.getArray()
        rhs_array[:] = np.asarray(self.rhs_local, dtype=rhs_array.dtype)
        rhs_vec.assemble()
        rhs_time = time.perf_counter() - stage
        return {
            "local_rows": float(self.local_size),
            "coo_value_fill": 0.0,
            "coo_matrix_update": matrix_time,
            "coo_rhs_update": rhs_time,
        }


def assemble_energy_fixed_system(ctx, settings, fields, fluxes):
    timing: Dict[str, float] = {}
    pattern = build_energy_pattern(ctx)
    workspace = initialize_solver_workspace(ctx)

    data = workspace.get("energy_local_matrix_data")
    if data is None or data.size != pattern.nnz:
        data = np.empty(pattern.nnz, dtype=float)
        workspace["energy_local_matrix_data"] = data
    rhs = workspace.get("energy_local_rhs")
    local_size = int(ctx["domain"].local_energy_size)
    if rhs is None or rhs.size != local_size:
        rhs = np.empty(local_size, dtype=float)
        workspace["energy_local_rhs"] = rhs

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
    fill_energy_local_coo_kernel(
        int(ctx["nx"]),
        int(ctx["domain"].owned_ny),
        int(ctx["domain"].halo),
        pattern.local_indptr,
        ctx["topology"]["energy_face_kind"],
        ctx["topology"]["energy_neighbor_gid"],
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
        raise NotImplementedError("Transient energy term is reserved for the transient solver.")
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
    T_owned_vec = linear_solver.solve(
        system,
        None,
        system_type="energy",
        metadata=system.metadata,
    )
    timing["energy_linear_solve"] = time.perf_counter() - stage
    if not np.all(np.isfinite(T_owned_vec)):
        raise RuntimeError("Energy linear solve produced non-finite values.")

    stage = time.perf_counter()
    owned = ctx["domain"].owned_slice
    T_owned = fields["T"][owned, :]
    T_new = np.asarray(T_owned_vec, dtype=float).reshape(
        (ctx["domain"].owned_ny, int(ctx["nx"]))
    )
    alpha_T = float(settings["relaxation"]["T"])
    T_owned[:] = alpha_T * T_new + (1.0 - alpha_T) * T_owned
    timing["energy_field_update"] = time.perf_counter() - stage

    stage = time.perf_counter()
    ctx["domain"].exchange_halo(fields["T"])
    timing["energy_field_halo"] = time.perf_counter() - stage

    backend_info = getattr(linear_solver, "last_info", None)
    backend_extra = getattr(backend_info, "extra", {}) if backend_info is not None else {}
    backend_timing = backend_extra.get("timing", {}) if isinstance(backend_extra, dict) else {}
    if backend_timing:
        timing["backend"] = dict(backend_timing)
    local_stats = backend_extra.get("local_assembly_stats", {}) if isinstance(backend_extra, dict) else {}
    if isinstance(local_stats, dict):
        for key in ("coo_value_fill", "coo_matrix_update", "coo_rhs_update"):
            if key in local_stats:
                timing[f"energy_{key}"] = float(local_stats[key])

    timing["energy_total"] = time.perf_counter() - total_start
    if timing_enabled:
        settings["_last_energy_timing"] = timing
    return fields["T"]


# ============================================================
# ENERGY BALANCE CONDUCTANCE HELPERS — final root reporting only
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