from __future__ import annotations

"""Optimized fixed-pattern coupled flow assembly for Solver V5.

Phase A/B/C implementation:
- static fluid topology is precomputed once in geometry.py;
- momentum and row-value kernels are Numba compiled;
- no per-row Python dictionaries are created;
- the sparse pattern is fixed once and updated through PETSc COO values;
- serial and MPI runs use the same finite-volume equations.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import sparse

from flow_scaling import FlowScaling, build_flow_scaling
from numerical_kernels import (
    build_state_vector_kernel,
    fill_flow_local_coo_kernel,
    momentum_pass_kernel,
)
from solver_utils import (
    compute_cell_gradients,
    compute_face_fluxes,
    compute_pressure_gradients,
    initialize_solver_workspace,
)
from geometry import (
    EAST,
    WEST,
    NORTH,
    SOUTH,
    FACE_FLUID_FLUID,
    FLOW_BC_OPEN,
)


@dataclass
class FixedCSRPattern:
    indptr: np.ndarray
    indices: np.ndarray
    pattern_key: Tuple[Any, ...]
    _local_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, int, int]] = field(
        default_factory=dict, repr=False
    )

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def local_coo(self, row_start: int, row_end: int):
        key = (int(row_start), int(row_end))
        cached = self._local_cache.get(key)
        if cached is not None:
            return cached

        nz_start = int(self.indptr[row_start])
        nz_end = int(self.indptr[row_end])
        counts = np.diff(self.indptr[row_start : row_end + 1])
        rows = np.repeat(
            np.arange(row_start, row_end, dtype=np.int64), counts
        )
        cols = np.asarray(self.indices[nz_start:nz_end], dtype=np.int64)
        cached = (rows, cols, nz_start, nz_end)
        self._local_cache[key] = cached
        return cached


@dataclass
class MomentumPass:
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
    timing: Dict[str, float]


def _gidx_f(fid: int, variable: int) -> int:
    return 3 * int(fid) + int(variable)


def _resolve_pressure_constraint(ctx: Dict[str, Any], settings: Dict[str, Any]) -> str:
    cfg = settings.get("pressure_reference", {})
    mode = str(cfg.get("mode", "auto")).lower().strip()
    if mode not in ("auto", "pin", "nullspace", "none"):
        raise ValueError(f"Unsupported pressure constraint mode: {mode!r}")

    topology = ctx["topology"]
    has_pressure_boundary = bool(
        np.any(topology["flow_bc_code"] == FLOW_BC_OPEN)
    )
    if mode == "auto":
        return "none" if has_pressure_boundary else "pin"
    if mode == "nullspace":
        return "pin"
    return mode


def _pressure_reference_cell(ctx, settings):
    cfg = settings.get("pressure_reference", {})
    i = int(cfg.get("i", ctx.get("p_ref_i", 0)))
    j = int(cfg.get("j", ctx.get("p_ref_j", 0)))
    if (
        0 <= i < int(ctx["nx"])
        and 0 <= j < int(ctx["ny"])
        and bool(ctx["is_fluid"][j, i])
    ):
        return i, j
    if int(ctx["Nf"]) == 0:
        raise RuntimeError("No fluid cells are available for the pressure reference.")
    return int(ctx["topology"]["fluid_i"][0]), int(ctx["topology"]["fluid_j"][0])


def _pressure_reference_row(ctx, settings, mode: str) -> Optional[int]:
    if mode != "pin":
        return None
    i, j = _pressure_reference_cell(ctx, settings)
    fid = int(ctx["cell_to_fid"][j, i])
    return _gidx_f(fid, 2)


def _ensure_momentum_workspace(ctx):
    workspace = initialize_solver_workspace(ctx)
    shape = (int(ctx["ny"]), int(ctx["nx"]))
    for name, fill in (
        ("mom_aE", 0.0),
        ("mom_aW", 0.0),
        ("mom_aN", 0.0),
        ("mom_aS", 0.0),
        ("mom_aPu", 1.0),
        ("mom_aPv", 1.0),
        ("mom_source_u", 0.0),
        ("mom_source_v", 0.0),
        ("mom_Fe", 0.0),
        ("mom_Fw", 0.0),
        ("mom_Fn", 0.0),
        ("mom_Fs", 0.0),
    ):
        array = workspace.get(name)
        if array is None or array.shape != shape:
            workspace[name] = np.full(shape, fill, dtype=float)
    return workspace


def build_momentum_pass(ctx, settings, fields, lagged_coeffs) -> MomentumPass:
    """Compute the complete nonlinear momentum coefficient pass in JIT kernels."""
    timing: Dict[str, float] = {}
    topology = ctx["topology"]
    workspace = _ensure_momentum_workspace(ctx)

    stage = time.perf_counter()
    dpdx, dpdy = compute_pressure_gradients(ctx, fields["p"])
    timing["momentum_pressure_gradient"] = time.perf_counter() - stage
    gradients = {"dpdx": dpdx, "dpdy": dpdy}

    stage = time.perf_counter()
    fluxes = compute_face_fluxes(
        ctx, settings, fields, lagged_coeffs, gradients
    )
    timing["momentum_pre_flux"] = time.perf_counter() - stage

    sou_enabled = str(settings["schemes"].get("momentum", "upwind")).lower() == "sou"
    stage = time.perf_counter()
    if sou_enabled:
        dudx, dudy = compute_cell_gradients(
            ctx, fields["u"], ctx["is_fluid"], workspace_slot=1
        )
        dvdx, dvdy = compute_cell_gradients(
            ctx, fields["v"], ctx["is_fluid"], workspace_slot=2
        )
    else:
        # Kernels still require typed arrays; these are never read when SOU is off.
        dudx = workspace["grad_x_1"]
        dudy = workspace["grad_y_1"]
        dvdx = workspace["grad_x_2"]
        dvdy = workspace["grad_y_2"]
    timing["momentum_sou_gradients"] = time.perf_counter() - stage

    stage = time.perf_counter()
    momentum_pass_kernel(
        topology["fluid_i"],
        topology["fluid_j"],
        topology["face_kind"],
        topology["flow_bc_code"],
        topology["flow_bc_u"],
        topology["flow_bc_v"],
        topology["flow_bc_p"],
        ctx["is_fluid"],
        ctx["rho"],
        ctx["mu"],
        fields["u"],
        fields["v"],
        fields["p"],
        lagged_coeffs["aPu"],
        lagged_coeffs["aPv"],
        dpdx,
        dpdy,
        fluxes["me"],
        fluxes["mn"],
        dudx,
        dudy,
        dvdx,
        dvdy,
        float(ctx["dx"]),
        float(ctx["dy"]),
        float(ctx["V"]),
        sou_enabled,
        float(settings["schemes"].get("momentum_blend", 0.0)),
        bool(ctx.get("enable_sou_limiter", True)),
        workspace["mom_aE"],
        workspace["mom_aW"],
        workspace["mom_aN"],
        workspace["mom_aS"],
        workspace["mom_aPu"],
        workspace["mom_aPv"],
        workspace["mom_source_u"],
        workspace["mom_source_v"],
        workspace["mom_Fe"],
        workspace["mom_Fw"],
        workspace["mom_Fn"],
        workspace["mom_Fs"],
    )
    timing["momentum_coefficients"] = time.perf_counter() - stage

    return MomentumPass(
        aE=workspace["mom_aE"],
        aW=workspace["mom_aW"],
        aN=workspace["mom_aN"],
        aS=workspace["mom_aS"],
        aPu=workspace["mom_aPu"],
        aPv=workspace["mom_aPv"],
        source_u=workspace["mom_source_u"],
        source_v=workspace["mom_source_v"],
        Fe=workspace["mom_Fe"],
        Fw=workspace["mom_Fw"],
        Fn=workspace["mom_Fn"],
        Fs=workspace["mom_Fs"],
        gradients=gradients,
        fluxes=fluxes,
        timing=timing,
    )


def build_flow_pattern(ctx) -> FixedCSRPattern:
    """Create the exact fixed coupled sparsity pattern once per case."""
    existing = ctx.get("flow_pattern")
    if existing is not None:
        return existing

    nf = int(ctx["Nf"])
    neighbor = ctx["topology"]["neighbor_fid"]
    indptr = [0]
    indices = []

    for fid in range(nf):
        nb_e = int(neighbor[fid, EAST])
        nb_w = int(neighbor[fid, WEST])
        nb_n = int(neighbor[fid, NORTH])
        nb_s = int(neighbor[fid, SOUTH])
        ru = 3 * fid
        rv = ru + 1
        rp = ru + 2

        # u row: self u, neighbour velocities, self p, x-neighbour pressures.
        cols = [ru]
        if nb_e >= 0:
            cols.append(3 * nb_e)
        if nb_w >= 0:
            cols.append(3 * nb_w)
        if nb_n >= 0:
            cols.append(3 * nb_n)
        if nb_s >= 0:
            cols.append(3 * nb_s)
        cols.append(rp)
        if nb_e >= 0:
            cols.append(3 * nb_e + 2)
        if nb_w >= 0:
            cols.append(3 * nb_w + 2)
        indices.extend(cols)
        indptr.append(len(indices))

        # v row: self v, neighbour velocities, self p, y-neighbour pressures.
        cols = [rv]
        if nb_e >= 0:
            cols.append(3 * nb_e + 1)
        if nb_w >= 0:
            cols.append(3 * nb_w + 1)
        if nb_n >= 0:
            cols.append(3 * nb_n + 1)
        if nb_s >= 0:
            cols.append(3 * nb_s + 1)
        cols.append(rp)
        if nb_n >= 0:
            cols.append(3 * nb_n + 2)
        if nb_s >= 0:
            cols.append(3 * nb_s + 2)
        indices.extend(cols)
        indptr.append(len(indices))

        # continuity row: self p/u/v followed by E/W/N/S neighbour pairs.
        cols = [rp, ru, rv]
        if nb_e >= 0:
            cols.extend((3 * nb_e, 3 * nb_e + 2))
        if nb_w >= 0:
            cols.extend((3 * nb_w, 3 * nb_w + 2))
        if nb_n >= 0:
            cols.extend((3 * nb_n + 1, 3 * nb_n + 2))
        if nb_s >= 0:
            cols.extend((3 * nb_s + 1, 3 * nb_s + 2))
        indices.extend(cols)
        indptr.append(len(indices))

    pattern = FixedCSRPattern(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        pattern_key=(
            "v5_phase_abc_flow_coo",
            int(ctx["nx"]),
            int(ctx["ny"]),
            nf,
            int(len(indices)),
        ),
    )
    ctx["flow_pattern"] = pattern
    return pattern


def build_state_vector(ctx, fields) -> np.ndarray:
    workspace = initialize_solver_workspace(ctx)
    ndof = 3 * int(ctx["Nf"])
    state = workspace.get("flow_state")
    if state is None or state.size != ndof:
        state = np.empty(ndof, dtype=float)
        workspace["flow_state"] = state
    build_state_vector_kernel(
        ctx["topology"]["fluid_i"],
        ctx["topology"]["fluid_j"],
        fields["u"],
        fields["v"],
        fields["p"],
        state,
    )
    return state


@dataclass
class FlowCOOLinearSystem:
    ctx: Dict[str, Any]
    settings: Dict[str, Any]
    fields: Dict[str, np.ndarray]
    momentum: MomentumPass
    old_state: np.ndarray
    scaling: FlowScaling
    pressure_constraint_mode: str
    pressure_reference_row: Optional[int]
    pattern: FixedCSRPattern
    assembly_timing: Dict[str, float]

    is_fixed_coo: bool = True
    is_distributed_local: bool = True
    block_size: int = 3

    @property
    def global_size(self) -> int:
        return int(self.old_state.size)

    @property
    def pattern_key(self):
        return self.pattern.pattern_key + (
            int(self.pressure_reference_row if self.pressure_reference_row is not None else -1),
        )

    @property
    def metadata(self):
        return {
            "Nf": int(self.ctx["Nf"]),
            "block_size": 3,
            "scaling": self.scaling,
            "pressure_constraint_mode": self.pressure_constraint_mode,
            "distributed_local_assembly": True,
            "fixed_coo": True,
            "pattern_key": self.pattern_key,
        }

    def local_coo_pattern(self, row_start: int, row_end: int):
        rows, cols, _nz0, _nz1 = self.pattern.local_coo(row_start, row_end)
        return rows, cols

    def assemble_petsc(self, mat, rhs_vec, row_start: int, row_end: int):
        from petsc4py import PETSc

        if row_start % 3 or row_end % 3:
            raise RuntimeError(
                "PETSc ownership must be aligned to complete [u,v,p] cell blocks."
            )
        fid_start = row_start // 3
        fid_end = row_end // 3
        _rows, _cols, nz_start, nz_end = self.pattern.local_coo(
            row_start, row_end
        )

        workspace = initialize_solver_workspace(self.ctx)
        cache_key = ("flow_local_coo", row_start, row_end, nz_end - nz_start)
        local = workspace.get(cache_key)
        if local is None:
            local = {
                "data": np.empty(nz_end - nz_start, dtype=float),
                "rhs": np.empty(row_end - row_start, dtype=float),
            }
            workspace[cache_key] = local

        stage = time.perf_counter()
        fill_flow_local_coo_kernel(
            fid_start,
            fid_end,
            nz_start,
            self.pattern.indptr,
            self.ctx["topology"]["fluid_i"],
            self.ctx["topology"]["fluid_j"],
            self.ctx["topology"]["neighbor_fid"],
            self.ctx["topology"]["face_kind"],
            self.ctx["topology"]["flow_bc_code"],
            self.ctx["topology"]["flow_bc_u"],
            self.ctx["topology"]["flow_bc_v"],
            self.ctx["topology"]["flow_bc_p"],
            self.ctx["rho"],
            self.ctx["beta"],
            self.ctx["sx"],
            self.ctx["sy"],
            self.fields["T"],
            self.momentum.aE,
            self.momentum.aW,
            self.momentum.aN,
            self.momentum.aS,
            self.momentum.aPu,
            self.momentum.aPv,
            self.momentum.source_u,
            self.momentum.source_v,
            self.momentum.gradients["dpdx"],
            self.momentum.gradients["dpdy"],
            self.old_state,
            self.scaling.left,
            self.scaling.right,
            float(self.ctx["dx"]),
            float(self.ctx["dy"]),
            float(self.ctx["V"]),
            float(self.ctx["T_ref"]),
            float(self.ctx["gy"]),
            bool(self.settings["physics"].get("buoyancy", False)),
            int(self.pressure_reference_row if self.pressure_reference_row is not None else -1),
            float(self.settings.get("pressure_reference", {}).get("value", 0.0)),
            local["data"],
            local["rhs"],
        )
        value_fill_time = time.perf_counter() - stage

        stage = time.perf_counter()
        mat.setValuesCOO(
            np.asarray(local["data"], dtype=PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        coo_update_time = time.perf_counter() - stage

        stage = time.perf_counter()
        rhs_array = rhs_vec.getArray()
        rhs_array[:] = np.asarray(local["rhs"], dtype=rhs_array.dtype)
        rhs_vec.assemble()
        rhs_time = time.perf_counter() - stage

        return {
            "local_fid_start": float(fid_start),
            "local_fid_end": float(fid_end),
            "local_cells": float(fid_end - fid_start),
            "coo_value_fill": value_fill_time,
            "coo_matrix_update": coo_update_time,
            "coo_rhs_update": rhs_time,
        }

    def recover_correction(self, scaled_solution: np.ndarray) -> np.ndarray:
        return self.scaling.unscale_solution(scaled_solution)

    def distributed_residual_metrics(self, mat, rhs_vec, solution_vec):
        from petsc4py import PETSc

        residual = rhs_vec.duplicate()
        mat.mult(solution_vec, residual)
        residual.aypx(-1.0, rhs_vec)

        scaled_abs = float(residual.norm(PETSc.NormType.NORM_INFINITY))
        scaled_rhs = max(
            float(rhs_vec.norm(PETSc.NormType.NORM_INFINITY)), 1.0e-30
        )
        scaled_rel = scaled_abs / scaled_rhs

        row_start, row_end = mat.getOwnershipRange()
        local_left = np.asarray(
            self.scaling.left[row_start:row_end], dtype=float
        )

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


def assemble_flow_correction_system(
    ctx: Dict[str, Any],
    settings: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    lagged_coeffs: Dict[str, np.ndarray],
    *,
    distributed: bool = True,
):
    """Build one fixed-pattern PETSc COO correction-system descriptor."""
    del distributed  # one canonical path is used for both serial and MPI.
    total = time.perf_counter()
    timing: Dict[str, float] = {}

    stage = time.perf_counter()
    momentum = build_momentum_pass(ctx, settings, fields, lagged_coeffs)
    timing.update(momentum.timing)
    timing["flow_momentum_pass"] = time.perf_counter() - stage

    stage = time.perf_counter()
    constraint_mode = _resolve_pressure_constraint(ctx, settings)
    reference_row = _pressure_reference_row(ctx, settings, constraint_mode)
    pattern = build_flow_pattern(ctx)
    timing["flow_pattern_lookup"] = time.perf_counter() - stage

    stage = time.perf_counter()
    old_state_workspace = build_state_vector(ctx, fields)
    # The system survives field updates until the linear solve completes.  The
    # reusable workspace may be overwritten on the next iteration, not before.
    old_state = old_state_workspace
    timing["flow_state_vector"] = time.perf_counter() - stage

    stage = time.perf_counter()
    scaling_cfg = (
        settings.get("linear_solver", {})
        .get("flow_coupled", {})
        .get("scaling", {})
    )
    scaling = build_flow_scaling(ctx, fields, old_state.size, scaling_cfg)
    timing["flow_scaling"] = time.perf_counter() - stage
    timing["flow_assembly_total"] = time.perf_counter() - total

    return FlowCOOLinearSystem(
        ctx=ctx,
        settings=settings,
        fields=fields,
        momentum=momentum,
        old_state=old_state,
        scaling=scaling,
        pressure_constraint_mode=constraint_mode,
        pressure_reference_row=reference_row,
        pattern=pattern,
        assembly_timing=timing,
    )


def reconstruct_serial_absolute_system(
    ctx: Dict[str, Any],
    settings: Dict[str, Any],
    fields: Dict[str, np.ndarray],
    momentum: MomentumPass,
    pressure_constraint_mode: str,
):
    """Regression helper: reconstruct unscaled absolute A and b from fixed COO."""
    pattern = build_flow_pattern(ctx)
    ndof = 3 * int(ctx["Nf"])
    zeros = np.zeros(ndof, dtype=float)
    ones = np.ones(ndof, dtype=float)
    data = np.empty(pattern.nnz, dtype=float)
    rhs = np.empty(ndof, dtype=float)
    reference_row = _pressure_reference_row(
        ctx, settings, pressure_constraint_mode
    )

    fill_flow_local_coo_kernel(
        0,
        int(ctx["Nf"]),
        0,
        pattern.indptr,
        ctx["topology"]["fluid_i"],
        ctx["topology"]["fluid_j"],
        ctx["topology"]["neighbor_fid"],
        ctx["topology"]["face_kind"],
        ctx["topology"]["flow_bc_code"],
        ctx["topology"]["flow_bc_u"],
        ctx["topology"]["flow_bc_v"],
        ctx["topology"]["flow_bc_p"],
        ctx["rho"],
        ctx["beta"],
        ctx["sx"],
        ctx["sy"],
        fields["T"],
        momentum.aE,
        momentum.aW,
        momentum.aN,
        momentum.aS,
        momentum.aPu,
        momentum.aPv,
        momentum.source_u,
        momentum.source_v,
        momentum.gradients["dpdx"],
        momentum.gradients["dpdy"],
        zeros,
        ones,
        ones,
        float(ctx["dx"]),
        float(ctx["dy"]),
        float(ctx["V"]),
        float(ctx["T_ref"]),
        float(ctx["gy"]),
        bool(settings["physics"].get("buoyancy", False)),
        int(reference_row if reference_row is not None else -1),
        float(settings.get("pressure_reference", {}).get("value", 0.0)),
        data,
        rhs,
    )
    matrix = sparse.csr_matrix(
        (data.copy(), pattern.indices.copy(), pattern.indptr.copy()),
        shape=(ndof, ndof),
    )
    return matrix, rhs.copy()


# Backward-compatible regression name used by earlier tests.
def reconstruct_serial_from_rows(
    ctx,
    settings,
    fields,
    momentum,
    pressure_constraint_mode,
):
    return reconstruct_serial_absolute_system(
        ctx, settings, fields, momentum, pressure_constraint_mode
    )