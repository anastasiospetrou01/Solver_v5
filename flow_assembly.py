from __future__ import annotations

"""Distributed fixed-COO coupled flow assembly for Solver V5 (Phase D/F)."""

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from flow_scaling import FlowScaling, build_flow_scaling
from numerical_kernels import (
    fill_flow_local_coo_distributed_kernel,
    momentum_pass_kernel,
)
from solver_utils import (
    compute_cell_gradients,
    compute_face_fluxes,
    compute_pressure_gradients,
    initialize_solver_workspace,
)
from geometry import EAST, WEST, NORTH, SOUTH, FLOW_BC_OPEN


@dataclass
class LocalCOOPattern:
    local_indptr: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    row_start: int
    row_end: int
    pattern_key: Tuple[Any, ...]

    @property
    def nnz(self) -> int:
        return int(self.cols.size)


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


def _resolve_pressure_constraint(ctx: Dict[str, Any], settings: Dict[str, Any]) -> str:
    cfg = settings.get("pressure_reference", {})
    mode = str(cfg.get("mode", "auto")).lower().strip()
    if mode not in ("auto", "pin", "nullspace", "none"):
        raise ValueError(f"Unsupported pressure constraint mode: {mode!r}")
    has_pressure_boundary = bool(
        np.any(ctx["topology"]["flow_bc_code"] == FLOW_BC_OPEN)
    )
    if mode == "auto":
        return "none" if has_pressure_boundary else "pin"
    if mode == "nullspace":
        # Direct LU requires a nonsingular matrix; retain validated pin behavior.
        return "pin"
    return mode


def _pressure_reference_row(ctx, settings, mode: str) -> Optional[int]:
    del settings
    if mode != "pin":
        return None
    fid = int(ctx.get("pressure_reference_fid", -1))
    if fid < 0:
        raise RuntimeError("A valid pressure-reference fluid cell is required.")
    return 3 * fid + 2


def _ensure_momentum_workspace(ctx):
    workspace = initialize_solver_workspace(ctx)
    shape = ctx["is_fluid"].shape
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
    timing: Dict[str, float] = {}
    topology = ctx["topology"]
    workspace = _ensure_momentum_workspace(ctx)

    stage = time.perf_counter()
    dpdx, dpdy = compute_pressure_gradients(ctx, fields["p"])
    timing["momentum_pressure_gradient"] = time.perf_counter() - stage
    gradients = {"dpdx": dpdx, "dpdy": dpdy}

    stage = time.perf_counter()
    fluxes = compute_face_fluxes(ctx, settings, fields, lagged_coeffs, gradients)
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

    # Continuity rows at partition interfaces use neighbour aP values.  Only
    # these two new coefficient fields require halo exchange before row fill.
    stage = time.perf_counter()
    ctx["domain"].exchange_many((workspace["mom_aPu"], workspace["mom_aPv"]))
    timing["momentum_coefficient_halo"] = time.perf_counter() - stage

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


def build_flow_pattern(ctx) -> LocalCOOPattern:
    existing = ctx.get("flow_pattern")
    if existing is not None:
        return existing

    topology = ctx["topology"]
    neighbor = topology["neighbor_fid"]
    local_nf = int(topology["local_Nf"])
    fid_start = int(topology["fid_start"])
    row_start = 3 * fid_start
    row_end = row_start + 3 * local_nf

    indptr = [0]
    indices = []
    for lfid in range(local_nf):
        fid = fid_start + lfid
        nb_e = int(neighbor[lfid, EAST])
        nb_w = int(neighbor[lfid, WEST])
        nb_n = int(neighbor[lfid, NORTH])
        nb_s = int(neighbor[lfid, SOUTH])
        ru = 3 * fid
        rv = ru + 1
        rp = ru + 2

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

    local_indptr = np.asarray(indptr, dtype=np.int64)
    cols = np.asarray(indices, dtype=np.int64)
    counts = np.diff(local_indptr)
    rows = np.repeat(np.arange(row_start, row_end, dtype=np.int64), counts)

    pattern = LocalCOOPattern(
        local_indptr=local_indptr,
        rows=rows,
        cols=cols,
        row_start=row_start,
        row_end=row_end,
        pattern_key=(
            "v5_phase_df_flow_local_coo",
            int(ctx["nx"]),
            int(ctx["ny"]),
            int(ctx["Nf"]),
            int(ctx["domain"].size),
            row_start,
            row_end,
            int(cols.size),
        ),
    )
    ctx["flow_pattern"] = pattern
    return pattern


@dataclass
class FlowCOOLinearSystem:
    ctx: Dict[str, Any]
    settings: Dict[str, Any]
    fields: Dict[str, np.ndarray]
    old_fields: Dict[str, np.ndarray]
    momentum: MomentumPass
    scaling: FlowScaling
    pressure_constraint_mode: str
    pressure_reference_row: Optional[int]
    pattern: LocalCOOPattern
    assembly_timing: Dict[str, float]

    is_fixed_coo: bool = True
    is_distributed_local: bool = True
    returns_local_solution: bool = True
    block_size: int = 3

    @property
    def global_size(self) -> int:
        return 3 * int(self.ctx["Nf"])

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
            "returns_local_solution": True,
            "pattern_key": self.pattern_key,
        }

    def local_coo_pattern(self, *_args):
        return self.pattern.rows, self.pattern.cols

    def assemble_petsc(self, mat, rhs_vec, row_start: int, row_end: int):
        from petsc4py import PETSc

        if int(row_start) != self.row_start or int(row_end) != self.row_end:
            raise RuntimeError(
                "PETSc flow ownership does not match the geometric y-slab ownership: "
                f"PETSc={row_start}:{row_end}, expected={self.row_start}:{self.row_end}."
            )

        workspace = initialize_solver_workspace(self.ctx)
        key = ("flow_local_coo_df", self.row_start, self.row_end, self.pattern.nnz)
        local = workspace.get(key)
        if local is None:
            local = {
                "data": np.empty(self.pattern.nnz, dtype=float),
                "rhs": np.empty(self.local_size, dtype=float),
            }
            workspace[key] = local

        stage = time.perf_counter()
        fill_flow_local_coo_distributed_kernel(
            int(self.ctx["topology"]["fid_start"]),
            self.pattern.local_indptr,
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
            self.old_fields["u"],
            self.old_fields["v"],
            self.old_fields["p"],
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
            float(self.scaling.momentum_equation_scale),
            float(self.scaling.continuity_equation_scale),
            float(self.scaling.velocity_scale),
            float(self.scaling.pressure_scale),
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
        value_fill = time.perf_counter() - stage

        stage = time.perf_counter()
        mat.setValuesCOO(
            np.asarray(local["data"], dtype=PETSc.ScalarType),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        matrix_update = time.perf_counter() - stage

        stage = time.perf_counter()
        rhs_array = rhs_vec.getArray()
        rhs_array[:] = np.asarray(local["rhs"], dtype=rhs_array.dtype)
        rhs_vec.assemble()
        rhs_update = time.perf_counter() - stage

        return {
            "local_cells": float(self.ctx["topology"]["local_Nf"]),
            "coo_value_fill": value_fill,
            "coo_matrix_update": matrix_update,
            "coo_rhs_update": rhs_update,
        }

    def recover_correction(self, scaled_solution: np.ndarray) -> np.ndarray:
        workspace = initialize_solver_workspace(self.ctx)
        out = workspace.get("flow_local_correction")
        if out is None or out.size != scaled_solution.size:
            out = np.empty_like(np.asarray(scaled_solution, dtype=float))
            workspace["flow_local_correction"] = out
        return self.scaling.unscale_local_solution(
            scaled_solution, self.row_start, out=out
        )

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

        local_left = self.scaling.local_left(self.row_start, self.row_end)
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

        unscaled_abs = float(physical_residual.norm(PETSc.NormType.NORM_INFINITY))
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
    old_fields: Dict[str, np.ndarray],
    distributed: bool = True,
):
    del distributed
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
    scaling_cfg = (
        settings.get("linear_solver", {})
        .get("flow_coupled", {})
        .get("scaling", {})
    )
    scaling = build_flow_scaling(
        ctx, fields, 3 * int(ctx["Nf"]), scaling_cfg
    )
    timing["flow_scaling"] = time.perf_counter() - stage
    timing["flow_assembly_total"] = time.perf_counter() - total

    return FlowCOOLinearSystem(
        ctx=ctx,
        settings=settings,
        fields=fields,
        old_fields=old_fields,
        momentum=momentum,
        scaling=scaling,
        pressure_constraint_mode=constraint_mode,
        pressure_reference_row=reference_row,
        pattern=pattern,
        assembly_timing=timing,
    )