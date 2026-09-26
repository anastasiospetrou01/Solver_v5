from __future__ import annotations

# ============================================================
# RUN SETTINGS — MUMPS REFERENCE + ITERATIVE FLOW RESEARCH
# ============================================================
# Edit this dictionary to select the case, solver controls and MPI ranks.
# Run the solver with:
#     python -m solver.steady_lam_v5
#
# When mpi_ranks > 1, this script relaunches itself with mpiexec before NumPy,
# SciPy or PETSc is imported.
RUN_SETTINGS = {
    "case_name": "buoyancy_cavity_1x1",
    "case_file": None,

    "use_previous_solution": False,
    "restart_file": None,

    # Number of CPU processors/workers to use.
    # 1 .. physical cores  -> one MPI rank per physical core
    # above physical cores -> SMT/logical processors are used automatically
    "processors": 8,

    "max_iter": 200,
    "tol_mass": 1.0e-6,
    "tol_T": 1.0e-6,

    "enable_energy": True,
    "enable_buoyancy": True,

    "T_ref": 25.0,
    "gx": 0.0,
    "gy": -9.81,

    # Nonlinear correction under-relaxation factors.
    "alpha_T": 0.3,
    "alpha_u": 0.4,
    "alpha_v": 0.4,
    "alpha_p": 0.2,

    # auto: open-pressure domains need no interior pin; closed domains are
    # pinned because direct LU requires a nonsingular matrix.
    "pressure_reference_mode": "auto",
    "use_pressure_reference": True,
    "p_ref_value": 0.0,
    "p_ref_i_fraction": 0.125,
    "p_ref_j_fraction": 0.5,

    "enable_sou_momentum": True,
    "enable_sou_energy": True,
    "sou_blend_momentum": 0.7,
    "sou_blend_energy": 0.7,
    "enable_sou_limiter": True,

    # Coupled-flow linear backend. Keep "mumps" as the production/reference
    # baseline; use "iterative" for the Stage-1 FGMRES/Schur research path.
    "linear_solver_type": "mumps",

    "iterative_solver": {
        "ksp_type": "fgmres",
        "rtol": 1.0e-6,
        "atol": 0.0,
        "dtol": 1.0e8,
        "max_it": 200,
        "restart": 50,
        "schur_fact_type": "lower",
        "schur_pre_type": "selfp",
        # Stage 1: deliberately strong block solves to validate the Schur
        # architecture before introducing AMG.
        "velocity_solver": "hypre",
        "velocity_ksp_type": "gmres",
        "velocity_ksp_rtol": 1.0e-2,
        "velocity_ksp_atol": 0.0,
        "velocity_ksp_dtol": 1.0e8,
        "velocity_ksp_max_it": 10,
        "velocity_ksp_restart": 10,
        "velocity_boomeramg_max_iter": 1,
        "velocity_boomeramg_tol": 0.0,

        # Stage 3E: use a single fixed BoomerAMG V-cycle for A00^{-1}
        # inside every matrix-free Schur matvec.
        "schur_inner_velocity_ksp_type": "preonly",
        "schur_inner_velocity_solver": "hypre",
        "schur_inner_velocity_boomeramg_max_iter": 1,
        "schur_inner_velocity_boomeramg_tol": 0.0,

        # Stage 3F: same cheap one-cycle AMG approximation for the separate
        # upper-factor A00 solve used by FULL Schur factorization.
        "schur_upper_velocity_ksp_type": "preonly",
        "schur_upper_velocity_solver": "hypre",
        "schur_upper_velocity_boomeramg_max_iter": 1,
        "schur_upper_velocity_boomeramg_tol": 0.0,

        "pressure_solver": "hypre",
        "pressure_ksp_type": "gmres",
        "pressure_ksp_rtol": 1.0e-2,
        "pressure_ksp_atol": 0.0,
        "pressure_ksp_dtol": 1.0e8,
        "pressure_ksp_max_it": 30,
        "pressure_ksp_restart": 30,
        "pressure_boomeramg_max_iter": 1,
        "pressure_boomeramg_tol": 0.0,
        "monitor": False,
        # FGMRES monitors the true residual of the scaled L A R system.
        # Because the production acceptance criterion is defined on the
        # original unscaled equations, automatically continue with a tighter
        # KSP tolerance if the physical residual is still above 1e-5.
        "norm_type": "unpreconditioned",
        "physical_residual_retry": True,
        "max_physical_residual_retries": 3,
        "retry_safety": 0.5,
        "minimum_rtol": 1.0e-12,
    },

    "direct_solver": {
        "solver_type": "mumps",
        "reuse_ordering": True,
        "reuse_fill": True,
        "ordering_type": None,
        "fill": None,
        "flow_preallocation_nnz": 20,
        "energy_preallocation_nnz": 5,
        "flow_true_residual_tolerance": 1.0e-5,
        "energy_true_residual_tolerance": 1.0e-8,
        "print_flow_diagnostics": True,
        "print_energy_diagnostics": False,
        "mumps_icntl": {},
        "mumps_cntl": {},
        "scaling": {
            "enabled": True,
            "velocity_scale": "auto",
            "pressure_scale": "auto",
            "rho_scale": "auto",
            "mu_scale": "auto",
            "minimum_velocity_scale": 1.0e-3,
        },
    },

    "performance": {
        "use_numba": True,
    },

    "profiling": {
        "enabled": True,
        "print_per_iteration": True,
        "save_timing_csv": True,
        "print_summary": True,
    },
}


# ============================================================
# MPI BOOTSTRAP — MUST RUN BEFORE NUMPY/SCIPY/PETSC IMPORTS
# ============================================================
import os
import shutil
import subprocess
import sys

_MPI_CHILD_FLAG = "V5_MPI_CHILD"
_PHYSICAL_CORES_ENV = "V5_PHYSICAL_CORES"
_LOGICAL_CPUS_ENV = "V5_LOGICAL_CPUS"


def _detected_mpi_size():
    for key in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
    ):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _detect_cpu_topology():
    """
    Return the CPU topology detected by the original parent process.

    Once MPI ranks are bound to individual cores, sched_getaffinity()
    only sees the CPUs assigned to that rank. Therefore the parent
    detects the full available topology before mpiexec and passes it
    to all MPI children through environment variables.
    """

    # --------------------------------------------------------
    # MPI children use the topology detected by the parent.
    # --------------------------------------------------------
    saved_physical = os.environ.get(_PHYSICAL_CORES_ENV)
    saved_logical = os.environ.get(_LOGICAL_CPUS_ENV)

    if saved_physical is not None and saved_logical is not None:
        return int(saved_physical), int(saved_logical)

    # --------------------------------------------------------
    # Parent process: determine CPUs currently available.
    # --------------------------------------------------------
    try:
        available_cpus = set(os.sched_getaffinity(0))
        logical_cpus = len(available_cpus)
    except (AttributeError, OSError):
        available_cpus = None
        logical_cpus = os.cpu_count() or 1

    physical_cores = None

    try:
        result = subprocess.run(
            ["lscpu", "-p=CPU,CORE,SOCKET"],
            capture_output=True,
            text=True,
            check=True,
        )

        cores = set()

        for line in result.stdout.splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            cpu_str, core_str, socket_str = line.split(",")

            cpu = int(cpu_str)
            core = int(core_str)
            socket = int(socket_str)

            # On the parent process this filters to CPUs actually
            # available to WSL/Linux.
            if available_cpus is not None and cpu not in available_cpus:
                continue

            cores.add((socket, core))

        if cores:
            physical_cores = len(cores)

    except Exception:
        pass

    if physical_cores is None:
        physical_cores = logical_cpus

    return physical_cores, logical_cpus


def _bootstrap_parallel_run() -> None:

    requested_processors = int(RUN_SETTINGS["processors"])

    # Detect whether we are already inside an MPI launch.
    detected_size = _detected_mpi_size()

    # --------------------------------------------------------
    # MPI CHILD PROCESS
    # --------------------------------------------------------
    if detected_size is not None:

        if detected_size != requested_processors:
            raise RuntimeError(
                "The active MPI launch has "
                f"{detected_size} ranks, but RUN_SETTINGS requests "
                f"{requested_processors} processors."
            )

        # Always keep numerical libraries single-threaded because
        # the current production parallel model is pure MPI.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"

        return

    # --------------------------------------------------------
    # ORIGINAL PARENT PROCESS
    # --------------------------------------------------------
    physical_cores, logical_cpus = _detect_cpu_topology()

    if requested_processors < 1:
        raise ValueError(
            "RUN_SETTINGS['processors'] must be at least 1."
        )

    if requested_processors > logical_cpus:
        raise ValueError(
            f"Requested {requested_processors} processors, "
            f"but Linux currently provides only "
            f"{logical_cpus} logical CPUs."
        )

    requested_ranks = requested_processors

    # Current implementation:
    # one MPI rank = one CPU worker.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

    # Single-rank case does not need mpiexec.
    if requested_processors == 1:

        # Preserve topology for reporting.
        os.environ[_PHYSICAL_CORES_ENV] = str(physical_cores)
        os.environ[_LOGICAL_CPUS_ENV] = str(logical_cpus)

        return

    launcher = shutil.which("mpiexec") or shutil.which("mpirun")

    if launcher is None:
        raise RuntimeError(
            "More than one processor was requested, but neither "
            "mpiexec nor mpirun was found in PATH."
        )

    environment = os.environ.copy()

    # Store the topology BEFORE MPI changes process affinity.
    environment[_MPI_CHILD_FLAG] = "1"
    environment[_PHYSICAL_CORES_ENV] = str(physical_cores)
    environment[_LOGICAL_CPUS_ENV] = str(logical_cpus)

    # --------------------------------------------------------
    # CPU PLACEMENT
    # --------------------------------------------------------
    if requested_processors <= physical_cores:

        # Prefer one MPI rank on each separate physical core.
        command = [
            launcher,
            "--report-bindings",
            "--map-by", "core",
            "--bind-to", "core",
            "-n", str(requested_ranks),
            sys.executable,
            "-m", "solver.steady_lam_v5",
            *sys.argv[1:],
        ]

    else:

        # More workers than physical cores:
        # deliberately use SMT hardware threads.
        command = [
            launcher,
            "--report-bindings",
            "--use-hwthread-cpus",
            "--map-by", "hwthread",
            "--bind-to", "hwthread",
            "-n", str(requested_ranks),
            sys.executable,
            "-m", "solver.steady_lam_v5",
            *sys.argv[1:],
        ]

    completed = subprocess.run(
        command,
        env=environment,
        check=False,
    )

    raise SystemExit(completed.returncode)


_bootstrap_parallel_run()


# ============================================================
# NORMAL IMPORTS
# ============================================================
import time
from pathlib import Path

import numpy as np

from case_io import load_case
from geometry import build_fluid_index_map, build_masks, build_solver_topology
from distributed_domain import StructuredSlabDomain, build_local_solver_topology
from initial_conditions import build_initial_flow, build_initial_temperature
from linear_backend import create_linear_solver
from materials import build_material_fields, build_source_fields, materials
from results_io import make_run_dir, save_case_with_results
from solver_equations import solve_energy, solve_pressure_velocity
from solver_reporting import (
    compute_global_energy_balance,
    make_profile_record,
    print_iteration_timing,
    print_timing_summary,
    save_residual_history,
    save_simulation_report,
    summarize_timing,
    write_timing_csv,
)
from solver_utils import (
    compute_face_fluxes,
    compute_global_mass_balance,
    compute_mass_residual,
    compute_pressure_gradients,
    initialize_solver_workspace,
    snapshot_fields,
)


def _resolve_case_path(project_root: Path):
    case_name = str(RUN_SETTINGS["case_name"])
    use_previous_solution = bool(RUN_SETTINGS["use_previous_solution"])
    restart_file = RUN_SETTINGS["restart_file"]
    case_file = RUN_SETTINGS["case_file"]

    if use_previous_solution and restart_file is not None:
        return Path(restart_file), "restart_file"
    if case_file is not None:
        mode = (
            "case_file_with_previous_solution"
            if use_previous_solution
            else "case_file_initial_conditions"
        )
        return Path(case_file), mode

    path = project_root / "case_files" / case_name / f"{case_name}.npz"
    mode = (
        "base_case_with_previous_solution"
        if use_previous_solution
        else "base_case_initial_conditions"
    )
    return path, mode


def _build_internal_settings(nx: int, ny: int):
    direct = RUN_SETTINGS["direct_solver"]
    iterative = RUN_SETTINGS["iterative_solver"]
    profiling = RUN_SETTINGS["profiling"]
    p_ref_i = int(
        round(float(RUN_SETTINGS["p_ref_i_fraction"]) * (nx - 1))
    )
    p_ref_j = int(
        round(float(RUN_SETTINGS["p_ref_j_fraction"]) * (ny - 1))
    )

    settings = {
        "max_iter": int(RUN_SETTINGS["max_iter"]),
        "tol_mass": float(RUN_SETTINGS["tol_mass"]),
        "tol_T": float(RUN_SETTINGS["tol_T"]),
        "physics": {
            "flow": True,
            "energy": bool(RUN_SETTINGS["enable_energy"]),
            "buoyancy": bool(RUN_SETTINGS["enable_buoyancy"]),
            "passive_scalar": False,
            "radiation": False,
            "transient": False,
        },
        "relaxation": {
            "u": float(RUN_SETTINGS["alpha_u"]),
            "v": float(RUN_SETTINGS["alpha_v"]),
            "p": float(RUN_SETTINGS["alpha_p"]),
            "T": float(RUN_SETTINGS["alpha_T"]),
        },
        "schemes": {
            "momentum": (
                "sou" if RUN_SETTINGS["enable_sou_momentum"] else "upwind"
            ),
            "energy": (
                "sou" if RUN_SETTINGS["enable_sou_energy"] else "upwind"
            ),
            "momentum_blend": float(RUN_SETTINGS["sou_blend_momentum"]),
            "energy_blend": float(RUN_SETTINGS["sou_blend_energy"]),
            "limiter": (
                "local_bounds"
                if RUN_SETTINGS["enable_sou_limiter"]
                else "none"
            ),
        },
        "pressure_reference": {
            "mode": str(RUN_SETTINGS["pressure_reference_mode"]),
            "enabled": bool(RUN_SETTINGS["use_pressure_reference"]),
            "value": float(RUN_SETTINGS["p_ref_value"]),
            "i": p_ref_i,
            "j": p_ref_j,
        },
        "linear_solver": {
            "linear_solver_type": str(RUN_SETTINGS["linear_solver_type"]),
            "solver_type": str(direct["solver_type"]),
            "reuse_ordering": bool(direct["reuse_ordering"]),
            "reuse_fill": bool(direct["reuse_fill"]),
            "ordering_type": direct["ordering_type"],
            "fill": direct["fill"],
            "mumps_icntl": dict(direct["mumps_icntl"]),
            "mumps_cntl": dict(direct["mumps_cntl"]),
            "error_on_nonconvergence": True,
            "use_options_database": True,
            "verbose": True,
            "profiling": dict(profiling),
            "iterative": {
                "options_prefix": "flowv5_iter_",
                "ksp_type": str(iterative["ksp_type"]),
                "rtol": float(iterative["rtol"]),
                "atol": float(iterative["atol"]),
                "dtol": float(iterative["dtol"]),
                "max_it": int(iterative["max_it"]),
                "restart": int(iterative["restart"]),
                "schur_fact_type": str(iterative["schur_fact_type"]),
                "schur_pre_type": str(iterative["schur_pre_type"]),
                "velocity_solver": str(iterative["velocity_solver"]),
                "velocity_ksp_type": str(iterative["velocity_ksp_type"]),
                "velocity_ksp_rtol": float(iterative["velocity_ksp_rtol"]),
                "velocity_ksp_atol": float(iterative["velocity_ksp_atol"]),
                "velocity_ksp_dtol": float(iterative["velocity_ksp_dtol"]),
                "velocity_ksp_max_it": int(iterative["velocity_ksp_max_it"]),
                "velocity_ksp_restart": int(iterative["velocity_ksp_restart"]),
                "velocity_boomeramg_max_iter": int(iterative["velocity_boomeramg_max_iter"]),
                "velocity_boomeramg_tol": float(iterative["velocity_boomeramg_tol"]),
                "schur_inner_velocity_ksp_type": str(
                    iterative["schur_inner_velocity_ksp_type"]
                ),
                "schur_inner_velocity_solver": str(
                    iterative["schur_inner_velocity_solver"]
                ),
                "schur_inner_velocity_boomeramg_max_iter": int(
                    iterative["schur_inner_velocity_boomeramg_max_iter"]
                ),
                "schur_inner_velocity_boomeramg_tol": float(
                    iterative["schur_inner_velocity_boomeramg_tol"]
                ),
                "schur_upper_velocity_ksp_type": str(
                    iterative["schur_upper_velocity_ksp_type"]
                ),
                "schur_upper_velocity_solver": str(
                    iterative["schur_upper_velocity_solver"]
                ),
                "schur_upper_velocity_boomeramg_max_iter": int(
                    iterative["schur_upper_velocity_boomeramg_max_iter"]
                ),
                "schur_upper_velocity_boomeramg_tol": float(
                    iterative["schur_upper_velocity_boomeramg_tol"]
                ),
                "pressure_solver": str(iterative["pressure_solver"]),
                "pressure_ksp_type": str(iterative["pressure_ksp_type"]),
                "pressure_ksp_rtol": float(iterative["pressure_ksp_rtol"]),
                "pressure_ksp_atol": float(iterative["pressure_ksp_atol"]),
                "pressure_ksp_dtol": float(iterative["pressure_ksp_dtol"]),
                "pressure_ksp_max_it": int(iterative["pressure_ksp_max_it"]),
                "pressure_ksp_restart": int(iterative["pressure_ksp_restart"]),
                "pressure_boomeramg_max_iter": int(
                    iterative["pressure_boomeramg_max_iter"]
                ),
                "pressure_boomeramg_tol": float(
                    iterative["pressure_boomeramg_tol"]
                ),
                "monitor": bool(iterative["monitor"]),
                "norm_type": str(iterative["norm_type"]),
                "physical_residual_retry": bool(iterative["physical_residual_retry"]),
                "max_physical_residual_retries": int(iterative["max_physical_residual_retries"]),
                "retry_safety": float(iterative["retry_safety"]),
                "minimum_rtol": float(iterative["minimum_rtol"]),
            },
            "flow_coupled": {
                "options_prefix": "flowv5_",
                "preallocation_nnz": int(
                    direct["flow_preallocation_nnz"]
                ),
                "true_residual_norm": "inf",
                "true_residual_tolerance": float(
                    direct["flow_true_residual_tolerance"]
                ),
                "print_diagnostics": bool(
                    direct["print_flow_diagnostics"]
                ),
                "scaling": dict(direct["scaling"]),
            },
            "energy": {
                "options_prefix": "energy_",
                "preallocation_nnz": int(
                    direct["energy_preallocation_nnz"]
                ),
                "true_residual_norm": "inf",
                "true_residual_tolerance": float(
                    direct["energy_true_residual_tolerance"]
                ),
                "print_diagnostics": bool(
                    direct["print_energy_diagnostics"]
                ),
            },
        },
        "profiling": dict(profiling),
    }
    return settings, p_ref_i, p_ref_j




def _choose_pressure_reference(full_is_fluid, p_ref_i, p_ref_j):
    ny, nx = full_is_fluid.shape
    if 0 <= p_ref_i < nx and 0 <= p_ref_j < ny and full_is_fluid[p_ref_j, p_ref_i]:
        return int(p_ref_i), int(p_ref_j)
    locations = np.argwhere(full_is_fluid)
    if locations.size == 0:
        raise RuntimeError("The case contains no fluid cells.")
    j, i = locations[0]
    return int(i), int(j)


def _pressure_reference_fid(full_is_fluid, p_ref_i, p_ref_j):
    row_counts = np.count_nonzero(full_is_fluid, axis=1).astype(np.int64)
    row_offsets = np.zeros(full_is_fluid.shape[0] + 1, dtype=np.int64)
    row_offsets[1:] = np.cumsum(row_counts)
    before = int(np.count_nonzero(full_is_fluid[p_ref_j, :p_ref_i]))
    return int(row_offsets[p_ref_j] + before)


def _build_distributed_context(
    geom,
    settings,
    domain,
    topology,
    local_masks,
    local_material,
    local_sources,
    p_ref_i,
    p_ref_j,
    pressure_reference_fid,
    rho_reference_scale,
    mu_reference_scale,
):
    nx = int(geom["nx"])
    ny = int(geom["ny"])
    lx = float(geom["Lx"])
    ly = float(geom["Ly"])
    dx = lx / nx
    dy = ly / ny
    return {
        "nx": nx,
        "ny": ny,
        "Lx": lx,
        "Ly": ly,
        "dx": dx,
        "dy": dy,
        "V": dx * dy,
        "domain": domain,
        "topology": topology,
        "is_fluid": local_masks["fluid"],
        "is_solid": local_masks["solid"],
        "Nf": int(topology["Nf"]),
        "local_Nf": int(topology["local_Nf"]),
        "BC_flow": geom["boundaries"]["flow"],
        "BC_heat": geom["boundaries"]["heat"],
        "rho": local_material["rho"],
        "mu": local_material["mu"],
        "cp": local_material["cp"],
        "k": local_material["k"],
        "beta": local_material["beta"],
        "qdot": local_sources["energy"],
        "sx": local_sources["momentum_x"],
        "sy": local_sources["momentum_y"],
        "rho_reference_scale": float(rho_reference_scale),
        "mu_reference_scale": float(mu_reference_scale),
        "T_ref": float(RUN_SETTINGS["T_ref"]),
        "gx": float(RUN_SETTINGS["gx"]),
        "gy": float(RUN_SETTINGS["gy"]),
        "use_pressure_reference": bool(RUN_SETTINGS["use_pressure_reference"]),
        "p_ref_value": float(RUN_SETTINGS["p_ref_value"]),
        "p_ref_i": int(p_ref_i),
        "p_ref_j": int(p_ref_j),
        "pressure_reference_fid": int(pressure_reference_fid),
        "enable_sou_limiter": settings["schemes"].get("limiter", "none") != "none",
        "use_numba": bool(RUN_SETTINGS.get("performance", {}).get("use_numba", True)),
    }


def _build_reporting_context(geom, settings, material_fields, source_fields, p_ref_i, p_ref_j):
    """Recreate the old full-grid context only once on Rank 0 after convergence."""
    geom["masks"] = build_masks(geom["region"], geom["region_defs"])
    index_data = build_fluid_index_map(geom["region"], geom["region_defs"])
    nx = int(geom["nx"])
    ny = int(geom["ny"])
    lx = float(geom["Lx"])
    ly = float(geom["Ly"])
    ctx = {
        "nx": nx,
        "ny": ny,
        "Lx": lx,
        "Ly": ly,
        "dx": lx / nx,
        "dy": ly / ny,
        "V": (lx / nx) * (ly / ny),
        "is_fluid": geom["masks"]["fluid"],
        "is_solid": geom["masks"]["solid"],
        "fluid_cells": index_data["fluid_cells"],
        "cell_to_fid": index_data["cell_to_fid"],
        "Nf": int(index_data["Nf"]),
        "BC_flow": geom["boundaries"]["flow"],
        "BC_heat": geom["boundaries"]["heat"],
        "rho": material_fields["rho"],
        "mu": material_fields["mu"],
        "cp": material_fields["cp"],
        "k": material_fields["k"],
        "beta": material_fields["beta"],
        "qdot": source_fields["energy"],
        "sx": source_fields["momentum_x"],
        "sy": source_fields["momentum_y"],
        "T_ref": float(RUN_SETTINGS["T_ref"]),
        "gx": float(RUN_SETTINGS["gx"]),
        "gy": float(RUN_SETTINGS["gy"]),
        "use_pressure_reference": bool(RUN_SETTINGS["use_pressure_reference"]),
        "p_ref_value": float(RUN_SETTINGS["p_ref_value"]),
        "p_ref_i": int(p_ref_i),
        "p_ref_j": int(p_ref_j),
        "enable_sou_limiter": settings["schemes"].get("limiter", "none") != "none",
        "use_numba": bool(RUN_SETTINGS.get("performance", {}).get("use_numba", True)),
    }
    ctx["topology"] = build_solver_topology(
        nx=nx,
        ny=ny,
        is_fluid=ctx["is_fluid"],
        is_solid=ctx["is_solid"],
        index_data=index_data,
        flow_boundaries=geom["boundaries"]["flow"],
        heat_boundaries=geom["boundaries"]["heat"],
    )
    initialize_solver_workspace(ctx)
    return ctx


def _local_inf_change(domain, new, old, mask=None):
    owned = domain.owned_slice
    diff = np.abs(new[owned, :] - old[owned, :])
    if mask is not None:
        owned_mask = mask[owned, :]
        local = float(np.max(diff[owned_mask])) if np.any(owned_mask) else 0.0
    else:
        local = float(np.max(diff)) if diff.size else 0.0
    return domain.allreduce_max(local)


def main() -> None:
    if int(RUN_SETTINGS["max_iter"]) < 1:
        raise ValueError("RUN_SETTINGS['max_iter'] must be at least 1.")

    project_root = Path(__file__).resolve().parent.parent
    case_path, initialization_mode = _resolve_case_path(project_root)
    if not case_path.exists():
        raise FileNotFoundError(f"Case/restart file not found: {case_path}")

    # Case metadata are initially loaded on every rank.  Full CFD arrays are
    # immediately localized and released; only Rank 0 retains the case geometry
    # needed for final NPZ output.
    geom = load_case(case_path)
    case_name = str(geom["case_name"])
    nx = int(geom["nx"])
    ny = int(geom["ny"])
    lx = float(geom["Lx"])
    ly = float(geom["Ly"])
    dx = lx / nx
    dy = ly / ny
    settings, p_ref_i, p_ref_j = _build_internal_settings(nx, ny)

    linear_solver = create_linear_solver(settings["linear_solver"])
    mpi_rank = linear_solver.rank
    mpi_size = linear_solver.size
    is_root = mpi_rank == 0
    requested_processors = int(RUN_SETTINGS["processors"])

    if mpi_size != requested_processors:
        raise RuntimeError(
            f"PETSc started with {mpi_size} MPI ranks, but "
            f"RUN_SETTINGS requests {requested_processors} processors."
        )
    mpi_comm = linear_solver.PETSc.COMM_WORLD.tompi4py()
    domain = StructuredSlabDomain(nx, ny, mpi_comm, halo=2)

    full_masks = build_masks(geom["region"], geom["region_defs"])
    full_is_fluid = full_masks["fluid"]
    full_is_solid = full_masks["solid"]
    p_ref_i, p_ref_j = _choose_pressure_reference(full_is_fluid, p_ref_i, p_ref_j)
    settings["pressure_reference"]["i"] = p_ref_i
    settings["pressure_reference"]["j"] = p_ref_j
    p_ref_fid = _pressure_reference_fid(full_is_fluid, p_ref_i, p_ref_j)

    topology = build_local_solver_topology(
        domain=domain,
        global_is_fluid=full_is_fluid,
        global_is_solid=full_is_solid,
        flow_boundaries=geom["boundaries"]["flow"],
        heat_boundaries=geom["boundaries"]["heat"],
    )

    full_material = build_material_fields(geom, materials)
    full_sources = build_source_fields(geom)
    if not np.any(full_is_fluid):
        raise RuntimeError("The case contains no fluid cells.")
    rho_reference_scale = float(np.median(full_material["rho"][full_is_fluid]))
    mu_reference_scale = float(np.median(full_material["mu"][full_is_fluid]))

    use_previous_solution = bool(RUN_SETTINGS["use_previous_solution"])
    full_flow_initial = build_initial_flow(geom, prefer_solved=use_previous_solution)
    full_temperature = build_initial_temperature(geom, prefer_solved=use_previous_solution)
    full_flow_initial["u"][full_is_solid] = 0.0
    full_flow_initial["v"][full_is_solid] = 0.0
    full_flow_initial["p"][full_is_solid] = 0.0

    local_masks = {
        "fluid": domain.localize(full_is_fluid, False),
        "solid": domain.localize(full_is_solid, False),
    }
    local_material = {name: domain.localize(value, 0.0) for name, value in full_material.items()}
    local_sources = {name: domain.localize(value, 0.0) for name, value in full_sources.items()}
    fields = {
        "u": domain.localize(full_flow_initial["u"], 0.0),
        "v": domain.localize(full_flow_initial["v"], 0.0),
        "p": domain.localize(full_flow_initial["p"], 0.0),
        "T": domain.localize(full_temperature, float(RUN_SETTINGS["T_ref"])),
    }

    ctx = _build_distributed_context(
        geom,
        settings,
        domain,
        topology,
        local_masks,
        local_material,
        local_sources,
        p_ref_i,
        p_ref_j,
        p_ref_fid,
        rho_reference_scale,
        mu_reference_scale,
    )
    initialize_solver_workspace(ctx)

    aPu_lag = np.maximum(4.0 * local_material["mu"].copy(), 1.0e-6)
    aPv_lag = np.maximum(4.0 * local_material["mu"].copy(), 1.0e-6)
    aPu_lag[local_masks["solid"]] = 1.0
    aPv_lag[local_masks["solid"]] = 1.0
    coeffs = {"aPu": aPu_lag, "aPv": aPv_lag}
    domain.exchange_many((fields["u"], fields["v"], fields["p"], fields["T"], coeffs["aPu"], coeffs["aPv"]))

    # Full runtime CFD arrays are no longer replicated after this point.
    del full_flow_initial, full_temperature, full_material, full_sources
    del full_masks, full_is_fluid, full_is_solid
    if not is_root:
        # Only Rank 0 keeps the global region map for final result-file output.
        geom.pop("region", None)
        geom.pop("solved_flow", None)
        geom.pop("solved_temperature", None)

    if is_root:
        print(f"Linear solver backend: {linear_solver.describe()}")
        physical_cores, logical_cpus = _detect_cpu_topology()
        using_smt = int(RUN_SETTINGS["processors"]) > physical_cores
        print("")
        print("Parallel configuration:")
        print(f"  Requested processors : {int(RUN_SETTINGS['processors'])}")
        print(f"  Physical CPU cores    : {physical_cores}")
        print(f"  Logical CPUs          : {logical_cpus}")
        print(f"  MPI ranks             : {mpi_size}")
        print("  Threads per MPI rank  : 1")
        print(f"  SMT used              : {'Yes' if using_smt else 'No'}")
        print("Performance path: local+halo Numba kernels + persistent PETSc fixed-COO")
        print(f"Domain decomposition: y-slabs with halo=2 ({ny} rows across {mpi_size} ranks)")
        print(f"Loaded case/restart file: {case_path}")
        print(f"Initialization mode: {initialization_mode}")
        print(f"Use previous solution: {use_previous_solution}")

    if is_root:
        run_dir_value, run_tag = make_run_dir(
            results_root="results", prefix=f"{case_name}_steady"
        )
        run_payload = (str(run_dir_value), str(run_tag))
    else:
        run_payload = None
    run_payload = linear_solver.broadcast(run_payload, root=0)
    run_dir = Path(run_payload[0])
    run_tag = str(run_payload[1])

    histories = {
        "hist_it": [], "hist_du": [], "hist_dv": [], "hist_dp": [],
        "hist_dT": [], "hist_mass": [],
    }
    profile_records = []
    profiling = settings["profiling"]
    profiling_enabled = bool(profiling["enabled"])
    max_iter = int(settings["max_iter"])
    tol_mass = float(settings["tol_mass"])
    tol_temperature = float(settings["tol_T"])

    dpdx, dpdy = compute_pressure_gradients(ctx, fields["p"])
    gradients = {"dpdx": dpdx, "dpdy": dpdy}
    fluxes = compute_face_fluxes(ctx, settings, fields, coeffs, gradients)
    fluxes.update({"dpdx": dpdx, "dpdy": dpdy, "gradients": gradients})

    start_time = time.perf_counter()
    try:
        for iteration in range(1, max_iter + 1):
            iteration_start = time.perf_counter()
            fields_old = snapshot_fields(ctx, fields)

            fields, coeffs, fluxes = solve_pressure_velocity(
                ctx, settings, fields, coeffs,
                transient=None,
                linear_solver=linear_solver,
                old_fields=fields_old,
            )
            flow_timing = fluxes.get("timing", {})
            gradients = fluxes["gradients"]

            if settings["physics"]["energy"]:
                owned = domain.owned_slice
                if not np.all(np.isfinite(fields["T"][owned, :])):
                    raise RuntimeError("Non-finite temperature values exist before the energy solve.")
                fields["T"] = solve_energy(
                    ctx, settings, fields, fluxes,
                    transient=None,
                    linear_solver=linear_solver,
                )
                energy_timing = settings.pop("_last_energy_timing", {})
                dT_inf = _local_inf_change(domain, fields["T"], fields_old["T"])
            else:
                energy_timing = {}
                dT_inf = 0.0

            metrics_start = time.perf_counter()
            du_inf = _local_inf_change(domain, fields["u"], fields_old["u"], ctx["is_fluid"])
            dv_inf = _local_inf_change(domain, fields["v"], fields_old["v"], ctx["is_fluid"])
            dp_inf = _local_inf_change(domain, fields["p"], fields_old["p"], ctx["is_fluid"])
            mass_residual = float(
                compute_mass_residual(ctx, settings, fields, coeffs, gradients, fluxes=fluxes)
            )
            metrics_time = time.perf_counter() - metrics_start
            outer_time = time.perf_counter() - iteration_start

            if profiling_enabled:
                flow_timing = linear_solver.reduce_timing_max(flow_timing)
                energy_timing = linear_solver.reduce_timing_max(energy_timing)
                metrics_time = linear_solver.allreduce_max(metrics_time)
                outer_time = linear_solver.allreduce_max(outer_time)

            histories["hist_it"].append(iteration)
            histories["hist_du"].append(du_inf)
            histories["hist_dv"].append(dv_inf)
            histories["hist_dp"].append(dp_inf)
            histories["hist_dT"].append(dT_inf)
            histories["hist_mass"].append(mass_residual)

            if is_root:
                print(
                    f"it {iteration:4d}/{max_iter}  "
                    f"|du|inf {du_inf:.3e} |dv|inf {dv_inf:.3e}  "
                    f"|dp|inf {dp_inf:.3e} |dT|inf {dT_inf:.3e} | "
                    f"massRes {mass_residual:.3e} | "
                    f"elapsed {time.perf_counter() - start_time:.2f}s"
                )

            if profiling_enabled:
                record = make_profile_record(
                    iteration, outer_time, metrics_time, flow_timing, energy_timing
                )
                profile_records.append(record)
                if is_root and profiling["print_per_iteration"]:
                    print_iteration_timing(record)

            converged = mass_residual < tol_mass
            if settings["physics"]["energy"]:
                converged = converged and dT_inf < tol_temperature
            if converged:
                if is_root:
                    print("Converged.")
                break

        # Final solution gather is intentionally performed only once, after the
        # iterative solve.  There is no per-iteration all-gather in Phase F.
        gathered_fields = {
            name: domain.gather_owned_field(array, root=0)
            for name, array in fields.items()
        }
        gathered_coeffs = {
            name: domain.gather_owned_field(array, root=0)
            for name, array in coeffs.items()
        }

        if is_root:
            total_time = time.perf_counter() - start_time
            full_masks_report = build_masks(geom["region"], geom["region_defs"])
            geom["masks"] = full_masks_report
            material_fields = build_material_fields(geom, materials)
            source_fields = build_source_fields(geom)
            report_ctx = _build_reporting_context(
                geom, settings, material_fields, source_fields, p_ref_i, p_ref_j
            )
            full_fields = gathered_fields
            full_coeffs = gathered_coeffs
            dpdx_full, dpdy_full = compute_pressure_gradients(report_ctx, full_fields["p"])
            gradients_full = {"dpdx": dpdx_full, "dpdy": dpdy_full}
            fluxes_full = compute_face_fluxes(
                report_ctx, settings, full_fields, full_coeffs, gradients_full
            )

            x_coordinates = np.linspace(dx / 2.0, lx - dx / 2.0, nx)
            y_coordinates = np.linspace(dy / 2.0, ly - dy / 2.0, ny)
            save_residual_history(
                run_dir,
                histories["hist_it"], histories["hist_du"], histories["hist_dv"],
                histories["hist_dp"], histories["hist_mass"], histories["hist_dT"],
                "residual_history.png",
            )
            save_case_with_results(
                case_path,
                geom,
                run_tag=run_tag,
                solution_fields=full_fields,
                histories={key: np.asarray(value) for key, value in histories.items()},
                coordinates={"x": x_coordinates, "y": y_coordinates},
                extra_meta={
                    "solver_name": "steady_v5_direct_mumps_phase_df",
                    "run_tag": run_tag,
                    "initialization_mode": initialization_mode,
                    "use_previous_solution": use_previous_solution,
                    "run_settings": RUN_SETTINGS,
                    "decomposition": "structured_y_slab_halo2",
                },
            )

            mass_balance = compute_global_mass_balance(
                report_ctx, full_fields, full_coeffs, gradients_full
            )
            energy_balance = compute_global_energy_balance(
                report_ctx, settings, full_fields, full_coeffs, fluxes_full, gradients_full
            )

            timing_csv_path = None
            if profiling_enabled and profiling["save_timing_csv"]:
                timing_csv_path = write_timing_csv(
                    run_dir / f"timing_history_{run_tag}.csv", profile_records
                )
            timing_summary = summarize_timing(profile_records, timing_csv_path)
            if profiling_enabled and profiling["print_summary"]:
                print_timing_summary(timing_summary)

            is_fluid_global = full_masks_report["fluid"]
            report_data = {
                "setup": {
                    "case_name": case_name,
                    "case_path": case_path,
                    "initialization_mode": initialization_mode,
                    "use_previous_solution": use_previous_solution,
                    "restart_file": RUN_SETTINGS["restart_file"],
                },
                "grid": {"nx": nx, "ny": ny, "dx": dx, "dy": dy},
                "cells": {
                    "total": nx * ny,
                    "fluid": int(np.sum(is_fluid_global)),
                    "solid": int(np.sum(full_masks_report["solid"])),
                },
                "properties": {
                    "rho_min": float(np.min(material_fields["rho"][is_fluid_global])),
                    "rho_max": float(np.max(material_fields["rho"][is_fluid_global])),
                    "mu_min": float(np.min(material_fields["mu"][is_fluid_global])),
                    "mu_max": float(np.max(material_fields["mu"][is_fluid_global])),
                    "beta_min": float(np.min(material_fields["beta"][is_fluid_global])),
                    "beta_max": float(np.max(material_fields["beta"][is_fluid_global])),
                },
                "flags": {
                    "ENABLE_ENERGY": bool(RUN_SETTINGS["enable_energy"]),
                    "ENABLE_BUOYANCY": bool(RUN_SETTINGS["enable_buoyancy"]),
                    "ENABLE_SOU_MOMENTUM": bool(RUN_SETTINGS["enable_sou_momentum"]),
                    "ENABLE_SOU_ENERGY": bool(RUN_SETTINGS["enable_sou_energy"]),
                    "SOU_BLEND_MOMENTUM": float(RUN_SETTINGS["sou_blend_momentum"]),
                    "SOU_BLEND_ENERGY": float(RUN_SETTINGS["sou_blend_energy"]),
                    "ENABLE_SOU_LIMITER": bool(RUN_SETTINGS["enable_sou_limiter"]),
                    "alpha_u": float(RUN_SETTINGS["alpha_u"]),
                    "alpha_v": float(RUN_SETTINGS["alpha_v"]),
                    "alpha_p": float(RUN_SETTINGS["alpha_p"]),
                    "alpha_T": float(RUN_SETTINGS["alpha_T"]),
                    "max_iter": max_iter,
                    "tol_mass": tol_mass,
                    "tol_T": tol_temperature,
                    "T_ref": float(RUN_SETTINGS["T_ref"]),
                    "gx": float(RUN_SETTINGS["gx"]),
                    "gy": float(RUN_SETTINGS["gy"]),
                    "processors": int(RUN_SETTINGS["processors"]),
                    "mpi_ranks": mpi_size,
                    "threads_per_rank": 1,
                    "physical_cores": physical_cores,
                    "logical_cpus": logical_cpus,
                    "smt_used": bool(using_smt),
                    "linear_solver": str(RUN_SETTINGS["linear_solver_type"]),
                    "direct_solver": str(RUN_SETTINGS["direct_solver"]["solver_type"]),
                    "use_numba": bool(RUN_SETTINGS.get("performance", {}).get("use_numba", True)),
                    "decomposition": "structured_y_slab_halo2",
                },
                "histories": histories,
                "temperature": {
                    "Tmin": float(np.min(full_fields["T"])),
                    "Tmax": float(np.max(full_fields["T"])),
                },
                "sources": {
                    "qdot_min": float(np.min(source_fields["energy"])),
                    "qdot_max": float(np.max(source_fields["energy"])),
                    "sx_min": float(np.min(source_fields["momentum_x"])),
                    "sx_max": float(np.max(source_fields["momentum_x"])),
                    "sy_min": float(np.min(source_fields["momentum_y"])),
                    "sy_max": float(np.max(source_fields["momentum_y"])),
                },
                "mass_balance": mass_balance,
                "energy_balance": energy_balance,
                "performance": {
                    "total_time": total_time,
                    "avg_time_per_iter": total_time / max(len(histories["hist_it"]), 1),
                    "profiling_enabled": profiling_enabled,
                    "timing_summary": timing_summary,
                },
            }
            save_simulation_report(
                run_dir, f"simulation_report_{run_tag}.txt", report_data
            )
            print(f"\nSaved thermo-flow results to: {run_dir}")
    except Exception:
        linear_solver.close()
        raise
    else:
        linear_solver.barrier()
        linear_solver.close()


if __name__ == "__main__":
    main()