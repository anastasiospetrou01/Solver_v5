from __future__ import annotations

# ============================================================
# RUN SETTINGS — DIRECT-LU PRODUCTION VERSION
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

    "mpi_ranks": 4,
    "threads_per_rank": 1,

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

    "profiling": {
        "enabled": True,
        "print_per_iteration": True,
        "save_timing_csv": False,
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


def _bootstrap_parallel_run() -> None:
    requested_ranks = int(RUN_SETTINGS["mpi_ranks"])
    threads = int(RUN_SETTINGS["threads_per_rank"])
    if requested_ranks < 1:
        raise ValueError("RUN_SETTINGS['mpi_ranks'] must be at least 1.")
    if threads < 1:
        raise ValueError("RUN_SETTINGS['threads_per_rank'] must be at least 1.")

    thread_value = str(threads)
    os.environ["OMP_NUM_THREADS"] = thread_value
    os.environ["OPENBLAS_NUM_THREADS"] = thread_value
    os.environ["MKL_NUM_THREADS"] = thread_value
    os.environ["NUMEXPR_NUM_THREADS"] = thread_value

    detected_size = _detected_mpi_size()
    if detected_size is not None:
        if detected_size != requested_ranks:
            raise RuntimeError(
                "The active MPI launch has "
                f"{detected_size} ranks, but RUN_SETTINGS requests "
                f"{requested_ranks}. Run without mpiexec or use matching values."
            )
        return

    if os.environ.get(_MPI_CHILD_FLAG) == "1" or requested_ranks == 1:
        return

    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        raise RuntimeError(
            "mpi_ranks is greater than 1, but neither mpiexec nor mpirun "
            "was found in PATH."
        )

    environment = os.environ.copy()
    environment[_MPI_CHILD_FLAG] = "1"
    command = [
        launcher,
        "-n",
        str(requested_ranks),
        sys.executable,
        "-m",
        "solver.steady_lam_v5",
        *sys.argv[1:],
    ]
    completed = subprocess.run(command, env=environment, check=False)
    raise SystemExit(completed.returncode)


_bootstrap_parallel_run()


# ============================================================
# NORMAL IMPORTS
# ============================================================
import time
from pathlib import Path

import numpy as np

from case_io import load_case
from geometry import build_fluid_index_map, build_masks
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
    fluid_at,
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


def _build_context(
    geom,
    settings,
    material_fields,
    source_fields,
    index_data,
    p_ref_i,
    p_ref_j,
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
        "use_pressure_reference": bool(
            RUN_SETTINGS["use_pressure_reference"]
        ),
        "p_ref_value": float(RUN_SETTINGS["p_ref_value"]),
        "p_ref_i": int(p_ref_i),
        "p_ref_j": int(p_ref_j),
        "enable_sou_limiter": (
            settings["schemes"].get("limiter", "none") != "none"
        ),
    }


def main() -> None:
    if int(RUN_SETTINGS["max_iter"]) < 1:
        raise ValueError("RUN_SETTINGS['max_iter'] must be at least 1.")

    project_root = Path(__file__).resolve().parent.parent
    case_path, initialization_mode = _resolve_case_path(project_root)
    if not case_path.exists():
        raise FileNotFoundError(f"Case/restart file not found: {case_path}")

    geom = load_case(case_path)
    geom["masks"] = build_masks(geom["region"], geom["region_defs"])
    case_name = str(geom["case_name"])
    nx = int(geom["nx"])
    ny = int(geom["ny"])
    lx = float(geom["Lx"])
    ly = float(geom["Ly"])
    dx = lx / nx
    dy = ly / ny
    is_fluid = geom["masks"]["fluid"]
    is_solid = geom["masks"]["solid"]

    settings, p_ref_i, p_ref_j = _build_internal_settings(nx, ny)
    material_fields = build_material_fields(geom, materials)
    source_fields = build_source_fields(geom)
    index_data = build_fluid_index_map(geom["region"], geom["region_defs"])
    ctx = _build_context(
        geom,
        settings,
        material_fields,
        source_fields,
        index_data,
        p_ref_i,
        p_ref_j,
    )

    linear_solver = create_linear_solver(settings["linear_solver"])
    mpi_rank = linear_solver.rank
    mpi_size = linear_solver.size
    is_root = mpi_rank == 0
    requested_ranks = int(RUN_SETTINGS["mpi_ranks"])
    if mpi_size != requested_ranks:
        raise RuntimeError(
            f"PETSc started with {mpi_size} ranks, but RUN_SETTINGS requests "
            f"{requested_ranks}."
        )

    if is_root:
        print(f"Linear solver backend: {linear_solver.describe()}")
        print(
            f"MPI ranks: {mpi_size} | "
            f"threads/rank: {int(RUN_SETTINGS['threads_per_rank'])}"
        )

    use_previous_solution = bool(RUN_SETTINGS["use_previous_solution"])
    flow_initial = build_initial_flow(
        geom, prefer_solved=use_previous_solution
    )
    fields = {
        "u": flow_initial["u"].copy(),
        "v": flow_initial["v"].copy(),
        "p": flow_initial["p"].copy(),
        "T": build_initial_temperature(
            geom, prefer_solved=use_previous_solution
        ),
    }
    fields["u"][is_solid] = 0.0
    fields["v"][is_solid] = 0.0
    fields["p"][is_solid] = 0.0

    aPu_lag = np.maximum(4.0 * material_fields["mu"].copy(), 1.0e-6)
    aPv_lag = np.maximum(4.0 * material_fields["mu"].copy(), 1.0e-6)
    aPu_lag[is_solid] = 1.0
    aPv_lag[is_solid] = 1.0
    coeffs = {"aPu": aPu_lag, "aPv": aPv_lag}

    if (
        RUN_SETTINGS["use_pressure_reference"]
        and not fluid_at(ctx, p_ref_i, p_ref_j)
    ):
        if not ctx["fluid_cells"]:
            raise RuntimeError("The case contains no fluid cells.")
        p_ref_i, p_ref_j = ctx["fluid_cells"][0]
        ctx["p_ref_i"] = p_ref_i
        ctx["p_ref_j"] = p_ref_j
        settings["pressure_reference"]["i"] = p_ref_i
        settings["pressure_reference"]["j"] = p_ref_j

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

    if is_root:
        print(f"Loaded case/restart file: {case_path}")
        print(f"Initialization mode: {initialization_mode}")
        print(f"Use previous solution: {use_previous_solution}")

    histories = {
        "hist_it": [],
        "hist_du": [],
        "hist_dv": [],
        "hist_dp": [],
        "hist_dT": [],
        "hist_mass": [],
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

    start_time = time.time()
    try:
        for iteration in range(1, max_iter + 1):
            iteration_start = time.perf_counter()
            fields_old = {
                "u": fields["u"].copy(),
                "v": fields["v"].copy(),
                "p": fields["p"].copy(),
                "T": fields["T"].copy(),
            }

            fields, coeffs, fluxes = solve_pressure_velocity(
                ctx,
                settings,
                fields,
                coeffs,
                transient=None,
                linear_solver=linear_solver,
            )
            flow_timing = fluxes.get("timing", {})
            gradients = fluxes["gradients"]

            if settings["physics"]["energy"]:
                if not np.all(np.isfinite(fields["T"])):
                    raise RuntimeError(
                        "Non-finite temperature values exist before the energy solve."
                    )
                fields["T"] = solve_energy(
                    ctx,
                    settings,
                    fields,
                    fluxes,
                    transient=None,
                    linear_solver=linear_solver,
                )
                energy_timing = settings.pop("_last_energy_timing", {})
                dT_inf = float(
                    np.max(np.abs(fields["T"] - fields_old["T"]))
                )
            else:
                energy_timing = {}
                dT_inf = 0.0

            metrics_start = time.perf_counter()
            du_inf = float(
                np.max(np.abs((fields["u"] - fields_old["u"])[is_fluid]))
            )
            dv_inf = float(
                np.max(np.abs((fields["v"] - fields_old["v"])[is_fluid]))
            )
            dp_inf = float(
                np.max(np.abs((fields["p"] - fields_old["p"])[is_fluid]))
            )
            mass_residual = float(
                compute_mass_residual(
                    ctx, settings, fields, coeffs, gradients
                )
            )
            metrics_time = time.perf_counter() - metrics_start
            outer_time = time.perf_counter() - iteration_start

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
                    f"elapsed {time.time() - start_time:.2f}s"
                )

            if profiling_enabled:
                record = make_profile_record(
                    iteration,
                    outer_time,
                    metrics_time,
                    flow_timing,
                    energy_timing,
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

        if is_root:
            x_coordinates = np.linspace(dx / 2.0, lx - dx / 2.0, nx)
            y_coordinates = np.linspace(dy / 2.0, ly - dy / 2.0, ny)

            save_residual_history(
                run_dir,
                histories["hist_it"],
                histories["hist_du"],
                histories["hist_dv"],
                histories["hist_dp"],
                histories["hist_mass"],
                histories["hist_dT"],
                "residual_history.png",
            )
            save_case_with_results(
                case_path,
                geom,
                run_tag=run_tag,
                solution_fields=fields,
                histories={
                    key: np.asarray(value)
                    for key, value in histories.items()
                },
                coordinates={"x": x_coordinates, "y": y_coordinates},
                extra_meta={
                    "solver_name": "steady_v5_direct_mumps",
                    "run_tag": run_tag,
                    "initialization_mode": initialization_mode,
                    "use_previous_solution": use_previous_solution,
                    "run_settings": RUN_SETTINGS,
                },
            )

            total_time = time.time() - start_time
            mass_balance = compute_global_mass_balance(
                ctx, fields, coeffs, gradients
            )
            energy_balance = compute_global_energy_balance(
                ctx, settings, fields, coeffs, fluxes, gradients
            )

            timing_csv_path = None
            if profiling_enabled and profiling["save_timing_csv"]:
                timing_csv_path = write_timing_csv(
                    run_dir / f"timing_history_{run_tag}.csv",
                    profile_records,
                )
            timing_summary = summarize_timing(
                profile_records, timing_csv_path
            )
            if profiling_enabled and profiling["print_summary"]:
                print_timing_summary(timing_summary)

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
                    "fluid": int(np.sum(is_fluid)),
                    "solid": int(np.sum(is_solid)),
                },
                "properties": {
                    "rho_min": float(np.min(material_fields["rho"][is_fluid])),
                    "rho_max": float(np.max(material_fields["rho"][is_fluid])),
                    "mu_min": float(np.min(material_fields["mu"][is_fluid])),
                    "mu_max": float(np.max(material_fields["mu"][is_fluid])),
                    "beta_min": float(
                        np.min(material_fields["beta"][is_fluid])
                    ),
                    "beta_max": float(
                        np.max(material_fields["beta"][is_fluid])
                    ),
                },
                "flags": {
                    "ENABLE_ENERGY": bool(RUN_SETTINGS["enable_energy"]),
                    "ENABLE_BUOYANCY": bool(
                        RUN_SETTINGS["enable_buoyancy"]
                    ),
                    "ENABLE_SOU_MOMENTUM": bool(
                        RUN_SETTINGS["enable_sou_momentum"]
                    ),
                    "ENABLE_SOU_ENERGY": bool(
                        RUN_SETTINGS["enable_sou_energy"]
                    ),
                    "SOU_BLEND_MOMENTUM": float(
                        RUN_SETTINGS["sou_blend_momentum"]
                    ),
                    "SOU_BLEND_ENERGY": float(
                        RUN_SETTINGS["sou_blend_energy"]
                    ),
                    "ENABLE_SOU_LIMITER": bool(
                        RUN_SETTINGS["enable_sou_limiter"]
                    ),
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
                    "mpi_ranks": mpi_size,
                    "threads_per_rank": int(
                        RUN_SETTINGS["threads_per_rank"]
                    ),
                    "direct_solver": str(
                        RUN_SETTINGS["direct_solver"]["solver_type"]
                    ),
                },
                "histories": histories,
                "temperature": {
                    "Tmin": float(np.min(fields["T"])),
                    "Tmax": float(np.max(fields["T"])),
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
                    "avg_time_per_iter": total_time
                    / max(len(histories["hist_it"]), 1),
                    "profiling_enabled": profiling_enabled,
                    "timing_summary": timing_summary,
                },
            }
            save_simulation_report(
                run_dir,
                f"simulation_report_{run_tag}.txt",
                report_data,
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