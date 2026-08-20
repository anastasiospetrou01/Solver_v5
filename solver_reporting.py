from __future__ import annotations

"""Residual plots, balances, timing reports and text simulation reports."""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from solver_utils import (
    area_e,
    area_n,
    area_s,
    area_w,
    boundary_mdot,
    heat_bc_T,
    heat_bc_q,
    heat_bc_type,
)
from solver_equations import (
    east_conductance,
    north_conductance,
    south_conductance,
    west_conductance,
)


def save_residual_history(
    run_dir,
    hist_it,
    hist_du,
    hist_dv,
    hist_dp,
    hist_mass,
    hist_dT,
    filename,
):
    figure = plt.figure(figsize=(8, 5))

    if np.any(np.asarray(hist_du) > 0.0):
        plt.semilogy(hist_it, hist_du, label="du_inf")
    if np.any(np.asarray(hist_dv) > 0.0):
        plt.semilogy(hist_it, hist_dv, label="dv_inf")
    if np.any(np.asarray(hist_dp) > 0.0):
        plt.semilogy(hist_it, hist_dp, label="dp_inf")
    if np.any(np.asarray(hist_mass) > 0.0):
        plt.semilogy(hist_it, hist_mass, label="mass residual")
    if np.any(np.asarray(hist_dT) > 0.0):
        plt.semilogy(hist_it, hist_dT, label="dT_inf")

    plt.xlabel("Iteration")
    plt.ylabel("Residual / update norm")
    plt.title("Residual history")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(os.path.join(run_dir, filename), dpi=300)
    plt.close(figure)


def compute_global_energy_balance(
    ctx,
    settings,
    fields,
    coeffs,
    fluxes,
    gradients,
) -> Dict[str, float]:
    del settings, fluxes
    temperature = fields["T"]
    cp = ctx["cp"]
    qdot = ctx["qdot"]
    nx = int(ctx["nx"])
    ny = int(ctx["ny"])
    reference_temperature = float(ctx["T_ref"])

    volumetric = float(np.sum(qdot * ctx["V"]))
    conductive_in = 0.0
    conductive_out = 0.0
    convective_in = 0.0
    convective_out = 0.0

    def add_conduction(heat_rate: float) -> None:
        nonlocal conductive_in, conductive_out
        if heat_rate >= 0.0:
            conductive_in += heat_rate
        else:
            conductive_out += -heat_rate

    def add_convection(side: str, i: int, j: int, mass_flow: float) -> None:
        nonlocal convective_in, convective_out
        inlet_temperature = heat_bc_T(ctx, side, reference_temperature)
        local_cp = cp[j, i]
        if side in ("west", "south"):
            if mass_flow >= 0.0:
                convective_in += mass_flow * local_cp * inlet_temperature
            else:
                convective_out += (-mass_flow) * local_cp * temperature[j, i]
        else:
            if mass_flow >= 0.0:
                convective_out += mass_flow * local_cp * temperature[j, i]
            else:
                convective_in += (-mass_flow) * local_cp * inlet_temperature

    for j in range(ny):
        west_type = heat_bc_type(ctx, "west")
        if west_type == "dirichlet":
            heat_rate = west_conductance(ctx, 0, j) * (
                heat_bc_T(ctx, "west") - temperature[j, 0]
            )
        elif west_type == "neumann":
            heat_rate = heat_bc_q(ctx, "west") * area_w(ctx, 0, j)
        else:
            heat_rate = 0.0
        add_conduction(heat_rate)
        add_convection(
            "west",
            0,
            j,
            boundary_mdot(
                ctx, "west", 0, j, fields, coeffs, gradients
            ),
        )

        east_i = nx - 1
        east_type = heat_bc_type(ctx, "east")
        if east_type == "dirichlet":
            heat_rate = east_conductance(ctx, east_i, j) * (
                heat_bc_T(ctx, "east") - temperature[j, east_i]
            )
        elif east_type == "neumann":
            heat_rate = heat_bc_q(ctx, "east") * area_e(ctx, east_i, j)
        else:
            heat_rate = 0.0
        add_conduction(heat_rate)
        add_convection(
            "east",
            east_i,
            j,
            boundary_mdot(
                ctx, "east", east_i, j, fields, coeffs, gradients
            ),
        )

    for i in range(nx):
        south_type = heat_bc_type(ctx, "south")
        if south_type == "dirichlet":
            heat_rate = south_conductance(ctx, i, 0) * (
                heat_bc_T(ctx, "south") - temperature[0, i]
            )
        elif south_type == "neumann":
            heat_rate = heat_bc_q(ctx, "south") * area_s(ctx, i, 0)
        else:
            heat_rate = 0.0
        add_conduction(heat_rate)
        add_convection(
            "south",
            i,
            0,
            boundary_mdot(
                ctx, "south", i, 0, fields, coeffs, gradients
            ),
        )

        north_j = ny - 1
        north_type = heat_bc_type(ctx, "north")
        if north_type == "dirichlet":
            heat_rate = north_conductance(ctx, i, north_j) * (
                heat_bc_T(ctx, "north") - temperature[north_j, i]
            )
        elif north_type == "neumann":
            heat_rate = heat_bc_q(ctx, "north") * area_n(ctx, i, north_j)
        else:
            heat_rate = 0.0
        add_conduction(heat_rate)
        add_convection(
            "north",
            i,
            north_j,
            boundary_mdot(
                ctx, "north", i, north_j, fields, coeffs, gradients
            ),
        )

    total_in = volumetric + conductive_in + convective_in
    total_out = conductive_out + convective_out
    net = total_in - total_out
    denominator = max(abs(total_in), abs(total_out), abs(volumetric), 1.0e-30)
    return {
        "Q_vol": volumetric,
        "Q_cond_in": conductive_in,
        "Q_cond_out": conductive_out,
        "Q_conv_in": convective_in,
        "Q_conv_out": convective_out,
        "Q_in": total_in,
        "Q_out": total_out,
        "Q_net": net,
        "rel_imbalance": abs(net) / denominator,
    }


def make_profile_record(
    iteration: int,
    outer_time: float,
    metrics_time: float,
    flow_timing: Dict[str, Any],
    energy_timing: Dict[str, Any],
) -> Dict[str, float]:
    """Flatten already MPI-MAX-reduced stage timings into one CSV record."""
    flow_timing = flow_timing or {}
    energy_timing = energy_timing or {}
    flow_backend = flow_timing.get("backend", {}) or {}
    energy_backend = energy_timing.get("backend", {}) or {}

    return {
        "iteration": int(iteration),
        "outer_total_s": float(outer_time),
        "metrics_s": float(metrics_time),

        "flow_total_s": float(flow_timing.get("flow_total", 0.0)),
        "flow_assembly_total_s": float(flow_timing.get("flow_assembly_total", 0.0)),
        "flow_momentum_pass_s": float(flow_timing.get("flow_momentum_pass", 0.0)),
        "momentum_pressure_gradient_s": float(flow_timing.get("momentum_pressure_gradient", 0.0)),
        "momentum_pre_flux_s": float(flow_timing.get("momentum_pre_flux", 0.0)),
        "momentum_sou_gradients_s": float(flow_timing.get("momentum_sou_gradients", 0.0)),
        "momentum_coefficients_s": float(flow_timing.get("momentum_coefficients", 0.0)),
        "flow_pattern_lookup_s": float(flow_timing.get("flow_pattern_lookup", 0.0)),
        "flow_state_vector_s": float(flow_timing.get("flow_state_vector", 0.0)),
        "flow_scaling_s": float(flow_timing.get("flow_scaling", 0.0)),
        "flow_linear_solve_s": float(flow_timing.get("flow_linear_solve", 0.0)),
        "flow_field_update_s": float(flow_timing.get("flow_field_update", 0.0)),
        "flow_coeff_update_s": float(flow_timing.get("flow_coeff_update", 0.0)),
        "flow_field_halo_s": float(flow_timing.get("flow_field_halo", 0.0)),
        "momentum_coefficient_halo_s": float(flow_timing.get("momentum_coefficient_halo", 0.0)),
        "flow_post_flux_s": float(flow_timing.get("flow_post_flux", 0.0)),
        "flow_coo_value_fill_s": float(flow_timing.get("flow_coo_value_fill", 0.0)),
        "flow_coo_matrix_update_s": float(flow_timing.get("flow_coo_matrix_update", 0.0)),
        "flow_coo_rhs_update_s": float(flow_timing.get("flow_coo_rhs_update", 0.0)),

        "flow_backend_total_s": float(flow_backend.get("total", 0.0)),
        "flow_backend_matrix_rhs_update_s": float(flow_backend.get("matrix_rhs_update", 0.0)),
        "flow_backend_factorization_s": float(flow_backend.get("factorization_setup", 0.0)),
        "flow_backend_solve_s": float(flow_backend.get("triangular_solve", 0.0)),
        "flow_backend_solution_gather_s": float(flow_backend.get("solution_gather", 0.0)),
        "flow_backend_residual_check_s": float(flow_backend.get("true_residual_check", 0.0)),

        "energy_total_s": float(energy_timing.get("energy_total", 0.0)),
        "energy_assembly_total_s": float(energy_timing.get("energy_assembly_total", 0.0)),
        "energy_gradient_s": float(energy_timing.get("energy_gradient", 0.0)),
        "energy_value_fill_s": float(energy_timing.get("energy_value_fill", 0.0)),
        "energy_linear_solve_s": float(energy_timing.get("energy_linear_solve", 0.0)),
        "energy_field_update_s": float(energy_timing.get("energy_field_update", 0.0)),
        "energy_field_halo_s": float(energy_timing.get("energy_field_halo", 0.0)),
        "energy_coo_matrix_update_s": float(energy_timing.get("energy_coo_matrix_update", 0.0)),
        "energy_coo_rhs_update_s": float(energy_timing.get("energy_coo_rhs_update", 0.0)),

        "energy_backend_total_s": float(energy_backend.get("total", 0.0)),
        "energy_backend_matrix_rhs_update_s": float(energy_backend.get("matrix_rhs_update", 0.0)),
        "energy_backend_factorization_s": float(energy_backend.get("factorization_setup", 0.0)),
        "energy_backend_solve_s": float(energy_backend.get("triangular_solve", 0.0)),
        "energy_backend_solution_gather_s": float(energy_backend.get("solution_gather", 0.0)),
        "energy_backend_residual_check_s": float(energy_backend.get("true_residual_check", 0.0)),
    }


def print_iteration_timing(record: Dict[str, float]) -> None:
    print(
        "    Timing(max-rank) | "
        f"outer={record['outer_total_s']:.3f}s | "
        f"mom={record['flow_momentum_pass_s']:.3f}s | "
        f"rowFill={record['flow_coo_value_fill_s']:.3f}s | "
        f"mat={record['flow_coo_matrix_update_s']:.3f}s | "
        f"MUMPS={record['flow_backend_factorization_s']:.3f}s | "
        f"solve={record['flow_backend_solve_s']:.3f}s | "
        f"gather={record['flow_backend_solution_gather_s']:.3f}s | "
        f"postFlux={record['flow_post_flux_s']:.3f}s | "
        f"energy={record['energy_total_s']:.3f}s"
    )


def write_timing_csv(path, records: List[Dict[str, float]]):
    if not records:
        return None
    output_path = Path(path)
    keys: List[str] = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def _average(records: Iterable[Dict[str, float]], key: str) -> float:
    values = [
        float(record[key])
        for record in records
        if key in record and record[key] is not None
    ]
    return sum(values) / len(values) if values else 0.0


def summarize_timing(records: List[Dict[str, float]], timing_csv_path=None) -> Dict[str, Any]:
    if not records:
        return {}
    keys = [key for key in records[0] if key != "iteration"]
    summary = {f"avg_{key}": _average(records, key) for key in keys}
    summary["timing_csv_path"] = str(timing_csv_path) if timing_csv_path is not None else None
    return summary


def print_timing_summary(summary: Dict[str, Any]) -> None:
    if not summary:
        return
    print("\n---------------- MPI-MAX TIMING SUMMARY ----------------")
    display = (
        ("outer iteration", "avg_outer_total_s"),
        ("flow total", "avg_flow_total_s"),
        ("momentum pass", "avg_flow_momentum_pass_s"),
        ("  pressure gradients", "avg_momentum_pressure_gradient_s"),
        ("  pre-solve fluxes", "avg_momentum_pre_flux_s"),
        ("  SOU gradients", "avg_momentum_sou_gradients_s"),
        ("  momentum coefficients", "avg_momentum_coefficients_s"),
        ("flow COO value fill", "avg_flow_coo_value_fill_s"),
        ("flow COO matrix update", "avg_flow_coo_matrix_update_s"),
        ("flow MUMPS factorization", "avg_flow_backend_factorization_s"),
        ("flow MUMPS solve", "avg_flow_backend_solve_s"),
        ("flow solution gather", "avg_flow_backend_solution_gather_s"),
        ("flow field halo", "avg_flow_field_halo_s"),
        ("momentum coeff halo", "avg_momentum_coefficient_halo_s"),
        ("flow post-flux", "avg_flow_post_flux_s"),
        ("metrics", "avg_metrics_s"),
        ("energy total", "avg_energy_total_s"),
        ("  energy gradients", "avg_energy_gradient_s"),
        ("  energy value fill", "avg_energy_value_fill_s"),
        ("  energy COO update", "avg_energy_coo_matrix_update_s"),
        ("  energy factorization", "avg_energy_backend_factorization_s"),
        ("  energy solve", "avg_energy_backend_solve_s"),
        ("  energy field halo", "avg_energy_field_halo_s"),
    )
    for label, key in display:
        if key in summary:
            print(f"    {label:<28s} = {summary[key]:.4f} s")
    if summary.get("timing_csv_path"):
        print(f"    timing CSV                  = {summary['timing_csv_path']}")


def save_simulation_report(run_dir, filename, report_data):
    history = report_data["histories"]
    setup = report_data["setup"]
    grid = report_data["grid"]
    cells = report_data["cells"]
    properties = report_data["properties"]
    flags = report_data["flags"]
    temperature = report_data["temperature"]
    sources = report_data["sources"]
    mass_balance = report_data["mass_balance"]
    energy_balance = report_data["energy_balance"]
    performance = report_data["performance"]

    report = f"""
================ CFD THERMO-FLOW SIMULATION REPORT ================

Run setup:
    case_name = {setup['case_name']}
    case_path = {setup['case_path']}
    initialization_mode = {setup['initialization_mode']}
    use_previous_solution = {setup['use_previous_solution']}
    restart_file = {setup['restart_file'] if setup['restart_file'] is not None else 'None'}
    MPI ranks = {flags.get('mpi_ranks', 1)}
    threads per rank = {flags.get('threads_per_rank', 1)}
    direct solver = {flags.get('direct_solver', 'mumps')}
    Numba kernels = {flags.get('use_numba', True)}
    sparse update = distributed fixed PETSc COO
    decomposition = {flags.get('decomposition', 'structured_y_slab_halo2')}
    profiling basis = MPI maximum rank time

Grid:
    nx = {grid['nx']}
    ny = {grid['ny']}
    dx = {grid['dx']:.4e}
    dy = {grid['dy']:.4e}

Cells:
    total = {cells['total']}
    fluid = {cells['fluid']}
    solid = {cells['solid']}

Fluid property ranges:
    rho  min/max = {properties['rho_min']:.6e} / {properties['rho_max']:.6e}
    mu   min/max = {properties['mu_min']:.6e} / {properties['mu_max']:.6e}
    beta min/max = {properties['beta_min']:.6e} / {properties['beta_max']:.6e}
    T_ref = {flags['T_ref']}
    gx = {flags['gx']}
    gy = {flags['gy']}

Flags:
    ENABLE_ENERGY = {flags['ENABLE_ENERGY']}
    ENABLE_BUOYANCY = {flags['ENABLE_BUOYANCY']}
    ENABLE_SOU_MOMENTUM = {flags['ENABLE_SOU_MOMENTUM']}
    ENABLE_SOU_ENERGY = {flags['ENABLE_SOU_ENERGY']}
    SOU_BLEND_MOMENTUM = {flags['SOU_BLEND_MOMENTUM']}
    SOU_BLEND_ENERGY = {flags['SOU_BLEND_ENERGY']}
    ENABLE_SOU_LIMITER = {flags['ENABLE_SOU_LIMITER']}
    alpha_u = {flags['alpha_u']}
    alpha_v = {flags['alpha_v']}
    alpha_p = {flags['alpha_p']}
    alpha_T = {flags['alpha_T']}
    max_iter = {flags['max_iter']}
    tol_mass = {flags['tol_mass']}
    tol_T = {flags['tol_T']}

Iterations:
    completed iterations = {len(history['hist_it'])}
    final mass residual  = {history['hist_mass'][-1]:.3e}
    final temperature residual = {history['hist_dT'][-1]:.3e}

Convergence:
    du_inf = {history['hist_du'][-1]:.3e}
    dv_inf = {history['hist_dv'][-1]:.3e}
    dp_inf = {history['hist_dp'][-1]:.3e}
    dT_inf = {history['hist_dT'][-1]:.3e}

Temperature range:
    Tmin = {temperature['Tmin']:.6f}
    Tmax = {temperature['Tmax']:.6f}

Source term ranges:
    qdot min/max = {sources['qdot_min']:.6e} / {sources['qdot_max']:.6e} W/m^3
    sx   min/max = {sources['sx_min']:.6e} / {sources['sx_max']:.6e} N/m^3
    sy   min/max = {sources['sy_min']:.6e} / {sources['sy_max']:.6e} N/m^3

---------------- GLOBAL MASS BALANCE ----------------
    west  in/out   = {mass_balance['west_in']:.6e} / {mass_balance['west_out']:.6e}
    east  in/out   = {mass_balance['east_in']:.6e} / {mass_balance['east_out']:.6e}
    south in/out   = {mass_balance['south_in']:.6e} / {mass_balance['south_out']:.6e}
    north in/out   = {mass_balance['north_in']:.6e} / {mass_balance['north_out']:.6e}
    total in       = {mass_balance['total_in']:.6e}
    total out      = {mass_balance['total_out']:.6e}
    net (out-in)   = {mass_balance['net']:.6e}
    relative error = {mass_balance['rel_imbalance']:.6e}

---------------- GLOBAL ENERGY BALANCE ----------------
    volumetric source Qvol = {energy_balance['Q_vol']:.6e} W
    conductive in/out      = {energy_balance['Q_cond_in']:.6e} / {energy_balance['Q_cond_out']:.6e} W
    convective in/out      = {energy_balance['Q_conv_in']:.6e} / {energy_balance['Q_conv_out']:.6e} W
    total Qin              = {energy_balance['Q_in']:.6e} W
    total Qout             = {energy_balance['Q_out']:.6e} W
    net (Qin-Qout)         = {energy_balance['Q_net']:.6e} W
    relative error         = {energy_balance['rel_imbalance']:.6e}

Performance:
    total runtime = {performance['total_time']:.2f} s
    avg time/iter = {performance['avg_time_per_iter']:.4f} s

===================================================================
"""
    output_path = Path(run_dir) / filename
    output_path.write_text(report, encoding="utf-8")
    print(report)