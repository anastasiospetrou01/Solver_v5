from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flow_assembly import assemble_flow_correction_system
from linear_backend import create_linear_solver
from solver_equations import solve_pressure_velocity


def make_cavity(n=6):
    nx = ny = int(n)
    dx = dy = 1.0 / n
    fluid = np.ones((ny, nx), dtype=bool)
    solid = ~fluid
    cells = [(i, j) for j in range(ny) for i in range(nx)]
    cell_to_fid = np.arange(nx * ny, dtype=int).reshape(ny, nx)

    ctx = {
        "nx": nx,
        "ny": ny,
        "Lx": 1.0,
        "Ly": 1.0,
        "dx": dx,
        "dy": dy,
        "V": dx * dy,
        "is_fluid": fluid,
        "is_solid": solid,
        "fluid_cells": cells,
        "cell_to_fid": cell_to_fid,
        "Nf": nx * ny,
        "BC_flow": {
            "west": {"type": "wall", "u": 0.0, "v": 0.0, "p": 0.0},
            "east": {"type": "wall", "u": 0.0, "v": 0.0, "p": 0.0},
            "south": {"type": "wall", "u": 0.0, "v": 0.0, "p": 0.0},
            "north": {"type": "wall", "u": 1.0, "v": 0.0, "p": 0.0},
        },
        "BC_heat": {
            side: {"type": "adiabatic", "T": 0.0, "q": 0.0}
            for side in ("west", "east", "south", "north")
        },
        "rho": np.ones((ny, nx)),
        "mu": np.full((ny, nx), 0.01),
        "cp": np.ones((ny, nx)),
        "k": np.ones((ny, nx)),
        "beta": np.zeros((ny, nx)),
        "qdot": np.zeros((ny, nx)),
        "sx": np.zeros((ny, nx)),
        "sy": np.zeros((ny, nx)),
        "T_ref": 0.0,
        "gx": 0.0,
        "gy": 0.0,
        "use_pressure_reference": True,
        "p_ref_value": 0.0,
        "p_ref_i": 0,
        "p_ref_j": 0,
        "enable_sou_limiter": True,
    }
    fields = {
        "u": np.zeros((ny, nx)),
        "v": np.zeros((ny, nx)),
        "p": np.zeros((ny, nx)),
        "T": np.zeros((ny, nx)),
    }
    coeffs = {
        "aPu": np.full((ny, nx), 0.04),
        "aPv": np.full((ny, nx), 0.04),
    }
    settings = {
        "physics": {"flow": True, "energy": False, "buoyancy": False},
        "relaxation": {"u": 0.7, "v": 0.7, "p": 0.3, "T": 1.0},
        "schemes": {
            "momentum": "upwind",
            "energy": "upwind",
            "momentum_blend": 0.0,
            "energy_blend": 0.0,
            "limiter": "none",
        },
        "pressure_reference": {
            "mode": "pin",
            "enabled": True,
            "value": 0.0,
            "i": 0,
            "j": 0,
        },
        "linear_solver": {
            "backend": "scipy",
            "flow_coupled": {
                "scaling": {"enabled": True},
                "rtol": 1.0e-8,
                "acceptable_true_residual_factor": 10.0,
                "minimum_acceptable_true_residual": 1.0e-10,
                "check_true_residual": True,
            },
        },
        "profiling": {"enabled": True},
    }
    return ctx, fields, coeffs, settings


class FlowV5Tests(unittest.TestCase):
    def test_correction_form_matches_absolute_solution(self):
        ctx, fields, coeffs, settings = make_cavity(4)
        system = assemble_flow_correction_system(
            ctx,
            settings,
            fields,
            coeffs,
            supports_pressure_nullspace=False,
        )
        absolute = spsolve(system.matrix, system.absolute_rhs)
        correction = spsolve(system.matrix, system.rhs)
        np.testing.assert_allclose(
            absolute,
            system.old_state + correction,
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    def test_direct_smoke_iterations_are_finite(self):
        ctx, fields, coeffs, settings = make_cavity(6)
        linear_solver = create_linear_solver(settings["linear_solver"])
        for _ in range(3):
            fields, coeffs, fluxes = solve_pressure_velocity(
                ctx,
                settings,
                fields,
                coeffs,
                linear_solver=linear_solver,
            )
        self.assertTrue(np.all(np.isfinite(fields["u"])))
        self.assertTrue(np.all(np.isfinite(fields["v"])))
        self.assertTrue(np.all(np.isfinite(fields["p"])))
        self.assertLess(fluxes["linear_true_relative_residual"], 1.0e-10)


if __name__ == "__main__":
    unittest.main()
