import numpy as np
from pathlib import Path

from case_io import save_case_bundle

# ============================================================
# DOMAIN
# ============================================================
nx = 200
ny = 200
Lx = 1.0
Ly = 1.0

# ============================================================
# REGION IDS
# ============================================================
REGION_FLUID = 0

region_defs = {
    REGION_FLUID: {
        "name": "low_Re_fluid",
        "material": "low_Re_fluid",
        "phase": "fluid",
    },
}

region = np.full((ny, nx), REGION_FLUID, dtype=int)

case_name = "kolmogorov_forcing"

# ============================================================
# BOUNDARIES
# ------------------------------------------------------------
# Not exact periodic Kolmogorov flow yet.
# This is a source-driven rectangular test compatible with the current solver.
# ============================================================
boundaries = {
    "heat": {
        "west": {"type": "open", "T": 25.0},
        "east": {"type": "open", "T": 25.0},
        "south": {"type": "adiabatic"},
        "north": {"type": "adiabatic"},
    },
    "flow": {
        "west": {"type": "open", "p": 0.0},
        "east": {"type": "open", "p": 0.0},
        "south": {"type": "wall", "u": 0.0, "v": 0.0},
        "north": {"type": "wall", "u": 0.0, "v": 0.0},
    },
}

initial_temperature = {
    "default": 25.0,
    "by_region": {
        "low_Re_fluid": 25.0,
    },
}

initial_flow = {
    "default": {
        "u": 0.0,
        "v": 0.0,
        "p": 0.0,
    },
    "by_region": {},
}

# Keep source values zero here.
# The sinusoidal spatial source is assigned in steady_lam_v4.py.
sources = {
    "energy": {
        "default": 0.0,
        "by_region": {
            "low_Re_fluid": 0.0,
        },
    },
    "momentum_x": {
        "default": 0.0,
        "by_region": {},
    },
    "momentum_y": {
        "default": 0.0,
        "by_region": {},
    },
}

geom = {
    "nx": nx,
    "ny": ny,
    "Lx": Lx,
    "Ly": Ly,
    "region": region,
    "region_defs": region_defs,
    "case_name": case_name,
    "boundaries": boundaries,
    "initial_temperature": initial_temperature,
    "initial_flow": initial_flow,
    "sources": sources,
}

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
cases_root = project_root / "case_files"

case_dir, case_npz_path, case_info_path = save_case_bundle(
    cases_root=cases_root,
    geom=geom,
)

print("Case exported successfully.")
print(f"Case folder : {case_dir}")
print(f"Case npz    : {case_npz_path}")
print(f"Case info   : {case_info_path}")