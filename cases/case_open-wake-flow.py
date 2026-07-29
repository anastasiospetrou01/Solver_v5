import numpy as np
from pathlib import Path

from geometry_primitives import fill_rectangle
from case_io import save_case_bundle

# ============================================================
# DOMAIN
# ============================================================
nx = 150
ny = 50
Lx = 3.0
Ly = 1.0
dx = Lx / nx
dy = Ly / ny

# ============================================================
# REGION IDS
# ============================================================
REGION_FLUID = 0
REGION_OBSTACLE = 1

region_defs = {
    REGION_FLUID: {
        "name": "glycerin",
        "material": "glycerin",
        "phase": "fluid",
    },
    REGION_OBSTACLE: {
        "name": "obstacle",
        "material": "glass",
        "phase": "solid",
    },
}

# ============================================================
# REGION MAP
# ============================================================
region = np.full((ny, nx), REGION_FLUID, dtype=int)
# ============================================================
# GEOMETRY DESIGN
# ============================================================
obs_w = max(4, int(round(0.30 / dx)))
obs_h = max(3, int(round(0.30 / dy)))
obs_i0 = nx // 4 - obs_w // 2
obs_i1 = obs_i0 + obs_w
obs_j0 = ny // 2 - obs_h // 2
obs_j1 = obs_j0 + obs_h

fill_rectangle(region, obs_i0, obs_i1, obs_j0, obs_j1, REGION_OBSTACLE)

# ============================================================
# CASE NAME
# ============================================================
case_name = "wake_test"

# ============================================================
# BOUNDARIES
# ------------------------------------------------------------
# Left wall hot, right wall cold
# Top/bottom adiabatic
# All walls no-slip
# ============================================================
boundaries = {
    "heat": {
        "west": {"type": "open", "T": 25.0},
        "east": {"type": "open", "T": 25.0},
        "south": {"type": "dirichlet", "T": 25.0},
        "north": {"type": "open", "T": 25.0},
        
    },
    "flow": {
        "west": {"type": "inlet", "u": 0.05},
        "east": {"type": "open", "p": 0.0},
        "south": {"type": "symmetry", "u": 0.0, "v": 0.0},
        "north": {"type": "symmetry"},
    },
}

# ============================================================
# INITIAL CONDITIONS
# ------------------------------------------------------------
# Start from uniform reference temperature
# ============================================================
initial_temperature = {
    "default": 25.0,
    "by_region": {
        "glycerin": 25.0,
    }
}

initial_flow = {
    "default": {
        "u": 0.0,
        "v": 0.0,
        "p": 0.0,
    },
    "by_region": {}
}

# ============================================================
# SOURCES
# ------------------------------------------------------------
# No explicit momentum source.
# Flow is driven only by buoyancy in the solver.
# ============================================================
sources = {
     "energy": {
        "default": 0.0,
        "by_region": {
            "glycerin": 0.0,
            "obstacle": 0.0,
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

# ============================================================
# BUILD CASE
# ============================================================
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

# ============================================================
# EXPORT
# ============================================================
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

#"alpha_T": 1.0,
#"alpha_u": 0.4,
#"alpha_v": 0.4,
#"alpha_p": 0.2,