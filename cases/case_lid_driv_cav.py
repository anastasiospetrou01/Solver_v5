import numpy as np
from pathlib import Path

from geometry_primitives import fill_rectangle
from case_io import save_case_bundle

# ============================================================
# DOMAIN
# ============================================================
nx = 256
ny = 256
Lx = 1.0
Ly = 1.0
dx = Lx / nx
dy = Ly / ny

# ============================================================
# REGION IDS
# ============================================================
REGION_FLUID = 0


region_defs = {
    REGION_FLUID: {
        "name": "fluid",
        "material": "low_Re_fluid",
        "phase": "fluid",
    },
    
}

# ============================================================
# REGION MAP
# ============================================================
region = np.full((ny, nx), REGION_FLUID, dtype=int)
# ============================================================
# GEOMETRY DESIGN
# ============================================================


# ============================================================
# CASE NAME
# ============================================================
case_name = "lidcav_20"

# ============================================================
# BOUNDARIES
# ------------------------------------------------------------
# Left wall hot, right wall cold
# Top/bottom adiabatic
# All walls no-slip
# ============================================================
boundaries = {
    "heat": {
        "west": {"type": "dirichlet", "T": 25.0},
        "east": {"type": "dirichlet", "T": 25.0},
        "south": {"type": "dirichlet", "T": 25.0},
        "north": {"type": "open", "T": 25.0},
        
    },
    "flow": {
        "west": {"type": "wall"},
        "east": {"type": "wall"},
        "south": {"type": "wall", "u": 0.0, "v": 0.0},
        "north": {"type": "inlet", "u": 1.0, "v": 0.0},
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
        "fluid": 25.0,
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
            "fluid": 0.0,
            
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