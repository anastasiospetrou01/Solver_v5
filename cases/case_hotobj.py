import numpy as np
from pathlib import Path

from geometry_primitives import fill_rectangle, fill_circle
from case_io import save_case_bundle

# ============================================================
# DOMAIN
# ============================================================
nx = 100
ny = 100
Lx = 1.0
Ly = 1.0

dx = Lx / nx
dy = Ly / ny

# ============================================================
# REGION IDS
# ============================================================
REGION_AIR = 0
REGION_OBSTACLE = 1

region_defs = {
    REGION_AIR: {
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
region = np.full((ny, nx), REGION_AIR, dtype=int)

# ============================================================
# GEOMETRY DESIGN
# ============================================================
obs_w = max(4, int(round(0.10 / dx)))
obs_h = max(3, int(round(0.10 / dy)))
obs_i0 = nx // 2 - obs_w // 2
obs_i1 = obs_i0 + obs_w
obs_j0 = ny // 3 - obs_h // 3
obs_j1 = obs_j0 + obs_h

fill_rectangle(region, obs_i0, obs_i1, obs_j0, obs_j1, REGION_OBSTACLE)

# ============================================================
# BOUNDARIES
# ============================================================
boundaries = {
    "heat": {
        "west": {"type": "symmetry", "T": 25.0},
        "east": {"type": "symmetry", "T": 25.0},
        "south": {"type": "adiabatic", "T": 25.0},
        "north": {"type": "symmetry", "T": 0.0},
    },
    "flow": {
        "west": {"type": "symmetry", "u": 0.0, "v": 0.0},
        "east": {"type": "symmetry", "u": 0.0, "v": 0.0},
        "south": {"type": "wall", "u": 0.0, "v": 0.0},
        "north": {"type": "outlet", "u": 0.0, "v": 0.0},
    },
}

# ============================================================
# INITIAL CONDITIONS
# ============================================================
initial_temperature = {
    "default": 25.0,
    "by_region": {
        "glycerin": 25.0,
        "obstacle": 25.0,
    },
}

initial_flow = {
    "default": {
        "u": 0.0,
        "v": 0.0,
        "p": 0.0,
    },
    "by_region": {
        "obstacle": {
            "u": 0.0,
            "v": 0.0,
            "p": 0.0,
        }
    },
}

# ============================================================
# SOURCES
# ============================================================
sources = {
    "energy": {
        "default": 0.0,
        "by_region": {
            "glycerin": 0.0,
            "obstacle": 300,
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
# EXPORT CASE
# ============================================================
geom = {
    "nx": nx,
    "ny": ny,
    "Lx": Lx,
    "Ly": Ly,
    "region": region,
    "region_defs": region_defs,
    "case_name": "hotobj4x1",
    "boundaries": boundaries,
    "initial_temperature": initial_temperature,
    "initial_flow": initial_flow,
    "sources": sources,
}

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
cases_root = project_root / "case_files"
cases_root.mkdir(exist_ok=True)

saved = save_case_bundle(cases_root, geom)

print(f"Case exported successfully to folder: {saved['case_dir']}")
print(f"Case npz: {saved['case_npz_path']}")
print(f"Case info: {saved['case_info_path']}")
