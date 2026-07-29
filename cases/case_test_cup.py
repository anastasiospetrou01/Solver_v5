import numpy as np
from pathlib import Path

from geometry_primitives import fill_rectangle, fill_circle
from case_io import save_case

# ============================================================
# DOMAIN
# ============================================================
nx = 200
ny = 100
Lx = 2.0
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
obs_w = max(4, int(round(0.20 / dx)))
obs_h = max(3, int(round(0.20 / dy)))
wall_w = max(3,int(round(0.03/dx)))
obs_i0 = nx // 2 - obs_w // 2
obs_i1 = obs_i0 + obs_w
obs_j0 = ny // 2 - obs_h // 2
obs_j1 = obs_j0 + obs_h

region[obs_j0:obs_j1, obs_i0:obs_i1] = REGION_OBSTACLE
region[obs_j0+wall_w:obs_j1, obs_i0+wall_w:obs_i1-wall_w] = REGION_AIR

# ============================================================
# BOUNDARIES
# ============================================================
boundaries = {
    "heat": {
        "west":  {"type": "dirichlet", "T": 25.0},
        "east":  {"type": "dirichlet", "T": 25.0},
        "south": {"type": "dirichlet", "T": 25.0},
        "north": {"type": "dirichlet", "T": 25.0},
    },
    "flow": {
        "west":  {"type": "inlet",    "u": 0.01, "v": 0.0},
        "east":  {"type": "outlet",   "u": 0.0,  "v": 0.0},
        "south": {"type": "symmetry", "u": 0.0,  "v": 0.0},
        "north": {"type": "symmetry", "u": 0.0,  "v": 0.0},
    },
}

# ============================================================
# INITIAL CONDITIONS
# ============================================================
initial_temperature = {
    "default": 25.0,
    "by_region": {
        "glycerin": 40.0,
        "obstacle": 25.0,
    }
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
    }
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
    "case_name": "glycerin_and_cup",
    "boundaries": boundaries,
    "initial_temperature": initial_temperature,
    "initial_flow": initial_flow,
}

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
case_folder = project_root / "case_files"
case_folder.mkdir(exist_ok=True)

save_path = case_folder / "glycerin_and_cup.npz"
save_case(save_path, geom)

print(f"Case exported successfully to: {save_path}")