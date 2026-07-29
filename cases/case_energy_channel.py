import numpy as np
from pathlib import Path

from case_io import save_case

# ============================================================
# DOMAIN
# ============================================================
nx = 100
ny = 60
Lx = 2.0
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

# ============================================================
# REGION MAP
# ============================================================
region = np.full((ny, nx), REGION_FLUID, dtype=int)

# ============================================================
# BOUNDARIES
# ============================================================
boundaries = {
    "flow": {
        "west":  {"type": "inlet",    "u": 0.01, "v": 0.0},
        "east":  {"type": "outlet",   "u": 0.0,  "v": 0.0},
        "south": {"type": "wall", "u": 0.0,  "v": 0.0},
        "north": {"type": "symmetry", "u": 0.0,  "v": 0.0},
    },
    "heat": {
        "west":  {"type": "dirichlet", "T": 40.0},
        "east":  {"type": "outlet"},
        "south": {"type": "dirichlet", "T": 25.0},
        "north": {"type": "adiabatic"},
    },
}

# ============================================================
# INITIAL CONDITIONS
# ============================================================
initial_temperature = {
    "default": 25.0,
    "by_region": {
        "low_Re_fluid": 25.0,
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
# EXPORT CASE
# ============================================================
geom = {
    "nx": nx,
    "ny": ny,
    "Lx": Lx,
    "Ly": Ly,
    "region": region,
    "region_defs": region_defs,
    "case_name": "energy_channel",
    "boundaries": boundaries,
    "initial_temperature": initial_temperature,
    "initial_flow": initial_flow,
}

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
case_folder = project_root / "case_files"
case_folder.mkdir(exist_ok=True)

save_path = case_folder / "energy_channel.npz"
save_case(save_path, geom)

print(f"Case exported successfully to: {save_path}")