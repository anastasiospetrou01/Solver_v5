import numpy as np
from pathlib import Path

from geometry_primitives import fill_hollow_rectangle
from case_io import save_case

# ============================================================
# DOMAIN
# ============================================================

nx = 200
ny = 200

Lx = 1.0
Ly = 1.0

REGION_AIR = 0
REGION_CUP_WALL = 1
REGION_WATER = 2

region_defs = {
    REGION_AIR: {"name": "air", "material": "air", "phase": "fluid"},
    REGION_CUP_WALL: {"name": "cup_wall", "material": "glass", "phase": "solid"},
    REGION_WATER: {"name": "water", "material": "water", "phase": "fluid"},
}

region = np.full((ny, nx), REGION_AIR, dtype=int)

# ============================================================
# GEOMETRY DESIGN
# ============================================================

cup_width = 40
cup_left = 80
cup_right = cup_left + cup_width
cup_bottom = 15
cup_top = 65
wall_thick = 3

fill_hollow_rectangle(
    region,
    cup_left,
    cup_right,
    cup_bottom,
    cup_top,
    wall_thick,
    REGION_CUP_WALL,
    REGION_WATER
)

# ============================================================
# INITIAL TEMPERATURE SPECIFICATION
# ============================================================

initial_temperature = {
    "default": 25.0,
    "by_region": {
        "air": 25.0,
        "cup_wall": 40.0,
        "water": 90.0,
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
    "case_name": "cup_case11",
    "initial_temperature": initial_temperature
}

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
case_folder = project_root / "case_files"

case_folder.mkdir(exist_ok=True)

save_path = case_folder / "cup_case12.npz"

save_case(save_path, geom)

print(f"Case exported successfully to: {save_path}")