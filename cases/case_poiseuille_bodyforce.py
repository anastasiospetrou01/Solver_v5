import numpy as np
from pathlib import Path

from case_io import save_case_bundle

# ============================================================
# DOMAIN
# ============================================================
nx = 400
ny = 100
Lx = 4.0
Ly = 1.0

dx = Lx / nx
dy = Ly / ny

# ============================================================
# REGION IDS
# ============================================================
REGION_FLUID = 0

region_defs = {
    REGION_FLUID: {
        "name": "glycerin",
        "material": "glycerin",
        "phase": "fluid",
    },
}

# ============================================================
# REGION MAP
# ============================================================
region = np.full((ny, nx), REGION_FLUID, dtype=int)

# ============================================================
# CASE NAME
# ============================================================
case_name = "poiseuille_4x1"

# ============================================================
# BOUNDARIES
# ------------------------------------------------------------
# West/east: outlet so streamwise gradients are weakly released
# South/north: no-slip walls
# Heat block is dummy only, since energy should be disabled
# ============================================================
boundaries = {
    "heat": {
        "west":  {"type": "outlet", "T": 25.0},
        "east":  {"type": "outlet", "T": 25.0},
        "south": {"type": "adiabatic", "T": 25.0},
        "north": {"type": "adiabatic", "T": 25.0},
    },
    "flow": {
        "west":  {"type": "Inlet", "u": 0.02, "v": 0.0},
        "east":  {"type": "outlet", "u": 0.0, "v": 0.0},
        "south": {"type": "wall",   "u": 0.0, "v": 0.0},
        "north": {"type": "wall",   "u": 0.0, "v": 0.0},
    },
}

# ============================================================
# INITIAL CONDITIONS
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
# Body-force-driven channel flow:
# mu * d2u/dy2 + Sx = 0
#
# With glycerin mu = 1.0 Pa.s in your materials.py,
# Ly = H = 1.0 m, and Sx = 0.04 N/m^3:
#
# u_max  = Sx*H^2/(8*mu)  = 0.005 m/s
# u_mean = Sx*H^2/(12*mu) = 0.003333... m/s
# ============================================================
Sx = 0.0 # N/m^3

sources = {
    "energy": {
        "default": 0.0,
        "by_region": {
            "glycerin": 0.0,
        },
    },
    "momentum_x": {
        "default": 0.0,
        "by_region": {
            "glycerin": Sx,
        },
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

print(f"Case exported successfully.")
print(f"Case folder : {case_dir}")
print(f"Case npz    : {case_npz_path}")
print(f"Case info   : {case_info_path}")

# ============================================================
# ANALYTICAL REFERENCE
# ============================================================
H = Ly
mu = 1.0  # glycerin from materials.py
u_max = Sx * H**2 / (8.0 * mu)
u_mean = Sx * H**2 / (12.0 * mu)

print("\nAnalytical reference:")
print(f"Sx     = {Sx:.6f} N/m^3")
print(f"mu     = {mu:.6f} Pa.s")
print(f"H      = {H:.6f} m")
print(f"u_max  = {u_max:.6e} m/s")
print(f"u_mean = {u_mean:.6e} m/s")
print("Profile: u(y) = (Sx/(2*mu))*y*(H-y)")