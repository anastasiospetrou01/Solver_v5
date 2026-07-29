import numpy as np
from pathlib import Path

from case_io import save_case_bundle

# ============================================================
# DOMAIN
# ============================================================
nx = 200
ny = 50
Lx = 4.0
Ly = 1.0

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
case_name = "couette_4x1"

# ============================================================
# PARAMETERS
# ============================================================
Uw = 0.01   # top wall velocity [m/s]
Sx = 0.0   # body force in x [N/m^3]
mu = 1.0    # glycerin viscosity from materials.py
H = Ly

# ============================================================
# BOUNDARIES
# ============================================================
boundaries = {
    "heat": {
        "west":  {"type": "outlet", "T": 25.0},
        "east":  {"type": "outlet", "T": 25.0},
        "south": {"type": "adiabatic", "T": 25.0},
        "north": {"type": "adiabatic", "T": 25.0},
    },
    "flow": {
        "west":  {"type": "outlet", "u": 0.0, "v": 0.0},
        "east":  {"type": "outlet", "u": 0.0, "v": 0.0},
        "south": {"type": "wall",   "u": 0.0, "v": 0.0},
        "north": {"type": "wall",   "u": Uw,  "v": 0.0},
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
# ============================================================
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

# ============================================================
# ANALYTICAL REFERENCE
# ============================================================
#u_mean = Uw / 2.0 + Sx * H**2 / (12.0 * mu)
#y_max = H / 2.0 + (mu * Uw) / (H * Sx)
#u_max = (Uw / H) * y_max + (Sx / (2.0 * mu)) * y_max * (H - y_max)

print("Case exported successfully.")
print(f"Case folder : {case_dir}")
print(f"Case npz    : {case_npz_path}")
print(f"Case info   : {case_info_path}")

print("\nAnalytical solution:")
print("u(y) = (Uw/H)*y + (Sx/(2*mu))*y*(H-y)")
print(f"Uw      = {Uw:.6e} m/s")
print(f"Sx      = {Sx:.6e} N/m^3")
print(f"mu      = {mu:.6e} Pa.s")
print(f"H       = {H:.6e} m")
#print(f"u_mean  = {u_mean:.6e} m/s")
#print(f"y_max   = {y_max:.6e} m")
#print(f"u_max   = {u_max:.6e} m/s")