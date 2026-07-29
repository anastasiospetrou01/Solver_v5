import numpy as np


def _region_name_to_id(region_defs):
    return {
        info["name"]: region_id
        for region_id, info in region_defs.items()
    }


def build_initial_temperature(geom, prefer_solved=True):
    if prefer_solved and "solved_temperature" in geom:
        return np.array(geom["solved_temperature"], dtype=float, copy=True)

    ny = geom["ny"]
    nx = geom["nx"]
    region = geom["region"]
    region_defs = geom["region_defs"]

    spec = geom["initial_temperature"]
    default_T = spec["default"]

    T0 = np.full((ny, nx), default_T, dtype=float)
    region_name_to_id = _region_name_to_id(region_defs)

    for region_name, T_value in spec["by_region"].items():
        if region_name not in region_name_to_id:
            raise ValueError(f"Region '{region_name}' not defined in region_defs")
        region_id = region_name_to_id[region_name]
        T0[region == region_id] = T_value

    return T0



def build_initial_flow(geom, prefer_solved=True):
    if prefer_solved and "solved_flow" in geom:
        return {
            "u": np.array(geom["solved_flow"]["u"], dtype=float, copy=True),
            "v": np.array(geom["solved_flow"]["v"], dtype=float, copy=True),
            "p": np.array(geom["solved_flow"]["p"], dtype=float, copy=True),
        }

    ny = geom["ny"]
    nx = geom["nx"]
    region = geom["region"]
    region_defs = geom["region_defs"]

    spec = geom["initial_flow"]

    u0 = np.full((ny, nx), spec["default"]["u"], dtype=float)
    v0 = np.full((ny, nx), spec["default"]["v"], dtype=float)
    p0 = np.full((ny, nx), spec["default"]["p"], dtype=float)

    region_name_to_id = _region_name_to_id(region_defs)

    for region_name, values in spec.get("by_region", {}).items():
        if region_name not in region_name_to_id:
            raise ValueError(f"Region '{region_name}' not defined in region_defs")
        region_id = region_name_to_id[region_name]
        mask = region == region_id

        u0[mask] = values.get("u", u0[mask])
        v0[mask] = values.get("v", v0[mask])
        p0[mask] = values.get("p", p0[mask])

    return {
        "u": u0,
        "v": v0,
        "p": p0,
    }
