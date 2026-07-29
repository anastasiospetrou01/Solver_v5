from __future__ import annotations

"""Material database and mapping of materials/sources onto the case grid."""

from typing import Any, Dict

import numpy as np


materials = {
    "air": {
        "rho": 1.225,
        "cp": 1005.0,
        "k": 0.024,
        "mu": 1.8e-5,
        "beta": 1.0 / 300.0,
        "phase": "fluid",
    },
    "low_Re_fluid": {
        "rho": 1.0,
        "cp": 1005.0,
        "k": 0.024,
        "mu": 0.01,
        "beta": 4.0e-4,
        "phase": "fluid",
    },
    "glycerin": {
        "rho": 1260.0,
        "cp": 2400.0,
        "k": 0.28,
        "mu": 1.0,
        "beta": 4.0e-4,
        "phase": "fluid",
    },
    "water": {
        "rho": 970.0,
        "cp": 4180.0,
        "k": 0.650,
        "mu": 0.0010005,
        "beta": 2.07e-4,
        "phase": "fluid",
    },
    "glass": {
        "rho": 2500.0,
        "cp": 800.0,
        "k": 1.0,
        "mu": 0.0,
        "phase": "solid",
    },
    "soil": {
        "rho": 1600.0,
        "cp": 800.0,
        "k": 0.30,
        "mu": 0.0,
        "phase": "solid",
    },
}


def build_material_fields(
    geom: Dict[str, Any],
    material_database: Dict[str, Dict[str, Any]] = materials,
) -> Dict[str, np.ndarray]:
    region = geom["region"]
    region_defs = geom["region_defs"]
    ny, nx = region.shape

    fields = {
        "rho": np.zeros((ny, nx), dtype=float),
        "cp": np.zeros((ny, nx), dtype=float),
        "k": np.zeros((ny, nx), dtype=float),
        "mu": np.zeros((ny, nx), dtype=float),
        "beta": np.zeros((ny, nx), dtype=float),
    }

    for region_id, info in region_defs.items():
        mask = region == region_id
        material_name = info["material"]
        if material_name not in material_database:
            raise KeyError(
                f"Material {material_name!r} used by region {info.get('name')!r} "
                "is not defined in materials.py."
            )
        material = material_database[material_name]
        for property_name in fields:
            fields[property_name][mask] = float(
                material.get(property_name, 0.0)
            )
    return fields


def _region_name_to_id(region_defs: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    return {
        str(info["name"]): int(region_id)
        for region_id, info in region_defs.items()
    }


def _build_one_source_field(
    region: np.ndarray,
    region_defs: Dict[int, Dict[str, Any]],
    specification: Dict[str, Any],
) -> np.ndarray:
    field = np.full(
        region.shape,
        float(specification.get("default", 0.0)),
        dtype=float,
    )
    name_to_id = _region_name_to_id(region_defs)
    for region_name, value in specification.get("by_region", {}).items():
        if region_name not in name_to_id:
            raise ValueError(
                f"Region {region_name!r} is not defined in region_defs."
            )
        field[region == name_to_id[region_name]] = float(value)
    return field


def _apply_case_specific_sources(
    geom: Dict[str, Any],
    source_fields: Dict[str, np.ndarray],
) -> None:
    """Apply legacy analytical forcing that is not stored in the NPZ case."""
    if geom.get("case_name") != "kolmogorov_forcing":
        return

    nx = int(geom["nx"])
    ny = int(geom["ny"])
    lx = float(geom["Lx"])
    ly = float(geom["Ly"])
    dx = lx / nx
    dy = ly / ny
    x_centers = (np.arange(nx) + 0.5) * dx
    y_centers = (np.arange(ny) + 0.5) * dy
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    forcing = 1.0  # N/m^3; retained from the validated V5 implementation.
    source_fields["momentum_x"][:, :] = (
        forcing * np.sin(2.0 * np.pi * y_grid / ly)
        + 0.35 * forcing * np.sin(4.0 * np.pi * y_grid / ly)
        + 0.20 * forcing * np.sin(6.0 * np.pi * y_grid / ly)
        + 0.10
        * forcing
        * np.sin(2.0 * np.pi * x_grid / lx)
        * np.sin(2.0 * np.pi * y_grid / ly)
    )
    source_fields["momentum_y"][:, :] = (
        0.05
        * forcing
        * np.sin(2.0 * np.pi * x_grid / lx)
        * np.sin(4.0 * np.pi * y_grid / ly)
    )


def build_source_fields(geom: Dict[str, Any]) -> Dict[str, np.ndarray]:
    region = geom["region"]
    region_defs = geom["region_defs"]
    source_specification = geom.get("sources", {})

    source_fields = {
        "energy": _build_one_source_field(
            region,
            region_defs,
            source_specification.get("energy", {}),
        ),
        "momentum_x": _build_one_source_field(
            region,
            region_defs,
            source_specification.get("momentum_x", {}),
        ),
        "momentum_y": _build_one_source_field(
            region,
            region_defs,
            source_specification.get("momentum_y", {}),
        ),
    }
    _apply_case_specific_sources(geom, source_fields)
    return source_fields