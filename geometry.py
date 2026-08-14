from __future__ import annotations

"""Geometry masks, compact fluid indexing, and static solver topology."""

from typing import Any, Dict

import numpy as np


# Direction order used by all optimized kernels.
EAST = 0
WEST = 1
NORTH = 2
SOUTH = 3

# Static face classifications, equivalent to the legacy string helpers.
FACE_FLUID_FLUID = 0
FACE_FLUID_SOLID = 1
FACE_BOUNDARY = 2

# Integer boundary codes remove strings/dictionaries from Numba kernels.
FLOW_BC_WALL = 0
FLOW_BC_INLET = 1
FLOW_BC_OUTLET = 2
FLOW_BC_OPEN = 3
FLOW_BC_SYMMETRY = 4
FLOW_BC_OTHER = 5

HEAT_BC_DIRICHLET = 0
HEAT_BC_NEUMANN = 1
HEAT_BC_ADIABATIC = 2
HEAT_BC_SYMMETRY = 3
HEAT_BC_OUTLET = 4
HEAT_BC_OPEN = 5
HEAT_BC_OTHER = 6


_FLOW_BC_CODES = {
    "wall": FLOW_BC_WALL,
    "inlet": FLOW_BC_INLET,
    "outlet": FLOW_BC_OUTLET,
    "open": FLOW_BC_OPEN,
    "symmetry": FLOW_BC_SYMMETRY,
}

_HEAT_BC_CODES = {
    "dirichlet": HEAT_BC_DIRICHLET,
    "neumann": HEAT_BC_NEUMANN,
    "adiabatic": HEAT_BC_ADIABATIC,
    "symmetry": HEAT_BC_SYMMETRY,
    "outlet": HEAT_BC_OUTLET,
    "open": HEAT_BC_OPEN,
}

_SIDES = ("east", "west", "north", "south")


def build_masks(region, region_defs):
    """Build named, fluid and solid boolean masks from the region map."""
    masks = {}

    for region_id, info in region_defs.items():
        masks[info["name"]] = region == region_id

    fluid_mask = np.zeros_like(region, dtype=bool)
    solid_mask = np.zeros_like(region, dtype=bool)

    for region_id, info in region_defs.items():
        this_mask = region == region_id
        phase = str(info["phase"]).lower()
        if phase == "fluid":
            fluid_mask |= this_mask
        elif phase == "solid":
            solid_mask |= this_mask

    masks["fluid"] = fluid_mask
    masks["solid"] = solid_mask
    return masks


def build_fluid_index_map(region, region_defs):
    """Build compact algebraic indexing for fluid cells only."""
    masks = build_masks(region, region_defs)
    is_fluid = masks["fluid"]

    ny, nx = region.shape
    cell_to_fid = -np.ones((ny, nx), dtype=np.int64)
    fluid_cells = []

    for j in range(ny):
        for i in range(nx):
            if is_fluid[j, i]:
                fid = len(fluid_cells)
                fluid_cells.append((i, j))
                cell_to_fid[j, i] = fid

    fluid_i = np.fromiter((cell[0] for cell in fluid_cells), dtype=np.int64)
    fluid_j = np.fromiter((cell[1] for cell in fluid_cells), dtype=np.int64)

    return {
        "fluid_cells": fluid_cells,
        "fluid_i": fluid_i,
        "fluid_j": fluid_j,
        "cell_to_fid": cell_to_fid,
        "Nf": len(fluid_cells),
    }


def _flow_bc_arrays(boundaries: Dict[str, Dict[str, Any]]):
    codes = np.empty(4, dtype=np.int8)
    u = np.zeros(4, dtype=float)
    v = np.zeros(4, dtype=float)
    p = np.zeros(4, dtype=float)

    for direction, side in enumerate(_SIDES):
        spec = boundaries.get(side, {})
        code = _FLOW_BC_CODES.get(
            str(spec.get("type", "wall")).lower(), FLOW_BC_OTHER
        )
        codes[direction] = code
        u[direction] = float(spec.get("u", 0.0))
        v[direction] = float(spec.get("v", 0.0))
        p[direction] = float(spec.get("p", 0.0))
    return codes, u, v, p


def _heat_bc_arrays(boundaries: Dict[str, Dict[str, Any]]):
    codes = np.empty(4, dtype=np.int8)
    temperature = np.zeros(4, dtype=float)
    flux = np.zeros(4, dtype=float)

    for direction, side in enumerate(_SIDES):
        spec = boundaries.get(side, {})
        code = _HEAT_BC_CODES.get(
            str(spec.get("type", "adiabatic")).lower(), HEAT_BC_OTHER
        )
        codes[direction] = code
        temperature[direction] = float(spec.get("T", 0.0))
        flux[direction] = float(spec.get("q", 0.0))
    return codes, temperature, flux


def _classify_neighbor(
    i: int,
    j: int,
    ni: int,
    nj: int,
    nx: int,
    ny: int,
    is_fluid: np.ndarray,
    is_solid: np.ndarray,
) -> int:
    if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
        return FACE_BOUNDARY
    if is_fluid[nj, ni]:
        return FACE_FLUID_FLUID
    if is_solid[nj, ni]:
        return FACE_FLUID_SOLID
    return FACE_BOUNDARY


def build_solver_topology(
    *,
    nx: int,
    ny: int,
    is_fluid: np.ndarray,
    is_solid: np.ndarray,
    index_data: Dict[str, Any],
    flow_boundaries: Dict[str, Dict[str, Any]],
    heat_boundaries: Dict[str, Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    """Precompute all static neighbour/face/boundary information.

    The output is intentionally numeric-only so it can be passed directly to
    Numba kernels without Python dictionaries or string comparisons.
    """
    fluid_i = np.asarray(index_data["fluid_i"], dtype=np.int64)
    fluid_j = np.asarray(index_data["fluid_j"], dtype=np.int64)
    cell_to_fid = np.asarray(index_data["cell_to_fid"], dtype=np.int64)
    nf = int(index_data["Nf"])

    face_kind = np.empty((nf, 4), dtype=np.int8)
    neighbor_fid = -np.ones((nf, 4), dtype=np.int64)

    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for fid in range(nf):
        i = int(fluid_i[fid])
        j = int(fluid_j[fid])
        for direction, (di, dj) in enumerate(offsets):
            ni = i + di
            nj = j + dj
            kind = _classify_neighbor(
                i, j, ni, nj, nx, ny, is_fluid, is_solid
            )
            face_kind[fid, direction] = kind
            if kind == FACE_FLUID_FLUID:
                neighbor_fid[fid, direction] = int(cell_to_fid[nj, ni])

    # Energy is solved for every cell, so retain a separate full-grid face map.
    ncell = nx * ny
    cell_face_kind = np.empty((ncell, 4), dtype=np.int8)
    cell_neighbor = -np.ones((ncell, 4), dtype=np.int64)
    for j in range(ny):
        for i in range(nx):
            cell = j * nx + i
            for direction, (di, dj) in enumerate(offsets):
                ni = i + di
                nj = j + dj
                kind = _classify_neighbor(
                    i, j, ni, nj, nx, ny, is_fluid, is_solid
                )
                cell_face_kind[cell, direction] = kind
                if kind != FACE_BOUNDARY:
                    cell_neighbor[cell, direction] = nj * nx + ni

    flow_bc_code, flow_bc_u, flow_bc_v, flow_bc_p = _flow_bc_arrays(
        flow_boundaries
    )
    heat_bc_code, heat_bc_T, heat_bc_q = _heat_bc_arrays(heat_boundaries)

    return {
        "fluid_i": fluid_i,
        "fluid_j": fluid_j,
        "neighbor_fid": neighbor_fid,
        "face_kind": face_kind,
        "cell_face_kind": cell_face_kind,
        "cell_neighbor": cell_neighbor,
        "flow_bc_code": flow_bc_code,
        "flow_bc_u": flow_bc_u,
        "flow_bc_v": flow_bc_v,
        "flow_bc_p": flow_bc_p,
        "heat_bc_code": heat_bc_code,
        "heat_bc_T": heat_bc_T,
        "heat_bc_q": heat_bc_q,
    }