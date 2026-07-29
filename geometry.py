import numpy as np


def build_masks(region, region_defs):
    """
    Build a dictionary of boolean masks from the integer region map.
    """
    masks = {}

    for region_id, info in region_defs.items():
        masks[info["name"]] = (region == region_id)

    fluid_mask = np.zeros_like(region, dtype=bool)
    solid_mask = np.zeros_like(region, dtype=bool)

    for region_id, info in region_defs.items():
        this_mask = (region == region_id)
        if info["phase"] == "fluid":
            fluid_mask |= this_mask
        elif info["phase"] == "solid":
            solid_mask |= this_mask

    masks["fluid"] = fluid_mask
    masks["solid"] = solid_mask

    return masks


def build_fluid_index_map(region, region_defs):
    """
    Build compact indexing for momentum solver:
    only fluid cells get algebraic unknowns.
    """
    masks = build_masks(region, region_defs)
    is_fluid = masks["fluid"]

    ny, nx = region.shape
    cell_to_fid = -np.ones((ny, nx), dtype=int)
    fluid_cells = []

    for j in range(ny):
        for i in range(nx):
            if is_fluid[j, i]:
                fid = len(fluid_cells)
                fluid_cells.append((i, j))
                cell_to_fid[j, i] = fid

    return {
        "fluid_cells": fluid_cells,
        "cell_to_fid": cell_to_fid,
        "Nf": len(fluid_cells),
    }