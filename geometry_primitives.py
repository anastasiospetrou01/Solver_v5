import numpy as np

# ============================================================
# GEOMETRY PRIMITIVES
# ------------------------------------------------------------
# These functions "paint" region IDs into the region array.
# This keeps slicing-based geometry, but makes it reusable.
# ============================================================


def clip_bounds(x0, x1, y0, y1, nx, ny):
    """
    Clip rectangle bounds to the computational domain.
    """
    x0 = max(0, int(x0))
    x1 = min(nx, int(x1))
    y0 = max(0, int(y0))
    y1 = min(ny, int(y1))
    return x0, x1, y0, y1


def fill_rectangle(region, x0, x1, y0, y1, region_id):
    """
    Fill a rectangular block with a given region ID.
    """
    ny, nx = region.shape
    x0, x1, y0, y1 = clip_bounds(x0, x1, y0, y1, nx, ny)
    if x1 > x0 and y1 > y0:
        region[y0:y1, x0:x1] = region_id


def fill_hollow_rectangle(region, x0, x1, y0, y1, thickness, wall_region_id, inner_region_id=None):
    """
    Draw a hollow rectangle (e.g. a cup wall).

    Parameters
    ----------
    region : 2D int array
        Region map.
    x0, x1, y0, y1 : int
        Outer rectangle bounds.
    thickness : int
        Wall thickness in cells.
    wall_region_id : int
        Region ID for the wall.
    inner_region_id : int or None
        If provided, fills the interior with this region ID.
    """
    fill_rectangle(region, x0, x1, y0, y1, wall_region_id)

    xi0 = x0 + thickness
    xi1 = x1 - thickness
    yi0 = y0 + thickness
    yi1 = y1

    if inner_region_id is not None and xi1 > xi0 and yi1 > yi0:
        fill_rectangle(region, xi0, xi1, yi0, yi1, inner_region_id)


def fill_circle(region, cx, cy, radius, region_id):
    """
    Fill a circle with a given region ID.
    """
    ny, nx = region.shape
    y, x = np.ogrid[:ny, :nx]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    region[mask] = region_id


def fill_annulus(region, cx, cy, r_outer, r_inner, wall_region_id, inner_region_id=None):
    """
    Draw a circular wall and optionally fill the interior.
    """
    ny, nx = region.shape
    y, x = np.ogrid[:ny, :nx]

    r2 = (x - cx) ** 2 + (y - cy) ** 2
    outer_mask = r2 <= r_outer ** 2
    inner_mask = r2 <= r_inner ** 2

    region[outer_mask] = wall_region_id

    if inner_region_id is not None:
        region[inner_mask] = inner_region_id


def fill_polygon_placeholder(region, vertices, region_id):
    """
    Placeholder for future polygon support.

    For now this is intentionally not implemented because
    your current workflow is based on slicing logic.
    """
    raise NotImplementedError("Polygon filling not implemented yet.")