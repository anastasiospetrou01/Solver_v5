from __future__ import annotations

"""Structured MPI domain decomposition for Solver V5.

Phase F deliberately uses an equivalent structured-grid decomposition rather
than forcing the compact fluid-only algebraic ordering into PETSc DMDA.  The
physical mesh is decomposed into contiguous y-slabs.  Each rank owns complete
rows and stores only its owned rows plus a two-cell north/south halo.  The
compact fluid IDs remain global row-major IDs, therefore each rank also owns a
contiguous block of [u,v,p] algebraic rows.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from geometry import (
    EAST,
    WEST,
    NORTH,
    SOUTH,
    FACE_FLUID_FLUID,
    FACE_FLUID_SOLID,
    FACE_BOUNDARY,
    _flow_bc_arrays,
    _heat_bc_arrays,
)


@dataclass(frozen=True)
class SlabLayout:
    rank: int
    size: int
    nx: int
    ny: int
    halo: int
    j_start: int
    j_end: int
    row_starts: np.ndarray
    row_ends: np.ndarray

    @property
    def owned_ny(self) -> int:
        return self.j_end - self.j_start

    @property
    def local_ny(self) -> int:
        return self.owned_ny + 2 * self.halo

    @property
    def owned_slice(self) -> slice:
        return slice(self.halo, self.halo + self.owned_ny)


class StructuredSlabDomain:
    """One-dimensional Cartesian decomposition with explicit ghost exchange.

    The current V5 grid is uniform and Cartesian, so a y-slab decomposition is
    sufficient and keeps global row-major cell/DOF ordering contiguous on every
    rank.  A halo width of two is used because the limited SOU reconstruction
    can inspect a neighbour-of-a-neighbour at partition interfaces.
    """

    def __init__(self, nx: int, ny: int, comm, halo: int = 2):
        self.comm = comm
        self.rank = int(comm.Get_rank())
        self.size = int(comm.Get_size())
        self.nx = int(nx)
        self.ny = int(ny)
        self.halo = int(halo)
        if self.nx < 1 or self.ny < 1:
            raise ValueError("StructuredSlabDomain requires positive nx and ny.")
        if self.halo < 1:
            raise ValueError("Halo width must be at least one.")

        counts = np.full(self.size, self.ny // self.size, dtype=np.int64)
        counts[: self.ny % self.size] += 1
        starts = np.zeros(self.size, dtype=np.int64)
        if self.size > 1:
            starts[1:] = np.cumsum(counts[:-1])
        ends = starts + counts

        if self.size > 1 and np.any(counts < self.halo):
            raise ValueError(
                "Too many MPI ranks for the requested halo width: every rank "
                f"must own at least {self.halo} y-rows, ownership={counts.tolist()}."
            )

        self.layout = SlabLayout(
            rank=self.rank,
            size=self.size,
            nx=self.nx,
            ny=self.ny,
            halo=self.halo,
            j_start=int(starts[self.rank]),
            j_end=int(ends[self.rank]),
            row_starts=starts,
            row_ends=ends,
        )
        self.south_rank = self.rank - 1 if self.rank > 0 else None
        self.north_rank = self.rank + 1 if self.rank < self.size - 1 else None

    @property
    def j_start(self) -> int:
        return self.layout.j_start

    @property
    def j_end(self) -> int:
        return self.layout.j_end

    @property
    def owned_ny(self) -> int:
        return self.layout.owned_ny

    @property
    def local_ny(self) -> int:
        return self.layout.local_ny

    @property
    def owned_slice(self) -> slice:
        return self.layout.owned_slice

    @property
    def energy_row_start(self) -> int:
        return self.j_start * self.nx

    @property
    def energy_row_end(self) -> int:
        return self.j_end * self.nx

    @property
    def local_energy_size(self) -> int:
        return self.owned_ny * self.nx

    def owns_global_j(self, j: int) -> bool:
        return self.j_start <= int(j) < self.j_end

    def global_to_local_j(self, j: int) -> int:
        return self.halo + int(j) - self.j_start

    def local_to_global_j(self, j_local: int) -> int:
        return self.j_start + int(j_local) - self.halo

    def localize(self, global_array: np.ndarray, fill_value=0) -> np.ndarray:
        """Copy only owned + halo rows from a global 2-D array."""
        source = np.asarray(global_array)
        if source.ndim != 2 or source.shape[0] != self.ny or source.shape[1] != self.nx:
            raise ValueError(
                f"Expected global array shape {(self.ny, self.nx)}, got {source.shape}."
            )
        local = np.full(
            (self.local_ny, self.nx), fill_value, dtype=source.dtype
        )
        gj0 = max(0, self.j_start - self.halo)
        gj1 = min(self.ny, self.j_end + self.halo)
        lj0 = self.halo + gj0 - self.j_start
        lj1 = lj0 + (gj1 - gj0)
        local[lj0:lj1, :] = source[gj0:gj1, :]
        return np.ascontiguousarray(local)

    def exchange_halo(self, array: np.ndarray) -> None:
        """Exchange north/south ghost rows in-place."""
        if self.size == 1:
            return
        a = np.asarray(array)
        if a.ndim < 2 or a.shape[0] != self.local_ny:
            raise ValueError(
                "Halo exchange expects an array whose first dimension is local_ny."
            )
        h = self.halo
        n = self.owned_ny
        from mpi4py import MPI

        south = self.south_rank if self.south_rank is not None else MPI.PROC_NULL
        north = self.north_rank if self.north_rank is not None else MPI.PROC_NULL

        # Send the south edge to the south neighbour and receive the north
        # neighbour's south edge into our north halo.
        self.comm.Sendrecv(
            sendbuf=a[h : h + h, ...],
            dest=south,
            sendtag=1101,
            recvbuf=a[h + n : h + n + h, ...],
            source=north,
            recvtag=1101,
        )
        # Send the north edge to the north neighbour and receive the south
        # neighbour's north edge into our south halo.
        self.comm.Sendrecv(
            sendbuf=a[h + n - h : h + n, ...],
            dest=north,
            sendtag=1102,
            recvbuf=a[0:h, ...],
            source=south,
            recvtag=1102,
        )

    def exchange_many(self, arrays: Iterable[np.ndarray]) -> None:
        for array in arrays:
            self.exchange_halo(array)

    def allreduce_max(self, value: float) -> float:
        if self.size == 1:
            return float(value)
        from mpi4py import MPI
        return float(self.comm.allreduce(float(value), op=MPI.MAX))

    def allreduce_sum(self, value: float) -> float:
        if self.size == 1:
            return float(value)
        from mpi4py import MPI
        return float(self.comm.allreduce(float(value), op=MPI.SUM))

    def gather_owned_field(self, local_array: np.ndarray, root: int = 0):
        """Gather a distributed y-slab field only when output/reporting needs it."""
        owned = np.ascontiguousarray(np.asarray(local_array)[self.owned_slice, :])
        pieces = self.comm.gather(owned, root=root)
        if self.rank != root:
            return None
        result = np.concatenate(pieces, axis=0)
        if result.shape != (self.ny, self.nx):
            raise RuntimeError(
                f"Gathered field has shape {result.shape}, expected {(self.ny, self.nx)}."
            )
        return result

    def description(self) -> str:
        return (
            f"structured y-slab decomposition, halo={self.halo}, "
            f"owned rows={self.j_start}:{self.j_end}"
        )


def _global_row_fid_maps(is_fluid: np.ndarray):
    """Return global row offsets and a helper for compact row-major fluid IDs."""
    row_counts = np.count_nonzero(is_fluid, axis=1).astype(np.int64)
    row_offsets = np.zeros(is_fluid.shape[0] + 1, dtype=np.int64)
    row_offsets[1:] = np.cumsum(row_counts)
    return row_counts, row_offsets


def _make_row_fid_map(mask_row: np.ndarray, row_offset: int) -> np.ndarray:
    mapping = -np.ones(mask_row.size, dtype=np.int64)
    fluid_x = np.flatnonzero(mask_row)
    mapping[fluid_x] = row_offset + np.arange(fluid_x.size, dtype=np.int64)
    return mapping


def build_local_solver_topology(
    *,
    domain: StructuredSlabDomain,
    global_is_fluid: np.ndarray,
    global_is_solid: np.ndarray,
    flow_boundaries: Dict[str, Dict[str, Any]],
    heat_boundaries: Dict[str, Dict[str, Any]],
) -> Dict[str, np.ndarray | int]:
    """Build only this rank's owned flow/energy topology.

    Global compact fluid IDs remain row-major.  Since each rank owns complete
    contiguous y-rows, its fluid IDs and therefore [u,v,p] rows are contiguous.
    """
    is_fluid = np.asarray(global_is_fluid, dtype=np.bool_)
    is_solid = np.asarray(global_is_solid, dtype=np.bool_)
    ny, nx = is_fluid.shape
    if (ny, nx) != (domain.ny, domain.nx):
        raise ValueError("Global masks do not match the distributed domain.")

    _row_counts, row_offsets = _global_row_fid_maps(is_fluid)
    nf_global = int(row_offsets[-1])
    fid_start = int(row_offsets[domain.j_start])
    fid_end = int(row_offsets[domain.j_end])
    local_nf = fid_end - fid_start

    relevant_j0 = max(0, domain.j_start - domain.halo)
    relevant_j1 = min(ny, domain.j_end + domain.halo)
    row_maps = {
        j: _make_row_fid_map(is_fluid[j], int(row_offsets[j]))
        for j in range(relevant_j0, relevant_j1)
    }

    fluid_i = np.empty(local_nf, dtype=np.int64)
    fluid_j = np.empty(local_nf, dtype=np.int64)
    fluid_j_global = np.empty(local_nf, dtype=np.int64)
    neighbor_fid = -np.ones((local_nf, 4), dtype=np.int64)
    face_kind = np.empty((local_nf, 4), dtype=np.int8)

    local_index = 0
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for jg in range(domain.j_start, domain.j_end):
        jl = domain.global_to_local_j(jg)
        for i in np.flatnonzero(is_fluid[jg]):
            fluid_i[local_index] = i
            fluid_j[local_index] = jl
            fluid_j_global[local_index] = jg
            for direction, (di, dj) in enumerate(directions):
                ni = int(i + di)
                nj = int(jg + dj)
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    kind = FACE_BOUNDARY
                elif is_fluid[nj, ni]:
                    kind = FACE_FLUID_FLUID
                    neighbor_fid[local_index, direction] = int(row_maps[nj][ni])
                elif is_solid[nj, ni]:
                    kind = FACE_FLUID_SOLID
                else:
                    kind = FACE_BOUNDARY
                face_kind[local_index, direction] = kind
            local_index += 1

    if local_index != local_nf:
        raise RuntimeError("Local fluid enumeration did not match row-offset ownership.")

    n_owned_cells = domain.local_energy_size
    energy_face_kind = np.empty((n_owned_cells, 4), dtype=np.int8)
    energy_neighbor_gid = -np.ones((n_owned_cells, 4), dtype=np.int64)
    energy_global_gid = np.arange(
        domain.energy_row_start, domain.energy_row_end, dtype=np.int64
    )
    for oj, jg in enumerate(range(domain.j_start, domain.j_end)):
        for i in range(nx):
            lc = oj * nx + i
            for direction, (di, dj) in enumerate(directions):
                ni = i + di
                nj = jg + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    kind = FACE_BOUNDARY
                elif is_fluid[nj, ni]:
                    kind = FACE_FLUID_FLUID
                    energy_neighbor_gid[lc, direction] = nj * nx + ni
                elif is_solid[nj, ni]:
                    kind = FACE_FLUID_SOLID
                    energy_neighbor_gid[lc, direction] = nj * nx + ni
                else:
                    kind = FACE_BOUNDARY
                energy_face_kind[lc, direction] = kind

    flow_bc_code, flow_bc_u, flow_bc_v, flow_bc_p = _flow_bc_arrays(flow_boundaries)
    heat_bc_code, heat_bc_T, heat_bc_q = _heat_bc_arrays(heat_boundaries)

    return {
        "fluid_i": fluid_i,
        "fluid_j": fluid_j,
        "fluid_j_global": fluid_j_global,
        "neighbor_fid": neighbor_fid,
        "face_kind": face_kind,
        "Nf": nf_global,
        "local_Nf": local_nf,
        "fid_start": fid_start,
        "fid_end": fid_end,
        "row_offsets": row_offsets,
        "energy_face_kind": energy_face_kind,
        "energy_neighbor_gid": energy_neighbor_gid,
        "energy_global_gid": energy_global_gid,
        "flow_bc_code": flow_bc_code,
        "flow_bc_u": flow_bc_u,
        "flow_bc_v": flow_bc_v,
        "flow_bc_p": flow_bc_p,
        "heat_bc_code": heat_bc_code,
        "heat_bc_T": heat_bc_T,
        "heat_bc_q": heat_bc_q,
    }
