from __future__ import annotations

"""Persistent PETSc/MUMPS direct solver with distributed fixed-COO updates."""

import hashlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import sparse


DEFAULT_DIRECT_SOLVER_SETTINGS: Dict[str, Any] = {
    "error_on_nonconvergence": True,
    "use_options_database": True,
    "verbose": True,
    "profiling": {"enabled": True},
    "solver_type": "mumps",
    "reuse_ordering": True,
    "reuse_fill": True,
    "ordering_type": None,
    "fill": None,
    "mumps_icntl": {},
    "mumps_cntl": {},
    "flow_coupled": {
        "options_prefix": "flowv5_",
        "preallocation_nnz": 20,
        "true_residual_norm": "inf",
        "true_residual_tolerance": 1.0e-5,
        "print_diagnostics": True,
    },
    "energy": {
        "options_prefix": "energy_",
        "preallocation_nnz": 5,
        "true_residual_norm": "inf",
        "true_residual_tolerance": 1.0e-8,
        "print_diagnostics": False,
    },
}


def _deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in base.items():
        result[key] = _deep_update(value, None) if isinstance(value, dict) else value
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _deep_update(result[key], value)
            else:
                result[key] = value
    return result


@dataclass
class DirectSolveInfo:
    backend: str = "petsc_direct_lu"
    system_type: str = "unknown"
    converged: bool = True
    reason: Optional[int] = None
    iterations: Optional[int] = None
    residual_norm: Optional[float] = None
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _PetscDirectCache:
    key: Any
    comm: Any
    mat: Any
    rhs: Any
    solution: Any
    ksp: Any
    global_size: int
    block_size: int
    pattern_signature: Any
    row_start: int
    row_end: int
    fixed_coo: bool


def _true_residual(matrix, rhs, solution, norm_type: str = "inf") -> Tuple[float, float]:
    matrix_csr = sparse.csr_matrix(matrix)
    rhs_array = np.asarray(rhs, dtype=float).reshape(-1)
    x = np.asarray(solution, dtype=float).reshape(-1)
    residual = rhs_array - matrix_csr @ x

    if str(norm_type).lower() in ("inf", "linf", "infinity"):
        absolute = float(np.max(np.abs(residual))) if residual.size else 0.0
        rhs_norm = float(max(np.max(np.abs(rhs_array)), 1.0e-30)) if rhs_array.size else 1.0
    else:
        absolute = float(np.linalg.norm(residual))
        rhs_norm = float(max(np.linalg.norm(rhs_array), 1.0e-30))
    return absolute / rhs_norm, absolute


class PetscDirectSolver:
    """Persistent sparse direct solver using PETSc LU and MUMPS.

    Phase D/F retains persistent fixed-COO matrices while allowing the CFD
    decomposition to prescribe PETSc local row ownership and return only each
    rank's local solution vector.
    """

    supports_pressure_nullspace = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = _deep_update(DEFAULT_DIRECT_SOLVER_SETTINGS, config)
        self.PETSc = self._load_petsc()
        self._caches: Dict[Any, _PetscDirectCache] = {}
        self.last_info = DirectSolveInfo()

    @staticmethod
    def _load_petsc():
        try:
            import petsc4py
            try:
                petsc4py.init(sys.argv)
            except Exception:
                pass
            from petsc4py import PETSc
        except Exception as exc:
            raise ImportError(
                "petsc4py/PETSc could not be loaded. This solver requires a "
                "PETSc build containing the MUMPS factorization package."
            ) from exc
        return PETSc

    @property
    def name(self) -> str:
        return "petsc_direct_lu"

    @property
    def rank(self) -> int:
        return int(self.PETSc.COMM_WORLD.getRank())

    @property
    def size(self) -> int:
        return int(self.PETSc.COMM_WORLD.getSize())

    @property
    def comm(self):
        return self.PETSc.COMM_WORLD if self.size > 1 else self.PETSc.COMM_SELF

    def uses_local_flow_assembly(self) -> bool:
        return True

    def barrier(self) -> None:
        self.PETSc.COMM_WORLD.barrier()

    def broadcast(self, value, root: int = 0):
        try:
            return self.PETSc.COMM_WORLD.tompi4py().bcast(value, root=root)
        except Exception as exc:
            if self.size > 1:
                raise RuntimeError("mpi4py is required for MPI control-data broadcasts.") from exc
            return value

    def allreduce_max(self, value: float) -> float:
        """Return the slowest-rank timing value."""
        if self.size == 1:
            return float(value)
        try:
            from mpi4py import MPI
            return float(
                self.PETSc.COMM_WORLD.tompi4py().allreduce(float(value), op=MPI.MAX)
            )
        except Exception as exc:
            raise RuntimeError("mpi4py is required for MPI max-reduced profiling.") from exc

    def reduce_timing_max(self, timing):
        """Recursively MPI-MAX reduce a timing dictionary.

        Timing dictionaries are generated deterministically on every rank, so
        all ranks traverse the same sorted key order.
        """
        if not isinstance(timing, dict):
            return timing
        reduced = {}
        for key in sorted(timing):
            value = timing[key]
            if isinstance(value, dict):
                reduced[key] = self.reduce_timing_max(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                reduced[key] = self.allreduce_max(float(value))
            else:
                reduced[key] = value
        return reduced

    def describe(self) -> str:
        mode = "distributed local-field fixed-COO assembly" if self.size > 1 else "serial fixed-COO assembly"
        return f"PETSc direct LU ({self.config.get('solver_type', 'mumps')}, {mode})"

    def _system_options(self, system_type: str) -> Dict[str, Any]:
        common = {
            "solver_type": self.config.get("solver_type", "mumps"),
            "reuse_ordering": self.config.get("reuse_ordering", True),
            "reuse_fill": self.config.get("reuse_fill", True),
            "ordering_type": self.config.get("ordering_type"),
            "fill": self.config.get("fill"),
            "mumps_icntl": dict(self.config.get("mumps_icntl", {})),
            "mumps_cntl": dict(self.config.get("mumps_cntl", {})),
        }
        return _deep_update(common, self.config.get(system_type, {}))

    @staticmethod
    def _prefix(prefix: str) -> str:
        prefix = str(prefix or "")
        return prefix if not prefix or prefix.endswith("_") else prefix + "_"

    def _set_option(self, key: str, value: Any) -> None:
        self.PETSc.Options()[str(key)] = str(value)

    @staticmethod
    def _as_csr(matrix) -> sparse.csr_matrix:
        matrix_csr = sparse.csr_matrix(matrix)
        matrix_csr.sum_duplicates()
        matrix_csr.sort_indices()
        return matrix_csr

    @staticmethod
    def _pattern_signature(matrix_csr: sparse.csr_matrix) -> str:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.asarray(matrix_csr.shape, dtype=np.int64).tobytes())
        digest.update(np.asarray(matrix_csr.indptr, dtype=np.int64).tobytes())
        digest.update(np.asarray(matrix_csr.indices, dtype=np.int64).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _local_size(global_size: int, block_size: int, comm) -> int:
        size = int(comm.getSize())
        rank = int(comm.getRank())
        if global_size % block_size:
            quotient, remainder = divmod(global_size, size)
            return quotient + (1 if rank < remainder else 0)
        global_blocks = global_size // block_size
        quotient, remainder = divmod(global_blocks, size)
        local_blocks = quotient + (1 if rank < remainder else 0)
        return block_size * local_blocks

    @classmethod
    def _local_layout(cls, global_size: int, block_size: int, comm):
        size = int(comm.getSize())
        rank = int(comm.getRank())
        local_sizes = []
        for r in range(size):
            class _RankProxy:
                def getSize(self_nonlocal):
                    return size
                def getRank(self_nonlocal):
                    return r
            local_sizes.append(cls._local_size(global_size, block_size, _RankProxy()))
        row_start = int(sum(local_sizes[:rank]))
        local_size = int(local_sizes[rank])
        return local_size, row_start, row_start + local_size

    def _set_factor_options(self, options: Dict[str, Any], prefix: str) -> None:
        self._set_option(f"{prefix}ksp_type", "preonly")
        self._set_option(f"{prefix}pc_type", "lu")
        self._set_option(f"{prefix}pc_factor_mat_solver_type", options.get("solver_type", "mumps"))
        self._set_option(f"{prefix}pc_factor_reuse_ordering", str(bool(options.get("reuse_ordering", True))).lower())
        self._set_option(f"{prefix}pc_factor_reuse_fill", str(bool(options.get("reuse_fill", True))).lower())

        ordering = options.get("ordering_type")
        if ordering not in (None, "", "auto"):
            self._set_option(f"{prefix}pc_factor_mat_ordering_type", ordering)
        fill = options.get("fill")
        if fill is not None:
            self._set_option(f"{prefix}pc_factor_fill", fill)
        for key, value in dict(options.get("mumps_icntl", {})).items():
            self._set_option(f"{prefix}mat_mumps_icntl_{int(key)}", value)
        for key, value in dict(options.get("mumps_cntl", {})).items():
            self._set_option(f"{prefix}mat_mumps_cntl_{int(key)}", value)

    def _create_cache(
        self,
        *,
        key,
        global_size: int,
        block_size: int,
        local_size: int,
        expected_row_start: int,
        expected_row_end: int,
        pattern_signature,
        preallocation_nnz: int,
        options: Dict[str, Any],
        comm,
        fixed_coo: bool,
        coo_rows=None,
        coo_cols=None,
    ) -> _PetscDirectCache:
        PETSc = self.PETSc

        matrix = PETSc.Mat().create(comm=comm)
        matrix.setSizes(((int(local_size), int(global_size)), (int(local_size), int(global_size))))
        matrix.setType(PETSc.Mat.Type.AIJ)
        matrix.setBlockSize(int(block_size))

        if fixed_coo:
            if not hasattr(matrix, "setPreallocationCOO"):
                raise RuntimeError(
                    "This PETSc/petsc4py build does not provide Mat.setPreallocationCOO."
                )
            matrix.setPreallocationCOO(
                np.asarray(coo_rows, dtype=PETSc.IntType).copy(),
                np.asarray(coo_cols, dtype=PETSc.IntType).copy(),
            )
        else:
            matrix.setPreallocationNNZ(max(int(preallocation_nnz), 1))
        matrix.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        matrix.setUp()

        row_start, row_end = matrix.getOwnershipRange()
        row_start = int(row_start)
        row_end = int(row_end)
        if row_start != int(expected_row_start) or row_end != int(expected_row_end):
            matrix.destroy()
            raise RuntimeError(
                "PETSc ownership does not match the structured CFD decomposition: "
                f"PETSc={row_start}:{row_end}, expected="
                f"{expected_row_start}:{expected_row_end}."
            )

        rhs = matrix.createVecLeft()
        solution = matrix.createVecRight()

        ksp = PETSc.KSP().create(comm=comm)
        prefix = self._prefix(options.get("options_prefix", "direct_"))
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType(str(options.get("solver_type", "mumps")))
        try:
            pc.setFactorOrdering(None, reuse=bool(options.get("reuse_ordering", True)))
        except Exception:
            pass

        self._set_factor_options(options, prefix)
        if bool(self.config.get("use_options_database", True)):
            ksp.setFromOptions()

        return _PetscDirectCache(
            key=key,
            comm=comm,
            mat=matrix,
            rhs=rhs,
            solution=solution,
            ksp=ksp,
            global_size=int(global_size),
            block_size=int(block_size),
            pattern_signature=pattern_signature,
            row_start=row_start,
            row_end=row_end,
            fixed_coo=bool(fixed_coo),
        )

    @staticmethod
    def _destroy_cache(cache: _PetscDirectCache) -> None:
        for obj in (cache.ksp, cache.mat, cache.rhs, cache.solution):
            try:
                obj.destroy()
            except Exception:
                pass

    def _get_cache(self, matrix_or_system, system_type: str, options, metadata):
        is_fixed_coo = bool(getattr(matrix_or_system, "is_fixed_coo", False))
        is_local_legacy = bool(getattr(matrix_or_system, "is_distributed_local", False)) and not is_fixed_coo
        matrix_csr = None
        coo_rows = coo_cols = None
        comm = self.comm

        if is_fixed_coo:
            global_size = int(matrix_or_system.global_size)
            block_size = int(getattr(matrix_or_system, "block_size", metadata.get("block_size", 1)))
            signature = getattr(matrix_or_system, "pattern_key", (global_size, block_size, "coo"))

            if hasattr(matrix_or_system, "local_size"):
                local_size = int(matrix_or_system.local_size)
                expected_row_start = int(matrix_or_system.row_start)
                expected_row_end = int(matrix_or_system.row_end)
            else:
                local_size, expected_row_start, expected_row_end = self._local_layout(
                    global_size, block_size, comm
                )
            coo_rows, coo_cols = matrix_or_system.local_coo_pattern(
                expected_row_start, expected_row_end
            )
            preallocation = max(int(options.get("preallocation_nnz", 1)), 1)
        elif is_local_legacy:
            global_size = int(matrix_or_system.global_size)
            block_size = int(getattr(matrix_or_system, "block_size", metadata.get("block_size", 1)))
            signature = getattr(matrix_or_system, "pattern_key", (global_size, block_size, "local"))
            local_size, expected_row_start, expected_row_end = self._local_layout(
                global_size, block_size, comm
            )
            preallocation = int(getattr(matrix_or_system, "preallocation_nnz", options.get("preallocation_nnz", 20)))
        else:
            matrix_csr = self._as_csr(matrix_or_system)
            if matrix_csr.shape[0] != matrix_csr.shape[1]:
                raise ValueError("Direct LU requires a square matrix.")
            global_size = int(matrix_csr.shape[0])
            block_size = int(metadata.get("block_size", 1))
            signature = self._pattern_signature(matrix_csr)
            row_nnz = np.diff(matrix_csr.indptr)
            preallocation = int(max(np.max(row_nnz) if row_nnz.size else 1, 1))
            local_size, expected_row_start, expected_row_end = self._local_layout(
                global_size, block_size, comm
            )

        key = (
            system_type, int(comm.getSize()), global_size, block_size,
            int(local_size), int(expected_row_start), int(expected_row_end)
        )
        cache = self._caches.get(key)
        if cache is not None and cache.pattern_signature != signature:
            self._destroy_cache(cache)
            self._caches.pop(key, None)
            cache = None

        if cache is None:
            cache = self._create_cache(
                key=key,
                global_size=global_size,
                block_size=block_size,
                local_size=local_size,
                expected_row_start=expected_row_start,
                expected_row_end=expected_row_end,
                pattern_signature=signature,
                preallocation_nnz=preallocation,
                options=options,
                comm=comm,
                fixed_coo=is_fixed_coo,
                coo_rows=coo_rows,
                coo_cols=coo_cols,
            )
            self._caches[key] = cache
        return cache, is_fixed_coo, is_local_legacy, matrix_csr

    def _fill_from_global_csr(self, cache, matrix_csr, rhs_global) -> None:
        PETSc = self.PETSc
        matrix = cache.mat
        matrix.zeroEntries()
        row_start, row_end = cache.row_start, cache.row_end
        for row in range(row_start, row_end):
            start = int(matrix_csr.indptr[row])
            end = int(matrix_csr.indptr[row + 1])
            if end <= start:
                continue
            matrix.setValues(
                np.asarray([row], dtype=PETSc.IntType),
                np.asarray(matrix_csr.indices[start:end], dtype=PETSc.IntType),
                np.asarray(matrix_csr.data[start:end], dtype=PETSc.ScalarType).reshape(1, -1),
            )
        matrix.assemble()
        rhs_array = cache.rhs.getArray()
        rhs_array[:] = np.asarray(rhs_global, dtype=float)[row_start:row_end]
        cache.rhs.assemble()
        cache.solution.set(0.0)

    def _gather_solution(self, vector, comm) -> np.ndarray:
        if int(comm.getSize()) == 1:
            return np.asarray(vector.getArray(readonly=True), dtype=float).copy()
        scatter, sequential = self.PETSc.Scatter.toAll(vector)
        scatter.scatter(vector, sequential)
        result = np.asarray(sequential.getArray(readonly=True), dtype=float).copy()
        scatter.destroy()
        sequential.destroy()
        return result

    def _distributed_residual_metrics(self, mat, rhs_vec, solution_vec):
        PETSc = self.PETSc
        residual = rhs_vec.duplicate()
        mat.mult(solution_vec, residual)
        residual.aypx(-1.0, rhs_vec)
        absolute = float(residual.norm(PETSc.NormType.NORM_INFINITY))
        rhs_norm = max(float(rhs_vec.norm(PETSc.NormType.NORM_INFINITY)), 1.0e-30)
        residual.destroy()
        relative = absolute / rhs_norm
        return {
            "scaled_true_rel_residual": relative,
            "scaled_true_abs_residual": absolute,
            "unscaled_true_rel_residual": relative,
            "unscaled_true_abs_residual": absolute,
            "true_rel_residual": relative,
            "true_abs_residual": absolute,
        }

    def solve(
        self,
        matrix_or_system,
        rhs,
        system_type: str = "flow_coupled",
        x0=None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> np.ndarray:
        del x0
        metadata = metadata or {}
        options = self._system_options(system_type)
        timing: Dict[str, float] = {}
        total_start = time.perf_counter()

        cache, is_fixed_coo, is_local_legacy, matrix_csr = self._get_cache(
            matrix_or_system, system_type, options, metadata
        )

        update_start = time.perf_counter()
        local_stats: Dict[str, float] = {}
        if is_fixed_coo:
            local_stats = dict(
                matrix_or_system.assemble_petsc(
                    cache.mat, cache.rhs, cache.row_start, cache.row_end
                )
            )
            cache.solution.set(0.0)
        elif is_local_legacy:
            local_stats = dict(matrix_or_system.assemble_petsc(cache.mat, cache.rhs))
            cache.solution.set(0.0)
        else:
            self._fill_from_global_csr(cache, matrix_csr, rhs)
        timing["matrix_rhs_update"] = time.perf_counter() - update_start
        timing["coo_value_fill"] = float(local_stats.get("coo_value_fill", 0.0))
        timing["coo_matrix_update"] = float(local_stats.get("coo_matrix_update", 0.0))
        timing["coo_rhs_update"] = float(local_stats.get("coo_rhs_update", 0.0))

        ksp = cache.ksp
        pc = ksp.getPC()
        ksp.setOperators(cache.mat)

        setup_start = time.perf_counter()
        ksp.setUp()
        timing["factorization_setup"] = time.perf_counter() - setup_start

        solve_start = time.perf_counter()
        ksp.solve(cache.rhs, cache.solution)
        timing["triangular_solve"] = time.perf_counter() - solve_start

        reason = int(ksp.getConvergedReason())
        iterations = int(ksp.getIterationNumber())
        petsc_residual = float(ksp.getResidualNorm())
        try:
            pc_failed = int(pc.getFailedReason())
        except Exception:
            pc_failed = 0

        gather_start = time.perf_counter()
        returns_local = bool(getattr(matrix_or_system, "returns_local_solution", False))
        if returns_local:
            solution = np.asarray(
                cache.solution.getArray(readonly=True), dtype=float
            ).copy()
        else:
            solution = self._gather_solution(cache.solution, cache.comm)
        timing["solution_gather"] = time.perf_counter() - gather_start

        residual_start = time.perf_counter()
        if is_fixed_coo or is_local_legacy:
            if hasattr(matrix_or_system, "distributed_residual_metrics"):
                metrics = matrix_or_system.distributed_residual_metrics(
                    cache.mat, cache.rhs, cache.solution
                )
            else:
                metrics = self._distributed_residual_metrics(
                    cache.mat, cache.rhs, cache.solution
                )
        else:
            true_matrix = metadata.get("true_matrix", matrix_csr)
            true_rhs = metadata.get("true_rhs", rhs)
            scaled_rel, scaled_abs = _true_residual(
                true_matrix, true_rhs, solution,
                options.get("true_residual_norm", "inf"),
            )
            unscaled_rel = None
            unscaled_abs = None
            scaling = metadata.get("scaling")
            unscaled_matrix = metadata.get("unscaled_true_matrix")
            unscaled_rhs = metadata.get("unscaled_true_rhs")
            if scaling is not None and unscaled_matrix is not None and unscaled_rhs is not None:
                physical_solution = scaling.unscale_solution(solution)
                unscaled_rel, unscaled_abs = _true_residual(
                    unscaled_matrix, unscaled_rhs, physical_solution,
                    options.get("true_residual_norm", "inf"),
                )
            metrics = {
                "scaled_true_rel_residual": scaled_rel,
                "scaled_true_abs_residual": scaled_abs,
                "unscaled_true_rel_residual": unscaled_rel,
                "unscaled_true_abs_residual": unscaled_abs,
                "true_rel_residual": max(scaled_rel, unscaled_rel if unscaled_rel is not None else scaled_rel),
                "true_abs_residual": max(scaled_abs, unscaled_abs if unscaled_abs is not None else scaled_abs),
            }
        timing["true_residual_check"] = time.perf_counter() - residual_start

        tolerance = float(options.get("true_residual_tolerance", 1.0e-5))
        true_ok = float(metrics["true_rel_residual"]) <= tolerance
        converged = reason > 0 and pc_failed == 0 and true_ok and np.all(np.isfinite(solution))

        factor_solver = str(options.get("solver_type", "mumps"))
        factor_nnz = None
        factor_memory = None
        try:
            factor_solver = str(pc.getFactorSolverType())
            factor = pc.getFactorMatrix()
            factor_info = factor.getInfo()
            factor_nnz = float(factor_info.get("nz_used", factor_info.get("nz_allocated", 0.0)))
            factor_memory = float(factor_info.get("memory", 0.0))
        except Exception:
            pass

        timing["total"] = time.perf_counter() - total_start
        self.last_info = DirectSolveInfo(
            system_type=system_type,
            converged=converged,
            reason=reason,
            iterations=iterations,
            residual_norm=petsc_residual,
            message="PETSc direct LU completed." if converged else "PETSc direct LU failed.",
            extra={
                "strategy": "direct_lu",
                "direct_solver_type": factor_solver,
                "mpi_size": int(cache.comm.getSize()),
                "persistent": True,
                "fixed_coo": is_fixed_coo,
                "local_assembly": is_fixed_coo or is_local_legacy,
                "factor_nnz": factor_nnz,
                "factor_memory": factor_memory,
                "pc_failed_reason": pc_failed,
                "acceptable_true_residual": tolerance,
                "true_residual_ok": true_ok,
                **metrics,
                "local_assembly_stats": local_stats,
                "timing": timing,
            },
        )

        if int(cache.comm.getRank()) == 0 and bool(
            options.get("print_diagnostics", self.config.get("verbose", True))
        ):
            unscaled_value = metrics.get("unscaled_true_rel_residual")
            print(
                "    PETSc direct LU | "
                f"solver={factor_solver} | mpi={int(cache.comm.getSize())} | "
                f"reason={reason} | pcFailed={pc_failed} | "
                f"scaledTrueRel={float(metrics['scaled_true_rel_residual']):.3e} | "
                f"unscaledTrueRel={float(unscaled_value) if unscaled_value is not None else float('nan'):.3e} | "
                f"allowed={tolerance:.3e} | persistent=True | fixedCOO={is_fixed_coo}"
            )

        if not converged and bool(self.config.get("error_on_nonconvergence", True)):
            raise RuntimeError(
                "PETSc direct LU failed: "
                f"solver={factor_solver}, reason={reason}, pc_failed={pc_failed}, "
                f"true relative residual={float(metrics['true_rel_residual']):.6e}, "
                f"allowed={tolerance:.6e}."
            )
        return solution

    def close(self) -> None:
        for cache in list(self._caches.values()):
            self._destroy_cache(cache)
        self._caches.clear()



@dataclass
class IterativeSolveInfo:
    backend: str = "petsc_iterative_schur"
    system_type: str = "flow_coupled"
    converged: bool = True
    reason: Optional[int] = None
    iterations: Optional[int] = None
    residual_norm: Optional[float] = None
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _PetscIterativeCache:
    key: Any
    comm: Any
    mat: Any
    rhs: Any
    solution: Any
    ksp: Any
    velocity_is: Any
    pressure_is: Any
    global_size: int
    row_start: int
    row_end: int
    pattern_signature: Any
    residual_history: list = field(default_factory=list)


class PetscIterativeFlowSolver:
    """Distributed FGMRES + PCFIELDSPLIT Schur solver for coupled [u,v,p].

    Stage 1 intentionally uses strong MUMPS subsolves for the velocity block
    and for the explicit SELFP Schur-preconditioner matrix. This validates the
    block formulation before AMG or inexact inner solves are introduced.
    """

    supports_pressure_nullspace = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = _deep_update(DEFAULT_DIRECT_SOLVER_SETTINGS, config)
        self.PETSc = PetscDirectSolver._load_petsc()
        self._caches: Dict[Any, _PetscIterativeCache] = {}
        self.last_info = IterativeSolveInfo()

    @property
    def rank(self) -> int:
        return int(self.PETSc.COMM_WORLD.getRank())

    @property
    def size(self) -> int:
        return int(self.PETSc.COMM_WORLD.getSize())

    @property
    def comm(self):
        return self.PETSc.COMM_WORLD if self.size > 1 else self.PETSc.COMM_SELF

    @property
    def name(self) -> str:
        return "petsc_iterative_schur"

    def describe(self) -> str:
        return (
            "PETSc FGMRES + PCFIELDSPLIT Schur "
            "(nested Schur KSP, distributed fixed-COO)"
        )

    def uses_local_flow_assembly(self) -> bool:
        return True

    def barrier(self) -> None:
        self.PETSc.COMM_WORLD.barrier()

    def broadcast(self, value, root: int = 0):
        try:
            return self.PETSc.COMM_WORLD.tompi4py().bcast(value, root=root)
        except Exception as exc:
            if self.size > 1:
                raise RuntimeError("mpi4py is required for MPI control-data broadcasts.") from exc
            return value

    def allreduce_max(self, value: float) -> float:
        if self.size == 1:
            return float(value)
        try:
            from mpi4py import MPI
            return float(
                self.PETSc.COMM_WORLD.tompi4py().allreduce(float(value), op=MPI.MAX)
            )
        except Exception as exc:
            raise RuntimeError("mpi4py is required for MPI max-reduced profiling.") from exc

    def reduce_timing_max(self, timing):
        if not isinstance(timing, dict):
            return timing
        reduced = {}
        for key in sorted(timing):
            value = timing[key]
            if isinstance(value, dict):
                reduced[key] = self.reduce_timing_max(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                reduced[key] = self.allreduce_max(float(value))
            else:
                reduced[key] = value
        return reduced

    @staticmethod
    def _prefix(prefix: str) -> str:
        prefix = str(prefix or "")
        return prefix if not prefix or prefix.endswith("_") else prefix + "_"

    def _system_options(self) -> Dict[str, Any]:
        result = dict(self.config.get("flow_coupled", {}))
        result.update(dict(self.config.get("iterative", {})))
        result.setdefault("options_prefix", "flowv5_iter_")
        result.setdefault("ksp_type", "fgmres")
        result.setdefault("rtol", 1.0e-6)
        result.setdefault("atol", 0.0)
        result.setdefault("dtol", 1.0e8)
        result.setdefault("max_it", 200)
        result.setdefault("restart", 50)
        result.setdefault("schur_fact_type", "full")
        result.setdefault("schur_pre_type", "selfp")
        result.setdefault("velocity_solver", "mumps")
        result.setdefault("velocity_ksp_type", "gmres")
        result.setdefault("velocity_ksp_rtol", 1.0e-2)
        result.setdefault("velocity_ksp_atol", 0.0)
        result.setdefault("velocity_ksp_dtol", 1.0e8)
        result.setdefault("velocity_ksp_max_it", 10)
        result.setdefault("velocity_ksp_restart", 10)
        result.setdefault("velocity_boomeramg_max_iter", 1)
        result.setdefault("velocity_boomeramg_tol", 0.0)

        # Stage 3E: PETSc has a separate A00 KSP used inside each matrix-free
        # Schur matvec.  Do not let it inherit the 2-3 iteration velocity GMRES;
        # use one fixed BoomerAMG V-cycle instead.
        result.setdefault("schur_inner_velocity_ksp_type", "preonly")
        result.setdefault("schur_inner_velocity_solver", "hypre")
        result.setdefault("schur_inner_velocity_boomeramg_max_iter", 1)
        result.setdefault("schur_inner_velocity_boomeramg_tol", 0.0)

        # Stage 3F: FULL Schur factorization has an additional upper-factor
        # A00 solve.  By default PETSc reuses the main velocity KSP; make this
        # a fixed one-cycle AMG action as well.
        result.setdefault("schur_upper_velocity_ksp_type", "preonly")
        result.setdefault("schur_upper_velocity_solver", "hypre")
        result.setdefault("schur_upper_velocity_boomeramg_max_iter", 1)
        result.setdefault("schur_upper_velocity_boomeramg_tol", 0.0)

        result.setdefault("pressure_solver", "mumps")
        result.setdefault("pressure_ksp_type", "gmres")
        result.setdefault("pressure_ksp_rtol", 1.0e-2)
        result.setdefault("pressure_ksp_atol", 0.0)
        result.setdefault("pressure_ksp_dtol", 1.0e8)
        result.setdefault("pressure_ksp_max_it", 30)
        result.setdefault("pressure_ksp_restart", 30)
        result.setdefault("pressure_boomeramg_max_iter", 1)
        result.setdefault("pressure_boomeramg_tol", 0.0)
        result.setdefault("monitor", False)
        result.setdefault("print_diagnostics", True)
        # The assembled operator is physically two-sided scaled.  FGMRES must
        # monitor the true residual of that scaled system, and the backend then
        # enforces the existing unscaled physical residual acceptance criterion.
        result.setdefault("norm_type", "unpreconditioned")
        result.setdefault("physical_residual_retry", True)
        result.setdefault("max_physical_residual_retries", 3)
        result.setdefault("retry_safety", 0.5)
        result.setdefault("minimum_rtol", 1.0e-12)
        return result

    def _set_option(self, key: str, value: Any) -> None:
        self.PETSc.Options()[str(key)] = str(value)

    def _set_fieldsplit_options(self, prefix: str, options: Dict[str, Any]) -> None:
        """Configure Stage-3A Schur field solves.

        The pressure Schur solve keeps the validated nested GMRES + SELFP/MUMPS
        path. The velocity A00 solve can use either the Stage-2 exact MUMPS
        baseline or a scalable GMRES + hypre BoomerAMG approximation.
        """
        vel_solver = str(options.get("velocity_solver", "mumps")).lower()
        pre_solver = str(options.get("pressure_solver", "mumps")).lower()

        # A00 velocity solve.
        vbase = f"{prefix}fieldsplit_velocity_"
        if vel_solver == "mumps":
            self._set_option(f"{vbase}ksp_type", "preonly")
            self._set_option(f"{vbase}pc_type", "lu")
            self._set_option(f"{vbase}pc_factor_mat_solver_type", "mumps")
            self._set_option(f"{vbase}pc_factor_reuse_ordering", "true")
            self._set_option(f"{vbase}pc_factor_reuse_fill", "true")
        elif vel_solver in ("hypre", "boomeramg", "amg"):
            vksp = str(options.get("velocity_ksp_type", "gmres")).lower()
            self._set_option(f"{vbase}ksp_type", vksp)
            self._set_option(f"{vbase}ksp_rtol", options.get("velocity_ksp_rtol", 1.0e-2))
            self._set_option(f"{vbase}ksp_atol", options.get("velocity_ksp_atol", 0.0))
            self._set_option(f"{vbase}ksp_divtol", options.get("velocity_ksp_dtol", 1.0e8))
            self._set_option(f"{vbase}ksp_max_it", options.get("velocity_ksp_max_it", 10))
            if vksp in ("gmres", "fgmres"):
                self._set_option(
                    f"{vbase}ksp_gmres_restart",
                    options.get("velocity_ksp_restart", 10),
                )
            self._set_option(f"{vbase}pc_type", "hypre")
            self._set_option(f"{vbase}pc_hypre_type", "boomeramg")
            # Fixed V-cycle count makes the AMG application a predictable PC.
            self._set_option(
                f"{vbase}pc_hypre_boomeramg_max_iter",
                options.get("velocity_boomeramg_max_iter", 1),
            )
            self._set_option(
                f"{vbase}pc_hypre_boomeramg_tol",
                options.get("velocity_boomeramg_tol", 0.0),
            )
        else:
            raise ValueError(
                "velocity_solver must be 'mumps' or 'hypre'/'boomeramg'."
            )

        # FULL Schur factorization has a separate upper-factor A00 KSP.
        # PETSc defaults it to the main velocity solver; Stage 3F replaces
        # that repeated GMRES solve with one fixed BoomerAMG V-cycle.
        # Because the second field split is explicitly named "pressure",
        # PETSc names the FULL-Schur upper KSP with the split-name prefix:
        #   fieldsplit_pressure_upper_
        # (not the generic fieldsplit_upper_ form used in some manual examples).
        ubase = f"{prefix}fieldsplit_pressure_upper_"
        upper_ksp = str(
            options.get("schur_upper_velocity_ksp_type", "preonly")
        ).lower()
        upper_solver = str(
            options.get("schur_upper_velocity_solver", "hypre")
        ).lower()
        if upper_ksp != "preonly":
            raise ValueError(
                "Stage-3F schur_upper_velocity_ksp_type must be 'preonly'."
            )
        if upper_solver not in ("hypre", "boomeramg", "amg"):
            raise ValueError(
                "Stage-3F schur_upper_velocity_solver must be hypre/BoomerAMG."
            )
        self._set_option(f"{ubase}ksp_type", "preonly")
        self._set_option(f"{ubase}pc_type", "hypre")
        self._set_option(f"{ubase}pc_hypre_type", "boomeramg")
        self._set_option(
            f"{ubase}pc_hypre_boomeramg_max_iter",
            options.get("schur_upper_velocity_boomeramg_max_iter", 1),
        )
        self._set_option(
            f"{ubase}pc_hypre_boomeramg_tol",
            options.get("schur_upper_velocity_boomeramg_tol", 0.0),
        )

        # Matrix-free Schur solve.
        pbase = f"{prefix}fieldsplit_pressure_"

        # PETSc's separate A00 KSP used *inside* application of
        # S = A11 - A10 A00^{-1} A01.  A fixed one-cycle AMG action avoids
        # nesting a 2-3 iteration velocity GMRES inside every Schur matvec.
        ibase = f"{pbase}inner_"
        inner_ksp = str(
            options.get("schur_inner_velocity_ksp_type", "preonly")
        ).lower()
        inner_solver = str(
            options.get("schur_inner_velocity_solver", "hypre")
        ).lower()
        if inner_ksp != "preonly":
            raise ValueError(
                "Stage-3E schur_inner_velocity_ksp_type must be 'preonly'."
            )
        if inner_solver not in ("hypre", "boomeramg", "amg"):
            raise ValueError(
                "Stage-3E schur_inner_velocity_solver must be hypre/BoomerAMG."
            )
        self._set_option(f"{ibase}ksp_type", "preonly")
        self._set_option(f"{ibase}pc_type", "hypre")
        self._set_option(f"{ibase}pc_hypre_type", "boomeramg")
        self._set_option(
            f"{ibase}pc_hypre_boomeramg_max_iter",
            options.get("schur_inner_velocity_boomeramg_max_iter", 1),
        )
        self._set_option(
            f"{ibase}pc_hypre_boomeramg_tol",
            options.get("schur_inner_velocity_boomeramg_tol", 0.0),
        )

        pksp = str(options.get("pressure_ksp_type", "gmres")).lower()
        self._set_option(f"{pbase}ksp_type", pksp)
        self._set_option(f"{pbase}ksp_rtol", options.get("pressure_ksp_rtol", 1.0e-2))
        self._set_option(f"{pbase}ksp_atol", options.get("pressure_ksp_atol", 0.0))
        self._set_option(f"{pbase}ksp_divtol", options.get("pressure_ksp_dtol", 1.0e8))
        self._set_option(f"{pbase}ksp_max_it", options.get("pressure_ksp_max_it", 30))
        if pksp in ("gmres", "fgmres"):
            self._set_option(
                f"{pbase}ksp_gmres_restart",
                options.get("pressure_ksp_restart", 30),
            )

        # Pressure preconditioner for the explicitly assembled SELFP P-matrix.
        # Stage 3B supports either the previous LU/MUMPS reference or scalable
        # hypre BoomerAMG. The Schur operator itself remains matrix-free.
        if pre_solver == "mumps":
            self._set_option(f"{pbase}pc_type", "lu")
            self._set_option(f"{pbase}pc_factor_mat_solver_type", "mumps")
            self._set_option(f"{pbase}pc_factor_reuse_ordering", "true")
            self._set_option(f"{pbase}pc_factor_reuse_fill", "true")
        elif pre_solver in ("hypre", "boomeramg", "amg"):
            self._set_option(f"{pbase}pc_type", "hypre")
            self._set_option(f"{pbase}pc_hypre_type", "boomeramg")
            self._set_option(
                f"{pbase}pc_hypre_boomeramg_max_iter",
                options.get("pressure_boomeramg_max_iter", 1),
            )
            self._set_option(
                f"{pbase}pc_hypre_boomeramg_tol",
                options.get("pressure_boomeramg_tol", 0.0),
            )
        else:
            raise ValueError(
                "pressure_solver must be 'mumps' or 'hypre'/'boomeramg'."
            )

    def _create_cache(self, system, options: Dict[str, Any]) -> _PetscIterativeCache:
        PETSc = self.PETSc
        comm = self.comm
        global_size = int(system.global_size)
        local_size = int(system.local_size)
        row_start = int(system.row_start)
        row_end = int(system.row_end)
        pattern_signature = system.pattern_key

        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes(((local_size, global_size), (local_size, global_size)))
        mat.setType(PETSc.Mat.Type.AIJ)
        mat.setBlockSize(3)
        rows, cols = system.local_coo_pattern(row_start, row_end)
        if not hasattr(mat, "setPreallocationCOO"):
            raise RuntimeError("This PETSc build does not provide Mat.setPreallocationCOO.")
        mat.setPreallocationCOO(
            np.asarray(rows, dtype=PETSc.IntType).copy(),
            np.asarray(cols, dtype=PETSc.IntType).copy(),
        )
        mat.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        mat.setUp()

        actual_start, actual_end = map(int, mat.getOwnershipRange())
        if (actual_start, actual_end) != (row_start, row_end):
            mat.destroy()
            raise RuntimeError(
                "PETSc ownership does not match the structured CFD decomposition: "
                f"PETSc={actual_start}:{actual_end}, expected={row_start}:{row_end}."
            )

        rhs = mat.createVecLeft()
        solution = mat.createVecRight()

        if local_size % 3 != 0 or row_start % 3 != 0:
            raise RuntimeError("Flow ownership must preserve complete [u,v,p] cell blocks.")
        local_nf = local_size // 3
        fid_start = row_start // 3
        fids = np.arange(fid_start, fid_start + local_nf, dtype=PETSc.IntType)
        velocity_idx = np.empty(2 * local_nf, dtype=PETSc.IntType)
        velocity_idx[0::2] = 3 * fids
        velocity_idx[1::2] = 3 * fids + 1
        pressure_idx = 3 * fids + 2
        velocity_is = PETSc.IS().createGeneral(velocity_idx, comm=comm)
        pressure_is = PETSc.IS().createGeneral(pressure_idx, comm=comm)

        ksp = PETSc.KSP().create(comm=comm)
        prefix = self._prefix(options.get("options_prefix", "flowv5_iter_"))
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(mat)
        ksp.setType(str(options.get("ksp_type", "fgmres")))
        ksp.setTolerances(
            rtol=float(options.get("rtol", 1.0e-6)),
            atol=float(options.get("atol", 0.0)),
            divtol=float(options.get("dtol", 1.0e8)),
            max_it=int(options.get("max_it", 200)),
        )
        try:
            ksp.setGMRESRestart(int(options.get("restart", 50)))
        except Exception:
            pass
        try:
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        except Exception:
            pass

        norm_type = str(options.get("norm_type", "unpreconditioned")).lower().strip()
        if norm_type in ("unpreconditioned", "true", "true_residual"):
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        elif norm_type in ("preconditioned", "preconditioner"):
            ksp.setNormType(PETSc.KSP.NormType.PRECONDITIONED)
        elif norm_type not in ("default", ""):
            raise ValueError(f"Unsupported iterative KSP norm_type: {norm_type!r}")

        ksp.setInitialGuessNonzero(False)

        residual_history = []
        if bool(options.get("monitor", False)):
            def _monitor(_ksp, its, rnorm):
                residual_history.append((int(its), float(rnorm)))
                if self.rank == 0:
                    print(
                        f"[flow_coupled] FGMRES it {int(its):4d} | "
                        f"residual {float(rnorm):.6e}"
                    )
            ksp.setMonitor(_monitor)

        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitIS(("velocity", velocity_is), ("pressure", pressure_is))
        try:
            pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        except Exception:
            pc.setFieldSplitType("schur")

        fact = str(options.get("schur_fact_type", "full")).lower()
        fact_enum = {
            "full": PETSc.PC.FieldSplitSchurFactType.FULL,
            "lower": PETSc.PC.FieldSplitSchurFactType.LOWER,
            "upper": PETSc.PC.FieldSplitSchurFactType.UPPER,
            "diag": PETSc.PC.FieldSplitSchurFactType.DIAG,
        }
        if fact not in fact_enum:
            raise ValueError(f"Unsupported Schur factorization type: {fact!r}")
        pc.setFieldSplitSchurFactType(fact_enum[fact])

        schur_pre = str(options.get("schur_pre_type", "selfp")).lower()
        pre_enum = {
            "selfp": PETSc.PC.FieldSplitSchurPreType.SELFP,
            "a11": PETSc.PC.FieldSplitSchurPreType.A11,
            "self": PETSc.PC.FieldSplitSchurPreType.SELF,
            "full": PETSc.PC.FieldSplitSchurPreType.FULL,
        }
        if schur_pre not in pre_enum:
            raise ValueError(f"Unsupported Schur preconditioner type: {schur_pre!r}")
        pc.setFieldSplitSchurPreType(pre_enum[schur_pre])

        self._set_fieldsplit_options(prefix, options)
        if bool(self.config.get("use_options_database", True)):
            ksp.setFromOptions()

        return _PetscIterativeCache(
            key=("flow_coupled", self.size, global_size, row_start, row_end),
            comm=comm,
            mat=mat,
            rhs=rhs,
            solution=solution,
            ksp=ksp,
            velocity_is=velocity_is,
            pressure_is=pressure_is,
            global_size=global_size,
            row_start=row_start,
            row_end=row_end,
            pattern_signature=pattern_signature,
            residual_history=residual_history,
        )

    @staticmethod
    def _destroy_cache(cache: _PetscIterativeCache) -> None:
        for obj in (
            cache.ksp,
            cache.velocity_is,
            cache.pressure_is,
            cache.mat,
            cache.rhs,
            cache.solution,
        ):
            try:
                obj.destroy()
            except Exception:
                pass

    def _get_cache(self, system, options: Dict[str, Any]) -> _PetscIterativeCache:
        if not bool(getattr(system, "is_fixed_coo", False)):
            raise TypeError(
                "The iterative flow backend requires the distributed fixed-COO flow system."
            )
        key = (
            "flow_coupled",
            self.size,
            int(system.global_size),
            int(system.row_start),
            int(system.row_end),
        )
        cache = self._caches.get(key)
        if cache is not None and cache.pattern_signature != system.pattern_key:
            self._destroy_cache(cache)
            self._caches.pop(key, None)
            cache = None
        if cache is None:
            cache = self._create_cache(system, options)
            self._caches[key] = cache
        return cache

    def _schur_subsolver_info(self, ksp) -> Dict[str, Any]:
        """Inspect the actual PETSc Schur subsolvers after PC setup.

        This is diagnostic only.  It verifies that the requested strong MUMPS
        block solves are really active instead of silently relying on defaults.
        """
        info: Dict[str, Any] = {}
        try:
            subksps = ksp.getPC().getFieldSplitSchurGetSubKSP()
        except Exception:
            return info

        labels = ("velocity", "pressure")
        for label, subksp in zip(labels, subksps):
            try:
                info[f"{label}_ksp_type"] = str(subksp.getType())
            except Exception:
                pass
            try:
                info[f"{label}_iterations_last"] = int(subksp.getIterationNumber())
            except Exception:
                pass
            try:
                info[f"{label}_residual_last"] = float(subksp.getResidualNorm())
            except Exception:
                pass
            try:
                info[f"{label}_reason_last"] = int(subksp.getConvergedReason())
            except Exception:
                pass
            try:
                subpc = subksp.getPC()
                info[f"{label}_pc_type"] = str(subpc.getType())
                try:
                    factor_solver = subpc.getFactorSolverType()
                    if factor_solver:
                        info[f"{label}_factor_solver"] = str(factor_solver)
                except Exception:
                    pass
            except Exception:
                pass
        return info

    def solve(
        self,
        matrix_or_system,
        rhs,
        system_type: str = "flow_coupled",
        x0=None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> np.ndarray:
        del rhs, x0, metadata
        if system_type != "flow_coupled":
            raise ValueError(
                "PetscIterativeFlowSolver is reserved for system_type='flow_coupled'."
            )

        PETSc = self.PETSc
        options = self._system_options()
        cache = self._get_cache(matrix_or_system, options)
        timing: Dict[str, float] = {}
        total_start = time.perf_counter()

        update_start = time.perf_counter()
        local_stats = dict(
            matrix_or_system.assemble_petsc(
                cache.mat, cache.rhs, cache.row_start, cache.row_end
            )
        )
        cache.solution.set(0.0)
        timing["matrix_rhs_update"] = time.perf_counter() - update_start
        timing["coo_value_fill"] = float(local_stats.get("coo_value_fill", 0.0))
        timing["coo_matrix_update"] = float(local_stats.get("coo_matrix_update", 0.0))
        timing["coo_rhs_update"] = float(local_stats.get("coo_rhs_update", 0.0))

        cache.ksp.setOperators(cache.mat)
        setup_start = time.perf_counter()
        cache.ksp.setUp()
        timing["pc_setup"] = time.perf_counter() - setup_start
        subsolver_info = self._schur_subsolver_info(cache.ksp)

        tolerance = float(options.get("true_residual_tolerance", 1.0e-5))
        requested_rtol = float(options.get("rtol", 1.0e-6))
        current_rtol = requested_rtol
        minimum_rtol = float(options.get("minimum_rtol", 1.0e-12))
        retry_enabled = bool(options.get("physical_residual_retry", True))
        max_retries = int(options.get("max_physical_residual_retries", 3)) if retry_enabled else 0
        retry_safety = float(options.get("retry_safety", 0.5))
        retry_safety = min(max(retry_safety, 1.0e-3), 1.0)

        total_iterations = 0
        total_ksp_solve_time = 0.0
        total_residual_check_time = 0.0
        attempts = []
        reason = 0
        iterations_last = 0
        petsc_residual = float("inf")
        pc_failed = 0
        metrics = {
            "scaled_true_rel_residual": float("inf"),
            "scaled_true_abs_residual": float("inf"),
            "unscaled_true_rel_residual": float("inf"),
            "unscaled_true_abs_residual": float("inf"),
            "true_rel_residual": float("inf"),
            "true_abs_residual": float("inf"),
        }

        try:
            max_it_reason = int(PETSc.KSP.ConvergedReason.DIVERGED_ITS)
        except Exception:
            max_it_reason = -3

        cache.ksp.setInitialGuessNonzero(False)
        for attempt in range(max_retries + 1):
            cache.ksp.setTolerances(
                rtol=current_rtol,
                atol=float(options.get("atol", 0.0)),
                divtol=float(options.get("dtol", 1.0e8)),
                max_it=int(options.get("max_it", 200)),
            )

            solve_start = time.perf_counter()
            cache.ksp.solve(cache.rhs, cache.solution)
            total_ksp_solve_time += time.perf_counter() - solve_start

            reason = int(cache.ksp.getConvergedReason())
            iterations_last = int(cache.ksp.getIterationNumber())
            total_iterations += iterations_last
            petsc_residual = float(cache.ksp.getResidualNorm())
            try:
                pc_failed = int(cache.ksp.getPC().getFailedReason())
            except Exception:
                pc_failed = 0

            residual_start = time.perf_counter()
            metrics = matrix_or_system.distributed_residual_metrics(
                cache.mat, cache.rhs, cache.solution
            )
            total_residual_check_time += time.perf_counter() - residual_start

            true_ok = float(metrics["true_rel_residual"]) <= tolerance
            attempts.append({
                "attempt": int(attempt),
                "rtol": float(current_rtol),
                "reason": int(reason),
                "iterations": int(iterations_last),
                "ksp_residual": float(petsc_residual),
                "scaled_true_rel_residual": float(metrics["scaled_true_rel_residual"]),
                "unscaled_true_rel_residual": float(metrics["unscaled_true_rel_residual"]),
            })

            # The independently recomputed true residual is the production
            # acceptance criterion.  PETSc DIVERGED_ITS only means the current
            # scaled-system KSP tolerance was not reached within max_it; it is
            # not a numerical failure when the physical residual already meets
            # the required tolerance.  Genuine negative breakdown reasons are
            # still rejected.
            acceptable_exit = reason > 0 or reason == max_it_reason
            if acceptable_exit and pc_failed == 0 and true_ok:
                break

            if attempt >= max_retries or pc_failed != 0:
                break

            # Continue only from a normal positive exit whose physical residual
            # is still too large, or from a max-iteration exit.  Other negative
            # PETSc reasons indicate a genuine Krylov/preconditioner breakdown.
            if reason < 0 and reason != max_it_reason:
                break

            physical_rel = max(float(metrics["true_rel_residual"]), 1.0e-300)
            ratio = tolerance / physical_rel
            candidate = current_rtol * ratio * retry_safety
            # Tighten by at least one decade when a physical-residual retry is
            # required.  The ratio-based candidate usually tightens more.
            next_rtol = min(current_rtol * 0.1, candidate)
            next_rtol = max(minimum_rtol, next_rtol)
            if next_rtol >= current_rtol:
                next_rtol = max(minimum_rtol, current_rtol * 0.1)
            if next_rtol >= current_rtol:
                break

            current_rtol = next_rtol
            cache.ksp.setInitialGuessNonzero(True)

        subsolver_info = self._schur_subsolver_info(cache.ksp)
        timing["ksp_solve"] = total_ksp_solve_time
        timing["true_residual_check"] = total_residual_check_time
        timing["krylov_iterations"] = float(total_iterations)
        timing["krylov_iterations_last"] = float(iterations_last)
        timing["ksp_residual"] = float(petsc_residual)
        timing["physical_residual_retries"] = float(max(0, len(attempts) - 1))
        timing["effective_rtol"] = float(current_rtol)

        solution_start = time.perf_counter()
        solution = np.asarray(cache.solution.getArray(readonly=True), dtype=float).copy()
        timing["solution_gather"] = time.perf_counter() - solution_start

        true_ok = float(metrics["true_rel_residual"]) <= tolerance
        finite = bool(np.all(np.isfinite(solution)))
        acceptable_exit = reason > 0 or reason == max_it_reason
        accepted_at_max_it = reason == max_it_reason and pc_failed == 0 and true_ok and finite
        converged = acceptable_exit and pc_failed == 0 and true_ok and finite
        timing["accepted_at_max_it"] = float(bool(accepted_at_max_it))
        timing["total"] = time.perf_counter() - total_start

        history = list(cache.residual_history)
        cache.residual_history.clear()

        self.last_info = IterativeSolveInfo(
            system_type=system_type,
            converged=converged,
            reason=reason,
            iterations=total_iterations,
            residual_norm=petsc_residual,
            message=(
                "PETSc FGMRES/Schur solve completed."
                if converged
                else "PETSc FGMRES/Schur solve failed acceptance checks."
            ),
            extra={
                "strategy": "fgmres_fieldsplit_schur",
                "schur_fact_type": str(options.get("schur_fact_type", "full")),
                "schur_pre_type": str(options.get("schur_pre_type", "selfp")),
                "velocity_solver": str(options.get("velocity_solver", "mumps")),
                "velocity_ksp_type": str(options.get("velocity_ksp_type", "gmres")),
                "velocity_ksp_rtol": float(options.get("velocity_ksp_rtol", 1.0e-2)),
                "velocity_ksp_max_it": int(options.get("velocity_ksp_max_it", 10)),
                "schur_inner_velocity_ksp_type": str(
                    options.get("schur_inner_velocity_ksp_type", "preonly")
                ),
                "schur_inner_velocity_solver": str(
                    options.get("schur_inner_velocity_solver", "hypre")
                ),
                "schur_inner_velocity_boomeramg_max_iter": int(
                    options.get("schur_inner_velocity_boomeramg_max_iter", 1)
                ),
                "schur_upper_velocity_ksp_type": str(
                    options.get("schur_upper_velocity_ksp_type", "preonly")
                ),
                "schur_upper_velocity_solver": str(
                    options.get("schur_upper_velocity_solver", "hypre")
                ),
                "schur_upper_velocity_boomeramg_max_iter": int(
                    options.get("schur_upper_velocity_boomeramg_max_iter", 1)
                ),
                "pressure_solver": str(options.get("pressure_solver", "mumps")),
                "pressure_ksp_type": str(options.get("pressure_ksp_type", "gmres")),
                "pressure_ksp_rtol": float(options.get("pressure_ksp_rtol", 1.0e-2)),
                "pressure_ksp_max_it": int(options.get("pressure_ksp_max_it", 30)),
                "pressure_boomeramg_max_iter": int(
                    options.get("pressure_boomeramg_max_iter", 1)
                ),
                "mpi_size": int(cache.comm.getSize()),
                "persistent": True,
                "fixed_coo": True,
                "local_assembly": True,
                "pc_failed_reason": pc_failed,
                "acceptable_true_residual": tolerance,
                "true_residual_ok": true_ok,
                "krylov_iterations": total_iterations,
                "krylov_iterations_last": iterations_last,
                "ksp_residual": petsc_residual,
                "requested_rtol": requested_rtol,
                "effective_rtol": current_rtol,
                "physical_residual_retries": max(0, len(attempts) - 1),
                "accepted_at_max_it": bool(accepted_at_max_it),
                "solve_attempts": attempts,
                **subsolver_info,
                "residual_history": history,
                **metrics,
                "local_assembly_stats": local_stats,
                "timing": timing,
            },
        )

        if self.rank == 0 and bool(options.get("print_diagnostics", True)):
            print(
                "    PETSc iterative | "
                f"FGMRES its={total_iterations} "
                f"(last={iterations_last}, retries={max(0, len(attempts) - 1)}) | "
                f"reason={reason}"
                f"{' [accepted: true residual]' if accepted_at_max_it else ''} | "
                f"pcFailed={pc_failed} | "
                f"Schur={options.get('schur_fact_type', 'full')}/"
                f"{options.get('schur_pre_type', 'selfp')} | "
                f"rtol={current_rtol:.1e} | "
                f"V={subsolver_info.get('velocity_ksp_type', '?')}/"
                f"{subsolver_info.get('velocity_pc_type', '?')}/"
                f"{subsolver_info.get('velocity_factor_solver', '?')}"
                f"(lastIts={subsolver_info.get('velocity_iterations_last', '?')},"
                f" reason={subsolver_info.get('velocity_reason_last', '?')}) | "
                f"Sinner={options.get('schur_inner_velocity_ksp_type', 'preonly')}/"
                f"{options.get('schur_inner_velocity_solver', 'hypre')}"
                f"({int(options.get('schur_inner_velocity_boomeramg_max_iter', 1))}V) | "
                f"Supper={options.get('schur_upper_velocity_ksp_type', 'preonly')}/"
                f"{options.get('schur_upper_velocity_solver', 'hypre')}"
                f"({int(options.get('schur_upper_velocity_boomeramg_max_iter', 1))}V) | "
                f"P={subsolver_info.get('pressure_ksp_type', '?')}/"
                f"{subsolver_info.get('pressure_pc_type', '?')}/"
                f"{subsolver_info.get('pressure_factor_solver', '?')}"
                f"(lastIts={subsolver_info.get('pressure_iterations_last', '?')},"
                f" reason={subsolver_info.get('pressure_reason_last', '?')}) | "
                f"scaledTrueRel={float(metrics['scaled_true_rel_residual']):.3e} | "
                f"unscaledTrueRel={float(metrics['unscaled_true_rel_residual']):.3e} | "
                f"allowed={tolerance:.3e}"
            )

        if not converged and bool(self.config.get("error_on_nonconvergence", True)):
            raise RuntimeError(
                "PETSc iterative flow solve failed: "
                f"reason={reason}, pc_failed={pc_failed}, iterations={total_iterations}, "
                f"acceptable_exit={acceptable_exit}, "
                f"true relative residual={float(metrics['true_rel_residual']):.6e}, "
                f"allowed={tolerance:.6e}."
            )
        return solution

    def close(self) -> None:
        for cache in list(self._caches.values()):
            self._destroy_cache(cache)
        self._caches.clear()


class HybridFlowIterativeSolver:
    """Iterative coupled-flow solver with direct MUMPS retained for energy."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.flow_solver = PetscIterativeFlowSolver(self.config)
        self.direct_solver = PetscDirectSolver(self.config)
        self.PETSc = self.direct_solver.PETSc
        self.last_info = self.flow_solver.last_info

    @property
    def rank(self) -> int:
        return self.direct_solver.rank

    @property
    def size(self) -> int:
        return self.direct_solver.size

    def describe(self) -> str:
        return "iterative flow (FGMRES/Schur) + direct MUMPS energy"

    def uses_local_flow_assembly(self) -> bool:
        return True

    def barrier(self) -> None:
        self.direct_solver.barrier()

    def broadcast(self, value, root: int = 0):
        return self.direct_solver.broadcast(value, root=root)

    def allreduce_max(self, value: float) -> float:
        return self.direct_solver.allreduce_max(value)

    def reduce_timing_max(self, timing):
        return self.direct_solver.reduce_timing_max(timing)

    def solve(self, matrix_or_system, rhs, system_type="flow_coupled", **kwargs):
        if system_type == "flow_coupled":
            result = self.flow_solver.solve(
                matrix_or_system, rhs, system_type=system_type, **kwargs
            )
            self.last_info = self.flow_solver.last_info
            return result
        result = self.direct_solver.solve(
            matrix_or_system, rhs, system_type=system_type, **kwargs
        )
        self.last_info = self.direct_solver.last_info
        return result

    def close(self) -> None:
        self.flow_solver.close()
        self.direct_solver.close()


def create_linear_solver(config: Optional[Dict[str, Any]] = None):
    cfg = config or {}
    mode = str(cfg.get("linear_solver_type", "mumps")).lower().strip()
    if mode in ("mumps", "direct", "direct_lu"):
        return PetscDirectSolver(cfg)
    if mode in ("iterative", "fgmres_schur", "schur"):
        return HybridFlowIterativeSolver(cfg)
    raise ValueError(
        "Unsupported linear_solver_type. Expected 'mumps' or 'iterative', "
        f"got {mode!r}."
    )
