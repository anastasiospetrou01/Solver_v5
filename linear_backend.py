from __future__ import annotations

"""Persistent PETSc/MUMPS direct solver for serial and MPI execution."""

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


def _deep_update(
    base: Dict[str, Any], override: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
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


def _true_residual(
    matrix,
    rhs,
    solution,
    norm_type: str = "inf",
) -> Tuple[float, float]:
    matrix_csr = sparse.csr_matrix(matrix)
    rhs_array = np.asarray(rhs, dtype=float).reshape(-1)
    x = np.asarray(solution, dtype=float).reshape(-1)
    residual = rhs_array - matrix_csr @ x

    if str(norm_type).lower() in ("inf", "linf", "infinity"):
        absolute = float(np.max(np.abs(residual))) if residual.size else 0.0
        rhs_norm = (
            float(max(np.max(np.abs(rhs_array)), 1.0e-30))
            if rhs_array.size
            else 1.0
        )
    else:
        absolute = float(np.linalg.norm(residual))
        rhs_norm = float(max(np.linalg.norm(rhs_array), 1.0e-30))
    return absolute / rhs_norm, absolute


class PetscDirectSolver:
    """Persistent sparse direct solver using PETSc LU and MUMPS.

    With one MPI rank the solver uses ``PETSc.COMM_SELF`` and receives a global
    CSR matrix. With multiple ranks it uses ``PETSc.COMM_WORLD``. Coupled flow
    rows are then assembled locally by ``flow_assembly.py``; scalar systems such
    as energy are currently assembled globally on every rank and only their
    owned rows are inserted into PETSc.
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
        return self.size > 1

    def barrier(self) -> None:
        self.PETSc.COMM_WORLD.barrier()

    def broadcast(self, value, root: int = 0):
        try:
            return self.PETSc.COMM_WORLD.tompi4py().bcast(value, root=root)
        except Exception as exc:
            if self.size > 1:
                raise RuntimeError(
                    "mpi4py is required for MPI control-data broadcasts."
                ) from exc
            return value

    def describe(self) -> str:
        mode = "distributed local-row assembly" if self.size > 1 else "serial CSR assembly"
        return (
            f"PETSc direct LU ({self.config.get('solver_type', 'mumps')}, "
            f"{mode})"
        )

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

    def _set_factor_options(self, options: Dict[str, Any], prefix: str) -> None:
        self._set_option(f"{prefix}ksp_type", "preonly")
        self._set_option(f"{prefix}pc_type", "lu")
        self._set_option(
            f"{prefix}pc_factor_mat_solver_type",
            options.get("solver_type", "mumps"),
        )
        self._set_option(
            f"{prefix}pc_factor_reuse_ordering",
            str(bool(options.get("reuse_ordering", True))).lower(),
        )
        self._set_option(
            f"{prefix}pc_factor_reuse_fill",
            str(bool(options.get("reuse_fill", True))).lower(),
        )

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
        pattern_signature,
        preallocation_nnz: int,
        options: Dict[str, Any],
        comm,
    ) -> _PetscDirectCache:
        PETSc = self.PETSc
        local_size = self._local_size(global_size, block_size, comm)

        matrix = PETSc.Mat().create(comm=comm)
        matrix.setSizes(
            ((local_size, global_size), (local_size, global_size))
        )
        matrix.setType(PETSc.Mat.Type.AIJ)
        matrix.setBlockSize(int(block_size))
        matrix.setPreallocationNNZ(max(int(preallocation_nnz), 1))
        matrix.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        matrix.setUp()

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
            pc.setFactorOrdering(
                None, reuse=bool(options.get("reuse_ordering", True))
            )
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
        )

    @staticmethod
    def _destroy_cache(cache: _PetscDirectCache) -> None:
        for obj in (cache.ksp, cache.mat, cache.rhs, cache.solution):
            try:
                obj.destroy()
            except Exception:
                pass

    def _get_cache(self, matrix_or_system, system_type: str, options, metadata):
        is_local = bool(
            getattr(matrix_or_system, "is_distributed_local", False)
        )
        if is_local:
            global_size = int(matrix_or_system.global_size)
            block_size = int(
                getattr(
                    matrix_or_system,
                    "block_size",
                    metadata.get("block_size", 1),
                )
            )
            signature = getattr(
                matrix_or_system,
                "pattern_key",
                (global_size, block_size, "local"),
            )
            preallocation = int(
                getattr(
                    matrix_or_system,
                    "preallocation_nnz",
                    options.get("preallocation_nnz", 20),
                )
            )
            matrix_csr = None
        else:
            matrix_csr = self._as_csr(matrix_or_system)
            if matrix_csr.shape[0] != matrix_csr.shape[1]:
                raise ValueError("Direct LU requires a square matrix.")
            global_size = int(matrix_csr.shape[0])
            block_size = int(metadata.get("block_size", 1))
            signature = self._pattern_signature(matrix_csr)
            row_nnz = np.diff(matrix_csr.indptr)
            preallocation = int(
                max(np.max(row_nnz) if row_nnz.size else 1, 1)
            )

        comm = self.comm
        key = (
            system_type,
            int(comm.getSize()),
            global_size,
            block_size,
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
                pattern_signature=signature,
                preallocation_nnz=preallocation,
                options=options,
                comm=comm,
            )
            self._caches[key] = cache
        return cache, is_local, matrix_csr

    def _fill_from_global_csr(
        self,
        cache: _PetscDirectCache,
        matrix_csr: sparse.csr_matrix,
        rhs_global,
    ) -> None:
        PETSc = self.PETSc
        matrix = cache.mat
        matrix.zeroEntries()
        row_start, row_end = matrix.getOwnershipRange()

        for row in range(row_start, row_end):
            start = int(matrix_csr.indptr[row])
            end = int(matrix_csr.indptr[row + 1])
            if end <= start:
                continue
            matrix.setValues(
                np.asarray([row], dtype=PETSc.IntType),
                np.asarray(
                    matrix_csr.indices[start:end], dtype=PETSc.IntType
                ),
                np.asarray(
                    matrix_csr.data[start:end], dtype=PETSc.ScalarType
                ).reshape(1, -1),
            )
        matrix.assemble()

        rhs_array = cache.rhs.getArray()
        rhs_array[:] = np.asarray(rhs_global, dtype=float)[row_start:row_end]
        cache.rhs.assemble()
        cache.solution.set(0.0)

    def _gather_solution(self, vector, comm) -> np.ndarray:
        if int(comm.getSize()) == 1:
            return np.asarray(
                vector.getArray(readonly=True), dtype=float
            ).copy()
        scatter, sequential = self.PETSc.Scatter.toAll(vector)
        scatter.scatter(vector, sequential)
        result = np.asarray(
            sequential.getArray(readonly=True), dtype=float
        ).copy()
        scatter.destroy()
        sequential.destroy()
        return result

    def solve(
        self,
        matrix_or_system,
        rhs,
        system_type: str = "flow_coupled",
        x0=None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> np.ndarray:
        del x0  # Direct LU always starts from an exact factorization solve.
        metadata = metadata or {}
        options = self._system_options(system_type)
        timing: Dict[str, float] = {}
        total_start = time.perf_counter()

        cache, is_local, matrix_csr = self._get_cache(
            matrix_or_system, system_type, options, metadata
        )

        update_start = time.perf_counter()
        local_stats: Dict[str, float] = {}
        if is_local:
            local_stats = dict(
                matrix_or_system.assemble_petsc(cache.mat, cache.rhs)
            )
            cache.solution.set(0.0)
        else:
            self._fill_from_global_csr(cache, matrix_csr, rhs)
        timing["matrix_rhs_update"] = time.perf_counter() - update_start

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
        solution = self._gather_solution(cache.solution, cache.comm)
        timing["solution_gather"] = time.perf_counter() - gather_start

        residual_start = time.perf_counter()
        if is_local:
            metrics = matrix_or_system.distributed_residual_metrics(
                cache.mat, cache.rhs, cache.solution
            )
        else:
            true_matrix = metadata.get("true_matrix", matrix_csr)
            true_rhs = metadata.get("true_rhs", rhs)
            scaled_rel, scaled_abs = _true_residual(
                true_matrix,
                true_rhs,
                solution,
                options.get("true_residual_norm", "inf"),
            )

            unscaled_rel = None
            unscaled_abs = None
            scaling = metadata.get("scaling")
            unscaled_matrix = metadata.get("unscaled_true_matrix")
            unscaled_rhs = metadata.get("unscaled_true_rhs")
            if (
                scaling is not None
                and unscaled_matrix is not None
                and unscaled_rhs is not None
            ):
                physical_solution = scaling.unscale_solution(solution)
                unscaled_rel, unscaled_abs = _true_residual(
                    unscaled_matrix,
                    unscaled_rhs,
                    physical_solution,
                    options.get("true_residual_norm", "inf"),
                )

            metrics = {
                "scaled_true_rel_residual": scaled_rel,
                "scaled_true_abs_residual": scaled_abs,
                "unscaled_true_rel_residual": unscaled_rel,
                "unscaled_true_abs_residual": unscaled_abs,
                "true_rel_residual": max(
                    scaled_rel,
                    unscaled_rel if unscaled_rel is not None else scaled_rel,
                ),
                "true_abs_residual": max(
                    scaled_abs,
                    unscaled_abs if unscaled_abs is not None else scaled_abs,
                ),
            }
        timing["true_residual_check"] = time.perf_counter() - residual_start

        tolerance = float(options.get("true_residual_tolerance", 1.0e-5))
        true_ok = float(metrics["true_rel_residual"]) <= tolerance
        converged = (
            reason > 0
            and pc_failed == 0
            and true_ok
            and np.all(np.isfinite(solution))
        )

        factor_solver = str(options.get("solver_type", "mumps"))
        factor_nnz = None
        factor_memory = None
        try:
            factor_solver = str(pc.getFactorSolverType())
            factor = pc.getFactorMatrix()
            factor_info = factor.getInfo()
            factor_nnz = float(
                factor_info.get(
                    "nz_used", factor_info.get("nz_allocated", 0.0)
                )
            )
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
            message=(
                "PETSc direct LU completed."
                if converged
                else "PETSc direct LU failed."
            ),
            extra={
                "strategy": "direct_lu",
                "direct_solver_type": factor_solver,
                "mpi_size": int(cache.comm.getSize()),
                "persistent": True,
                "local_assembly": is_local,
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

        if (
            int(cache.comm.getRank()) == 0
            and bool(
                options.get(
                    "print_diagnostics", self.config.get("verbose", True)
                )
            )
        ):
            unscaled_value = metrics.get("unscaled_true_rel_residual")
            print(
                "    PETSc direct LU | "
                f"solver={factor_solver} | mpi={int(cache.comm.getSize())} | "
                f"reason={reason} | pcFailed={pc_failed} | "
                f"scaledTrueRel={float(metrics['scaled_true_rel_residual']):.3e} | "
                f"unscaledTrueRel={float(unscaled_value) if unscaled_value is not None else float('nan'):.3e} | "
                f"allowed={tolerance:.3e} | persistent=True | "
                f"localAssembly={is_local}"
            )

        if not converged and bool(
            self.config.get("error_on_nonconvergence", True)
        ):
            raise RuntimeError(
                "PETSc direct LU failed: "
                f"solver={factor_solver}, reason={reason}, "
                f"pc_failed={pc_failed}, true relative residual="
                f"{float(metrics['true_rel_residual']):.6e}, "
                f"allowed={tolerance:.6e}."
            )
        return solution

    def close(self) -> None:
        for cache in list(self._caches.values()):
            self._destroy_cache(cache)
        self._caches.clear()


def create_linear_solver(
    config: Optional[Dict[str, Any]] = None,
) -> PetscDirectSolver:
    """Create the only production linear solver: persistent PETSc/MUMPS LU."""
    return PetscDirectSolver(config)