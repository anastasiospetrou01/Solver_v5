# Fully Coupled CFD Solver V5

This directory contains a clean redesign of the steady finite-volume pressure–velocity solver. It keeps the monolithic `[u, v, p]` formulation and does not introduce a segregated pressure-correction solver.

## Implemented changes

### 1. Two-pass assembly

`flow_assembly.py` first computes the complete current momentum coefficients and diagonals. The continuity and Rhie–Chow pressure blocks are then assembled in a second pass using those current `aPu` and `aPv` values.

This removes the previous dependency of the Schur approximation on the preceding nonlinear iteration's momentum diagonal.

### 2. Correction-form coupled solve

The code now forms

```text
A delta_x = b - A x_old
```

and applies the nonlinear URFs to the correction:

```text
u <- u + alpha_u delta_u
v <- v + alpha_v delta_v
p <- p + alpha_p delta_p
```

For a direct solve with all URFs equal to one, this is algebraically equivalent to solving the absolute linearized system.

### 3. Physical two-sided scaling

`flow_scaling.py` applies

```text
(L A R) y = L r
delta_x = R y
```

with separate velocity, pressure, momentum-equation and continuity-equation scales. Automatic reference scales are derived from boundary velocities, the current field and fluid properties.

### 4. Pressure gauge handling

The pressure reference supports:

```text
auto
pin
nullspace
none
```

`auto` uses a PETSc pressure nullspace for closed incompressible systems, no gauge constraint when an open pressure boundary anchors the pressure, and a row pin for SciPy direct solves.

### 5. PETSc Schur strategies

`linear_backend.py` provides:

```text
current_diag_schur
lsc
pcd
augmented_pcd
block_amg_hypre
block_amg_gamg
```

The primary candidate is `pcd`.

The PCD Python preconditioner applies:

```text
Kp^-1 Fp Mp^-1
```

where:

- `Mp` is the pressure mass matrix;
- `Kp` is a density-weighted pressure Laplacian with Rhie–Chow stabilization;
- `Fp` is a pressure convection–diffusion operator built from current face fluxes and kinematic viscosity.

The LSC mode uses PETSc's Schur-complement LSC preconditioner as a prototype. PETSc's current LSC implementation ignores a nonzero `A11`, so it should be treated as a comparison mode rather than the expected final production option for this collocated formulation.

### 6. Coupled block AMG alternatives

Two experimental whole-system AMG modes are included:

```text
block_amg_hypre
block_amg_gamg
```

Both retain the interleaved three-variable block per cell. The HYPRE mode uses nodal/system coarsening. The GAMG mode uses additive Schwarz with local ILU smoothing on multigrid levels.

These are experimental because general AMG is not automatically robust for an indefinite saddle-point system. They must be compared against PCD and augmented PCD.

### 7. Augmented-Lagrangian option

`augmented_pcd` applies an exact block row operation before solving:

```text
F_aug = F + gamma G Mp^-1 D
G_aug = G + gamma G Mp^-1 C
r_u_aug = r_u + gamma G Mp^-1 r_p
```

The linear-system solution is unchanged. The default is `gamma = 1.0`.

### 8. DMDA / COMM_WORLD infrastructure

`dmda_flow.py` creates a structured PETSc `DMDA` with three DOFs per cell and solves the fully coupled matrix using block AMG on `COMM_WORLD`.

The included `CompactSystemDMDAAdapter` is a validation bridge: PETSc matrix/vector storage and AMG are distributed, but the compact SciPy assembly is still created before distribution. It is not yet the final million-cell assembly path.

The final phase-6 integration is to replace this adapter with direct local finite-volume insertion into the DMDA matrix and DMDA-owned field/material vectors. This separation is intentional so the new algebra and preconditioners can be validated before changing data ownership and case I/O simultaneously.

## Validation completed here

The supplied test suite verifies:

1. correction form and absolute form give the same direct solution when URFs are one;
2. the new two-pass direct solver produces finite fields and very small true linear residuals on a small cavity case;
3. all supplied Python modules compile.

PETSc, HYPRE and MPI execution could not be run in the artifact environment because `petsc4py` is not installed there. The PETSc paths therefore require validation in the existing `cfd-petsc` environment.

Run the local tests:

```bash
cd "/home/anast/Solver - v3/solver"
bash run_tests.sh
```

## Installation into the project

Back up the current solver directory. Copy these files into the project's `solver` directory:

```text
flow_assembly.py
flow_preconditioners.py
flow_scaling.py
linear_backend.py
solver_equations.py
dmda_flow.py
steady_lam_v5.py
```

The other included modules are copies of the current support modules for completeness.

Run the new steady solver:

```bash
cd "/home/anast/Solver - v3"
conda activate cfd-petsc

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python -m solver.steady_lam_v5
```

Do **not** launch `steady_lam_v5.py` under MPI yet. The current main solver uses the compact serial validation path (`COMM_SELF`). `dmda_flow.py` contains the separate `COMM_WORLD` validation adapter, but direct rank-local finite-volume assembly is still the remaining phase-6 integration milestone.

## Initial solver settings

Primary test:

```python
"strategy": "pcd",
"fallback_to_direct_on_failure": False,

"alpha_u": 0.7,
"alpha_v": 0.7,
"alpha_p": 0.3,
```

Keep the same URFs that make the obstacle case converge with direct LU. The linear solver must solve each resulting coupled correction system without case-specific KSP tuning.

If PCD cannot solve the first obstacle iteration, change only:

```python
"strategy": "augmented_pcd",
"augmented_gamma": 1.0,
```

Then test:

```python
"augmented_gamma": 10.0
```

Do not loosen the true-residual acceptance criterion to hide a failed linear solve.

After PCD or augmented PCD works, compare the block AMG modes:

```python
"strategy": "block_amg_hypre"
```

and:

```python
"strategy": "block_amg_gamg"
```

## Acceptance criteria

A candidate production strategy must:

- complete every nonlinear iteration of the obstacle case;
- require no direct-LU fallback;
- use the same nonlinear URFs as the direct-LU run;
- satisfy the unscaled original-system residual check;
- retain bounded linear iteration counts under grid refinement;
- converge on the cavity, obstacle and buoyancy cases;
- later run through direct local DMDA assembly on `COMM_WORLD`.
