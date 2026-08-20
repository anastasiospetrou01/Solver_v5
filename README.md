# Solver_v5 — Distributed Finite-Volume CFD Solver

**Solver_v5** is a research-oriented computational fluid dynamics (CFD) solver developed for steady, incompressible thermo-fluid simulations using the **finite-volume method (FVM)**.

The solver combines a custom CFD formulation with a modern high-performance computing stack built around **MPI**, **PETSc**, **MUMPS**, and **Numba**. It is designed as a transparent development platform for investigating pressure–velocity coupling, sparse linear systems, distributed-memory parallelism, thermal transport, buoyancy, and scalable CFD solver architecture.

> **Current status:** actively developed research code. The focus is numerical transparency, solver architecture, validation, and performance scaling rather than replacing established commercial CFD packages.

---

## Highlights

- Fully coupled **pressure–velocity** formulation
- Steady incompressible **Navier–Stokes** solution
- Finite-volume discretization on structured Cartesian grids
- Optional **energy equation**
- Optional **buoyancy** using the Boussinesq approximation
- First-order upwind and blended **second-order upwind (SOU)** convection
- Local boundedness limiter for higher-order reconstruction
- **Rhie–Chow-type pressure–velocity coupling**
- Sparse distributed linear systems through **PETSc**
- Parallel sparse direct solution using **MUMPS**
- Distributed-memory domain decomposition with **MPI**
- Local CFD fields with **halo / ghost-cell exchange**
- Persistent **fixed-COO PETSc matrix structure**
- Numba-compiled numerical kernels
- Runtime profiling using **MPI maximum-rank timing**
- Restart support and automated result/report generation
- Global mass- and energy-balance diagnostics

---

# Solver Architecture

The current solver is built around a distributed CFD workflow:

```text
                      Solver_v5
                          │
                          ▼
                Finite-Volume Discretization
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
   Momentum + Pressure                 Energy Equation
          │                                │
          ▼                                ▼
   Local MPI Assembly                Local MPI Assembly
          │                                │
          └───────────────┬────────────────┘
                          ▼
                Distributed PETSc Matrix
                          │
                          ▼
                  MUMPS Sparse Direct LU
                          │
                          ▼
               Local Solution Corrections
                          │
                          ▼
                 Halo / Ghost Exchange
                          │
                          ▼
                  Next Nonlinear Iteration
```

The computational domain is distributed between MPI ranks. Each rank stores and operates primarily on its own local subdomain together with a small halo region required for stencil operations.

---

# Parallel Computing

Solver_v5 currently uses a structured **y-slab decomposition**.

For example, a `400 × 400` grid running on 8 MPI ranks is approximately partitioned as:

```text
400 rows
│
├── Rank 0  → ~50 rows
├── Rank 1  → ~50 rows
├── Rank 2  → ~50 rows
├── Rank 3  → ~50 rows
├── Rank 4  → ~50 rows
├── Rank 5  → ~50 rows
├── Rank 6  → ~50 rows
└── Rank 7  → ~50 rows
```

Each rank maintains a **two-row halo**, allowing local evaluation of gradients, fluxes, higher-order reconstruction, and neighbouring-cell coefficients without globally replicating the CFD solution.

### Distributed components

The following are distributed across MPI ranks:

- velocity fields `u`, `v`
- pressure field `p`
- temperature field `T`
- momentum coefficients
- pressure gradients
- face fluxes
- residual calculations
- flow matrix rows
- energy matrix rows
- PETSc vectors and sparse matrices
- MUMPS direct factorization and solution

Global solution fields are gathered only when required for final output and reporting.

---

# Numerical Formulation

## Governing equations

The solver currently targets steady incompressible flow.

### Continuity

$$
\nabla \cdot \mathbf{u} = 0
$$

### Momentum

$$
\rho (\mathbf{u}\cdot\nabla)\mathbf{u}
=
-\nabla p
+
\mu\nabla^2\mathbf{u}
+
\mathbf{S}
$$

### Energy

$$
\rho c_p (\mathbf{u}\cdot\nabla T)
=
\nabla\cdot(k\nabla T)
+
\dot{q}
$$

For buoyant flows, a Boussinesq body-force contribution can be enabled.

---

# Pressure–Velocity Coupling

Velocity and pressure are assembled into one coupled sparse system using the ordering

```text
[u0, v0, p0, u1, v1, p1, ...]
```

The solver applies nonlinear under-relaxation and pressure–velocity coupling with Rhie–Chow-type face-flux treatment to avoid pressure–velocity decoupling on the collocated finite-volume grid.

---

# Spatial Discretization

Current convection options include:

- First-order upwind
- Second-order upwind reconstruction
- User-selectable SOU blending
- Local-bounds limiter

Typical configuration:

```python
"enable_sou_momentum": True,
"enable_sou_energy": True,
"sou_blend_momentum": 0.7,
"sou_blend_energy": 0.7,
"enable_sou_limiter": True,
```

---

# Linear Algebra Backend

Solver_v5 uses:

- **PETSc** for distributed sparse matrix/vector management
- **MUMPS** for parallel sparse direct LU factorization
- persistent KSP / PC / matrix objects
- fixed sparse structure with repeated numerical-value updates

The sparse structure is precomputed once and reused throughout the nonlinear simulation.

```text
Fixed sparsity pattern
        │
        ▼
Update numerical coefficients only
        │
        ▼
PETSc Mat.setValuesCOO(...)
        │
        ▼
Parallel MUMPS factorization
        │
        ▼
Distributed solution
```

This removes repeated Python sparse-matrix construction from the nonlinear loop.

---

# High-Performance Numerical Kernels

Performance-critical CFD operations are implemented using **Numba JIT compilation**.

Compiled kernels include:

- pressure gradients
- scalar gradients
- face mass fluxes
- higher-order reconstruction
- momentum coefficient assembly
- flow sparse-matrix value filling
- energy sparse-matrix value filling
- solution updates
- residual evaluation

The goal is to keep Python primarily responsible for orchestration while numerical loops execute as compiled code.

---

# MPI + Thread Configuration

Parallel execution is controlled directly from `solver/steady_lam_v5.py`.

Example:

```python
"mpi_ranks": 8,
"threads_per_rank": 1,
```

For an 8-core CPU this corresponds to a pure-MPI configuration using one physical core per rank.

Hybrid configurations are also supported:

```python
"mpi_ranks": 4,
"threads_per_rank": 2,
```

The launcher binds the requested processing elements to physical CPU cores.

Example binding:

```text
Rank 0 → cores 0–1
Rank 1 → cores 2–3
Rank 2 → cores 4–5
Rank 3 → cores 6–7
```

This enables controlled testing of pure-MPI and hybrid MPI/thread execution.

---

# Validated Benchmark Results

The solver has been regression-tested on canonical laminar CFD cases, including wake flow and buoyancy-driven cavity flow.

## 400 × 400 buoyancy cavity

| Parameter | Value |
|---|---:|
| Grid | `400 × 400` |
| Cells | `160,000` |
| Coupled flow DOF | `480,000` |
| MPI ranks | `8` |
| Threads/rank | `1` |
| Nonlinear iterations | `92` |
| Total runtime | **116.80 s** |
| Average time/iteration | **1.2696 s** |
| Final mass residual | `1.594 × 10⁻⁸` |
| Final temperature change | `9.919 × 10⁻⁷` |
| Tmin | `20.008895 °C` |
| Tmax | `29.991105 °C` |
| Relative energy-balance error | `~6.6 × 10⁻¹⁴` |

### Average MPI-max timing

```text
Outer iteration              1.2685 s
Flow total                   1.1207 s
Momentum pass                0.0220 s
Flow MUMPS factorization     0.9996 s
Flow MUMPS solve             0.0746 s
Energy total                 0.1467 s
Energy factorization         0.1128 s
Energy solve                 0.0210 s
```

The current dominant cost is the sparse direct LU factorization rather than CFD equation assembly.

---

## Parallel scaling example

Same `400 × 400` buoyancy cavity:

| Configuration | Total Runtime | Avg. Time / Iteration |
|---|---:|---:|
| 4 MPI × 1 thread | 154.71 s | 1.6817 s |
| 4 MPI × 2 threads | 121.23 s | 1.3177 s |
| **8 MPI × 1 thread** | **116.80 s** | **1.2696 s** |

This demonstrates that the distributed CFD architecture benefits from increased MPI decomposition, while hybrid MPI/thread execution is also supported.

---

# Performance Development

The solver has undergone several major optimization stages.

### Initial implementation

- Python-heavy finite-volume assembly
- replicated nonlinear CFD work across MPI ranks
- repeated sparse data-structure construction
- limited parallel scalability

### Performance refactor

Implemented:

- precomputed topology
- reusable numerical workspaces
- Numba CFD kernels
- fixed sparse patterns
- persistent PETSc matrices
- distributed local fields
- local matrix assembly
- halo communication
- distributed residual calculations
- local solution updates

On the validated `200 × 200` buoyancy cavity benchmark, the optimized solver reduced runtime from approximately:

```text
622.37 s
```

to:

```text
32.90 s
```

for the corresponding optimized benchmark configuration — an approximately **18.9× speedup** while preserving the validated numerical solution.

---

# Conservation Diagnostics

Every simulation can report global conservation quantities.

### Mass balance

```text
total inlet mass flow
total outlet mass flow
net imbalance
relative mass error
```

### Energy balance

```text
volumetric heat generation
conductive heat input/output
convective heat input/output
total Qin
total Qout
net energy imbalance
relative energy error
```

These diagnostics provide an additional validation layer beyond nonlinear residual convergence.

---

# Boundary Conditions

Current infrastructure supports numerical treatment for common CFD boundary types including:

### Flow

- wall
- velocity inlet
- pressure/open outlet
- symmetry
- open boundary

### Thermal

- prescribed temperature
- prescribed heat flux
- adiabatic
- symmetry
- outlet/open thermal treatment

---

# Case and Result Workflow

Cases are stored in NumPy-based `.npz` files.

Typical structure:

```text
case_files/
└── <case_name>/
    └── <case_name>.npz
```

A simulation is launched with:

```bash
python -m solver.steady_lam_v5
```

Solver settings are controlled from the `RUN_SETTINGS` dictionary.

Results are automatically written to:

```text
results/
└── <case_name>_steady_<timestamp>/
```

The solver also creates a solved `.npz` case containing the final CFD fields and metadata.

---

# Example Solver Configuration

```python
RUN_SETTINGS = {
    "case_name": "buoyancy_cavity_1x1",

    "mpi_ranks": 8,
    "threads_per_rank": 1,

    "max_iter": 200,
    "tol_mass": 1.0e-6,
    "tol_T": 1.0e-6,

    "enable_energy": True,
    "enable_buoyancy": True,

    "alpha_u": 0.4,
    "alpha_v": 0.4,
    "alpha_p": 0.2,
    "alpha_T": 0.3,

    "enable_sou_momentum": True,
    "enable_sou_energy": True,

    "direct_solver": {
        "solver_type": "mumps",
    },

    "performance": {
        "use_numba": True,
    },
}
```

---

# Core Technology Stack

| Technology | Role |
|---|---|
| Python | solver orchestration and research development |
| NumPy | numerical arrays and case data |
| Numba | compiled CFD kernels |
| MPI | distributed-memory parallel execution |
| PETSc / petsc4py | distributed sparse linear algebra |
| MUMPS | parallel sparse direct LU solver |
| Matplotlib | diagnostics and post-processing |

---

# Current Project Structure

```text
Solver_v5/
├── solver/
│   ├── __init__.py
│   └── steady_lam_v5.py
│
├── distributed_domain.py
├── flow_assembly.py
├── flow_scaling.py
├── linear_backend.py
├── numerical_kernels.py
├── solver_equations.py
├── solver_reporting.py
├── solver_utils.py
├── geometry.py
├── initial_conditions.py
├── materials.py
├── case_io.py
├── results_io.py
│
├── case_files/
├── case_builders/
├── results/
└── tests/
```

---

# Development Direction

Current and planned development areas include:

- scalable iterative alternatives to direct LU
- block preconditioning
- algebraic multigrid
- larger structured meshes
- improved multi-rank scaling
- more general domain decomposition
- unstructured-mesh support
- turbulence models
- transient formulation
- passive scalar/species transport
- expanded thermal models
- automated benchmark suites
- graphical pre/post-processing
- workstation, HPC, and cloud deployment

A major long-term objective is to evolve the solver from a research direct-solver architecture toward a scalable CFD platform capable of handling substantially larger systems.

---

# Design Philosophy

Solver_v5 is intentionally developed as more than a black-box CFD program.

The project is intended to expose and investigate the complete numerical pipeline:

```text
Governing equations
        ↓
Finite-volume discretization
        ↓
Pressure–velocity coupling
        ↓
Sparse matrix construction
        ↓
Parallel domain decomposition
        ↓
Distributed linear algebra
        ↓
Nonlinear convergence
        ↓
Conservation verification
        ↓
Performance profiling
```

This makes the code useful not only for CFD simulations, but also for research into **numerical methods, sparse solvers, parallel computing, and CFD software architecture**.

---

# Platform

Development is currently performed primarily under:

- Linux / Ubuntu
- WSL2 for Windows-based development
- Conda/Miniforge environments

The computational backend is based on technologies widely used in scientific and high-performance computing environments, making native Linux and HPC deployment natural future targets.

---

# Disclaimer

Solver_v5 is an actively developed **research CFD solver**. Numerical results should be independently verified before being used for safety-critical, regulatory, or commercial engineering decisions.

---

# Author / Project

Developed as an ongoing CFD and high-performance scientific-computing project focused on:

- finite-volume CFD
- pressure–velocity coupling
- sparse numerical linear algebra
- distributed-memory parallelism
- solver optimization
- thermal and buoyancy-driven flows
- scalable scientific software development
