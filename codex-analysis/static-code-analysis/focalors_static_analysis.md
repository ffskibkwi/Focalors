# Focalors Static Code Analysis Report

Scope note: this report is based on static source inspection only. No project code was compiled, linked, or executed. The `case/` directory was used as the primary entry index, and the algorithmic cross-reference uses the paper source in `codex-analysis/ref-files/main.tex` with `codex-analysis/ref-files/main.pdf` treated as its rendered companion.

## 1. Introduction & Repository Topology

### Project Positioning

Focalors is a C++17 incompressible-flow solver whose numerical center of gravity is a low-complexity Poisson/Helmholtz-Poisson solver for domains made by stitching axis-aligned rectangles or rectangular boxes. The paper describes the key mathematical idea: decompose a composite geometry into rectangular subdomains, solve each regular-domain block with FFT-accelerated direct solvers, and couple the subdomains through Schur-complement interface equations solved by a preconditioned GMRES iteration.

The repository has grown this Poisson core into a broader CFD framework:

- FFT-based Poisson solvers for single regular 2D/3D domains.
- Concatenated-domain Poisson solvers for stitched geometries.
- Incompressible Navier-Stokes pressure-projection pipelines.
- Scalar transport, physical pressure-equation solvers, MHD coupling, immersed-boundary components, particle properties, and VTK/CSV/savepoint IO.
- MPI/OpenMP variants, especially slab-oriented Poisson/domain decomposition support.

The practical instantiation pattern is visible in `case/`: define domains, stitch them into a `Geometry*`, bind semantic variables to per-domain fields, configure boundary conditions, construct a solver, then run a time loop or precision/benchmark loop.

### Code Directory Topology

| Path | Responsibility | Interaction Pattern |
| --- | --- | --- |
| `case/` | Executable examples, validation cases, benchmarks, precision studies. | Each `.cpp` is turned into an executable by `case/CMakeLists.txt`; cases instantiate domains, variables, solvers, and IO. |
| `poisson_base/` | Modular Poisson core, corresponding to the solver submodule. | Provides domain/geometry/field abstractions, FFT Poisson kernels, Schur/GMRES stitched-domain assembly, MPI/OpenMP helpers. |
| `poisson_base/base/` | Semantic primitives: fields, domains, geometry graphs/trees, variables, boundaries, shapes, math helpers, parallel macros. | Used by both pure Poisson solvers and higher CFD modules. |
| `poisson_base/pe/poisson/` | Single-rectangle FFT Poisson/Helmholtz-Poisson solvers. | Implements transform selection by boundary type, FFT workspaces, eigenvalue construction, and tridiagonal chasing in the remaining axis. |
| `poisson_base/pe/concat/` | Stitched-domain Poisson solver. | Builds subdomain solver tree, Schur matrices, and GMRES interface solves. |
| `poisson_base/pe/parallel/` | MPI slab/pencil transpose and parallel Poisson variants. | Extends field/domain/solver abstractions for distributed decomposition. |
| `ns/` | Navier-Stokes and scalar transport solvers. | Uses `Variable*` and concatenated Poisson solvers for pressure projection and velocity/scalar updates. |
| `ibm_Uhlmann/`, `ibm_MirrorPoint/` | Immersed-boundary implementations. | Use particle coordinate maps and variable field pointers to interpolate/spread forces or enforce solid velocities. |
| `particle/` | Particle property containers and particle-to-domain maps. | Stores property arrays and maps particles onto domain-indexed lookup structures. |
| `io/` | Case base, CSV/VTK writers, parameter readers, savepoint utilities. | Case files use these modules to ingest parameters and emit fields. |
| `codex-analysis/ref-files/` | Paper reference files. | `main.tex` documents the FFT-Schur-GMRES method implemented in `poisson_base/pe`. |

### Build & Compilation Environment

The top-level `CMakeLists.txt` and `poisson_base/CMakeLists.txt` both require CMake 3.10 and C++17:

- `set(CMAKE_CXX_STANDARD 17)`
- `set(CMAKE_CXX_STANDARD_REQUIRED True)`

The configured build modes are:

- `Debug`: `-g -O0`
- `Hybrid`: `-g -O3 -DHYBRID -DNDEBUG`
- `PureMPI`: `-g -O3 -DPUREMPI -DNDEBUG`

External dependencies include FFTW, OpenMP, MPI, and VTK. The top-level `Solver` library globs sources from `poisson_base`, `ns`, `ibm_*`, `particle`, and `io`, then links against FFTW, `OpenMP::OpenMP_CXX`, `MPI::MPI_CXX`, and VTK.

OpenMP is integrated through `poisson_base/base/parallel/omp/enable_openmp.h`. The macro `OPENMP_PARALLEL_FOR(...)` expands to `_Pragma("omp parallel for schedule(dynamic) ...")` only when `HYBRID` is defined; otherwise it expands to nothing. This means the same source compiles as serial loop code in `Debug` and `PureMPI` configurations unless direct OpenMP calls are used.

## 2. Semantic Data Structures & Memory Topology

### Core Data Objects

| Object Family | Main Files | Meaning |
| --- | --- | --- |
| Dense fields | `poisson_base/base/field/field2.*`, `field3.*` | Contiguous 2D/3D scalar arrays used for physical variables, RHS buffers, transform buffers, and temporary fields. |
| Domains | `poisson_base/base/domain/domain2d.*`, `domain3d.*`, MPI variants | Rectangular semantic blocks with dimensions, physical lengths, spacing, boundary types, offsets, and a back-pointer to owning geometry. |
| Geometry graphs | `poisson_base/base/geometry/geometry2d.*`, `geometry3d.*` | Stitched-domain graph plus tree decomposition used by concatenated solvers. |
| Variables | `poisson_base/base/variable/variable2d.*`, `variable3d.*`, slab variants | Semantic physical variables attached to a geometry; own boundary and inter-domain buffers while borrowing user field storage. |
| Poisson kernels | `poisson_base/pe/poisson/*` | FFT workspaces, FFTW plans, tridiagonal chasing objects, and boundary-dependent eigenvalue transforms. |
| Concatenated solvers | `poisson_base/pe/concat/*` | Own temporary fields, child solvers, Schur matrices, and GMRES work arrays for stitched geometries. |
| CFD solvers | `ns/*` | Orchestrate velocity/scalar predictor-corrector steps, pressure solves, MHD, and boundary updates. |
| Particles/IBM | `particle/*`, `ibm_Uhlmann/*`, `ibm_MirrorPoint/*` | Particle property arrays, particle-domain maps, immersed-boundary force/interpolation logic. |

### Data Members & Memory Layout

`field2` owns one contiguous `double* value` buffer with logical shape `(nx, ny)` and index expression `value[i * ny + j]`. `field3` owns one contiguous `double* value` buffer with logical shape `(nx, ny, nz)` and index expression `value[i * ny * nz + j * nz + k]`. These field classes are the only broadly used data containers with flat dense memory.

`Domain2DUniform` and `Domain3DUniform` are descriptors rather than containers. They store sizes, spacings, lengths, offsets, names, boundary-type metadata, and a raw `parent` pointer back to the geometry. They do not own heap memory.

`Geometry2D` and `Geometry3D` store raw domain pointers in vectors and maps:

- `std::vector<Domain*> domains`
- adjacency maps keyed by `Domain*`
- tree maps, parent maps, hierarchical solve levels

The geometry does not allocate or delete domains. Cases commonly put domains on the stack and pass their addresses into `add_domain()`.

`Variable2D` and `Variable3D` borrow the actual physical fields through `field_map<Domain*, field*>`. They do own the auxiliary boundary/interface storage:

- `Variable2D`: face buffers are raw `double*`; boundary values are raw `double*`.
- `Variable3D`: face buffers and boundary values are heap-allocated `field2*` planes; some face-centered corner data use raw `double*` arrays.

Poisson FFT classes own FFTW plans and workspace arrays. The 2D classes store row-wise `double**`/`fftw_complex**` style buffers; 3D classes use analogous nested workspaces. The base Poisson solver owns transform objects, tridiagonal/chasing helpers, diagonal arrays, and dense temporary fields.

`ConcatPoissonSolver2D/3D` borrow the external `Variable*`, copy the variable's `field_map` pointers, and allocate internal `temp_fields` plus solver objects per subdomain. `GMRESSolver2D/3D` own a local rectangular Poisson solver, Schur-matrix objects, Krylov vectors, and multiple mutable field buffers.

### Lifecycle Management

The repository uses a mixed ownership model:

- Dense fields usually own their internal numeric buffer.
- Domains are usually stack-owned by case code.
- Geometries borrow domain pointers and do not delete domains.
- Variables borrow geometry and physical field pointers, but own boundary and exchange buffers.
- Concatenated Poisson solvers borrow variables and physical fields, but own temporary fields and child solver objects.
- CFD solvers usually borrow variables and pressure solvers, but some allocate temporary fields internally.
- MHD code is more modern: `MHDModule2D` uses `std::unique_ptr` for owned variables, solvers, and fields.
- Particle base objects own property arrays and are explicitly non-copyable.

This split is workable but fragile. Future code should treat the case-level objects as the root lifetime owner: domains and fields must outlive geometries, variables, and solvers that store their addresses. Solver objects are not safely reusable after the geometry/variable/field topology is mutated.

### Copy Behaviors: Deep vs. Shallow

| Type | Copy Behavior | Developer Warning |
| --- | --- | --- |
| `field2` | Has deep copy constructor and deep copy assignment; move swaps ownership. | Generally safe to copy, but repeated arithmetic can allocate temporaries. |
| `field3` | Has copy assignment but no explicit copy constructor; because move operations exist, copy construction is effectively disabled. Assignment does not resize destination storage. | `field3 dst; dst = src;` leaves `dst` empty. Assignment between mismatched shapes risks partial copies or out-of-bounds reads. |
| `Domain2DUniform`, `Domain3DUniform` | Default shallow copy of metadata and `parent` pointer. | Copying a domain after it is attached to geometry can leave confusing parent/graph relationships. |
| `Geometry2D`, `Geometry3D` | Default shallow copy of raw domain pointers and maps. | Do not copy. A copied geometry points to the same domains, but domains still point to the original geometry. |
| `Variable2D`, `Variable3D`, `Variable2DSlabX` | No custom copy/move controls; default copy shallow-copies owning buffer pointers. | High double-free risk. Direct assignment or pass-by-value is unsafe. |
| `PoissonFFT*`, `PoissonSolver*Base`, `PoissonSolver2D/3D` | Own raw FFTW plans/workspaces/helpers; copy is not disabled. | Default copy can double-destroy plans and buffers. Treat as non-copyable. |
| `ConcatPoissonSolver2D/3D` | Own temp fields and solver pointers; copy is not disabled. | Default copy can double-delete `temp_fields` and `solver_map` entries. |
| `GMRESSolver2D/3D` | Own local solver and Schur matrix pointers; copy is not disabled. | Default copy can double-delete `pe_solver` and `S_params`. |
| `SchurMat2D/3D` | Owns dense `field2 value`; derived classes currently add no extra resources. | Base classes are polymorphic but lack an explicit virtual destructor; deleting through base pointer is structurally unsafe if derived classes later own resources. |
| `ConcatNSSolver2D/3D` | Constructors allocate temporary field maps; no destructor was found in the inspected headers/sources. | Leaks internal temp fields and is unsafe to copy. |
| `ScalarSolver2D/3D` | Constructors allocate scalar temp field maps; no destructor was found. | Leaks internal temp fields and is unsafe to copy. |
| `ParticlesBase` | Inherits `NonCopyable`; owns `double*` property arrays and deletes them. | Safer model. Preserve this pattern for particle extensions. |
| `PCoordMap2D/3D` | Owns vectors/maps of particle-coordinate pointers; copy is not disabled. | Default copy can double-delete map entries. Returned raw-pointer maps are only valid while the map object lives. |
| `IBVelocitySolver*`, `IBScalarSolver*` | Constructors allocate particle-property maps; no destructor was found. | Likely leaks `PIB*` entries. Also has OpenMP write hazards discussed below. |

The main rule for secondary development: any class with raw owning pointers should either explicitly delete copy/assignment or be converted to RAII containers such as `std::vector`, `std::unique_ptr`, and value-owned fields. The most urgent copy-danger classes are `Variable2D`, `Variable3D`, `PoissonFFT*`, `PoissonSolver*`, `ConcatPoissonSolver*`, `GMRESSolver*`, and `PCoordMap*`.

## 3. Algorithmic Logic & Calculation Pipelines

### Main Computational Flow

A representative Poisson-only case, such as `case/precision/2d/five_domain_pe.cpp`, follows this flow:

1. Construct several `Domain2DUniform` objects.
2. Add them to a `Geometry2D`.
3. Stitch neighboring domains with `geo.connect(...)`.
4. Set global spacing with `geo.set_global_spatial_step(...)`.
5. Construct a semantic pressure variable `Variable2D p`.
6. Attach one `field2` per domain with `p.set_center_field(domain, field)`.
7. Set physical boundary types and values.
8. Construct `ConcatPoissonSolver2D solver(&p)`.
9. Fill RHS fields, call `solver.solve()`, then inspect/write field values.

A representative CFD benchmark, such as `case/benchmark/2d/cross_shaped_channel.cpp`, extends the same assembly:

1. Build stitched geometry and pressure/velocity variables.
2. Attach staggered or centered fields to each domain.
3. Configure pressure and velocity boundary conditions.
4. Construct `ConcatPoissonSolver2D p_solver(&p)`.
5. Construct `ConcatNSSolver2D ns_solver(&u, &v, &p, &p_solver)`.
6. Optionally initialize MHD with `ns_solver.init_mhd(&phi)`.
7. In the time loop, call `ns_solver.solve()`, update time-dependent boundaries, and write output.

In 3D validation cases, the pattern is analogous: create `Geometry3D`, `Variable3D` objects for `u/v/w/p`, attach `field3` data, then use `ConcatPoissonSolver3D` and optional physical pressure-equation solvers.

### Key Algorithm-to-Code Mapping

| Paper Concept | Mathematical/Logical Definition | Code Implementation Anchor |
| --- | --- | --- |
| Regular-domain Poisson solve | Discrete operator is diagonalized along transform directions; the remaining direction is solved as independent tridiagonal systems. | `PoissonSolver2D::solve(field2&)` in `poisson_base/pe/poisson/poisson_solver2d.cpp`; `PoissonSolver3D::solve(field3&)` in `poisson_solver3d.cpp`. |
| Boundary-specific transforms | Dirichlet, Neumann, periodic, and mixed boundary pairs map to different low-complexity transform matrices and eigenvalues. | `PoissonSolver2DBase::create_fft`, `PoissonSolver3DBase::create_fft`, and `cal_lambda` in `poisson_solver2d.cpp` / `poisson_solver3d.cpp`; concrete transform classes in `poisson_fft2d.cpp` and `poisson_fft3d.cpp`. |
| Tridiagonal line solve | After FFT diagonalization, each spectral line solves an x-direction tridiagonal or periodic/singular variant. | `ChasingMethod2D::chasing` and `ChasingMethod3D::chasing` in `poisson_base/pe/poisson/chasing_method*.cpp`; base routines in `chasing_method_base.cpp`. |
| Domain decomposition | Each rectangle/box is an `A_i` block; neighboring interfaces contribute coupling operators `R_ij`. | Geometry adjacency and tree assembly in `Geometry2D::solve_prepare`, `Geometry3D::solve_prepare`, and `TreeUtils`. |
| Schur complement | Interface response matrices are assembled as `S_ij = R_ij A_i^{-1} R_ji`. | `SchurMat2D*::construct` in `poisson_base/pe/concat/schur_mat2d.cpp`; `SchurMat3D*::construct` in `schur_mat3d.cpp`. |
| Preconditioned GMRES | The coupled domain solves `(I - A^{-1}S)x = A^{-1}f'`, matching the paper's FFT-preconditioned GMRES. | `GMRESSolver2D::Afun`, `GMRESSolver2D::solve`; `GMRESSolver3D::Afun`, `GMRESSolver3D::solve`. |
| Hierarchical concatenation | Leaf/branch domains are solved upward into modified RHS values; root is solved with Schur/GMRES; results propagate downward. | `ConcatPoissonSolver2D::construct_solver_map_at_domain`, `ConcatPoissonSolver2D::solve`; 3D analogues in `concat_poisson_solver3d.cpp`. |
| Pressure projection CFD | Predictor velocity update, pressure RHS/divergence, pressure solve, pressure-gradient correction. | `ConcatNSSolver2D::solve` / `ConcatNSSolver3D::solve` in `ns/concat_ns_solver*.cpp`. |
| Physical pressure equation | RHS derived from velocity-gradient invariants, followed by pressure Poisson solve. | `PhysicalPESolver2D::solve`, `PhysicalPESolver3D::solve` in `ns/physical_pe_solver*.cpp`. |

### FFT-Based Stitched Geometry Pipeline

The source follows the paper's algorithmic structure closely:

1. `ConcatPoissonSolver*` calls `boundary_assembly()` to insert physical and adjacent-domain boundary contributions into each subdomain RHS.
2. RHS fields are scaled by `hx * hx`, matching the non-dimensionalized discrete operator used by the internal Poisson kernels.
3. The solver tree is traversed from leaves toward the root. For each child domain, its temporary solution is computed and its interface response is subtracted from the parent RHS through `bond_add(...)`.
4. A root or branching domain with children uses `GMRESSolver*`, not a plain `PoissonSolver*`.
5. `GMRESSolver*::solve` first applies the rectangular Poisson inverse to the RHS. Its matrix-vector product computes Schur interface terms, applies the rectangular inverse again, and returns `x - A^{-1}Sx`.
6. After the root solve, the solution is propagated back down the tree. Each branch subtracts the known parent interface data and solves its own rectangular or GMRES-coupled problem.

This is the code-level realization of the paper's leaf-solve, coupled-center GMRES, and branch-solve algorithm. The implementation generalizes the paper's cross-shaped demonstration by using geometry-derived trees, not a hard-coded five-domain formula.

### Control Flow & Loop Analysis

The computation-heavy zones are:

- FFT transform loops in `poisson_fft2d.cpp` and `poisson_fft3d.cpp`: outer loops over independent rows/planes call FFTW plans over contiguous or workspace-copied slices.
- Tridiagonal chasing in `chasing_method2d.cpp` and `chasing_method3d.cpp`: outer spectral-line loops are independent; each line has forward/backward sweeps with strong data dependency along the line.
- Schur construction in `schur_mat2d.cpp` and `schur_mat3d.cpp`: for each interface basis vector, the code injects a unit boundary signal, solves a child rectangular problem, then records the interface response. This can be expensive because it performs many Poisson solves during setup.
- GMRES in `gmres_solver2d.cpp` and `gmres_solver3d.cpp`: nested restart/iteration loops repeatedly compute `Afun`, orthogonalize Krylov vectors, apply Givens rotations, and update the solution field.
- Navier-Stokes and scalar stencils in `ns/`: large nested loops over grid cells update predictor velocities, divergence, scalar advection-diffusion, pressure gradients, and boundary values.

Spatial locality is strongest inside flat `field2/field3` scans and line solves. It is weaker at module boundaries where maps keyed by `Domain*`, per-face buffer arrays, and tree traversal determine what field or interface is touched next.

## 4. Parallel Design & Thread Safety

### OpenMP Annotation Map

OpenMP parallelism is mostly routed through `OPENMP_PARALLEL_FOR(...)`, which becomes an OpenMP parallel `for` only under the `HYBRID` compile definition.

| Area | Files / Functions | Parallel Pattern |
| --- | --- | --- |
| Field operations | `field2.cpp`, `field3.cpp` | Parallel dense loops for clear, arithmetic, assignment, transpose, and slice operations. |
| Variables | `variable2d.cpp`, `variable3d.cpp`, `variable2d_slab_x.cpp` | Parallel value initialization/copy over domain fields or boundary buffers. |
| FFT transforms | `poisson_fft2d.cpp`, `poisson_fft3d.cpp` | Parallel outer loops over independent transform rows/planes. |
| Tridiagonal solvers | `chasing_method2d.cpp`, `chasing_method3d.cpp` | Parallel outer loops over independent spectral systems. |
| Schur matrix application | `schur_mat2d.cpp`, `schur_mat3d.cpp`, slab variants | Parallel interface-index loops for dense matrix-vector actions. |
| MPI transposes | `transpose_slab.cpp`, `transpose_pencil.cpp` | Parallel local packing/unpacking or transpose loops. |
| Navier-Stokes/scalar/MHD/PPE | `ns/*.cpp` | Parallel stencil loops over interior cells and boundary strips. |
| Particles/IBM | `particle/*.cpp`, `ibm_Uhlmann/*.cpp` | Parallel particle-coordinate updates and force/interpolation loops. |
| Cases | `case/precision/*`, `case/benchmark/*` | Parallel reductions for errors, norms, or benchmark statistics. |

### Granularity Assessment

The dominant granularity is fine to medium-grained loop parallelism:

- Dense field loops split rows, planes, or flattened cell ranges.
- FFT loops split independent transform batches.
- Chasing loops split independent tridiagonal systems.
- Stencil solvers split spatial cells or boundary strips.

The higher-level orchestration is mostly serial:

- Geometry construction is serial.
- Solver-tree construction is serial.
- Schur-matrix construction iterates interface bases and child solves in serial at the outer algorithmic level.
- `ConcatPoissonSolver*::solve` traverses hierarchy levels serially.
- GMRES Krylov iteration is serial at the algorithm-control level, with parallelism inside field operations or Schur matrix-vector products.

This design avoids many coarse-grained synchronization problems but leaves potential performance on the table for multi-domain assemblies.

### Data Sharing & Isolation Analysis

Shared read-mostly state inside OpenMP regions includes:

- Domain dimensions, spacings, and boundary metadata.
- Geometry-derived maps after construction.
- FFTW plans and transform coefficients.
- Input field arrays when a loop writes to a distinct output buffer.
- Schur dense matrix values during matrix-vector products.

Thread-private or effectively private state includes:

- Loop indices and local scalar temporaries.
- Per-row/per-plane work buffers in FFT transform loops when indexed by the parallel loop index.
- Independent tridiagonal line buffers in chasing methods.
- Cell-local stencil intermediates in Navier-Stokes/scalar/PPE loops.

Reduction usage is explicit mainly in cases, such as precision/error calculations and benchmark statistics using `OPENMP_PARALLEL_FOR(reduction(+ : local_sum))` or similar reductions. Core GMRES dot products and norms are not broadly expressed as OpenMP reductions; many are serial or delegated to field helper methods.

### Concurrency Vulnerabilities

The current parallel regions are mostly safe when the data topology is frozen before entering a solver call. The hazards appear when ownership or write semantics are changed carelessly:

- `IBVelocitySolver*_Uhlmann::apply_ib_force` and scalar analogues spread particle forces with `+=` into grid fields. Multiple immersed-boundary points can target the same grid cell concurrently, causing data races unless non-overlap is externally guaranteed.
- Some immersed-boundary fallback accessors return a reference to `static double zero`. If missing-cell paths are written through under OpenMP, unrelated threads can race on this shared dummy value and silently discard intended force.
- Solver objects are not reentrant. `ConcatPoissonSolver*`, `GMRESSolver*`, `PoissonSolver*`, and CFD solvers mutate internal buffers during `solve()`. The same solver instance must not be used concurrently from multiple threads.
- Geometry, variable boundary maps, field maps, and solver maps must not be modified while a parallel loop reads them.
- FFTW plan creation/destruction is not protected. The code assumes plans are built before parallel execution and only executed inside transform loops with separate data workspaces.
- `field3::operator=` assumes destination shape/storage is already compatible. Under parallel execution, a shape mismatch becomes harder to diagnose and can corrupt memory faster.
- The macro uses `schedule(dynamic)`. This is robust for uneven work but may reduce cache locality in regular dense stencil loops compared with static scheduling.

## 5. Module Assembly & Extension Guide

### Assembly Architecture

The repository assembles solvers as layered building blocks:

1. **Field layer**: `field2`/`field3` own numeric memory.
2. **Domain layer**: `Domain*Uniform` objects describe rectangular blocks.
3. **Geometry layer**: `Geometry*` stitches domains and derives the solve tree.
4. **Variable layer**: `Variable*` binds semantic physical variables to per-domain fields and boundary buffers.
5. **Poisson layer**: `PoissonSolver*` solves one block; `ConcatPoissonSolver*` solves stitched geometries.
6. **Physics layer**: `ConcatNSSolver*`, `ScalarSolver*`, `PhysicalPESolver*`, MHD, and IBM modules consume variables and Poisson solvers.
7. **Case layer**: files in `case/` instantiate all objects, set parameters and boundary conditions, and run loops.

No central `Manager` class owns the whole lifecycle. The case file is effectively the composition root. This makes examples easy to read, but it also means extension authors must preserve object lifetimes manually.

### Scenario A: Adding Operators

For a new scalar or Navier-Stokes finite-difference operator:

- Add or extend the relevant scheme enum in `poisson_base/base/scheme_type.h`.
- Implement the operator in the matching solver source, such as `ns/scalar_solver2d.cpp`, `ns/scalar_solver3d.cpp`, or `ns/concat_ns_solver*.cpp`.
- Update the existing `switch`/dispatch point that selects `center2nd`, `upwind1st`, `QUICK`, or `TVD`-style logic.
- Keep interior-cell loops separate from boundary/outer loops, following the current organization.
- Use existing `Variable*` boundary-update functions rather than directly mutating adjacent-domain buffers.

For a new Poisson boundary transform or regular-domain operator:

- Add a concrete FFT transform class under `poisson_base/pe/poisson/poisson_fft2d.*` or `poisson_fft3d.*`.
- Update `PoissonSolver2DBase::create_fft` or `PoissonSolver3DBase::create_fft` so the boundary-pair mapping selects the new transform.
- Update `cal_lambda` consistently with the transform eigenvalues.
- If the x-direction linear system changes shape, extend `ChasingMethodBase` or add a new chasing variant.
- Add validation cases under `case/precision` or `case/validation` following existing Poisson examples.

For a new stitched-interface coupling:

- Add a `SchurMat2D*` or `SchurMat3D*` subclass for the new interface orientation or condition.
- Implement both `construct(...)` and `operator*(...)` so the dense response matrix is built and applied consistently.
- Update `GMRESSolver*::schur_mat_construct` switch logic to instantiate the new Schur object.
- Keep ownership explicit: if the new Schur subclass owns resources, first add virtual destructors to `SchurMat2D` and `SchurMat3D`.

For a new body-force or physics module:

- Prefer the `MHDModule2D` style: own internal variables/fields with `std::unique_ptr`, borrow external variables explicitly, and expose a small `solve/update/apply` surface.
- Hook the module into `ConcatNSSolver*::solve()` at a clear phase boundary, such as after predictor velocities and before pressure correction.
- Avoid hidden writes into pressure/velocity fields during OpenMP loops unless writes are one-cell-per-thread or protected by a reduction/accumulation strategy.

### Scenario B: Data Expansion

To add a new semantic variable in a case:

- Declare one `Variable2D` or `Variable3D`.
- Call `set_geometry(&geo)`.
- Allocate one `field2` or `field3` per domain at case scope.
- Attach fields with the correct centering function: `set_center_field`, `set_x_edge_field`, `set_y_edge_field`, or `set_z_edge_field`.
- Configure all non-adjacent physical boundary types and values before solver construction.
- Ensure the fields outlive the variable and every solver that borrows them.

To add new attributes to existing structures:

- For `Domain*`, prefer value metadata with no heap ownership unless the geometry/tree code must own it.
- For `Variable*`, any new owning pointer or container must be cleaned in the destructor and must force an explicit copy policy. The safest policy is `delete` copy constructor and copy assignment.
- For solver classes, do not add raw owning pointers without also adding destructor, move, and deleted-copy semantics.
- For particle properties, follow the existing `ParticlesBase`/macro pattern so the base class owns property arrays and remains non-copyable.
- When adding boundary data, update boundary assembly, shared-boundary update, and save/load paths together; these are tightly coupled through `LocationType`, `BoundaryType`, and variable centering.

## 6. Architectural Limitations & Optimization Anchors

### Structural Bottlenecks

The main performance risks visible from static structure are:

- Dense Schur matrices scale poorly with interface size. In 3D, a face interface of size `ny * nz`, `nx * nz`, or `nx * ny` yields a dense `cn x cn` matrix, which can dominate setup memory and matrix-vector cost.
- Schur construction performs many child Poisson solves, one per interface basis vector. This is a natural setup bottleneck and is currently not parallelized at the outer basis-vector level.
- Geometry and variable dispatch rely heavily on `std::map`/pointer-key lookups. This is convenient semantically but less cache-friendly than compact domain IDs and contiguous vectors.
- Boundary buffers are fragmented across raw pointers, nested maps, and heap-allocated face fields. This increases pointer chasing and complicates ownership.
- Some FFT workspaces are represented as nested pointer arrays rather than single flat workspaces, which can reduce locality and makes copy/destruction hazards more likely.
- `field2` arithmetic returns temporary fields in several operators. In GMRES and Schur applications, repeated temporary allocation can be expensive.
- `OPENMP_PARALLEL_FOR` always uses dynamic scheduling in `HYBRID`; regular stencils and transforms may prefer static scheduling for locality.
- Solver orchestration is mostly serial across domains, levels, and Schur basis construction. The implementation exploits loop-level parallelism but not much task-level parallelism.
- `ConcatNSSolver*`, `ScalarSolver*`, and several IBM classes appear to leak internal allocations. Besides correctness, leaks prevent long-running simulations or repeated solver construction from scaling cleanly.

### Coupling Constraints

Several modules are tightly coupled by convention rather than type-enforced contracts:

- `Geometry*::solve_prepare()` must build a valid tree before concatenated solvers construct their solver maps. Changing geometry after solver construction invalidates the solver's assumptions.
- `Variable*` boundary maps, buffer maps, and position types must remain consistent. A field attached with the wrong centering can make boundary updates numerically wrong without a compiler error.
- `ConcatPoissonSolver*::solve()` scales RHS values by `hx * hx`, so callers must treat the solver as owning that discretization convention.
- Boundary transform selection assumes specific `BoundaryType` pairs. Adding a new boundary type requires coordinated changes in transform selection, eigenvalue calculation, boundary assembly, and tests.
- Schur matrix subclasses assume exact interface orientation and index flattening. 3D changes need careful consistency across `construct`, `operator*`, and `bond_add`.
- Pure Neumann problems require normalization or singular-mode handling. The code contains special paths for singular tridiagonal cases and pressure normalization; extensions must preserve these zero-mean constraints.
- MPI slab variants cast and store specialized domain/variable forms. Any domain-layout refactor must account for serial, hybrid, and pure-MPI code paths together.
- MHD, IBM, and pressure projection all mutate velocity/pressure fields in sequenced phases. Inserting a new physics module at the wrong point can break projection consistency or boundary synchronization.

The strongest long-term architectural improvement would be to make ownership explicit first: delete unsafe copies, add missing destructors or smart pointers, and flatten hot-path maps into domain-indexed vectors after geometry finalization. That would reduce both memory hazards and performance ambiguity while preserving the current semantic API used by `case/`.
