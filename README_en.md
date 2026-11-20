<p align="center">
  <img src="Focalors_logo.png" alt="Focalors Logo" width="220"/>
</p>

<div align="center">

# Focalors

High-Performance Incompressible Multi-Physics CFD Solver Based on FFT-DDM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/Build-CMake-green.svg)](https://cmake.org/)
[![Parallel](https://img.shields.io/badge/Parallel-MPI%20%2B%20OpenMP-orange.svg)](https://www.open-mpi.org/)

[中文](./README.md)&nbsp;&nbsp;|&nbsp;&nbsp;English&nbsp;&nbsp;|&nbsp;&nbsp;[日本語](./README_ja.md)

</div>

---

## Overview

**Focalors** is a high-performance computational fluid dynamics (CFD) solver designed for incompressible flows in composite block-structured orthogonal geometries (e.g., cross-slot channels, T-mixers, backward-facing steps).

In traditional incompressible CFD simulations, solving the pressure Poisson equation on composite multi-block domains typically relies on multigrid or Krylov subspace iterative solvers. Focalors uses an FFT-based Domain Decomposition Method (FFT-DDM), maintaining a quasi-linear O(N log N) complexity on composite domains, comparable to a single-domain FFT.

The solver supports the incompressible Navier-Stokes equations and incorporates physical modules for non-Newtonian rheology, low magnetic Reynolds number magnetohydrodynamics (MHD), immersed boundary methods (IBM), and Lagrangian particle tracking.

---

## Modules and Capabilities

- **Poisson Solver (`poisson_base`)**
  - Solves the Poisson equation on 2D/3D composite orthogonal domains based on FFT-DDM with O(N log N) complexity.
  - Supports physical static pressure recovery (`Physical PE Solver`) and solvability compatibility projection for all-Neumann boundary conditions.
- **Flow Solver (`ns`)**
  - Solves the incompressible Navier-Stokes equations via the fractional-step projection method.
  - Supports MAC staggered grid and collocated grid discretizations.
  - Includes a passive scalar convection-diffusion solver (supporting high-order schemes such as QUICK).
- **Non-Newtonian Rheology (`ns`)**
  - Supports Power-law, Carreau, and Bird-Carreau constitutive models.
  - Implements viscosity under-relaxation and non-linear strain-rate iterations.
- **Low-Rm Magnetohydrodynamics (`ns`)**
  - Couples fluid flow and electromagnetic fields under the low magnetic Reynolds number approximation; solves the electric potential Poisson equation and evaluates the J × B Lorentz force.
  - Supports insulating (J · n = 0) and conducting wall boundary conditions.
- **Immersed Boundary Method & Particle Tracking (`ibm_Uhlmann`, `ibm_MirrorPoint`, `particle`)**
  - Includes the Uhlmann direct-forcing method (regularized discrete delta functions) and the sharp-interface mirror-point method.
  - Supports Lagrangian particle tracking and spatial mapping.
- **Parallel Computing Architecture**
  - Hybrid MPI (inter-process communication) and OpenMP (shared-memory multithreading) execution model.
- **Data I/O and Storage (`io`)**
  - Supports consolidated VTK XML unstructured grid (`.vtu`) export for 3D multi-domain cases for direct inspection in ParaView.
  - Outputs structured `.csv` field data by domain and probe time histories for 2D cases, enabling convenient post-processing and analysis.

---

## Directory Structure

```text
Focalors/
├── case/               # Test cases (benchmark, validation, tests)
├── ibm_MirrorPoint/    # Sharp-interface mirror-point IBM
├── ibm_Uhlmann/        # Direct-forcing immersed boundary method
├── io/                 # Data I/O (VTK XML, CSV probes, checkpoints)
├── ns/                 # Navier-Stokes solver core (MAC discretization, MHD, non-Newtonian, scalar)
├── particle/           # Lagrangian particle tracking module
├── poisson_base/       # FFT domain decomposition Poisson solver core
└── tools/              # Auxiliary preprocessing and symmetry tools
```

---

## Prerequisites

- **Operating System**: Linux (CentOS 7+, Ubuntu 20.04+, or HPC cluster environments)
- **Compiler**: C++17 compliant compiler (GCC >= 9.0, Clang >= 9.0, or Intel oneAPI icpx)
- **CMake**: >= 3.10
- **FFTW3**: >= 3.3.8 (Fast Fourier Transform library, multi-threading enabled recommended)
- **MPI**: OpenMPI, MPICH, or Intel MPI
- **OpenMP**: Multi-threading support
- **VTK**: 8.x / 9.x (requires CommonCore, IOXML, IOParallelXML, ParallelMPI components)

---

## Build and Configuration

### 1. Build Types

Configured via `-DCMAKE_BUILD_TYPE`:

- **`Hybrid`** (Default / Recommended): Enables OpenMP + MPI hybrid parallelism with flags `-g -O3 -DHYBRID -DNDEBUG`.
- **`PureMPI`**: Disables OpenMP and relies solely on MPI with flags `-g -O3 -DPUREMPI -DNDEBUG`.
- **`Debug`**: Disables OpenMP and optimizations for debugging with flags `-g -O0`.

### 2. Build Instructions

```bash
# 1. Export library paths if installed in non-standard locations
export FFTW_ROOT=/path/to/fftw3
export VTK_DIR=/path/to/vtk/lib/cmake/vtk-9.x

# 2. Configure build directory (Hybrid mode recommended)
cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=Hybrid

# 3. Parallel compilation
cmake --build build --parallel $(nproc)
```

Executables are placed in `build/bin/`.

---

## Usage and Examples

### 1. Target Naming Convention

All `.cpp` files in `case/` are automatically registered as CMake executable targets with path separators replaced by underscores:

- `case/benchmark/2d/cross_shaped_channel.cpp`  
  -> `build/bin/benchmark_2d_cross_shaped_channel`
- `case/benchmark/3d/cross_shaped_channel.cpp`  
  -> `build/bin/benchmark_3d_cross_shaped_channel`

### 2. Running a Case

Example using the 2D cross-slot benchmark case:

```bash
# Build the specified case target
cmake --build build --target benchmark_2d_cross_shaped_channel --parallel $(nproc)

# Set OpenMP thread count (adjust based on system resources)
export OMP_NUM_THREADS=$(nproc)

# Run simulation
./build/bin/benchmark_2d_cross_shaped_channel
```

For multi-process MPI cases (e.g., 2D multi-domain Poisson test):

```bash
mpirun -np 4 ./build/bin/precision_2d_five_domain_pe_mpi
```

Output field data (3D `.vtu` or 2D `.csv`) are saved in `result/` or the case-designated directory. 3D fields can be inspected directly in ParaView, and 2D arrays/probes can be processed using standard analysis tools.

---

## Creating Custom Cases

Add a `.cpp` file in `case/` following this structure:

```cpp
#include "base/domain/geometry2d.h"
#include "ns/ns_solver2d.h"
#include "io/csv_writer_2d.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // 1. Define composite geometry and grid
    // 2. Set boundary conditions for velocity, pressure, potential, or scalars
    // 3. Set physical parameters (Re, Ha, power-law index n, apparent viscosity, etc.)
    // 4. Initialize solver and advance time steps
    // 5. Write VTK field files and CSV probes

    MPI_Finalize();
    return 0;
}
```

Rerunning `cmake --build build` will automatically detect and compile the new case.

---

## References

Core algorithm and fluid solver publications:

1. **FFT-based Poisson Solver in Concatenated Geometries (FFT-DDM / GMRES)**:
   - Zichao Jiang, Jiacheng Lian, Zhuolin Wang. *A FFT-based GMRES for fast solving of Poisson equation in concatenated geometry*.
   - **arXiv**: [https://arxiv.org/abs/2509.23180](https://arxiv.org/abs/2509.23180)
   - **DOI**: [https://doi.org/10.48550/arXiv.2509.23180](https://doi.org/10.48550/arXiv.2509.23180)

2. **Parallel Incompressible Flow and Overlapping IBM Algorithm (IBM-PISO & FFT Solver)**:
   - Jiacheng Lian, Qinghe Yao, Zichao Jiang. *An Overlapping IBM-PISO Algorithm with an FFT-Based Poisson Solver for Parallel Incompressible Flow Simulations*.
   - ***Fluids***, 2025, 10(7), 176.
   - **DOI**: [https://doi.org/10.3390/fluids10070176](https://doi.org/10.3390/fluids10070176)

---

## Development Team

This project is developed and maintained by Zichao Jiang, Jiacheng Lian, and Zhaowei Zhang.

---

## License

This project is licensed under the [MIT License](LICENSE).
