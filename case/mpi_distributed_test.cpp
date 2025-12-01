#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "base/location_boundary.h"
#include "pe_mpi_distributed/d_domain_2d.h"
#include "pe_mpi_distributed/d_field_2d.h"
#include "pe_mpi_distributed/mpi_poisson_solver2d.h"

// Analytical solution for testing (same as pe_test.cpp logic simplified)
// u(x, y) = sin(pi*x) * sin(pi*y)
// f(x, y) = -2*pi^2 * sin(pi*x) * sin(pi*y)
// Domain: [0, 1] x [0, 1]

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int    nx = 128;
    int    ny = 128;
    double lx = 1.0;
    double ly = 1.0;
    double hx = lx / nx;
    double hy = ly / ny;

    // 1. Create Distributed Domain
    DDomain2D domain(MPI_COMM_WORLD, nx, ny, hx, hy);

    // 2. Create Distributed Fields
    DField2D f(&domain, "f");
    DField2D u(&domain, "u");

    // 3. Initialize RHS (f) locally
    // Each rank initializes its own part of the domain
    field2& f_local  = f.get_local_data();
    int     i_start  = domain.get_local_i_start();
    int     local_nx = domain.get_local_nx();

    double pi = 3.14159265358979323846;

    for (int i = 0; i < local_nx; ++i)
    {
        int    i_global = i_start + i;
        double x        = (i_global + 0.5) * hx; // Cell center
        for (int j = 0; j < ny; ++j)
        {
            double y      = (j + 0.5) * hy; // Cell center
            double val    = -2.0 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
            f_local(i, j) = val;
        }
    }

    // 4. Create Solver
    // Dirichlet boundaries on all sides
    MPIDistributedPoissonSolver2D solver(&domain,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet,
                                         PDEBoundaryType::Dirichlet);

    // 5. Solve
    // Input f is RHS, Output f is Solution (in-place)
    solver.solve(f);

    // 6. Verify Result
    // Compare f (now u) with analytical solution
    double local_error_sq = 0.0;
    for (int i = 0; i < local_nx; ++i)
    {
        int    i_global = i_start + i;
        double x        = (i_global + 0.5) * hx;
        for (int j = 0; j < ny; ++j)
        {
            double y     = (j + 0.5) * hy;
            double exact = std::sin(pi * x) * std::sin(pi * y);
            double diff  = f_local(i, j) - exact;
            local_error_sq += diff * diff;
        }
    }

    double global_error_sq = 0.0;
    MPI_Reduce(&local_error_sq, &global_error_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double l2_error = std::sqrt(global_error_sq * hx * hy);
        std::cout << "[Test] Grid: " << nx << "x" << ny << std::endl;
        std::cout << "[Test] L2 Error: " << l2_error << std::endl;

        if (l2_error < 1e-3)
            std::cout << "[Test] PASSED" << std::endl;
        else
            std::cout << "[Test] FAILED" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
