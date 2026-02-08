#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"

#include "ns/ns_solver3d.h"

#include "io/config.h"
#include "io/csv_writer_3d.h"

#include "pe/concat/concat_solver3d.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showGmresRes       = true;

    int rank = 4;
    int nx1  = rank;
    int ny1  = 2 * rank;
    int nz1  = 3 * rank;

    double H = 1.0 / rank;

    double lx1 = nx1 * H;
    double ly1 = ny1 * H;
    double lz1 = nz1 * H;

    Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");

    geo.add_domain(&A1);

    geo.set_global_spatial_step(H, H, H);

    Variable3D p("p");
    p.set_geometry(geo);

    field3 p_A1;
    p.set_center_field(&A1, p_A1);

    geo.axis(&A1, LocationType::Left);
    geo.axis(&A1, LocationType::Front);
    geo.axis(&A1, LocationType::Down);

    // p.fill_boundary_type(PDEBoundaryType::Neumann);
    p.set_boundary_type(&A1, LocationType::Left, PDEBoundaryType::Neumann);
    p.set_boundary_type(&A1, LocationType::Right, PDEBoundaryType::Neumann);
    p.set_boundary_type(&A1, LocationType::Front, PDEBoundaryType::Neumann);
    p.set_boundary_type(&A1, LocationType::Back, PDEBoundaryType::Neumann);
    p.set_boundary_type(&A1, LocationType::Down, PDEBoundaryType::Dirichlet);
    p.set_boundary_type(&A1, LocationType::Up, PDEBoundaryType::Dirichlet);

    for (int i = 0; i < nx1; i++)
    {
        for (int j = 0; j < ny1; j++)
        {
            for (int k = 0; k < nz1; k++)
            {
                p_A1(i, j, k) = i * ny1 * nz1 + j * ny1 + k;
            }
        }
    }

    ConcatPoissonSolver3D solver(&p);
    solver.solve();

    int         rank_int = static_cast<int>(rank);
    std::string out_base = "result/p_rank_" + std::to_string(rank_int);
    IO::write_csv(p, out_base);
}
