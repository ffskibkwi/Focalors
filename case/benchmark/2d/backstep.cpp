#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"

#include "ns/ns_solver2d.h"

#include "io/config.h"
#include "io/csv_writer_2d.h"

#include "pe/concat/concat_solver2d.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    // Geometry: Cross shape
    Geometry2D geo_cross;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showGmresRes       = true;

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = 0.001;
    time_cfg.num_iterations       = 1e5;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(1000);

    double lx1 = 10;
    double lx2 = 40;
    double ly1 = 0.5;
    double ly3 = 0.5;

    double ly2 = ly1;
    double lx3 = lx2;

    double h = 1e-2;

    int nx1 = lx1 / h;
    int ny1 = ly1 / h;
    int nx2 = lx2 / h;
    int ny2 = ly2 / h;
    int nx3 = lx3 / h;
    int ny3 = ly3 / h;

    Domain2DUniform A1(nx1, ny1, lx1, ly1, "A1");
    Domain2DUniform A2(nx2, ny2, lx2, ly2, "A2");
    Domain2DUniform A3(nx3, ny3, lx3, ly3, "A3");

    geo_cross.add_domain({&A1, &A2, &A3});

    // Construct cross connectivity
    geo_cross.connect(&A2, LocationType::Left, &A1);
    geo_cross.connect(&A2, LocationType::Down, &A3);

    // Variable2Ds
    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo_cross);
    v.set_geometry(geo_cross);
    p.set_geometry(geo_cross);

    // Fields on each domain
    field2 u_A1, u_A2, u_A3;
    field2 v_A1, v_A2, v_A3;
    field2 p_A1, p_A2, p_A3;

    u.set_x_edge_field(&A1, u_A1);
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    v.set_y_edge_field(&A1, v_A1);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);

    // Helper setters
    auto set_dirichlet_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto isdjacented = [&](Domain2DUniform* d, LocationType loc) {
        return geo_cross.adjacency.count(d) && geo_cross.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3};
    std::vector<LocationType> dirs = {LocationType::Left, LocationType::Right, LocationType::Down, LocationType::Up};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (isdjacented(d, loc))
                continue; // internal boundaries handled automatically
            // velocity: default wall (Dirichlet 0)
            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            // pressure: default Neumann (zero gradient)
            set_neumann_zero(p, d, loc);
        }
    }

    // Inlet
    u.has_boundary_value_map[&A1][LocationType::Left] = true;
    for (int j = 0; j < u_A1.get_ny(); ++j)
    {
        double y                                         = j * h + 0.5 * h;
        u.boundary_value_map[&A1][LocationType::Left][j] = -24.0 * y * y + 12.0 * y;
    }
    // Outlet
    u.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    u.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);

    ConcatNSSolver2D solver(&u, &v, &p);

    for (int iter = 0; iter < time_cfg.num_iterations; iter++)
    {
        if (iter % 200 == 0)
            std::cout << "iter: " << iter << "/" << time_cfg.num_iterations << "\n";

        ConcatNSSolver2D ns_solver(&u, &v, &p);
        ns_solver.solve();
    }

    IO::write_csv(u, "result/u");
    IO::write_csv(v, "result/v");
    IO::write_csv(p, "result/p");
}