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
    auto main_start_time = std::chrono::steady_clock::now();

    // Geometry: Cross shape
    Geometry3D geo_cross;

    EnvironmentConfig* env_config = new EnvironmentConfig();
    env_config->showGmresRes      = true;

    TimeAdvancingConfig* time_config = new TimeAdvancingConfig();
    time_config->dt                  = 0.001;
    time_config->num_iterations      = 1e5;

    PhysicsConfig* physics_config = new PhysicsConfig();
    physics_config->set_Re(200);

    double lx1 = 0.5;
    double lx2 = 0.5;
    double ly1 = 0.5;
    double lz1 = 0.5;
    double lz3 = 0.5;

    double ly2 = ly1;
    double ly3 = ly1;
    double lz2 = lz1;
    double lx3 = lx2;

    double h = 1e-2;

    int nx1 = lx1 / h;
    int ny1 = ly1 / h;
    int nz1 = lz1 / h;
    int nx2 = lx2 / h;
    int ny2 = ly2 / h;
    int nz2 = lz2 / h;
    int nx3 = lx3 / h;
    int ny3 = ly3 / h;
    int nz3 = lz3 / h;

    Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");
    Domain3DUniform A2(nx2, ny2, nz2, lx2, ly2, lz2, "A2");
    Domain3DUniform A3(nx3, ny3, nz3, lx3, ly3, lz3, "A3");

    geo_cross.add_domain(A1);
    geo_cross.add_domain(A2);
    geo_cross.add_domain(A3);

    // Construct cross connectivity
    geo_cross.connect(A2, LocationType::Left, A1);
    geo_cross.connect(A2, LocationType::Down, A3);

    // Variables
    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo_cross);
    v.set_geometry(geo_cross);
    w.set_geometry(geo_cross);
    p.set_geometry(geo_cross);

    // Fields on each domain
    field3 u_A1("u_A1"), u_A2("u_A2"), u_A3("u_A3");
    field3 v_A1("v_A1"), v_A2("v_A2"), v_A3("v_A3");
    field3 w_A1("w_A1"), w_A2("w_A2"), w_A3("w_A3");
    field3 p_A1("p_A1"), p_A2("p_A2"), p_A3("p_A3");

    u.set_x_face_center_field(&A1, u_A1);
    u.set_x_face_center_field(&A2, u_A2);
    u.set_x_face_center_field(&A3, u_A3);
    v.set_y_face_center_field(&A1, v_A1);
    v.set_y_face_center_field(&A2, v_A2);
    v.set_y_face_center_field(&A3, v_A3);
    w.set_z_face_center_field(&A1, w_A1);
    w.set_z_face_center_field(&A2, w_A2);
    w.set_z_face_center_field(&A3, w_A3);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);

    // Helper setters
    auto set_dirichlet_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto isdjacented = [&](Domain3DUniform* d, LocationType loc) {
        return geo_cross.adjacency.count(d) && geo_cross.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain3DUniform*> domains = {&A1, &A2, &A3};
    std::vector<LocationType>     dirs    = {LocationType::Left,
                                             LocationType::Right,
                                             LocationType::Front,
                                             LocationType::Back,
                                             LocationType::Down,
                                             LocationType::Up};

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
    {
        u.has_boundary_value_map[&A1][LocationType::Left] = true;
        field2& u_inlet_buffer                            = *u.boundary_value_map[&A1][LocationType::Left];
        for (int j = 0; j < u_A1.get_ny(); ++j)
        {
            for (int k = 0; k < u_A1.get_nz(); ++k)
            {
                double z             = k * h + 0.5 * h;
                u_inlet_buffer(j, k) = -24.0 * z * z + 12.0 * z;
            }
        }
    }
    // Outlet
    u.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    u.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A2, LocationType::Right, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Neumann);

    ConcatNSSolver3D solver(&u, &v, &w, &p, time_config, physics_config, env_config);

    std::chrono::steady_clock::time_point iter_start_time, iter_end_time;

    auto   main_end_time = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(main_end_time - main_start_time).count();
    std::cout << "Total init time: " << std::fixed << std::setprecision(2) << total_elapsed << " seconds.\n";

    for (int iter = 0; iter <= time_config->num_iterations; iter++)
    {
        if (iter % 200 == 0)
        {
            std::cout << "iter: " << iter << "/" << time_config->num_iterations;
            iter_start_time = std::chrono::steady_clock::now();
        }

        ConcatNSSolver3D ns_solver(&u, &v, &w, &p, time_config, physics_config, env_config);
        ns_solver.solve();

        if (iter % 200 == 0)
        {
            iter_end_time = std::chrono::steady_clock::now();
            total_elapsed = std::chrono::duration<double>(iter_end_time - iter_start_time).count();
            std::cout << " iter wall time: " << std::fixed << std::setprecision(2) << total_elapsed << " seconds.\n";
        }

        if (iter % 10000 == 0 && iter != 0)
        {
            IO::var_to_csv(u, "result/" + std::to_string(iter) + "u");
            IO::var_to_csv(v, "result/" + std::to_string(iter) + "v");
            IO::var_to_csv(w, "result/" + std::to_string(iter) + "w");
            IO::var_to_csv(p, "result/" + std::to_string(iter) + "p");
        }
    }

    main_end_time = std::chrono::steady_clock::now();
    total_elapsed = std::chrono::duration<double>(main_end_time - main_start_time).count();
    std::cout << "Total wall time: " << std::fixed << std::setprecision(2) << total_elapsed << " seconds.\n";
}