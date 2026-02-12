#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"

#include "ns/ns_solver3d.h"

#include "base/config.h"
#include "io/vtk_writer.h"

#include "pe/concat/concat_solver3d.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

/**
 *
 * y
 * ▲
 * │
 * │
 * │
 * │
 * │
 * ├──────┬──────┬──────┐
 * │      │      │      │
 * │  A1  │  A2  │  A3  │
 * │      │      │      │
 * ├──────┼──────┼──────┘
 * │      │      │
 * │      │  A4  │
 * │      │      │
 * └──────┴──────┴──────────►
 * O                         x
 *
 */
int main(int argc, char* argv[])
{
    TIMER_BEGIN(Init, "Init", TimeRecordType::None, true);

    double lx1 = 0.3;
    double ly1 = 0.02;
    double lz1 = 0.02;

    double lx2 = 0.04;
    double ly2 = 0.02;
    double lz2 = 0.02;

    double lx3 = lx1; // symmetry
    double ly3 = ly1; // symmetry
    double lz3 = lz1; // symmetry

    double lx4 = 0.04;
    double ly4 = 0.6;
    double lz4 = 0.02;

    double h = 0.001;

    int nx1 = lx1 / h;
    int ny1 = ly1 / h;
    int nz1 = lz1 / h;
    int nx2 = lx2 / h;
    int ny2 = ly2 / h;
    int nz2 = lz2 / h;
    int nx3 = lx3 / h;
    int ny3 = ly3 / h;
    int nz3 = lz3 / h;
    int nx4 = lx4 / h;
    int ny4 = ly4 / h;
    int nz4 = lz4 / h;

    double Re                  = 175;
    double density             = 1.03;
    double dynamic_viscosity   = 1.57e-5;
    double feature_velocity    = Re * dynamic_viscosity / (density * ly1);
    double kinematic_viscosity = dynamic_viscosity / density;

    // Geometry: Cross shape
    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.debugOutputDir     = "./result/backstep_3d/Re" + std::to_string(Re);

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = 0.0001;
    time_cfg.num_iterations       = 1e5;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_nu(kinematic_viscosity);

    Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");
    Domain3DUniform A2(nx2, ny2, nz2, lx2, ly2, lz2, "A2");
    Domain3DUniform A3(nx3, ny3, nz3, lx3, ly3, lz3, "A3");
    Domain3DUniform A4(nx4, ny4, nz4, lx4, ly4, lz4, "A4");

    geo.add_domain(&A1);
    geo.add_domain(&A2);
    geo.add_domain(&A3);
    geo.add_domain(&A4);

    // Construct cross connectivity
    geo.connect(&A2, LocationType::Left, &A1);
    geo.connect(&A2, LocationType::Right, &A3);
    geo.connect(&A2, LocationType::Front, &A4);

    geo.axis(&A1, LocationType::Left);
    geo.axis(&A1, LocationType::Front);
    geo.axis(&A1, LocationType::Down);

    // Variable2Ds
    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    // Fields on each domain
    field3 u_A1, u_A2, u_A3, u_A4;
    field3 v_A1, v_A2, v_A3, v_A4;
    field3 w_A1, w_A2, w_A3, w_A4;
    field3 p_A1, p_A2, p_A3, p_A4;

    u.set_x_face_center_field(&A1, u_A1);
    u.set_x_face_center_field(&A2, u_A2);
    u.set_x_face_center_field(&A3, u_A3);
    u.set_x_face_center_field(&A4, u_A4);
    v.set_y_face_center_field(&A1, v_A1);
    v.set_y_face_center_field(&A2, v_A2);
    v.set_y_face_center_field(&A3, v_A3);
    v.set_y_face_center_field(&A4, v_A4);
    w.set_z_face_center_field(&A1, w_A1);
    w.set_z_face_center_field(&A2, w_A2);
    w.set_z_face_center_field(&A3, w_A3);
    w.set_z_face_center_field(&A4, w_A4);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);

    // Helper setters
    auto set_dirichlet_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto isdjacented = [&](Domain3DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain3DUniform*> domains = {&A1, &A2, &A3, &A4};
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
            set_dirichlet_zero(w, d, loc);
            // pressure: default Neumann (zero gradient)
            set_neumann_zero(p, d, loc);
        }
    }

    // Inlet
    {
        u.has_boundary_value_map[&A1][LocationType::Left]  = true;
        u.has_boundary_value_map[&A3][LocationType::Right] = true;

        field2& u_inlet_buffer_left  = *u.boundary_value_map[&A1][LocationType::Left];
        field2& u_inlet_buffer_right = *u.boundary_value_map[&A3][LocationType::Right];

        for (int j = 0; j < u_A1.get_ny(); ++j)
        {
            for (int k = 0; k < u_A1.get_nz(); ++k)
            {
                double z = k * h + 0.5 * h;
                z /= lz1;
                double vel                 = 6.0 * feature_velocity * (1.0 - z) * z;
                u_inlet_buffer_left(j, k)  = vel;
                u_inlet_buffer_right(j, k) = vel;
            }
        }
    }
    // Outlet
    u.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);

    ConcatNSSolver3D solver(&u, &v, &w, &p);

    VTKWriter vtk_writer;
    vtk_writer.add_vector_as_cell_data(&u, &v, &w, "velocity");
    vtk_writer.validate();

    TIMER_END(Init);

    for (int iter = 0; iter <= time_cfg.num_iterations; iter++)
    {
        SCOPE_TIMER("Iteration", TimeRecordType::None, iter % 100 == 0);

        if (iter % 100 == 0)
        {
            std::cout << "iter: " << iter << "/" << time_cfg.num_iterations << "\n";

            env_cfg.track_pe_solve_detail_time = true;
            env_cfg.showGmresRes               = true;
        }

        solver.solve();

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (iter % 5000 == 0 && iter != 0)
        {
            vtk_writer.write(env_cfg.debugOutputDir + "/" + std::to_string(iter));
        }
    }

    std::cout << "Finished" << std::endl;
}