#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "io/csv_handler.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver3d.h"
#include "pe/concat/concat_solver3d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

// Basu, S., Dirbude, S.B. (2024). Flow, Thermal, and Mass Mixing Analysis in a T-Shaped Mixer. In: Sikarwar, B.S.,
// Sharma, S.K. (eds) Scientific and Technological Advances in Materials for Energy Storage and Conversions. FLUTE 2023.
// Lecture Notes in Mechanical Engineering. Springer, Singapore. https://doi.org/10.1007/978-981-97-2481-9_17

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

    if (argc != 2)
    {
        std::cerr << "Error argument! Usage: program rank[double > 0]" << std::endl;
        return 0;
    }

    double rank = std::stod(argv[1]);

    double lx1 = 0.2;
    double ly1 = 0.02;
    double lz1 = 0.02;

    double lx2 = 0.04;
    double ly2 = 0.02;
    double lz2 = 0.02;

    double lx3 = lx1; // symmetry
    double ly3 = ly1; // symmetry
    double lz3 = lz1; // symmetry

    double lx4 = 0.04;
    double ly4 = 0.4;
    double lz4 = 0.02;

    double hx = 0.001 / rank;
    double hy = 0.001 / rank;
    double hz = 0.001 / rank / 2.0;

    int nx1 = lx1 / hx;
    int ny1 = ly1 / hy;
    int nz1 = lz1 / hz;
    int nx2 = lx2 / hx;
    int ny2 = ly2 / hy;
    int nz2 = lz2 / hz;
    int nx3 = lx3 / hx;
    int ny3 = ly3 / hy;
    int nz3 = lz3 / hz;
    int nx4 = lx4 / hx;
    int ny4 = ly4 / hy;
    int nz4 = lz4 / hz;

    double Re                  = 350;
    double density             = 1e3;
    double dynamic_viscosity   = 1.01e-3;
    double feature_velocity    = Re * dynamic_viscosity / (density * ly1);
    double kinematic_viscosity = dynamic_viscosity / density;

    std::cout << "feature_velocity = " << feature_velocity << std::endl;

    // Geometry: Cross shape
    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.debugOutputDir     = "./result/T-shaped_mixer_grid_independence/" + std::to_string(rank);

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = 0.0001 * rank;
    time_cfg.num_iterations       = 2e5 * rank;

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

    std::cout << "mesh num = " << u_A1.get_size_n() + u_A2.get_size_n() + u_A3.get_size_n() + u_A4.get_size_n()
              << std::endl;

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
                double z = k * hz + 0.5 * hz;
                z /= lz1;
                double vel                 = 6.0 * feature_velocity * (1.0 - z) * z;
                u_inlet_buffer_left(j, k)  = vel;
                u_inlet_buffer_right(j, k) = -vel;
            }
        }
    }
    // Outlet
    u.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);

    ConcatNSSolver3D solver(&u, &v, &w, &p);

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

        if (iter % static_cast<int>(20 * rank) == 0)
        {
            CSVHandler v_A4_mean_j0_file(env_cfg.debugOutputDir + "/v_A4_mean_j0");
            CSVHandler v_A4_mean_jend_file(env_cfg.debugOutputDir + "/v_A4_mean_jend");

            double v_A4_mean_j0   = v_A4.mean_at_xz_plane(0);
            double v_A4_mean_jend = v_A4.mean_at_xz_plane(v_A4.get_ny() - 1);
            if (std::isnan(v_A4_mean_j0) || std::isnan(v_A4_mean_jend))
            {
                std::cout << "Error: Find nan! Break solving." << std::endl;
                break;
            }

            v_A4_mean_j0_file.stream << v_A4_mean_j0 << std::endl;
            v_A4_mean_jend_file.stream << v_A4_mean_jend << std::endl;
        }

        if (std::isnan(u_A1(0, 0, 0)))
        {
            std::cout << "Error: Find nan! Break solving." << std::endl;
            break;
        }
    }

    std::cout << "Finished" << std::endl;
}