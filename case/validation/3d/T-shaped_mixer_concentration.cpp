#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "base/math/random.h"
#include "io/csv_handler.h"
#include "io/stat.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver3d.h"
#include "ns/scalar_solver3d.h"
#include "pe/concat/concat_solver3d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Steady and unsteady regimes in a T-shaped micro-mixer: Synergic experimental and numerical investigation, Alessandro
// Mariotti and Chiara Galletti and Roberto Mauri and Maria Vittoria Salvetti and Elisabetta Brunazzi

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

    if (argc != 3)
    {
        std::cerr << "Error argument! Usage: program Re[double > 0] Sc[double > 0]" << std::endl;
        return 0;
    }

    double Height = 1e-3;

    double lx1 = 20 * Height;
    double ly1 = Height;
    double lz1 = Height;

    double lx2 = 2.0 * Height;
    double ly2 = Height;
    double lz2 = Height;

    double lx3 = lx1; // symmetry
    double ly3 = ly1; // symmetry
    double lz3 = lz1; // symmetry

    double lx4 = 2.0 * Height;
    double ly4 = 40.0 * Height;
    double lz4 = Height;

    double hx = Height / 20.0;
    double hy = Height / 20.0;
    double hz = Height / 20.0;

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

    double Re = std::stod(argv[1]);
    double Sc = std::stod(argv[2]);
    double Pe = Sc * Re;
    double nr = 1.0 / Pe;

    double dt = hx / 10.0;

    double density                           = 1e3;
    double dynamic_viscosity                 = 1.01e-3;
    double mixing_channel_hydraulic_diameter = 4.0 * Height / 3.0;
    double inlet_velocity                    = Re * dynamic_viscosity / (density * mixing_channel_hydraulic_diameter);
    double convective_time                   = mixing_channel_hydraulic_diameter / inlet_velocity;

    DifferenceSchemeType c_scheme = DifferenceSchemeType::Conv_QUICK_Diff_Center2nd;

    lx1 /= mixing_channel_hydraulic_diameter;
    ly1 /= mixing_channel_hydraulic_diameter;
    lz1 /= mixing_channel_hydraulic_diameter;

    lx2 /= mixing_channel_hydraulic_diameter;
    ly2 /= mixing_channel_hydraulic_diameter;
    lz2 /= mixing_channel_hydraulic_diameter;

    lx3 /= mixing_channel_hydraulic_diameter;
    ly3 /= mixing_channel_hydraulic_diameter;
    lz3 /= mixing_channel_hydraulic_diameter;

    lx4 /= mixing_channel_hydraulic_diameter;
    ly4 /= mixing_channel_hydraulic_diameter;
    lz4 /= mixing_channel_hydraulic_diameter;

    hx /= mixing_channel_hydraulic_diameter;
    hy /= mixing_channel_hydraulic_diameter;
    hz /= mixing_channel_hydraulic_diameter;

    dt /= convective_time;

    std::cout << "mixing_channel_hydraulic_diameter = " << mixing_channel_hydraulic_diameter << std::endl;
    std::cout << "inlet_velocity = " << inlet_velocity << std::endl;
    std::cout << "convective_time = " << convective_time << std::endl;

    std::cout << "convection trem CFL = " << dt / hx << std::endl;
    std::cout << "Petlet cell = " << hx * Pe << std::endl;

    // Geometry: Cross shape
    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    {
        std::stringstream ss;
        ss << "./result/T-shaped_mixer_concentration/";
        ss << "Re";
        ss << std::to_string((int)Re);
        env_cfg.debugOutputDir = ss.str();
    }

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = dt;
    time_cfg.num_iterations       = 2e5;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(Re);

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
    Variable3D u("u"), v("v"), w("w"), p("p"), c("concentration");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);
    c.set_geometry(geo);

    // Fields on each domain
    field3 u_A1, u_A2, u_A3, u_A4;
    field3 v_A1, v_A2, v_A3, v_A4;
    field3 w_A1, w_A2, w_A3, w_A4;
    field3 p_A1, p_A2, p_A3, p_A4;
    field3 c_A1, c_A2, c_A3, c_A4;

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
    c.set_center_field(&A1, c_A1);
    c.set_center_field(&A2, c_A2);
    c.set_center_field(&A3, c_A3);
    c.set_center_field(&A4, c_A4);

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
            set_dirichlet_zero(c, d, loc);
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
                double vel                 = 6.0 * (1.0 - z) * z;
                u_inlet_buffer_left(j, k)  = vel;
                u_inlet_buffer_right(j, k) = -vel;
            }
        }

        c.has_boundary_value_map[&A3][LocationType::Right] = true;

        field2& c_inlet_buffer_right = *c.boundary_value_map[&A3][LocationType::Right];

        for (int j = 0; j < c_A3.get_ny(); ++j)
        {
            for (int k = 0; k < c_A3.get_nz(); ++k)
            {
                c_inlet_buffer_right(j, k) = 1.0;
            }
        }
    }
    // Outlet
    u.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);
    c.set_boundary_type(&A4, LocationType::Front, PDEBoundaryType::Neumann);

    add_random_number(u_A1, -0.01, 0.01, 42);
    add_random_number(u_A2, -0.01, 0.01, 42);
    add_random_number(u_A3, -0.01, 0.01, 42);
    add_random_number(u_A4, -0.01, 0.01, 42);

    add_random_number(v_A1, -0.01, 0.01, 42);
    add_random_number(v_A2, -0.01, 0.01, 42);
    add_random_number(v_A3, -0.01, 0.01, 42);
    add_random_number(v_A4, -0.01, 0.01, 42);

    add_random_number(w_A1, -0.01, 0.01, 42);
    add_random_number(w_A2, -0.01, 0.01, 42);
    add_random_number(w_A3, -0.01, 0.01, 42);
    add_random_number(w_A4, -0.01, 0.01, 42);

    ConcatNSSolver3D solver(&u, &v, &w, &p);
    ScalarSolver3D   solver_c(&u, &v, &w, &c, nr, c_scheme);

    VTKWriter vtk_writer;
    vtk_writer.add_vector_as_cell_data(&u, &v, &w, "velocity");
    vtk_writer.add_scalar_as_cell_data(&c);
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
        solver_c.solve();

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (iter % static_cast<int>(1e4) == 0)
        {
            static int count = 0;
            vtk_writer.write(env_cfg.debugOutputDir + "/vtk/" + std::to_string(count++));
        }

        if (iter % 20 == 0)
        {
            CSVHandler c_rms_file(env_cfg.debugOutputDir + "/c_rms");
            c_rms_file.stream << calc_rms(c) << std::endl;
        }

        if (std::isnan(u_A1(0, 0, 0)))
        {
            std::cout << "Error: Find nan at u_A1! Break solving." << std::endl;
            break;
        }

        if (std::isnan(c_A1(0, 0, 0)))
        {
            std::cout << "Error: Find nan at c_A1! Break solving." << std::endl;
            break;
        }
    }

    std::cout << "Finished" << std::endl;
}