#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "base/math/random.h"
#include "ibm_Uhlmann/ib_velocity_solver_2d_Uhlmann.h"
#include "io/csv_handler.h"
#include "io/stat.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver2d.h"
#include "particle/particles_coordinate_map_2d.h"
#include "pe/concat/concat_solver2d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Tata Rao, Lanka & Goel, Sanket & Dubey, Satish & Javed, Arshad. (2019). Performance Investigation of T-Shaped
// Micromixer with Different Obstacles. Journal of Physics: Conference Series. 1276.
// 012003. 10.1088/1742-6596/1276/1/012003.

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
        std::cerr << "Error argument! Usage: program Re[double > 0]" << std::endl;
        return 0;
    }

    double Height = 0.2e-3;

    double lx1 = 5.0 * Height; // in reference paper no say
    double ly1 = Height;
    double lz1 = Height;

    double lx2 = Height;
    double ly2 = Height;

    double lx3 = lx1; // symmetry
    double ly3 = ly1; // symmetry

    double lx4 = Height;
    double ly4 = 5.0 * Height;

    double hx = Height / 10.0;
    double hy = Height / 10.0;
    double hz = Height / 10.0;

    int nx1 = lx1 / hx;
    int ny1 = ly1 / hy;
    int nx2 = lx2 / hx;
    int ny2 = ly2 / hy;
    int nx3 = lx3 / hx;
    int ny3 = ly3 / hy;
    int nx4 = lx4 / hx;
    int ny4 = ly4 / hy;

    double Re = std::stod(argv[1]);

    double dt = hx / 20.0;

    double density                           = 1e3;
    double dynamic_viscosity_1               = 26.46e-3;
    double dynamic_viscosity_2               = 1e-3;
    double mixing_channel_hydraulic_diameter = Height;
    double inlet_velocity                    = Re * dynamic_viscosity_2 / (density * mixing_channel_hydraulic_diameter);
    double kinematic_viscosity               = dynamic_viscosity_2 / density;

    double diffusion_coefficient = 3.23e-10;

    std::cout << "mixing_channel_hydraulic_diameter = " << mixing_channel_hydraulic_diameter << std::endl;
    std::cout << "inlet_velocity = " << inlet_velocity << std::endl;

    std::cout << "convection trem CFL = " << dt / hx << std::endl;

    // Geometry: Cross shape
    Geometry2D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    {
        std::stringstream ss;
        ss << "./result/T-shaped_mixer_obstacle_Uhlmann_Dirichlet/";
        ss << "Re=";
        ss << std::to_string((int)Re);
        ss << "TVD_VanLeer";
        env_cfg.debugOutputDir = ss.str();
    }

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = dt;
    time_cfg.num_iterations       = 4e5;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_nu(kinematic_viscosity);

    Domain2DUniform A1(nx1, ny1, lx1, ly1, "A1");
    Domain2DUniform A2(nx2, ny2, lx2, ly2, "A2");
    Domain2DUniform A3(nx3, ny3, lx3, ly3, "A3");
    Domain2DUniform A4(nx4, ny4, lx4, ly4, "A4");

    geo.add_domain(&A1);
    geo.add_domain(&A2);
    geo.add_domain(&A3);
    geo.add_domain(&A4);

    // Construct cross connectivity
    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);

    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);

    // Variable2Ds
    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    // Fields on each domain
    field2 u_A1, u_A2, u_A3, u_A4;
    field2 v_A1, v_A2, v_A3, v_A4;
    field2 p_A1, p_A2, p_A3, p_A4;

    u.set_x_edge_field(&A1, u_A1);
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    u.set_x_edge_field(&A4, u_A4);
    v.set_y_edge_field(&A1, v_A1);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    v.set_y_edge_field(&A4, v_A4);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);

    std::cout << "mesh num = " << u_A1.get_size_n() + u_A2.get_size_n() + u_A3.get_size_n() + u_A4.get_size_n()
              << std::endl;

    // Helper setters
    auto set_dirichlet_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto isdjacented = [&](Domain2DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4};
    std::vector<LocationType>     dirs    = {
        LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive};

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
        u.has_boundary_value_map[&A1][LocationType::XNegative] = true;
        u.has_boundary_value_map[&A3][LocationType::XPositive] = true;

        double* u_inlet_buffer_xneg = u.boundary_value_map[&A1][LocationType::XNegative];
        double* u_inlet_buffer_xpos = u.boundary_value_map[&A3][LocationType::XPositive];

        for (int j = 0; j < u_A1.get_ny(); ++j)
        {
            double y = j * hy + 0.5 * hy;
            y /= ly1;
            double vel             = 6.0 * inlet_velocity * (1.0 - y) * y;
            u_inlet_buffer_xneg[j] = vel;
            u_inlet_buffer_xpos[j] = -vel;
        }
    }
    // Outlet
    u.set_boundary_type(&A4, LocationType::YNegative, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A4, LocationType::YNegative, PDEBoundaryType::Neumann);

    add_random_number(u_A1, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);
    add_random_number(u_A2, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);
    add_random_number(u_A3, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);
    add_random_number(u_A4, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);

    add_random_number(v_A1, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);
    add_random_number(v_A2, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);
    add_random_number(v_A3, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);
    add_random_number(v_A4, -0.01 * inlet_velocity, 0.01 * inlet_velocity, 42);

    // IBM setup: sphere at T-junction center
    // Note: A2 starts at x=20*H/d, y=0, z=0
    double sphere_radius   = Height / 3.0;                                    // Radius = H/3
    double sphere_center_x = (A1.get_lx() + A2.get_lx() + A3.get_lx()) / 2.0; // Center of A2 domain
    double sphere_center_y = 0.0;                                             // T-junction y coordinate (within A2)

    std::cout << "IBM sphere (non-dim): center = (" << sphere_center_x << ", " << sphere_center_y
              << "), radius = " << sphere_radius << std::endl;

    PCoordMap2D coord_map;
    coord_map.add_cylinder(hx, sphere_radius, sphere_center_x, sphere_center_y);
    coord_map.generate_map(&geo);

    auto coord_map_raw = coord_map.get_map();

    // Velocity IBM solver
    IBVelocitySolver2D_Uhlmann ibm_solver(&u, &v, coord_map_raw);
    ibm_solver.set_parameters(coord_map.get_h(), hx);

    // Initialize IBM particle velocities to zero (solid sphere)
    for (auto& kv : coord_map_raw)
    {
        auto* p_coord = kv.second;
        auto* ib_data = ibm_solver.get_ib_data(kv.first);

        EXPOSE_PCOORD2D(p_coord)
        EXPOSE_PIB2D(ib_data)

        for (int i = 0; i < p_coord->cur_n; i++)
        {
            Up[i] = 0.0;
            Vp[i] = 0.0;
        }
    }

    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);

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

        ns_solver.euler_conv_diff_inner();
        ns_solver.euler_conv_diff_outer();

        ibm_solver.solve();

        ns_solver.phys_boundary_update();
        ns_solver.nondiag_shared_boundary_update();
        ns_solver.diag_shared_boundary_update();

        // divu
        ns_solver.velocity_div_inner();
        ns_solver.velocity_div_outer();

        // PE
        ns_solver.normalize_pressure();
        p_solver.solve();

        // update buffer for p
        ns_solver.pressure_buffer_update();

        // p grad
        ns_solver.add_pressure_gradient();

        // Boundary update
        ns_solver.phys_boundary_update();
        ns_solver.nondiag_shared_boundary_update();
        ns_solver.diag_shared_boundary_update();

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (iter % 100 == 0)
        {
            CSVHandler u_rms_file(env_cfg.debugOutputDir + "/u_rms");
            u_rms_file.stream << calc_rms(u) << std::endl;
        }

        if (std::isnan(u_A1(0, 0)))
        {
            std::cout << "Error: Find nan at u_A1! Break solving." << std::endl;
            break;
        }
    }

    std::cout << "Finished" << std::endl;
}