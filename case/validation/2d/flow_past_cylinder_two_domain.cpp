#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "ibm/ib_velocity_solver_2d_Uhlmann.h"
#include "ibm/particles_coordinate_map_2d.h"
#include "io/case_base.hpp"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"
#include "pe/concat/concat_solver2d.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

/**
 * Flow Past Cylinder - Two Domain Validation
 *
 * This case tests the IBM solver with a cylinder placed exactly on the boundary
 * between two domains along x-direction.
 *
 * Reference parameters from cylinder_case.hpp:
 * - Domain: 16 x 16, Grid: 1024 x 1024
 * - Cylinder center: (1.85, 3.2), radius: 0.15
 * - Reynolds number: 100
 * - Time step: 1/640
 * - Inlet velocity: 1.0
 *
 * Domain split:
 * - Left domain: x in [0, 1.859375] (119 grid points)
 * - Right domain: x in [1.859375, 16] (905 grid points)
 * - Cylinder center x = 1.85 is very close to the split boundary x = 1.859375
 */

class FlowPastCylinderTwoDomainCase : public CaseBase
{
public:
    FlowPastCylinderTwoDomainCase(int argc, char* argv[])
        : CaseBase(argc, argv)
    {}

    void read_paras() override
    {
        CaseBase::read_paras();

        // Grid parameters
        IO::read_number(para_map, "H", H);
        IO::read_number(para_map, "nx_total", nx_total);
        IO::read_number(para_map, "ny_total", ny_total);

        // Time stepping
        IO::read_number(para_map, "time_step", time_step);
        IO::read_number(para_map, "max_step", max_step);
        IO::read_number(para_map, "pv_output_step", pv_output_step);
        IO::read_number(para_map, "statistics_output_step", statistics_output_step);

        // Cylinder parameters
        IO::read_number(para_map, "cylinder_center_x", cylinder_center_x);
        IO::read_number(para_map, "cylinder_center_y", cylinder_center_y);
        IO::read_number(para_map, "cylinder_radius", cylinder_radius);

        // Physics parameters
        IO::read_number(para_map, "feature_fluid_velocity", feature_fluid_velocity);
        IO::read_number(para_map, "fluid_density", fluid_density);
        IO::read_number(para_map, "Reynolds_number", Reynolds_number);

        // IBM parameters
        IO::read_number(para_map, "ibm_repeat_number", ibm_repeat_number);
        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        // Calculate derived parameters
        Lx = nx_total * H;
        Ly = ny_total * H;

        dynamic_viscosity   = fluid_density * feature_fluid_velocity * cylinder_radius * 2.0 / Reynolds_number;
        kinematic_viscosity = dynamic_viscosity / fluid_density;

        // Domain split: find the grid index closest to cylinder center x
        int split_idx = static_cast<int>(std::round(cylinder_center_x / H));
        nx_left       = split_idx;
        nx_right      = nx_total - split_idx;
        split_x       = split_idx * H;

        // Update cylinder center to match actual split position (slight adjustment)
        cylinder_center_x = split_x;
    }

    bool record_paras() override
    {
        if (!CaseBase::record_paras())
            return false;

        paras_record.record("H", H)
            .record("nx_total", nx_total)
            .record("ny_total", ny_total)
            .record("Lx", Lx)
            .record("Ly", Ly)
            .record("time_step", time_step)
            .record("pv_output_step", pv_output_step)
            .record("statistics_output_step", statistics_output_step)
            .record("cylinder_center_x", cylinder_center_x)
            .record("cylinder_center_y", cylinder_center_y)
            .record("cylinder_radius", cylinder_radius)
            .record("feature_fluid_velocity", feature_fluid_velocity)
            .record("fluid_density", fluid_density)
            .record("Reynolds_number", Reynolds_number)
            .record("dynamic_viscosity", dynamic_viscosity)
            .record("kinematic_viscosity", kinematic_viscosity)
            .record("ibm_repeat_number", ibm_repeat_number)
            .record("nx_left", nx_left)
            .record("nx_right", nx_right)
            .record("split_x", split_x);

        return true;
    }

    // Grid parameters
    double H        = 1.0 / 64.0;
    int    nx_total = 512;
    int    ny_total = 512;
    double Lx = 8.0, Ly = 8.0;

    // Domain split
    int    nx_left  = 0;
    int    nx_right = 0;
    double split_x  = 0.0;

    // Time stepping
    double time_step              = 1.0 / 640.0;
    int    pv_output_step         = 1000;
    int    statistics_output_step = 100;

    // Cylinder parameters
    double cylinder_center_x = 1.85;
    double cylinder_center_y = 3.2;
    double cylinder_radius   = 0.15;

    // Physics parameters
    double feature_fluid_velocity = 1.0;
    double fluid_density          = 1.0;
    double Reynolds_number        = 100.0;
    double dynamic_viscosity      = 1.0;
    double kinematic_viscosity    = 1.0;

    // IBM parameters
    int    ibm_repeat_number = 1;
    int    gmres_m           = 20;
    double gmres_tol         = 1e-6;
    int    gmres_max_iter    = 100;
};

int main(int argc, char* argv[])
{
    FlowPastCylinderTwoDomainCase case_param(argc, argv);
    case_param.read_paras();

    // Configuration
    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showGmresRes       = false;
    env_cfg.showCurrentStep    = false;

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.time_step;
    time_cfg.set_t_max(case_param.time_step * case_param.max_step);

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Reynolds_number);

    case_param.record_paras();

    // Geometry setup
    Geometry2D geo;
    double     h = case_param.H;

    std::cout << "=== Flow Past Cylinder - Two Domain Validation ===\n";
    std::cout << "Computational domain: [" << 0 << "," << case_param.Lx << "] x [" << 0 << "," << case_param.Ly
              << "]\n";
    std::cout << "Grid size: " << case_param.nx_total << " x " << case_param.ny_total << "\n";
    std::cout << "Grid spacing: " << h << "\n";
    std::cout << "Domain split at x = " << case_param.split_x << "\n";
    std::cout << "Left domain: " << case_param.nx_left << " grid points\n";
    std::cout << "Right domain: " << case_param.nx_right << " grid points\n";
    std::cout << "Cylinder center: (" << case_param.cylinder_center_x << ", " << case_param.cylinder_center_y << ")\n";
    std::cout << "Cylinder radius: " << case_param.cylinder_radius << "\n";
    std::cout << "Reynolds number: " << case_param.Reynolds_number << "\n";
    std::cout << "Time step: " << case_param.time_step << "\n";
    std::cout << "Max steps: " << case_param.max_step << "\n\n";

    // Create domains
    double lx_left  = case_param.nx_left * h;
    double ly_left  = case_param.Ly;
    double lx_right = case_param.nx_right * h;
    double ly_right = case_param.Ly;

    Domain2DUniform d_left(case_param.nx_left, case_param.ny_total, lx_left, ly_left, "Left");
    Domain2DUniform d_right(case_param.nx_right, case_param.ny_total, lx_right, ly_right, "Right");

    geo.add_domain({&d_left, &d_right});
    geo.connect(&d_left, LocationType::XPositive, &d_right);
    geo.set_global_spatial_step(h, h);
    geo.axis(&d_left, LocationType::XNegative);

    // Variables
    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    // Fields
    field2 u_left, u_right;
    field2 v_left, v_right;
    field2 p_left, p_right;

    u.set_x_edge_field(&d_left, u_left);
    u.set_x_edge_field(&d_right, u_right);
    v.set_y_edge_field(&d_left, v_left);
    v.set_y_edge_field(&d_right, v_right);
    p.set_center_field(&d_left, p_left);
    p.set_center_field(&d_right, p_right);

    // Boundary conditions
    auto set_dirichlet = [](Variable2D& var, Domain2DUniform* d, LocationType loc, double value) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, value);
    };

    auto set_neumann = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };

    // Left boundary (inlet): u = U0, v = 0
    u.set_boundary_type(&d_left, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&d_left, LocationType::XNegative, 0.0);
    u.has_boundary_value_map[&d_left][LocationType::XNegative] = true;
    for (int j = 0; j < u_left.get_ny(); ++j)
    {
        double y_norm = (j + 0.5) / static_cast<double>(u_left.get_ny());
        double u_val  = 6.0 * case_param.feature_fluid_velocity * y_norm * (1.0 - y_norm);
        u.boundary_value_map[&d_left][LocationType::XNegative][j] = u_val;
    }
    set_dirichlet(v, &d_left, LocationType::XNegative, 0.0);

    // Right boundary (outlet): Neumann for all
    set_neumann(u, &d_right, LocationType::XPositive);
    set_neumann(v, &d_right, LocationType::XPositive);
    set_neumann(p, &d_right, LocationType::XPositive);

    // Top and bottom walls (no-slip)
    set_dirichlet(u, &d_left, LocationType::YNegative, 0.0);
    set_dirichlet(u, &d_left, LocationType::YPositive, 0.0);
    set_dirichlet(u, &d_right, LocationType::YNegative, 0.0);
    set_dirichlet(u, &d_right, LocationType::YPositive, 0.0);

    set_dirichlet(v, &d_left, LocationType::YNegative, 0.0);
    set_dirichlet(v, &d_left, LocationType::YPositive, 0.0);
    set_dirichlet(v, &d_right, LocationType::YNegative, 0.0);
    set_dirichlet(v, &d_right, LocationType::YPositive, 0.0);

    // Pressure boundaries
    set_neumann(p, &d_left, LocationType::XNegative);
    set_neumann(p, &d_left, LocationType::YNegative);
    set_neumann(p, &d_left, LocationType::YPositive);
    set_neumann(p, &d_right, LocationType::YNegative);
    set_neumann(p, &d_right, LocationType::YPositive);

    // IBM setup
    PCoordMap2D coord_map;
    coord_map.add_cylinder(h, case_param.cylinder_radius, case_param.cylinder_center_x, case_param.cylinder_center_y);
    coord_map.generate_map(&geo);

    auto coord_map_raw = coord_map.get_map();

    IBVelocitySolver2D_Uhlmann ibm_solver(&u, &v, coord_map_raw);
    ibm_solver.set_parameters(coord_map.get_h(), h);

    // Initialize IBM particle velocities to zero (solid cylinder)
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

    // Solvers
    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);

    ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

    // Output directory
    std::string nowtime_dir = case_param.root_dir;

    // Main time loop
    for (int step = 1; step <= case_param.max_step; step++)
    {
        SCOPE_TIMER("Iteration", TimeRecordType::None, step % 100 == 0);

        // Enable output periodically
        if (step % case_param.statistics_output_step == 0)
        {
            env_cfg.showGmresRes = true;
            std::cout << "Step: " << step << "/" << case_param.max_step << "\n";
        }
        else
        {
            env_cfg.showGmresRes = (step <= 5);
        }

        // Speed up transition to vortex street
        if (step == 1)
        {
            for (auto& kv : coord_map_raw)
            {
                auto* p_coord = kv.second;
                auto* ib_data = ibm_solver.get_ib_data(kv.first);

                EXPOSE_PCOORD2D(p_coord)
                EXPOSE_PIB2D(ib_data)

                for (int i = 0; i < p_coord->cur_n; i++)
                {
                    // Tangential velocity
                    Up[i] = (Y[i] - case_param.cylinder_center_y) / case_param.cylinder_radius;
                    Vp[i] = -(X[i] - case_param.cylinder_center_x) / case_param.cylinder_radius;
                }
            }
        }
        else if (step == 5)
        {
            for (auto& kv : coord_map_raw)
            {
                auto* ib_data = ibm_solver.get_ib_data(kv.first);
                EXPOSE_PIB2D(ib_data)
                for (int i = 0; i < ib_data->cur_n; i++)
                {
                    Up[i] = 0.0;
                    Vp[i] = 0.0;
                }
            }
        }

        ns_solver.euler_conv_diff_inner();
        ns_solver.euler_conv_diff_outer();

        // IBM solve (multiple iterations for better convergence)
        for (int ib_iter = 0; ib_iter < case_param.ibm_repeat_number; ib_iter++)
        {
            ibm_solver.solve();
        }

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

        // Output velocity fields
        if (step % case_param.pv_output_step == 0)
        {
            std::cout << "Saving step " << step << " to CSV files.\n";
            IO::write_csv(u, nowtime_dir + "/u/u_" + std::to_string(step));
            IO::write_csv(v, nowtime_dir + "/v/v_" + std::to_string(step));
        }

        // Check for divergence
        if (std::isnan(u_left(1, 1)))
        {
            std::cout << "=== DIVERGENCE ===" << std::endl;
            return -1;
        }
    }

    std::cout << "Simulation finished.\n";
    return 0;
}
