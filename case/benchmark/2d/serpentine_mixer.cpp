#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "base/math/random.h"
#include "io/case_base.hpp"
#include "io/csv_handler.h"
#include "io/csv_writer_2d.h"
#include "io/stat.h"
#include "ns/ns_solver2d.h"
#include "ns/scalar_solver2d.h"
#include "pe/concat/concat_solver2d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
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
 * └──────┼──────┼──────┴───►
 * O      │      │           x
 *        │  A4  │
 *        │      │
 *        ├──────┼──────┐
 *        │      │      │
 *        │  A5  │  A6  │
 *        │      │      │
 *        └──────┼──────┤
 *               │      │
 *               │  A7  │
 *               │      │
 *        ┌──────┼──────┤
 *        │      │      │
 *        │  A9  │  A8  │
 *        │      │      │
 *        ├──────┼──────┘
 *        │      │
 *        │  A10 │
 *        │      │
 *        └──────┘
 */
class TShapedMixerCase : public CaseBase
{
public:
    TShapedMixerCase(int argc, char* argv[])
        : CaseBase(argc, argv)
    {}

    void read_paras() override
    {
        CaseBase::read_paras();

        // Geometry parameters
        IO::read_number(para_map, "Height", Height);
        IO::read_number(para_map, "lx1_ratio", lx1_ratio);
        IO::read_number(para_map, "lx2_ratio", lx2_ratio);
        IO::read_number(para_map, "ly4_ratio", ly4_ratio);
        IO::read_number(para_map, "mesh_density", mesh_density);

        IO::read_number(para_map, "ly7_ratio", ly7_ratio);

        // Time stepping
        IO::read_number(para_map, "cfl", cfl);
        IO::read_number(para_map, "pv_output_step", pv_output_step);
        IO::read_number(para_map, "statistics_output_step", statistics_output_step);

        // Physics parameters
        IO::read_number(para_map, "Re", Reynolds_number);
        IO::read_number(para_map, "Sc", Schmidt_number);

        // Calculate derived parameters
        lx1 = lx1_ratio * Height;
        ly1 = Height;

        lx2 = lx2_ratio * Height;
        ly2 = Height;

        lx3 = lx1;
        ly3 = ly1;

        lx4 = lx2;
        ly4 = ly4_ratio * Height;

        lx5 = lx4;
        ly5 = lx4;

        lx6 = lx4;
        ly6 = ly5;

        lx7 = lx6;
        ly7 = ly7_ratio * Height;

        lx8 = lx7;
        ly8 = ly6;

        lx9 = lx5;
        ly9 = ly8;

        lx10 = lx9;
        ly10 = ly4;

        hx = Height / mesh_density;
        hy = Height / mesh_density;

        nx1  = static_cast<int>(lx1 / hx);
        ny1  = static_cast<int>(ly1 / hy);
        nx2  = static_cast<int>(lx2 / hx);
        ny2  = static_cast<int>(ly2 / hy);
        nx3  = static_cast<int>(lx3 / hx);
        ny3  = static_cast<int>(ly3 / hy);
        nx4  = static_cast<int>(lx4 / hx);
        ny4  = static_cast<int>(ly4 / hy);
        nx5  = static_cast<int>(lx5 / hx);
        ny5  = static_cast<int>(ly5 / hy);
        nx6  = static_cast<int>(lx6 / hx);
        ny6  = static_cast<int>(ly6 / hy);
        nx7  = static_cast<int>(lx7 / hx);
        ny7  = static_cast<int>(ly7 / hy);
        nx8  = static_cast<int>(lx8 / hx);
        ny8  = static_cast<int>(ly8 / hy);
        nx9  = static_cast<int>(lx9 / hx);
        ny9  = static_cast<int>(ly9 / hy);
        nx10 = static_cast<int>(lx10 / hx);
        ny10 = static_cast<int>(ly10 / hy);

        dt = cfl * hx;

        Pe = Schmidt_number * Reynolds_number;
        nr = 1.0 / Pe;

        double mixing_channel_hydraulic_diameter = 2.0 * lx2 * ly2 / (lx2 + ly2);
        double density                           = 1e3;
        double dynamic_viscosity                 = 1.01e-3;
        double inlet_velocity  = Reynolds_number * dynamic_viscosity / (density * mixing_channel_hydraulic_diameter);
        double convective_time = mixing_channel_hydraulic_diameter / inlet_velocity;

        paras_record.record("mixing_channel_hydraulic_diameter", mixing_channel_hydraulic_diameter)
            .record("density", density)
            .record("dynamic_viscosity", dynamic_viscosity)
            .record("inlet_velocity", inlet_velocity)
            .record("convective_time", convective_time);

        // Non-dimensionalize
        lx1 /= mixing_channel_hydraulic_diameter;
        ly1 /= mixing_channel_hydraulic_diameter;

        lx2 /= mixing_channel_hydraulic_diameter;
        ly2 /= mixing_channel_hydraulic_diameter;

        lx3 /= mixing_channel_hydraulic_diameter;
        ly3 /= mixing_channel_hydraulic_diameter;

        lx4 /= mixing_channel_hydraulic_diameter;
        ly4 /= mixing_channel_hydraulic_diameter;

        lx5 /= mixing_channel_hydraulic_diameter;
        ly5 /= mixing_channel_hydraulic_diameter;

        lx6 /= mixing_channel_hydraulic_diameter;
        ly6 /= mixing_channel_hydraulic_diameter;

        lx7 /= mixing_channel_hydraulic_diameter;
        ly7 /= mixing_channel_hydraulic_diameter;

        lx8 /= mixing_channel_hydraulic_diameter;
        ly8 /= mixing_channel_hydraulic_diameter;

        lx9 /= mixing_channel_hydraulic_diameter;
        ly9 /= mixing_channel_hydraulic_diameter;

        lx10 /= mixing_channel_hydraulic_diameter;
        ly10 /= mixing_channel_hydraulic_diameter;

        hx /= mixing_channel_hydraulic_diameter;
        hy /= mixing_channel_hydraulic_diameter;

        dt /= convective_time;
    }

    bool record_paras() override
    {
        if (!CaseBase::record_paras())
            return false;

        paras_record.record("Height", Height)
            .record("lx1_ratio", lx1_ratio)
            .record("lx2_ratio", lx2_ratio)
            .record("ly4_ratio", ly4_ratio)
            .record("ly7_ratio", ly7_ratio)
            .record("mesh_density", mesh_density)
            .record("lx1", lx1)
            .record("ly1", ly1)
            .record("lx2", lx2)
            .record("ly2", ly2)
            .record("lx3", lx3)
            .record("ly3", ly3)
            .record("lx4", lx4)
            .record("ly4", ly4)
            .record("lx5", lx5)
            .record("ly5", ly5)
            .record("lx6", lx6)
            .record("ly6", ly6)
            .record("lx7", lx7)
            .record("ly7", ly7)
            .record("lx8", lx8)
            .record("ly8", ly8)
            .record("lx9", lx9)
            .record("ly9", ly9)
            .record("lx10", lx10)
            .record("ly10", ly10)
            .record("hx", hx)
            .record("hy", hy)
            .record("nx1", nx1)
            .record("ny1", ny1)
            .record("nx2", nx2)
            .record("ny2", ny2)
            .record("nx3", nx3)
            .record("ny3", ny3)
            .record("nx4", nx4)
            .record("ny4", ny4)
            .record("nx5", nx5)
            .record("ny5", ny5)
            .record("nx6", nx6)
            .record("ny6", ny6)
            .record("nx7", nx7)
            .record("ny7", ny7)
            .record("nx8", nx8)
            .record("ny8", ny8)
            .record("nx9", nx9)
            .record("ny9", ny9)
            .record("nx10", nx10)
            .record("ny10", ny10)
            .record("cfl", cfl)
            .record("dt", dt)
            .record("Reynolds_number", Reynolds_number)
            .record("Schmidt_number", Schmidt_number)
            .record("Pe", Pe)
            .record("nr", nr)
            .record("pv_output_step", pv_output_step)
            .record("statistics_output_step", statistics_output_step);

        return true;
    }

    // Geometry parameters
    double Height       = 1e-3;
    double lx1_ratio    = 5.0;
    double lx2_ratio    = 1.0;
    double ly4_ratio    = 5.0;
    double ly7_ratio    = 1.0;
    int    mesh_density = 20;

    // Derived geometry
    double lx1, ly1;
    double lx2, ly2;
    double lx3, ly3;
    double lx4, ly4;

    double lx5, ly5;
    double lx6, ly6;
    double lx7, ly7;
    double lx8, ly8;
    double lx9, ly9;
    double lx10, ly10;

    double hx, hy;
    int    nx1, ny1;
    int    nx2, ny2;
    int    nx3, ny3;
    int    nx4, ny4;

    int nx5, ny5;
    int nx6, ny6;
    int nx7, ny7;
    int nx8, ny8;
    int nx9, ny9;
    int nx10, ny10;

    // Time stepping
    double cfl                    = 0.1;
    double dt                     = 0.0;
    int    pv_output_step         = 10000;
    int    statistics_output_step = 20;

    // Physics parameters
    double Reynolds_number = 100.0;
    double Schmidt_number  = 0.1;
    double Pe              = 0.0;
    double nr              = 0.0;
};

int main(int argc, char* argv[])
{
    TShapedMixerCase case_param(argc, argv);
    case_param.read_paras();

    // Configuration
    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showGmresRes       = false;
    env_cfg.showCurrentStep    = false;

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.dt;
    time_cfg.num_iterations       = case_param.max_step;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Reynolds_number);

    case_param.record_paras();

    // Geometry: Cross shape
    Geometry2D geo;

    std::cout << "=== T-shaped Mixer ===\n";
    std::cout << "Domain dimensions (non-dim):\n";
    std::cout << "  A1: " << case_param.lx1 << " x " << case_param.ly1 << "\n";
    std::cout << "  A2: " << case_param.lx2 << " x " << case_param.ly2 << "\n";
    std::cout << "  A3: " << case_param.lx3 << " x " << case_param.ly3 << "\n";
    std::cout << "  A4: " << case_param.lx4 << " x " << case_param.ly4 << "\n";
    std::cout << "  A5: " << case_param.lx5 << " x " << case_param.ly5 << "\n";
    std::cout << "  A6: " << case_param.lx6 << " x " << case_param.ly6 << "\n";
    std::cout << "  A7: " << case_param.lx7 << " x " << case_param.ly7 << "\n";
    std::cout << "  A8: " << case_param.lx8 << " x " << case_param.ly8 << "\n";
    std::cout << "  A9: " << case_param.lx9 << " x " << case_param.ly9 << "\n";
    std::cout << "  A10: " << case_param.lx10 << " x " << case_param.ly10 << "\n";
    std::cout << "Grid: " << case_param.nx1 << "x" << case_param.ny1 << " (A1)\n";
    std::cout << "      " << case_param.nx2 << "x" << case_param.ny2 << " (A2)\n";
    std::cout << "      " << case_param.nx3 << "x" << case_param.ny3 << " (A3)\n";
    std::cout << "      " << case_param.nx4 << "x" << case_param.ny4 << " (A4)\n";
    std::cout << "      " << case_param.nx5 << "x" << case_param.ny5 << " (A5)\n";
    std::cout << "      " << case_param.nx6 << "x" << case_param.ny6 << " (A6)\n";
    std::cout << "      " << case_param.nx7 << "x" << case_param.ny7 << " (A7)\n";
    std::cout << "      " << case_param.nx8 << "x" << case_param.ny8 << " (A8)\n";
    std::cout << "      " << case_param.nx9 << "x" << case_param.ny9 << " (A9)\n";
    std::cout << "      " << case_param.nx10 << "x" << case_param.ny10 << " (A10)\n";
    std::cout << "Grid spacing: " << case_param.hx << " x " << case_param.hy << "\n";
    std::cout << "Re = " << case_param.Reynolds_number << ", Sc = " << case_param.Schmidt_number << "\n";
    std::cout << "dt = " << case_param.dt << ", max_step = " << case_param.max_step << "\n\n";

    Domain2DUniform A1(case_param.nx1, case_param.ny1, case_param.lx1, case_param.ly1, "A1");
    Domain2DUniform A2(case_param.nx2, case_param.ny2, case_param.lx2, case_param.ly2, "A2");
    Domain2DUniform A3(case_param.nx3, case_param.ny3, case_param.lx3, case_param.ly3, "A3");
    Domain2DUniform A4(case_param.nx4, case_param.ny4, case_param.lx4, case_param.ly4, "A4");
    Domain2DUniform A5(case_param.nx5, case_param.ny5, case_param.lx5, case_param.ly5, "A5");
    Domain2DUniform A6(case_param.nx6, case_param.ny6, case_param.lx6, case_param.ly6, "A6");
    Domain2DUniform A7(case_param.nx7, case_param.ny7, case_param.lx7, case_param.ly7, "A7");
    Domain2DUniform A8(case_param.nx8, case_param.ny8, case_param.lx8, case_param.ly8, "A8");
    Domain2DUniform A9(case_param.nx9, case_param.ny9, case_param.lx9, case_param.ly9, "A9");
    Domain2DUniform A10(case_param.nx10, case_param.ny10, case_param.lx10, case_param.ly10, "A10");

    geo.add_domain(&A1);
    geo.add_domain(&A2);
    geo.add_domain(&A3);
    geo.add_domain(&A4);
    geo.add_domain(&A5);
    geo.add_domain(&A6);
    geo.add_domain(&A7);
    geo.add_domain(&A8);
    geo.add_domain(&A9);
    geo.add_domain(&A10);

    // Construct cross connectivity
    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);
    geo.connect(&A4, LocationType::YNegative, &A5);
    geo.connect(&A5, LocationType::XPositive, &A6);
    geo.connect(&A6, LocationType::YNegative, &A7);
    geo.connect(&A7, LocationType::YNegative, &A8);
    geo.connect(&A8, LocationType::XNegative, &A9);
    geo.connect(&A9, LocationType::YNegative, &A10);

    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);

    // Variable2Ds
    Variable2D u("u"), v("v"), p("p"), c("concentration");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);
    c.set_geometry(geo);

    // Fields on each domain
    field2 u_A1, u_A2, u_A3, u_A4, u_A5, u_A6, u_A7, u_A8, u_A9, u_A10;
    field2 v_A1, v_A2, v_A3, v_A4, v_A5, v_A6, v_A7, v_A8, v_A9, v_A10;
    field2 p_A1, p_A2, p_A3, p_A4, p_A5, p_A6, p_A7, p_A8, p_A9, p_A10;
    field2 c_A1, c_A2, c_A3, c_A4, c_A5, c_A6, c_A7, c_A8, c_A9, c_A10;

    u.set_x_edge_field(&A1, u_A1);
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    u.set_x_edge_field(&A4, u_A4);
    u.set_x_edge_field(&A5, u_A5);
    u.set_x_edge_field(&A6, u_A6);
    u.set_x_edge_field(&A7, u_A7);
    u.set_x_edge_field(&A8, u_A8);
    u.set_x_edge_field(&A9, u_A9);
    u.set_x_edge_field(&A10, u_A10);

    v.set_y_edge_field(&A1, v_A1);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    v.set_y_edge_field(&A4, v_A4);
    v.set_y_edge_field(&A4, v_A4);
    v.set_y_edge_field(&A5, v_A5);
    v.set_y_edge_field(&A6, v_A6);
    v.set_y_edge_field(&A7, v_A7);
    v.set_y_edge_field(&A8, v_A8);
    v.set_y_edge_field(&A9, v_A9);
    v.set_y_edge_field(&A10, v_A10);

    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);
    p.set_center_field(&A5, p_A5);
    p.set_center_field(&A6, p_A6);
    p.set_center_field(&A7, p_A7);
    p.set_center_field(&A8, p_A8);
    p.set_center_field(&A9, p_A9);
    p.set_center_field(&A10, p_A10);

    c.set_center_field(&A1, c_A1);
    c.set_center_field(&A2, c_A2);
    c.set_center_field(&A3, c_A3);
    c.set_center_field(&A4, c_A4);
    c.set_center_field(&A5, c_A5);
    c.set_center_field(&A6, c_A6);
    c.set_center_field(&A7, c_A7);
    c.set_center_field(&A8, c_A8);
    c.set_center_field(&A9, c_A9);
    c.set_center_field(&A10, c_A10);

    std::cout << "mesh num = "
              << u_A1.get_size_n() + u_A2.get_size_n() + u_A3.get_size_n() + u_A4.get_size_n() + u_A5.get_size_n() +
                     u_A6.get_size_n() + u_A7.get_size_n() + u_A8.get_size_n() + u_A9.get_size_n() + u_A10.get_size_n()
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
    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5, &A6, &A7, &A8, &A9, &A10};
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
            set_dirichlet_zero(c, d, loc);
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

            double y = j * case_param.hy + 0.5 * case_param.hy;
            y /= case_param.ly1;
            double vel             = 6.0 * (1.0 - y) * y;
            u_inlet_buffer_xneg[j] = vel;
            u_inlet_buffer_xpos[j] = -vel;
        }

        c.has_boundary_value_map[&A3][LocationType::XPositive] = true;

        double* c_inlet_buffer_xpos = c.boundary_value_map[&A3][LocationType::XPositive];

        for (int j = 0; j < c_A3.get_ny(); ++j)
        {
            c_inlet_buffer_xpos[j] = 1.0;
        }
    }
    // Outlet
    u.set_boundary_type(&A10, LocationType::YNegative, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A10, LocationType::YNegative, PDEBoundaryType::Neumann);
    c.set_boundary_type(&A10, LocationType::YNegative, PDEBoundaryType::Neumann);

    add_random_number(u_A1, -0.01, 0.01, 42);
    add_random_number(u_A2, -0.01, 0.01, 42);
    add_random_number(u_A3, -0.01, 0.01, 42);
    add_random_number(u_A4, -0.01, 0.01, 42);
    add_random_number(u_A5, -0.01, 0.01, 42);
    add_random_number(u_A6, -0.01, 0.01, 42);
    add_random_number(u_A7, -0.01, 0.01, 42);
    add_random_number(u_A8, -0.01, 0.01, 42);
    add_random_number(u_A9, -0.01, 0.01, 42);
    add_random_number(u_A10, -0.01, 0.01, 42);

    add_random_number(v_A1, -0.01, 0.01, 42);
    add_random_number(v_A2, -0.01, 0.01, 42);
    add_random_number(v_A3, -0.01, 0.01, 42);
    add_random_number(v_A4, -0.01, 0.01, 42);
    add_random_number(v_A5, -0.01, 0.01, 42);
    add_random_number(v_A6, -0.01, 0.01, 42);
    add_random_number(v_A7, -0.01, 0.01, 42);
    add_random_number(v_A8, -0.01, 0.01, 42);
    add_random_number(v_A9, -0.01, 0.01, 42);
    add_random_number(v_A10, -0.01, 0.01, 42);

    DifferenceSchemeType c_scheme = DifferenceSchemeType::Conv_QUICK_Diff_Center2nd;

    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);
    ScalarSolver2D        solver_c(&u, &v, &c, case_param.nr, c_scheme);

    auto calc_MI = [&](int j) {
        double c_mean = c_A10.mean_at_y_axis(j);
        double sigma  = 0.0;
        OPENMP_PARALLEL_FOR(reduction(+ : sigma))
        for (int i = 0; i < case_param.nx10; i++)
            sigma += (c_A10(i, j) - c_mean) * (c_A10(i, j) - c_mean);
        sigma = std::sqrt(sigma / case_param.nx10);
        return 1 - sigma / 0.5;
    };

    for (int iter = 0; iter <= time_cfg.num_iterations; iter++)
    {
        SCOPE_TIMER("Iteration", TimeRecordType::None, iter % 100 == 0);

        if (iter % 100 == 0)
        {
            std::cout << "iter: " << iter << "/" << time_cfg.num_iterations << "\n";

            env_cfg.track_pe_solve_detail_time = true;
            env_cfg.showGmresRes               = true;
        }

        ns_solver.solve();
        solver_c.solve();

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (iter % static_cast<int>(1e4) == 0)
        {
            IO::write_csv(u, case_param.root_dir + '/' + std::to_string(iter) + "/u");
            IO::write_csv(v, case_param.root_dir + '/' + std::to_string(iter) + "/v");
            IO::write_csv(c, case_param.root_dir + '/' + std::to_string(iter) + "/c");

            IO::matlab_read_var(u, case_param.root_dir + '/' + std::to_string(iter) + "/u.m");
            IO::matlab_read_var(v, case_param.root_dir + '/' + std::to_string(iter) + "/v.m");
            IO::matlab_read_var(c, case_param.root_dir + '/' + std::to_string(iter) + "/c.m");
        }

        if (iter % case_param.statistics_output_step == 0)
        {
            CSVHandler u_rms_file(case_param.root_dir + "/u_rms");
            u_rms_file.stream << calc_rms(u) << std::endl;

            CSVHandler c_rms_file(case_param.root_dir + "/c_rms");
            c_rms_file.stream << calc_rms(c) << std::endl;

            CSVHandler MI_file(case_param.root_dir + "/MI");
            for (int j = case_param.ny10 - 1; j >= 0; j--)
            {
                MI_file.stream << calc_MI(j);
                if (j != 0)
                    MI_file.stream << ',';
                else
                    MI_file.stream << std::endl;
            }
        }

        if (std::isnan(u_A1(0, 0)))
        {
            std::cout << "Error: Find nan at u_A1! Break solving." << std::endl;
            break;
        }

        if (std::isnan(c_A1(0, 0)))
        {
            std::cout << "Error: Find nan at c_A1! Break solving." << std::endl;
            break;
        }
    }

    std::cout << "Finished" << std::endl;
}
