#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "case/cross_shaped_channel_2d.h"
#include "instrumentor/timer.h"
#include "io/common.h"
#include "io/config.h"
#include "io/csv_writer_2d.h"
#include "ns/mhd_module_2d.h"
#include "ns/ns_solver2d.h"
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
 * │      ┌──────┐
 * │      │      │
 * │      │  A5  │
 * │      │      │
 * ├──────┼──────┼──────┐
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
 * MHD + Non-Newtonian fluid simulation for cross-shaped channel
 */
int main(int argc, char* argv[])
{
    // 参数读取与物理配置
    CrossShapedChannel2DCase case_param(argc, argv);
    case_param.read_paras();
    case_param.record_paras();

    Geometry2D geo_cross;
    double     h = case_param.h;

    EnvironmentConfig& env_cfg    = EnvironmentConfig::Get();
    env_cfg.showGmresRes          = false;
    env_cfg.showCurrentStep       = false;
    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.dt_factor * h; // CFL condition
    time_cfg.set_t_max(case_param.T_total);
    // time_cfg.num_iterations   = 10;
    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Re);

    // MHD parameters (must be set before MHDModule2D initialization)
    physics_cfg.Ha = case_param.Ha;
    physics_cfg.Bx = case_param.Bx;
    physics_cfg.By = case_param.By;

    std::cout << "MHD Parameters:" << std::endl;
    std::cout << "  Ha: " << case_param.Ha << std::endl;
    std::cout << "  Bx: " << case_param.Bx << std::endl;
    std::cout << "  By: " << case_param.By << std::endl;

    // Set Non-Newtonian parameters based on model type
    physics_cfg.set_model_type(case_param.model_type);
    if (case_param.model_type == 1) // Power Law
    {
        // Use dimensionless setter if Re_PL is provided in config (which it is by default in class)
        // Note: You can add logic to choose between dimensional and dimensionless if needed.
        // Here we prioritize dimensionless for this task.
        physics_cfg.set_power_law_dimensionless(case_param.Re_PL, case_param.n_index);
        std::cout << "Configuring Power Law Model (Dimensionless):" << std::endl;
        std::cout << "  Re_PL: " << case_param.Re_PL << std::endl;
        std::cout << "  n:     " << case_param.n_index << std::endl;
    }
    else if (case_param.model_type == 2) // Carreau
    {
        physics_cfg.set_carreau_dimensionless(
            case_param.Re_0, case_param.Re_inf, case_param.Wi, case_param.a, case_param.n_index);
        std::cout << "Configuring Carreau Model (Dimensionless):" << std::endl;
        std::cout << "  Re_0:   " << case_param.Re_0 << std::endl;
        std::cout << "  Re_inf: " << case_param.Re_inf << std::endl;
        std::cout << "  Wi:     " << case_param.Wi << std::endl;
        std::cout << "  a:      " << case_param.a << std::endl;
        std::cout << "  n:      " << case_param.n_index << std::endl;
    }

    // 计算循环输出步数间隔 pv_output_step（如果未指定则使用 num_iterations/10）
    int pv_output_step = case_param.pv_output_step > 0 ? case_param.pv_output_step : time_cfg.num_iterations / 10;
    // 计算最终保存步数（如果未指定则使用 num_iterations）
    int final_step_to_save = case_param.step_to_save > 0 ? case_param.step_to_save : time_cfg.num_iterations;

    double lx2 = case_param.lx_2;
    double ly2 = case_param.ly_2;
    double lx1 = case_param.lx_1;
    double lx3 = case_param.lx_3;
    double ly4 = case_param.ly_4;
    double ly5 = case_param.ly_5;

    int nx2 = lx2 / h;
    int ny2 = ly2 / h;
    int nx1 = lx1 / h;
    int nx3 = lx3 / h;
    int ny4 = ly4 / h;
    int ny5 = ly5 / h;

    int ny1 = ny2;
    int ny3 = ny2;
    int nx4 = nx2;
    int nx5 = nx2;
    int ly1 = ly2;
    int ly3 = ly2;
    int lx4 = lx2;
    int lx5 = lx2;

    std::cout << "Construct cross-shaped channel geometry (MHD + Non-Newtonian)..." << std::endl;
    std::cout << "Domain A2: " << nx2 << " x " << ny2 << " (" << lx2 << " x " << ly2 << ")" << std::endl;
    std::cout << "Domain A1: " << nx1 << " x " << ny1 << " (" << lx1 << " x " << ly1 << ")" << std::endl;
    std::cout << "Domain A3: " << nx3 << " x " << ny3 << " (" << lx3 << " x " << ly3 << ")" << std::endl;
    std::cout << "Domain A4: " << nx4 << " x " << ny4 << " (" << lx4 << " x " << ly4 << ")" << std::endl;
    std::cout << "Domain A5: " << nx5 << " x " << ny5 << " (" << lx5 << " x " << ly5 << ")" << std::endl;

    // 输出总网格数
    int total_grid = nx1 * ny1 + nx2 * ny2 + nx3 * ny3 + nx4 * ny4 + nx5 * ny5;
    std::cout << "Total grid points: " << total_grid << std::endl;

    Domain2DUniform A2(nx2, ny2, lx2, ly2, "A2");
    Domain2DUniform A1(nx1, ny1, lx1, ly1, "A1");
    Domain2DUniform A3(nx3, ny3, lx3, ly3, "A3");
    Domain2DUniform A4(nx4, ny4, lx4, ly4, "A4");
    Domain2DUniform A5(nx5, ny5, lx5, ly5, "A5");
    // geo_cross.add_domain(&A1);
    // geo_cross.add_domain(&A2);
    // geo_cross.add_domain(&A3);
    // geo_cross.add_domain(&A4);
    // geo_cross.add_domain(&A5);

    // Construct cross connectivity
    geo_cross.connect(&A2, LocationType::Left, &A1);
    geo_cross.connect(&A2, LocationType::Right, &A3);
    geo_cross.connect(&A2, LocationType::Down, &A4);
    geo_cross.connect(&A2, LocationType::Up, &A5);

    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo_cross);
    v.set_geometry(geo_cross);
    p.set_geometry(geo_cross);

    // Electric potential variable (center type, like pressure p)
    Variable2D phi("phi");
    phi.set_geometry(geo_cross);

    // Non-Newtonian Variable2Ds
    Variable2D mu("mu"), tau_xx("tau_xx"), tau_yy("tau_yy"), tau_xy("tau_xy");
    mu.set_geometry(geo_cross);
    tau_xx.set_geometry(geo_cross);
    tau_yy.set_geometry(geo_cross);
    tau_xy.set_geometry(geo_cross);

    // Fields on each domain
    field2 u_A1, u_A2, u_A3, u_A4, u_A5;
    field2 v_A1, v_A2, v_A3, v_A4, v_A5;
    field2 p_A1, p_A2, p_A3, p_A4, p_A5;

    u.set_x_edge_field(&A1, u_A1);
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    u.set_x_edge_field(&A4, u_A4);
    u.set_x_edge_field(&A5, u_A5);
    v.set_y_edge_field(&A1, v_A1);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    v.set_y_edge_field(&A4, v_A4);
    v.set_y_edge_field(&A5, v_A5);
    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);
    p.set_center_field(&A5, p_A5);

    // Electric potential fields (center type)
    field2 phi_A1("phi_A1"), phi_A2("phi_A2"), phi_A3("phi_A3"), phi_A4("phi_A4"), phi_A5("phi_A5");
    phi.set_center_field(&A1, phi_A1);
    phi.set_center_field(&A2, phi_A2);
    phi.set_center_field(&A3, phi_A3);
    phi.set_center_field(&A4, phi_A4);
    phi.set_center_field(&A5, phi_A5);

    // Non-Newtonian Fields
    field2 mu_A1("mu_A1"), mu_A2("mu_A2"), mu_A3("mu_A3"), mu_A4("mu_A4"), mu_A5("mu_A5");
    field2 txx_A1("txx_A1"), txx_A2("txx_A2"), txx_A3("txx_A3"), txx_A4("txx_A4"), txx_A5("txx_A5");
    field2 tyy_A1("tyy_A1"), tyy_A2("tyy_A2"), tyy_A3("tyy_A3"), tyy_A4("tyy_A4"), tyy_A5("tyy_A5");
    field2 txy_A1("txy_A1"), txy_A2("txy_A2"), txy_A3("txy_A3"), txy_A4("txy_A4"), txy_A5("txy_A5");

    mu.set_corner_field(&A1, mu_A1);
    mu.set_corner_field(&A2, mu_A2);
    mu.set_corner_field(&A3, mu_A3);
    mu.set_corner_field(&A4, mu_A4);
    mu.set_corner_field(&A5, mu_A5);

    tau_xx.set_center_field(&A1, txx_A1);
    tau_xx.set_center_field(&A2, txx_A2);
    tau_xx.set_center_field(&A3, txx_A3);
    tau_xx.set_center_field(&A4, txx_A4);
    tau_xx.set_center_field(&A5, txx_A5);

    tau_yy.set_center_field(&A1, tyy_A1);
    tau_yy.set_center_field(&A2, tyy_A2);
    tau_yy.set_center_field(&A3, tyy_A3);
    tau_yy.set_center_field(&A4, tyy_A4);
    tau_yy.set_center_field(&A5, tyy_A5);

    tau_xy.set_corner_field(&A1, txy_A1);
    tau_xy.set_corner_field(&A2, txy_A2);
    tau_xy.set_corner_field(&A3, txy_A3);
    tau_xy.set_corner_field(&A4, txy_A4);
    tau_xy.set_corner_field(&A5, txy_A5);

    // Helper setters
    auto set_dirichlet_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
    };
    auto is_adjacented = [&](Domain2DUniform* d, LocationType loc) {
        return geo_cross.adjacency.count(d) && geo_cross.adjacency[d].count(loc);
    };
    // Default outer boundaries
    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};
    std::vector<LocationType> dirs = {LocationType::Left, LocationType::Right, LocationType::Down, LocationType::Up};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(d, loc))
                continue; // internal boundaries handled automatically
            // velocity: default wall (Dirichlet 0)
            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            // pressure: default Neumann (zero gradient)
            set_neumann_zero(p, d, loc);
        }
    }

    // ========== Phi boundary conditions ==========
    // Default: Neumann (zero gradient) for all physical boundaries (walls)
    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(d, loc))
                continue; // skip internal connections (Adjacented is set by set_geometry)
            // Set Neumann BC for walls
            set_neumann_zero(phi, d, loc);
        }
    }

    // Inlet Dirichlet: A1 Left, phi = 0
    phi.set_boundary_type(&A1, LocationType::Left, PDEBoundaryType::Dirichlet);
    phi.set_boundary_value(&A1, LocationType::Left, 0.0);
    phi.has_boundary_value_map[&A1][LocationType::Left] = true;

    // Outlet Dirichlet: A3 Right, phi = 0
    phi.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Dirichlet);
    phi.set_boundary_value(&A3, LocationType::Right, 0.0);
    phi.has_boundary_value_map[&A3][LocationType::Right] = true;

    // Inlet profiles for symmetry validation (Poiseuille)
    const double U0 = case_param.U0;

    u.set_boundary_type(&A1, LocationType::Left, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A1, LocationType::Left, 0.0); // ← 添加这行来分配内存
    u.has_boundary_value_map[&A1][LocationType::Left] = true;
    set_dirichlet_zero(v, &A1, LocationType::Left);
    // A1 Left: u(y_norm) = +6*U0*y_norm*(1-y_norm)
    for (int j = 0; j < u_A1.get_ny(); ++j)
    {
        double y_norm                                    = (j + 0.5) / static_cast<double>(u_A1.get_ny());
        double u_val                                     = 6.0 * U0 * y_norm * (1.0 - y_norm);
        u.boundary_value_map[&A1][LocationType::Left][j] = u_val;
    }
    set_dirichlet_zero(v, &A1, LocationType::Left);

    // A3 Right: u(y_norm) = -6*U0*y_norm*(1-y_norm)
    u.set_boundary_type(&A3, LocationType::Right, PDEBoundaryType::Dirichlet);
    u.has_boundary_value_map[&A3][LocationType::Right] = true;
    u.set_boundary_value(&A3, LocationType::Right, 0.0); // ← 添加这行来分配内存
    set_dirichlet_zero(v, &A3, LocationType::Right);
    for (int j = 0; j < u_A3.get_ny(); ++j)
    {
        double y_norm                                     = (j + 0.5) / static_cast<double>(u_A3.get_ny());
        double u_val                                      = -6.0 * U0 * y_norm * (1.0 - y_norm);
        u.boundary_value_map[&A3][LocationType::Right][j] = u_val;
    }

    // A4 Down: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A4, LocationType::Down);
    set_neumann_zero(v, &A4, LocationType::Down);

    // A5 Up: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A5, LocationType::Up);
    set_neumann_zero(v, &A5, LocationType::Up);

    ConcatNSSolver2D ns_solver(&u, &v, &p);
    ns_solver.init_nonnewton(&mu, &tau_xx, &tau_yy, &tau_xy);

    ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

    // ========== Initialize MHD module ==========
    MHDModule2D mhd_module(&u, &v);
    mhd_module.init(&phi); // Pass pre-configured phi variable

    // Generate timestamp directory
    std::string nowtime_dir = case_param.root_dir;

    // 注册计时器用于输出
    TimerSingleton::Get().RegisterStdCout("step_time");

    std::cout << "Starting MHD + Non-Newtonian simulation..." << std::endl;

    // Solve
    for (int step = 1; step <= time_cfg.num_iterations; step++)
    {
        // 每200步启用计时输出
        if (step % 200 == 0)
        {
            TimerSingleton::Get().EnableStdCout(true);
            env_cfg.showGmresRes = true;
            std::cout << "step: " << step << "/" << time_cfg.num_iterations << "\n";
        }
        else
        {
            TimerSingleton::Get().EnableStdCout(false);
            env_cfg.showGmresRes = (step <= 5);
        }

        {
            Timer step_timer("step_time");
            // Non-Newtonian solve
            ns_solver.solve_nonnewton();

            // MHD solve
            mhd_module.solveElectricPotential();
            mhd_module.updateCurrentDensity();
            mhd_module.applyLorentzForce();
        }

        // 使用 pv_output_step 控制循环输出
        if (step % pv_output_step == 0)
        {
            std::cout << "Saving step " << (step) << " to CSV files." << std::endl;
            // update boundary for NS
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
            IO::write_csv(u, nowtime_dir + "/u/u_" + std::to_string(step));
            IO::write_csv(v, nowtime_dir + "/v/v_" + std::to_string(step));
            IO::write_csv(p, nowtime_dir + "/p/p_" + std::to_string(step));
            IO::write_csv(mu, nowtime_dir + "/mu/mu_" + std::to_string(step));
            IO::write_csv(phi, nowtime_dir + "/phi/phi_" + std::to_string(step));
        }
        if (std::isnan(u_A1(1, 1)))
        {
            std::cout << "=== DIVERGENCE ===" << std::endl;
            return -1;
        }
    }
    std::cout << "Simulation finished." << std::endl;
    // 使用 step_to_save 控制最终保存
    IO::write_csv(u, nowtime_dir + "/final/u_" + std::to_string(final_step_to_save));
    IO::write_csv(v, nowtime_dir + "/final/v_" + std::to_string(final_step_to_save));
    IO::write_csv(p, nowtime_dir + "/final/p_" + std::to_string(final_step_to_save));
    IO::write_csv(mu, nowtime_dir + "/final/mu_" + std::to_string(final_step_to_save));
    IO::write_csv(phi, nowtime_dir + "/final/phi_" + std::to_string(final_step_to_save));
    return 0;
}
