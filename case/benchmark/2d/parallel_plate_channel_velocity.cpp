#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/common.h"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"
#include "parallel_plate_channel.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

/**
 * @brief Power-Law Analytical Solution for Parallel Plate Channel Flow
 *
 * u(y) = n/(n+1) * ( (-dp/dx)/K )^(1/n) * [ H^((n+1)/n) - |y|^((n+1)/n) ]
 */
double power_law_analytical(double y, double n, double dp_dx, double k_pl, double H)
{
    if (std::abs(y) > H)
        return 0.0;

    double K             = k_pl;
    double pressure_term = -dp_dx / K;
    if (pressure_term < 0)
        pressure_term = 0.0;

    double exponent       = (n + 1.0) / n;
    double prefactor      = (n / (n + 1.0)) * std::pow(pressure_term, 1.0 / n);
    double geometric_term = std::pow(H, exponent) - std::pow(std::abs(y), exponent);

    return prefactor * geometric_term;
}

/**
 * @brief 计算 Variable 的相对残差（L2 范数）
 * @param var 当前速度场 Variable
 * @param prev_map 上一时间步速度场的 field2 映射
 * @return 相对残差值
 */
double compute_velocity_residual(Variable2D& var, std::unordered_map<Domain2DUniform*, field2>& prev_map)
{
    double total_diff_sq = 0.0;
    double total_norm_sq = 0.0;

    for (auto* domain : var.geometry->domains)
    {
        field2& curr = *var.field_map[domain];
        field2& prev = prev_map[domain];

        // 使用 field2 减法和 squared_sum
        field2 diff = curr - prev;
        total_diff_sq += diff.squared_sum();
        total_norm_sq += curr.squared_sum();
    }

    return (total_norm_sq > 1e-14) ? std::sqrt(total_diff_sq / total_norm_sq) : std::sqrt(total_diff_sq);
}

/**
 * @brief 更新前一时间步速度场
 * @param var 当前速度场 Variable
 * @param prev_map 上一时间步速度场的 field2 映射（将被更新）
 */
void update_prev_velocity(Variable2D& var, std::unordered_map<Domain2DUniform*, field2>& prev_map)
{
    for (auto* domain : var.geometry->domains)
    {
        field2& curr = *var.field_map[domain];
        field2& prev = prev_map[domain];

        // 复制当前速度场到 prev
        int nx = curr.get_nx();
        int ny = curr.get_ny();
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                prev(i, j) = curr(i, j);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    // Parameter setup
    ParallelPlateChannel2DCase case_param(argc, argv);
    case_param.read_paras();
    case_param.record_paras();

    double h  = case_param.h;
    double Lx = case_param.Lx;
    double Ly = case_param.Ly;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.dt_factor * h;
    time_cfg.set_t_max(case_param.T_total);

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Re);
    physics_cfg.set_model_type(case_param.model_type);

    if (case_param.model_type == 1) // Power Law
    {
        physics_cfg.set_power_law_dimensionless(
            case_param.k_pl,
            case_param.n_index,
            case_param.Re,
            case_param.mu_ref,
            case_param.use_dimensionless_viscosity,
            case_param.mu_min_pl,
            case_param.mu_max_pl);
        std::cout << "Configuring Power Law Model (Dimensionless):" << std::endl;
        std::cout << "  k_pl:      " << case_param.k_pl << std::endl;
        std::cout << "  Re:        " << case_param.Re << std::endl;
        std::cout << "  mu_ref:    " << case_param.mu_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
        std::cout << "  n:         " << case_param.n_index << std::endl;
        std::cout << "  mu_min_pl: " << case_param.mu_min_pl << " (use default if -1)" << std::endl;
        std::cout << "  mu_max_pl: " << case_param.mu_max_pl << " (use default if -1)" << std::endl;
    }
    else if (case_param.model_type == 2) // Carreau
    {
        physics_cfg.set_carreau_dimensionless(case_param.mu_0,
                                              case_param.mu_inf,
                                              case_param.a,
                                              case_param.lambda,
                                              case_param.n_index,
                                              case_param.Re,
                                              case_param.mu_ref,
                                              case_param.use_dimensionless_viscosity);
        std::cout << "Configuring Carreau Model (Dimensionless):" << std::endl;
        std::cout << "  mu_0:           " << case_param.mu_0 << std::endl;
        std::cout << "  mu_inf:         " << case_param.mu_inf << std::endl;
        std::cout << "  lambda:         " << case_param.lambda << std::endl;
        std::cout << "  Re:             " << case_param.Re << std::endl;
        std::cout << "  mu_ref:         " << case_param.mu_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
        std::cout << "  a:      " << case_param.a << std::endl;
        std::cout << "  n:      " << case_param.n_index << std::endl;
    }

    // Output stepping - 根据更大的迭代次数调整输出频率
    int pv_output_step =
        case_param.pv_output_step > 0 ? case_param.pv_output_step : std::max(1, time_cfg.num_iterations / 10);
    int final_step_to_save = case_param.step_to_save > 0 ? case_param.step_to_save : time_cfg.num_iterations;

    // Grid construction - Split into two domains D1 (Left) and D2 (Right)
    int nx_total = static_cast<int>(Lx / h);
    int nx1      = nx_total / 2;
    int nx2      = nx_total - nx1;
    int ny       = static_cast<int>(Ly / h);

    double lx1 = nx1 * h;
    double lx2 = nx2 * h;

    std::cout << "Construct parallel plate channel geometry (Velocity Driven, Multi-domain)..." << std::endl;
    std::cout << "Domain D1: " << nx1 << " x " << ny << " (" << lx1 << " x " << Ly << ")" << std::endl;
    std::cout << "Domain D2: " << nx2 << " x " << ny << " (" << lx2 << " x " << Ly << ")" << std::endl;
    std::cout << "Total grid points: " << (nx1 + nx2) * ny << std::endl;

    Geometry2D      geo;
    Domain2DUniform D1(nx1, ny, lx1, Ly, "D1");
    Domain2DUniform D2(nx2, ny, lx2, Ly, "D2");
    // geo.add_domain({&D1, &D2});

    // Connectivity
    // D1 Right -> D2 Left (Internal)
    geo.connect(&D1, LocationType::Right, &D2);

    // Variable2Ds
    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    // Non-Newtonian Variable2Ds
    Variable2D mu("mu"), tau_xx("tau_xx"), tau_yy("tau_yy"), tau_xy("tau_xy");
    mu.set_geometry(geo);
    tau_xx.set_geometry(geo);
    tau_yy.set_geometry(geo);
    tau_xy.set_geometry(geo);

    // Fields
    field2 u_D1, v_D1, p_D1;
    field2 u_D2, v_D2, p_D2;

    u.set_x_edge_field(&D1, u_D1);
    u.set_x_edge_field(&D2, u_D2);
    v.set_y_edge_field(&D1, v_D1);
    v.set_y_edge_field(&D2, v_D2);
    p.set_center_field(&D1, p_D1);
    p.set_center_field(&D2, p_D2);

    field2 mu_D1, txx_D1, tyy_D1, txy_D1;
    field2 mu_D2, txx_D2, tyy_D2, txy_D2;

    mu.set_corner_field(&D1, mu_D1);
    mu.set_corner_field(&D2, mu_D2);

    tau_xx.set_center_field(&D1, txx_D1);
    tau_xx.set_center_field(&D2, txx_D2);

    tau_yy.set_center_field(&D1, tyy_D1);
    tau_yy.set_center_field(&D2, tyy_D2);

    tau_xy.set_corner_field(&D1, txy_D1);
    tau_xy.set_corner_field(&D2, txy_D2);

    // Helper setters
    auto set_dirichlet_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
        var.set_boundary_value(d, loc, 0.0);
    };

    // Boundary Conditions
    // Walls (Top/Bottom) for BOTH domains
    for (auto* d : {&D1, &D2})
    {
        set_dirichlet_zero(u, d, LocationType::Down);
        set_dirichlet_zero(u, d, LocationType::Up);
        set_dirichlet_zero(v, d, LocationType::Down);
        set_dirichlet_zero(v, d, LocationType::Up);
        set_neumann_zero(p, d, LocationType::Down);
        set_neumann_zero(p, d, LocationType::Up);
    }

    // Inlet at D1 Left (Velocity Profile)
    u.set_boundary_type(&D1, LocationType::Left, PDEBoundaryType::Dirichlet);
    // CRITICAL FIX: Allocate memory for boundary values first!
    // TODO  WHY if comment out the following line, u.boundary_value_map[&D1][LocationType::Left] will cause segfault
    // later.
    u.set_boundary_value(&D1, LocationType::Left, 0.0);
    u.has_boundary_value_map[&D1][LocationType::Left] = true;
    set_dirichlet_zero(v, &D1, LocationType::Left);
    set_neumann_zero(p, &D1, LocationType::Left);

    // Calculate Inlet Profile and overwrite
    double H = Ly / 2.0;
    for (int j = 0; j < ny; ++j)
    {
        double y_coord    = (j + 0.5) * h;
        double y_centered = y_coord - H;

        // Use analytical profile for inlet
        double u_val;
        if (case_param.model_type == 0) // Newtonian
        {
            u_val = power_law_analytical(y_centered, 1.0, case_param.dp_dx, case_param.k_pl, H);
        }
        else // Power Law / Carreau
        {
            u_val = power_law_analytical(y_centered, case_param.n_index, case_param.dp_dx, case_param.k_pl, H);
        }

        // Now it's safe to access
        u.boundary_value_map[&D1][LocationType::Left][j] = u_val;
    }

    // Outlet at D2 Right (Open/Neumann)
    set_neumann_zero(u, &D2, LocationType::Right);
    set_neumann_zero(v, &D2, LocationType::Right);
    set_dirichlet_zero(p, &D2, LocationType::Right); // Pressure outlet p=0

    // Solver Initialization
    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);
    ns_solver.init_nonnewton(&mu, &tau_xx, &tau_yy, &tau_xy);
    ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

    std::string nowtime_dir = case_param.root_dir;

    std::cout << "Starting simulation..." << std::endl;

    // 收敛监测变量: 用于存储上一时间步的速度场
    std::unordered_map<Domain2DUniform*, field2> prev_u_map;
    std::unordered_map<Domain2DUniform*, field2> prev_v_map;

    // 使用 Domain 遍历初始化 prev_map
    for (auto* domain : u.geometry->domains)
    {
        field2& curr_u = *u.field_map[domain];
        field2& curr_v = *v.field_map[domain];

        prev_u_map[domain].init(curr_u.get_nx(), curr_u.get_ny(), "prev_u_" + domain->name);
        prev_v_map[domain].init(curr_v.get_nx(), curr_v.get_ny(), "prev_v_" + domain->name);

        prev_u_map[domain].clear(0.0);
        prev_v_map[domain].clear(0.0);
    }

    // 收敛容差和标志
    double convergence_tol = 1e-6;
    bool   converged       = false;
    int    final_step      = time_cfg.num_iterations;

    std::cout << "Convergence monitoring enabled with tolerance = " << convergence_tol << std::endl;

    // Simulation Loop
    for (int step = 1; step <= time_cfg.num_iterations; step++)
    {
        if (step % 200 == 0)
        {
            env_cfg.showGmresRes = true;
            std::cout << "step: " << step << "/" << time_cfg.num_iterations << "\n";
        }
        else
        {
            env_cfg.showGmresRes = (step <= 5);
        }

        {
            Timer step_timer("step_time", TimeRecordType::None, step % 200 == 0);
            ns_solver.solve_nonnewton();
        }

        // 收敛检测: 计算当前速度场与上一时间步速度场的相对残差
        if (step > 1) // 从第2步开始检测
        {
            // 使用辅助函数计算残差
            double u_residual = compute_velocity_residual(u, prev_u_map);
            double v_residual = compute_velocity_residual(v, prev_v_map);

            // 取 u 和 v 残差的最大值作为总残差
            double max_residual = std::max(u_residual, v_residual);

            // 每100步或每次输出时显示残差
            if (step % 100 == 0 || step % pv_output_step == 0)
            {
                std::cout << "Step " << step << ": u_residual = " << u_residual << ", v_residual = " << v_residual
                          << std::endl;
            }

            // 检查是否收敛
            if (max_residual < convergence_tol)
            {
                std::cout << "\n========================================" << std::endl;
                std::cout << "CONVERGED at step " << step << std::endl;
                std::cout << "u_residual = " << u_residual << std::endl;
                std::cout << "v_residual = " << v_residual << std::endl;
                std::cout << "max_residual = " << max_residual << " < " << convergence_tol << std::endl;
                std::cout << "========================================\n" << std::endl;
                converged  = true;
                final_step = step;
                break; // 提前退出循环
            }
        }

        // 使用辅助函数更新上一时间步的速度场
        update_prev_velocity(u, prev_u_map);
        update_prev_velocity(v, prev_v_map);

        if (step % pv_output_step == 0)
        {
            std::cout << "Saving step " << step << " to CSV files." << std::endl;
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
            IO::write_csv(u, nowtime_dir + "/u/u_" + std::to_string(step));
            IO::write_csv(v, nowtime_dir + "/v/v_" + std::to_string(step));
            IO::write_csv(p, nowtime_dir + "/p/p_" + std::to_string(step));
            IO::write_csv(mu, nowtime_dir + "/mu/mu_" + std::to_string(step));
        }

        if (std::isnan(u_D1(nx1 / 2, ny / 2)))
        {
            std::cout << "=== DIVERGENCE DETECTED ===" << std::endl;
            return -1;
        }
    }

    std::cout << "Simulation finished." << std::endl;

    // Final Save
    IO::write_csv(u, nowtime_dir + "/final/u_" + std::to_string(final_step));
    IO::write_csv(v, nowtime_dir + "/final/v_" + std::to_string(final_step));
    IO::write_csv(p, nowtime_dir + "/final/p_" + std::to_string(final_step));
    IO::write_csv(mu, nowtime_dir + "/final/mu_" + std::to_string(final_step));

    // Analytical Verification (At Outlet D2 Right)
    if (case_param.model_type == 1)
    {
        std::cout << "Calculating analytical solution verification at outlet..." << std::endl;

        // Check profile at the middle of D2 (or end)
        int    i_check      = nx2 - 2; // Near outlet
        double error_sum_sq = 0.0;
        double ref_sum_sq   = 0.0;

        std::string   ana_path = nowtime_dir + "/analytical_profile_outlet.csv";
        std::ofstream ana_file(ana_path);
        ana_file << "y,u_numerical,u_analytical,error\n";

        for (int j = 0; j < ny; ++j)
        {
            double y_coord    = (j + 0.5) * h;
            double y_centered = y_coord - H;

            double u_num = u_D2(i_check, j);
            double u_ana = power_law_analytical(y_centered, case_param.n_index, case_param.dp_dx, case_param.k_pl, H);

            double error = u_num - u_ana;
            error_sum_sq += error * error;
            ref_sum_sq += u_ana * u_ana;

            ana_file << y_centered << "," << u_num << "," << u_ana << "," << error << "\n";
        }
        ana_file.close();

        double l2_error = (ref_sum_sq > 1e-10) ? std::sqrt(error_sum_sq / ref_sum_sq) : std::sqrt(error_sum_sq);

        std::cout << "========================================" << std::endl;
        std::cout << "Verification Result (Power Law n=" << case_param.n_index << ")" << std::endl;
        std::cout << "L2 Error Norm (Outlet): " << l2_error << std::endl;
        std::cout << "Analytical profile saved to: " << ana_path << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return 0;
}
