#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "cross_shaped_channel.h"
#include "io/common.h"
#include "io/csv_writer_2d.h"
#include "ns/mhd_module_2d_mac.h"
#include "ns/ns_solver2d.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace
{
    // Keep the explicit non-Newtonian diffusion update inside a conservative
    // 2D FTCS-like stability range: dt <= C * h^2 / mu_max, with C < 0.25.
    constexpr double EXPLICIT_DIFFUSION_DT_FACTOR = 0.20;
    // Mirror the Hartmann validation case: explicit Lorentz forcing should also
    // obey a magnetic time-step restriction when B is active.
    constexpr double MAGNETIC_DT_FACTOR = 0.50;

    /** @brief Convert the viscosity bound used in input/output units to solver units. */
    double scale_viscosity_to_solver_units(double viscosity_value, const PhysicsConfig& physics_cfg)
    {
        if (physics_cfg.model_type == 0 || !physics_cfg.use_dimensionless_viscosity)
            return viscosity_value;

        const double solver_scale = physics_cfg.mu_ref * physics_cfg.Re;
        if (solver_scale <= 0.0)
            return viscosity_value;

        return viscosity_value / solver_scale;
    }

    struct TimeStepSelection
    {
        double convective_dt                   = 0.0;
        double diffusion_dt_limit              = std::numeric_limits<double>::infinity();
        double magnetic_dt_limit               = std::numeric_limits<double>::infinity();
        double selected_dt                     = 0.0;
        double viscosity_upper_bound_raw       = 0.0;
        double viscosity_upper_bound_effective = 0.0;
        double magnetic_factor_sq              = 0.0;
        bool   diffusion_limited               = false;
        bool   magnetic_limited                = false;
    };

    struct TimeStepSchedule
    {
        double base_dt        = 0.0;
        double startup_dt     = 0.0;
        double startup_t_end  = 0.0;
        bool   has_startup_dt = false;
        double initial_dt() const { return has_startup_dt ? startup_dt : base_dt; }
    };

    /** @brief Select the smaller of the convective and explicit diffusion time-step limits. */
    TimeStepSelection select_time_step(double h, double dt_factor, const PhysicsConfig& physics_cfg)
    {
        TimeStepSelection selection;

        selection.convective_dt             = dt_factor * h;
        selection.viscosity_upper_bound_raw = physics_cfg.model_type == 0 ? physics_cfg.nu : physics_cfg.mu_max;
        selection.viscosity_upper_bound_effective =
            scale_viscosity_to_solver_units(selection.viscosity_upper_bound_raw, physics_cfg);

        if (selection.viscosity_upper_bound_effective > 0.0)
        {
            selection.diffusion_dt_limit =
                EXPLICIT_DIFFUSION_DT_FACTOR * h * h / selection.viscosity_upper_bound_effective;
        }

        selection.magnetic_factor_sq = physics_cfg.Bx * physics_cfg.Bx + physics_cfg.By * physics_cfg.By +
                                       physics_cfg.Bz * physics_cfg.Bz;
        if (std::abs(physics_cfg.Ha) > 0.0 && selection.magnetic_factor_sq > 0.0 && physics_cfg.Re > 0.0)
        {
            selection.magnetic_dt_limit =
                MAGNETIC_DT_FACTOR * physics_cfg.Re / (physics_cfg.Ha * physics_cfg.Ha * selection.magnetic_factor_sq);
        }

        selection.selected_dt =
            std::min(selection.convective_dt, std::min(selection.diffusion_dt_limit, selection.magnetic_dt_limit));
        selection.diffusion_limited = selection.diffusion_dt_limit < selection.convective_dt &&
                                      selection.diffusion_dt_limit <= selection.magnetic_dt_limit;
        selection.magnetic_limited = selection.magnetic_dt_limit < selection.convective_dt &&
                                     selection.magnetic_dt_limit < selection.diffusion_dt_limit;

        return selection;
    }

    double compute_step_dt(double current_time, double total_time, const TimeStepSchedule& schedule)
    {
        const double eps = 128.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, total_time);
        if (current_time >= total_time - eps)
            return 0.0;

        const bool   in_startup_phase = schedule.has_startup_dt && current_time < schedule.startup_t_end - eps;
        const double phase_dt         = in_startup_phase ? schedule.startup_dt : schedule.base_dt;
        const double phase_end        = in_startup_phase ? schedule.startup_t_end : total_time;
        const double phase_remain     = std::max(0.0, phase_end - current_time);
        const double total_remain     = std::max(0.0, total_time - current_time);

        return std::min(phase_dt, std::min(phase_remain, total_remain));
    }

    int estimate_num_steps(double total_time, const TimeStepSchedule& schedule)
    {
        int    step         = 0;
        double current_time = 0.0;

        while (true)
        {
            const double dt_step = compute_step_dt(current_time, total_time, schedule);
            if (dt_step <= 0.0)
                break;

            current_time += dt_step;
            ++step;
        }

        return step;
    }
} // namespace
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
 * MHD + Non-Newtonian fluid simulation for cross-shaped channel (MAC grid)
 */
int main(int argc, char* argv[])
{
    // 参数读取与物理配置
    CrossShapedChannel2DCase case_param(argc, argv);
    case_param.read_paras();

    Geometry2D geo;
    double     h = case_param.h;

    EnvironmentConfig& env_cfg       = EnvironmentConfig::Get();
    env_cfg.showGmresRes             = false;
    env_cfg.showCurrentStep          = false;
    TimeAdvancingConfig& time_cfg    = TimeAdvancingConfig::Get();
    PhysicsConfig&       physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Re);

    const bool enable_mhd = (std::abs(case_param.Ha) > 0.0);
    physics_cfg.set_enable_mhd(enable_mhd);

    // MHD parameters (must be set before MHDModule2D initialization)
    physics_cfg.Ha = case_param.Ha;
    physics_cfg.set_magnetic_field(case_param.Bx, case_param.By, case_param.Bz);

    std::cout << "MHD Parameters:" << std::endl;
    std::cout << "  enable_mhd: " << enable_mhd << std::endl;
    std::cout << "  Ha: " << case_param.Ha << std::endl;
    std::cout << "  Bx: " << case_param.Bx << std::endl;
    std::cout << "  By: " << case_param.By << std::endl;
    std::cout << "  Bz: " << case_param.Bz << std::endl;

    // Set Non-Newtonian parameters based on model type
    physics_cfg.set_model_type(case_param.model_type);
    physics_cfg.set_gamma_ref(case_param.gamma_ref);
    if (case_param.model_type == 1) // Power Law
    {
        physics_cfg.set_power_law_dimensionless(case_param.k_pl,
                                                case_param.n_index,
                                                case_param.Re,
                                                case_param.mu_ref,
                                                case_param.use_dimensionless_viscosity,
                                                case_param.mu_min_pl,
                                                case_param.mu_max_pl);

        std::cout << "Configuring Power Law Model (Dimensionless):" << std::endl;
        std::cout << "  k_pl: " << case_param.k_pl << std::endl;
        std::cout << "  Re: " << case_param.Re << std::endl;
        std::cout << "  mu_ref: " << case_param.mu_ref << std::endl;
        std::cout << "  gamma_ref: " << case_param.gamma_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
        std::cout << "  n:     " << case_param.n_index << std::endl;
        std::cout << "  mu_min_pl: " << case_param.mu_min_pl << std::endl;
        std::cout << "  mu_max_pl: " << case_param.mu_max_pl << std::endl;
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
                                              case_param.use_dimensionless_viscosity,
                                              case_param.mu_min_pl,
                                              case_param.mu_max_pl);
        std::cout << "Configuring Carreau Model (Dimensionless):" << std::endl;
        std::cout << "  mu_0:           " << case_param.mu_0 << std::endl;
        std::cout << "  mu_inf:         " << case_param.mu_inf << std::endl;
        std::cout << "  lambda:         " << case_param.lambda << std::endl;
        std::cout << "  Re:             " << case_param.Re << std::endl;
        std::cout << "  mu_ref:         " << case_param.mu_ref << std::endl;
        std::cout << "  gamma_ref:      " << case_param.gamma_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
        std::cout << "  a:      " << case_param.a << std::endl;
        std::cout << "  n:      " << case_param.n_index << std::endl;
        std::cout << "  mu_min_pl: " << case_param.mu_min_pl << std::endl;
        std::cout << "  mu_max_pl: " << case_param.mu_max_pl << std::endl;
    }
    else if (case_param.model_type == 3) // Casson
    {
        physics_cfg.set_casson_dimensionless(case_param.casson_mu,
                                             case_param.casson_tau0,
                                             case_param.Re,
                                             case_param.mu_ref,
                                             case_param.use_dimensionless_viscosity,
                                             case_param.mu_min_pl,
                                             case_param.mu_max_pl);
        std::cout << "Configuring Casson Model (Dimensionless):" << std::endl;
        std::cout << "  casson_mu:      " << case_param.casson_mu << std::endl;
        std::cout << "  casson_tau0:    " << case_param.casson_tau0 << std::endl;
        std::cout << "  Re:             " << case_param.Re << std::endl;
        std::cout << "  mu_ref:         " << case_param.mu_ref << std::endl;
        std::cout << "  gamma_ref:      " << case_param.gamma_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
    }
    else
    {
        std::cout << "Configuring Newtonian Model." << std::endl;
    }

    const TimeStepSelection base_time_step_selection = select_time_step(h, case_param.dt_factor, physics_cfg);

    const bool        has_requested_startup_dt = case_param.startup_dt_factor > 0.0 && case_param.startup_t_end > 0.0;
    TimeStepSelection startup_time_step_selection;
    if (has_requested_startup_dt)
        startup_time_step_selection = select_time_step(h, case_param.startup_dt_factor, physics_cfg);

    TimeStepSchedule time_step_schedule;
    time_step_schedule.base_dt = base_time_step_selection.selected_dt;
    if (has_requested_startup_dt)
    {
        time_step_schedule.startup_dt    = startup_time_step_selection.selected_dt;
        time_step_schedule.startup_t_end = std::min(case_param.startup_t_end, case_param.T_total);
        time_step_schedule.has_startup_dt =
            time_step_schedule.startup_t_end > 0.0 && time_step_schedule.startup_dt < time_step_schedule.base_dt;
    }

    const int estimated_total_steps = estimate_num_steps(case_param.T_total, time_step_schedule);

    time_cfg.dt             = time_step_schedule.initial_dt();
    time_cfg.t_max          = case_param.T_total;
    time_cfg.num_iterations = estimated_total_steps;
    // time_cfg.num_iterations   = 10;

    std::cout << "Time Step Selection:" << std::endl;
    std::cout << "  base_convective_dt: " << base_time_step_selection.convective_dt << std::endl;
    std::cout << "  base_diffusion_dt_limit: " << base_time_step_selection.diffusion_dt_limit << std::endl;
    std::cout << "  base_magnetic_dt_limit: " << base_time_step_selection.magnetic_dt_limit << std::endl;
    std::cout << "  viscosity_upper_bound_raw: " << base_time_step_selection.viscosity_upper_bound_raw << std::endl;
    std::cout << "  viscosity_upper_bound_effective: " << base_time_step_selection.viscosity_upper_bound_effective
              << std::endl;
    std::cout << "  magnetic_factor_sq: " << base_time_step_selection.magnetic_factor_sq << std::endl;
    std::cout << "  base_selected_dt: " << base_time_step_selection.selected_dt << std::endl;
    std::cout << "  initial_dt: " << time_cfg.dt << std::endl;
    std::cout << "  base_diffusion_limited: " << std::boolalpha << base_time_step_selection.diffusion_limited
              << std::noboolalpha << std::endl;
    std::cout << "  base_magnetic_limited: " << std::boolalpha << base_time_step_selection.magnetic_limited
              << std::noboolalpha << std::endl;
    if (has_requested_startup_dt)
    {
        std::cout << "  startup_convective_dt: " << startup_time_step_selection.convective_dt << std::endl;
        std::cout << "  startup_diffusion_dt_limit: " << startup_time_step_selection.diffusion_dt_limit << std::endl;
        std::cout << "  startup_magnetic_dt_limit: " << startup_time_step_selection.magnetic_dt_limit << std::endl;
        std::cout << "  startup_selected_dt: " << startup_time_step_selection.selected_dt << std::endl;
        std::cout << "  startup_t_end: " << time_step_schedule.startup_t_end << std::endl;
        std::cout << "  startup_active: " << std::boolalpha << time_step_schedule.has_startup_dt << std::noboolalpha
                  << std::endl;
        std::cout << "  startup_magnetic_limited: " << std::boolalpha << startup_time_step_selection.magnetic_limited
                  << std::noboolalpha << std::endl;
    }
    std::cout << "  estimated_total_steps: " << estimated_total_steps << std::endl;

    if (estimated_total_steps <= 0)
    {
        std::cerr << "No valid time step schedule produced for T_total=" << case_param.T_total << std::endl;
        return -1;
    }

    // 计算循环输出步数间隔 pv_output_step（如果未指定则使用 estimated_total_steps/10）
    int pv_output_step =
        case_param.pv_output_step > 0 ? case_param.pv_output_step : std::max(1, estimated_total_steps / 10);
    // 计算最终保存步数（如果未指定则使用 estimated_total_steps）
    int final_step_to_save = case_param.step_to_save > 0 ? case_param.step_to_save : estimated_total_steps;

    case_param.max_step     = estimated_total_steps;
    case_param.step_to_save = final_step_to_save;

    const bool should_record_paras = case_param.record_paras();
    if (should_record_paras)
    {
        case_param.paras_record.record("mhd_grid", std::string("mac"))
            .record("dt", time_cfg.dt)
            .record("dt_base", base_time_step_selection.selected_dt)
            .record("dt_base_convective", base_time_step_selection.convective_dt)
            .record("dt_base_diffusion_limit", base_time_step_selection.diffusion_dt_limit)
            .record("dt_base_magnetic_limit", base_time_step_selection.magnetic_dt_limit)
            .record("dt_startup", has_requested_startup_dt ? startup_time_step_selection.selected_dt : 0.0)
            .record("dt_startup_convective", has_requested_startup_dt ? startup_time_step_selection.convective_dt : 0.0)
            .record("dt_startup_diffusion_limit",
                    has_requested_startup_dt ? startup_time_step_selection.diffusion_dt_limit : 0.0)
            .record("dt_startup_magnetic_limit",
                    has_requested_startup_dt ? startup_time_step_selection.magnetic_dt_limit : 0.0)
            .record("dt_startup_active", time_step_schedule.has_startup_dt ? 1 : 0)
            .record("dt_startup_t_end", time_step_schedule.startup_t_end)
            .record("estimated_total_steps", estimated_total_steps)
            .record("viscosity_upper_bound_raw", base_time_step_selection.viscosity_upper_bound_raw)
            .record("viscosity_upper_bound_effective", base_time_step_selection.viscosity_upper_bound_effective)
            .record("viscosity_upper_bound", base_time_step_selection.viscosity_upper_bound_effective)
            .record("magnetic_factor_sq", base_time_step_selection.magnetic_factor_sq)
            .record("dt_diffusion_limited", base_time_step_selection.diffusion_limited ? 1 : 0)
            .record("dt_magnetic_limited", base_time_step_selection.magnetic_limited ? 1 : 0)
            .record("dt_startup_diffusion_limited",
                    has_requested_startup_dt && startup_time_step_selection.diffusion_limited ? 1 : 0)
            .record("dt_startup_magnetic_limited",
                    has_requested_startup_dt && startup_time_step_selection.magnetic_limited ? 1 : 0);
    }

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
    // geo.add_domain(&A1);
    // geo.add_domain(&A2);
    // geo.add_domain(&A3);
    // geo.add_domain(&A4);
    // geo.add_domain(&A5);

    // Construct cross connectivity
    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);
    geo.connect(&A2, LocationType::YPositive, &A5);

    // Set the center block as the geometric reference so domain offsets match the
    // post-processing layout: A2 at (0, 0), A1/A3 along x, A4/A5 along y.
    geo.axis(&A2, LocationType::XNegative);
    geo.axis(&A2, LocationType::YNegative);
    geo.check();
    geo.solve_prepare();

    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    // Electric potential variable (center type, like pressure p)
    Variable2D phi("phi");
    if (enable_mhd)
        phi.set_geometry(geo);

    // Non-Newtonian Variable2Ds
    Variable2D mu("mu"), tau_xx("tau_xx"), tau_yy("tau_yy"), tau_xy("tau_xy");
    mu.set_geometry(geo);
    tau_xx.set_geometry(geo);
    tau_yy.set_geometry(geo);
    tau_xy.set_geometry(geo);

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
    if (enable_mhd)
    {
        phi.set_center_field(&A1, phi_A1);
        phi.set_center_field(&A2, phi_A2);
        phi.set_center_field(&A3, phi_A3);
        phi.set_center_field(&A4, phi_A4);
        phi.set_center_field(&A5, phi_A5);
    }

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
        var.set_boundary_value(d, loc, 0.0);
    };
    auto is_adjacented = [&](Domain2DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };
    // Default outer boundaries
    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};
    std::vector<LocationType>     dirs    = {
        LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive};

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

    // Outlet pressure: Dirichlet p = 0 (A4 YNegative, A5 YPositive)
    set_dirichlet_zero(p, &A4, LocationType::YNegative);
    set_dirichlet_zero(p, &A5, LocationType::YPositive);

    // ========== Phi boundary conditions ==========
    if (enable_mhd)
    {
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

        // Inlet Dirichlet: A1 XNegative, phi = 0
        phi.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A1, LocationType::XNegative, 0.0);
        phi.has_boundary_value_map[&A1][LocationType::XNegative] = true;

        // Inlet Dirichlet: A3 XPositive, phi = 0
        phi.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A3, LocationType::XPositive, 0.0);
        phi.has_boundary_value_map[&A3][LocationType::XPositive] = true;

        // Outlets Dirichlet: A4 YNegative, A5 YPositive, phi = 0
        phi.set_boundary_type(&A4, LocationType::YNegative, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A4, LocationType::YNegative, 0.0);
        phi.has_boundary_value_map[&A4][LocationType::YNegative] = true;

        phi.set_boundary_type(&A5, LocationType::YPositive, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A5, LocationType::YPositive, 0.0);
        phi.has_boundary_value_map[&A5][LocationType::YPositive] = true;
    }

    // Uniform inlet velocity to match the paper setup (nondimensional amplitude = 1.0)

    u.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A1, LocationType::XNegative, 0.0); // ← 添加这行来分配内存
    u.has_boundary_value_map[&A1][LocationType::XNegative] = true;
    set_dirichlet_zero(v, &A1, LocationType::XNegative);
    // A1 XNegative: u = +1.0
    for (int j = 0; j < u_A1.get_ny(); ++j)
    {
        u.boundary_value_map[&A1][LocationType::XNegative][j] = 1.0;
    }
    set_dirichlet_zero(v, &A1, LocationType::XNegative);

    // A3 XPositive: u = -1.0
    u.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
    u.has_boundary_value_map[&A3][LocationType::XPositive] = true;
    u.set_boundary_value(&A3, LocationType::XPositive, 0.0); // ← 添加这行来分配内存
    set_dirichlet_zero(v, &A3, LocationType::XPositive);
    for (int j = 0; j < u_A3.get_ny(); ++j)
    {
        u.boundary_value_map[&A3][LocationType::XPositive][j] = -1.0;
    }

    // A4 YNegative: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A4, LocationType::YNegative);
    set_neumann_zero(v, &A4, LocationType::YNegative);

    // A5 YPositive: open/symmetry as Neumann for u and v
    set_neumann_zero(u, &A5, LocationType::YPositive);
    set_neumann_zero(v, &A5, LocationType::YPositive);

    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);
    ns_solver.init_nonnewton(&mu, &tau_xx, &tau_yy, &tau_xy, enable_mhd ? &phi : nullptr);

    ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

    // Generate timestamp directory
    std::string nowtime_dir = case_param.root_dir;

    std::cout << "Starting MHD + Non-Newtonian simulation..." << std::endl;

    double current_time = 0.0;
    int    step         = 0;

    // Solve
    while (step < estimated_total_steps)
    {
        const double dt_step = compute_step_dt(current_time, case_param.T_total, time_step_schedule);
        if (dt_step <= 0.0)
            break;

        ++step;
        time_cfg.dt = dt_step;
        ns_solver.setTimeStep(dt_step);

        // 每200步启用计时输出
        if (step % 200 == 0)
        {
            env_cfg.showGmresRes = true;
            std::cout << "step: " << step << "/" << estimated_total_steps << ", t = " << current_time
                      << ", dt = " << dt_step << "\n";
        }
        else
        {
            env_cfg.showGmresRes = (step <= 5);
        }

        {
            Timer step_timer("step_time", TimeRecordType::None, step % 200 == 0);
            // Non-Newtonian solve (MHD is internally handled by ns_solver when enable_mhd=true)
            ns_solver.solve_nonnewton();
        }

        current_time += dt_step;

        // 使用 pv_output_step 控制循环输出
        if (step % pv_output_step == 0)
        {
            std::cout << "Saving step " << (step) << " to CSV files." << std::endl;
            // update boundary
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
            IO::write_csv(u, nowtime_dir + "/u/u_" + std::to_string(step));
            IO::write_csv(v, nowtime_dir + "/v/v_" + std::to_string(step));
            IO::write_csv(p, nowtime_dir + "/p/p_" + std::to_string(step));
            IO::write_csv(mu, nowtime_dir + "/mu/mu_" + std::to_string(step));
            if (enable_mhd)
                IO::write_csv(phi, nowtime_dir + "/phi/phi_" + std::to_string(step));
        }
        if (std::isnan(u_A1(1, 1)))
        {
            std::cout << "=== DIVERGENCE ===" << std::endl;
            return -1;
        }
    }
    std::cout << "Simulation finished." << std::endl;
    const int runtime_final_step = case_param.step_to_save > 0 ? case_param.step_to_save : step;
    // 使用 step_to_save 控制最终保存
    IO::write_csv(u, nowtime_dir + "/final/u_" + std::to_string(runtime_final_step));
    IO::write_csv(v, nowtime_dir + "/final/v_" + std::to_string(runtime_final_step));
    IO::write_csv(p, nowtime_dir + "/final/p_" + std::to_string(runtime_final_step));
    IO::write_csv(mu, nowtime_dir + "/final/mu_" + std::to_string(runtime_final_step));
    if (enable_mhd)
        IO::write_csv(phi, nowtime_dir + "/final/phi_" + std::to_string(runtime_final_step));
    return 0;
}
