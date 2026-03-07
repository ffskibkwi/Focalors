#include "parallel_plate_microchannel_mhd.h"
#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

namespace
{
    double compute_velocity_residual(Variable2D& var, std::unordered_map<Domain2DUniform*, field2>& prev_map)
    {
        double total_diff_sq = 0.0;
        double total_norm_sq = 0.0;

        for (auto* domain : var.geometry->domains)
        {
            field2& curr = *var.field_map[domain];
            field2& prev = prev_map[domain];
            field2  diff = curr - prev;
            total_diff_sq += diff.squared_sum();
            total_norm_sq += curr.squared_sum();
        }

        if (total_norm_sq > 1.0e-14)
            return std::sqrt(total_diff_sq / total_norm_sq);

        return std::sqrt(total_diff_sq);
    }

    void update_prev_velocity(Variable2D& var, std::unordered_map<Domain2DUniform*, field2>& prev_map)
    {
        for (auto* domain : var.geometry->domains)
        {
            field2&   curr = *var.field_map[domain];
            field2&   prev = prev_map[domain];
            const int nx   = curr.get_nx();
            const int ny   = curr.get_ny();
            for (int i = 0; i < nx; ++i)
            {
                for (int j = 0; j < ny; ++j)
                {
                    prev(i, j) = curr(i, j);
                }
            }
        }
    }

    int find_probe_i_center(const Domain2DUniform& domain, double x_probe)
    {
        const double hx = domain.get_hx();
        const int    nx = domain.get_nx();
        int          i  = static_cast<int>(std::round((x_probe - 0.5 * hx) / hx));
        i               = std::max(0, std::min(i, nx - 1));
        return i;
    }
} // namespace

int main(int argc, char* argv[])
{
    ParallelPlateMicrochannelMhd2DCase case_param(argc, argv);
    case_param.read_paras();
    case_param.record_paras();

    const double h_ref = case_param.half_height;
    const double lx    = case_param.getLx();
    const double ly    = case_param.getLy();

    const int nx = case_param.nx;
    const int ny = case_param.ny;
    if (nx < 3 || ny < 3)
    {
        std::cerr << "Invalid grid: nx and ny must both be >= 3." << std::endl;
        return -1;
    }

    Geometry2D      geo;
    Domain2DUniform d0(nx, ny, lx, ly, "D0");
    geo.add_domain(&d0);
    geo.axis(&d0, LocationType::XNegative);
    geo.axis(&d0, LocationType::YNegative);
    geo.check();
    geo.solve_prepare();

    const double hx = d0.get_hx();
    const double hy = d0.get_hy();

    EnvironmentConfig& env_cfg    = EnvironmentConfig::Get();
    env_cfg.showCurrentStep       = false;
    env_cfg.showGmresRes          = false;
    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.dt_factor * std::min(hx, hy);
    time_cfg.set_t_max(case_param.T_total);
    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    const bool     enable_mhd  = (std::abs(case_param.Ha) > 0.0);
    const int      pv_output_step =
        case_param.pv_output_step > 0 ? case_param.pv_output_step : std::max(1, time_cfg.num_iterations / 20);
    const int converged_hits_n = std::max(case_param.converged_hits, 1);

    physics_cfg.set_Re(case_param.Re);
    physics_cfg.set_enable_mhd(enable_mhd);
    physics_cfg.set_Ha(case_param.Ha);
    physics_cfg.set_magnetic_field(case_param.Bx, case_param.By, case_param.Bz);
    physics_cfg.set_model_type(case_param.model_type);
    if (case_param.model_type == 1)
    {
        physics_cfg.set_power_law_dimensionless(case_param.k_pl,
                                                case_param.n_index,
                                                case_param.Re,
                                                case_param.mu_ref,
                                                case_param.use_dimensionless_viscosity,
                                                case_param.mu_min_pl,
                                                case_param.mu_max_pl);
    }
    else
    {
        std::cout << "[Warn] This case is intended for power-law model_type=1. Running with current model_type="
                  << case_param.model_type << std::endl;
    }

    std::cout << "Parallel-plate microchannel MHD case" << std::endl;
    std::cout << "  Grid: nx=" << nx << ", ny=" << ny << std::endl;
    std::cout << "  Geometry: Lx=" << lx << ", Ly=" << ly << ", h_ref=" << h_ref << std::endl;
    std::cout << "  Model: n=" << case_param.n_index << ", Ha=" << case_param.Ha << ", Re=" << case_param.Re
              << std::endl;
    std::cout << "  MHD enable=" << enable_mhd << ", B=(" << case_param.Bx << "," << case_param.By << ","
              << case_param.Bz << ")" << std::endl;

    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    Variable2D phi("phi");
    if (enable_mhd)
        phi.set_geometry(geo);

    Variable2D mu("mu"), tau_xx("tau_xx"), tau_yy("tau_yy"), tau_xy("tau_xy");
    mu.set_geometry(geo);
    tau_xx.set_geometry(geo);
    tau_yy.set_geometry(geo);
    tau_xy.set_geometry(geo);

    field2 u_d0, v_d0, p_d0;
    u.set_x_edge_field(&d0, u_d0);
    v.set_y_edge_field(&d0, v_d0);
    p.set_center_field(&d0, p_d0);

    field2 mu_d0, txx_d0, tyy_d0, txy_d0;
    mu.set_corner_field(&d0, mu_d0);
    tau_xx.set_center_field(&d0, txx_d0);
    tau_yy.set_center_field(&d0, tyy_d0);
    tau_xy.set_corner_field(&d0, txy_d0);

    field2 phi_d0;
    if (enable_mhd)
        phi.set_center_field(&d0, phi_d0);

    auto set_dirichlet = [](Variable2D& var, Domain2DUniform* domain, LocationType loc, double value) {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(domain, loc, value);
    };
    auto set_neumann = [](Variable2D& var, Domain2DUniform* domain, LocationType loc, [[maybe_unused]] double value) {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Neumann);
    };

    // Wall no-slip.
    set_dirichlet(u, &d0, LocationType::YNegative, 0.0);
    set_dirichlet(u, &d0, LocationType::YPositive, 0.0);
    set_dirichlet(v, &d0, LocationType::YNegative, 0.0);
    set_dirichlet(v, &d0, LocationType::YPositive, 0.0);
    set_neumann(p, &d0, LocationType::YNegative, 0.0);
    set_neumann(p, &d0, LocationType::YPositive, 0.0);

    // Pressure-driven inlet/outlet.
    const double p_out = 0.0;
    const double p_in  = -case_param.dp_dx * lx;
    set_neumann(u, &d0, LocationType::XNegative, 0.0);
    set_neumann(u, &d0, LocationType::XPositive, 0.0);
    set_dirichlet(v, &d0, LocationType::XNegative, 0.0);
    set_dirichlet(v, &d0, LocationType::XPositive, 0.0);
    set_dirichlet(p, &d0, LocationType::XNegative, p_in);
    set_dirichlet(p, &d0, LocationType::XPositive, p_out);

    if (enable_mhd)
    {
        set_neumann(phi, &d0, LocationType::YNegative, 0.0);
        set_neumann(phi, &d0, LocationType::YPositive, 0.0);
        set_dirichlet(phi, &d0, LocationType::XNegative, 0.0);
        set_dirichlet(phi, &d0, LocationType::XPositive, 0.0);

        // 兼容 IO::write_csv(Variable2D) 对 Center 变量边界类型查询四边的实现。
        phi.set_boundary_type(PDEBoundaryType::Neumann);
    }

    // 角点变量写 CSV 时会读取边界类型映射，需补全四边。
    mu.set_boundary_type(PDEBoundaryType::Neumann);

    // Zero initialization.
    u_d0.clear(0.0);
    v_d0.clear(0.0);
    p_d0.clear(0.0);
    mu_d0.clear(case_param.k_pl);
    txx_d0.clear(0.0);
    tyy_d0.clear(0.0);
    txy_d0.clear(0.0);
    if (enable_mhd)
        phi_d0.clear(0.0);

    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);
    ns_solver.init_nonnewton(&mu, &tau_xx, &tau_yy, &tau_xy, enable_mhd ? &phi : nullptr);
    ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

    std::unordered_map<Domain2DUniform*, field2> prev_u_map;
    std::unordered_map<Domain2DUniform*, field2> prev_v_map;
    prev_u_map[&d0].init(u_d0.get_nx(), u_d0.get_ny(), "prev_u_d0");
    prev_v_map[&d0].init(v_d0.get_nx(), v_d0.get_ny(), "prev_v_d0");
    prev_u_map[&d0].clear(0.0);
    prev_v_map[&d0].clear(0.0);

    std::cout << "Starting solve: iterations=" << time_cfg.num_iterations << ", dt=" << time_cfg.dt << std::endl;

    int    final_step = time_cfg.num_iterations;
    int    hits       = 0;
    double residual   = std::numeric_limits<double>::infinity();

    for (int step = 1; step <= time_cfg.num_iterations; ++step)
    {
        ns_solver.solve_nonnewton();

        if (step > 1)
        {
            const double u_residual = compute_velocity_residual(u, prev_u_map);
            const double v_residual = compute_velocity_residual(v, prev_v_map);
            residual                = std::max(u_residual, v_residual);

            if (residual < case_param.convergence_tol)
                ++hits;
            else
                hits = 0;

            if (step % 100 == 0 || step % pv_output_step == 0)
            {
                std::cout << "step=" << step << " residual(u,v,max)=(" << u_residual << "," << v_residual << ","
                          << residual << "), hits=" << hits << "/" << converged_hits_n << std::endl;
            }

            if (hits >= converged_hits_n)
            {
                final_step = step;
                std::cout << "Converged at step=" << final_step << ", residual=" << residual << std::endl;
                break;
            }
        }

        update_prev_velocity(u, prev_u_map);
        update_prev_velocity(v, prev_v_map);

        if (step % pv_output_step == 0)
        {
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
            IO::write_csv(u, case_param.root_dir + "/u/u_" + std::to_string(step));
            IO::write_csv(v, case_param.root_dir + "/v/v_" + std::to_string(step));
            IO::write_csv(p, case_param.root_dir + "/p/p_" + std::to_string(step));
            IO::write_csv(mu, case_param.root_dir + "/mu/mu_" + std::to_string(step));
            if (enable_mhd)
                IO::write_csv(phi, case_param.root_dir + "/phi/phi_" + std::to_string(step));
        }

        if (!std::isfinite(u_d0(nx / 2, ny / 2)) || !std::isfinite(v_d0(nx / 2, ny / 2)))
        {
            std::cerr << "Divergence detected at step=" << step << std::endl;
            return -1;
        }
    }

    IO::write_csv(u, case_param.root_dir + "/final/u_" + std::to_string(final_step));
    IO::write_csv(v, case_param.root_dir + "/final/v_" + std::to_string(final_step));
    IO::write_csv(p, case_param.root_dir + "/final/p_" + std::to_string(final_step));
    IO::write_csv(mu, case_param.root_dir + "/final/mu_" + std::to_string(final_step));
    if (enable_mhd)
        IO::write_csv(phi, case_param.root_dir + "/final/phi_" + std::to_string(final_step));

    const double x_probe = case_param.x_probe_over_h * h_ref;
    const int    i_probe = find_probe_i_center(d0, x_probe);
    const double denom   = h_ref * h_ref * std::abs(case_param.dp_dx) / std::max(case_param.mu_ref, 1.0e-12);

    std::ofstream profile(case_param.root_dir + "/profile_x_probe.csv");
    profile << "y_over_h,u_norm,u_raw\n";
    for (int j = 0; j < ny; ++j)
    {
        const double y        = (j + 0.5) * hy - h_ref;
        const double y_over_h = y / std::max(h_ref, 1.0e-12);
        const double u_raw    = u_d0(i_probe, j);
        const double u_norm   = u_raw / std::max(denom, 1.0e-12);
        profile << y_over_h << "," << u_norm << "," << u_raw << "\n";
    }
    profile.close();

    if (!profile.good())
    {
        std::cerr << "Failed writing profile_x_probe.csv" << std::endl;
        return -1;
    }

    std::cout << "Finished. final_step=" << final_step
              << ", profile saved: " << case_param.root_dir + "/profile_x_probe.csv" << std::endl;

    return 0;
}
