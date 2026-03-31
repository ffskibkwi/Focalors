#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/csv_handler.h"
#include "ns/ns_solver2d.h"
#include "ns/physical_pe_solver2d.h"
#include "pe/concat/concat_solver2d.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

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

        return total_norm_sq > 1e-14 ? std::sqrt(total_diff_sq / total_norm_sq) : std::sqrt(total_diff_sq);
    }

    void update_prev_velocity(Variable2D& var, std::unordered_map<Domain2DUniform*, field2>& prev_map)
    {
        for (auto* domain : var.geometry->domains)
        {
            field2& curr = *var.field_map[domain];
            field2& prev = prev_map[domain];

            for (int i = 0; i < curr.get_nx(); ++i)
                for (int j = 0; j < curr.get_ny(); ++j)
                    prev(i, j) = curr(i, j);
        }
    }

    double compute_linear_fit_slope(const std::vector<double>& x, const std::vector<double>& y)
    {
        double x_mean = 0.0;
        double y_mean = 0.0;

        for (size_t i = 0; i < x.size(); ++i)
        {
            x_mean += x[i];
            y_mean += y[i];
        }

        x_mean /= static_cast<double>(x.size());
        y_mean /= static_cast<double>(y.size());

        double numerator   = 0.0;
        double denominator = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            numerator += (x[i] - x_mean) * (y[i] - y_mean);
            denominator += (x[i] - x_mean) * (x[i] - x_mean);
        }

        return denominator > 0.0 ? numerator / denominator : 0.0;
    }
} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Error argument! Usage: program Re[double > 0]" << std::endl;
        return 0;
    }

    const double channel_height = 1.0;
    const double lx1            = 20.0 * channel_height;
    const double ly1            = channel_height;

    const int nx1 = 256;
    const int ny1 = 64;

    const double hx                  = lx1 / nx1;
    const double hy                  = ly1 / ny1;
    const double Re                  = std::stod(argv[1]);
    const double density             = 1.0;
    const double hydraulic_diameter  = 2.0 * channel_height;
    const double inlet_velocity      = 1.0;
    const double dynamic_viscosity   = density * inlet_velocity * hydraulic_diameter / Re;
    const double kinematic_viscosity = dynamic_viscosity / density;
    const double convection_dt       = 0.5 * std::min(hx, hy) / std::max(std::abs(inlet_velocity), 1e-12);
    const double diffusion_dt        = 0.5 / (2.0 * kinematic_viscosity * (1.0 / (hx * hx) + 1.0 / (hy * hy)));
    const double dt                  = std::min(convection_dt, diffusion_dt);
    const double dpdx_theory         = -12.0 * dynamic_viscosity * inlet_velocity / (channel_height * channel_height);

    Geometry2D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.debugOutputDir =
        "./result/straight_channel_pressure_drop_2d_dimless/Re" + std::to_string(static_cast<int>(Re));

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = dt;
    time_cfg.num_iterations       = 12000;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_nu(kinematic_viscosity);

    Domain2DUniform A1(nx1, ny1, lx1, ly1, "A1");
    geo.add_domain(&A1);
    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);

    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    field2 u_A1;
    field2 v_A1;
    field2 p_A1;

    u.set_x_edge_field(&A1, u_A1);
    v.set_y_edge_field(&A1, v_A1);
    p.set_center_field(&A1, p_A1);

    std::cout << "mesh num = " << u_A1.get_size_n() << std::endl;

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

    std::vector<Domain2DUniform*> domains = {&A1};
    std::vector<LocationType>     dirs    = {
        LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(d, loc))
                continue;

            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            set_neumann_zero(p, d, loc);
        }
    }

    u.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A1, LocationType::XNegative, [&](double, double y) {
        const double y_norm = y / ly1;
        return 6.0 * inlet_velocity * (1.0 - y_norm) * y_norm;
    });

    u.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    u.set_boundary_value(&A1, LocationType::XPositive, 0.0);
    v.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    v.set_boundary_value(&A1, LocationType::XPositive, 0.0);
    ConcatPoissonSolver2D ns_p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &ns_p_solver);

    std::unordered_map<Domain2DUniform*, field2> prev_u;
    std::unordered_map<Domain2DUniform*, field2> prev_v;
    prev_u.emplace(&A1, field2(u_A1.get_nx(), u_A1.get_ny(), "u_prev_A1"));
    prev_v.emplace(&A1, field2(v_A1.get_nx(), v_A1.get_ny(), "v_prev_A1"));
    update_prev_velocity(u, prev_u);
    update_prev_velocity(v, prev_v);

    double u_res      = 0.0;
    double v_res      = 0.0;
    int    final_iter = time_cfg.num_iterations;

    for (int iter = 1; iter <= time_cfg.num_iterations; ++iter)
    {
        ns_solver.solve();

        if (iter % 200 == 0)
        {
            u_res = compute_velocity_residual(u, prev_u);
            v_res = compute_velocity_residual(v, prev_v);
            update_prev_velocity(u, prev_u);
            update_prev_velocity(v, prev_v);

            std::cout << "iter: " << iter << "/" << time_cfg.num_iterations << ", u_res = " << u_res
                      << ", v_res = " << v_res << std::endl;

            if (iter >= 2000 && std::max(u_res, v_res) < 1e-6)
            {
                final_iter = iter;
                break;
            }
        }

        if (std::isnan(u_A1(0, 0)))
        {
            std::cout << "Error: Find nan! Break solving." << std::endl;
            final_iter = iter;
            break;
        }
    }

    const auto calc_pressure = [&](double x, double) { return dpdx_theory * x; };

    // Use the same physical-pressure pattern as the validated 3D straight-channel
    // case: streamwise Dirichlet pressure on inlet/outlet and zero-Neumann on walls.
    p.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    p.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Dirichlet);
    p.set_boundary_value(&A1, LocationType::XNegative, calc_pressure);
    p.set_boundary_value(&A1, LocationType::XPositive, calc_pressure);
    p.set_buffer_value(&A1, LocationType::XNegative, calc_pressure);
    p.set_buffer_value(&A1, LocationType::XPositive, calc_pressure);
    p.set_value(calc_pressure);

    // Physical PPE uses streamwise Dirichlet pressure boundaries, which differ
    // from the NS pressure-correction stage. Build a dedicated Poisson solver
    // after the physical pressure BC types are finalized.
    ConcatPoissonSolver2D physical_p_solver(&p);
    PhysicalPESolver2D    ppe_solver(&u, &v, &p, &physical_p_solver, density);
    ppe_solver.solve();

    const double x_inlet_center       = 0.5 * hx;
    const double x_outlet_center      = lx1 - 0.5 * hx;
    const double pressure_drop_theory = -dpdx_theory * (x_outlet_center - x_inlet_center);

    std::vector<double> x_profile;
    std::vector<double> p_profile;
    x_profile.reserve(nx1);
    p_profile.reserve(nx1);

    for (int i = 0; i < nx1; ++i)
    {
        x_profile.push_back((i + 0.5) * hx);
        p_profile.push_back(p_A1.mean_at_x_axis(i));
    }

    const double p_inlet_mean            = p_profile.front();
    const double p_outlet_mean           = p_profile.back();
    const double pressure_drop_numerical = p_inlet_mean - p_outlet_mean;
    const double dpdx_numerical_fit      = compute_linear_fit_slope(x_profile, p_profile);

    double theory_sq = 0.0;
    double diff_sq   = 0.0;
    for (int i = 0; i < nx1; ++i)
    {
        const double p_theory = p_outlet_mean + dpdx_theory * (x_profile[i] - x_outlet_center);
        const double diff     = p_profile[i] - p_theory;
        theory_sq += p_theory * p_theory;
        diff_sq += diff * diff;
    }

    const double pressure_profile_l2_rel =
        theory_sq > 0.0 ? std::sqrt(diff_sq / theory_sq) : std::sqrt(diff_sq / static_cast<double>(nx1));
    const double pressure_drop_rel_err =
        std::abs(pressure_drop_numerical - pressure_drop_theory) / std::max(std::abs(pressure_drop_theory), 1e-14);
    const double dpdx_rel_err = std::abs(dpdx_numerical_fit - dpdx_theory) / std::max(std::abs(dpdx_theory), 1e-14);

    CSVHandler p_x_file(env_cfg.debugOutputDir + "/p_x");
    p_x_file.stream << "x,p_mean,p_theory_shifted\n";
    for (int i = 0; i < nx1; ++i)
    {
        const double p_theory = p_outlet_mean + dpdx_theory * (x_profile[i] - x_outlet_center);
        p_x_file.stream << x_profile[i] << ',' << p_profile[i] << ',' << p_theory << '\n';
    }

    CSVHandler summary_file(env_cfg.debugOutputDir + "/summary");
    summary_file.stream << "Re,iterations,u_res,v_res,dpdx_theory,dpdx_numerical_fit,dpdx_rel_err,pressure_drop_theory,"
                           "pressure_drop_numerical,pressure_drop_rel_err,pressure_profile_l2_rel\n";
    summary_file.stream << Re << ',' << final_iter << ',' << u_res << ',' << v_res << ',' << dpdx_theory << ','
                        << dpdx_numerical_fit << ',' << dpdx_rel_err << ',' << pressure_drop_theory << ','
                        << pressure_drop_numerical << ',' << pressure_drop_rel_err << ',' << pressure_profile_l2_rel
                        << '\n';

    std::cout << std::setprecision(12) << "dpdx_theory = " << dpdx_theory << std::endl;
    std::cout << std::setprecision(12) << "dpdx_numerical_fit = " << dpdx_numerical_fit << std::endl;
    std::cout << std::setprecision(12) << "pressure_drop_theory = " << pressure_drop_theory << std::endl;
    std::cout << std::setprecision(12) << "pressure_drop_numerical = " << pressure_drop_numerical << std::endl;
    std::cout << std::setprecision(12) << "pressure_drop_rel_err = " << pressure_drop_rel_err << std::endl;
    std::cout << std::setprecision(12) << "pressure_profile_l2_rel = " << pressure_profile_l2_rel << std::endl;
    std::cout << "Finished" << std::endl;
}
