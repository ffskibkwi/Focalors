#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "io/csv_handler.h"
#include "io/vtk_writer.h"
#include "ns/ns_solver3d.h"
#include "ns/physical_pe_solver3d.h"
#include "pe/concat/concat_solver3d.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
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

    double compute_l2_norm(const std::vector<double>& values)
    {
        double sum_sq = 0.0;
        for (double value : values)
            sum_sq += value * value;
        return std::sqrt(sum_sq);
    }

    double rectangular_duct_mean_coefficient(double width, double height, int max_odd_mode)
    {
        const double pi  = std::acos(-1.0);
        const double pi6 = std::pow(pi, 6);

        double coeff = 0.0;
        for (int m = 1; m <= max_odd_mode; m += 2)
        {
            const double m_val = static_cast<double>(m);
            for (int n = 1; n <= max_odd_mode; n += 2)
            {
                const double n_val       = static_cast<double>(n);
                const double lambda_no_pi = (m_val * m_val) / (width * width) + (n_val * n_val) / (height * height);
                const double denominator = (m_val * m_val) * (n_val * n_val) * pi6 * lambda_no_pi;
                coeff += 64.0 / denominator;
            }
        }
        return coeff;
    }

    double rectangular_duct_velocity(double y,
                                     double z,
                                     double width,
                                     double height,
                                     double dynamic_viscosity,
                                     double dpdx,
                                     int    max_odd_mode)
    {
        const double pi  = std::acos(-1.0);
        const double pi4 = std::pow(pi, 4);

        double series = 0.0;
        for (int m = 1; m <= max_odd_mode; m += 2)
        {
            const double m_val  = static_cast<double>(m);
            const double sin_my = std::sin(m_val * pi * y / width);
            for (int n = 1; n <= max_odd_mode; n += 2)
            {
                const double n_val       = static_cast<double>(n);
                const double lambda_no_pi = (m_val * m_val) / (width * width) + (n_val * n_val) / (height * height);
                const double coeff       = 16.0 / (m_val * n_val * pi4 * lambda_no_pi);
                series += coeff * sin_my * std::sin(n_val * pi * z / height);
            }
        }

        return -(dpdx / dynamic_viscosity) * series;
    }

    double rectangular_duct_mean_velocity_at_z(double z,
                                               double width,
                                               double height,
                                               double dynamic_viscosity,
                                               double dpdx,
                                               int    max_odd_mode)
    {
        const double pi  = std::acos(-1.0);
        const double pi5 = std::pow(pi, 5);

        double series = 0.0;
        for (int m = 1; m <= max_odd_mode; m += 2)
        {
            const double m_val = static_cast<double>(m);
            for (int n = 1; n <= max_odd_mode; n += 2)
            {
                const double n_val       = static_cast<double>(n);
                const double lambda_no_pi = (m_val * m_val) / (width * width) + (n_val * n_val) / (height * height);
                const double coeff       = 32.0 / (m_val * m_val * n_val * pi5 * lambda_no_pi);
                series += coeff * std::sin(n_val * pi * z / height);
            }
        }

        return -(dpdx / dynamic_viscosity) * series;
    }
} // namespace

int main(int argc, char* argv[])
{
    TIMER_BEGIN(Init, "Init", TimeRecordType::None, true);

    if (argc != 2)
    {
        std::cerr << "Error argument! Usage: program Re[double > 0]" << std::endl;
        return 0;
    }

    double channel_height = 1.0;
    double alpha          = 0.5;

    double lx1 = 100 * channel_height;
    double ly1 = 1.0 / alpha * channel_height;
    double lz1 = channel_height;

    int nx1 = 512;
    int ny1 = 64;
    int nz1 = 64;

    double hx = lx1 / nx1;
    double hy = ly1 / ny1;
    double hz = lz1 / nz1;

    double dt = hx / 10.0;

    double Re                = std::stod(argv[1]);
    double density           = 1e3;
    double dynamic_viscosity = 1.01e-3;
    // hydrodynamic_diameter = 2ab/(a+b) = 2aαa/(a+αa) = 2aα/(1+α)
    double hydrodynamic_diameter = 2.0 * channel_height * alpha / (1 + alpha);
    double inlet_velocity        = Re * dynamic_viscosity / (density * hydrodynamic_diameter);
    double kinematic_viscosity   = dynamic_viscosity / density;
    double channel_span          = ly1;
    int    rectangular_series_max_odd = 99;
    double rectangular_mean_coeff =
        rectangular_duct_mean_coefficient(channel_span, channel_height, rectangular_series_max_odd);
    double dpdx_theory = -dynamic_viscosity * inlet_velocity / rectangular_mean_coeff;

    // Geometry: Straight channel
    Geometry3D geo;

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.debugOutputDir =
        "./result/straight_channel_pressure_drop_3d_rectangular_duct/Re" + std::to_string(static_cast<int>(Re));

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = dt;
    time_cfg.num_iterations       = 3e4;

    PhysicsConfig& physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_nu(kinematic_viscosity);

    Domain3DUniform A1(nx1, ny1, nz1, lx1, ly1, lz1, "A1");

    geo.add_domain(&A1);

    geo.axis(&A1, LocationType::XNegative);
    geo.axis(&A1, LocationType::YNegative);
    geo.axis(&A1, LocationType::ZNegative);

    // Variable2Ds
    Variable3D u("u"), v("v"), w("w"), p_ns("p_ns"), p_phys("p_phys");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p_ns.set_geometry(geo);
    p_phys.set_geometry(geo);

    // Fields on each domain
    field3 u_A1;
    field3 v_A1;
    field3 w_A1;
    field3 p_ns_A1;
    field3 p_phys_A1;

    u.set_x_face_center_field(&A1, u_A1);
    v.set_y_face_center_field(&A1, v_A1);
    w.set_z_face_center_field(&A1, w_A1);
    p_ns.set_center_field(&A1, p_ns_A1);
    p_phys.set_center_field(&A1, p_phys_A1);

    std::cout << "mesh num = " << u_A1.get_size_n() << std::endl;

    // Helper setters
    auto set_dirichlet_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable3D& var, Domain3DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto isdjacented = [&](Domain3DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };

    // Default outer boundaries
    std::vector<Domain3DUniform*> domains = {&A1};
    std::vector<LocationType>     dirs    = {LocationType::XNegative,
                                             LocationType::XPositive,
                                             LocationType::YNegative,
                                             LocationType::YPositive,
                                             LocationType::ZNegative,
                                             LocationType::ZPositive};

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
            set_neumann_zero(p_ns, d, loc);
            set_neumann_zero(p_phys, d, loc);
        }
    }

    const auto calc_zero = [](double, double, double) { return 0.0; };
    const auto calc_velocity = [&](double, double y, double z) {
        return rectangular_duct_velocity(y, z, channel_span, channel_height, dynamic_viscosity, dpdx_theory,
                                         rectangular_series_max_odd);
    };
    const auto calc_velocity_mean_at_z = [&](double z) {
        return rectangular_duct_mean_velocity_at_z(z, channel_span, channel_height, dynamic_viscosity, dpdx_theory,
                                                   rectangular_series_max_odd);
    };
    const auto calc_pressure = [&](double x, double, double) { return dpdx_theory * x; };

    u.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A1, LocationType::XNegative, calc_velocity);
    v.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    v.set_boundary_value(&A1, LocationType::XNegative, 0.0);
    w.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    w.set_boundary_value(&A1, LocationType::XNegative, 0.0);

    // Outlet
    u.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    v.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);
    w.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Neumann);

    u.set_value(calc_velocity);
    v.set_value(calc_zero);
    w.set_value(calc_zero);
    p_ns.set_value(calc_zero);

    p_phys.set_buffer(calc_zero);
    p_phys.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    p_phys.set_boundary_type(&A1, LocationType::XPositive, PDEBoundaryType::Dirichlet);
    p_phys.set_boundary_value(&A1, LocationType::XNegative, calc_pressure);
    p_phys.set_boundary_value(&A1, LocationType::XPositive, calc_pressure);
    p_phys.set_buffer_value(&A1, LocationType::XNegative, calc_pressure);
    p_phys.set_buffer_value(&A1, LocationType::XPositive, calc_pressure);
    p_phys.set_value(calc_pressure);

    ConcatPoissonSolver3D ns_p_solver(&p_ns);
    ConcatNSSolver3D      ns_solver(&u, &v, &w, &p_ns, &ns_p_solver);

    ConcatPoissonSolver3D physical_p_solver(&p_phys);
    PhysicalPESolver3D    ppe_solver(&u, &v, &w, &p_phys, &physical_p_solver, density);

    VTKWriter vtk_writer;
    vtk_writer.add_vector_as_cell_data(&u, &v, &w, "velocity");
    vtk_writer.add_scalar_as_cell_data(&p_phys);
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

        ns_solver.solve();

        if (iter % 200 == 0)
        {
            ppe_solver.solve();

            CSVHandler p_YZ_file(env_cfg.debugOutputDir + "/p_YZ");
            for (int i = 0; i < nx1; i++)
            {
                p_YZ_file.stream << p_phys_A1.mean_at_yz_plane(i);
                if (i != nx1 - 1)
                    p_YZ_file.stream << ',';
                else
                    p_YZ_file.stream << std::endl;
            }
        }

        if (iter % 100 == 0)
        {
            env_cfg.track_pe_solve_detail_time = false;
            env_cfg.showGmresRes               = false;
        }

        if (false && iter % static_cast<int>(1e4) == 0)
        {
            static int count = 0;
            vtk_writer.write(env_cfg.debugOutputDir + "/vtk/" + std::to_string(count++));
        }

        if (std::isnan(u_A1(0, 0, 0)))
        {
            std::cout << "Error: Find nan! Break solving." << std::endl;
            break;
        }
    }

    const double x_inlet_center       = 0.5 * hx;
    const double x_outlet_center      = lx1 - 0.5 * hx;
    const double pressure_drop_theory = -dpdx_theory * (x_outlet_center - x_inlet_center);

    std::vector<double> x_profile;
    std::vector<double> p_profile;
    std::vector<double> p_theory_profile;
    std::vector<double> p_diff_profile;
    x_profile.reserve(nx1);
    p_profile.reserve(nx1);
    p_theory_profile.reserve(nx1);
    p_diff_profile.reserve(nx1);

    for (int i = 0; i < nx1; ++i)
    {
        const double x         = (i + 0.5) * hx;
        const double p_numeric = p_phys_A1.mean_at_yz_plane(i);
        const double p_theory  = calc_pressure(x, 0.0, 0.0);
        x_profile.push_back(x);
        p_profile.push_back(p_numeric);
        p_theory_profile.push_back(p_theory);
        p_diff_profile.push_back(p_numeric - p_theory);
    }

    const double p_inlet_mean            = p_profile.front();
    const double p_outlet_mean           = p_profile.back();
    const double pressure_drop_numerical = p_inlet_mean - p_outlet_mean;
    const double dpdx_numerical_fit      = compute_linear_fit_slope(x_profile, p_profile);
    const double pressure_profile_l2_rel =
        compute_l2_norm(p_diff_profile) / std::max(compute_l2_norm(p_theory_profile), 1e-14);
    const double pressure_drop_rel_err =
        std::abs(pressure_drop_numerical - pressure_drop_theory) / std::max(std::abs(pressure_drop_theory), 1e-14);
    const double dpdx_rel_err = std::abs(dpdx_numerical_fit - dpdx_theory) / std::max(std::abs(dpdx_theory), 1e-14);

    std::vector<double> z_profile;
    std::vector<double> u_profile;
    std::vector<double> u_theory_profile;
    std::vector<double> u_diff_profile;
    z_profile.reserve(nz1);
    u_profile.reserve(nz1);
    u_theory_profile.reserve(nz1);
    u_diff_profile.reserve(nz1);

    for (int k = 0; k < nz1; ++k)
    {
        double u_sum = 0.0;
        for (int i = 0; i < u_A1.get_nx(); ++i)
        {
            for (int j = 0; j < u_A1.get_ny(); ++j)
                u_sum += u_A1(i, j, k);
        }

        const double z        = (k + 0.5) * hz;
        const double u_mean   = u_sum / static_cast<double>(u_A1.get_nx() * u_A1.get_ny());
        const double u_theory = calc_velocity_mean_at_z(z);
        z_profile.push_back(z);
        u_profile.push_back(u_mean);
        u_theory_profile.push_back(u_theory);
        u_diff_profile.push_back(u_mean - u_theory);
    }

    const double u_profile_l2_rel = compute_l2_norm(u_diff_profile) / std::max(compute_l2_norm(u_theory_profile), 1e-14);

    CSVHandler pressure_profile_file(env_cfg.debugOutputDir + "/pressure_profile");
    pressure_profile_file.stream << "x,p_mean,p_theory,p_diff\n";
    for (int i = 0; i < nx1; ++i)
    {
        pressure_profile_file.stream << x_profile[i] << ',' << p_profile[i] << ',' << p_theory_profile[i] << ','
                                     << p_diff_profile[i] << '\n';
    }

    CSVHandler velocity_profile_file(env_cfg.debugOutputDir + "/velocity_profile_z");
    velocity_profile_file.stream << "z,u_mean,u_theory,u_diff\n";
    for (int k = 0; k < nz1; ++k)
    {
        velocity_profile_file.stream << z_profile[k] << ',' << u_profile[k] << ',' << u_theory_profile[k] << ','
                                     << u_diff_profile[k] << '\n';
    }

    CSVHandler summary_file(env_cfg.debugOutputDir + "/summary");
    summary_file.stream << "Re,dpdx_theory,dpdx_numerical_fit,dpdx_rel_err,pressure_drop_theory,"
                           "pressure_drop_numerical,pressure_drop_rel_err,pressure_profile_l2_rel,u_profile_l2_rel\n";
    summary_file.stream << Re << ',' << dpdx_theory << ',' << dpdx_numerical_fit << ',' << dpdx_rel_err << ','
                        << pressure_drop_theory << ',' << pressure_drop_numerical << ',' << pressure_drop_rel_err
                        << ',' << pressure_profile_l2_rel << ',' << u_profile_l2_rel << '\n';

    std::cout << std::setprecision(12) << "dpdx_theory = " << dpdx_theory << std::endl;
    std::cout << std::setprecision(12) << "dpdx_numerical_fit = " << dpdx_numerical_fit << std::endl;
    std::cout << std::setprecision(12) << "pressure_drop_theory = " << pressure_drop_theory << std::endl;
    std::cout << std::setprecision(12) << "pressure_drop_numerical = " << pressure_drop_numerical << std::endl;
    std::cout << std::setprecision(12) << "pressure_drop_rel_err = " << pressure_drop_rel_err << std::endl;
    std::cout << std::setprecision(12) << "pressure_profile_l2_rel = " << pressure_profile_l2_rel << std::endl;
    std::cout << std::setprecision(12) << "u_profile_l2_rel = " << u_profile_l2_rel << std::endl;
    std::cout << "Finished" << std::endl;
}
