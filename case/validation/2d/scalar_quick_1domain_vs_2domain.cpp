#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/case_base.hpp"
#include "io/csv_handler.h"
#include "io/csv_writer_2d.h"
#include "ns/scalar_solver2d.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr double kPi = 3.14159265358979323846;

    struct ErrorStats
    {
        double l1   = 0.0;
        double l2   = 0.0;
        double linf = 0.0;
    };

    void fill_constant_buffers(Variable2D& var, double value)
    {
        for (auto& [domain, buffer_map] : var.buffer_map)
        {
            for (auto& [loc, buffer] : buffer_map)
            {
                const int length =
                    (loc == LocationType::XNegative || loc == LocationType::XPositive) ? domain->ny : domain->nx;
                std::fill(buffer, buffer + length, value);
            }
        }
    }

    double initial_scalar(double x, double y) { return x + 0.05 * std::sin(kPi * x) * std::cos(2.0 * kPi * y); }

    ErrorStats compute_difference(const field2& a, const field2& b)
    {
        ErrorStats stats;

        double    l1_sum = 0.0;
        double    l2_sum = 0.0;
        const int nx     = a.get_nx();
        const int ny     = a.get_ny();

        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                const double diff     = a(i, j) - b(i, j);
                const double abs_diff = std::abs(diff);
                l1_sum += abs_diff;
                l2_sum += diff * diff;
                stats.linf = std::max(stats.linf, abs_diff);
            }
        }

        const double count = static_cast<double>(nx * ny);
        stats.l1           = l1_sum / count;
        stats.l2           = std::sqrt(l2_sum / count);
        return stats;
    }

    void assemble_global_center_field(const Variable2D& scalar_var, field2& global_field, double hx, double hy)
    {
        global_field.clear(0.0);

        for (const auto& [domain, local_ptr] : scalar_var.field_map)
        {
            const field2& local = *local_ptr;
            for (int i = 0; i < local.get_nx(); ++i)
            {
                for (int j = 0; j < local.get_ny(); ++j)
                {
                    const double x  = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
                    const double y  = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
                    const int    gi = static_cast<int>(std::lround(x / hx - 0.5));
                    const int    gj = static_cast<int>(std::lround(y / hy - 0.5));

                    if (gi < 0 || gi >= global_field.get_nx() || gj < 0 || gj >= global_field.get_ny())
                        throw std::runtime_error("assemble_global_center_field: index out of range");

                    global_field(gi, gj) = local(i, j);
                }
            }
        }
    }

    void set_channel_like_scalar_bc(Variable2D& scalar_var, Domain2DUniform* left_domain, Domain2DUniform* right_domain)
    {
        scalar_var.set_boundary_type(left_domain, LocationType::XNegative, PDEBoundaryType::Dirichlet);
        scalar_var.set_boundary_value(left_domain, LocationType::XNegative, 0.0);

        scalar_var.set_boundary_type(right_domain, LocationType::XPositive, PDEBoundaryType::Dirichlet);
        scalar_var.set_boundary_value(right_domain, LocationType::XPositive, 1.0);

        scalar_var.set_boundary_type(left_domain, LocationType::YNegative, PDEBoundaryType::Neumann);
        scalar_var.set_boundary_value(left_domain, LocationType::YNegative, 0.0);
        scalar_var.set_boundary_type(left_domain, LocationType::YPositive, PDEBoundaryType::Neumann);
        scalar_var.set_boundary_value(left_domain, LocationType::YPositive, 0.0);

        if (right_domain != left_domain)
        {
            scalar_var.set_boundary_type(right_domain, LocationType::YNegative, PDEBoundaryType::Neumann);
            scalar_var.set_boundary_value(right_domain, LocationType::YNegative, 0.0);
            scalar_var.set_boundary_type(right_domain, LocationType::YPositive, PDEBoundaryType::Neumann);
            scalar_var.set_boundary_value(right_domain, LocationType::YPositive, 0.0);
        }
    }

    field2 run_case(bool split_x, int n, double u0, double v0, double diffusivity, double dt, int num_steps)
    {
        Geometry2D geo;

        Domain2DUniform d_single(n, n, 1.0, 1.0, "A1");
        Domain2DUniform d_left(n / 2, n, 0.5, 1.0, "A1");
        Domain2DUniform d_right(n / 2, n, 0.5, 1.0, "A2");

        Domain2DUniform* left_domain  = nullptr;
        Domain2DUniform* right_domain = nullptr;

        if (split_x)
        {
            geo.connect(&d_left, LocationType::XPositive, &d_right);
            geo.axis(&d_left, LocationType::XNegative);
            geo.axis(&d_left, LocationType::YNegative);
            left_domain  = &d_left;
            right_domain = &d_right;
        }
        else
        {
            geo.add_domain(&d_single);
            geo.axis(&d_single, LocationType::XNegative);
            geo.axis(&d_single, LocationType::YNegative);
            left_domain  = &d_single;
            right_domain = &d_single;
        }

        Variable2D u("u"), v("v"), c("c");
        u.set_geometry(geo);
        v.set_geometry(geo);
        c.set_geometry(geo);

        std::vector<field2> u_fields;
        std::vector<field2> v_fields;
        std::vector<field2> c_fields;

        if (split_x)
        {
            u_fields.emplace_back("u_A1");
            u_fields.emplace_back("u_A2");
            v_fields.emplace_back("v_A1");
            v_fields.emplace_back("v_A2");
            c_fields.emplace_back("c_A1");
            c_fields.emplace_back("c_A2");

            u.set_x_edge_field(&d_left, u_fields[0]);
            u.set_x_edge_field(&d_right, u_fields[1]);
            v.set_y_edge_field(&d_left, v_fields[0]);
            v.set_y_edge_field(&d_right, v_fields[1]);
            c.set_center_field(&d_left, c_fields[0]);
            c.set_center_field(&d_right, c_fields[1]);
        }
        else
        {
            u_fields.emplace_back("u_A1");
            v_fields.emplace_back("v_A1");
            c_fields.emplace_back("c_A1");

            u.set_x_edge_field(&d_single, u_fields[0]);
            v.set_y_edge_field(&d_single, v_fields[0]);
            c.set_center_field(&d_single, c_fields[0]);
        }

        set_channel_like_scalar_bc(c, left_domain, right_domain);

        u.set_value([&](double, double) { return u0; });
        v.set_value([&](double, double) { return v0; });
        c.set_value([](double x, double y) { return initial_scalar(x, y); });

        fill_constant_buffers(u, u0);
        fill_constant_buffers(v, v0);

        TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
        time_cfg.dt                   = dt;
        time_cfg.num_iterations       = num_steps;

        ScalarSolver2D solver(&u, &v, &c, diffusivity, DifferenceSchemeType::Conv_QUICK_Diff_Center2nd);
        solver.variable_check();

        for (int step = 0; step < num_steps; ++step)
            solver.solve();

        field2 global_field(n, n, split_x ? "c_two_domain" : "c_one_domain");
        assemble_global_center_field(c, global_field, 1.0 / static_cast<double>(n), 1.0 / static_cast<double>(n));
        return global_field;
    }

    class ScalarQuickSplitCompareCase : public CaseBase
    {
    public:
        ScalarQuickSplitCompareCase(int argc, char* argv[])
            : CaseBase(argc, argv)
        {}

        void read_paras() override
        {
            CaseBase::read_paras();

            IO::read_number(para_map, "n", n);
            IO::read_number(para_map, "u0", u0);
            IO::read_number(para_map, "v0", v0);
            IO::read_number(para_map, "diffusivity", diffusivity);
            IO::read_number(para_map, "final_time", final_time);
            IO::read_number(para_map, "tolerance", tolerance);

            if (n < 16 || n % 2 != 0)
                throw std::runtime_error("n must be an even integer >= 16");
            if (diffusivity < 0.0)
                throw std::runtime_error("diffusivity must be >= 0");
            if (final_time <= 0.0)
                throw std::runtime_error("final_time must be > 0");
            if (tolerance <= 0.0)
                throw std::runtime_error("tolerance must be > 0");
        }

        bool record_paras() override
        {
            if (!CaseBase::record_paras())
                return false;

            paras_record.record("n", n)
                .record("u0", u0)
                .record("v0", v0)
                .record("diffusivity", diffusivity)
                .record("final_time", final_time)
                .record("tolerance", tolerance)
                .record("scheme", "quick");
            return true;
        }

        int    n           = 64;
        double u0          = 1.0;
        double v0          = 0.25;
        double diffusivity = 1e-3;
        double final_time  = 0.005;
        double tolerance   = 1e-4;
    };
} // namespace

int main(int argc, char* argv[])
{
    ScalarQuickSplitCompareCase case_param(argc, argv);
    case_param.read_paras();
    case_param.record_paras();

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showCurrentStep    = false;
    env_cfg.showGmresRes       = false;
    env_cfg.debugOutputDir     = case_param.root_dir;

    const double h            = 1.0 / static_cast<double>(case_param.n);
    double       dt_candidate = std::numeric_limits<double>::infinity();
    if (std::abs(case_param.u0) > 1e-14)
        dt_candidate = std::min(dt_candidate, 0.02 * h / std::abs(case_param.u0));
    if (std::abs(case_param.v0) > 1e-14)
        dt_candidate = std::min(dt_candidate, 0.02 * h / std::abs(case_param.v0));
    if (case_param.diffusivity > 0.0)
    {
        const double dt_diff = 0.2 / (2.0 * case_param.diffusivity * (2.0 / (h * h)));
        dt_candidate         = std::min(dt_candidate, dt_diff);
    }
    if (!std::isfinite(dt_candidate))
        dt_candidate = case_param.final_time;

    const int    num_steps = std::max(1, static_cast<int>(std::ceil(case_param.final_time / dt_candidate)));
    const double dt        = case_param.final_time / static_cast<double>(num_steps);

    field2 one_domain =
        run_case(false, case_param.n, case_param.u0, case_param.v0, case_param.diffusivity, dt, num_steps);
    field2 two_domain =
        run_case(true, case_param.n, case_param.u0, case_param.v0, case_param.diffusivity, dt, num_steps);
    field2 diff_field = one_domain - two_domain;

    const ErrorStats diff_stats = compute_difference(one_domain, two_domain);
    const bool       pass       = std::isfinite(diff_stats.linf) && diff_stats.linf <= case_param.tolerance;

    CSVHandler summary_file(case_param.root_dir + "/summary");
    summary_file.stream << "n,u0,v0,diffusivity,final_time,dt,num_steps,l1,l2,linf,tolerance,pass\n";
    summary_file.stream << case_param.n << ',' << case_param.u0 << ',' << case_param.v0 << ',' << case_param.diffusivity
                        << ',' << case_param.final_time << ',' << dt << ',' << num_steps << ',' << diff_stats.l1 << ','
                        << diff_stats.l2 << ',' << diff_stats.linf << ',' << case_param.tolerance << ','
                        << (pass ? 1 : 0) << '\n';

    IO::write_csv(one_domain, case_param.root_dir + "/scalar_one_domain");
    IO::write_csv(two_domain, case_param.root_dir + "/scalar_two_domain");
    IO::write_csv(diff_field, case_param.root_dir + "/scalar_diff");

    std::cout << std::setprecision(16) << "scalar_quick_1domain_vs_2domain: dt=" << dt << ", num_steps=" << num_steps
              << ", L1=" << diff_stats.l1 << ", L2=" << diff_stats.l2 << ", Linf=" << diff_stats.linf
              << ", tolerance=" << case_param.tolerance << ", pass=" << pass << std::endl;

    return pass ? 0 : 2;
}
