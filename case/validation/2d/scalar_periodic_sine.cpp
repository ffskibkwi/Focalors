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
#include <cctype>
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
        double l1     = 0.0;
        double l2     = 0.0;
        double linf   = 0.0;
        double l2_rel = 0.0;
    };

    struct LevelResult
    {
        int    nx        = 0;
        int    ny        = 0;
        double h         = 0.0;
        double dt        = 0.0;
        int    num_steps = 0;
        ErrorStats error;
    };

    std::string to_lower(std::string text)
    {
        std::transform(text.begin(),
                       text.end(),
                       text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    DifferenceSchemeType parse_scheme(const std::string& scheme_name)
    {
        const std::string key = to_lower(scheme_name);
        if (key == "cd2" || key == "center" || key == "central" || key == "conv_center2nd_diff_center2nd")
            return DifferenceSchemeType::Conv_Center2nd_Diff_Center2nd;
        if (key == "uw1" || key == "upwind" || key == "upwind1st" || key == "conv_upwind1st_diff_center2nd")
            return DifferenceSchemeType::Conv_Upwind1st_Diff_Center2nd;
        if (key == "quick" || key == "conv_quick_diff_center2nd")
            return DifferenceSchemeType::Conv_QUICK_Diff_Center2nd;
        if (key == "tvd" || key == "vanleer" || key == "tvd_vanleer" || key == "conv_tvd_vanleer_diff_center2nd")
            return DifferenceSchemeType::Conv_TVD_VanLeer_Diff_Center2nd;

        throw std::runtime_error("Unknown scalar scheme: " + scheme_name);
    }

    double wrap_periodic(double x, double length)
    {
        double wrapped = std::fmod(x, length);
        if (wrapped < 0.0)
            wrapped += length;
        return wrapped;
    }

    double exact_solution(double x,
                          double y,
                          double time,
                          double lx,
                          double ly,
                          double u0,
                          double v0,
                          double diffusivity)
    {
        const double x_shift = wrap_periodic(x - u0 * time, lx);
        const double y_shift = wrap_periodic(y - v0 * time, ly);
        const double kx      = 2.0 * kPi / lx;
        const double ky      = 2.0 * kPi / ly;
        const double decay   = std::exp(-diffusivity * (kx * kx + ky * ky) * time);

        return decay * std::sin(kx * x_shift) * std::cos(ky * y_shift);
    }

    ErrorStats compute_error(const field2& numerical, const field2& exact)
    {
        ErrorStats stats;

        double l1_sum    = 0.0;
        double l2_sum    = 0.0;
        double exact_sum = 0.0;

        const int nx = numerical.get_nx();
        const int ny = numerical.get_ny();

        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                const double diff     = numerical(i, j) - exact(i, j);
                const double abs_diff = std::abs(diff);

                l1_sum += abs_diff;
                l2_sum += diff * diff;
                exact_sum += exact(i, j) * exact(i, j);
                stats.linf = std::max(stats.linf, abs_diff);
            }
        }

        const double count = static_cast<double>(nx * ny);
        stats.l1           = l1_sum / count;
        stats.l2           = std::sqrt(l2_sum / count);
        stats.l2_rel       = exact_sum > 0.0 ? std::sqrt(l2_sum / exact_sum) : 0.0;
        return stats;
    }

    void fill_velocity_buffers(Variable2D& u, Variable2D& v, Domain2DUniform* domain, double u0, double v0)
    {
        for (int j = 0; j < domain->ny; ++j)
        {
            u.buffer_map[domain][LocationType::XNegative][j] = u0;
            u.buffer_map[domain][LocationType::XPositive][j] = u0;
        }

        for (int i = 0; i < domain->nx; ++i)
        {
            u.buffer_map[domain][LocationType::YNegative][i] = u0;
            u.buffer_map[domain][LocationType::YPositive][i] = u0;
            v.buffer_map[domain][LocationType::YNegative][i] = v0;
            v.buffer_map[domain][LocationType::YPositive][i] = v0;
        }

        for (int j = 0; j < domain->ny; ++j)
        {
            v.buffer_map[domain][LocationType::XNegative][j] = v0;
            v.buffer_map[domain][LocationType::XPositive][j] = v0;
        }
    }

    double compute_observed_order(double coarse_error, double fine_error, double coarse_h, double fine_h)
    {
        if (coarse_error <= 0.0 || fine_error <= 0.0 || coarse_h <= 0.0 || fine_h <= 0.0 || coarse_h == fine_h)
            return std::numeric_limits<double>::quiet_NaN();

        return std::log(coarse_error / fine_error) / std::log(coarse_h / fine_h);
    }

    class ScalarPeriodicSineCase : public CaseBase
    {
    public:
        ScalarPeriodicSineCase(int argc, char* argv[])
            : CaseBase(argc, argv)
        {}

        void read_paras() override
        {
            CaseBase::read_paras();

            IO::read_number(para_map, "base_n", base_n);
            IO::read_number(para_map, "levels", levels);
            IO::read_number(para_map, "final_time", final_time);
            IO::read_number(para_map, "u0", u0);
            IO::read_number(para_map, "v0", v0);
            IO::read_number(para_map, "diffusivity", diffusivity);
            IO::read_number(para_map, "adv_cfl", adv_cfl);
            IO::read_number(para_map, "diff_cfl", diff_cfl);
            IO::read_string(para_map, "scheme", scheme_name);

            if (base_n < 8)
                throw std::runtime_error("base_n must be >= 8");
            if (levels < 1)
                throw std::runtime_error("levels must be >= 1");
            if (final_time <= 0.0)
                throw std::runtime_error("final_time must be > 0");
            if (diffusivity < 0.0)
                throw std::runtime_error("diffusivity must be >= 0");
            if (adv_cfl <= 0.0 || diff_cfl <= 0.0)
                throw std::runtime_error("adv_cfl and diff_cfl must be > 0");

            scheme = parse_scheme(scheme_name);
        }

        bool record_paras() override
        {
            if (!CaseBase::record_paras())
                return false;

            paras_record.record("base_n", base_n)
                .record("levels", levels)
                .record("final_time", final_time)
                .record("u0", u0)
                .record("v0", v0)
                .record("diffusivity", diffusivity)
                .record("adv_cfl", adv_cfl)
                .record("diff_cfl", diff_cfl)
                .record("scheme", scheme_name);

            return true;
        }

        int                  base_n      = 32;
        int                  levels      = 3;
        double               final_time  = 0.05;
        double               u0          = 1.0;
        double               v0          = 0.5;
        double               diffusivity = 1e-3;
        double               adv_cfl     = 0.02;
        double               diff_cfl    = 0.2;
        std::string          scheme_name = "quick";
        DifferenceSchemeType scheme      = DifferenceSchemeType::Conv_QUICK_Diff_Center2nd;
    };
} // namespace

int main(int argc, char* argv[])
{
    ScalarPeriodicSineCase case_param(argc, argv);
    case_param.read_paras();
    case_param.record_paras();

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showCurrentStep    = false;
    env_cfg.showGmresRes       = false;
    env_cfg.debugOutputDir     = case_param.root_dir;

    std::cout << "2D scalar periodic sine validation" << std::endl;
    std::cout << "root_dir = " << case_param.root_dir << std::endl;
    std::cout << "scheme = " << case_param.scheme << std::endl;
    std::cout << "base_n = " << case_param.base_n << ", levels = " << case_param.levels << std::endl;
    std::cout << "u0 = " << case_param.u0 << ", v0 = " << case_param.v0
              << ", diffusivity = " << case_param.diffusivity << std::endl;
    std::cout << "final_time = " << case_param.final_time << std::endl;

    CSVHandler summary_file(case_param.root_dir + "/summary");
    summary_file.stream << "level,nx,ny,h,dt,num_steps,l1,l2,linf,l2_rel,order_l1,order_l2,order_linf\n";

    std::vector<LevelResult> results;
    results.reserve(static_cast<std::size_t>(case_param.levels));

    for (int level = 0; level < case_param.levels; ++level)
    {
        const int    n  = case_param.base_n * (1 << level);
        const double lx = 1.0;
        const double ly = 1.0;
        const double hx = lx / static_cast<double>(n);
        const double hy = ly / static_cast<double>(n);

        double dt_adv = std::numeric_limits<double>::infinity();
        if (std::abs(case_param.u0) > 1e-14)
            dt_adv = std::min(dt_adv, case_param.adv_cfl * hx / std::abs(case_param.u0));
        if (std::abs(case_param.v0) > 1e-14)
            dt_adv = std::min(dt_adv, case_param.adv_cfl * hy / std::abs(case_param.v0));
        if (!std::isfinite(dt_adv))
            dt_adv = case_param.final_time;

        double dt_diff = std::numeric_limits<double>::infinity();
        if (case_param.diffusivity > 0.0)
        {
            dt_diff = case_param.diff_cfl /
                      (2.0 * case_param.diffusivity * (1.0 / (hx * hx) + 1.0 / (hy * hy)));
        }

        const double dt_candidate = std::min(dt_adv, dt_diff);
        const int    num_steps =
            std::max(1, static_cast<int>(std::ceil(case_param.final_time / std::max(dt_candidate, 1e-14))));
        const double dt = case_param.final_time / static_cast<double>(num_steps);

        Geometry2D geo;
        Domain2DUniform domain(n, n, lx, ly, "A1");
        geo.add_domain(&domain);
        geo.axis(&domain, LocationType::XNegative);
        geo.axis(&domain, LocationType::YNegative);

        Variable2D u("u"), v("v"), c("c");
        u.set_geometry(geo);
        v.set_geometry(geo);
        c.set_geometry(geo);

        field2 u_field;
        field2 v_field;
        field2 c_field;
        field2 c_exact;
        field2 c_error;

        u.set_x_edge_field(&domain, u_field);
        v.set_y_edge_field(&domain, v_field);
        c.set_center_field(&domain, c_field);
        c_exact.init(n, n, "c_exact");
        c_error.init(n, n, "c_error");

        u.set_boundary_type(PDEBoundaryType::Periodic);
        v.set_boundary_type(PDEBoundaryType::Periodic);
        c.set_boundary_type(PDEBoundaryType::Periodic);

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                u_field(i, j) = case_param.u0;
                v_field(i, j) = case_param.v0;

                const double x = (static_cast<double>(i) + 0.5) * hx;
                const double y = (static_cast<double>(j) + 0.5) * hy;
                c_field(i, j)  = exact_solution(
                    x, y, 0.0, lx, ly, case_param.u0, case_param.v0, case_param.diffusivity);
            }
        }

        fill_velocity_buffers(u, v, &domain, case_param.u0, case_param.v0);

        TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
        time_cfg.dt                   = dt;
        time_cfg.num_iterations       = num_steps;

        ScalarSolver2D solver(&u, &v, &c, case_param.diffusivity, case_param.scheme);
        solver.variable_check();

        for (int step = 0; step < num_steps; ++step)
            solver.solve();

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                const double x = (static_cast<double>(i) + 0.5) * hx;
                const double y = (static_cast<double>(j) + 0.5) * hy;
                c_exact(i, j)  = exact_solution(
                    x, y, case_param.final_time, lx, ly, case_param.u0, case_param.v0, case_param.diffusivity);
                c_error(i, j) = c_field(i, j) - c_exact(i, j);
            }
        }

        LevelResult result;
        result.nx        = n;
        result.ny        = n;
        result.h         = hx;
        result.dt        = dt;
        result.num_steps = num_steps;
        result.error     = compute_error(c_field, c_exact);
        results.push_back(result);

        const bool finest_level = (level == case_param.levels - 1);
        if (finest_level)
        {
            IO::write_csv(c_field, case_param.root_dir + "/scalar_numeric_finest");
            IO::write_csv(c_exact, case_param.root_dir + "/scalar_exact_finest");
            IO::write_csv(c_error, case_param.root_dir + "/scalar_error_finest");

            CSVHandler centerline_file(case_param.root_dir + "/centerline_y_mid_finest");
            centerline_file.stream << "x,numeric,exact,error\n";
            const int j_mid = n / 2;
            for (int i = 0; i < n; ++i)
            {
                const double x = (static_cast<double>(i) + 0.5) * hx;
                centerline_file.stream << x << ',' << c_field(i, j_mid) << ',' << c_exact(i, j_mid) << ','
                                       << c_error(i, j_mid) << '\n';
            }
        }
    }

    for (std::size_t idx = 0; idx < results.size(); ++idx)
    {
        const auto& result = results[idx];

        double order_l1   = std::numeric_limits<double>::quiet_NaN();
        double order_l2   = std::numeric_limits<double>::quiet_NaN();
        double order_linf = std::numeric_limits<double>::quiet_NaN();
        if (idx > 0)
        {
            const auto& coarse = results[idx - 1];
            order_l1           = compute_observed_order(coarse.error.l1, result.error.l1, coarse.h, result.h);
            order_l2           = compute_observed_order(coarse.error.l2, result.error.l2, coarse.h, result.h);
            order_linf         = compute_observed_order(coarse.error.linf, result.error.linf, coarse.h, result.h);
        }

        summary_file.stream << idx << ',' << result.nx << ',' << result.ny << ',' << result.h << ',' << result.dt
                            << ',' << result.num_steps << ',' << result.error.l1 << ',' << result.error.l2 << ','
                            << result.error.linf << ',' << result.error.l2_rel << ',';
        if (std::isnan(order_l1))
            summary_file.stream << ',';
        else
            summary_file.stream << order_l1 << ',';
        if (std::isnan(order_l2))
            summary_file.stream << ',';
        else
            summary_file.stream << order_l2 << ',';
        if (std::isnan(order_linf))
            summary_file.stream << '\n';
        else
            summary_file.stream << order_linf << '\n';

        std::cout << std::setprecision(10) << "level " << idx << ": N=" << result.nx << ", dt=" << result.dt
                  << ", steps=" << result.num_steps << ", L1=" << result.error.l1 << ", L2=" << result.error.l2
                  << ", Linf=" << result.error.linf << ", L2_rel=" << result.error.l2_rel;
        if (!std::isnan(order_l2))
            std::cout << ", order(L2)=" << order_l2;
        std::cout << std::endl;
    }

    std::cout << "Finished" << std::endl;
    return 0;
}
