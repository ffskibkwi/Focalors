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
#include <stdexcept>
#include <string>

namespace
{
    struct ErrorStats
    {
        double l1   = 0.0;
        double l2   = 0.0;
        double linf = 0.0;
    };

    double exact_solution(double x) { return x; }

    ErrorStats compute_error(const field2& numerical, double hx)
    {
        ErrorStats stats;

        double    l1_sum = 0.0;
        double    l2_sum = 0.0;
        const int nx     = numerical.get_nx();
        const int ny     = numerical.get_ny();

        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                const double x        = (static_cast<double>(i) + 0.5) * hx;
                const double diff     = numerical(i, j) - exact_solution(x);
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

    class ScalarCrossSlotLinearBcCase : public CaseBase
    {
    public:
        ScalarCrossSlotLinearBcCase(int argc, char* argv[])
            : CaseBase(argc, argv)
        {}

        void read_paras() override
        {
            CaseBase::read_paras();

            IO::read_number(para_map, "n", n);
            IO::read_number(para_map, "num_steps", num_steps);
            IO::read_number(para_map, "dt", dt);
            IO::read_number(para_map, "diffusivity", diffusivity);
            IO::read_number(para_map, "tolerance", tolerance);

            if (n < 8)
                throw std::runtime_error("n must be >= 8");
            if (num_steps < 1)
                throw std::runtime_error("num_steps must be >= 1");
            if (dt <= 0.0)
                throw std::runtime_error("dt must be > 0");
            if (diffusivity < 0.0)
                throw std::runtime_error("diffusivity must be >= 0");
            if (tolerance <= 0.0)
                throw std::runtime_error("tolerance must be > 0");
        }

        bool record_paras() override
        {
            if (!CaseBase::record_paras())
                return false;

            paras_record.record("n", n)
                .record("num_steps", num_steps)
                .record("dt", dt)
                .record("diffusivity", diffusivity)
                .record("tolerance", tolerance)
                .record("scheme", "quick");
            return true;
        }

        int    n           = 64;
        int    num_steps   = 100;
        double dt          = 1e-3;
        double diffusivity = 1e-3;
        double tolerance   = 1e-10;
    };
} // namespace

int main(int argc, char* argv[])
{
    ScalarCrossSlotLinearBcCase case_param(argc, argv);
    case_param.read_paras();
    case_param.record_paras();

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showCurrentStep    = false;
    env_cfg.showGmresRes       = false;
    env_cfg.debugOutputDir     = case_param.root_dir;

    TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
    time_cfg.dt                   = case_param.dt;
    time_cfg.num_iterations       = case_param.num_steps;

    Geometry2D      geo;
    Domain2DUniform domain(case_param.n, case_param.n, 1.0, 1.0, "A1");
    geo.add_domain(&domain);
    geo.axis(&domain, LocationType::XNegative);
    geo.axis(&domain, LocationType::YNegative);

    Variable2D u("u"), v("v"), c("c");
    u.set_geometry(geo);
    v.set_geometry(geo);
    c.set_geometry(geo);

    field2 u_field("u_field"), v_field("v_field"), c_field("c_field");
    u.set_x_edge_field(&domain, u_field);
    v.set_y_edge_field(&domain, v_field);
    c.set_center_field(&domain, c_field);

    c.set_boundary_type(&domain, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    c.set_boundary_value(&domain, LocationType::XNegative, 0.0);
    c.set_boundary_type(&domain, LocationType::XPositive, PDEBoundaryType::Dirichlet);
    c.set_boundary_value(&domain, LocationType::XPositive, 1.0);
    c.set_boundary_type(&domain, LocationType::YNegative, PDEBoundaryType::Neumann);
    c.set_boundary_value(&domain, LocationType::YNegative, 0.0);
    c.set_boundary_type(&domain, LocationType::YPositive, PDEBoundaryType::Neumann);
    c.set_boundary_value(&domain, LocationType::YPositive, 0.0);

    u_field.clear(0.0);
    v_field.clear(0.0);
    fill_constant_buffers(u, 0.0);
    fill_constant_buffers(v, 0.0);

    const double hx = domain.get_hx();
    for (int i = 0; i < case_param.n; ++i)
    {
        for (int j = 0; j < case_param.n; ++j)
        {
            const double x = (static_cast<double>(i) + 0.5) * hx;
            c_field(i, j)  = exact_solution(x);
        }
    }

    ScalarSolver2D solver(&u, &v, &c, case_param.diffusivity, DifferenceSchemeType::Conv_QUICK_Diff_Center2nd);
    solver.variable_check();

    for (int step = 0; step < case_param.num_steps; ++step)
        solver.solve();

    const ErrorStats error = compute_error(c_field, hx);
    const bool       pass  = std::isfinite(error.linf) && error.linf <= case_param.tolerance;

    CSVHandler summary_file(case_param.root_dir + "/summary");
    summary_file.stream << "n,num_steps,dt,diffusivity,l1,l2,linf,tolerance,pass\n";
    summary_file.stream << case_param.n << ',' << case_param.num_steps << ',' << case_param.dt << ','
                        << case_param.diffusivity << ',' << error.l1 << ',' << error.l2 << ',' << error.linf << ','
                        << case_param.tolerance << ',' << (pass ? 1 : 0) << '\n';

    IO::write_csv(c_field, case_param.root_dir + "/scalar_final");

    std::cout << std::setprecision(16) << "scalar_cross_slot_linear_bc: L1=" << error.l1 << ", L2=" << error.l2
              << ", Linf=" << error.linf << ", tolerance=" << case_param.tolerance << ", pass=" << pass << std::endl;

    return pass ? 0 : 2;
}
