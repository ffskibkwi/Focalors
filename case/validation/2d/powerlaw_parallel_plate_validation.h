#pragma once

#include "io/case_base.hpp"

#include <algorithm>
#include <cmath>
#include <string>

class PowerLawParallelPlateValidation2DCase : public CaseBase
{
public:
    static constexpr double kShearRateFloor = 1.0e-4;

    PowerLawParallelPlateValidation2DCase(int argc, char* argv[])
        : CaseBase(argc, argv)
    {}

    void read_paras() override
    {
        CaseBase::read_paras();

        IO::read_number(para_map, "nx", nx);
        IO::read_number(para_map, "ny", ny);
        IO::read_number(para_map, "half_height", half_height);
        IO::read_number(para_map, "length_over_h", length_over_h);
        IO::read_number(para_map, "x_probe_over_h", x_probe_over_h);
        IO::read_number(para_map, "x_window_half_width_over_h", x_window_half_width_over_h);
        IO::read_bool(para_map, "split_domain", split_domain);

        IO::read_number(para_map, "n_index", n_index);
        IO::read_number(para_map, "k_pl", k_pl);
        IO::read_number(para_map, "dp_dx", dp_dx);
        IO::read_number(para_map, "mu_min", mu_min);
        IO::read_number(para_map, "mu_max", mu_max);

        IO::read_number(para_map, "dt_factor", dt_factor);
        IO::read_number(para_map, "steady_max_steps", steady_max_steps);
        IO::read_number(para_map, "corr_iter", corr_iter);
        IO::read_number(para_map, "pv_output_step", pv_output_step);

        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        double legacy_convergence_tol = steady_u_tol;
        if (IO::read_number(para_map, "convergence_tol", legacy_convergence_tol))
        {
            steady_u_tol    = legacy_convergence_tol;
            steady_bulk_tol = legacy_convergence_tol;
            steady_peak_tol = legacy_convergence_tol;
            steady_v_peak_tol = legacy_convergence_tol;
        }

        IO::read_number(para_map, "steady_u_tol", steady_u_tol);
        IO::read_number(para_map, "steady_bulk_tol", steady_bulk_tol);
        IO::read_number(para_map, "steady_peak_tol", steady_peak_tol);
        IO::read_number(para_map, "steady_v_peak_tol", steady_v_peak_tol);
        IO::read_number(para_map, "steady_v_ratio_tol", steady_v_ratio_tol);
        IO::read_number(para_map, "converged_hits", converged_hits);
        IO::read_bool(para_map, "require_steady_exit", require_steady_exit);
        IO::read_bool(para_map, "use_analytical_initialization", use_analytical_initialization);
        IO::read_bool(para_map, "use_accuracy_dt_limit", use_accuracy_dt_limit);
        IO::read_number(para_map, "dt_accuracy_factor", dt_accuracy_factor);
    }

    bool record_paras() override
    {
        if (!CaseBase::record_paras())
            return false;

        paras_record.record("nx", nx)
            .record("ny", ny)
            .record("half_height", half_height)
            .record("length_over_h", length_over_h)
            .record("Lx", getLx())
            .record("Ly", getLy())
            .record("hx", getHx())
            .record("hy", getHy())
            .record("x_probe_over_h", x_probe_over_h)
            .record("x_window_half_width_over_h", x_window_half_width_over_h)
            .record("split_domain", split_domain ? 1 : 0)
            .record("topology", std::string(split_domain ? "split" : "single"))
            .record("n_index", n_index)
            .record("k_pl", k_pl)
            .record("dp_dx", dp_dx)
            .record("pressure_force", getPressureForce())
            .record("mu_min_input", mu_min)
            .record("mu_max_input", mu_max)
            .record("mu_min_for_config", getMuMinForConfig())
            .record("mu_max_for_config", getMuMaxForConfig())
            .record("wall_shear_rate", getWallShearRateMagnitude())
            .record("dt_factor", dt_factor)
            .record("steady_max_steps", steady_max_steps)
            .record("corr_iter", corr_iter)
            .record("pv_output_step", pv_output_step)
            .record("gmres_m", gmres_m)
            .record("gmres_tol", gmres_tol)
            .record("gmres_max_iter", gmres_max_iter)
            .record("convergence_tol", steady_u_tol)
            .record("steady_u_tol", steady_u_tol)
            .record("steady_bulk_tol", steady_bulk_tol)
            .record("steady_peak_tol", steady_peak_tol)
            .record("steady_v_peak_tol", steady_v_peak_tol)
            .record("steady_v_ratio_tol", steady_v_ratio_tol)
            .record("converged_hits", converged_hits)
            .record("require_steady_exit", require_steady_exit ? 1 : 0)
            .record("use_analytical_initialization", use_analytical_initialization ? 1 : 0)
            .record("use_accuracy_dt_limit", use_accuracy_dt_limit ? 1 : 0)
            .record("dt_accuracy_factor", dt_accuracy_factor);

        return true;
    }

    double getLx() const { return length_over_h * half_height; }

    double getLy() const { return 2.0 * half_height; }

    double getHx() const { return getLx() / std::max(nx, 1); }

    double getHy() const { return getLy() / std::max(ny, 1); }

    double getPressureForce() const { return -dp_dx; }

    bool hasAnalyticalReference() const
    {
        return nx > 0 && ny > 0 && half_height > 0.0 && length_over_h > 0.0 && n_index > 0.0 && k_pl > 0.0 &&
               getPressureForce() > 0.0;
    }

    double getWallShearRateMagnitude() const
    {
        if (!hasAnalyticalReference())
            return 0.0;

        return std::pow(getPressureForce() * half_height / k_pl, 1.0 / n_index);
    }

    double evaluatePowerLawViscosity(double shear_rate) const
    {
        const double shear_safe = std::max(shear_rate, kShearRateFloor);
        return k_pl * std::pow(shear_safe, n_index - 1.0);
    }

    double getMuMinForConfig() const
    {
        if (mu_min > 0.0)
            return mu_min;

        const double mu_floor = evaluatePowerLawViscosity(kShearRateFloor);
        const double mu_wall  = evaluatePowerLawViscosity(getWallShearRateMagnitude());
        return std::min(mu_floor, mu_wall);
    }

    double getMuMaxForConfig() const
    {
        if (mu_max > 0.0)
            return mu_max;

        const double mu_floor = evaluatePowerLawViscosity(kShearRateFloor);
        const double mu_wall  = evaluatePowerLawViscosity(getWallShearRateMagnitude());
        return std::max(mu_floor, mu_wall);
    }

public:
    int    nx                         = 192;
    int    ny                         = 64;
    double half_height                = 1.0;
    double length_over_h              = 12.0;
    double x_probe_over_h             = -1.0;
    double x_window_half_width_over_h = 0.5;
    bool   split_domain               = false;

    double n_index = 0.7;
    double k_pl    = 1.0;
    double dp_dx   = -1.0;
    double mu_min  = -1.0;
    double mu_max  = -1.0;

    double dt_factor        = 0.25;
    int    steady_max_steps = 12000;
    int    corr_iter        = 2;
    int    pv_output_step   = 0;

    int    gmres_m        = 30;
    double gmres_tol      = 1.0e-7;
    int    gmres_max_iter = 1000;

    double steady_u_tol                  = 1.0e-7;
    double steady_bulk_tol               = 1.0e-7;
    double steady_peak_tol               = 1.0e-7;
    double steady_v_peak_tol             = 1.0e-7;
    double steady_v_ratio_tol            = 5.0e-3;
    int    converged_hits                = 10;
    bool   require_steady_exit           = true;
    bool   use_analytical_initialization = true;
    bool   use_accuracy_dt_limit         = true;
    double dt_accuracy_factor            = 2.0;
};
