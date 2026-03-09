#pragma once

#include "io/case_base.hpp"

#include <cmath>
#include <string>

class HartmannFlowValidation2DCase : public CaseBase
{
public:
    static constexpr double POWERLAW_ETA_C = 0.00604;
    static constexpr double CARREAU_ETA_C  = 0.00596;
    static constexpr double CASSON_ETA_C   = 0.00514;
    static constexpr double POWERLAW_N     = 0.708;
    static constexpr double CARREAU_N      = 0.3568;
    static constexpr double CASSON_MU      = 0.00276;
    static constexpr double CASSON_TAU0    = 0.0108;

    HartmannFlowValidation2DCase(int argc, char* argv[])
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

        IO::read_number(para_map, "Re", Re);
        IO::read_number(para_map, "Ha", Ha);
        IO::read_number(para_map, "dp_dx", dp_dx);
        IO::read_number(para_map, "Bx", Bx);
        IO::read_number(para_map, "By", By);
        IO::read_number(para_map, "Bz", Bz);

        IO::read_number(para_map, "dt_factor", dt_factor);
        IO::read_number(para_map, "T_total", T_total);
        IO::read_number(para_map, "corr_iter", corr_iter);
        IO::read_number(para_map, "pv_output_step", pv_output_step);

        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        IO::read_number(para_map, "model_type", model_type);
        const bool has_n_index = IO::read_number(para_map, "n_index", n_index);
        if (!has_n_index)
        {
            n_index = (model_type == 2) ? CARREAU_N : ((model_type == 3) ? 1.0 : POWERLAW_N);
        }

        IO::read_number(para_map, "mu_0", mu_0);
        IO::read_number(para_map, "mu_inf", mu_inf);
        IO::read_number(para_map, "lambda", lambda);
        IO::read_number(para_map, "a", a);
        IO::read_number(para_map, "casson_mu", casson_mu);
        IO::read_number(para_map, "casson_tau0", casson_tau0);

        IO::read_number(para_map, "k_pl", k_pl);
        IO::read_number(para_map, "mu_min_pl", mu_min_pl);
        IO::read_number(para_map, "mu_max_pl", mu_max_pl);
        if (!IO::read_number(para_map, "mu_ref", mu_ref))
        {
            mu_ref = (model_type == 0) ?
                         1.0 :
                         ((model_type == 2) ? CARREAU_ETA_C : ((model_type == 3) ? CASSON_ETA_C : POWERLAW_ETA_C));
        }
        IO::read_bool(para_map, "use_dimensionless_viscosity", use_dimensionless_viscosity);

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

        const double lx = getLx();
        const double ly = getLy();
        const double hx = (nx > 0) ? (lx / static_cast<double>(nx)) : 0.0;
        const double hy = (ny > 0) ? (ly / static_cast<double>(ny)) : 0.0;

        paras_record.record("nx", nx)
            .record("ny", ny)
            .record("half_height", half_height)
            .record("length_over_h", length_over_h)
            .record("x_probe_over_h", x_probe_over_h)
            .record("x_window_half_width_over_h", x_window_half_width_over_h)
            .record("Lx", lx)
            .record("Ly", ly)
            .record("hx", hx)
            .record("hy", hy)
            .record("split_domain", split_domain ? 1 : 0)
            .record("topology", std::string(split_domain ? "split" : "single"))
            .record("Re", Re)
            .record("Ha", Ha)
            .record("dp_dx", dp_dx)
            .record("Bx", Bx)
            .record("By", By)
            .record("Bz", Bz)
            .record("dt_factor", dt_factor)
            .record("T_total", T_total)
            .record("corr_iter", corr_iter)
            .record("pv_output_step", pv_output_step)
            .record("gmres_m", gmres_m)
            .record("gmres_tol", gmres_tol)
            .record("gmres_max_iter", gmres_max_iter)
            .record("model_type", model_type)
            .record("n_index", n_index)
            .record("mu_0", mu_0)
            .record("mu_inf", mu_inf)
            .record("lambda", lambda)
            .record("a", a)
            .record("casson_mu", casson_mu)
            .record("casson_tau0", casson_tau0)
            .record("k_pl", k_pl)
            .record("mu_min_pl", mu_min_pl)
            .record("mu_max_pl", mu_max_pl)
            .record("mu_ref", mu_ref)
            .record("use_dimensionless_viscosity", use_dimensionless_viscosity ? 1 : 0)
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

    double getEffectiveHartmann() const { return std::abs(Ha) * std::abs(By); }

    bool hasHartmannAnalyticalReference() const
    {
        return model_type == 0 && std::abs(Bx) <= 1.0e-12 && std::abs(Bz) <= 1.0e-12;
    }

public:
    int    nx                         = 160;
    int    ny                         = 80;
    double half_height                = 1.0;
    double length_over_h              = 20.0;
    double x_probe_over_h             = -1.0;
    double x_window_half_width_over_h = 0.5;
    bool   split_domain               = false;

    double Re    = 20.0;
    double Ha    = 4.0;
    double dp_dx = -1.0;
    double Bx    = 0.0;
    double By    = 1.0;
    double Bz    = 0.0;

    double dt_factor      = 0.10;
    double T_total        = 80.0;
    int    corr_iter      = 2;
    int    pv_output_step = 0;

    int    gmres_m        = 30;
    double gmres_tol      = 1.0e-7;
    int    gmres_max_iter = 1000;

    int    model_type  = 0; // 0: Newtonian, 1: Power Law, 2: Carreau, 3: Casson
    double n_index     = POWERLAW_N;
    double mu_0        = 0.056;
    double mu_inf      = 0.00345;
    double lambda      = 3.313;
    double a           = 2.0;
    double casson_mu   = CASSON_MU;
    double casson_tau0 = CASSON_TAU0;

    double k_pl                        = 0.017;
    double mu_min_pl                   = 0.00345;
    double mu_max_pl                   = 0.056;
    double mu_ref                      = 1.0;
    bool   use_dimensionless_viscosity = true;

    double steady_u_tol                  = 1.0e-8;
    double steady_bulk_tol               = 1.0e-8;
    double steady_peak_tol               = 1.0e-8;
    double steady_v_peak_tol             = 1.0e-8;
    double steady_v_ratio_tol            = 1.0e-4;
    int    converged_hits                = 5;
    bool   require_steady_exit           = true;
    bool   use_analytical_initialization = true;
    bool   use_accuracy_dt_limit         = true;
    double dt_accuracy_factor            = 1.0;
};
