#pragma once

#include "io/case_base.hpp"

/**
 * @brief Case class for Parallel Plate Channel 2D simulation (Non-Newtonian).
 *
 * Manages parameters specific to the parallel plate channel geometry and flow conditions.
 * Supports both Newtonian and Non-Newtonian fluid parameters.
 * Provides analytical solution parameters for verification.
 */
class ParallelPlateChannel2DCase : public CaseBase
{
public:
    static constexpr double POWERLAW_ETA_C = 0.00604;
    static constexpr double CARREAU_ETA_C  = 0.00596;
    static constexpr double CASSON_ETA_C   = 0.00514;
    static constexpr double POWERLAW_N      = 0.708;
    static constexpr double CARREAU_N       = 0.3568;
    static constexpr double CASSON_MU       = 0.00276;
    static constexpr double CASSON_TAU0     = 0.0108;

    ParallelPlateChannel2DCase(int argc, char* argv[])
        : CaseBase(argc, argv)
    {}

    void read_paras() override
    {
        CaseBase::read_paras();

        // Grid and Geometry Parameters
        IO::read_number(para_map, "h", h);
        IO::read_number(para_map, "Lx", Lx);
        IO::read_number(para_map, "Ly", Ly);

        // Physics Parameters
        IO::read_number(para_map, "Re", Re);
        IO::read_number(para_map, "U0", U0);

        // Time Stepping
        IO::read_number(para_map, "dt_factor", dt_factor);
        IO::read_number(para_map, "T_total", T_total);
        IO::read_number(para_map, "pv_output_step", pv_output_step);

        // Solver Parameters
        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        // Non-Newtonian Parameters
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

        // Non-Newtonian parameters actually used by config setters
        IO::read_number(para_map, "k_pl", k_pl);
        IO::read_number(para_map, "mu_min_pl", mu_min_pl);
        IO::read_number(para_map, "mu_max_pl", mu_max_pl);
        if (!IO::read_number(para_map, "mu_ref", mu_ref))
        {
            mu_ref = (model_type == 2) ? CARREAU_ETA_C : ((model_type == 3) ? CASSON_ETA_C : POWERLAW_ETA_C);
        }
        IO::read_bool(para_map, "use_dimensionless_viscosity", use_dimensionless_viscosity);

        // Analytical Solution Parameters
        IO::read_number(para_map, "dp_dx", dp_dx);
    }

    bool record_paras() override
    {
        if (!CaseBase::record_paras())
            return false;

        paras_record.record("h", h)
            .record("Lx", Lx)
            .record("Ly", Ly)
            .record("Re", Re)
            .record("U0", U0)
            .record("dt_factor", dt_factor)
            .record("T_total", T_total)
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
            .record("dp_dx", dp_dx);

        return true;
    }

public:
    // Grid Spacing
    double h = 0.02;

    // Geometry Dimensions
    double Lx = 2.0;
    double Ly = 1.0;

    // Physics
    double Re = 100.0;
    double U0 = 1.0;

    // Time Stepping
    double dt_factor      = 0.1; // dt = dt_factor * h
    double T_total        = 1000.0;
    int    pv_output_step = 0; // 0 means use default (num_iterations/10)

    // Solver Settings
    int    gmres_m        = 10;
    double gmres_tol      = 1.e-7;
    int    gmres_max_iter = 1000;

    // Non-Newtonian Model Parameters
    int    model_type = 1;   // 0: Newtonian, 1: Power Law (default), 2: Carreau, 3: Casson
    double n_index    = POWERLAW_N; // Power-law index (paper)
    double mu_0       = 0.056;   // Zero-shear viscosity (Carreau, paper)
    double mu_inf     = 0.00345; // Infinite-shear viscosity (Carreau, paper)
    double lambda     = 3.313;   // Relaxation time (Carreau, paper)
    double a          = 2.0; // Carreau model parameter
    double casson_mu   = CASSON_MU;
    double casson_tau0 = CASSON_TAU0;

    // Non-Newtonian parameters actually used by config setters
    double k_pl      = 0.017;   // Power-law consistency index (paper)
    double mu_min_pl = 0.00345; // Power-law viscosity lower limit (paper)
    double mu_max_pl = 0.056;   // Power-law viscosity upper limit (paper)
    double mu_ref    = POWERLAW_ETA_C;
    bool   use_dimensionless_viscosity = true;

    // Analytical Solution Parameters
    double dp_dx =
        -12.0; // Default pressure gradient for Newtonian channel flow with U_max=1.5, mu=1, H=0.5 -> dp/dx =
               // -2*mu*U_max/H^2 * ? Actually for Poiseuille u(y) = 1/(2mu) (-dp/dx) (H^2 - y^2). If U_avg=1, H=0.5.
               // U_max = 1.5. 1.5 = 1/(2*1) * (-dp/dx) * 0.25 => 1.5 = (-dp/dx) * 0.125 => -dp/dx = 12.
};
