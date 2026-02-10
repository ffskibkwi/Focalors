#pragma once

#include "poisson_base/io/case_base.hpp"

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
        IO::read_number(para_map, "n_index", n_index);
        IO::read_number(para_map, "mu_0", mu_0);
        IO::read_number(para_map, "mu_inf", mu_inf);
        IO::read_number(para_map, "lambda", lambda);
        IO::read_number(para_map, "a", a);

        // Dimensionless Parameters
        IO::read_number(para_map, "Re_PL", Re_PL);
        IO::read_number(para_map, "mu_min_pl", mu_min_pl);
        IO::read_number(para_map, "mu_max_pl", mu_max_pl);
        IO::read_number(para_map, "Re_0", Re_0);
        IO::read_number(para_map, "Re_inf", Re_inf);
        IO::read_number(para_map, "Wi", Wi);

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
            .record("Re_PL", Re_PL)
            .record("mu_min_pl", mu_min_pl)
            .record("mu_max_pl", mu_max_pl)
            .record("Re_0", Re_0)
            .record("Re_inf", Re_inf)
            .record("Wi", Wi)
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
    int    model_type = 1;   // 0: Newtonian, 1: Power Law (default), 2: Carreau
    double n_index    = 1.0; // Power-law index
    double mu_0       = 1.0; // Zero-shear viscosity (Carreau)
    double mu_inf     = 0.0; // Infinite-shear viscosity (Carreau)
    double lambda     = 0.0; // Relaxation time (Carreau)
    double a          = 2.0; // Carreau model parameter

    // Dimensionless Parameters
    double Re_PL     = 100.0;
    double mu_min_pl = -1.0; // Minimum viscosity limit for Power Law (-1.0 means use default)
    double mu_max_pl = -1.0; // Maximum viscosity limit for Power Law (-1.0 means use default)
    double Re_0      = 10.0;
    double Re_inf = 1000.0;
    double Wi     = 1.0;

    // Analytical Solution Parameters
    double dp_dx =
        -12.0; // Default pressure gradient for Newtonian channel flow with U_max=1.5, mu=1, H=0.5 -> dp/dx =
               // -2*mu*U_max/H^2 * ? Actually for Poiseuille u(y) = 1/(2mu) (-dp/dx) (H^2 - y^2). If U_avg=1, H=0.5.
               // U_max = 1.5. 1.5 = 1/(2*1) * (-dp/dx) * 0.25 => 1.5 = (-dp/dx) * 0.125 => -dp/dx = 12.
};
