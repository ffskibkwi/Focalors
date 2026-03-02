#pragma once

#include "io/case_base.hpp"

/**
 * @brief Case class for Cross-Shaped Channel 2D simulation.
 *
 * Manages parameters specific to the cross-shaped channel geometry and flow conditions.
 * Supports both Newtonian and Non-Newtonian fluid parameters.
 */
class CrossShapedChannel2DCase : public CaseBase
{
public:
    static constexpr double POWERLAW_ETA_C = 0.00604;
    static constexpr double CARREAU_ETA_C  = 0.00596;
    static constexpr double POWERLAW_N      = 0.708;
    static constexpr double CARREAU_N       = 0.3568;

    CrossShapedChannel2DCase(int argc, char* argv[])
        : CaseBase(argc, argv)
    {}

    void read_paras() override
    {
        CaseBase::read_paras();

        // Grid and Geometry Parameters
        IO::read_number(para_map, "h", h);
        IO::read_number(para_map, "lx_2", lx_2);
        IO::read_number(para_map, "ly_2", ly_2);
        IO::read_number(para_map, "lx_1", lx_1);
        IO::read_number(para_map, "lx_3", lx_3);
        IO::read_number(para_map, "ly_4", ly_4);
        IO::read_number(para_map, "ly_5", ly_5);
        std::cout << "1" << std::endl;

        // Physics Parameters
        IO::read_number(para_map, "Re", Re);
        IO::read_number(para_map, "U0", U0);

        // MHD Parameters
        IO::read_number(para_map, "Ha", Ha);
        IO::read_number(para_map, "Bx", Bx);
        IO::read_number(para_map, "By", By);

        // Time Stepping
        IO::read_number(para_map, "dt_factor", dt_factor);
        IO::read_number(para_map, "T_total", T_total);
        IO::read_number(para_map, "pv_output_step", pv_output_step);

        // Solver Parameters
        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        // Non-Newtonian Parameters (Optional, with defaults)
        IO::read_number(para_map, "model_type", model_type);
        const bool has_n_index = IO::read_number(para_map, "n_index", n_index);
        if (!has_n_index)
        {
            n_index = (model_type == 2) ? CARREAU_N : POWERLAW_N;
        }
        IO::read_number(para_map, "mu_0", mu_0);
        IO::read_number(para_map, "mu_inf", mu_inf);
        IO::read_number(para_map, "lambda", lambda);
        IO::read_number(para_map, "a", a);

        // Non-Newtonian parameters actually used by config setters
        IO::read_number(para_map, "k_pl", k_pl);
        IO::read_number(para_map, "mu_min_pl", mu_min_pl);
        IO::read_number(para_map, "mu_max_pl", mu_max_pl);
        if (!IO::read_number(para_map, "mu_ref", mu_ref))
        {
            mu_ref = (model_type == 2) ? CARREAU_ETA_C : POWERLAW_ETA_C;
        }
        IO::read_bool(para_map, "use_dimensionless_viscosity", use_dimensionless_viscosity);
    }

    bool record_paras() override
    {
        if (!CaseBase::record_paras())
            return false;

        paras_record.record("h", h)
            .record("lx_2", lx_2)
            .record("ly_2", ly_2)
            .record("lx_1", lx_1)
            .record("lx_3", lx_3)
            .record("ly_4", ly_4)
            .record("ly_5", ly_5)
            .record("Re", Re)
            .record("U0", U0)
            .record("Ha", Ha)
            .record("Bx", Bx)
            .record("By", By)
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
            .record("k_pl", k_pl)
            .record("mu_min_pl", mu_min_pl)
            .record("mu_max_pl", mu_max_pl)
            .record("mu_ref", mu_ref)
            .record("use_dimensionless_viscosity", use_dimensionless_viscosity ? 1 : 0);

        return true;
    }

public:
    // Grid Spacing
    double h = 0.01;

    // Geometry Dimensions
    double lx_2 = 1.0;
    double ly_2 = 1.0;
    double lx_1 = 15.0;
    double lx_3 = 15.0;
    double ly_4 = 15.0;
    double ly_5 = 15.0;

    // Physics
    double Re = 100.0;
    double U0 = 1.0;

    // MHD Parameters
    double Ha = 10.0; // Hartmann number
    double Bx = 0.0;  // Magnetic field x-component
    double By = 1.0;  // Magnetic field y-component

    // Time Stepping
    double dt_factor      = 0.1;   // dt = dt_factor * h
    double T_total        = 105.0; // 70 * 1.5
    int    pv_output_step = 0;     // 循环输出时刻间隔步数 (0 表示使用 num_iterations/10)

    // Solver Settings
    int    gmres_m        = 30;
    double gmres_tol      = 1.e-6;
    int    gmres_max_iter = 1000;

    // Non-Newtonian Model Parameters
    int    model_type = 1;       // 0: Newtonian, 1: Power Law (paper default), 2: Carreau
    double n_index    = POWERLAW_N; // Power-law index (paper)
    double mu_0       = 0.056;   // Zero-shear viscosity (Bird/Carreau)
    double mu_inf     = 0.00345; // Infinite-shear viscosity (Bird/Carreau)
    double lambda     = 3.313;   // Relaxation time (paper Bird-Carreau)
    double a          = 2.0;     // Carreau model parameter

    // Non-Newtonian parameters actually used by config setters
    double k_pl      = 0.017;   // Power-law consistency index (paper)
    double mu_min_pl = 0.00345; // Power-Law viscosity lower limit
    double mu_max_pl = 0.056;   // Power-Law viscosity upper limit
    double mu_ref    = POWERLAW_ETA_C;
    bool   use_dimensionless_viscosity = true;
};
