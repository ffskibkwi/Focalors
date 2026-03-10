#pragma once

#include "io/case_base.hpp"

/**
 * @brief 2D 平行板微通道 MHD 幂律流体案例参数。
 *
 * 物理量约定：
 * - 参考半高记为 h_ref（对应后处理中 y/h_ref）。
 * - 通道全高 Ly = 2 * h_ref。
 * - 通道长度 Lx = length_over_h * h_ref。
 */
class ParallelPlateMicrochannelMhd2DCase : public CaseBase
{
public:
    static constexpr double POWERLAW_ETA_C = 0.00604;
    static constexpr double POWERLAW_K      = 0.017;

    ParallelPlateMicrochannelMhd2DCase(int argc, char* argv[])
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

        IO::read_number(para_map, "Re", Re);
        IO::read_number(para_map, "U0", U0);

        IO::read_number(para_map, "Ha", Ha);
        IO::read_number(para_map, "B0", B0);
        IO::read_number(para_map, "sigma", sigma);
        IO::read_number(para_map, "Bx", Bx);
        IO::read_number(para_map, "By", By);
        IO::read_number(para_map, "Bz", Bz);

        IO::read_number(para_map, "dt_factor", dt_factor);
        IO::read_number(para_map, "T_total", T_total);
        IO::read_number(para_map, "pv_output_step", pv_output_step);

        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        IO::read_number(para_map, "model_type", model_type);
        IO::read_number(para_map, "n_index", n_index);
        IO::read_number(para_map, "k_pl", k_pl);
        IO::read_number(para_map, "mu_min_pl", mu_min_pl);
        IO::read_number(para_map, "mu_max_pl", mu_max_pl);
        IO::read_number(para_map, "mu_ref", mu_ref);
        IO::read_bool(para_map, "use_dimensionless_viscosity", use_dimensionless_viscosity);
        if (!IO::read_number(para_map, "gamma_ref", gamma_ref))
        {
            const double ref_length = (half_height > 0.0) ? half_height : 1.0;
            gamma_ref               = use_dimensionless_viscosity ? (U0 / ref_length) : 1.0;
        }

        IO::read_number(para_map, "dp_dx", dp_dx);

        IO::read_number(para_map, "convergence_tol", convergence_tol);
        IO::read_number(para_map, "converged_hits", converged_hits);
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
            .record("Lx", lx)
            .record("Ly", ly)
            .record("hx", hx)
            .record("hy", hy)
            .record("Re", Re)
            .record("U0", U0)
            .record("Ha", Ha)
            .record("B0", B0)
            .record("sigma", sigma)
            .record("Bx", Bx)
            .record("By", By)
            .record("Bz", Bz)
            .record("dt_factor", dt_factor)
            .record("T_total", T_total)
            .record("pv_output_step", pv_output_step)
            .record("gmres_m", gmres_m)
            .record("gmres_tol", gmres_tol)
            .record("gmres_max_iter", gmres_max_iter)
            .record("model_type", model_type)
            .record("n_index", n_index)
            .record("k_pl", k_pl)
            .record("mu_min_pl", mu_min_pl)
            .record("mu_max_pl", mu_max_pl)
            .record("mu_ref", mu_ref)
            .record("use_dimensionless_viscosity", use_dimensionless_viscosity ? 1 : 0)
            .record("gamma_ref", gamma_ref)
            .record("dp_dx", dp_dx)
            .record("convergence_tol", convergence_tol)
            .record("converged_hits", converged_hits);

        return true;
    }

    double getLx() const { return length_over_h * half_height; }
    double getLy() const { return 2.0 * half_height; }

public:
    int    nx             = 101;
    int    ny             = 51;
    double half_height    = 1.0;
    double length_over_h = 15.0;
    double x_probe_over_h = 7.5;

    double Re = 0.1;
    double U0 = 1.0;

    double Ha    = 6.0;
    double B0    = 1.0;
    double sigma = 0.0;
    double Bx    = 0.0;
    double By    = 0.0;
    double Bz    = 1.0;

    double dt_factor      = 0.05;
    double T_total        = 120.0;
    int    pv_output_step = 0;

    int    gmres_m        = 30;
    double gmres_tol      = 1.e-6;
    int    gmres_max_iter = 1000;

    int    model_type                 = 1;   // 0: Newtonian, 1: Power Law
    double n_index                    = 0.5; // 默认剪切变稀
    double k_pl                       = POWERLAW_K;
    double mu_min_pl                  = 0.00345;
    double mu_max_pl                  = 0.056;
    double mu_ref                     = POWERLAW_ETA_C;
    bool   use_dimensionless_viscosity = true;
    double gamma_ref                   = 1.0;

    double dp_dx = -1.0;

    double convergence_tol = 1.0e-6;
    int    converged_hits  = 5;
};
