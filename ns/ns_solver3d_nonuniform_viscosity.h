#pragma once

#include "ns_solver3d.h"

// SELVAM B, MERK S, GOVINDARAJAN R, MEIBURG E. Stability of miscible core–annular flows with viscosity stratification.
// Journal of Fluid Mechanics. 2007;592:23-49. doi:10.1017/S0022112007008269
class NSSolver3DNonUniVisc : public ConcatNSSolver3D
{
    NSSolver3DNonUniVisc(Variable3D*            in_u_var,
                         Variable3D*            in_v_var,
                         Variable3D*            in_w_var,
                         Variable3D*            in_p_var,
                         ConcatPoissonSolver3D* in_p_solver,
                         Variable3D*            in_c_var,
                         double                 mu1,
                         double                 mu2)
        : ConcatNSSolver3D(in_u_var, in_v_var, in_w_var, in_p_var, in_p_solver)
        , c_var(in_c_var)
    {
        if (mu1 < 0 || mu2 < 0)
            std::cerr << "mu must not be negative!" << std::endl;

        if (mu1 < mu2)
            std::swap(mu1, mu2);
        ln_mu1_mu2 = std::log(mu1 / mu2);
    }

    void euler_conv_diff_inner();
    void euler_conv_diff_outer();

private:
    Variable3D* c_var      = nullptr;
    double      ln_mu1_mu2 = 0.0;
};