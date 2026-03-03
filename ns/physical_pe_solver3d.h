#pragma once

#include "base/pch.h"

#include "pe/concat/concat_solver3d.h"

#include <unordered_map>
#include <vector>

class PhysicalPESolver3D
{
public:
    PhysicalPESolver3D(Variable3D*            in_u_var,
                       Variable3D*            in_v_var,
                       Variable3D*            in_w_var,
                       Variable3D*            in_p_var,
                       ConcatPoissonSolver3D* in_p_solver,
                       double                 in_rho);
    ~PhysicalPESolver3D();

    void solve();

    void diag_shared_boundary_update();
    void calc_rhs();

    Variable3D *           u_var = nullptr, *v_var = nullptr, *w_var = nullptr, *p_var = nullptr;
    ConcatPoissonSolver3D* p_solver = nullptr;

    std::unordered_map<Domain3DUniform*, double*> u_xpos_ypos_corner_map, u_xpos_zpos_corner_map,
        v_xpos_ypos_corner_map, v_ypos_zpos_corner_map, w_xpos_zpos_corner_map, w_ypos_zpos_corner_map;

    // debug
    std::unordered_map<Domain3DUniform*, field3*> dudx_map, dudy_map, dudz_map;
    std::unordered_map<Domain3DUniform*, field3*> dvdx_map, dvdy_map, dvdz_map;
    std::unordered_map<Domain3DUniform*, field3*> dwdx_map, dwdy_map, dwdz_map;

    double rho;
};