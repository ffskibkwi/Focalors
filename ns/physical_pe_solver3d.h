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

    void calc_conv_inner();
    void calc_conv_outer();
    void nondiag_shared_boundary_update();
    void calc_rhs();

    Variable3D *                                  u_var = nullptr, *v_var = nullptr, *w_var = nullptr, *p_var = nullptr;
    ConcatPoissonSolver3D*                        p_solver = nullptr;
    std::unordered_map<Domain3DUniform*, field3*> c_u_map, c_v_map, c_w_map;
    std::unordered_map<Domain3DUniform*, std::unordered_map<LocationType, field2*>> c_u_buffer_map, c_v_buffer_map,
        c_w_buffer_map;

    // debug
    std::unordered_map<Domain3DUniform*, field3*> conv_u_x_map, conv_u_y_map, conv_u_z_map;
    std::unordered_map<Domain3DUniform*, field3*> conv_v_x_map, conv_v_y_map, conv_v_z_map;
    std::unordered_map<Domain3DUniform*, field3*> conv_w_x_map, conv_w_y_map, conv_w_z_map;

    double rho;
};