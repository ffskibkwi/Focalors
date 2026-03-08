#pragma once

#include "base/pch.h"

#include "pe/concat/concat_solver2d.h"

#include <unordered_map>
#include <vector>

class PhysicalPESolver2D
{
public:
    PhysicalPESolver2D(Variable2D*            in_u_var,
                       Variable2D*            in_v_var,
                       Variable2D*            in_p_var,
                       ConcatPoissonSolver2D* in_p_solver,
                       double                 in_rho);
    ~PhysicalPESolver2D();

    void solve();

    void phys_boundary_update();
    void diag_shared_boundary_update();
    void calc_rhs();

    Variable2D *           u_var = nullptr, *v_var = nullptr, *p_var = nullptr;
    ConcatPoissonSolver2D* p_solver = nullptr;

    std::unordered_map<Domain2DUniform*, double> u_xpos_ypos_corner_map;
    std::unordered_map<Domain2DUniform*, double> v_xpos_ypos_corner_map;

    std::unordered_map<Domain2DUniform*, field2*> dudx_map, dudy_map;
    std::unordered_map<Domain2DUniform*, field2*> dvdx_map, dvdy_map;

    double rho;

private:
    std::vector<Domain2DUniform*>                                                            domains;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, Domain2DUniform*>> adjacency;
};
