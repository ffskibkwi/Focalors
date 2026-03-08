#pragma once

#include "ib_kernel.hpp"
#include "particles_coordinate_3d.h"
#include "particles_ib_3d.h"

#include <algorithm>

/**
 * @brief Immersed boundary method solver.
 *
 * Grid space dx = dy = dz should be ensured.
 *
 * We associate a discrete volume ΔV with each force point such that the union of
 * all these volumes forms a thin shell (of thickness equal to one mesh width) around each particle.
 * It means that ImmersedBoundary point volume
 * ΔV = ib_h * ib_h * grid_h in 3D
 * ΔV = ib_h * grid_h in 3D
 */
class ImmersedBoundarySolver3D
{
public:
    ImmersedBoundarySolver3D(Variable3D*                                      _u_var,
                             Variable3D*                                      _v_var,
                             Variable3D*                                      _w_var,
                             std::unordered_map<Domain3DUniform*, PCoord3D*>& _coord_map);

    void solve();

    void u3F();

    void apply_ib_force();

private:
    Variable3D* u_var;
    Variable3D* v_var;
    Variable3D* w_var;

    std::unordered_map<Domain3DUniform*, PCoord3D*> coord_map;
    std::unordered_map<Domain3DUniform*, PIB3D*>    ib_map;

    double ib_h;
    double grid_h;
};