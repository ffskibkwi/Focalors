#pragma once

#include "ib_kernel.hpp"
#include "particles_coordinate_2d.h"
#include "particles_ib_2d.h"

/**
 * @brief Immersed boundary method solver.
 *
 * Grid space dx = dy should be ensured.
 *
 * We associate a discrete volume ΔV with each force point such that the union of
 * all these volumes forms a thin shell (of thickness equal to one mesh width) around each particle.
 * It means that ImmersedBoundary point volume
 * ΔV = ib_h * ib_h * grid_h in 3D
 * ΔV = ib_h * grid_h in 2D
 */
class ImmersedBoundarySolver2D
{
public:
    ImmersedBoundarySolver2D(Variable2D*                                      _u_var,
                             Variable2D*                                      _v_var,
                             std::unordered_map<Domain2DUniform*, PCoord2D*>& _coord_map);

    void solve();

    void u2F();

    void apply_ib_force();

private:
    Variable2D* u_var;
    Variable2D* v_var;

    std::unordered_map<Domain2DUniform*, PCoord2D*> coord_map;
    std::unordered_map<Domain2DUniform*, PIB2D*>    ib_map;

    double ib_h;
    double grid_h;
};