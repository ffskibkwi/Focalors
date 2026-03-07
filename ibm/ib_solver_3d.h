#pragma once

#include "ib_kernel.hpp"
#include "ib_particles_3d.h"

#include <algorithm>

/**
 * @brief Immersed boundary method solver.
 *
 * We associate a discrete volume ΔV with each force point such that the union of
 * all these volumes forms a thin shell (of thickness equal to one mesh width) around each particle.
 * It means that ImmersedBoundary point volume
 * ΔV = solid.h * solid.h * context.h in 3D
 * ΔV = solid.h * context.h in 2D
 */
class ImmersedBoundarySolver3D
{
public:
    ImmersedBoundarySolver3D() {}

    void solve(FluidContext3D&              context,
               field3&                      u_recv,
               field3&                      v_recv,
               field3&                      w_recv,
               ImmersedBoundaryParticles3D& ib_particles);

    void u2F(FluidContext3D& context, ImmersedBoundaryParticles3D& particles);

    void apply_ib_force(FluidContext3D&              context,
                        field3&                      u_recv,
                        field3&                      v_recv,
                        field3&                      w_recv,
                        ImmersedBoundaryParticles3D& particles,
                        double                       ib_hx,
                        double                       ib_hy,
                        double                       ib_hz);
};