#pragma once

#include "ib_kernel.hpp"
#include "ib_particles_2d.h"

/**
 * @brief Immersed boundary method solver.
 *
 * We associate a discrete volume ΔV with each force point such that the union of
 * all these volumes forms a thin shell (of thickness equal to one mesh width) around each particle.
 * It means that ImmersedBoundary point volume
 * ΔV = solid.h * solid.h * context.h in 3D
 * ΔV = solid.h * context.h in 2D
 */
class ImmersedBoundarySolver2D
{
public:
    ImmersedBoundarySolver2D() {}

    void solve(FluidContext2D& context, field2& u_recv, field2& v_recv, ImmersedBoundaryParticles2D& ib_particles);

    void u2F(FluidContext2D& context, ImmersedBoundaryParticles2D& particles);

    void apply_ib_force(FluidContext2D&              context,
                        field2&                      u_recv,
                        field2&                      v_recv,
                        ImmersedBoundaryParticles2D& particles,
                        double                       ib_hx,
                        double                       ib_hy);
};