#pragma once

#include "particles_2d_base.h"

#include <limits>

struct ImmersedBoundaryParticles2D : public Particles2DBase
{
    double* Fx     = nullptr; // ib force x at point position
    double* Fy     = nullptr; // ib force y at point position
    double* Fx_sum = nullptr; // ib force x sum at point position
    double* Fy_sum = nullptr; // ib force y sum at point position

    double min_X = std::numeric_limits<double>::max();
    double max_X = std::numeric_limits<double>::min();
    double min_Y = std::numeric_limits<double>::max();
    double max_Y = std::numeric_limits<double>::min();

    // double position -> context index -> velocity u index

    int min_ix_u = std::numeric_limits<int>::max();
    int max_ix_u = std::numeric_limits<int>::min();
    int min_iy_u = std::numeric_limits<int>::max();
    int max_iy_u = std::numeric_limits<int>::min();

    // double position -> context index -> velocity v index

    int min_ix_v = std::numeric_limits<int>::max();
    int max_ix_v = std::numeric_limits<int>::min();
    int min_iy_v = std::numeric_limits<int>::max();
    int max_iy_v = std::numeric_limits<int>::min();

    ImmersedBoundaryParticles2D() {}

    ImmersedBoundaryParticles2D(int max_n)
        : Particles2DBase(max_n)
    {
        INITIALIZE_PROPERTY(properties, Fx, max_n);
        INITIALIZE_PROPERTY(properties, Fy, max_n);
        INITIALIZE_PROPERTY(properties, Fx_sum, max_n);
        INITIALIZE_PROPERTY(properties, Fy_sum, max_n);
    }

    ImmersedBoundaryParticles2D(ImmersedBoundaryParticles2D&& rhs) noexcept { swap(*this, rhs); }

    ImmersedBoundaryParticles2D& operator=(ImmersedBoundaryParticles2D&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    void refresh_bounding_box(Domain2DUniform* domain, double offset_x = 0.0, double offset_y = 0.0);

    void clear_force_sum();

    friend void swap(ImmersedBoundaryParticles2D& lhs, ImmersedBoundaryParticles2D& rhs);
};

#define EXPOSE_POINTS2D_BOUND(solid) \
    int min_ix_u = solid.min_ix_u;   \
    int max_ix_u = solid.max_ix_u;   \
    int min_iy_u = solid.min_iy_u;   \
    int max_iy_u = solid.max_iy_u;   \
    int min_ix_v = solid.min_ix_v;   \
    int max_ix_v = solid.max_ix_v;   \
    int min_iy_v = solid.min_iy_v;   \
    int max_iy_v = solid.max_iy_v;
