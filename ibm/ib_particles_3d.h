#pragma once

#include "particles_3d_base.h"

#include <limits>

struct ImmersedBoundaryParticles3D : public Particles3DBase
{
    double* Fx     = nullptr; // ib force x at point position
    double* Fy     = nullptr; // ib force y at point position
    double* Fz     = nullptr; // ib force z at point position
    double* Fx_sum = nullptr; // ib force x sum at point position
    double* Fy_sum = nullptr; // ib force y sum at point position
    double* Fz_sum = nullptr; // ib force z sum at point position

    double min_X = std::numeric_limits<double>::max();
    double max_X = std::numeric_limits<double>::min();
    double min_Y = std::numeric_limits<double>::max();
    double max_Y = std::numeric_limits<double>::min();
    double min_Z = std::numeric_limits<double>::max();
    double max_Z = std::numeric_limits<double>::min();

    // double position -> context index -> velocity u index

    int min_ix_u = std::numeric_limits<int>::max();
    int max_ix_u = std::numeric_limits<int>::min();
    int min_iy_u = std::numeric_limits<int>::max();
    int max_iy_u = std::numeric_limits<int>::min();
    int min_iz_u = std::numeric_limits<int>::max();
    int max_iz_u = std::numeric_limits<int>::min();

    // double position -> context index -> velocity v index

    int min_ix_v = std::numeric_limits<int>::max();
    int max_ix_v = std::numeric_limits<int>::min();
    int min_iy_v = std::numeric_limits<int>::max();
    int max_iy_v = std::numeric_limits<int>::min();
    int min_iz_v = std::numeric_limits<int>::max();
    int max_iz_v = std::numeric_limits<int>::min();

    // double position -> context index -> velocity w index

    int min_ix_w = std::numeric_limits<int>::max();
    int max_ix_w = std::numeric_limits<int>::min();
    int min_iy_w = std::numeric_limits<int>::max();
    int max_iy_w = std::numeric_limits<int>::min();
    int min_iz_w = std::numeric_limits<int>::max();
    int max_iz_w = std::numeric_limits<int>::min();

    ImmersedBoundaryParticles3D() {}

    ImmersedBoundaryParticles3D(int max_n)
        : Particles3DBase(max_n)
    {
        INITIALIZE_PROPERTY(properties, Fx, max_n);
        INITIALIZE_PROPERTY(properties, Fy, max_n);
        INITIALIZE_PROPERTY(properties, Fz, max_n);
        INITIALIZE_PROPERTY(properties, Fx_sum, max_n);
        INITIALIZE_PROPERTY(properties, Fy_sum, max_n);
        INITIALIZE_PROPERTY(properties, Fz_sum, max_n);
    }

    ImmersedBoundaryParticles3D(ImmersedBoundaryParticles3D&& rhs) noexcept { swap(*this, rhs); }

    ImmersedBoundaryParticles3D& operator=(ImmersedBoundaryParticles3D&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    void
    refresh_bounding_box(Domain3DUniform* domain, double offset_x = 0.0, double offset_y = 0.0, double offset_z = 0.0);

    void clear_force_sum();

    friend void swap(ImmersedBoundaryParticles3D& lhs, ImmersedBoundaryParticles3D& rhs);
};

#define EXPOSE_POINTS3D_BOUND(solid) \
    int min_ix_u = solid.min_ix_u;   \
    int max_ix_u = solid.max_ix_u;   \
    int min_iy_u = solid.min_iy_u;   \
    int max_iy_u = solid.max_iy_u;   \
    int min_iz_u = solid.min_iz_u;   \
    int max_iz_u = solid.max_iz_u;   \
    int min_ix_v = solid.min_ix_v;   \
    int max_ix_v = solid.max_ix_v;   \
    int min_iy_v = solid.min_iy_v;   \
    int max_iy_v = solid.max_iy_v;   \
    int min_iz_v = solid.min_iz_v;   \
    int max_iz_v = solid.max_iz_v;   \
    int min_ix_w = solid.min_ix_w;   \
    int max_ix_w = solid.max_ix_w;   \
    int min_iy_w = solid.min_iy_w;   \
    int max_iy_w = solid.max_iy_w;   \
    int min_iz_w = solid.min_iz_w;   \
    int max_iz_w = solid.max_iz_w;
