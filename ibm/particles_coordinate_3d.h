#pragma once

#include "base/pch.h"

#include "particles_base.h"

#include <vector>

struct PCoord3D : public ParticlesBase
{
    DECLARE_PROPERTY(X)
    DECLARE_PROPERTY(Y)
    DECLARE_PROPERTY(Z)

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

    PCoord3D() {}

    PCoord3D(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(X)
        INITIALIZE_PROPERTY(Y)
        INITIALIZE_PROPERTY(Z)
    }

    PCoord3D(PCoord3D&& rhs) noexcept { swap(*this, rhs); }

    PCoord3D& operator=(PCoord3D&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }
        return *this;
    }

    void translate(double x, double y, double z);

    void rotate_y(double cx, double cz, double angle);

    void
    refresh_bounding_box(Domain3DUniform* domain, double offset_x = 0.0, double offset_y = 0.0, double offset_z = 0.0);

    friend void swap(PCoord3D& lhs, PCoord3D& rhs);
};

#define EXPOSE_PCOORD3D(p) \
    EXPOSE_PROPERTY(p, X)  \
    EXPOSE_PROPERTY(p, Y)  \
    EXPOSE_PROPERTY(p, Z)

#define EXPOSE_PCOORD3D_BOUND(p) \
    int min_ix_u = p.min_ix_u;   \
    int max_ix_u = p.max_ix_u;   \
    int min_iy_u = p.min_iy_u;   \
    int max_iy_u = p.max_iy_u;   \
    int min_iz_u = p.min_iz_u;   \
    int max_iz_u = p.max_iz_u;   \
    int min_ix_v = p.min_ix_v;   \
    int max_ix_v = p.max_ix_v;   \
    int min_iy_v = p.min_iy_v;   \
    int max_iy_v = p.max_iy_v;   \
    int min_iz_v = p.min_iz_v;   \
    int max_iz_v = p.max_iz_v;   \
    int min_ix_w = p.min_ix_w;   \
    int max_ix_w = p.max_ix_w;   \
    int min_iy_w = p.min_iy_w;   \
    int max_iy_w = p.max_iy_w;   \
    int min_iz_w = p.min_iz_w;   \
    int max_iz_w = p.max_iz_w;
