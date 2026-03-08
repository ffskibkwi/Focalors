
#pragma once

#include "base/pch.h"

#include "particles_base.h"

struct PCoord2D : public ParticlesBase
{
    DECLARE_PROPERTY(X)
    DECLARE_PROPERTY(Y)

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

    PCoord2D() {}

    PCoord2D(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(X)
        INITIALIZE_PROPERTY(Y)
    }

    PCoord2D(PCoord2D&& rhs) noexcept { swap(*this, rhs); }

    PCoord2D& operator=(PCoord2D&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    void translate(double x, double y);

    void rotate(double cx, double cy, double angle);

    void refresh_bounding_box(Domain2DUniform* domain, double offset_x = 0.0, double offset_y = 0.0);

    friend void swap(PCoord2D& lhs, PCoord2D& rhs);
};

#define EXPOSE_PCOORD2D(p) \
    EXPOSE_PROPERTY(p, X)  \
    EXPOSE_PROPERTY(p, Y)

#define EXPOSE_PCOORD2D_BOUND(p) \
    int min_ix_u = p.min_ix_u;   \
    int max_ix_u = p.max_ix_u;   \
    int min_iy_u = p.min_iy_u;   \
    int max_iy_u = p.max_iy_u;   \
    int min_ix_v = p.min_ix_v;   \
    int max_ix_v = p.max_ix_v;   \
    int min_iy_v = p.min_iy_v;   \
    int max_iy_v = p.max_iy_v;
