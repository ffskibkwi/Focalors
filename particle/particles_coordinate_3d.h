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
