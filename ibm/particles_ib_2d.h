#pragma once

#include "particles_base.h"

#include <limits>

struct PIB2D : public ParticlesBase
{
    DECLARE_PROPERTY(Uf)
    DECLARE_PROPERTY(Vf)
    DECLARE_PROPERTY(Up)
    DECLARE_PROPERTY(Vp)

    DECLARE_PROPERTY(Fx)
    DECLARE_PROPERTY(Fy)
    DECLARE_PROPERTY(Fx_sum)
    DECLARE_PROPERTY(Fy_sum)

    PIB2D() {}

    PIB2D(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(Uf)
        INITIALIZE_PROPERTY(Vf)
        INITIALIZE_PROPERTY(Up)
        INITIALIZE_PROPERTY(Vp)

        INITIALIZE_PROPERTY(Fx)
        INITIALIZE_PROPERTY(Fy)
        INITIALIZE_PROPERTY(Fx_sum)
        INITIALIZE_PROPERTY(Fy_sum)
    }

    PIB2D(PIB2D&& rhs) noexcept { swap(*this, rhs); }

    PIB2D& operator=(PIB2D&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    void clear_force_sum();
};

#define EXPOSE_PIB2D(p)        \
    EXPOSE_PROPERTY(p, Uf)     \
    EXPOSE_PROPERTY(p, Vf)     \
    EXPOSE_PROPERTY(p, Up)     \
    EXPOSE_PROPERTY(p, Vp)     \
    EXPOSE_PROPERTY(p, Fx)     \
    EXPOSE_PROPERTY(p, Fy)     \
    EXPOSE_PROPERTY(p, Fx_sum) \
    EXPOSE_PROPERTY(p, Fy_sum)
