#pragma once

#include "particles_base.h"

#include <limits>

struct PIB3D : public ParticlesBase
{
    DECLARE_PROPERTY(Uf)
    DECLARE_PROPERTY(Vf)
    DECLARE_PROPERTY(Wf)
    DECLARE_PROPERTY(Up)
    DECLARE_PROPERTY(Vp)
    DECLARE_PROPERTY(Wp)

    DECLARE_PROPERTY(Fx)
    DECLARE_PROPERTY(Fy)
    DECLARE_PROPERTY(Fz)
    DECLARE_PROPERTY(Fx_sum)
    DECLARE_PROPERTY(Fy_sum)
    DECLARE_PROPERTY(Fz_sum)

    PIB3D() {}

    PIB3D(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(Uf)
        INITIALIZE_PROPERTY(Vf)
        INITIALIZE_PROPERTY(Wf)
        INITIALIZE_PROPERTY(Up)
        INITIALIZE_PROPERTY(Vp)
        INITIALIZE_PROPERTY(Wp)

        INITIALIZE_PROPERTY(Fx)
        INITIALIZE_PROPERTY(Fy)
        INITIALIZE_PROPERTY(Fz)
        INITIALIZE_PROPERTY(Fx_sum)
        INITIALIZE_PROPERTY(Fy_sum)
        INITIALIZE_PROPERTY(Fz_sum)
    }

    PIB3D(PIB3D&& rhs) noexcept { swap(*this, rhs); }

    PIB3D& operator=(PIB3D&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    void clear_force_sum();
};

#define EXPOSE_PIB3D(p)        \
    EXPOSE_PROPERTY(p, Uf)     \
    EXPOSE_PROPERTY(p, Vf)     \
    EXPOSE_PROPERTY(p, Wf)     \
    EXPOSE_PROPERTY(p, Up)     \
    EXPOSE_PROPERTY(p, Vp)     \
    EXPOSE_PROPERTY(p, Wp)     \
    EXPOSE_PROPERTY(p, Fx)     \
    EXPOSE_PROPERTY(p, Fy)     \
    EXPOSE_PROPERTY(p, Fz)     \
    EXPOSE_PROPERTY(p, Fx_sum) \
    EXPOSE_PROPERTY(p, Fy_sum) \
    EXPOSE_PROPERTY(p, Fz_sum)
