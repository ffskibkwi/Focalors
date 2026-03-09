#pragma once

#include "particles_base.h"

#include <limits>

struct PIBScalar : public ParticlesBase
{
    DECLARE_PROPERTY(Sf)
    DECLARE_PROPERTY(Sp)
    DECLARE_PROPERTY(Fs)

    PIBScalar() {}

    PIBScalar(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(Sf)
        INITIALIZE_PROPERTY(Sp)
        INITIALIZE_PROPERTY(Fs)
    }

    PIBScalar(PIBScalar&& rhs) noexcept { swap(*this, rhs); }

    PIBScalar& operator=(PIBScalar&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    void clear_force_sum();
};

#define EXPOSE_PIBSCALAR(p) \
    EXPOSE_PROPERTY(p, Sf)  \
    EXPOSE_PROPERTY(p, Sp)  \
    EXPOSE_PROPERTY(p, Fs)
