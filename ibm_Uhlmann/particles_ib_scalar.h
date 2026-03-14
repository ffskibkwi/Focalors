#pragma once

#include "particle/particles_base.h"

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

/**
 * @brief Normal vectors for Neumann boundary condition
 * @details Only created when solver detects Neumann boundary type
 */
struct PIBNormal : public ParticlesBase
{
    DECLARE_PROPERTY(Nx)
    DECLARE_PROPERTY(Ny)
    DECLARE_PROPERTY(Nz)

    PIBNormal() {}

    PIBNormal(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(Nx)
        INITIALIZE_PROPERTY(Ny)
        INITIALIZE_PROPERTY(Nz)
    }

    PIBNormal(PIBNormal&& rhs) noexcept { swap(*this, rhs); }

    PIBNormal& operator=(PIBNormal&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }
};

#define EXPOSE_PIBSCALAR(p) \
    EXPOSE_PROPERTY(p, Sf)  \
    EXPOSE_PROPERTY(p, Sp)  \
    EXPOSE_PROPERTY(p, Fs)

#define EXPOSE_PIBNORMAL(p) \
    EXPOSE_PROPERTY(p, Nx) \
    EXPOSE_PROPERTY(p, Ny) \
    EXPOSE_PROPERTY(p, Nz)
