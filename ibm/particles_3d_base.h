#pragma once

#include "base/pch.h"

#include "particles_base.h"

#include <vector>

struct Particles3DBase : public ParticlesBase
{
    double* X  = nullptr; // position x of point
    double* Y  = nullptr; // position y of point
    double* Z  = nullptr; // position z of point
    double* Uf = nullptr; // velocity u of fluid at point position
    double* Vf = nullptr; // velocity v of fluid at point position
    double* Wf = nullptr; // velocity w of fluid at point position
    double* Up = nullptr; // velocity u of point
    double* Vp = nullptr; // velocity v of point
    double* Wp = nullptr; // velocity w of point

    Particles3DBase() {}

    Particles3DBase(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(properties, X, max_n);
        INITIALIZE_PROPERTY(properties, Y, max_n);
        INITIALIZE_PROPERTY(properties, Z, max_n);
        INITIALIZE_PROPERTY(properties, Uf, max_n);
        INITIALIZE_PROPERTY(properties, Vf, max_n);
        INITIALIZE_PROPERTY(properties, Wf, max_n);
        INITIALIZE_PROPERTY(properties, Up, max_n);
        INITIALIZE_PROPERTY(properties, Vp, max_n);
        INITIALIZE_PROPERTY(properties, Wp, max_n);
    }

    Particles3DBase(Particles3DBase&& rhs) noexcept { swap(*this, rhs); }

    Particles3DBase& operator=(Particles3DBase&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }
        return *this;
    }

    void translate(double x, double y, double z);

    void rotate_y(double cx, double cz, double angle);

    bool swap_two_particles(int lhs, int rhs);

    friend void swap(Particles3DBase& lhs, Particles3DBase& rhs);
};