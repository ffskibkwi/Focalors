#pragma once

#include "base/pch.h"

#include "particles_base.h"

struct Particles2DBase : public ParticlesBase
{
    double* X  = nullptr; // position x of point
    double* Y  = nullptr; // position y of point
    double* Uf = nullptr; // velocity u of fluid at point position
    double* Vf = nullptr; // velocity v of fluid at point position
    double* Up = nullptr; // velocity u of point
    double* Vp = nullptr; // velocity v of point

    Particles2DBase() {}

    Particles2DBase(int max_n)
    {
        this->max_n = max_n;
        cur_n       = max_n;

        INITIALIZE_PROPERTY(properties, X, max_n);
        INITIALIZE_PROPERTY(properties, Y, max_n);
        INITIALIZE_PROPERTY(properties, Uf, max_n);
        INITIALIZE_PROPERTY(properties, Vf, max_n);
        INITIALIZE_PROPERTY(properties, Up, max_n);
        INITIALIZE_PROPERTY(properties, Vp, max_n);
    }

    Particles2DBase(Particles2DBase&& rhs) noexcept { swap(*this, rhs); }

    Particles2DBase& operator=(Particles2DBase&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    void translate(double x, double y);

    void rotate(double cx, double cy, double angle);

    bool swap_two_particles(int lhs, int rhs);

    friend void swap(Particles2DBase& lhs, Particles2DBase& rhs);
};
