#pragma once

#include "base/pch.h"

#include <cmath>

// Peskin, Charles S. "The immersed boundary method." Acta numerica 11 (2002): 479-517.
//
// https://doi.org/10.1017/S0962492902000077

inline double ib_phi(double r, double h)
{

    if (std::abs(r) <= h)
    {
        return 1.0 / 8.0 *
               (3.0 - 2.0 * std::abs(r) / h +
                std::sqrt(1.0 + 4.0 * std::abs(r) / h - 4.0 * (std::abs(r) / h) * (std::abs(r) / h)));
    }
    else if (std::abs(r) <= 2.0 * h)
    {
        return 1.0 / 8.0 *
               (5.0 - 2.0 * std::abs(r) / h -
                std::sqrt(-7.0 + 12.0 * std::abs(r) / h - 4.0 * (std::abs(r) / h) * (std::abs(r) / h)));
    }
    else
    {
        return 0;
    }

    // return std::abs(r) <= 2 ? (1.0 / 4.0 * (1.0 + std::cos(pi * r / 2))) : 0.0;
}

inline double ib_delta(double x, double y, double hx, double hy)
{
    return 1.0 / hx / hy * ib_phi(x, hx) * ib_phi(y, hy);
}

inline double ib_delta(double x, double y, double z, double hx, double hy, double hz)
{
    return 1.0 / hx / hy / hz * ib_phi(x, hx) * ib_phi(y, hy) * ib_phi(z, hz);
}

inline double ib_phi_short(double r) { return std::abs(r) <= 2 ? (1.0 / 4.0 * (1.0 + std::cos(pi * r / 2))) : 0.0; }

inline double ib_delta_short(double x, double y, double hx, double hy)
{
    return 1.0 / hx / hy * ib_phi_short(x / hx) * ib_phi_short(y / hy);
}

inline double ib_delta_short(double x, double y, double z, double hx, double hy, double hz)
{
    return 1.0 / hx / hy / hz * ib_phi_short(x / hx) * ib_phi_short(y / hy) * ib_phi_short(z / hz);
}
