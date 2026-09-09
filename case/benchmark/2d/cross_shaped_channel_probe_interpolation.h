#pragma once

#include <cmath>
#include <stdexcept>

namespace CrossSlotProbe
{
    struct Stencil
    {
        int    i, j;
        double wx, wy;
    };

    inline Stencil cell_center_stencil(int nx, int ny, double hx, double hy, double ox, double oy, double x, double y)
    {
        if (nx < 2 || ny < 2 || hx <= 0 || hy <= 0 || !std::isfinite(x) || !std::isfinite(y))
            throw std::runtime_error("Invalid fixed-probe grid or coordinate");
        const double xi  = (x - ox) / hx - 0.5;
        const double eta = (y - oy) / hy - 0.5;
        if (xi < 0 || xi > nx - 1 || eta < 0 || eta > ny - 1)
            throw std::runtime_error("Fixed probe must be inside the cell-centre interpolation domain");
        int i = static_cast<int>(std::floor(xi));
        int j = static_cast<int>(std::floor(eta));
        if (i == nx - 1)
            --i;
        if (j == ny - 1)
            --j;
        return {i, j, xi - i, eta - j};
    }

    template<class CellSampler>
    double interpolate(const Stencil& s, CellSampler value)
    {
        return (1 - s.wy) * ((1 - s.wx) * value(s.i, s.j) + s.wx * value(s.i + 1, s.j)) +
               s.wy * ((1 - s.wx) * value(s.i, s.j + 1) + s.wx * value(s.i + 1, s.j + 1));
    }
} // namespace CrossSlotProbe
