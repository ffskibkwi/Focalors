#include "case/benchmark/2d/cross_shaped_channel_probe_interpolation.h"
#include <cmath>
#include <iostream>

int main()
{
    for (const int n : {80, 100, 125})
    {
        const double h = 1.0 / n;
        for (const double x : {0.5, 0.75})
        {
            const auto s     = CrossSlotProbe::cell_center_stencil(n, 15 * n, h, h, 0., 1., x, 1.5);
            const auto value = [h](int i, int j) { return 2 * ((i + .5) * h) - 3 * (1 + (j + .5) * h) + 1; };
            if (std::abs(CrossSlotProbe::interpolate(s, value) - (2 * x - 3 * 1.5 + 1)) > 1e-13)
                return 1;
        }
    }
    bool rejected = false;
    try
    {
        (void)CrossSlotProbe::cell_center_stencil(100, 1500, .01, .01, 0, 1, .5, .5);
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    if (!rejected)
        return 1;
    std::cout << "Fixed physical probe interpolation: PASS\n";
    return 0;
}
