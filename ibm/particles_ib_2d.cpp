#include "particles_ib_2d.h"

void PIB2D::clear_force_sum()
{
    EXPOSE_PIB2D(this)

    for (int i = 0; i < cur_n; i++)
    {
        Fx_sum[i] = 0.0;
        Fy_sum[i] = 0.0;
    }
}