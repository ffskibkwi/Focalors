#include "particles_ib_3d.h"

void PIB3D::clear_force_sum()
{
    EXPOSE_PIB3D(this)

    for (int i = 0; i < cur_n; i++)
    {
        Fx_sum[i] = 0.0;
        Fy_sum[i] = 0.0;
        Fz_sum[i] = 0.0;
    }
}