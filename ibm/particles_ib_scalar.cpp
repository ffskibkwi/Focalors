#include "particles_ib_scalar.h"

void PIBScalar::clear_force_sum()
{
    EXPOSE_PIBSCALAR(this)

    for (int i = 0; i < cur_n; i++)
    {
        Fs_sum[i] = 0.0;
    }
}
