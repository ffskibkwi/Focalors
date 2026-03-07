#include "particles_base.h"

void swap(ParticlesBase& lhs, ParticlesBase& rhs)
{
    using std::swap;

    swap(lhs.max_n, rhs.max_n);
    swap(lhs.cur_n, rhs.cur_n);
    swap(lhs.properties, rhs.properties);
}