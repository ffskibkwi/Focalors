#include "ib_particles_2d.h"

void ImmersedBoundaryParticles2D::refresh_bounding_box(Domain2DUniform* domain, double offset_x, double offset_y)
{
    double hx = domain->get_hx();
    double hy = domain->get_hy();
    double nx = domain->get_nx();
    double ny = domain->get_ny();

    if (cur_n == 0)
    {
        min_X = 0.0;
        max_X = 0.0;
        min_Y = 0.0;
        max_Y = 0.0;

        // double position -> context index -> velocity u index

        min_ix_u = 0;
        max_ix_u = 0;
        min_iy_u = 0;
        max_iy_u = 0;

        // double position -> context index -> velocity v index

        min_ix_v = 0;
        max_ix_v = 0;
        min_iy_v = 0;
        max_iy_v = 0;
    }

    min_X = std::numeric_limits<double>::max();
    max_X = std::numeric_limits<double>::min();
    min_Y = std::numeric_limits<double>::max();
    max_Y = std::numeric_limits<double>::min();

    for (int i = 0; i < cur_n; ++i)
    {
        min_X = std::min(offset_x + X[i], min_X);
        max_X = std::max(offset_x + X[i], max_X);
        min_Y = std::min(offset_y + Y[i], min_Y);
        max_Y = std::max(offset_y + Y[i], max_Y);
    }

    // 2h range

    // double position -> context index -> velocity u index

    min_ix_u = std::floor(min_X / hx) - 1;
    max_ix_u = std::floor(max_X / hx) + 2;
    min_iy_u = std::floor(min_Y / hy) - 2;
    max_iy_u = std::floor(max_Y / hy) + 2;

    // double position -> context index -> velocity v index

    min_ix_v = std::floor(min_X / hx) - 2;
    max_ix_v = std::floor(max_X / hx) + 2;
    min_iy_v = std::floor(min_Y / hy) - 1;
    max_iy_v = std::floor(max_Y / hy) + 2;

    // clamp to inner field

    min_ix_u = std::clamp(min_ix_u, 0, nx);
    max_ix_u = std::clamp(max_ix_u, 0, nx);
    min_iy_u = std::clamp(min_iy_u, 0, ny - 1);
    max_iy_u = std::clamp(max_iy_u, 0, ny - 1);

    min_ix_v = std::clamp(min_ix_v, 0, nx - 1);
    max_ix_v = std::clamp(max_ix_v, 0, nx - 1);
    min_iy_v = std::clamp(min_iy_v, 0, ny);
    max_iy_v = std::clamp(max_iy_v, 0, ny);
}

void ImmersedBoundaryParticles2D::clear_force_sum()
{
    for (int i = 0; i < cur_n; i++)
    {
        Fx_sum[i] = 0.0;
        Fy_sum[i] = 0.0;
    }
}

void swap(ImmersedBoundaryParticles2D& lhs, ImmersedBoundaryParticles2D& rhs)
{
    using std::swap;

    swap(static_cast<Particles2DBase&>(lhs), static_cast<Particles2DBase&>(rhs));

    swap(lhs.Fx, rhs.Fx);
    swap(lhs.Fy, rhs.Fy);
    swap(lhs.Fx_sum, rhs.Fx_sum);
    swap(lhs.Fy_sum, rhs.Fy_sum);
    swap(lhs.min_X, rhs.min_X);
    swap(lhs.max_X, rhs.max_X);
    swap(lhs.min_Y, rhs.min_Y);
    swap(lhs.max_Y, rhs.max_Y);
    swap(lhs.min_ix_u, rhs.min_ix_u);
    swap(lhs.max_ix_u, rhs.max_ix_u);
    swap(lhs.min_iy_u, rhs.min_iy_u);
    swap(lhs.max_iy_u, rhs.max_iy_u);
    swap(lhs.min_ix_v, rhs.min_ix_v);
    swap(lhs.max_ix_v, rhs.max_ix_v);
    swap(lhs.min_iy_v, rhs.min_iy_v);
    swap(lhs.max_iy_v, rhs.max_iy_v);
}