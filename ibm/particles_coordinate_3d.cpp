#include "particles_coordinate_3d.h"

void PCoord3D::translate(double x, double y, double z)
{
    EXPOSE_PCOORD3D(this)

    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < cur_n; ++i)
    {
        X[i] = X[i] + x;
        Y[i] = Y[i] + y;
        Z[i] = Z[i] + z;
    }
}

void PCoord3D::rotate_y(double cx, double cz, double angle)
{
    EXPOSE_PCOORD3D(this)

    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < cur_n; ++i)
    {
        double x1 = X[i] - cx;
        double z1 = Z[i] - cz;

        double x2 = std::cos(angle) * x1 - std::sin(angle) * z1;
        double z2 = std::sin(angle) * x1 + std::cos(angle) * z1;

        X[i] = x2 + cx;
        Z[i] = z2 + cz;
    }
}

void PCoord3D::refresh_bounding_box(Domain3DUniform* domain, double offset_x, double offset_y, double offset_z)
{
    EXPOSE_PCOORD3D(this)

    double hx = domain->get_hx();
    double hy = domain->get_hy();
    double hz = domain->get_hz();
    double nx = domain->get_nx();
    double ny = domain->get_ny();
    double nz = domain->get_nz();

    if (cur_n == 0)
    {
        min_X = 0.0;
        max_X = 0.0;
        min_Y = 0.0;
        max_Y = 0.0;
        min_Z = 0.0;
        max_Z = 0.0;

        // double position -> context index -> velocity u index

        min_ix_u = 0;
        max_ix_u = 0;
        min_iy_u = 0;
        max_iy_u = 0;
        min_iz_u = 0;
        max_iz_u = 0;

        // double position -> context index -> velocity v index

        min_ix_v = 0;
        max_ix_v = 0;
        min_iy_v = 0;
        max_iy_v = 0;
        min_iz_v = 0;
        max_iz_v = 0;

        // double position -> context index -> velocity w index

        min_ix_w = 0;
        max_ix_w = 0;
        min_iy_w = 0;
        max_iy_w = 0;
        min_iz_w = 0;
        max_iz_w = 0;

        return;
    }

    min_X = std::numeric_limits<double>::max();
    max_X = std::numeric_limits<double>::min();
    min_Y = std::numeric_limits<double>::max();
    max_Y = std::numeric_limits<double>::min();
    min_Z = std::numeric_limits<double>::max();
    max_Z = std::numeric_limits<double>::min();

    for (int i = 0; i < cur_n; ++i)
    {
        min_X = std::min(offset_x + X[i], min_X);
        max_X = std::max(offset_x + X[i], max_X);
        min_Y = std::min(offset_y + Y[i], min_Y);
        max_Y = std::max(offset_y + Y[i], max_Y);
        min_Z = std::min(offset_z + Z[i], min_Z);
        max_Z = std::max(offset_z + Z[i], max_Z);
    }

    // 2h range

    // double position -> context index -> velocity u index

    min_ix_u = std::floor(min_X / hx) - 1;
    max_ix_u = std::floor(max_X / hx) + 2;
    min_iy_u = std::floor(min_Y / hy) - 2;
    max_iy_u = std::floor(max_Y / hy) + 2;
    min_iz_u = std::floor(min_Z / hz) - 2;
    max_iz_u = std::floor(max_Z / hz) + 2;

    // double position -> context index -> velocity v index

    min_ix_v = std::floor(min_X / hx) - 2;
    max_ix_v = std::floor(max_X / hx) + 2;
    min_iy_v = std::floor(min_Y / hy) - 1;
    max_iy_v = std::floor(max_Y / hy) + 2;
    min_iz_v = std::floor(min_Z / hz) - 2;
    max_iz_v = std::floor(max_Z / hz) + 2;

    // double position -> context index -> velocity w index

    min_ix_w = std::floor(min_X / hx) - 2;
    max_ix_w = std::floor(max_X / hx) + 2;
    min_iy_w = std::floor(min_Y / hy) - 2;
    max_iy_w = std::floor(max_Y / hy) + 2;
    min_iz_w = std::floor(min_Z / hz) - 1;
    max_iz_w = std::floor(max_Z / hz) + 2;

    // clamp to inner field

    min_ix_u = std::clamp(min_ix_u, 0, nx);
    max_ix_u = std::clamp(max_ix_u, 0, nx);
    min_iy_u = std::clamp(min_iy_u, 0, ny - 1);
    max_iy_u = std::clamp(max_iy_u, 0, ny - 1);
    min_iz_u = std::clamp(min_iz_u, 0, nz - 1);
    max_iz_u = std::clamp(max_iz_u, 0, nz - 1);

    min_ix_v = std::clamp(min_ix_v, 0, nx - 1);
    max_ix_v = std::clamp(max_ix_v, 0, nx - 1);
    min_iy_v = std::clamp(min_iy_v, 0, ny);
    max_iy_v = std::clamp(max_iy_v, 0, ny);
    min_iz_v = std::clamp(min_iz_v, 0, nz - 1);
    max_iz_v = std::clamp(max_iz_v, 0, nz - 1);

    min_ix_w = std::clamp(min_ix_w, 0, nx - 1);
    max_ix_w = std::clamp(max_ix_w, 0, nx - 1);
    min_iy_w = std::clamp(min_iy_w, 0, ny - 1);
    max_iy_w = std::clamp(max_iy_w, 0, ny - 1);
    min_iz_w = std::clamp(min_iz_w, 0, nz);
    max_iz_w = std::clamp(max_iz_w, 0, nz);
}

void swap(PCoord3D& lhs, PCoord3D& rhs)
{
    using std::swap;

    swap(static_cast<ParticlesBase&>(lhs), static_cast<ParticlesBase&>(rhs));

    swap(lhs.min_X, rhs.min_X);
    swap(lhs.max_X, rhs.max_X);
    swap(lhs.min_Y, rhs.min_Y);
    swap(lhs.max_Y, rhs.max_Y);
    swap(lhs.min_Z, rhs.min_Z);
    swap(lhs.max_Z, rhs.max_Z);
    swap(lhs.min_ix_u, rhs.min_ix_u);
    swap(lhs.max_ix_u, rhs.max_ix_u);
    swap(lhs.min_iy_u, rhs.min_iy_u);
    swap(lhs.max_iy_u, rhs.max_iy_u);
    swap(lhs.min_iz_u, rhs.min_iz_u);
    swap(lhs.max_iz_u, rhs.max_iz_u);
    swap(lhs.min_ix_v, rhs.min_ix_v);
    swap(lhs.max_ix_v, rhs.max_ix_v);
    swap(lhs.min_iy_v, rhs.min_iy_v);
    swap(lhs.max_iy_v, rhs.max_iy_v);
    swap(lhs.min_iz_v, rhs.min_iz_v);
    swap(lhs.max_iz_v, rhs.max_iz_v);
    swap(lhs.min_ix_w, rhs.min_ix_w);
    swap(lhs.max_ix_w, rhs.max_ix_w);
    swap(lhs.min_iy_w, rhs.min_iy_w);
    swap(lhs.max_iy_w, rhs.max_iy_w);
    swap(lhs.min_iz_w, rhs.min_iz_w);
    swap(lhs.max_iz_w, rhs.max_iz_w);
}