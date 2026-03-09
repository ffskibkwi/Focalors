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

    if (cur_n == 0)
    {
        min_X = 0.0;
        max_X = 0.0;
        min_Y = 0.0;
        max_Y = 0.0;
        min_Z = 0.0;
        max_Z = 0.0;
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
}