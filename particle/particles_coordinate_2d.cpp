#include "particles_coordinate_2d.h"

void PCoord2D::translate(double x, double y)
{
    EXPOSE_PCOORD2D(this)

    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < cur_n; ++i)
    {
        X[i] = X[i] + x;
        Y[i] = Y[i] + y;
    }
}

void PCoord2D::rotate(double cx, double cy, double angle)
{
    EXPOSE_PCOORD2D(this)

    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < cur_n; ++i)
    {
        double x1 = X[i] - cx;
        double y1 = Y[i] - cy;

        double x2 = std::cos(angle) * x1 - std::sin(angle) * y1;
        double y2 = std::sin(angle) * x1 + std::cos(angle) * y1;

        X[i] = x2 + cx;
        Y[i] = y2 + cy;
    }
}

void PCoord2D::refresh_bounding_box(Domain2DUniform* domain, double offset_x, double offset_y)
{
    EXPOSE_PCOORD2D(this)

    double hx = domain->get_hx();
    double hy = domain->get_hy();

    if (cur_n == 0)
    {
        min_X = 0.0;
        max_X = 0.0;
        min_Y = 0.0;
        max_Y = 0.0;
        return;
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

}

void swap(PCoord2D& lhs, PCoord2D& rhs)
{
    using std::swap;

    swap(static_cast<ParticlesBase&>(lhs), static_cast<ParticlesBase&>(rhs));

    swap(lhs.min_X, rhs.min_X);
    swap(lhs.max_X, rhs.max_X);
    swap(lhs.min_Y, rhs.min_Y);
    swap(lhs.max_Y, rhs.max_Y);
}