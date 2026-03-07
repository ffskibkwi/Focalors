#include "particles_2d_base.h"

void Particles2DBase::translate(double x, double y)
{
    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < cur_n; ++i)
    {
        X[i] = X[i] + x;
        Y[i] = Y[i] + y;
    }
}

void Particles2DBase::rotate(double cx, double cy, double angle)
{
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

bool Particles2DBase::swap_two_particles(int lhs, int rhs)
{
    if (lhs == rhs)
    {
        return false;
    }

    if (lhs < 0)
    {
        std::cerr << "Swap two particles but lhs < 0!" << std::endl;
        return false;
    }

    if (rhs < 0)
    {
        std::cerr << "Swap two particles but rhs < 0!" << std::endl;
        return false;
    }

    if (lhs >= max_n)
    {
        std::cerr << "Swap two particles but lhs >= max_n!" << std::endl;
        return false;
    }

    if (rhs >= max_n)
    {
        std::cerr << "Swap two particles but rhs >= max_n!" << std::endl;
        return false;
    }

    for (int i = 0; i < properties.size(); ++i)
    {
        if (properties[i] == nullptr)
        {
            std::cerr << "Swap two particles but properties[" << i << "] is nullptr!" << std::endl;
            return false;
        }

        std::swap(properties[i][lhs], properties[i][rhs]);
    }

    return true;
}

void swap(Particles2DBase& lhs, Particles2DBase& rhs)
{
    using std::swap;

    swap(static_cast<ParticlesBase&>(lhs), static_cast<ParticlesBase&>(rhs));

    swap(lhs.X, rhs.X);
    swap(lhs.Y, rhs.Y);
    swap(lhs.Uf, rhs.Uf);
    swap(lhs.Vf, rhs.Vf);
    swap(lhs.Up, rhs.Up);
    swap(lhs.Vp, rhs.Vp);
}