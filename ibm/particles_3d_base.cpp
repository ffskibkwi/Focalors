#include "particles_3d_base.h"

void Particles3DBase::translate(double x, double y, double z)
{
    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < cur_n; ++i)
    {
        X[i] = X[i] + x;
        Y[i] = Y[i] + y;
        Z[i] = Z[i] + z;
    }
}

void Particles3DBase::rotate_y(double cx, double cz, double angle)
{
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

bool Particles3DBase::swap_two_particles(int lhs, int rhs)
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

void swap(Particles3DBase& lhs, Particles3DBase& rhs)
{
    using std::swap;

    swap(static_cast<ParticlesBase&>(lhs), static_cast<ParticlesBase&>(rhs));

    swap(lhs.X, rhs.X);
    swap(lhs.Y, rhs.Y);
    swap(lhs.Z, rhs.Z);
    swap(lhs.Uf, rhs.Uf);
    swap(lhs.Vf, rhs.Vf);
    swap(lhs.Wf, rhs.Wf);
    swap(lhs.Up, rhs.Up);
    swap(lhs.Vp, rhs.Vp);
    swap(lhs.Wp, rhs.Wp);
}