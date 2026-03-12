#include "particles_spawner.h"

#include <cmath>

void spawn_cylinder(double* X, double* Y, int n, double r, double cx, double cy, double& h)
{
    h = 2 * M_PI * r / n;

    double dtheta = 2.0 * M_PI / n;
    for (int i = 0; i < n; i++)
    {
        X[i] = cx + r * std::cos(dtheta * i);
        Y[i] = cy + r * std::sin(dtheta * i);
    }
}

void spawn_sphere(double* X, double* Y, double* Z, int n, double r, double cx, double cy, double cz, double& h)
{
    double h_acc = 0.0;
    for (int i = 0; i < n; i++)
    {
        double phi   = std::acos(-1.0 + (2.0 * (i + 1) - 1.0) / n);
        double theta = std::sqrt(n * M_PI) * phi;

        X[i] = cx + r * std::cos(theta) * std::sin(phi);
        Y[i] = cy + r * std::sin(theta) * std::sin(phi);
        Z[i] = cz + r * std::cos(phi);

        if (i > 0)
        {
            h_acc += std::sqrt(std::pow(X[i - 1] - X[i], 2.0) + std::pow(Y[i - 1] - Y[i], 2.0) +
                               std::pow(Z[i - 1] - Z[i], 2.0));
        }
    }

    h = h_acc / (n - 1);
}