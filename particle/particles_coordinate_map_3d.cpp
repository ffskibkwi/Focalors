#include "particles_coordinate_map_3d.h"

#include "particles_spawner.h"

#include <cstring>

void PCoordMap3D::add_sphere(double grid_h, double r, double cx, double cy, double cz)
{
    double n = M_PI / 3.0 * (12.0 * r * r / grid_h / grid_h + 1.0);

    PCoord3D* p_coord = new PCoord3D(n);
    collections.push_back(p_coord);

    EXPOSE_PCOORD3D(p_coord)

    // TODO: validate h
    spawn_sphere(X, Y, Z, n, r, cx, cy, cz, h);
}

void PCoordMap3D::generate_map(Geometry3D* geo)
{
    auto domains = geo->domains;

    std::vector<std::vector<double>> X_list(domains.size());
    std::vector<std::vector<double>> Y_list(domains.size());
    std::vector<std::vector<double>> Z_list(domains.size());

    for (PCoord3D* p_coord : collections)
    {
        EXPOSE_PCOORD3D(p_coord)

        for (int i = 0; i < p_coord->max_n; i++)
        {
            for (int domain_idx = 0; domain_idx < domains.size(); domain_idx++)
            {
                Domain3DUniform* domain = domains[domain_idx];

                if (domain->get_offset_x() <= X[i] && X[i] <= domain->get_offset_x() + domain->get_lx() &&
                    domain->get_offset_y() <= Y[i] && Y[i] <= domain->get_offset_y() + domain->get_ly() &&
                    domain->get_offset_z() <= Z[i] && Z[i] <= domain->get_offset_z() + domain->get_lz())
                {
                    X_list[domain_idx].push_back(X[i]);
                    Y_list[domain_idx].push_back(Y[i]);
                    Z_list[domain_idx].push_back(Z[i]);
                    break;
                }
            }
        }
    }

    for (int domain_idx = 0; domain_idx < domains.size(); domain_idx++)
    {
        Domain3DUniform* domain = domains[domain_idx];

        PCoord3D* p_coord = new PCoord3D(X_list[domain_idx].size());
        coord_map[domain] = p_coord;

        EXPOSE_PCOORD3D(coord_map[domain])

        std::memcpy(X, X_list[domain_idx].data(), X_list[domain_idx].size() * sizeof(double));
        std::memcpy(Y, Y_list[domain_idx].data(), Y_list[domain_idx].size() * sizeof(double));
        std::memcpy(Z, Z_list[domain_idx].data(), Z_list[domain_idx].size() * sizeof(double));

        // Calculate bounding box
        for (size_t i = 0; i < X_list[domain_idx].size(); i++)
        {
            if (X[i] < p_coord->min_X)
                p_coord->min_X = X[i];
            if (X[i] > p_coord->max_X)
                p_coord->max_X = X[i];
            if (Y[i] < p_coord->min_Y)
                p_coord->min_Y = Y[i];
            if (Y[i] > p_coord->max_Y)
                p_coord->max_Y = Y[i];
            if (Z[i] < p_coord->min_Z)
                p_coord->min_Z = Z[i];
            if (Z[i] > p_coord->max_Z)
                p_coord->max_Z = Z[i];
        }
    }
}