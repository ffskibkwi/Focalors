#include "particles_coordinate_map_2d.h"

#include "particles_spawner.h"

#include <cstring>

void PCoordMap2D::add_cylinder(int n, double r, double cx, double cy)
{
    PCoord2D* p_coord = new PCoord2D(n);
    collections.push_back(p_coord);

    EXPOSE_PCOORD2D(p_coord)

    // TODO: validate h
    spawn_cylinder(X, Y, n, r, cx, cy, h);
}

void PCoordMap2D::generate_map(Geometry2D* geo)
{
    auto domains = geo->domains;

    std::vector<std::vector<double>> X_list(domains.size());
    std::vector<std::vector<double>> Y_list(domains.size());

    for (PCoord2D* p_coord : collections)
    {
        EXPOSE_PCOORD2D(p_coord)

        for (int i = 0; i < p_coord->max_n; i++)
        {
            for (int domain_idx = 0; domain_idx < domains.size(); domain_idx++)
            {
                Domain2DUniform* domain = domains[domain_idx];

                if (domain->get_offset_x() <= X[i] && X[i] <= domain->get_offset_x() + domain->get_lx() &&
                    domain->get_offset_y() <= Y[i] && Y[i] <= domain->get_offset_y() + domain->get_ly())
                {
                    X_list[domain_idx].push_back(X[i]);
                    Y_list[domain_idx].push_back(Y[i]);
                    break;
                }
            }
        }
    }

    for (int domain_idx = 0; domain_idx < domains.size(); domain_idx++)
    {
        Domain2DUniform* domain = domains[domain_idx];

        coord_map[domain] = new PCoord2D(X_list[domain_idx].size());

        EXPOSE_PCOORD2D(coord_map[domain])

        std::memcpy(X, X_list[domain_idx].data(), X_list[domain_idx].size() * sizeof(double));
        std::memcpy(Y, Y_list[domain_idx].data(), Y_list[domain_idx].size() * sizeof(double));
    }
}