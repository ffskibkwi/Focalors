#include "particles_coordinate_map_3d.h"

#include "particles_spawner.h"

void PCoordMap3D::add_sphere(int n, double r, double cx, double cy, double cz)
{
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

        coord_map[domain] = new PCoord3D(X_list[domain_idx].size());

        EXPOSE_PCOORD3D(coord_map[domain])

        std::memcpy(X, X_list[domain_idx].data(), X_list[domain_idx].size() * sizeof(double));
        std::memcpy(Y, Y_list[domain_idx].data(), Y_list[domain_idx].size() * sizeof(double));
        std::memcpy(Z, Z_list[domain_idx].data(), Z_list[domain_idx].size() * sizeof(double));
    }
}