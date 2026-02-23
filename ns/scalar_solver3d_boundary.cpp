#include "scalar_solver3d.h"

#include "boundary_2d_utils.h"
#include "boundary_3d_utils.h"

void ScalarSolver3D::phys_boundary_update()
{
    for (auto& domain : domains)
    {
        field3& s = *s_field_map[domain];

        int nx = s.get_nx();
        int ny = s.get_ny();
        int nz = s.get_nz();

        {
            auto& bound_type_map    = s_var->boundary_type_map[domain];
            auto& has_bound_val_map = s_var->has_boundary_value_map[domain];
            auto& bound_val_map     = s_var->boundary_value_map[domain];
            auto& buffer_map        = s_buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                field2* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::Left)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], s, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], s, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], s, nx - 1);
                }
                else if (loc == LocationType::Right)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], s, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], s, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], s, 0);
                }
                else if (loc == LocationType::Front)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], s, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], s, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], s, ny - 1);
                }
                else if (loc == LocationType::Back)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], s, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], s, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], s, 0);
                }
                else if (loc == LocationType::Down)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], s, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], s, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], s, nz - 1);
                }
                else if (loc == LocationType::Up)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], s, nz - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], s, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], s, 0);
                }
            }
        }
    }
}

void ScalarSolver3D::nondiag_shared_boundary_update()
{
    for (auto& domain : domains)
    {
        field3& s = *s_field_map[domain];

        int nx = s.get_nx();
        int ny = s.get_ny();
        int nz = s.get_nz();

        for (auto& [loc, type] : s_var->boundary_type_map[domain])
        {
            if (type == PDEBoundaryType::Adjacented)
            {
                field2& s_buffer = *s_buffer_map[domain][loc];

                Domain3DUniform* adj_domain = adjacency[domain][loc];
                field3&          adj_s      = *s_field_map[adj_domain];
                int              adj_nx     = adj_s.get_nx();
                int              adj_ny     = adj_s.get_ny();
                int              adj_nz     = adj_s.get_nz();
                switch (loc)
                {
                    case LocationType::Left:
                        copy_x_to_buffer(s_buffer, adj_s, adj_nx - 1);
                        break;
                    case LocationType::Right:
                        copy_x_to_buffer(s_buffer, adj_s, 0);
                        break;
                    case LocationType::Front:
                        copy_y_to_buffer(s_buffer, adj_s, adj_ny - 1);
                        break;
                    case LocationType::Back:
                        copy_y_to_buffer(s_buffer, adj_s, 0);
                        break;
                    case LocationType::Down:
                        copy_z_to_buffer(s_buffer, adj_s, adj_nz - 1);
                        break;
                    case LocationType::Up:
                        copy_z_to_buffer(s_buffer, adj_s, 0);
                        break;
                    default:
                        throw std::runtime_error("ScalarSolver3D: invalid location type");
                }
            }
        }
    }
}
