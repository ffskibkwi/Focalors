#include "scalar_solver2d.h"

#include "boundary_2d_utils.h"

void ScalarSolver2D::phys_boundary_update()
{
    for (auto& domain : domains)
    {
        field2& s = *s_field_map[domain];

        int nx = s.get_nx();
        int ny = s.get_ny();

        {
            auto& bound_type_map    = s_var->boundary_type_map[domain];
            auto& has_bound_val_map = s_var->has_boundary_value_map[domain];
            auto& bound_val_map     = s_var->boundary_value_map[domain];
            auto& buffer_map        = s_buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                double* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::XNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], s, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], s, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], s, nx - 1);
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], s, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], s, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], s, 0);
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], s, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], s, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], s, ny - 1);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], s, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], s, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], s, 0);
                }
            }
        }
    }
}

void ScalarSolver2D::nondiag_shared_boundary_update()
{
    for (auto& domain : domains)
    {
        field2& s = *s_field_map[domain];

        int nx = s.get_nx();
        int ny = s.get_ny();

        for (auto& [loc, type] : s_var->boundary_type_map[domain])
        {
            if (type == PDEBoundaryType::Adjacented)
            {
                double*    s_buffer = s_buffer_map[domain][loc];

                Domain2DUniform* adj_domain = adjacency[domain][loc];
                field2&          adj_s      = *s_field_map[adj_domain];
                int              adj_nx     = adj_s.get_nx();
                int              adj_ny     = adj_s.get_ny();
                switch (loc)
                {
                    case LocationType::XNegative:
                        copy_x_to_buffer(s_buffer, adj_s, adj_nx - 1);
                        break;
                    case LocationType::XPositive:
                        copy_x_to_buffer(s_buffer, adj_s, 0);
                        break;
                    case LocationType::YNegative:
                        copy_y_to_buffer(s_buffer, adj_s, adj_ny - 1);
                        break;
                    case LocationType::YPositive:
                        copy_y_to_buffer(s_buffer, adj_s, 0);
                        break;
                    default:
                        throw std::runtime_error("ScalarSolver2D: invalid location type");
                }
            }
        }
    }
}
