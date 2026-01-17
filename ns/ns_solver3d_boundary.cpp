#include "ns_solver3d.h"

#include "boundary_3d_utils.h"

void ConcatNSSolver3D::phys_boundary_update()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();
        int nz = u.get_nz();

        {
            auto& bound_type_map    = u_var->boundary_type_map[domain];
            auto& has_bound_val_map = u_var->has_boundary_value_map[domain];
            auto& bound_val_map     = u_var->boundary_value_map[domain];
            auto& buffer_map        = u_buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                auto bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::Left)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_x(u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x(u, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x(u, nx - 1, 0);
                }
                else if (loc == LocationType::Right)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], u, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], u, 1);
                }
                else if (loc == LocationType::Front)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);
                }
                else if (loc == LocationType::Back)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, 0);
                }
                else if (loc == LocationType::Down)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(buffer_map[loc], u, nz - 1);
                }
                else if (loc == LocationType::Up)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(buffer_map[loc], u, nz - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(buffer_map[loc], u, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(buffer_map[loc], u, 0);
                }
            }
        }

        {
            auto& bound_type_map    = v_var->boundary_type_map[domain];
            auto& has_bound_val_map = v_var->has_boundary_value_map[domain];
            auto& bound_val_map     = v_var->boundary_value_map[domain];
            auto& buffer_map        = v_buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                auto bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::Left)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], v, nx - 1);
                }
                else if (loc == LocationType::Right)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(buffer_map[loc], v, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], v, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], v, 0);
                }
                else if (loc == LocationType::Front)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_y(v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y(v, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y(v, ny - 1, 0);
                }
                else if (loc == LocationType::Back)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], v, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], v, 1);
                }
                else if (loc == LocationType::Down)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(buffer_map[loc], v, nz - 1);
                }
                else if (loc == LocationType::Up)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(buffer_map[loc], v, nz - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(buffer_map[loc], v, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(buffer_map[loc], v, 0);
                }
            }
        }
    }
}

void ConcatNSSolver3D::nondiag_shared_boundary_update()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& p = *p_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            // While u has adjacented boundary, it means v and p also have adjacented boundary
            if (type == PDEBoundaryType::Adjacented)
            {
                double* u_buffer = u_buffer_map[domain][loc];
                double* v_buffer = v_buffer_map[domain][loc];

                Domain3DUniform* adj_domain = adjacency[domain][loc];
                field3&          adj_u      = *u_field_map[adj_domain];
                field3&          adj_v      = *v_field_map[adj_domain];
                int              adj_nx     = adj_u.get_nx();
                int              adj_ny     = adj_u.get_ny();
                switch (loc)
                {
                    case LocationType::Left:
                        copy_x_to_buffer(u_buffer, adj_u, adj_nx - 1);
                        copy_x_to_buffer(v_buffer, adj_v, adj_nx - 1);
                        left_up_corner_value_map[domain] = v_buffer_map[adj_domain][LocationType::Up][adj_nx - 1];
                        break;
                    case LocationType::Right:
                        copy_x_to_buffer(u_buffer, adj_u, 0);
                        copy_x_to_buffer(v_buffer, adj_v, 0);
                        break;
                    case LocationType::Down:
                        copy_y_to_buffer(u_buffer, adj_u, adj_ny - 1);
                        copy_y_to_buffer(v_buffer, adj_v, adj_ny - 1);
                        right_down_corner_value_map[domain] = u_buffer_map[adj_domain][LocationType::Right][adj_ny - 1];
                        break;
                    case LocationType::Up:
                        copy_y_to_buffer(u_buffer, adj_u, 0);
                        copy_y_to_buffer(v_buffer, adj_v, 0);
                        break;
                    default:
                        throw std::runtime_error("ConcatNSSolver3D: invalid location type");
                }
            }
        }
    }
}

void ConcatNSSolver3D::diag_shared_boundary_update()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            // While u has adjacented boundary, it means v and p also have adjacented boundary
            if (type == PDEBoundaryType::Adjacented)
            {
                Domain3DUniform* adj_domain         = adjacency[domain][loc];
                auto&            adj_bound_type_map = u_var->boundary_type_map[adj_domain];

                if (loc == LocationType::Left)
                {
                    if (adj_bound_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Down];
                        double*          diag_buffer  = u_buffer_map[diag_domain][LocationType::Right];
                        double*          local_buffer = u_buffer_map[domain][LocationType::Down];

                        local_buffer[0] = diag_buffer[diag_domain->get_ny() - 1];
                    }
                    if (adj_bound_type_map[LocationType::Up] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Up];
                        double*          diag_buffer  = u_buffer_map[diag_domain][LocationType::Right];
                        double*          local_buffer = u_buffer_map[domain][LocationType::Up];

                        local_buffer[0] = diag_buffer[0];
                    }
                }
                else if (loc == LocationType::Right)
                {
                    if (adj_bound_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Down];
                        auto&            diag_u      = *u_field_map[diag_domain];

                        right_down_corner_value_map[domain] = diag_u(0, diag_domain->get_ny() - 1);
                    }
                }
                else if (loc == LocationType::Down)
                {
                    if (adj_bound_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Left];
                        double*          diag_buffer  = v_buffer_map[diag_domain][LocationType::Up];
                        double*          local_buffer = v_buffer_map[domain][LocationType::Left];

                        local_buffer[0] = diag_buffer[diag_domain->get_nx() - 1];
                    }
                    if (adj_bound_type_map[LocationType::Right] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Right];
                        double*          diag_buffer  = v_buffer_map[diag_domain][LocationType::Up];
                        double*          local_buffer = v_buffer_map[domain][LocationType::Right];

                        local_buffer[0] = diag_buffer[0];
                    }
                }
                else if (loc == LocationType::Up)
                {
                    if (adj_bound_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Left];
                        auto&            diag_v      = *v_field_map[diag_domain];

                        left_up_corner_value_map[domain] = diag_v(diag_domain->get_nx() - 1, 0);
                    }
                }
            }
        }
    }
}