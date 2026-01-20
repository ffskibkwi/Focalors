#include "ns_solver3d.h"

#include "boundary_2d_utils.h"
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
                field2* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

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
                        assign_val_to_buffer(*buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], u, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], u, 1);
                }
                else if (loc == LocationType::Front)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], u, ny - 1);
                }
                else if (loc == LocationType::Back)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], u, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], u, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], u, 0);
                }
                else if (loc == LocationType::Down)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], u, nz - 1);
                }
                else if (loc == LocationType::Up)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], u, nz - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], u, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], u, 0);
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
                field2* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::Left)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], v, nx - 1);
                }
                else if (loc == LocationType::Right)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], v, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], v, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], v, 0);
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
                        assign_val_to_buffer(*buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], v, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], v, 1);
                }
                else if (loc == LocationType::Down)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], v, nz - 1);
                }
                else if (loc == LocationType::Up)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], v, nz - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], v, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], v, 0);
                }
            }
        }

        {
            auto& bound_type_map    = w_var->boundary_type_map[domain];
            auto& has_bound_val_map = w_var->has_boundary_value_map[domain];
            auto& bound_val_map     = w_var->boundary_value_map[domain];
            auto& buffer_map        = w_buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                field2* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::Left)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], w, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], w, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], w, nx - 1);
                }
                else if (loc == LocationType::Right)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], w, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], w, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], w, 0);
                }
                else if (loc == LocationType::Front)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], w, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], w, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], w, ny - 1);
                }
                else if (loc == LocationType::Back)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], w, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], w, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], w, 0);
                }
                else if (loc == LocationType::Down)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_z(w, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z(w, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z(w, nz - 1, 0);
                }
                else if (loc == LocationType::Up)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(*buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], w, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], w, 1);
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
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();
        int nz = u.get_nz();

        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            // While u has adjacented boundary, it means v and p also have adjacented boundary
            if (type == PDEBoundaryType::Adjacented)
            {
                field2& u_buffer = *u_buffer_map[domain][loc];
                field2& v_buffer = *v_buffer_map[domain][loc];
                field2& w_buffer = *w_buffer_map[domain][loc];

                Domain3DUniform* adj_domain = adjacency[domain][loc];
                field3&          adj_u      = *u_field_map[adj_domain];
                field3&          adj_v      = *v_field_map[adj_domain];
                field3&          adj_w      = *w_field_map[adj_domain];
                int              adj_nx     = adj_u.get_nx();
                int              adj_ny     = adj_u.get_ny();
                int              adj_nz     = adj_u.get_nz();
                switch (loc)
                {
                    case LocationType::Left:
                        copy_x_to_buffer(u_buffer, adj_u, adj_nx - 1);
                        copy_x_to_buffer(v_buffer, adj_v, adj_nx - 1);
                        copy_x_to_buffer(w_buffer, adj_w, adj_nx - 1);

                        copy_x_to_buffer(
                            v_corner_value_map_z[domain], *v_buffer_map[adj_domain][LocationType::Back], adj_nx - 1);
                        copy_x_to_buffer(
                            w_corner_value_map_y[domain], *w_buffer_map[adj_domain][LocationType::Up], adj_nx - 1);
                        break;
                    case LocationType::Right:
                        copy_x_to_buffer(u_buffer, adj_u, 0);
                        copy_x_to_buffer(v_buffer, adj_v, 0);
                        copy_x_to_buffer(w_buffer, adj_w, 0);
                        break;
                    case LocationType::Front:
                        copy_y_to_buffer(u_buffer, adj_u, adj_ny - 1);
                        copy_y_to_buffer(v_buffer, adj_v, adj_ny - 1);
                        copy_y_to_buffer(w_buffer, adj_w, adj_ny - 1);

                        copy_x_to_buffer(
                            u_corner_value_map_z[domain], *u_buffer_map[adj_domain][LocationType::Right], adj_ny - 1);
                        copy_y_to_buffer(
                            w_corner_value_map_x[domain], *w_buffer_map[adj_domain][LocationType::Up], adj_ny - 1);
                        break;
                    case LocationType::Back:
                        copy_y_to_buffer(u_buffer, adj_u, 0);
                        copy_y_to_buffer(v_buffer, adj_v, 0);
                        copy_y_to_buffer(w_buffer, adj_w, 0);
                        break;
                    case LocationType::Down:
                        copy_z_to_buffer(u_buffer, adj_u, adj_nz - 1);
                        copy_z_to_buffer(v_buffer, adj_v, adj_nz - 1);
                        copy_z_to_buffer(w_buffer, adj_w, adj_nz - 1);

                        copy_y_to_buffer(
                            u_corner_value_map_y[domain], *u_buffer_map[adj_domain][LocationType::Right], adj_nz - 1);
                        copy_y_to_buffer(
                            v_corner_value_map_x[domain], *v_buffer_map[adj_domain][LocationType::Back], adj_nz - 1);
                        break;
                    case LocationType::Up:
                        copy_z_to_buffer(u_buffer, adj_u, 0);
                        copy_z_to_buffer(v_buffer, adj_v, 0);
                        copy_z_to_buffer(w_buffer, adj_w, 0);
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
        int nz = u.get_nz();

        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            // While u has adjacented boundary, it means v and p also have adjacented boundary
            if (type == PDEBoundaryType::Adjacented)
            {
                Domain3DUniform* adj_domain         = adjacency[domain][loc];
                auto&            adj_bound_type_map = u_var->boundary_type_map[adj_domain];

                // Project to xy

                if (loc == LocationType::Left)
                {
                    if (adj_bound_type_map[LocationType::Front] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Front];
                        field2&          diag_buffer  = *u_buffer_map[diag_domain][LocationType::Right];
                        field2&          local_buffer = *u_buffer_map[domain][LocationType::Front];

                        copy_src_x_to_buffer_x(local_buffer, diag_buffer, diag_domain->get_ny() - 1, 0);
                    }
                    if (adj_bound_type_map[LocationType::Back] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Back];
                        field2&          diag_buffer  = *u_buffer_map[diag_domain][LocationType::Right];
                        field2&          local_buffer = *u_buffer_map[domain][LocationType::Back];

                        copy_src_x_to_buffer_x(local_buffer, diag_buffer, 0, 0);
                    }
                }
                else if (loc == LocationType::Right)
                {
                    if (adj_bound_type_map[LocationType::Front] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Front];
                        auto&            diag_u      = *u_field_map[diag_domain];

                        copy_z_to_buffer(u_corner_value_map_z[domain], diag_u, 0, diag_domain->get_ny() - 1);
                    }
                }
                else if (loc == LocationType::Front)
                {
                    if (adj_bound_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Left];
                        field2&          diag_buffer  = *v_buffer_map[diag_domain][LocationType::Back];
                        field2&          local_buffer = *v_buffer_map[domain][LocationType::Left];

                        copy_src_x_to_buffer_x(local_buffer, diag_buffer, diag_domain->get_nx() - 1, 0);
                    }
                    if (adj_bound_type_map[LocationType::Right] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Right];
                        field2&          diag_buffer  = *v_buffer_map[diag_domain][LocationType::Back];
                        field2&          local_buffer = *v_buffer_map[domain][LocationType::Right];

                        copy_src_x_to_buffer_x(local_buffer, diag_buffer, 0, 0);
                    }
                }
                else if (loc == LocationType::Back)
                {
                    if (adj_bound_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Left];
                        auto&            diag_v      = *v_field_map[diag_domain];

                        copy_z_to_buffer(v_corner_value_map_z[domain], diag_v, diag_domain->get_nx() - 1, 0);
                    }
                }

                // Project to xz

                if (loc == LocationType::Left)
                {
                    if (adj_bound_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Down];
                        field2&          diag_buffer  = *u_buffer_map[diag_domain][LocationType::Right];
                        field2&          local_buffer = *u_buffer_map[domain][LocationType::Down];

                        copy_src_y_to_buffer_x(local_buffer, diag_buffer, diag_domain->get_nz() - 1, 0);
                    }
                    if (adj_bound_type_map[LocationType::Up] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Up];
                        field2&          diag_buffer  = *u_buffer_map[diag_domain][LocationType::Right];
                        field2&          local_buffer = *u_buffer_map[domain][LocationType::Up];

                        copy_src_y_to_buffer_x(local_buffer, diag_buffer, 0, 0);
                    }
                }
                else if (loc == LocationType::Right)
                {
                    if (adj_bound_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Down];
                        auto&            diag_u      = *u_field_map[diag_domain];

                        copy_y_to_buffer(u_corner_value_map_y[domain], diag_u, 0, diag_domain->get_nz() - 1);
                    }
                }
                else if (loc == LocationType::Down)
                {
                    if (adj_bound_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Left];
                        field2&          diag_buffer  = *w_buffer_map[diag_domain][LocationType::Up];
                        field2&          local_buffer = *w_buffer_map[domain][LocationType::Left];

                        copy_src_x_to_buffer_y(local_buffer, diag_buffer, diag_domain->get_nx() - 1, 0);
                    }
                    if (adj_bound_type_map[LocationType::Right] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Right];
                        field2&          diag_buffer  = *w_buffer_map[diag_domain][LocationType::Up];
                        field2&          local_buffer = *w_buffer_map[domain][LocationType::Right];

                        copy_src_x_to_buffer_y(local_buffer, diag_buffer, 0, 0);
                    }
                }
                else if (loc == LocationType::Up)
                {
                    if (adj_bound_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Left];
                        auto&            diag_w      = *w_field_map[diag_domain];

                        copy_y_to_buffer(w_corner_value_map_y[domain], diag_w, diag_domain->get_nx() - 1, 0);
                    }
                }

                // Project to yz

                if (loc == LocationType::Front)
                {
                    if (adj_bound_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Down];
                        field2&          diag_buffer  = *v_buffer_map[diag_domain][LocationType::Back];
                        field2&          local_buffer = *v_buffer_map[domain][LocationType::Down];

                        copy_src_y_to_buffer_y(local_buffer, diag_buffer, diag_domain->get_nz() - 1, 0);
                    }
                    if (adj_bound_type_map[LocationType::Up] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Up];
                        field2&          diag_buffer  = *v_buffer_map[diag_domain][LocationType::Back];
                        field2&          local_buffer = *v_buffer_map[domain][LocationType::Up];

                        copy_src_y_to_buffer_y(local_buffer, diag_buffer, 0, 0);
                    }
                }
                else if (loc == LocationType::Back)
                {
                    if (adj_bound_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Down];
                        auto&            diag_v      = *v_field_map[diag_domain];

                        copy_x_to_buffer(v_corner_value_map_x[domain], diag_v, 0, diag_domain->get_nz() - 1);
                    }
                }
                else if (loc == LocationType::Down)
                {
                    if (adj_bound_type_map[LocationType::Front] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Front];
                        field2&          diag_buffer  = *w_buffer_map[diag_domain][LocationType::Up];
                        field2&          local_buffer = *w_buffer_map[domain][LocationType::Front];

                        copy_src_y_to_buffer_y(local_buffer, diag_buffer, diag_domain->get_ny() - 1, 0);
                    }
                    if (adj_bound_type_map[LocationType::Back] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain  = adjacency[adj_domain][LocationType::Back];
                        field2&          diag_buffer  = *w_buffer_map[diag_domain][LocationType::Up];
                        field2&          local_buffer = *w_buffer_map[domain][LocationType::Back];

                        copy_src_y_to_buffer_y(local_buffer, diag_buffer, 0, 0);
                    }
                }
                else if (loc == LocationType::Up)
                {
                    if (adj_bound_type_map[LocationType::Front] == PDEBoundaryType::Adjacented)
                    {
                        Domain3DUniform* diag_domain = adjacency[adj_domain][LocationType::Front];
                        auto&            diag_w      = *w_field_map[diag_domain];

                        copy_x_to_buffer(w_corner_value_map_x[domain], diag_w, 0, diag_domain->get_nz() - 1);
                    }
                }
            }
        }
    }
}