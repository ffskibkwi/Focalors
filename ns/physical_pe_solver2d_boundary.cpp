#include "physical_pe_solver2d.h"

#include "boundary_2d_utils.h"

void PhysicalPESolver2D::phys_boundary_update()
{
    auto update_corner_from_buffer = [](double&         corner,
                                        PDEBoundaryType type,
                                        const double*   buffer,
                                        int             idx_periodic,
                                        int             idx_neumann,
                                        const double*   bound_val,
                                        int             idx_dirichlet,
                                        double          dirichlet_default) {
        if (type == PDEBoundaryType::Dirichlet)
            corner = bound_val ? bound_val[idx_dirichlet] : dirichlet_default;
        else if (type == PDEBoundaryType::Neumann)
            corner = buffer[idx_neumann];
        else if (type == PDEBoundaryType::Periodic)
            corner = buffer[idx_periodic];
    };

    for (auto* domain : domains)
    {
        field2& u = *u_var->field_map[domain];
        field2& v = *v_var->field_map[domain];

        const int nx = u.get_nx();
        const int ny = u.get_ny();

        {
            auto& bound_type_map    = u_var->boundary_type_map[domain];
            auto& has_bound_val_map = u_var->has_boundary_value_map[domain];
            auto& bound_val_map     = u_var->boundary_value_map[domain];
            auto& buffer_map        = u_var->buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                auto* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::XNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_x(u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x(u, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x(u, nx - 1, 0);
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(buffer_map[loc], ny, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], u, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], u, 1);

                    update_corner_from_buffer(u_var->xpos_yneg_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::YNegative],
                                              1,
                                              nx - 1,
                                              bound_val,
                                              0,
                                              0.0);
                    update_corner_from_buffer(u_xpos_ypos_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::YPositive],
                                              1,
                                              nx - 1,
                                              bound_val,
                                              ny - 1,
                                              0.0);
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);

                    update_corner_from_buffer(u_var->xpos_yneg_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::XPositive],
                                              ny - 1,
                                              0,
                                              bound_val,
                                              nx - 1,
                                              0.0);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, 0);

                    update_corner_from_buffer(u_xpos_ypos_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::XPositive],
                                              0,
                                              ny - 1,
                                              bound_val,
                                              nx - 1,
                                              0.0);
                }
            }
        }

        {
            auto& bound_type_map    = v_var->boundary_type_map[domain];
            auto& has_bound_val_map = v_var->has_boundary_value_map[domain];
            auto& bound_val_map     = v_var->boundary_value_map[domain];
            auto& buffer_map        = v_var->buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                auto* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::XNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], v, nx - 1);

                    update_corner_from_buffer(v_var->xneg_ypos_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::YPositive],
                                              nx - 1,
                                              0,
                                              bound_val,
                                              ny - 1,
                                              0.0);
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(buffer_map[loc], v, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], v, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], v, 0);

                    update_corner_from_buffer(v_xpos_ypos_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::YPositive],
                                              0,
                                              nx - 1,
                                              bound_val,
                                              ny - 1,
                                              0.0);
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_y(v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y(v, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y(v, ny - 1, 0);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(buffer_map[loc], nx, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], v, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], v, 1);

                    update_corner_from_buffer(v_var->xneg_ypos_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::XNegative],
                                              1,
                                              ny - 1,
                                              bound_val,
                                              0,
                                              0.0);
                    update_corner_from_buffer(v_xpos_ypos_corner_map[domain],
                                              type,
                                              buffer_map[LocationType::XPositive],
                                              1,
                                              ny - 1,
                                              bound_val,
                                              nx - 1,
                                              0.0);
                }
            }
        }
    }
}

void PhysicalPESolver2D::diag_shared_boundary_update()
{
    for (auto* domain : domains)
    {
        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            if (type != PDEBoundaryType::Adjacented)
                continue;

            Domain2DUniform* adj_domain         = adjacency[domain][loc];
            auto&            adj_bound_type_map = u_var->boundary_type_map[adj_domain];

            if (loc == LocationType::XNegative)
            {
                if (adj_bound_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::YNegative];
                    double*          diag_buffer  = u_var->buffer_map[diag_domain][LocationType::XPositive];
                    double*          local_buffer = u_var->buffer_map[domain][LocationType::YNegative];

                    local_buffer[0] = diag_buffer[diag_domain->get_ny() - 1];
                }
                else
                {
                    double* adj_buffer   = u_var->buffer_map[adj_domain][LocationType::YNegative];
                    double* local_buffer = u_var->buffer_map[domain][LocationType::YNegative];

                    local_buffer[0] = adj_buffer[0];
                }

                if (adj_bound_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::YPositive];
                    double*          diag_buffer  = u_var->buffer_map[diag_domain][LocationType::XPositive];
                    double*          local_buffer = u_var->buffer_map[domain][LocationType::YPositive];

                    local_buffer[0] = diag_buffer[0];
                }
                else
                {
                    double* adj_buffer   = u_var->buffer_map[adj_domain][LocationType::YPositive];
                    double* local_buffer = u_var->buffer_map[domain][LocationType::YPositive];

                    local_buffer[0] = adj_buffer[0];
                }
            }
            else if (loc == LocationType::XPositive)
            {
                if (adj_bound_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::YNegative];
                    field2&          diag_u      = *u_var->field_map[diag_domain];

                    u_var->xpos_yneg_corner_map[domain] = diag_u(0, diag_domain->get_ny() - 1);
                }
                else
                {
                    double* adj_buffer                  = u_var->buffer_map[adj_domain][LocationType::YNegative];
                    u_var->xpos_yneg_corner_map[domain] = adj_buffer[0];
                }

                if (adj_bound_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::YPositive];
                    field2&          diag_u      = *u_var->field_map[diag_domain];
                    field2&          diag_v      = *v_var->field_map[diag_domain];

                    u_xpos_ypos_corner_map[domain] = diag_u(0, 0);
                    v_xpos_ypos_corner_map[domain] = diag_v(0, 0);
                }
                else
                {
                    double* adj_u_buffer = u_var->buffer_map[adj_domain][LocationType::YPositive];
                    double* adj_v_buffer = v_var->buffer_map[adj_domain][LocationType::YPositive];

                    u_xpos_ypos_corner_map[domain] = adj_u_buffer[0];
                    v_xpos_ypos_corner_map[domain] = adj_v_buffer[0];
                }
            }
            else if (loc == LocationType::YNegative)
            {
                if (adj_bound_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::XNegative];
                    double*          diag_buffer  = v_var->buffer_map[diag_domain][LocationType::YPositive];
                    double*          local_buffer = v_var->buffer_map[domain][LocationType::XNegative];

                    local_buffer[0] = diag_buffer[diag_domain->get_nx() - 1];
                }
                else
                {
                    double* adj_buffer   = v_var->buffer_map[adj_domain][LocationType::XNegative];
                    double* local_buffer = v_var->buffer_map[domain][LocationType::XNegative];

                    local_buffer[0] = adj_buffer[0];
                }

                if (adj_bound_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain  = adjacency[adj_domain][LocationType::XPositive];
                    double*          diag_buffer  = v_var->buffer_map[diag_domain][LocationType::YPositive];
                    double*          local_buffer = v_var->buffer_map[domain][LocationType::XPositive];

                    local_buffer[0] = diag_buffer[0];
                }
                else
                {
                    double* adj_buffer   = v_var->buffer_map[adj_domain][LocationType::XPositive];
                    double* local_buffer = v_var->buffer_map[domain][LocationType::XPositive];

                    local_buffer[0] = adj_buffer[0];
                }
            }
            else if (loc == LocationType::YPositive)
            {
                if (adj_bound_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::XNegative];
                    field2&          diag_v      = *v_var->field_map[diag_domain];

                    v_var->xneg_ypos_corner_map[domain] = diag_v(diag_domain->get_nx() - 1, 0);
                }
                else
                {
                    double* adj_buffer                  = v_var->buffer_map[adj_domain][LocationType::XNegative];
                    v_var->xneg_ypos_corner_map[domain] = adj_buffer[0];
                }

                if (adj_bound_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::XPositive];
                    field2&          diag_u      = *u_var->field_map[diag_domain];
                    field2&          diag_v      = *v_var->field_map[diag_domain];

                    u_xpos_ypos_corner_map[domain] = diag_u(0, 0);
                    v_xpos_ypos_corner_map[domain] = diag_v(0, 0);
                }
                else
                {
                    double* adj_u_buffer = u_var->buffer_map[adj_domain][LocationType::XPositive];
                    double* adj_v_buffer = v_var->buffer_map[adj_domain][LocationType::XPositive];

                    u_xpos_ypos_corner_map[domain] = adj_u_buffer[0];
                    v_xpos_ypos_corner_map[domain] = adj_v_buffer[0];
                }
            }
        }
    }
}
