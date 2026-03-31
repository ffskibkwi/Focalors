#include "physical_pe_solver2d.h"

#include "boundary_2d_utils.h"

void PhysicalPESolver2D::phys_boundary_update()
{
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
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(buffer_map[loc], u, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(buffer_map[loc], u, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], u, 0);
                }
            }

            const PDEBoundaryType xpos_type = bound_type_map.count(LocationType::XPositive) ?
                                                  bound_type_map.at(LocationType::XPositive) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType yneg_type = bound_type_map.count(LocationType::YNegative) ?
                                                  bound_type_map.at(LocationType::YNegative) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType ypos_type = bound_type_map.count(LocationType::YPositive) ?
                                                  bound_type_map.at(LocationType::YPositive) :
                                                  PDEBoundaryType::Null;

            const bool xpos_physical = xpos_type != PDEBoundaryType::Adjacented && xpos_type != PDEBoundaryType::Null;
            const bool yneg_physical = yneg_type != PDEBoundaryType::Adjacented && yneg_type != PDEBoundaryType::Null;
            const bool ypos_physical = ypos_type != PDEBoundaryType::Adjacented && ypos_type != PDEBoundaryType::Null;

            // NS does not materialize physical corner ghosts here. PPE does,
            // because calc_rhs() reads xpos_yneg directly at the lower-right cell.
            if (xpos_physical)
            {
                double* xpos_buffer = buffer_map[LocationType::XPositive];
                if (ny == 1)
                    u_var->xpos_yneg_corner_map[domain] = xpos_buffer[0];
                else
                    u_var->xpos_yneg_corner_map[domain] = 2.0 * xpos_buffer[0] - xpos_buffer[1];
            }
            else if (yneg_physical)
            {
                double* yneg_buffer = buffer_map[LocationType::YNegative];
                if (nx == 1)
                    u_var->xpos_yneg_corner_map[domain] = yneg_buffer[nx - 1];
                else
                    u_var->xpos_yneg_corner_map[domain] = 2.0 * yneg_buffer[nx - 1] - yneg_buffer[nx - 2];
            }

            // PPE-only upper-right ghost: rhs(dudy) needs u(i+1,j+1) at the
            // top-right cell, while NS has no corresponding corner storage.
            if (xpos_physical && ypos_physical)
            {
                double* xpos_buffer = buffer_map[LocationType::XPositive];
                if (ny == 1)
                    u_xpos_ypos_corner_map[domain] = xpos_buffer[ny - 1];
                else
                    u_xpos_ypos_corner_map[domain] = 2.0 * xpos_buffer[ny - 1] - xpos_buffer[ny - 2];
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
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(buffer_map[loc], v, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(buffer_map[loc], v, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], v, 0);
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
                }
            }

            const PDEBoundaryType xneg_type = bound_type_map.count(LocationType::XNegative) ?
                                                  bound_type_map.at(LocationType::XNegative) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType xpos_type = bound_type_map.count(LocationType::XPositive) ?
                                                  bound_type_map.at(LocationType::XPositive) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType ypos_type = bound_type_map.count(LocationType::YPositive) ?
                                                  bound_type_map.at(LocationType::YPositive) :
                                                  PDEBoundaryType::Null;

            const bool xneg_physical = xneg_type != PDEBoundaryType::Adjacented && xneg_type != PDEBoundaryType::Null;
            const bool xpos_physical = xpos_type != PDEBoundaryType::Adjacented && xpos_type != PDEBoundaryType::Null;
            const bool ypos_physical = ypos_type != PDEBoundaryType::Adjacented && ypos_type != PDEBoundaryType::Null;

            // NS does not materialize physical corner ghosts here. PPE does,
            // because calc_rhs() reads xneg_ypos directly at the upper-left cell.
            if (ypos_physical)
            {
                double* ypos_buffer = buffer_map[LocationType::YPositive];
                if (nx == 1)
                    v_var->xneg_ypos_corner_map[domain] = ypos_buffer[0];
                else
                    v_var->xneg_ypos_corner_map[domain] = 2.0 * ypos_buffer[0] - ypos_buffer[1];
            }
            else if (xneg_physical)
            {
                double* xneg_buffer = buffer_map[LocationType::XNegative];
                if (ny == 1)
                    v_var->xneg_ypos_corner_map[domain] = xneg_buffer[ny - 1];
                else
                    v_var->xneg_ypos_corner_map[domain] = 2.0 * xneg_buffer[ny - 1] - xneg_buffer[ny - 2];
            }

            // PPE-only upper-right ghost: rhs(dvdx) needs v(i+1,j+1) at the
            // top-right cell, while NS has no corresponding corner storage.
            if (xpos_physical && ypos_physical)
            {
                double* ypos_buffer = buffer_map[LocationType::YPositive];
                if (nx == 1)
                    v_xpos_ypos_corner_map[domain] = ypos_buffer[nx - 1];
                else
                    v_xpos_ypos_corner_map[domain] = 2.0 * ypos_buffer[nx - 1] - ypos_buffer[nx - 2];
            }
        }
    }
}

void PhysicalPESolver2D::nondiag_shared_boundary_update()
{
    for (auto* domain : domains)
    {
        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            if (type != PDEBoundaryType::Adjacented)
                continue;

            double* u_buffer = u_var->buffer_map[domain][loc];
            double* v_buffer = v_var->buffer_map[domain][loc];

            Domain2DUniform* adj_domain = adjacency[domain][loc];
            field2&          adj_u      = *u_var->field_map[adj_domain];
            field2&          adj_v      = *v_var->field_map[adj_domain];
            const int        adj_nx     = adj_u.get_nx();
            const int        adj_ny     = adj_u.get_ny();
            auto&            adj_bound_type_map = u_var->boundary_type_map[adj_domain];

            switch (loc)
            {
                case LocationType::XNegative:
                    copy_x_to_buffer(u_buffer, adj_u, adj_nx - 1);
                    copy_x_to_buffer(v_buffer, adj_v, adj_nx - 1);
                    v_var->xneg_ypos_corner_map[domain] =
                        v_var->buffer_map[adj_domain][LocationType::YPositive][adj_nx - 1];
                    break;
                case LocationType::XPositive:
                    copy_x_to_buffer(u_buffer, adj_u, 0);
                    copy_x_to_buffer(v_buffer, adj_v, 0);
                    // Unlike NS, PPE must also supply the new upper-right
                    // ghost when the orthogonal neighbor owns the physical top boundary.
                    if (adj_bound_type_map[LocationType::YPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::YPositive] != PDEBoundaryType::Null)
                    {
                        double* adj_u_ypos_buffer = u_var->buffer_map[adj_domain][LocationType::YPositive];
                        double* adj_v_ypos_buffer = v_var->buffer_map[adj_domain][LocationType::YPositive];

                        u_xpos_ypos_corner_map[domain] = adj_u_ypos_buffer[0];
                        v_xpos_ypos_corner_map[domain] = adj_v_ypos_buffer[0];
                    }
                    break;
                case LocationType::YNegative:
                    copy_y_to_buffer(u_buffer, adj_u, adj_ny - 1);
                    copy_y_to_buffer(v_buffer, adj_v, adj_ny - 1);
                    u_var->xpos_yneg_corner_map[domain] =
                        u_var->buffer_map[adj_domain][LocationType::XPositive][adj_ny - 1];
                    break;
                case LocationType::YPositive:
                    copy_y_to_buffer(u_buffer, adj_u, 0);
                    copy_y_to_buffer(v_buffer, adj_v, 0);
                    // Symmetric PPE-only case: no diagonal domain, so the
                    // upper-right ghost comes from the orthogonal right neighbor.
                    if (adj_bound_type_map[LocationType::XPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::XPositive] != PDEBoundaryType::Null)
                    {
                        double* adj_u_xpos_buffer = u_var->buffer_map[adj_domain][LocationType::XPositive];
                        double* adj_v_xpos_buffer = v_var->buffer_map[adj_domain][LocationType::XPositive];

                        u_xpos_ypos_corner_map[domain] = adj_u_xpos_buffer[0];
                        v_xpos_ypos_corner_map[domain] = adj_v_xpos_buffer[0];
                    }
                    break;
                default:
                    throw std::runtime_error("PhysicalPESolver2D: invalid location type");
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
                // Unlike NS, PPE does not let diag overwrite shared-face
                // endpoints here. calc_rhs() should see the face values copied
                // by nondiag_shared_boundary_update(), and diag only repairs corners.
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

                // PPE-only upper-right corner: when a true diagonal domain exists,
                // rhs reads the ghost from that diagonal field instead of an edge buffer.
                if (adj_bound_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::YPositive];
                    field2&          diag_u      = *u_var->field_map[diag_domain];
                    field2&          diag_v      = *v_var->field_map[diag_domain];

                    u_xpos_ypos_corner_map[domain] = diag_u(0, 0);
                    v_xpos_ypos_corner_map[domain] = diag_v(0, 0);
                }
            }
            else if (loc == LocationType::YNegative)
            {
                // Unlike NS, PPE does not let diag overwrite shared-face
                // endpoints here. calc_rhs() should see the face values copied
                // by nondiag_shared_boundary_update(), and diag only repairs corners.
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

                // Symmetric PPE-only upper-right corner from the true diagonal domain.
                if (adj_bound_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain2DUniform* diag_domain = adjacency[adj_domain][LocationType::XPositive];
                    field2&          diag_u      = *u_var->field_map[diag_domain];
                    field2&          diag_v      = *v_var->field_map[diag_domain];

                    u_xpos_ypos_corner_map[domain] = diag_u(0, 0);
                    v_xpos_ypos_corner_map[domain] = diag_v(0, 0);
                }
            }
        }
    }
}
