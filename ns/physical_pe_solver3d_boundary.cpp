#include "physical_pe_solver3d.h"

#include "boundary_2d_utils.h"
#include "boundary_3d_utils.h"

void PhysicalPESolver3D::phys_boundary_update()
{
    for (auto* domain : u_var->geometry->domains)
    {
        field3& u = *u_var->field_map[domain];
        field3& v = *v_var->field_map[domain];
        field3& w = *w_var->field_map[domain];

        const int nx = u.get_nx();
        const int ny = u.get_ny();
        const int nz = u.get_nz();

        {
            auto& bound_type_map    = u_var->boundary_type_map[domain];
            auto& has_bound_val_map = u_var->has_boundary_value_map[domain];
            auto& bound_val_map     = u_var->boundary_value_map[domain];
            auto& buffer_map        = u_var->buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                field2* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

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
                        assign_val_to_buffer(*buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], u, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], u, 1);
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], u, ny - 1);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], u, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], u, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], u, 0);
                }
                else if (loc == LocationType::ZNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], u, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], u, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], u, nz - 1);
                }
                else if (loc == LocationType::ZPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], u, nz - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], u, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], u, 0);
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
            const PDEBoundaryType zneg_type = bound_type_map.count(LocationType::ZNegative) ?
                                                  bound_type_map.at(LocationType::ZNegative) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType zpos_type = bound_type_map.count(LocationType::ZPositive) ?
                                                  bound_type_map.at(LocationType::ZPositive) :
                                                  PDEBoundaryType::Null;

            const bool xpos_physical = xpos_type != PDEBoundaryType::Adjacented && xpos_type != PDEBoundaryType::Null;
            const bool yneg_physical = yneg_type != PDEBoundaryType::Adjacented && yneg_type != PDEBoundaryType::Null;
            const bool ypos_physical = ypos_type != PDEBoundaryType::Adjacented && ypos_type != PDEBoundaryType::Null;
            const bool zneg_physical = zneg_type != PDEBoundaryType::Adjacented && zneg_type != PDEBoundaryType::Null;
            const bool zpos_physical = zpos_type != PDEBoundaryType::Adjacented && zpos_type != PDEBoundaryType::Null;

            if (xpos_physical)
            {
                field2& xpos_buffer = *buffer_map[LocationType::XPositive];

                for (int k = 0; k < nz; ++k)
                {
                    if (ny == 1)
                        u_var->corner_z_map[domain][k] = xpos_buffer(0, k);
                    else
                        u_var->corner_z_map[domain][k] = 2.0 * xpos_buffer(0, k) - xpos_buffer(1, k);
                }

                for (int j = 0; j < ny; ++j)
                {
                    if (nz == 1)
                        u_var->corner_y_map[domain][j] = xpos_buffer(j, 0);
                    else
                        u_var->corner_y_map[domain][j] = 2.0 * xpos_buffer(j, 0) - xpos_buffer(j, 1);
                }

                if (ypos_physical)
                {
                    for (int k = 0; k < nz; ++k)
                    {
                        if (ny == 1)
                            u_xpos_ypos_corner_map[domain][k] = xpos_buffer(ny - 1, k);
                        else
                            u_xpos_ypos_corner_map[domain][k] =
                                2.0 * xpos_buffer(ny - 1, k) - xpos_buffer(ny - 2, k);
                    }
                }

                if (zpos_physical)
                {
                    for (int j = 0; j < ny; ++j)
                    {
                        if (nz == 1)
                            u_xpos_zpos_corner_map[domain][j] = xpos_buffer(j, nz - 1);
                        else
                            u_xpos_zpos_corner_map[domain][j] =
                                2.0 * xpos_buffer(j, nz - 1) - xpos_buffer(j, nz - 2);
                    }
                }
            }
            else
            {
                if (yneg_physical)
                {
                    field2& yneg_buffer = *buffer_map[LocationType::YNegative];
                    for (int k = 0; k < nz; ++k)
                    {
                        if (nx == 1)
                            u_var->corner_z_map[domain][k] = yneg_buffer(nx - 1, k);
                        else
                            u_var->corner_z_map[domain][k] =
                                2.0 * yneg_buffer(nx - 1, k) - yneg_buffer(nx - 2, k);
                    }
                }

                if (zneg_physical)
                {
                    field2& zneg_buffer = *buffer_map[LocationType::ZNegative];
                    for (int j = 0; j < ny; ++j)
                    {
                        if (nx == 1)
                            u_var->corner_y_map[domain][j] = zneg_buffer(nx - 1, j);
                        else
                            u_var->corner_y_map[domain][j] =
                                2.0 * zneg_buffer(nx - 1, j) - zneg_buffer(nx - 2, j);
                    }
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
                field2* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::XNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], v, nx - 1);
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], v, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], v, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], v, 0);
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
                        assign_val_to_buffer(*buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], v, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], v, 1);
                }
                else if (loc == LocationType::ZNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], v, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], v, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], v, nz - 1);
                }
                else if (loc == LocationType::ZPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_z_to_buffer(*buffer_map[loc], v, nz - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], v, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], v, 0);
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
            const PDEBoundaryType zneg_type = bound_type_map.count(LocationType::ZNegative) ?
                                                  bound_type_map.at(LocationType::ZNegative) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType zpos_type = bound_type_map.count(LocationType::ZPositive) ?
                                                  bound_type_map.at(LocationType::ZPositive) :
                                                  PDEBoundaryType::Null;

            const bool xneg_physical = xneg_type != PDEBoundaryType::Adjacented && xneg_type != PDEBoundaryType::Null;
            const bool xpos_physical = xpos_type != PDEBoundaryType::Adjacented && xpos_type != PDEBoundaryType::Null;
            const bool ypos_physical = ypos_type != PDEBoundaryType::Adjacented && ypos_type != PDEBoundaryType::Null;
            const bool zneg_physical = zneg_type != PDEBoundaryType::Adjacented && zneg_type != PDEBoundaryType::Null;
            const bool zpos_physical = zpos_type != PDEBoundaryType::Adjacented && zpos_type != PDEBoundaryType::Null;

            if (ypos_physical)
            {
                field2& ypos_buffer = *buffer_map[LocationType::YPositive];

                for (int k = 0; k < nz; ++k)
                {
                    if (nx == 1)
                        v_var->corner_z_map[domain][k] = ypos_buffer(0, k);
                    else
                        v_var->corner_z_map[domain][k] = 2.0 * ypos_buffer(0, k) - ypos_buffer(1, k);
                }

                for (int i = 0; i < nx; ++i)
                {
                    if (nz == 1)
                        v_var->corner_x_map[domain][i] = ypos_buffer(i, 0);
                    else
                        v_var->corner_x_map[domain][i] = 2.0 * ypos_buffer(i, 0) - ypos_buffer(i, 1);
                }

                if (xpos_physical)
                {
                    for (int k = 0; k < nz; ++k)
                    {
                        if (nx == 1)
                            v_xpos_ypos_corner_map[domain][k] = ypos_buffer(nx - 1, k);
                        else
                            v_xpos_ypos_corner_map[domain][k] =
                                2.0 * ypos_buffer(nx - 1, k) - ypos_buffer(nx - 2, k);
                    }
                }

                if (zpos_physical)
                {
                    for (int i = 0; i < nx; ++i)
                    {
                        if (nz == 1)
                            v_ypos_zpos_corner_map[domain][i] = ypos_buffer(i, nz - 1);
                        else
                            v_ypos_zpos_corner_map[domain][i] =
                                2.0 * ypos_buffer(i, nz - 1) - ypos_buffer(i, nz - 2);
                    }
                }
            }
            else
            {
                if (xneg_physical)
                {
                    field2& xneg_buffer = *buffer_map[LocationType::XNegative];
                    for (int k = 0; k < nz; ++k)
                    {
                        if (ny == 1)
                            v_var->corner_z_map[domain][k] = xneg_buffer(ny - 1, k);
                        else
                            v_var->corner_z_map[domain][k] =
                                2.0 * xneg_buffer(ny - 1, k) - xneg_buffer(ny - 2, k);
                    }
                }

                if (zneg_physical)
                {
                    field2& zneg_buffer = *buffer_map[LocationType::ZNegative];
                    for (int i = 0; i < nx; ++i)
                    {
                        if (ny == 1)
                            v_var->corner_x_map[domain][i] = zneg_buffer(i, ny - 1);
                        else
                            v_var->corner_x_map[domain][i] =
                                2.0 * zneg_buffer(i, ny - 1) - zneg_buffer(i, ny - 2);
                    }
                }
            }
        }

        {
            auto& bound_type_map    = w_var->boundary_type_map[domain];
            auto& has_bound_val_map = w_var->has_boundary_value_map[domain];
            auto& bound_val_map     = w_var->boundary_value_map[domain];
            auto& buffer_map        = w_var->buffer_map[domain];

            for (auto& [loc, type] : bound_type_map)
            {
                field2* bound_val = has_bound_val_map[loc] ? bound_val_map[loc] : nullptr;

                if (loc == LocationType::XNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], w, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], w, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], w, nx - 1);
                }
                else if (loc == LocationType::XPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_x_to_buffer(*buffer_map[loc], w, nx - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_x_to_buffer(*buffer_map[loc], w, nx - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(*buffer_map[loc], w, 0);
                }
                else if (loc == LocationType::YNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], w, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], w, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], w, ny - 1);
                }
                else if (loc == LocationType::YPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        mirror_y_to_buffer(*buffer_map[loc], w, ny - 1, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_y_to_buffer(*buffer_map[loc], w, ny - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(*buffer_map[loc], w, 0);
                }
                else if (loc == LocationType::ZNegative)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_z(w, 0, bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z(w, 1, 0);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z(w, nz - 1, 0);
                }
                else if (loc == LocationType::ZPositive)
                {
                    if (type == PDEBoundaryType::Dirichlet)
                        assign_val_to_buffer(*buffer_map[loc], bound_val, 0.0);
                    else if (type == PDEBoundaryType::Neumann)
                        copy_z_to_buffer(*buffer_map[loc], w, nz - 1);
                    else if (type == PDEBoundaryType::Periodic)
                        copy_z_to_buffer(*buffer_map[loc], w, 1);
                }
            }

            const PDEBoundaryType xneg_type = bound_type_map.count(LocationType::XNegative) ?
                                                  bound_type_map.at(LocationType::XNegative) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType xpos_type = bound_type_map.count(LocationType::XPositive) ?
                                                  bound_type_map.at(LocationType::XPositive) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType yneg_type = bound_type_map.count(LocationType::YNegative) ?
                                                  bound_type_map.at(LocationType::YNegative) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType ypos_type = bound_type_map.count(LocationType::YPositive) ?
                                                  bound_type_map.at(LocationType::YPositive) :
                                                  PDEBoundaryType::Null;
            const PDEBoundaryType zpos_type = bound_type_map.count(LocationType::ZPositive) ?
                                                  bound_type_map.at(LocationType::ZPositive) :
                                                  PDEBoundaryType::Null;

            const bool xneg_physical = xneg_type != PDEBoundaryType::Adjacented && xneg_type != PDEBoundaryType::Null;
            const bool xpos_physical = xpos_type != PDEBoundaryType::Adjacented && xpos_type != PDEBoundaryType::Null;
            const bool yneg_physical = yneg_type != PDEBoundaryType::Adjacented && yneg_type != PDEBoundaryType::Null;
            const bool ypos_physical = ypos_type != PDEBoundaryType::Adjacented && ypos_type != PDEBoundaryType::Null;
            const bool zpos_physical = zpos_type != PDEBoundaryType::Adjacented && zpos_type != PDEBoundaryType::Null;

            if (zpos_physical)
            {
                field2& zpos_buffer = *buffer_map[LocationType::ZPositive];

                for (int i = 0; i < nx; ++i)
                {
                    if (ny == 1)
                        w_var->corner_x_map[domain][i] = zpos_buffer(i, 0);
                    else
                        w_var->corner_x_map[domain][i] = 2.0 * zpos_buffer(i, 0) - zpos_buffer(i, 1);
                }

                for (int j = 0; j < ny; ++j)
                {
                    if (nx == 1)
                        w_var->corner_y_map[domain][j] = zpos_buffer(0, j);
                    else
                        w_var->corner_y_map[domain][j] = 2.0 * zpos_buffer(0, j) - zpos_buffer(1, j);
                }

                if (xpos_physical)
                {
                    for (int j = 0; j < ny; ++j)
                    {
                        if (nx == 1)
                            w_xpos_zpos_corner_map[domain][j] = zpos_buffer(nx - 1, j);
                        else
                            w_xpos_zpos_corner_map[domain][j] =
                                2.0 * zpos_buffer(nx - 1, j) - zpos_buffer(nx - 2, j);
                    }
                }

                if (ypos_physical)
                {
                    for (int i = 0; i < nx; ++i)
                    {
                        if (ny == 1)
                            w_ypos_zpos_corner_map[domain][i] = zpos_buffer(i, ny - 1);
                        else
                            w_ypos_zpos_corner_map[domain][i] =
                                2.0 * zpos_buffer(i, ny - 1) - zpos_buffer(i, ny - 2);
                    }
                }
            }
            else
            {
                if (yneg_physical)
                {
                    field2& yneg_buffer = *buffer_map[LocationType::YNegative];
                    for (int i = 0; i < nx; ++i)
                    {
                        if (nz == 1)
                            w_var->corner_x_map[domain][i] = yneg_buffer(i, nz - 1);
                        else
                            w_var->corner_x_map[domain][i] =
                                2.0 * yneg_buffer(i, nz - 1) - yneg_buffer(i, nz - 2);
                    }
                }

                if (xneg_physical)
                {
                    field2& xneg_buffer = *buffer_map[LocationType::XNegative];
                    for (int j = 0; j < ny; ++j)
                    {
                        if (nz == 1)
                            w_var->corner_y_map[domain][j] = xneg_buffer(j, nz - 1);
                        else
                            w_var->corner_y_map[domain][j] =
                                2.0 * xneg_buffer(j, nz - 1) - xneg_buffer(j, nz - 2);
                    }
                }
            }
        }
    }
}

void PhysicalPESolver3D::nondiag_shared_boundary_update()
{
    for (auto* domain : u_var->geometry->domains)
    {
        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            if (type != PDEBoundaryType::Adjacented)
                continue;

            field2& u_buffer = *u_var->buffer_map[domain][loc];
            field2& v_buffer = *v_var->buffer_map[domain][loc];
            field2& w_buffer = *w_var->buffer_map[domain][loc];

            Domain3DUniform* adj_domain = u_var->geometry->adjacency[domain][loc];
            field3&          adj_u      = *u_var->field_map[adj_domain];
            field3&          adj_v      = *v_var->field_map[adj_domain];
            field3&          adj_w      = *w_var->field_map[adj_domain];
            const int        adj_nx     = adj_u.get_nx();
            const int        adj_ny     = adj_u.get_ny();
            const int        adj_nz     = adj_u.get_nz();
            auto&            adj_bound_type_map = u_var->boundary_type_map[adj_domain];

            switch (loc)
            {
                case LocationType::XNegative:
                    copy_x_to_buffer(u_buffer, adj_u, adj_nx - 1);
                    copy_x_to_buffer(v_buffer, adj_v, adj_nx - 1);
                    copy_x_to_buffer(w_buffer, adj_w, adj_nx - 1);

                    copy_x_to_buffer(v_var->corner_z_map[domain], *v_var->buffer_map[adj_domain][LocationType::YPositive],
                                     adj_nx - 1);
                    copy_x_to_buffer(w_var->corner_y_map[domain], *w_var->buffer_map[adj_domain][LocationType::ZPositive],
                                     adj_nx - 1);
                    break;
                case LocationType::XPositive:
                    copy_x_to_buffer(u_buffer, adj_u, 0);
                    copy_x_to_buffer(v_buffer, adj_v, 0);
                    copy_x_to_buffer(w_buffer, adj_w, 0);

                    if (adj_bound_type_map[LocationType::YPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::YPositive] != PDEBoundaryType::Null)
                    {
                        copy_x_to_buffer(u_xpos_ypos_corner_map[domain], *u_var->buffer_map[adj_domain][LocationType::YPositive],
                                         0);
                        copy_x_to_buffer(v_xpos_ypos_corner_map[domain], *v_var->buffer_map[adj_domain][LocationType::YPositive],
                                         0);
                    }

                    if (adj_bound_type_map[LocationType::ZPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::ZPositive] != PDEBoundaryType::Null)
                    {
                        copy_x_to_buffer(u_xpos_zpos_corner_map[domain], *u_var->buffer_map[adj_domain][LocationType::ZPositive],
                                         0);
                        copy_x_to_buffer(w_xpos_zpos_corner_map[domain], *w_var->buffer_map[adj_domain][LocationType::ZPositive],
                                         0);
                    }
                    break;
                case LocationType::YNegative:
                    copy_y_to_buffer(u_buffer, adj_u, adj_ny - 1);
                    copy_y_to_buffer(v_buffer, adj_v, adj_ny - 1);
                    copy_y_to_buffer(w_buffer, adj_w, adj_ny - 1);

                    copy_x_to_buffer(u_var->corner_z_map[domain], *u_var->buffer_map[adj_domain][LocationType::XPositive],
                                     adj_ny - 1);
                    copy_y_to_buffer(w_var->corner_x_map[domain], *w_var->buffer_map[adj_domain][LocationType::ZPositive],
                                     adj_ny - 1);
                    break;
                case LocationType::YPositive:
                    copy_y_to_buffer(u_buffer, adj_u, 0);
                    copy_y_to_buffer(v_buffer, adj_v, 0);
                    copy_y_to_buffer(w_buffer, adj_w, 0);

                    if (adj_bound_type_map[LocationType::XPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::XPositive] != PDEBoundaryType::Null)
                    {
                        copy_x_to_buffer(u_xpos_ypos_corner_map[domain], *u_var->buffer_map[adj_domain][LocationType::XPositive],
                                         0);
                        copy_x_to_buffer(v_xpos_ypos_corner_map[domain], *v_var->buffer_map[adj_domain][LocationType::XPositive],
                                         0);
                    }

                    if (adj_bound_type_map[LocationType::ZPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::ZPositive] != PDEBoundaryType::Null)
                    {
                        copy_y_to_buffer(v_ypos_zpos_corner_map[domain], *v_var->buffer_map[adj_domain][LocationType::ZPositive],
                                         0);
                        copy_y_to_buffer(w_ypos_zpos_corner_map[domain], *w_var->buffer_map[adj_domain][LocationType::ZPositive],
                                         0);
                    }
                    break;
                case LocationType::ZNegative:
                    copy_z_to_buffer(u_buffer, adj_u, adj_nz - 1);
                    copy_z_to_buffer(v_buffer, adj_v, adj_nz - 1);
                    copy_z_to_buffer(w_buffer, adj_w, adj_nz - 1);

                    copy_y_to_buffer(u_var->corner_y_map[domain], *u_var->buffer_map[adj_domain][LocationType::XPositive],
                                     adj_nz - 1);
                    copy_y_to_buffer(v_var->corner_x_map[domain], *v_var->buffer_map[adj_domain][LocationType::YPositive],
                                     adj_nz - 1);
                    break;
                case LocationType::ZPositive:
                    copy_z_to_buffer(u_buffer, adj_u, 0);
                    copy_z_to_buffer(v_buffer, adj_v, 0);
                    copy_z_to_buffer(w_buffer, adj_w, 0);

                    if (adj_bound_type_map[LocationType::XPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::XPositive] != PDEBoundaryType::Null)
                    {
                        copy_y_to_buffer(u_xpos_zpos_corner_map[domain], *u_var->buffer_map[adj_domain][LocationType::XPositive],
                                         0);
                        copy_y_to_buffer(w_xpos_zpos_corner_map[domain], *w_var->buffer_map[adj_domain][LocationType::XPositive],
                                         0);
                    }

                    if (adj_bound_type_map[LocationType::YPositive] != PDEBoundaryType::Adjacented &&
                        adj_bound_type_map[LocationType::YPositive] != PDEBoundaryType::Null)
                    {
                        copy_y_to_buffer(v_ypos_zpos_corner_map[domain], *v_var->buffer_map[adj_domain][LocationType::YPositive],
                                         0);
                        copy_y_to_buffer(w_ypos_zpos_corner_map[domain], *w_var->buffer_map[adj_domain][LocationType::YPositive],
                                         0);
                    }
                    break;
                default:
                    throw std::runtime_error("PhysicalPESolver3D: invalid location type");
            }
        }
    }
}

void PhysicalPESolver3D::diag_shared_boundary_update()
{
    for (auto* domain : u_var->geometry->domains)
    {
        for (auto& [loc, type] : u_var->boundary_type_map[domain])
        {
            if (type != PDEBoundaryType::Adjacented)
                continue;

            Domain3DUniform* adj_domain         = u_var->geometry->adjacency[domain][loc];
            auto&            adj_bound_type_map = u_var->boundary_type_map[adj_domain];

            // Project to xy

            if (loc == LocationType::XPositive)
            {
                if (adj_bound_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::YNegative];
                    field3&          diag_u      = *u_var->field_map[diag_domain];

                    copy_z_to_buffer(u_var->corner_z_map[domain], diag_u, 0, diag_domain->get_ny() - 1);
                }
                else
                {
                    field2& adj_u_yneg_buffer = *u_var->buffer_map[adj_domain][LocationType::YNegative];
                    copy_x_to_buffer(u_var->corner_z_map[domain], adj_u_yneg_buffer, 0);
                }

                if (adj_bound_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::YPositive];
                    field3&          diag_u      = *u_var->field_map[diag_domain];
                    field3&          diag_v      = *v_var->field_map[diag_domain];

                    copy_z_to_buffer(u_xpos_ypos_corner_map[domain], diag_u, 0, 0);
                    copy_z_to_buffer(v_xpos_ypos_corner_map[domain], diag_v, 0, 0);
                }
            }
            else if (loc == LocationType::YPositive)
            {
                if (adj_bound_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::XNegative];
                    field3&          diag_v      = *v_var->field_map[diag_domain];

                    copy_z_to_buffer(v_var->corner_z_map[domain], diag_v, diag_domain->get_nx() - 1, 0);
                }
                else
                {
                    field2& adj_v_xneg_buffer = *v_var->buffer_map[adj_domain][LocationType::XNegative];
                    copy_x_to_buffer(v_var->corner_z_map[domain], adj_v_xneg_buffer, 0);
                }

                if (adj_bound_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::XPositive];
                    field3&          diag_u      = *u_var->field_map[diag_domain];
                    field3&          diag_v      = *v_var->field_map[diag_domain];

                    copy_z_to_buffer(u_xpos_ypos_corner_map[domain], diag_u, 0, 0);
                    copy_z_to_buffer(v_xpos_ypos_corner_map[domain], diag_v, 0, 0);
                }
            }

            // Project to xz

            if (loc == LocationType::XPositive)
            {
                if (adj_bound_type_map[LocationType::ZNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::ZNegative];
                    field3&          diag_u      = *u_var->field_map[diag_domain];

                    copy_y_to_buffer(u_var->corner_y_map[domain], diag_u, 0, diag_domain->get_nz() - 1);
                }
                else
                {
                    field2& adj_u_zneg_buffer = *u_var->buffer_map[adj_domain][LocationType::ZNegative];
                    copy_x_to_buffer(u_var->corner_y_map[domain], adj_u_zneg_buffer, 0);
                }

                if (adj_bound_type_map[LocationType::ZPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::ZPositive];
                    field3&          diag_u      = *u_var->field_map[diag_domain];
                    field3&          diag_w      = *w_var->field_map[diag_domain];

                    copy_y_to_buffer(u_xpos_zpos_corner_map[domain], diag_u, 0, 0);
                    copy_y_to_buffer(w_xpos_zpos_corner_map[domain], diag_w, 0, 0);
                }
            }
            else if (loc == LocationType::ZPositive)
            {
                if (adj_bound_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::XNegative];
                    field3&          diag_w      = *w_var->field_map[diag_domain];

                    copy_y_to_buffer(w_var->corner_y_map[domain], diag_w, diag_domain->get_nx() - 1, 0);
                }
                else
                {
                    field2& adj_w_xneg_buffer = *w_var->buffer_map[adj_domain][LocationType::XNegative];
                    copy_y_to_buffer(w_var->corner_y_map[domain], adj_w_xneg_buffer, 0);
                }

                if (adj_bound_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::XPositive];
                    field3&          diag_u      = *u_var->field_map[diag_domain];
                    field3&          diag_w      = *w_var->field_map[diag_domain];

                    copy_y_to_buffer(u_xpos_zpos_corner_map[domain], diag_u, 0, 0);
                    copy_y_to_buffer(w_xpos_zpos_corner_map[domain], diag_w, 0, 0);
                }
            }

            // Project to yz

            if (loc == LocationType::YPositive)
            {
                if (adj_bound_type_map[LocationType::ZNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::ZNegative];
                    field3&          diag_v      = *v_var->field_map[diag_domain];

                    copy_x_to_buffer(v_var->corner_x_map[domain], diag_v, 0, diag_domain->get_nz() - 1);
                }
                else
                {
                    field2& adj_v_zneg_buffer = *v_var->buffer_map[adj_domain][LocationType::ZNegative];
                    copy_y_to_buffer(v_var->corner_x_map[domain], adj_v_zneg_buffer, 0);
                }

                if (adj_bound_type_map[LocationType::ZPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::ZPositive];
                    field3&          diag_v      = *v_var->field_map[diag_domain];
                    field3&          diag_w      = *w_var->field_map[diag_domain];

                    copy_x_to_buffer(v_ypos_zpos_corner_map[domain], diag_v, 0, 0);
                    copy_x_to_buffer(w_ypos_zpos_corner_map[domain], diag_w, 0, 0);
                }
            }
            else if (loc == LocationType::ZPositive)
            {
                if (adj_bound_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::YNegative];
                    field3&          diag_w      = *w_var->field_map[diag_domain];

                    copy_x_to_buffer(w_var->corner_x_map[domain], diag_w, 0, diag_domain->get_nz() - 1);
                }
                else
                {
                    field2& adj_w_yneg_buffer = *w_var->buffer_map[adj_domain][LocationType::YNegative];
                    copy_y_to_buffer(w_var->corner_x_map[domain], adj_w_yneg_buffer, 0);
                }

                if (adj_bound_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
                {
                    Domain3DUniform* diag_domain = u_var->geometry->adjacency[adj_domain][LocationType::YPositive];
                    field3&          diag_v      = *v_var->field_map[diag_domain];
                    field3&          diag_w      = *w_var->field_map[diag_domain];

                    copy_x_to_buffer(v_ypos_zpos_corner_map[domain], diag_v, 0, 0);
                    copy_x_to_buffer(w_ypos_zpos_corner_map[domain], diag_w, 0, 0);
                }
            }
        }
    }
}
