#include "physical_pe_solver3d.h"

#include "boundary_2d_utils.h"
#include "boundary_3d_utils.h"

void PhysicalPESolver3D::diag_shared_boundary_update()
{
    for (auto& domain : u_var->geometry->domains)
    {
        auto& bound_type_map = u_var->boundary_type_map[domain];

        for (auto& [loc, type] : bound_type_map)
        {
            if (type == PDEBoundaryType::Adjacented)
            {
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

                        copy_z_to_buffer(u_xpos_ypos_corner_map[domain], diag_u, 0, 0);
                    }
                    else
                    {
                        field2& adj_u_ypos_buffer = *u_var->buffer_map[adj_domain][LocationType::YPositive];

                        copy_x_to_buffer(u_xpos_ypos_corner_map[domain], adj_u_ypos_buffer, 0);
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
                        field3&          diag_v      = *v_var->field_map[diag_domain];

                        copy_z_to_buffer(v_xpos_ypos_corner_map[domain], diag_v, 0, 0);
                    }
                    else
                    {
                        field2& adj_v_xpos_buffer = *v_var->buffer_map[adj_domain][LocationType::XPositive];

                        copy_x_to_buffer(v_xpos_ypos_corner_map[domain], adj_v_xpos_buffer, 0);
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

                        copy_y_to_buffer(u_xpos_zpos_corner_map[domain], diag_u, 0, 0);
                    }
                    else
                    {
                        field2& adj_u_zpos_buffer = *u_var->buffer_map[adj_domain][LocationType::ZPositive];

                        copy_x_to_buffer(u_xpos_zpos_corner_map[domain], adj_u_zpos_buffer, 0);
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
                        field3&          diag_w      = *w_var->field_map[diag_domain];

                        copy_y_to_buffer(w_xpos_zpos_corner_map[domain], diag_w, 0, 0);
                    }
                    else
                    {
                        field2& adj_w_xpos_buffer = *w_var->buffer_map[adj_domain][LocationType::XPositive];

                        copy_y_to_buffer(w_xpos_zpos_corner_map[domain], adj_w_xpos_buffer, 0);
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

                        copy_x_to_buffer(v_ypos_zpos_corner_map[domain], diag_v, 0, 0);
                    }
                    else
                    {
                        field2& adj_v_zpos_buffer = *v_var->buffer_map[adj_domain][LocationType::ZPositive];

                        copy_y_to_buffer(v_ypos_zpos_corner_map[domain], adj_v_zpos_buffer, 0);
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
                        field3&          diag_w      = *w_var->field_map[diag_domain];

                        copy_x_to_buffer(w_ypos_zpos_corner_map[domain], diag_w, 0, 0);
                    }
                    else
                    {
                        field2& adj_w_ypos_buffer = *w_var->buffer_map[adj_domain][LocationType::YPositive];

                        copy_y_to_buffer(w_ypos_zpos_corner_map[domain], adj_w_ypos_buffer, 0);
                    }
                }
            }
        }
    }
}
