#include "physical_pe_solver3d.h"

#include "boundary_2d_utils.h"
#include "boundary_3d_utils.h"

void PhysicalPESolver3D::nondiag_shared_boundary_update()
{
    // redirect in convenience of using ns code
    auto& u_field_map = c_u_map;
    auto& v_field_map = c_v_map;
    auto& w_field_map = c_w_map;

    auto& domains   = u_var->geometry->domains;
    auto& adjacency = u_var->geometry->adjacency;
    for (auto& domain : domains)
    {
        // redirect in convenience of using ns code
        field3& u = *c_u_map[domain];
        field3& v = *c_v_map[domain];
        field3& w = *c_w_map[domain];

        // redirect in convenience of using ns code
        auto& u_buffer_map = c_u_buffer_map;
        auto& v_buffer_map = c_v_buffer_map;
        auto& w_buffer_map = c_w_buffer_map;

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