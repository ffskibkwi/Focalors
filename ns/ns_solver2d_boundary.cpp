#include "ns_solver2d.h"

void ConcatNSSolver2D::buffer_pass()
{
    //Only adjacented boundaries
    for (auto &domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& p = *p_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        for (auto &[loc, type] : u_var->boundary_type_map[domain])
        {
            //While u has adjacented boundary, it means v and p also have adjacented boundary
            if (type == PDEBoundaryType::Adjacented)
            {
                double* u_buffer = u_buffer_map[domain][loc];
                double* v_buffer = v_buffer_map[domain][loc];
                double* p_buffer = p_buffer_map[domain][loc];

                Domain2DUniform* adj_domain = adjacency[domain][loc];
                field2& adj_u = *u_field_map[adj_domain];
                field2& adj_v = *v_field_map[adj_domain];
                field2& adj_p = *p_field_map[adj_domain];
                switch (loc)
                {
                    case LocationType::Left:
                        for (int j = 0; j < ny; j++)
                        {
                            u_buffer[j] = adj_u(nx - 1, j);
                            v_buffer[j] = adj_v(nx - 1, j);
                            p_buffer[j] = adj_p(nx - 1, j);
                        }
                        break;
                    case LocationType::Right:
                        for (int j = 0; j < ny; j++)
                        {
                            u_buffer[j] = adj_u(0, j);
                            v_buffer[j] = adj_v(0, j);
                        }
                        break;
                    case LocationType::Down:
                        for (int i = 0; i < nx; i++)
                        {
                            u_buffer[i] = adj_u(i, ny - 1);
                            v_buffer[i] = adj_v(i, ny - 1);
                            p_buffer[i] = adj_p(i, ny - 1);
                        }
                        break;
                    case LocationType::Up:
                        for (int i = 0; i < nx; i++)
                        {
                            u_buffer[i] = adj_u(i, 0);
                            v_buffer[i] = adj_v(i, 0);
                        }
                        break;
                    default:
                        throw std::runtime_error("ConcatNSSolver2D: invalid location type");
                }
            }
        }
        
    }
}

void ConcatNSSolver2D::boundary_init()
{
    //It only be used once at the start of time advancing
    //Including u-left/right v-down/up
    for (auto &domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        for (auto &[loc, type] : u_var->boundary_type_map[domain])
        {
            if (loc == LocationType::Left)
            {
                //left-u boundary is at physical boundary
                if (type == PDEBoundaryType::Dirichlet)
                {
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = u_var->boundary_value_map[domain][loc];
                        for (int j = 0; j < ny; j++)
                            u(0, j) = boundary_value[j];
                    }else
                    {
                        for (int j = 0; j < ny; j++)
                            u(0, j) = 0.0;
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    throw std::runtime_error("ConcatNSSolver2D: left-u boundary is not supported for Neumann boundary");
                }else if (type == PDEBoundaryType::Periodic)
                {
                    throw std::runtime_error("ConcatNSSolver2D: left-u boundary is not supported for Periodic boundary (Under development)");
                }
            }

            if (loc == LocationType::Right)
            {
                //right-u boundary is at physical boundary
                if (type == PDEBoundaryType::Dirichlet)
                {
                    double* u_buffer_right = u_buffer_map[domain][loc];
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = u_var->boundary_value_map[domain][loc];
                        for (int j = 0; j < ny; j++)
                            u_buffer_right[j] = boundary_value[j];
                    }else
                    {
                        for (int j = 0; j < ny; j++)
                            u_buffer_right[j] = 0.0;
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    double* u_buffer_right = u_buffer_map[domain][loc];
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        throw std::runtime_error("ConcatNSSolver2D: right-u Neumann boundary should not set value");
                    }else
                    {
                        for (int j = 0; j < ny; j++)
                            u_buffer_right[j] = u(nx - 1, j);
                    }
                }else if (type == PDEBoundaryType::Periodic)
                {
                    throw std::runtime_error("ConcatNSSolver2D: right-u boundary is not supported for Periodic boundary (Under development)");
                }
            }
        }

        for (auto &[loc, type] : v_var->boundary_type_map[domain])
        {
            if (loc == LocationType::Up)
            {
                //up-v boundary is at physical boundary
                if (type == PDEBoundaryType::Dirichlet)
                {
                    double* v_buffer_up = v_buffer_map[domain][loc];
                    if (v_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = v_var->boundary_value_map[domain][loc];
                        for (int i = 0; i < nx; i++)
                            v_buffer_up[i] = boundary_value[i];
                    }else
                    {
                        for (int i = 0; i < nx; i++)
                            v_buffer_up[i] = 0.0;
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    double* v_buffer_up = v_buffer_map[domain][loc];
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        throw std::runtime_error("ConcatNSSolver2D: up-v Neumann boundary should not set value");
                    }else
                    {
                        for (int i = 0; i < nx; i++)
                            v_buffer_up[i] = v(i, ny - 1);
                    }
                }else if (type == PDEBoundaryType::Periodic)
                {
                    throw std::runtime_error("ConcatNSSolver2D: up-v boundary is not supported for Periodic boundary (Under development)");
                }
            }

            if (loc == LocationType::Down)
            {
                //down-v boundary is at physical boundary
                if (type == PDEBoundaryType::Dirichlet)
                {
                    if (v_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = v_var->boundary_value_map[domain][loc];
                        for (int i = 0; i < nx; i++)
                            v(i, 0) = boundary_value[i];
                    }else
                    {
                        for (int i = 0; i < nx; i++)
                            v(i, 0) = 0.0;
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    throw std::runtime_error("ConcatNSSolver2D: down-v boundary is not supported for Neumann boundary");
                }else if (type == PDEBoundaryType::Periodic)
                {
                    throw std::runtime_error("ConcatNSSolver2D: down-v boundary is not supported for Periodic boundary (Under development)");
                }
            }
        }
    }
}

void ConcatNSSolver2D::boundary_update()
{
    //It is used in every time step
    //Including u-up/down v-left/right
    //The corner node is calcualted here

    for (auto &domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();

        for (auto &[loc, type] : u_var->boundary_type_map[domain])
        {
            if (loc == LocationType::Up)
            {
                if (type == PDEBoundaryType::Dirichlet)
                {
                    double* u_buffer_up = u_buffer_map[domain][loc];
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = u_var->boundary_value_map[domain][loc];
                        for (int i = 0; i < nx; i++)
                            u_buffer_up[i] = 2.0 * boundary_value[i] - u(i, ny - 1);
                    }else
                    {
                        for (int i = 0; i < nx; i++)
                            u_buffer_up[i] = -1.0 * u(i, ny - 1);
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    double* u_buffer_up = u_buffer_map[domain][loc];
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        throw std::runtime_error("ConcatNSSolver2D: up-u Neumann boundary should not set value");
                    }else
                    {
                        for (int i = 0; i < nx; i++)
                            u_buffer_up[i] = u(i, ny - 1);
                    }
                }else if (type == PDEBoundaryType::Periodic)
                {
                    throw std::runtime_error("ConcatNSSolver2D: up-u boundary is not supported for Periodic boundary (Under development)");
                }
            }else if (loc == LocationType::Down)
            {
                if (type == PDEBoundaryType::Dirichlet)
                {
                    double* u_buffer_down = u_buffer_map[domain][loc];
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = u_var->boundary_value_map[domain][loc];
                        for (int i = 0; i < nx; i++)
                            u_buffer_down[i] = 2.0 * boundary_value[i] - u(i, 0);
                    }else
                    {
                        for (int i = 0; i < nx; i++)
                            u_buffer_down[i] = -1.0 * u(i, 0);
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    double* u_buffer_down = u_buffer_map[domain][loc];
                    if (u_var->has_boundary_value_map[domain][loc])
                    {
                        throw std::runtime_error("ConcatNSSolver2D: down-u Neumann boundary should not set value");
                    }else
                    {
                        for (int i = 0; i < nx; i++)
                            u_buffer_down[i] = u(i, 0);
                    }
                }else if (type == PDEBoundaryType::Periodic)
                {
                    throw std::runtime_error("ConcatNSSolver2D: down-u boundary is not supported for Periodic boundary (Under development)");
                }
            }
        }

        for (auto &[loc, type] : v_var->boundary_type_map[domain])
        {
            if (loc == LocationType::Left)
            {
                if (type == PDEBoundaryType::Dirichlet)
                {
                    double* v_buffer_left = v_buffer_map[domain][loc];
                    if (v_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = v_var->boundary_value_map[domain][loc];
                        for (int j = 0; j < ny; j++)
                            v_buffer_left[j] = 2.0 * boundary_value[j] - v(0, j);
                    }else
                    {
                        for (int j = 0; j < ny; j++)
                            v_buffer_left[j] = -1.0 * v(0, j);
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    if (v_var->has_boundary_value_map[domain][loc])
                    {
                        throw std::runtime_error("ConcatNSSolver2D: left-v Neumann boundary should not set value");
                    }else
                    {
                        double* v_buffer_left = v_buffer_map[domain][loc];
                        for (int j = 0; j < ny; j++)
                            v_buffer_left[j] = v(0, j);
                    }
                }
            }else if (loc == LocationType::Right)
            {
                if (type == PDEBoundaryType::Dirichlet)
                {
                    double* v_buffer_right = v_buffer_map[domain][loc];
                    if (v_var->has_boundary_value_map[domain][loc])
                    {
                        double* boundary_value = v_var->boundary_value_map[domain][loc];
                        for (int j = 0; j < ny; j++)
                            v_buffer_right[j] = 2.0 * boundary_value[j] - v(nx - 1, j);
                    }else
                    {
                        for (int j = 0; j < ny; j++)
                            v_buffer_right[j] = -1.0 * v(nx - 1, j);
                    }
                }else if (type == PDEBoundaryType::Neumann)
                {
                    if (v_var->has_boundary_value_map[domain][loc])
                    {
                        throw std::runtime_error("ConcatNSSolver2D: right-v Neumann boundary should not set value");
                    }else
                    {
                        double* v_buffer_right = v_buffer_map[domain][loc];
                        for (int j = 0; j < ny; j++)
                            v_buffer_right[j] = v(nx - 1, j);
                    }
                }
            }
        }

        //Corner nodes
        if (u_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Dirichlet)
        {    
            double right_down_corner_value = u_var->right_down_corner_boundary_map[domain];
            right_down_corner_value_map[domain] =  2.0 * right_down_corner_value - u_buffer_map[domain][LocationType::Down][0];
        }else if (u_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Neumann)
        {
            right_down_corner_value_map[domain] =  u_buffer_map[domain][LocationType::Down][0];
        }else if (u_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Periodic)
        {
            throw std::runtime_error("ConcatNSSolver2D (corner): down-u boundary is not supported for Periodic boundary (Under development)");
        }else if (u_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Adjacented)
        {
            Domain2DUniform* adj_domain = adjacency[domain][LocationType::Left];
            left_up_corner_value_map[domain] = u_buffer_map[adj_domain][LocationType::Up][adj_domain->get_nx() - 1];
        }
        
        if (v_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Dirichlet)
        {
           double left_up_corner_value = v_var->left_up_corner_boundary_map[domain];
           left_up_corner_value_map[domain] =  2.0 * left_up_corner_value - v_buffer_map[domain][LocationType::Left][0];
        }else if (v_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Neumann)
        {
            left_up_corner_value_map[domain] =  v_buffer_map[domain][LocationType::Left][0];
        }else if (v_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Periodic)
        {
            throw std::runtime_error("ConcatNSSolver2D (corner): left-v boundary is not supported for Periodic boundary (Under development)");
        }else if (v_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Adjacented)
        {
            Domain2DUniform* adj_domain = adjacency[domain][LocationType::Left];
            left_up_corner_value_map[domain] = v_buffer_map[adj_domain][LocationType::Right][adj_domain->get_ny() - 1];
        }
    }
}