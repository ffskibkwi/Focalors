#include "ns_solver2d.h"

void ConcatNSSolver2D::pressure_calculate()
{
    velocity_div_calculate();
    p_solver->solve();
    pressure_buffer_pass();
    velocity_update();
}

void ConcatNSSolver2D::velocity_div_calculate()
{
    velocity_div_inner();
    velocity_div_outer();
}

void ConcatNSSolver2D::velocity_div_inner()
{
    for (auto &domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& p = *p_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny; j++)
                p(i, j) = (u(i, j) - u(i - 1, j)) / hx + (v(i, j) - v(i, j - 1)) / hy;
        }
    }
}

void ConcatNSSolver2D::velocity_div_outer()
{
    for (auto &domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& p = *p_field_map[domain];

        double* u_buffer_right = u_buffer_map[domain][LocationType::Right];
        double* v_buffer_up = v_buffer_map[domain][LocationType::Up];

        int nx = u.get_nx();
        int ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx - 1; i++)
        {
            for (int j = 0; j < ny - 1; j++)
                p(i, j) = (u(i + 1, j) - u(i, j)) / hx + (v(i, j + 1) - v(i, j)) / hy;
        }
        
        for (int i = 0; i < nx - 1; i++)
            p(i, ny - 1) = (u(i + 1, ny - 1) - u(i, ny - 1)) / hx + (v_buffer_up[i] - v(i, ny - 1)) / hy;
        
        for (int j = 0; j < ny - 1; j++)
            p(nx - 1, j) = (u_buffer_right[j] - u(nx - 1, j)) / hx + (v(nx - 1, j + 1) - v(nx - 1, j)) / hy;
        
        p(nx - 1, ny - 1) = (u_buffer_right[ny - 1] - u(nx - 1, ny - 1)) / hx + (v_buffer_up[nx - 1] - v(nx - 1, ny - 1)) / hy;
    }
}

void ConcatNSSolver2D::velocity_update()
{
    for (auto &domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& p = *p_field_map[domain];

        double* p_buffer_down = u_buffer_map[domain][LocationType::Right];
        double* p_buffer_left = v_buffer_map[domain][LocationType::Up];

        int nx = u.get_nx();
        int ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx; i++)
        {
            for (int j = 1; j < ny; j++)
            {
                u(i, j) = u(i, j) - (p(i, j) - p(i - 1, j)) / hx;
                v(i, j) = v(i, j) - (p(i, j) - p(i, j - 1)) / hy;
            }
        }

        if (u_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Adjacented)
        {
            for (int i = 1; i < nx; i++)
            {
                u(i, 0) = u(i, 0) - (p(i, 0) - p(i - 1, 0)) / hx;
                v(i, 0) = v(i, 0) - (p(i, 0) - p_buffer_down[i]) / hy;
            }
            v(0, 0) = v(0, 0) - (p(0, 0) - p_buffer_down[0]) / hy;
        }        

        if (u_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Adjacented)
        {
            for (int j = 1; j < ny; j++)
            {
                u(0, j) = u(0, j) - (p(0, j) - p_buffer_left[j]) / hx;
                v(0, j) = v(0, j) - (p(0, j) - p(0, j - 1)) / hy;
            }
            u(0, 0) = u(0, 0) - (p(0, 0) - p_buffer_left[0]) / hx;
        }
    }
}

void ConcatNSSolver2D::pressure_buffer_pass()
{
    //Only adjacented boundaries
    for (auto &domain : domains)
    {
        field2& p = *p_field_map[domain];

        int nx = p.get_nx();
        int ny = p.get_ny();

        for (auto &[loc, type] : p_var->boundary_type_map[domain])
        {
            if (type == PDEBoundaryType::Adjacented)
            {
                double* p_buffer = p_buffer_map[domain][loc];

                Domain2DUniform* adj_domain = adjacency[domain][loc];
                field2& adj_p = *p_field_map[adj_domain];
                switch (loc)
                {
                    case LocationType::Left:
                        for (int j = 0; j < ny; j++)
                            p_buffer[j] = adj_p(nx - 1, j);
                        break;
                    case LocationType::Down:
                        for (int i = 0; i < nx; i++)
                            p_buffer[i] = adj_p(i, ny - 1);
                        break;
                    default:
                        throw std::runtime_error("ConcatNSSolver2D: invalid location type");
                }
            }
        }
    }
}