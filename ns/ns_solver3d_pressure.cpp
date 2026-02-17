#include "boundary_3d_utils.h"
#include "ns_solver3d.h"

#include <iomanip>

void ConcatNSSolver3D::velocity_div_inner()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx - 1; i++)
            for (int j = 0; j < ny - 1; j++)
                for (int k = 0; k < nz - 1; k++)
                    p(i, j, k) = (u(i + 1, j, k) - u(i, j, k)) / hx + (v(i, j + 1, k) - v(i, j, k)) / hy +
                                 (w(i, j, k + 1) - w(i, j, k)) / hz;
    }
}

void ConcatNSSolver3D::velocity_div_outer()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];

        field2& u_buffer_right = *u_buffer_map[domain][LocationType::Right];
        field2& v_buffer_back  = *v_buffer_map[domain][LocationType::Back];
        field2& w_buffer_up    = *w_buffer_map[domain][LocationType::Up];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx - 1; i++)
            for (int j = 0; j < ny - 1; j++)
                p(i, j, nz - 1) = (u(i + 1, j, nz - 1) - u(i, j, nz - 1)) / hx +
                                  (v(i, j + 1, nz - 1) - v(i, j, nz - 1)) / hy +
                                  (w_buffer_up(i, j) - w(i, j, nz - 1)) / hz;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx - 1; i++)
            for (int k = 0; k < nz - 1; k++)
                p(i, ny - 1, k) = (u(i + 1, ny - 1, k) - u(i, ny - 1, k)) / hx +
                                  (v_buffer_back(i, k) - v(i, ny - 1, k)) / hy +
                                  (w(i, ny - 1, k + 1) - w(i, ny - 1, k)) / hz;

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny - 1; j++)
            for (int k = 0; k < nz - 1; k++)
                p(nx - 1, j, k) = (u_buffer_right(j, k) - u(nx - 1, j, k)) / hx +
                                  (v(nx - 1, j + 1, k) - v(nx - 1, j, k)) / hy +
                                  (w(nx - 1, j, k + 1) - w(nx - 1, j, k)) / hz;

        for (int i = 0; i < nx - 1; i++)
            p(i, ny - 1, nz - 1) = (u(i + 1, ny - 1, nz - 1) - u(i, ny - 1, nz - 1)) / hx +
                                   (v_buffer_back(i, nz - 1) - v(i, ny - 1, nz - 1)) / hy +
                                   (w_buffer_up(i, ny - 1) - w(i, ny - 1, nz - 1)) / hz;

        for (int j = 0; j < ny - 1; j++)
            p(nx - 1, j, nz - 1) = (u_buffer_right(j, nz - 1) - u(nx - 1, j, nz - 1)) / hx +
                                   (v(nx - 1, j + 1, nz - 1) - v(nx - 1, j, nz - 1)) / hy +
                                   (w_buffer_up(nx - 1, j) - w(nx - 1, j, nz - 1)) / hz;

        for (int k = 0; k < nz - 1; k++)
            p(nx - 1, ny - 1, k) = (u_buffer_right(ny - 1, k) - u(nx - 1, ny - 1, k)) / hx +
                                   (v_buffer_back(nx - 1, k) - v(nx - 1, ny - 1, k)) / hy +
                                   (w(nx - 1, ny - 1, k + 1) - w(nx - 1, ny - 1, k)) / hz;

        p(nx - 1, ny - 1, nz - 1) = (u_buffer_right(ny - 1, nz - 1) - u(nx - 1, ny - 1, nz - 1)) / hx +
                                    (v_buffer_back(nx - 1, nz - 1) - v(nx - 1, ny - 1, nz - 1)) / hy +
                                    (w_buffer_up(nx - 1, nx - 1) - w(nx - 1, ny - 1, nz - 1)) / hz;
    }
}

void ConcatNSSolver3D::pressure_buffer_update()
{
    // Only adjacented boundaries
    for (auto& domain : domains)
    {
        field3& p = *p_field_map[domain];

        int nx = p.get_nx();
        int ny = p.get_ny();
        int nz = p.get_nz();

        for (auto& [loc, type] : p_var->boundary_type_map[domain])
        {
            if (type == PDEBoundaryType::Adjacented)
            {
                field2& p_buffer = *p_buffer_map[domain][loc];

                Domain3DUniform* adj_domain = adjacency[domain][loc];
                field3&          adj_p      = *p_field_map[adj_domain];
                int              adj_nx     = adj_p.get_nx();
                int              adj_ny     = adj_p.get_ny();
                int              adj_nz     = adj_p.get_nz();
                switch (loc)
                {
                    case LocationType::Left:
                        copy_x_to_buffer(p_buffer, adj_p, adj_nx - 1);
                        break;
                    case LocationType::Front:
                        copy_y_to_buffer(p_buffer, adj_p, adj_ny - 1);
                        break;
                    case LocationType::Down:
                        copy_z_to_buffer(p_buffer, adj_p, adj_nz - 1);
                        break;
                    default:
                        break;
                }
            }
        }
    }
}

void ConcatNSSolver3D::add_pressure_gradient()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];

        field2& p_buffer_left  = *p_buffer_map[domain][LocationType::Left];
        field2& p_buffer_front = *p_buffer_map[domain][LocationType::Front];
        field2& p_buffer_down  = *p_buffer_map[domain][LocationType::Down];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    u(i, j, k) -= (p(i, j, k) - p(i - 1, j, k)) / hx;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
            for (int j = 1; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    v(i, j, k) -= (p(i, j, k) - p(i, j - 1, k)) / hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 1; k < nz; k++)
                    w(i, j, k) -= (p(i, j, k) - p(i, j, k - 1)) / hz;

        if (u_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Adjacented)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    u(0, j, k) -= (p(0, j, k) - p_buffer_left(j, k)) / hx;

        if (u_var->boundary_type_map[domain][LocationType::Front] == PDEBoundaryType::Adjacented)
            for (int i = 0; i < nx; i++)
                for (int k = 0; k < nz; k++)
                    v(i, 0, k) -= (p(i, 0, k) - p_buffer_front(i, k)) / hy;

        if (u_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Adjacented)
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    w(i, j, 0) -= (p(i, j, 0) - p_buffer_down(i, j)) / hz;
    }
}

void ConcatNSSolver3D::normalize_pressure()
{
    double total_sum  = 0.0;
    size_t total_size = 0;
    for (auto& domain : domains)
    {
        field3& p = *p_field_map[domain];
        total_sum += p.sum();
        total_size += p.get_size_n();
    }
    double mean = total_sum / total_size;
    for (auto& domain : domains)
    {
        field3& p  = *p_field_map[domain];
        int     nx = p.get_nx();
        int     ny = p.get_ny();
        int     nz = p.get_nz();
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                    p(i, j, k) -= mean;
    }
}