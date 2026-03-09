#include "ib_solver_2d.h"

#include <cmath>

ImmersedBoundarySolver2D::ImmersedBoundarySolver2D(Variable2D*                                      _u_var,
                                                   Variable2D*                                      _v_var,
                                                   std::unordered_map<Domain2DUniform*, PCoord2D*>& _coord_map)
    : u_var(_u_var)
    , v_var(_v_var)
    , coord_map(_coord_map)
{
    for (auto* domain : u_var->geometry->domains)
    {
        ib_map[domain] = new PIB2D(coord_map[domain]->max_n);
    }

    // Setup domain context accessor
    get_domain_context = [&](Domain2DUniform* domain) -> DomainContext {
        auto& u = *u_var->field_map[domain];
        auto& v = *v_var->field_map[domain];

        auto* u_buffer_x_neg = u_var->buffer_map[domain][LocationType::XNegative];
        auto* u_buffer_x_pos = u_var->buffer_map[domain][LocationType::XPositive];
        auto* u_buffer_y_neg = u_var->buffer_map[domain][LocationType::YNegative];
        auto* u_buffer_y_pos = u_var->buffer_map[domain][LocationType::YPositive];

        auto* v_buffer_x_neg = v_var->buffer_map[domain][LocationType::XNegative];
        auto* v_buffer_x_pos = v_var->buffer_map[domain][LocationType::XPositive];
        auto* v_buffer_y_neg = v_var->buffer_map[domain][LocationType::YNegative];
        auto* v_buffer_y_pos = v_var->buffer_map[domain][LocationType::YPositive];

        return DomainContext {domain,
                              [&](int i, int j) -> double {
                                  // u is x-face centered; allow i=-1/nx (x buffers) and j=-1/ny (y buffers)
                                  if (i == -1)
                                  {
                                      j = std::clamp(j, 0, u.get_ny() - 1);
                                      return u_buffer_x_neg[j];
                                  }
                                  else if (i == u.get_nx())
                                  {
                                      j = std::clamp(j, 0, u.get_ny() - 1);
                                      return u_buffer_x_pos[j];
                                  }
                                  else if (j == -1)
                                  {
                                      i = std::clamp(i, 0, u.get_nx() - 1);
                                      return u_buffer_y_neg[i];
                                  }
                                  else if (j == u.get_ny())
                                  {
                                      i = std::clamp(i, 0, u.get_nx() - 1);
                                      return u_buffer_y_pos[i];
                                  }
                                  else
                                  {
                                      return u(i, j);
                                  }
                              },
                              [&](int i, int j) -> double {
                                  // v is y-face centered; allow j=-1/ny (y buffers) and i=-1/nx (x buffers)
                                  if (j == -1)
                                  {
                                      i = std::clamp(i, 0, v.get_nx() - 1);
                                      return v_buffer_y_neg[i];
                                  }
                                  else if (j == v.get_ny())
                                  {
                                      i = std::clamp(i, 0, v.get_nx() - 1);
                                      return v_buffer_y_pos[i];
                                  }
                                  else if (i == -1)
                                  {
                                      j = std::clamp(j, 0, v.get_ny() - 1);
                                      return v_buffer_x_neg[j];
                                  }
                                  else if (i == v.get_nx())
                                  {
                                      j = std::clamp(j, 0, v.get_ny() - 1);
                                      return v_buffer_x_pos[j];
                                  }
                                  else
                                  {
                                      return v(i, j);
                                  }
                              }};
    };
}

void ImmersedBoundarySolver2D::solve()
{
    calc_ib_force();
    apply_ib_force();
}

double& ImmersedBoundarySolver2D::get_u_value(Domain2DUniform* domain, int iix, int iiy)
{
    // Compute global position of this u-face
    double px = iix * grid_h;
    double py = iiy * grid_h + 0.5 * grid_h;

    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();

    // Helper lambda: try map a global position to a cell in given domain for u
    auto try_map_u = [&](Domain2DUniform* d, double gx, double gy, double*& field_ptr, int& li, int& lj) -> bool {
        double hx = d->get_hx();
        double hy = d->get_hy();

        double local_x = gx - d->get_offset_x();
        double local_y = gy - d->get_offset_y();

        int ui = static_cast<int>(std::floor(local_x / hx));
        int uj = static_cast<int>(std::floor((local_y - 0.5 * hy) / hy));

        auto& u = *u_var->field_map[d];
        if (ui >= 0 && ui < u.get_nx() && uj >= 0 && uj < u.get_ny())
        {
            field_ptr = &u(ui, uj);
            li        = ui;
            lj        = uj;
            return true;
        }
        return false;
    };

    // First try current domain
    {
        double* cell = nullptr;
        int     li   = 0;
        int     lj   = 0;
        if (try_map_u(domain, global_x, global_y, cell, li, lj))
        {
            return *cell;
        }
    }

    // Then try neighbor domains via adjacency
    if (u_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : u_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double* cell = nullptr;
            int     li   = 0;
            int     lj   = 0;
            if (try_map_u(other_domain, global_x, global_y, cell, li, lj))
            {
                return *cell;
            }
        }
    }

    static double zero = 0.0;
    return zero;
}

double& ImmersedBoundarySolver2D::get_v_value(Domain2DUniform* domain, int iix, int iiy)
{
    // Compute global position of this v-face
    double px = iix * grid_h + 0.5 * grid_h;
    double py = iiy * grid_h;

    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();

    // Helper lambda: try map a global position to a cell in given domain for v
    auto try_map_v = [&](Domain2DUniform* d, double gx, double gy, double*& field_ptr, int& li, int& lj) -> bool {
        double hx = d->get_hx();
        double hy = d->get_hy();

        double local_x = gx - d->get_offset_x();
        double local_y = gy - d->get_offset_y();

        int vi = static_cast<int>(std::floor((local_x - 0.5 * hx) / hx));
        int vj = static_cast<int>(std::floor(local_y / hy));

        auto& v = *v_var->field_map[d];
        if (vi >= 0 && vi < v.get_nx() && vj >= 0 && vj < v.get_ny())
        {
            field_ptr = &v(vi, vj);
            li        = vi;
            lj        = vj;
            return true;
        }
        return false;
    };

    // First try current domain
    {
        double* cell = nullptr;
        int     li   = 0;
        int     lj   = 0;
        if (try_map_v(domain, global_x, global_y, cell, li, lj))
        {
            return *cell;
        }
    }

    // Then try neighbor domains via adjacency
    if (v_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : v_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double* cell = nullptr;
            int     li   = 0;
            int     lj   = 0;
            if (try_map_v(other_domain, global_x, global_y, cell, li, lj))
            {
                return *cell;
            }
        }
    }

    static double zero = 0.0;
    return zero;
}

void ImmersedBoundarySolver2D::calc_ib_force()
{
    // Process each domain in the geometry tree
    for (auto* domain : u_var->geometry->domains)
    {
        auto* particles = coord_map[domain];
        auto* ib_data   = ib_map[domain];

        EXPOSE_PCOORD2D(particles)
        EXPOSE_PIB2D(ib_data)

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < particles->cur_n; i++)
        {
            // Convert particle position to grid indices (local coordinates)
            int ix = std::floor(X[i] / grid_h);
            int iy = std::floor(Y[i] / grid_h);

            int min_iix = std::clamp(ix - 1, 0, u_var->field_map[domain]->get_nx() - 1);
            int max_iix = std::clamp(ix + 2, 0, u_var->field_map[domain]->get_nx() - 1);
            int min_iiy = std::clamp(iy - 2, -1, u_var->field_map[domain]->get_ny());
            int max_iiy = std::clamp(iy + 2, -1, u_var->field_map[domain]->get_ny());

            Uf[i] = 0.0;
            for (int iix = min_iix; iix <= max_iix; iix++)
            {
                for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                {
                    double xi      = iix * grid_h;
                    double yi      = iiy * grid_h + 0.5 * grid_h;
                    double u_value = get_u_value(domain, iix, iiy);

                    Uf[i] += u_value * ib_delta(X[i] - xi, Y[i] - yi, grid_h) * grid_h * grid_h;
                }
            }
            Fx[i] = Up[i] - Uf[i];
            Fx_sum[i] += Fx[i];

            min_iix = std::clamp(ix - 2, -1, v_var->field_map[domain]->get_nx());
            max_iix = std::clamp(ix + 2, -1, v_var->field_map[domain]->get_nx());
            min_iiy = std::clamp(iy - 1, 0, v_var->field_map[domain]->get_ny() - 1);
            max_iiy = std::clamp(iy + 2, 0, v_var->field_map[domain]->get_ny() - 1);

            Vf[i] = 0.0;
            for (int iix = min_iix; iix <= max_iix; iix++)
            {
                for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                {
                    double xi      = iix * grid_h + 0.5 * grid_h;
                    double yi      = iiy * grid_h;
                    double v_value = get_v_value(domain, iix, iiy);

                    Vf[i] += v_value * ib_delta(X[i] - xi, Y[i] - yi, grid_h) * grid_h * grid_h;
                }
            }
            Fy[i] = Vp[i] - Vf[i];
            Fy_sum[i] += Fy[i];
        }
    }
}

void ImmersedBoundarySolver2D::apply_ib_force()
{
    // Process each domain in the geometry tree
    for (auto* domain : u_var->geometry->domains)
    {
        auto* particles = coord_map[domain];
        auto* ib_data   = ib_map[domain];

        EXPOSE_PCOORD2D(particles)
        EXPOSE_PIB2D(ib_data)

        auto& u_recv = *u_var->field_map[domain];
        auto& v_recv = *v_var->field_map[domain];

        if (particles->cur_n == 0)
        {
            continue;
        }

        // 使用 PCoord 中缓存的全局 bounding box（2h 支持域索引允许跨越 domain，不 clamp）
        double min_X = particles->min_X;
        double max_X = particles->max_X;
        double min_Y = particles->min_Y;
        double max_Y = particles->max_Y;

        int min_ix_u = static_cast<int>(std::floor(min_X / grid_h)) - 1;
        int max_ix_u = static_cast<int>(std::floor(max_X / grid_h)) + 2;
        int min_iy_u = static_cast<int>(std::floor(min_Y / grid_h)) - 2;
        int max_iy_u = static_cast<int>(std::floor(max_Y / grid_h)) + 2;

        int min_ix_v = static_cast<int>(std::floor(min_X / grid_h)) - 2;
        int max_ix_v = static_cast<int>(std::floor(max_X / grid_h)) + 2;
        int min_iy_v = static_cast<int>(std::floor(min_Y / grid_h)) - 1;
        int max_iy_v = static_cast<int>(std::floor(max_Y / grid_h)) + 2;

        // u
        OPENMP_PARALLEL_FOR()
        for (int i = min_ix_u; i <= max_ix_u; i++)
        {
            for (int j = min_iy_u; j <= max_iy_u; j++)
            {
                double xx = i * grid_h;
                double yy = j * grid_h + 0.5 * grid_h;

                for (int ib = 0; ib < particles->cur_n; ib++)
                {
                    double ib_force = Fx[ib] * ib_delta(xx - X[ib], yy - Y[ib], grid_h) * ib_h * grid_h;
                    get_u_value(domain, i, j) += ib_force;
                }
            }
        }

        // v
        OPENMP_PARALLEL_FOR()
        for (int i = min_ix_v; i <= max_ix_v; i++)
        {
            for (int j = min_iy_v; j <= max_iy_v; j++)
            {
                double xx = i * grid_h + 0.5 * grid_h;
                double yy = j * grid_h;

                for (int ib = 0; ib < particles->cur_n; ib++)
                {
                    double ib_force = Fy[ib] * ib_delta(xx - X[ib], yy - Y[ib], grid_h) * ib_h * grid_h;
                    get_v_value(domain, i, j) += ib_force;
                }
            }
        }
    }
}