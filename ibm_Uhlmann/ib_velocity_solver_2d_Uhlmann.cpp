#include "ib_velocity_solver_2d_Uhlmann.h"

#include <cmath>

IBVelocitySolver2D_Uhlmann::IBVelocitySolver2D_Uhlmann(Variable2D*                                      _u_var,
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

void IBVelocitySolver2D_Uhlmann::solve()
{
    calc_ib_force();
    apply_ib_force();
}

double& IBVelocitySolver2D_Uhlmann::get_u_value(Domain2DUniform* domain, int iix, int iiy)
{
    // iix, iiy are GLOBAL grid indices
    // Compute global position of this u-face
    double global_x = iix * grid_h;
    double global_y = iiy * grid_h + 0.5 * grid_h;

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

double& IBVelocitySolver2D_Uhlmann::get_v_value(Domain2DUniform* domain, int iix, int iiy)
{
    // iix, iiy are GLOBAL grid indices
    // Compute global position of this v-face
    double global_x = iix * grid_h + 0.5 * grid_h;
    double global_y = iiy * grid_h;

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

void IBVelocitySolver2D_Uhlmann::calc_ib_force()
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
            // X[i], Y[i] are global coordinates
            // ix, iy are global grid indices
            int ix = static_cast<int>(std::floor(X[i] / grid_h));
            int iy = static_cast<int>(std::floor(Y[i] / grid_h));

            // For u: support domain is 4 points in x, 5 points in y
            // Remove clamps to allow cross-domain access via get_u_value
            int min_iix_u = ix - 1;
            int max_iix_u = ix + 2;
            int min_iiy_u = iy - 2;
            int max_iiy_u = iy + 2;

            Uf[i] = 0.0;
            for (int iix = min_iix_u; iix <= max_iix_u; iix++)
            {
                for (int iiy = min_iiy_u; iiy <= max_iiy_u; iiy++)
                {
                    double xi      = iix * grid_h;
                    double yi      = iiy * grid_h + 0.5 * grid_h;
                    double u_value = get_u_value(domain, iix, iiy);

                    Uf[i] += u_value * ib_delta(X[i] - xi, Y[i] - yi, grid_h) * grid_h * grid_h;
                }
            }
            Fx[i] = Up[i] - Uf[i];

            // For v: support domain is 5 points in x, 4 points in y
            int min_iix_v = ix - 2;
            int max_iix_v = ix + 2;
            int min_iiy_v = iy - 1;
            int max_iiy_v = iy + 2;

            Vf[i] = 0.0;
            for (int iix = min_iix_v; iix <= max_iix_v; iix++)
            {
                for (int iiy = min_iiy_v; iiy <= max_iiy_v; iiy++)
                {
                    double xi      = iix * grid_h + 0.5 * grid_h;
                    double yi      = iiy * grid_h;
                    double v_value = get_v_value(domain, iix, iiy);

                    Vf[i] += v_value * ib_delta(X[i] - xi, Y[i] - yi, grid_h) * grid_h * grid_h;
                }
            }
            Fy[i] = Vp[i] - Vf[i];
        }
    }
}

void IBVelocitySolver2D_Uhlmann::apply_ib_force()
{
    // Process each domain in the geometry tree
    for (auto* domain : u_var->geometry->domains)
    {
        auto* particles = coord_map[domain];
        auto* ib_data   = ib_map[domain];

        EXPOSE_PCOORD2D(particles)
        EXPOSE_PIB2D(ib_data)

        if (particles->cur_n == 0)
        {
            continue;
        }

        OPENMP_PARALLEL_FOR()
        for (int ib = 0; ib < particles->cur_n; ib++)
        {
            // Get particle global grid indices
            int ix = static_cast<int>(std::floor(X[ib] / grid_h));
            int iy = static_cast<int>(std::floor(Y[ib] / grid_h));

            // U support domain: ix in [ix-1, ix+2], iy in [iy-2, iy+2]
            int min_iix_u = ix - 1;
            int max_iix_u = ix + 2;
            int min_iiy_u = iy - 2;
            int max_iiy_u = iy + 2;

            for (int iix = min_iix_u; iix <= max_iix_u; iix++)
            {
                for (int iiy = min_iiy_u; iiy <= max_iiy_u; iiy++)
                {
                    double xx = iix * grid_h;
                    double yy = iiy * grid_h + 0.5 * grid_h;

                    double delta    = ib_delta(xx - X[ib], yy - Y[ib], grid_h);
                    double ib_force = Fx[ib] * delta * ib_h * grid_h;

                    get_u_value(domain, iix, iiy) += ib_force;
                }
            }

            // V support domain: ix in [ix-2, ix+2], iy in [iy-1, iy+2]
            int min_iix_v = ix - 2;
            int max_iix_v = ix + 2;
            int min_iiy_v = iy - 1;
            int max_iiy_v = iy + 2;

            for (int iix = min_iix_v; iix <= max_iix_v; iix++)
            {
                for (int iiy = min_iiy_v; iiy <= max_iiy_v; iiy++)
                {
                    double xx = iix * grid_h + 0.5 * grid_h;
                    double yy = iiy * grid_h;

                    double delta    = ib_delta(xx - X[ib], yy - Y[ib], grid_h);
                    double ib_force = Fy[ib] * delta * ib_h * grid_h;

                    get_v_value(domain, iix, iiy) += ib_force;
                }
            }
        }
    }
}