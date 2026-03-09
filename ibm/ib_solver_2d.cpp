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
    u2F();
    apply_ib_force();
}

double ImmersedBoundarySolver2D::get_u_value(Domain2DUniform* domain, int iix, int iiy)
{
    auto  ctx = get_domain_context(domain);
    auto& u   = *u_var->field_map[domain];

    // Check if indices are within current domain bounds
    // DomainContext can handle boundary buffer layers: i in [-1,nx], j in [-1,ny]
    if (iix >= -1 && iix <= u.get_nx() && iiy >= -1 && iiy <= u.get_ny())
    {
        return ctx.get_u(iix, iiy);
    }

    // Check if there's a neighbor domain that can provide the value
    double px = iix * grid_h;
    double py = iiy * grid_h + 0.5 * grid_h;

    // Convert to global coordinates
    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();

    // Find which domain contains this global position
    // Check neighbors through adjacency
    if (u_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : u_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double other_offset_x = other_domain->get_offset_x();
            double other_offset_y = other_domain->get_offset_y();
            double other_hx       = other_domain->get_hx();
            double other_hy       = other_domain->get_hy();
            int    other_nx       = other_domain->get_nx();
            int    other_ny       = other_domain->get_ny();

            // Convert global to local coordinates
            double local_x = global_x - other_offset_x;
            double local_y = global_y - other_offset_y;

            // Check if within domain bounds (considering buffer regions)
            if (local_x >= -other_hx && local_x <= other_nx * other_hx && local_y >= -0.5 * other_hy &&
                local_y <= other_ny * other_hy + 0.5 * other_hy)
            {
                // Convert to grid indices for this domain
                int local_iix = static_cast<int>(std::floor(local_x / other_hx));
                int local_iiy = static_cast<int>(std::floor((local_y - 0.5 * other_hy) / other_hy));

                auto  other_ctx = get_domain_context(other_domain);
                auto& other_u   = *u_var->field_map[other_domain];

                // Check if valid indices for this domain
                // DomainContext can handle boundary buffer layers: i in [-1,nx], j in [-1,ny]
                if (local_iix >= -1 && local_iix <= other_u.get_nx() && local_iiy >= -1 &&
                    local_iiy <= other_u.get_ny())
                {
                    return other_ctx.get_u(local_iix, local_iiy);
                }
            }
        }
    }

    // If no neighbor found, return 0 (or handle boundary condition)
    return 0.0;
}

double ImmersedBoundarySolver2D::get_v_value(Domain2DUniform* domain, int iix, int iiy)
{
    auto  ctx = get_domain_context(domain);
    auto& v   = *v_var->field_map[domain];

    // Check if indices are within current domain bounds
    // DomainContext can handle boundary buffer layers: i in [-1,nx], j in [-1,ny]
    if (iix >= -1 && iix <= v.get_nx() && iiy >= -1 && iiy <= v.get_ny())
    {
        return ctx.get_v(iix, iiy);
    }

    // Check if there's a neighbor domain that can provide the value
    double px = iix * grid_h + 0.5 * grid_h;
    double py = iiy * grid_h;

    // Convert to global coordinates
    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();

    // Find which domain contains this global position
    // Check neighbors through adjacency
    if (v_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : v_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double other_offset_x = other_domain->get_offset_x();
            double other_offset_y = other_domain->get_offset_y();
            double other_hx       = other_domain->get_hx();
            double other_hy       = other_domain->get_hy();
            int    other_nx       = other_domain->get_nx();
            int    other_ny       = other_domain->get_ny();

            // Convert global to local coordinates
            double local_x = global_x - other_offset_x;
            double local_y = global_y - other_offset_y;

            // Check if within domain bounds (considering buffer regions)
            if (local_x >= -0.5 * other_hx && local_x <= other_nx * other_hx + 0.5 * other_hx && local_y >= -other_hy &&
                local_y <= other_ny * other_hy)
            {
                // Convert to grid indices for this domain
                int local_iix = static_cast<int>(std::floor((local_x - 0.5 * other_hx) / other_hx));
                int local_iiy = static_cast<int>(std::floor(local_y / other_hy));

                auto  other_ctx = get_domain_context(other_domain);
                auto& other_v   = *v_var->field_map[other_domain];

                // Check if valid indices for this domain
                // DomainContext can handle boundary buffer layers: i in [-1,nx], j in [-1,ny]
                if (local_iix >= -1 && local_iix <= other_v.get_nx() && local_iiy >= -1 &&
                    local_iiy <= other_v.get_ny())
                {
                    return other_ctx.get_v(local_iix, local_iiy);
                }
            }
        }
    }

    // If no neighbor found, return 0 (or handle boundary condition)
    return 0.0;
}

void ImmersedBoundarySolver2D::u2F()
{
    // Process each domain in the geometry tree
    for (auto* domain : u_var->geometry->domains)
    {
        auto& particles = *coord_map[domain];
        auto& ib_data   = *ib_map[domain];

        auto ctx = get_domain_context(domain);

        EXPOSE_PCOORD2D(&particles)
        EXPOSE_PIB2D(&ib_data)

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < particles.cur_n; i++)
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
                    double xi = iix * grid_h;
                    double yi = iiy * grid_h + 0.5 * grid_h;

                    double u_value;
                    if (iiy == -1)
                    {
                        u_value = ctx.get_u(iix, iiy);
                    }
                    else if (iiy == u_var->field_map[domain]->get_ny())
                    {
                        u_value = ctx.get_u(iix, iiy);
                    }
                    else
                    {
                        u_value = get_u_value(domain, iix, iiy);
                    }

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
                    double xi = iix * grid_h + 0.5 * grid_h;
                    double yi = iiy * grid_h;

                    double v_value;
                    if (iix == -1)
                    {
                        v_value = ctx.get_v(iix, iiy);
                    }
                    else if (iix == v_var->field_map[domain]->get_nx())
                    {
                        v_value = ctx.get_v(iix, iiy);
                    }
                    else
                    {
                        v_value = get_v_value(domain, iix, iiy);
                    }

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
        auto& particles = *coord_map[domain];
        auto& ib_data   = *ib_map[domain];

        EXPOSE_PCOORD2D_BOUND(&particles)
        EXPOSE_PIB2D(&ib_data)

        auto& u_recv = *u_var->field_map[domain];
        auto& v_recv = *v_var->field_map[domain];

        // u
        OPENMP_PARALLEL_FOR()
        for (int i = min_ix_u; i <= max_ix_u; i++)
        {
            for (int j = min_iy_u; j <= max_iy_u; j++)
            {
                double xx = i * grid_h;
                double yy = j * grid_h + 0.5 * grid_h;

                for (int ib = 0; ib < particles.cur_n; ib++)
                {
                    double ib_force = Fx[ib] * ib_delta(xx - X[ib], yy - Y[ib], grid_h) * ib_h * grid_h;
                    u_recv(i, j) += ib_force;
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

                for (int ib = 0; ib < particles.cur_n; ib++)
                {
                    double ib_force = Fy[ib] * ib_delta(xx - X[ib], yy - Y[ib], grid_h) * ib_h * grid_h;
                    v_recv(i, j) += ib_force;
                }
            }
        }
    }
}