#include "ib_solver_3d.h"

#include <cmath>

ImmersedBoundarySolver3D::ImmersedBoundarySolver3D(Variable3D*                                      _u_var,
                                                   Variable3D*                                      _v_var,
                                                   Variable3D*                                      _w_var,
                                                   std::unordered_map<Domain3DUniform*, PCoord3D*>& _coord_map)
    : u_var(_u_var)
    , v_var(_v_var)
    , w_var(_w_var)
    , coord_map(_coord_map)
{
    for (auto* domain : u_var->geometry->domains)
    {
        ib_map[domain] = new PIB3D(coord_map[domain]->max_n);
    }

    // Setup domain context accessor
    get_domain_context = [&](Domain3DUniform* domain) -> DomainContext {
        auto& u = *u_var->field_map[domain];
        auto& v = *v_var->field_map[domain];
        auto& w = *w_var->field_map[domain];

        auto& u_buffer_x_neg = *u_var->buffer_map[domain][LocationType::XNegative];
        auto& u_buffer_x_pos = *u_var->buffer_map[domain][LocationType::XPositive];
        auto& u_buffer_y_neg = *u_var->buffer_map[domain][LocationType::YNegative];
        auto& u_buffer_y_pos = *u_var->buffer_map[domain][LocationType::YPositive];
        auto& u_buffer_z_neg = *u_var->buffer_map[domain][LocationType::ZNegative];
        auto& u_buffer_z_pos = *u_var->buffer_map[domain][LocationType::ZPositive];

        auto& v_buffer_x_neg = *v_var->buffer_map[domain][LocationType::XNegative];
        auto& v_buffer_x_pos = *v_var->buffer_map[domain][LocationType::XPositive];
        auto& v_buffer_y_neg = *v_var->buffer_map[domain][LocationType::YNegative];
        auto& v_buffer_y_pos = *v_var->buffer_map[domain][LocationType::YPositive];
        auto& v_buffer_z_neg = *v_var->buffer_map[domain][LocationType::ZNegative];
        auto& v_buffer_z_pos = *v_var->buffer_map[domain][LocationType::ZPositive];

        auto& w_buffer_x_neg = *w_var->buffer_map[domain][LocationType::XNegative];
        auto& w_buffer_x_pos = *w_var->buffer_map[domain][LocationType::XPositive];
        auto& w_buffer_y_neg = *w_var->buffer_map[domain][LocationType::YNegative];
        auto& w_buffer_y_pos = *w_var->buffer_map[domain][LocationType::YPositive];
        auto& w_buffer_z_neg = *w_var->buffer_map[domain][LocationType::ZNegative];
        auto& w_buffer_z_pos = *w_var->buffer_map[domain][LocationType::ZPositive];

        return DomainContext {domain,
                              [&](int i, int j, int k) -> double {
                                  // u is x-face centered; allow i=-1/nx (x buffers)
                                  if (i == -1)
                                  {
                                      j = std::clamp(j, 0, u.get_ny() - 1);
                                      k = std::clamp(k, 0, u.get_nz() - 1);
                                      return u_buffer_x_neg(j, k);
                                  }
                                  else if (i == u.get_nx())
                                  {
                                      j = std::clamp(j, 0, u.get_ny() - 1);
                                      k = std::clamp(k, 0, u.get_nz() - 1);
                                      return u_buffer_x_pos(j, k);
                                  }
                                  else if (j == -1)
                                  {
                                      i = std::clamp(i, 0, u.get_nx() - 1);
                                      k = std::clamp(k, 0, u.get_nz() - 1);
                                      return u_buffer_y_neg(i, k);
                                  }
                                  else if (j == u.get_ny())
                                  {
                                      i = std::clamp(i, 0, u.get_nx() - 1);
                                      k = std::clamp(k, 0, u.get_nz() - 1);
                                      return u_buffer_y_pos(i, k);
                                  }
                                  else if (k == -1)
                                  {
                                      i = std::clamp(i, 0, u.get_nx() - 1);
                                      j = std::clamp(j, 0, u.get_ny() - 1);
                                      return u_buffer_z_neg(i, j);
                                  }
                                  else if (k == u.get_nz())
                                  {
                                      i = std::clamp(i, 0, u.get_nx() - 1);
                                      j = std::clamp(j, 0, u.get_ny() - 1);
                                      return u_buffer_z_pos(i, j);
                                  }
                                  else
                                  {
                                      return u(i, j, k);
                                  }
                              },
                              [&](int i, int j, int k) -> double {
                                  // v is y-face centered; allow j=-1/ny (y buffers)
                                  if (j == -1)
                                  {
                                      i = std::clamp(i, 0, v.get_nx() - 1);
                                      k = std::clamp(k, 0, v.get_nz() - 1);
                                      return v_buffer_y_neg(i, k);
                                  }
                                  else if (j == v.get_ny())
                                  {
                                      i = std::clamp(i, 0, v.get_nx() - 1);
                                      k = std::clamp(k, 0, v.get_nz() - 1);
                                      return v_buffer_y_pos(i, k);
                                  }
                                  else if (i == -1)
                                  {
                                      j = std::clamp(j, 0, v.get_ny() - 1);
                                      k = std::clamp(k, 0, v.get_nz() - 1);
                                      return v_buffer_x_neg(j, k);
                                  }
                                  else if (i == v.get_nx())
                                  {
                                      j = std::clamp(j, 0, v.get_ny() - 1);
                                      k = std::clamp(k, 0, v.get_nz() - 1);
                                      return v_buffer_x_pos(j, k);
                                  }
                                  else if (k == -1)
                                  {
                                      i = std::clamp(i, 0, v.get_nx() - 1);
                                      j = std::clamp(j, 0, v.get_ny() - 1);
                                      return v_buffer_z_neg(i, j);
                                  }
                                  else if (k == v.get_nz())
                                  {
                                      i = std::clamp(i, 0, v.get_nx() - 1);
                                      j = std::clamp(j, 0, v.get_ny() - 1);
                                      return v_buffer_z_pos(i, j);
                                  }
                                  else
                                  {
                                      return v(i, j, k);
                                  }
                              },
                              [&](int i, int j, int k) -> double {
                                  // w is z-face centered; allow k=-1/nz (z buffers)
                                  if (k == -1)
                                  {
                                      i = std::clamp(i, 0, w.get_nx() - 1);
                                      j = std::clamp(j, 0, w.get_ny() - 1);
                                      return w_buffer_z_neg(i, j);
                                  }
                                  else if (k == w.get_nz())
                                  {
                                      i = std::clamp(i, 0, w.get_nx() - 1);
                                      j = std::clamp(j, 0, w.get_ny() - 1);
                                      return w_buffer_z_pos(i, j);
                                  }
                                  else if (i == -1)
                                  {
                                      j = std::clamp(j, 0, w.get_ny() - 1);
                                      k = std::clamp(k, 0, w.get_nz() - 1);
                                      return w_buffer_x_neg(j, k);
                                  }
                                  else if (i == w.get_nx())
                                  {
                                      j = std::clamp(j, 0, w.get_ny() - 1);
                                      k = std::clamp(k, 0, w.get_nz() - 1);
                                      return w_buffer_x_pos(j, k);
                                  }
                                  else if (j == -1)
                                  {
                                      i = std::clamp(i, 0, w.get_nx() - 1);
                                      k = std::clamp(k, 0, w.get_nz() - 1);
                                      return w_buffer_y_neg(i, k);
                                  }
                                  else if (j == w.get_ny())
                                  {
                                      i = std::clamp(i, 0, w.get_nx() - 1);
                                      k = std::clamp(k, 0, w.get_nz() - 1);
                                      return w_buffer_y_pos(i, k);
                                  }
                                  else
                                  {
                                      return w(i, j, k);
                                  }
                              }};
    };
}

void ImmersedBoundarySolver3D::solve()
{
    u2F();
    apply_ib_force();
}

double ImmersedBoundarySolver3D::get_u_value(Domain3DUniform* domain, int iix, int iiy, int iiz)
{
    auto& u = *u_var->field_map[domain];

    // Check if indices are within current domain bounds
    // DomainContext can handle boundary buffer layers: i/j/k in [-1, nx/ny/nz]
    if (iix >= -1 && iix <= u.get_nx() && iiy >= -1 && iiy <= u.get_ny() && iiz >= -1 && iiz <= u.get_nz())
    {
        auto ctx = get_domain_context(domain);
        return ctx.get_u(iix, iiy, iiz);
    }

    // Check if there's a neighbor domain that can provide the value
    double px = iix * grid_h;
    double py = iiy * grid_h + 0.5 * grid_h;
    double pz = iiz * grid_h + 0.5 * grid_h;

    // Convert to global coordinates
    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();
    double global_z = pz + domain->get_offset_z();

    // Find which domain contains this global position
    // Check neighbors through adjacency
    if (u_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : u_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double other_offset_x = other_domain->get_offset_x();
            double other_offset_y = other_domain->get_offset_y();
            double other_offset_z = other_domain->get_offset_z();
            double other_hx       = other_domain->get_hx();
            double other_hy       = other_domain->get_hy();
            double other_hz       = other_domain->get_hz();
            int    other_nx       = other_domain->get_nx();
            int    other_ny       = other_domain->get_ny();
            int    other_nz       = other_domain->get_nz();

            // Convert global to local coordinates
            double local_x = global_x - other_offset_x;
            double local_y = global_y - other_offset_y;
            double local_z = global_z - other_offset_z;

            // Check if within domain bounds (considering buffer regions)
            if (local_x >= -other_hx && local_x <= other_nx * other_hx && local_y >= -0.5 * other_hy &&
                local_y <= other_ny * other_hy + 0.5 * other_hy && local_z >= -0.5 * other_hz &&
                local_z <= other_nz * other_hz + 0.5 * other_hz)
            {
                // Convert to grid indices for this domain
                int local_iix = static_cast<int>(std::floor(local_x / other_hx));
                int local_iiy = static_cast<int>(std::floor((local_y - 0.5 * other_hy) / other_hy));
                int local_iiz = static_cast<int>(std::floor((local_z - 0.5 * other_hz) / other_hz));

                auto  other_ctx = get_domain_context(other_domain);
                auto& other_u   = *u_var->field_map[other_domain];

                // Check if valid indices for this domain
                // DomainContext can handle boundary buffer layers: i/j/k in [-1, nx/ny/nz]
                if (local_iix >= -1 && local_iix <= other_u.get_nx() && local_iiy >= -1 &&
                    local_iiy <= other_u.get_ny() && local_iiz >= -1 && local_iiz <= other_u.get_nz())
                {
                    return other_ctx.get_u(local_iix, local_iiy, local_iiz);
                }
            }
        }
    }

    // If no neighbor found, return 0 (or handle boundary condition)
    return 0.0;
}

double ImmersedBoundarySolver3D::get_v_value(Domain3DUniform* domain, int iix, int iiy, int iiz)
{
    auto& v = *v_var->field_map[domain];

    // Check if indices are within current domain bounds
    // DomainContext can handle boundary buffer layers: i/j/k in [-1, nx/ny/nz]
    if (iix >= -1 && iix <= v.get_nx() && iiy >= -1 && iiy <= v.get_ny() && iiz >= -1 && iiz <= v.get_nz())
    {
        auto ctx = get_domain_context(domain);
        return ctx.get_v(iix, iiy, iiz);
    }

    // Check if there's a neighbor domain that can provide the value
    double px = iix * grid_h + 0.5 * grid_h;
    double py = iiy * grid_h;
    double pz = iiz * grid_h + 0.5 * grid_h;

    // Convert to global coordinates
    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();
    double global_z = pz + domain->get_offset_z();

    // Find which domain contains this global position
    // Check neighbors through adjacency
    if (v_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : v_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double other_offset_x = other_domain->get_offset_x();
            double other_offset_y = other_domain->get_offset_y();
            double other_offset_z = other_domain->get_offset_z();
            double other_hx       = other_domain->get_hx();
            double other_hy       = other_domain->get_hy();
            double other_hz       = other_domain->get_hz();
            int    other_nx       = other_domain->get_nx();
            int    other_ny       = other_domain->get_ny();
            int    other_nz       = other_domain->get_nz();

            // Convert global to local coordinates
            double local_x = global_x - other_offset_x;
            double local_y = global_y - other_offset_y;
            double local_z = global_z - other_offset_z;

            // Check if within domain bounds (considering buffer regions)
            if (local_x >= -0.5 * other_hx && local_x <= other_nx * other_hx + 0.5 * other_hx && local_y >= -other_hy &&
                local_y <= other_ny * other_hy && local_z >= -0.5 * other_hz &&
                local_z <= other_nz * other_hz + 0.5 * other_hz)
            {
                // Convert to grid indices for this domain
                int local_iix = static_cast<int>(std::floor((local_x - 0.5 * other_hx) / other_hx));
                int local_iiy = static_cast<int>(std::floor(local_y / other_hy));
                int local_iiz = static_cast<int>(std::floor((local_z - 0.5 * other_hz) / other_hz));

                auto  other_ctx = get_domain_context(other_domain);
                auto& other_v   = *v_var->field_map[other_domain];

                // Check if valid indices for this domain
                // DomainContext can handle boundary buffer layers: i/j/k in [-1, nx/ny/nz]
                if (local_iix >= -1 && local_iix <= other_v.get_nx() && local_iiy >= -1 &&
                    local_iiy <= other_v.get_ny() && local_iiz >= -1 && local_iiz <= other_v.get_nz())
                {
                    return other_ctx.get_v(local_iix, local_iiy, local_iiz);
                }
            }
        }
    }

    // If no neighbor found, return 0 (or handle boundary condition)
    return 0.0;
}

double ImmersedBoundarySolver3D::get_w_value(Domain3DUniform* domain, int iix, int iiy, int iiz)
{
    auto& w = *w_var->field_map[domain];

    // Check if indices are within current domain bounds
    // DomainContext can handle boundary buffer layers: i/j/k in [-1, nx/ny/nz]
    if (iix >= -1 && iix <= w.get_nx() && iiy >= -1 && iiy <= w.get_ny() && iiz >= -1 && iiz <= w.get_nz())
    {
        auto ctx = get_domain_context(domain);
        return ctx.get_w(iix, iiy, iiz);
    }

    // Check if there's a neighbor domain that can provide the value
    double px = iix * grid_h + 0.5 * grid_h;
    double py = iiy * grid_h + 0.5 * grid_h;
    double pz = iiz * grid_h;

    // Convert to global coordinates
    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();
    double global_z = pz + domain->get_offset_z();

    // Find which domain contains this global position
    // Check neighbors through adjacency
    if (w_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : w_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double other_offset_x = other_domain->get_offset_x();
            double other_offset_y = other_domain->get_offset_y();
            double other_offset_z = other_domain->get_offset_z();
            double other_hx       = other_domain->get_hx();
            double other_hy       = other_domain->get_hy();
            double other_hz       = other_domain->get_hz();
            int    other_nx       = other_domain->get_nx();
            int    other_ny       = other_domain->get_ny();
            int    other_nz       = other_domain->get_nz();

            // Convert global to local coordinates
            double local_x = global_x - other_offset_x;
            double local_y = global_y - other_offset_y;
            double local_z = global_z - other_offset_z;

            // Check if within domain bounds (considering buffer regions)
            if (local_x >= -0.5 * other_hx && local_x <= other_nx * other_hx + 0.5 * other_hx &&
                local_y >= -0.5 * other_hy && local_y <= other_ny * other_hy + 0.5 * other_hy && local_z >= -other_hz &&
                local_z <= other_nz * other_hz)
            {
                // Convert to grid indices for this domain
                int local_iix = static_cast<int>(std::floor((local_x - 0.5 * other_hx) / other_hx));
                int local_iiy = static_cast<int>(std::floor((local_y - 0.5 * other_hy) / other_hy));
                int local_iiz = static_cast<int>(std::floor(local_z / other_hz));

                auto  other_ctx = get_domain_context(other_domain);
                auto& other_w   = *w_var->field_map[other_domain];

                // Check if valid indices for this domain
                // DomainContext can handle boundary buffer layers: i/j/k in [-1, nx/ny/nz]
                if (local_iix >= -1 && local_iix <= other_w.get_nx() && local_iiy >= -1 &&
                    local_iiy <= other_w.get_ny() && local_iiz >= -1 && local_iiz <= other_w.get_nz())
                {
                    return other_ctx.get_w(local_iix, local_iiy, local_iiz);
                }
            }
        }
    }

    // If no neighbor found, return 0 (or handle boundary condition)
    return 0.0;
}

void ImmersedBoundarySolver3D::u2F()
{
    // Process each domain in the geometry tree
    for (auto* domain : u_var->geometry->domains)
    {
        auto& particles = *coord_map[domain];
        auto& ib_data   = *ib_map[domain];

        EXPOSE_PCOORD3D(&particles)
        EXPOSE_PIB3D(&ib_data)

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < particles.cur_n; i++)
        {
            int ix = std::floor(X[i] / grid_h);
            int iy = std::floor(Y[i] / grid_h);
            int iz = std::floor(Z[i] / grid_h);

            int min_iix = std::clamp(ix - 1, 0, u_var->field_map[domain]->get_nx() - 1);
            int max_iix = std::clamp(ix + 2, 0, u_var->field_map[domain]->get_nx() - 1);
            int min_iiy = std::clamp(iy - 2, -1, u_var->field_map[domain]->get_ny());
            int max_iiy = std::clamp(iy + 2, -1, u_var->field_map[domain]->get_ny());
            int min_iiz = std::clamp(iz - 2, -1, u_var->field_map[domain]->get_nz());
            int max_iiz = std::clamp(iz + 2, -1, u_var->field_map[domain]->get_nz());

            Uf[i] = 0.0;
            for (int iix = min_iix; iix <= max_iix; iix++)
            {
                for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                {
                    for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                    {
                        double xi      = iix * grid_h;
                        double yi      = iiy * grid_h + 0.5 * grid_h;
                        double zi      = iiz * grid_h + 0.5 * grid_h;
                        double u_value = get_u_value(domain, iix, iiy, iiz);

                        Uf[i] += u_value * ib_delta(X[i] - xi, Y[i] - yi, Z[i] - zi, grid_h) * grid_h * grid_h * grid_h;
                    }
                }
            }
            Fx[i] = Up[i] - Uf[i];
            Fx_sum[i] += Fx[i];

            min_iix = std::clamp(ix - 2, -1, v_var->field_map[domain]->get_nx());
            max_iix = std::clamp(ix + 2, -1, v_var->field_map[domain]->get_nx());
            min_iiy = std::clamp(iy - 1, 0, v_var->field_map[domain]->get_ny() - 1);
            max_iiy = std::clamp(iy + 2, 0, v_var->field_map[domain]->get_ny() - 1);
            min_iiz = std::clamp(iz - 2, -1, v_var->field_map[domain]->get_nz());
            max_iiz = std::clamp(iz + 2, -1, v_var->field_map[domain]->get_nz());

            Vf[i] = 0.0;
            for (int iix = min_iix; iix <= max_iix; iix++)
            {
                for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                {
                    for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                    {
                        double xi      = iix * grid_h + 0.5 * grid_h;
                        double yi      = iiy * grid_h;
                        double zi      = iiz * grid_h + 0.5 * grid_h;
                        double v_value = get_v_value(domain, iix, iiy, iiz);

                        Vf[i] += v_value * ib_delta(X[i] - xi, Y[i] - yi, Z[i] - zi, grid_h) * grid_h * grid_h * grid_h;
                    }
                }
            }
            Fy[i] = Vp[i] - Vf[i];
            Fy_sum[i] += Fy[i];

            min_iix = std::clamp(ix - 2, -1, w_var->field_map[domain]->get_nx());
            max_iix = std::clamp(ix + 2, -1, w_var->field_map[domain]->get_nx());
            min_iiy = std::clamp(iy - 2, -1, w_var->field_map[domain]->get_ny());
            max_iiy = std::clamp(iy + 2, -1, w_var->field_map[domain]->get_ny());
            min_iiz = std::clamp(iz - 1, 0, w_var->field_map[domain]->get_nz() - 1);
            max_iiz = std::clamp(iz + 2, 0, w_var->field_map[domain]->get_nz() - 1);

            Wf[i] = 0.0;
            for (int iix = min_iix; iix <= max_iix; iix++)
            {
                for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                {
                    for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                    {
                        double xi      = iix * grid_h + 0.5 * grid_h;
                        double yi      = iiy * grid_h + 0.5 * grid_h;
                        double zi      = iiz * grid_h;
                        double w_value = get_w_value(domain, iix, iiy, iiz);

                        Wf[i] += w_value * ib_delta(X[i] - xi, Y[i] - yi, Z[i] - zi, grid_h) * grid_h * grid_h * grid_h;
                    }
                }
            }
            Fz[i] = Wp[i] - Wf[i];
            Fz_sum[i] += Fz[i];
        }
    }
}

void ImmersedBoundarySolver3D::apply_ib_force()
{
    // Process each domain in geometry tree
    for (auto* domain : u_var->geometry->domains)
    {
        auto& particles = *coord_map[domain];
        auto& ib_data   = *ib_map[domain];

        EXPOSE_PCOORD3D_BOUND(&particles)
        EXPOSE_PIB3D(&ib_data)

        auto& u_recv = *u_var->field_map[domain];
        auto& v_recv = *v_var->field_map[domain];
        auto& w_recv = *w_var->field_map[domain];

        // u
        OPENMP_PARALLEL_FOR()
        for (int i = min_ix_u; i <= max_ix_u; i++)
        {
            for (int j = min_iy_u; j <= max_iy_u; j++)
            {
                for (int k = min_iz_u; k <= max_iz_u; k++)
                {
                    double xx = i * grid_h;
                    double yy = j * grid_h + 0.5 * grid_h;
                    double zz = k * grid_h + 0.5 * grid_h;

                    for (int ib = 0; ib < particles.cur_n; ib++)
                    {
                        double ib_force =
                            Fx[ib] * ib_delta(xx - X[ib], yy - Y[ib], zz - Z[ib], grid_h) * ib_h * ib_h * grid_h;

                        u_recv(i, j, k) += ib_force;
                    }
                }
            }
        }

        // v
        OPENMP_PARALLEL_FOR()
        for (int i = min_ix_v; i <= max_ix_v; i++)
        {
            for (int j = min_iy_v; j <= max_iy_v; j++)
            {
                for (int k = min_iz_v; k <= max_iz_v; k++)
                {
                    double xx = i * grid_h + 0.5 * grid_h;
                    double yy = j * grid_h;
                    double zz = k * grid_h + 0.5 * grid_h;

                    for (int ib = 0; ib < particles.cur_n; ib++)
                    {
                        double ib_force =
                            Fy[ib] * ib_delta(xx - X[ib], yy - Y[ib], zz - Z[ib], grid_h) * ib_h * ib_h * grid_h;

                        v_recv(i, j, k) += ib_force;
                    }
                }
            }
        }

        // w
        OPENMP_PARALLEL_FOR()
        for (int i = min_ix_w; i <= max_ix_w; i++)
        {
            for (int j = min_iy_w; j <= max_iy_w; j++)
            {
                for (int k = min_iz_w; k <= max_iz_w; k++)
                {
                    double xx = i * grid_h + 0.5 * grid_h;
                    double yy = j * grid_h + 0.5 * grid_h;
                    double zz = k * grid_h;

                    for (int ib = 0; ib < particles.cur_n; ib++)
                    {
                        double ib_force =
                            Fz[ib] * ib_delta(xx - X[ib], yy - Y[ib], zz - Z[ib], grid_h) * ib_h * ib_h * grid_h;

                        w_recv(i, j, k) += ib_force;
                    }
                }
            }
        }
    }
}
