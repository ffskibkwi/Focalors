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
    calc_ib_force();
    apply_ib_force();
}

double& ImmersedBoundarySolver3D::get_u_value(Domain3DUniform* domain, int iix, int iiy, int iiz)
{
    // Compute global position of this u-face
    double px = iix * grid_h;
    double py = iiy * grid_h + 0.5 * grid_h;
    double pz = iiz * grid_h + 0.5 * grid_h;

    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();
    double global_z = pz + domain->get_offset_z();

    auto try_map_u =
        [&](Domain3DUniform* d, double gx, double gy, double gz, double*& ptr, int& li, int& lj, int& lk) -> bool {
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        double local_x = gx - d->get_offset_x();
        double local_y = gy - d->get_offset_y();
        double local_z = gz - d->get_offset_z();

        int ui = static_cast<int>(std::floor(local_x / hx));
        int uj = static_cast<int>(std::floor((local_y - 0.5 * hy) / hy));
        int uk = static_cast<int>(std::floor((local_z - 0.5 * hz) / hz));

        auto& u = *u_var->field_map[d];
        if (ui >= 0 && ui < u.get_nx() && uj >= 0 && uj < u.get_ny() && uk >= 0 && uk < u.get_nz())
        {
            ptr = &u(ui, uj, uk);
            li  = ui;
            lj  = uj;
            lk  = uk;
            return true;
        }
        return false;
    };

    // Try current domain
    {
        double* cell = nullptr;
        int     li   = 0;
        int     lj   = 0;
        int     lk   = 0;
        if (try_map_u(domain, global_x, global_y, global_z, cell, li, lj, lk))
        {
            return *cell;
        }
    }

    // Try neighbors
    if (u_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : u_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double* cell = nullptr;
            int     li   = 0;
            int     lj   = 0;
            int     lk   = 0;
            if (try_map_u(other_domain, global_x, global_y, global_z, cell, li, lj, lk))
            {
                return *cell;
            }
        }
    }

    static double zero = 0.0;
    return zero;
}

double& ImmersedBoundarySolver3D::get_v_value(Domain3DUniform* domain, int iix, int iiy, int iiz)
{
    // Compute global position of this v-face
    double px = iix * grid_h + 0.5 * grid_h;
    double py = iiy * grid_h;
    double pz = iiz * grid_h + 0.5 * grid_h;

    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();
    double global_z = pz + domain->get_offset_z();

    auto try_map_v =
        [&](Domain3DUniform* d, double gx, double gy, double gz, double*& ptr, int& li, int& lj, int& lk) -> bool {
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        double local_x = gx - d->get_offset_x();
        double local_y = gy - d->get_offset_y();
        double local_z = gz - d->get_offset_z();

        int vi = static_cast<int>(std::floor((local_x - 0.5 * hx) / hx));
        int vj = static_cast<int>(std::floor(local_y / hy));
        int vk = static_cast<int>(std::floor((local_z - 0.5 * hz) / hz));

        auto& v = *v_var->field_map[d];
        if (vi >= 0 && vi < v.get_nx() && vj >= 0 && vj < v.get_ny() && vk >= 0 && vk < v.get_nz())
        {
            ptr = &v(vi, vj, vk);
            li  = vi;
            lj  = vj;
            lk  = vk;
            return true;
        }
        return false;
    };

    // Try current domain
    {
        double* cell = nullptr;
        int     li   = 0;
        int     lj   = 0;
        int     lk   = 0;
        if (try_map_v(domain, global_x, global_y, global_z, cell, li, lj, lk))
        {
            return *cell;
        }
    }

    // Try neighbors
    if (v_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : v_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double* cell = nullptr;
            int     li   = 0;
            int     lj   = 0;
            int     lk   = 0;
            if (try_map_v(other_domain, global_x, global_y, global_z, cell, li, lj, lk))
            {
                return *cell;
            }
        }
    }

    static double zero = 0.0;
    return zero;
}

double& ImmersedBoundarySolver3D::get_w_value(Domain3DUniform* domain, int iix, int iiy, int iiz)
{
    // Compute global position of this w-face
    double px = iix * grid_h + 0.5 * grid_h;
    double py = iiy * grid_h + 0.5 * grid_h;
    double pz = iiz * grid_h;

    double global_x = px + domain->get_offset_x();
    double global_y = py + domain->get_offset_y();
    double global_z = pz + domain->get_offset_z();

    auto try_map_w =
        [&](Domain3DUniform* d, double gx, double gy, double gz, double*& ptr, int& li, int& lj, int& lk) -> bool {
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        double local_x = gx - d->get_offset_x();
        double local_y = gy - d->get_offset_y();
        double local_z = gz - d->get_offset_z();

        int wi = static_cast<int>(std::floor((local_x - 0.5 * hx) / hx));
        int wj = static_cast<int>(std::floor((local_y - 0.5 * hy) / hy));
        int wk = static_cast<int>(std::floor(local_z / hz));

        auto& w = *w_var->field_map[d];
        if (wi >= 0 && wi < w.get_nx() && wj >= 0 && wj < w.get_ny() && wk >= 0 && wk < w.get_nz())
        {
            ptr = &w(wi, wj, wk);
            li  = wi;
            lj  = wj;
            lk  = wk;
            return true;
        }
        return false;
    };

    // Try current domain
    {
        double* cell = nullptr;
        int     li   = 0;
        int     lj   = 0;
        int     lk   = 0;
        if (try_map_w(domain, global_x, global_y, global_z, cell, li, lj, lk))
        {
            return *cell;
        }
    }

    // Try neighbors
    if (w_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : w_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double* cell = nullptr;
            int     li   = 0;
            int     lj   = 0;
            int     lk   = 0;
            if (try_map_w(other_domain, global_x, global_y, global_z, cell, li, lj, lk))
            {
                return *cell;
            }
        }
    }

    static double zero = 0.0;
    return zero;
}

void ImmersedBoundarySolver3D::calc_ib_force()
{
    // Process each domain in the geometry tree
    for (auto* domain : u_var->geometry->domains)
    {
        auto* particles = coord_map[domain];
        auto* ib_data   = ib_map[domain];

        EXPOSE_PCOORD3D(particles)
        EXPOSE_PIB3D(ib_data)

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < particles->cur_n; i++)
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
        auto* particles = coord_map[domain];
        auto* ib_data   = ib_map[domain];

        EXPOSE_PCOORD3D(particles)
        EXPOSE_PIB3D(ib_data)

        if (particles->cur_n == 0)
        {
            continue;
        }

        // 使用 PCoord 中缓存的全局 bounding box
        double min_X = particles->min_X;
        double max_X = particles->max_X;
        double min_Y = particles->min_Y;
        double max_Y = particles->max_Y;
        double min_Z = particles->min_Z;
        double max_Z = particles->max_Z;

        int min_ix_u = static_cast<int>(std::floor(min_X / grid_h)) - 1;
        int max_ix_u = static_cast<int>(std::floor(max_X / grid_h)) + 2;
        int min_iy_u = static_cast<int>(std::floor(min_Y / grid_h)) - 2;
        int max_iy_u = static_cast<int>(std::floor(max_Y / grid_h)) + 2;
        int min_iz_u = static_cast<int>(std::floor(min_Z / grid_h)) - 2;
        int max_iz_u = static_cast<int>(std::floor(max_Z / grid_h)) + 2;

        int min_ix_v = static_cast<int>(std::floor(min_X / grid_h)) - 2;
        int max_ix_v = static_cast<int>(std::floor(max_X / grid_h)) + 2;
        int min_iy_v = static_cast<int>(std::floor(min_Y / grid_h)) - 1;
        int max_iy_v = static_cast<int>(std::floor(max_Y / grid_h)) + 2;
        int min_iz_v = static_cast<int>(std::floor(min_Z / grid_h)) - 2;
        int max_iz_v = static_cast<int>(std::floor(max_Z / grid_h)) + 2;

        int min_ix_w = static_cast<int>(std::floor(min_X / grid_h)) - 2;
        int max_ix_w = static_cast<int>(std::floor(max_X / grid_h)) + 2;
        int min_iy_w = static_cast<int>(std::floor(min_Y / grid_h)) - 2;
        int max_iy_w = static_cast<int>(std::floor(max_Y / grid_h)) + 2;
        int min_iz_w = static_cast<int>(std::floor(min_Z / grid_h)) - 1;
        int max_iz_w = static_cast<int>(std::floor(max_Z / grid_h)) + 2;

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

                    for (int ib = 0; ib < particles->cur_n; ib++)
                    {
                        double delta = ib_delta(xx - X[ib], yy - Y[ib], zz - Z[ib], grid_h);
                        double ib_force = Fx[ib] * delta * ib_h * ib_h * grid_h;

                        get_u_value(domain, i, j, k) += ib_force;
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

                    for (int ib = 0; ib < particles->cur_n; ib++)
                    {
                        double ib_force =
                            Fy[ib] * ib_delta(xx - X[ib], yy - Y[ib], zz - Z[ib], grid_h) * ib_h * ib_h * grid_h;

                        get_v_value(domain, i, j, k) += ib_force;
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

                    for (int ib = 0; ib < particles->cur_n; ib++)
                    {
                        double ib_force =
                            Fz[ib] * ib_delta(xx - X[ib], yy - Y[ib], zz - Z[ib], grid_h) * ib_h * ib_h * grid_h;

                        get_w_value(domain, i, j, k) += ib_force;
                    }
                }
            }
        }
    }
}
