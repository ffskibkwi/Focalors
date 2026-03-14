#include "ib_scalar_solver_3d_Uhlmann.h"

#include <cmath>

IBScalarSolver3D_Uhlmann::IBScalarSolver3D_Uhlmann(Variable3D*                                       _scalar_var,
                                                   std::unordered_map<Domain3DUniform*, PCoord3D*>&  _coord_map,
                                                   std::unordered_map<Domain3DUniform*, PIBNormal*>& _normal_map)
    : scalar_var(_scalar_var)
    , coord_map(_coord_map)
    , normal_map(_normal_map)
{
    for (auto* domain : scalar_var->geometry->domains)
    {
        ib_map[domain] = new PIBScalar(coord_map[domain]->max_n);
    }

    // Setup domain context accessor
    get_domain_context = [&](Domain3DUniform* domain) -> DomainContext {
        auto& scalar = *scalar_var->field_map[domain];

        auto& scalar_buffer_x_neg = *scalar_var->buffer_map[domain][LocationType::XNegative];
        auto& scalar_buffer_x_pos = *scalar_var->buffer_map[domain][LocationType::XPositive];
        auto& scalar_buffer_y_neg = *scalar_var->buffer_map[domain][LocationType::YNegative];
        auto& scalar_buffer_y_pos = *scalar_var->buffer_map[domain][LocationType::YPositive];
        auto& scalar_buffer_z_neg = *scalar_var->buffer_map[domain][LocationType::ZNegative];
        auto& scalar_buffer_z_pos = *scalar_var->buffer_map[domain][LocationType::ZPositive];

        return DomainContext {domain, [&](int i, int j, int k) -> double {
                                  // scalar is cell-centered
                                  if (i == -1)
                                  {
                                      j = std::clamp(j, 0, scalar.get_ny() - 1);
                                      k = std::clamp(k, 0, scalar.get_nz() - 1);
                                      return scalar_buffer_x_neg(j, k);
                                  }
                                  else if (i == scalar.get_nx())
                                  {
                                      j = std::clamp(j, 0, scalar.get_ny() - 1);
                                      k = std::clamp(k, 0, scalar.get_nz() - 1);
                                      return scalar_buffer_x_pos(j, k);
                                  }
                                  else if (j == -1)
                                  {
                                      i = std::clamp(i, 0, scalar.get_nx() - 1);
                                      k = std::clamp(k, 0, scalar.get_nz() - 1);
                                      return scalar_buffer_y_neg(i, k);
                                  }
                                  else if (j == scalar.get_ny())
                                  {
                                      i = std::clamp(i, 0, scalar.get_nx() - 1);
                                      k = std::clamp(k, 0, scalar.get_nz() - 1);
                                      return scalar_buffer_y_pos(i, k);
                                  }
                                  else if (k == -1)
                                  {
                                      i = std::clamp(i, 0, scalar.get_nx() - 1);
                                      j = std::clamp(j, 0, scalar.get_ny() - 1);
                                      return scalar_buffer_z_neg(i, j);
                                  }
                                  else if (k == scalar.get_nz())
                                  {
                                      i = std::clamp(i, 0, scalar.get_nx() - 1);
                                      j = std::clamp(j, 0, scalar.get_ny() - 1);
                                      return scalar_buffer_z_pos(i, j);
                                  }
                                  else
                                  {
                                      return scalar(i, j, k);
                                  }
                              }};
    };
}

void IBScalarSolver3D_Uhlmann::solve()
{
    calc_ib_scalar();
    apply_ib_scalar();
}

double& IBScalarSolver3D_Uhlmann::get_scalar_value(Domain3DUniform* domain, int iix, int iiy, int iiz)
{
    // iix, iiy, iiz are GLOBAL grid indices
    // Compute global position of this cell center
    double global_x = iix * grid_h;
    double global_y = iiy * grid_h;
    double global_z = iiz * grid_h;

    // Helper lambda: try map a global position to a cell in given domain for scalar
    auto try_map_scalar =
        [&](Domain3DUniform* d, double gx, double gy, double gz, double*& ptr, int& li, int& lj, int& lk) -> bool {
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        double local_x = gx - d->get_offset_x();
        double local_y = gy - d->get_offset_y();
        double local_z = gz - d->get_offset_z();

        int si = static_cast<int>(std::floor(local_x / hx));
        int sj = static_cast<int>(std::floor(local_y / hy));
        int sk = static_cast<int>(std::floor(local_z / hz));

        auto& s = *scalar_var->field_map[d];
        if (si >= 0 && si < s.get_nx() && sj >= 0 && sj < s.get_ny() && sk >= 0 && sk < s.get_nz())
        {
            ptr = &s(si, sj, sk);
            li  = si;
            lj  = sj;
            lk  = sk;
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
        if (try_map_scalar(domain, global_x, global_y, global_z, cell, li, lj, lk))
        {
            return *cell;
        }
    }

    // Try neighbors
    if (scalar_var->geometry->adjacency.count(domain))
    {
        for (auto& loc_neighbor_pair : scalar_var->geometry->adjacency[domain])
        {
            auto* other_domain = loc_neighbor_pair.second;

            double* cell = nullptr;
            int     li   = 0;
            int     lj   = 0;
            int     lk   = 0;
            if (try_map_scalar(other_domain, global_x, global_y, global_z, cell, li, lj, lk))
            {
                return *cell;
            }
        }
    }

    static double zero = 0.0;
    return zero;
}

void IBScalarSolver3D_Uhlmann::calc_ib_scalar()
{
    // Process each domain in the geometry tree
    for (auto* domain : scalar_var->geometry->domains)
    {
        auto* particles = coord_map[domain];
        auto* ib_data   = ib_map[domain];
        auto* normal    = normal_map[domain];

        // Skip domains without particles
        if (!particles || particles->cur_n == 0)
            continue;

        EXPOSE_PCOORD3D(particles)
        EXPOSE_PIBSCALAR(ib_data)
        EXPOSE_PIBNORMAL(normal)

        if (pde_type == PDEBoundaryType::Dirichlet)
        {
            // Dirichlet: F = Sp - Sf (force to enforce scalar value at IB point)
            OPENMP_PARALLEL_FOR()
            for (int i = 0; i < particles->cur_n; i++)
            {
                // X[i], Y[i], Z[i] are global coordinates
                // ix, iy, iz are global grid indices
                int ix = static_cast<int>(std::floor(X[i] / grid_h));
                int iy = static_cast<int>(std::floor(Y[i] / grid_h));
                int iz = static_cast<int>(std::floor(Z[i] / grid_h));

                // For scalar: support domain is 5 points in x, y, and z
                // range: [i-2, i+2] in each direction
                int min_iix = ix - 2;
                int max_iix = ix + 2;
                int min_iiy = iy - 2;
                int max_iiy = iy + 2;
                int min_iiz = iz - 2;
                int max_iiz = iz + 2;

                Sf[i] = 0.0;
                for (int iix = min_iix; iix <= max_iix; iix++)
                {
                    for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                    {
                        for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                        {
                            double xi         = iix * grid_h;
                            double yi         = iiy * grid_h;
                            double zi         = iiz * grid_h;
                            double scalar_val = get_scalar_value(domain, iix, iiy, iiz);

                            Sf[i] += scalar_val * ib_delta(X[i] - xi, Y[i] - yi, Z[i] - zi, grid_h) * grid_h * grid_h *
                                     grid_h;
                        }
                    }
                }
                Fs[i] = Sp[i] - Sf[i];
            }
        }
        else if (pde_type == PDEBoundaryType::Neumann)
        {
            // Neumann: F = phi_ghost - grid_h * BC - Sf
            // where phi_ghost is interpolated at ghost point (X + Nx*dx, Y + Ny*dx, Z + Nz*dx)
            OPENMP_PARALLEL_FOR()
            for (int i = 0; i < particles->cur_n; i++)
            {
                // X[i], Y[i], Z[i] are global coordinates
                // ix, iy, iz are global grid indices
                int ix = static_cast<int>(std::floor(X[i] / grid_h));
                int iy = static_cast<int>(std::floor(Y[i] / grid_h));
                int iz = static_cast<int>(std::floor(Z[i] / grid_h));

                // For scalar: support domain is 5 points in x, y, and z
                int min_iix = ix - 2;
                int max_iix = ix + 2;
                int min_iiy = iy - 2;
                int max_iiy = iy + 2;
                int min_iiz = iz - 2;
                int max_iiz = iz + 2;

                // Interpolate fluid scalar at IB point
                Sf[i] = 0.0;
                for (int iix = min_iix; iix <= max_iix; iix++)
                {
                    for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                    {
                        for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                        {
                            double xi         = iix * grid_h;
                            double yi         = iiy * grid_h;
                            double zi         = iiz * grid_h;
                            double scalar_val = get_scalar_value(domain, iix, iiy, iiz);

                            Sf[i] += scalar_val * ib_delta(X[i] - xi, Y[i] - yi, Z[i] - zi, grid_h) * grid_h * grid_h *
                                     grid_h;
                        }
                    }
                }

                // Calculate ghost point position: X_ghost = X + Nx * dx
                double ghost_x = X[i] + Nx[i] * grid_h;
                double ghost_y = Y[i] + Ny[i] * grid_h;
                double ghost_z = Z[i] + Nz[i] * grid_h;

                // Get ghost point global grid indices
                int ghost_ix = static_cast<int>(std::floor(ghost_x / grid_h));
                int ghost_iy = static_cast<int>(std::floor(ghost_y / grid_h));
                int ghost_iz = static_cast<int>(std::floor(ghost_z / grid_h));

                // Ghost point support domain
                int ghost_min_iix = ghost_ix - 2;
                int ghost_max_iix = ghost_ix + 2;
                int ghost_min_iiy = ghost_iy - 2;
                int ghost_max_iiy = ghost_iy + 2;
                int ghost_min_iiz = ghost_iz - 2;
                int ghost_max_iiz = ghost_iz + 2;

                // Interpolate fluid scalar at ghost point
                double phi_ghost = 0.0;
                for (int iix = ghost_min_iix; iix <= ghost_max_iix; iix++)
                {
                    for (int iiy = ghost_min_iiy; iiy <= ghost_max_iiy; iiy++)
                    {
                        for (int iiz = ghost_min_iiz; iiz <= ghost_max_iiz; iiz++)
                        {
                            double xi         = iix * grid_h;
                            double yi         = iiy * grid_h;
                            double zi         = iiz * grid_h;
                            double scalar_val = get_scalar_value(domain, iix, iiy, iiz);

                            phi_ghost += scalar_val * ib_delta(ghost_x - xi, ghost_y - yi, ghost_z - zi, grid_h) *
                                         grid_h * grid_h * grid_h;
                        }
                    }
                }

                // Neumann BC: (phi_ghost - phi_ideal) / grid_h = BC
                // phi_ideal = phi_ghost - grid_h * BC
                // Force: F = phi_ideal - phi_ib = phi_ghost - grid_h * BC - Sf
                Fs[i] = phi_ghost - grid_h * neumann_bc - Sf[i];
            }
        }
    }
}

void IBScalarSolver3D_Uhlmann::apply_ib_scalar()
{
    // Process each domain in geometry tree
    for (auto* domain : scalar_var->geometry->domains)
    {
        auto* particles = coord_map[domain];
        auto* ib_data   = ib_map[domain];

        // Skip domains without particles
        if (!particles || particles->cur_n == 0)
            continue;

        EXPOSE_PCOORD3D(particles)
        EXPOSE_PIBSCALAR(ib_data)

        OPENMP_PARALLEL_FOR()
        for (int ib = 0; ib < particles->cur_n; ib++)
        {
            // Get particle global grid indices
            int ix = static_cast<int>(std::floor(X[ib] / grid_h));
            int iy = static_cast<int>(std::floor(Y[ib] / grid_h));
            int iz = static_cast<int>(std::floor(Z[ib] / grid_h));

            // Scalar support domain: ix in [ix-2, ix+2], iy in [iy-2, iy+2], iz in [iz-2, iz+2]
            int min_iix = ix - 2;
            int max_iix = ix + 2;
            int min_iiy = iy - 2;
            int max_iiy = iy + 2;
            int min_iiz = iz - 2;
            int max_iiz = iz + 2;

            for (int iix = min_iix; iix <= max_iix; iix++)
            {
                for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
                {
                    for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                    {
                        double xx = iix * grid_h;
                        double yy = iiy * grid_h;
                        double zz = iiz * grid_h;

                        double delta    = ib_delta(xx - X[ib], yy - Y[ib], zz - Z[ib], grid_h);
                        double ib_force = Fs[ib] * delta * ib_h * ib_h * grid_h;

                        get_scalar_value(domain, iix, iiy, iiz) += ib_force;
                    }
                }
            }
        }
    }
}
