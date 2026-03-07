#include "ib_solver_3d.h"

#include <cmath>

void ImmersedBoundarySolver3D::solve(FluidContext3D&              context,
                                     field3&                      u_recv,
                                     field3&                      v_recv,
                                     field3&                      w_recv,
                                     ImmersedBoundaryParticles3D& ib_particles)
{

    u2F(context, ib_particles);
    apply_ib_force(context, u_recv, v_recv, w_recv, ib_particles, ib_particles.hx, ib_particles.hy, ib_particles.hz);
}

void ImmersedBoundarySolver3D::u2F(FluidContext3D& context, ImmersedBoundaryParticles3D& particles)
{
    auto& u = *context.get_field("u");
    auto& v = *context.get_field("v");
    auto& w = *context.get_field("w");

    auto& u_buffer_y_neg = context.get_buffer("u_buffer_y_neg");
    auto& u_buffer_y_pos = context.get_buffer("u_buffer_y_pos");
    auto& u_buffer_z_neg = context.get_buffer("u_buffer_z_neg");
    auto& u_buffer_z_pos = context.get_buffer("u_buffer_z_pos");

    auto& v_buffer_x_neg = context.get_buffer("v_buffer_x_neg");
    auto& v_buffer_x_pos = context.get_buffer("v_buffer_x_pos");
    auto& v_buffer_z_neg = context.get_buffer("v_buffer_z_neg");
    auto& v_buffer_z_pos = context.get_buffer("v_buffer_z_pos");

    auto& w_buffer_x_neg = context.get_buffer("w_buffer_x_neg");
    auto& w_buffer_x_pos = context.get_buffer("w_buffer_x_pos");
    auto& w_buffer_y_neg = context.get_buffer("w_buffer_y_neg");
    auto& w_buffer_y_pos = context.get_buffer("w_buffer_y_pos");

    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < particles.cur_n; i++)
    {
        // double position -> context index

        int ix = std::floor(particles.X[i] / hz);
        int iy = std::floor(particles.Y[i] / hy);
        int iz = std::floor(particles.Z[i] / hz);

        int min_iix = std::clamp(ix - 1, 0, u.get_nx() - 1);
        int max_iix = std::clamp(ix + 2, 0, u.get_nx() - 1);
        int min_iiy = std::clamp(iy - 2, -1, u.get_ny());
        int max_iiy = std::clamp(iy + 2, -1, u.get_ny());
        int min_iiz = std::clamp(iz - 2, -1, u.get_nz());
        int max_iiz = std::clamp(iz + 2, -1, u.get_nz());

        particles.Uf[i] = 0.0;
        // context index -> velocity u index
        for (int iix = min_iix; iix <= max_iix; iix++)
        {
            for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
            {
                for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                {
                    // velocity u index -> double position

                    double xi = iix * hx;
                    double yi = iiy * hy + 0.5 * hy;
                    double zi = iiz * hz + 0.5 * hz;

                    double u_value = 0.0;
                    if (iiy == -1)
                    {
                        u_value = u_buffer_y_neg(iix, iiz);
                    }
                    else if (iiy == u.get_ny())
                    {
                        u_value = u_buffer_y_pos(iix, iiz);
                    }
                    else if (iiz == -1)
                    {
                        u_value = u_buffer_z_neg(iix, iiy);
                    }
                    else if (iiz == u.get_nz())
                    {
                        u_value = u_buffer_z_pos(iix, iiy);
                    }
                    else
                    {
                        u_value = u(iix, iiy, iiz);
                    }

                    particles.Uf[i] +=
                        u_value * ib_delta(particles.X[i] - xi, particles.Y[i] - yi, particles.Z[i] - zi, hx, hy, hz) *
                        hx * hy * hz;
                }
            }
        }
        particles.Fx[i] = particles.Up[i] - particles.Uf[i];
        particles.Fx_sum[i] += particles.Fx[i];

        min_iix = std::clamp(ix - 2, -1, v.get_nx());
        max_iix = std::clamp(ix + 2, -1, v.get_nx());
        min_iiy = std::clamp(iy - 1, 0, v.get_ny() - 1);
        max_iiy = std::clamp(iy + 2, 0, v.get_ny() - 1);
        min_iiz = std::clamp(iz - 2, -1, v.get_nz());
        max_iiz = std::clamp(iz + 2, -1, v.get_nz());

        particles.Vf[i] = 0.0;
        // context index -> velocity v index
        for (int iix = min_iix; iix <= max_iix; iix++)
        {
            for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
            {
                for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                {
                    // velocity v index -> double position

                    double xi = iix * hx + 0.5 * hx;
                    double yi = iiy * hy;
                    double zi = iiz * hz + 0.5 * hz;

                    double v_value = 0.0;
                    if (iix == -1)
                    {
                        v_value = v_buffer_x_neg(iiy, iiz);
                    }
                    else if (iix == v.get_nx())
                    {
                        v_value = v_buffer_x_pos(iiy, iiz);
                    }
                    else if (iiz == -1)
                    {
                        v_value = v_buffer_z_neg(iix, iiy);
                    }
                    else if (iiz == v.get_nz())
                    {
                        v_value = v_buffer_z_pos(iix, iiy);
                    }
                    else
                    {
                        v_value = v(iix, iiy, iiz);
                    }

                    particles.Vf[i] +=
                        v_value * ib_delta(particles.X[i] - xi, particles.Y[i] - yi, particles.Z[i] - zi, hx, hy, hz) *
                        hx * hy * hz;
                }
            }
        }
        particles.Fy[i] = particles.Vp[i] - particles.Vf[i];
        particles.Fy_sum[i] += particles.Fy[i];

        min_iix = std::clamp(ix - 2, -1, w.get_nx());
        max_iix = std::clamp(ix + 2, -1, w.get_nx());
        min_iiy = std::clamp(iy - 2, -1, w.get_ny());
        max_iiy = std::clamp(iy + 2, -1, w.get_ny());
        min_iiz = std::clamp(iz - 1, 0, w.get_nz() - 1);
        max_iiz = std::clamp(iz + 2, 0, w.get_nz() - 1);

        particles.Wf[i] = 0.0;
        // context index -> velocity v index
        for (int iix = min_iix; iix <= max_iix; iix++)
        {
            for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
            {
                for (int iiz = min_iiz; iiz <= max_iiz; iiz++)
                {
                    // velocity w index -> double position

                    double xi = iix * hx + 0.5 * hx;
                    double yi = iiy * hy + 0.5 * hy;
                    double zi = iiz * hz;

                    double w_value = 0.0;
                    if (iix == -1)
                    {
                        w_value = w_buffer_x_neg(iix, iiy);
                    }
                    else if (iix == w.get_nx())
                    {
                        w_value = w_buffer_x_pos(iix, iiy);
                    }
                    else if (iiy == -1)
                    {
                        w_value = w_buffer_y_neg(iix, iiz);
                    }
                    else if (iiy == w.get_ny())
                    {
                        w_value = w_buffer_y_pos(iix, iiz);
                    }
                    else
                    {
                        w_value = w(iix, iiy, iiz);
                    }

                    particles.Wf[i] +=
                        w_value * ib_delta(particles.X[i] - xi, particles.Y[i] - yi, particles.Z[i] - zi, hx, hy, hz) *
                        hx * hy * hz;
                }
            }
        }
        particles.Fz[i] = particles.Wp[i] - particles.Wf[i];
        particles.Fz_sum[i] += particles.Fz[i];
    }
}

void ImmersedBoundarySolver3D::apply_ib_force(FluidContext3D&              context,
                                              field3&                      u_recv,
                                              field3&                      v_recv,
                                              field3&                      w_recv,
                                              ImmersedBoundaryParticles3D& particles,
                                              double                       ib_hx,
                                              double                       ib_hy,
                                              double                       ib_hz)
{
    EXPOSE_POINTS3D_BOUND(particles);

    // u
    OPENMP_PARALLEL_FOR()
    for (int i = min_ix_u; i <= max_ix_u; i++)
    {
        for (int j = min_iy_u; j <= max_iy_u; j++)
        {
            for (int k = min_iz_u; k <= max_iz_u; k++)
            {
                // velocity u index -> double position

                double xx = i * hx;
                double yy = j * hy + 0.5 * hy;
                double zz = k * hz + 0.5 * hz;

                for (int ib = 0; ib < particles.cur_n; ib++)
                {
                    double ib_force =
                        particles.Fx[ib] *
                        ib_delta(xx - particles.X[ib], yy - particles.Y[ib], zz - particles.Z[ib], hx, hy, hz) * ib_hx *
                        ib_hy * hz;

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
                // velocity v index -> double position

                double xx = i * hx + 0.5 * hx;
                double yy = j * hy;
                double zz = k * hz + 0.5 * hz;

                for (int ib = 0; ib < particles.cur_n; ib++)
                {
                    double ib_force =
                        particles.Fy[ib] *
                        ib_delta(xx - particles.X[ib], yy - particles.Y[ib], zz - particles.Z[ib], hx, hy, hz) * ib_hx *
                        hy * ib_hz;

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
                // velocity w index -> double position

                double xx = i * hx + 0.5 * hx;
                double yy = j * hy + 0.5 * hy;
                double zz = k * hz;

                for (int ib = 0; ib < particles.cur_n; ib++)
                {
                    double ib_force = particles.Fz[ib] *
                                      ib_delta(xx - particles.X[ib], yy - particles.Y[ib], zz - particles.Z[ib], h) *
                                      hx * ib_hy * ib_hz;

                    w_recv(i, j, k) += ib_force;
                }
            }
        }
    }
}
