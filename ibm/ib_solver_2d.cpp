#include "ib_solver_2d.h"

#include <cmath>

void ImmersedBoundarySolver2D::solve(FluidContext2D&              context,
                                     field2&                      u_recv,
                                     field2&                      v_recv,
                                     ImmersedBoundaryParticles2D& ib_particles)
{
    u2F(context, ib_particles);
    apply_ib_force(context, u_recv, v_recv, ib_particles, ib_particles.hx, ib_particles.hy);
}

void ImmersedBoundarySolver2D::u2F(FluidContext2D& context, ImmersedBoundaryParticles2D& particles)
{
    auto& u = *context.get_field("u");
    auto& v = *context.get_field("v");

    auto* u_buffer_y_neg = context.get_buffer("u_buffer_y_neg");
    auto* u_buffer_y_pos = context.get_buffer("u_buffer_y_pos");

    auto* v_buffer_x_neg = context.get_buffer("v_buffer_x_neg");
    auto* v_buffer_x_pos = context.get_buffer("v_buffer_x_pos");

    OPENMP_PARALLEL_FOR()
    for (int i = 0; i < particles.cur_n; i++)
    {
        // double position -> context index

        int ix = std::floor(particles.X[i] / hx);
        int iy = std::floor(particles.Y[i] / hy);

        int min_iix = std::clamp(ix - 1, 0, u.get_nx() - 1);
        int max_iix = std::clamp(ix + 2, 0, u.get_nx() - 1);
        int min_iiy = std::clamp(iy - 2, -1, u.get_ny());
        int max_iiy = std::clamp(iy + 2, -1, u.get_ny());

        particles.Uf[i] = 0.0;
        // context index -> velocity u index
        for (int iix = min_iix; iix <= max_iix; iix++)
        {
            for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
            {
                // velocity u index -> double position

                double xi = iix * hx;
                double yi = iiy * hy + 0.5 * hy;

                double u_value = 0.0;
                if (iiy == -1)
                {
                    u_value = u_buffer_y_neg[iix];
                }
                else if (iiy == u.get_ny())
                {
                    u_value = u_buffer_y_pos[iix];
                }
                else
                {
                    u_value = u(iix, iiy);
                }

                particles.Uf[i] += u_value * ib_delta(particles.X[i] - xi, particles.Y[i] - yi, hx, hy) * hx * hy;
            }
        }
        particles.Fx[i] = particles.Up[i] - particles.Uf[i];
        particles.Fx_sum[i] += particles.Fx[i];

        min_iix = std::clamp(ix - 2, -1, v.get_nx());
        max_iix = std::clamp(ix + 2, -1, v.get_nx());
        min_iiy = std::clamp(iy - 1, 0, v.get_ny() - 1);
        max_iiy = std::clamp(iy + 2, 0, v.get_ny() - 1);

        particles.Vf[i] = 0.0;
        // context index -> velocity v index
        for (int iix = min_iix; iix <= max_iix; iix++)
        {
            for (int iiy = min_iiy; iiy <= max_iiy; iiy++)
            {
                // velocity v index -> double position

                double xi = iix * hx + 0.5 * hx;
                double yi = iiy * hy;

                double v_value = 0.0;
                if (iix == -1)
                {
                    v_value = v_buffer_x_neg[iiy];
                }
                else if (iix == v.get_nx())
                {
                    v_value = v_buffer_x_pos[iiy];
                }
                else
                {
                    v_value = v(iix, iiy);
                }

                particles.Vf[i] += v_value * ib_delta(particles.X[i] - xi, particles.Y[i] - yi, hx, hy) * hx * hy;
            }
        }
        particles.Fy[i] = particles.Vp[i] - particles.Vf[i];
        particles.Fy_sum[i] += particles.Fy[i];
    }
}

void ImmersedBoundarySolver2D::apply_ib_force(FluidContext2D&              context,
                                              field2&                      u_recv,
                                              field2&                      v_recv,
                                              ImmersedBoundaryParticles2D& particles,
                                              double                       ib_hx,
                                              double                       ib_hy)
{
    EXPOSE_POINTS2D_BOUND(particles);

    // u
    OPENMP_PARALLEL_FOR()
    for (int i = min_ix_u; i <= max_ix_u; i++)
    {
        for (int j = min_iy_u; j <= max_iy_u; j++)
        {
            // velocity u index -> double position

            double xx = i * hx;
            double yy = j * hy + 0.5 * hy;

            for (int ib = 0; ib < particles.cur_n; ib++)
            {
                double ib_force =
                    particles.Fx[ib] * ib_delta(xx - particles.X[ib], yy - particles.Y[ib], hx, hy) * ib_hx * hy;

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
            // velocity v index -> double position

            double xx = i * hx + 0.5 * hx;
            double yy = j * hy;

            for (int ib = 0; ib < particles.cur_n; ib++)
            {
                double ib_force =
                    particles.Fy[ib] * ib_delta(xx - particles.X[ib], yy - particles.Y[ib], hx, hy) * hx * ib_hy;

                v_recv(i, j) += ib_force;
            }
        }
    }
}