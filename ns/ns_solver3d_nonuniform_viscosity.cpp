#include "ns_solver3d_nonuniform_viscosity.h"

#include "boundary_3d_utils.h"
#include <iomanip>
#include <iostream>

void NSSolver3DNonUniVisc::euler_conv_diff_inner()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];
        field3& c = *c_var->field_map[domain];

        field3& u_temp = *u_temp_field_map[domain];
        field3& v_temp = *v_temp_field_map[domain];
        field3& w_temp = *w_temp_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        // u (interior only; boundaries handled in euler_conv_diff_outer)
        // OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    double c_avg = (c(i, j, k) + c(i - 1, j, k)) / 2.0;
                    double mu    = std::pow(mu1_mu2, std::clamp(c_avg, 0.0, 1.0));

                    double conv_x = 0.25 / hx *
                                    (u(i + 1, j, k) * (u(i + 1, j, k) + 2.0 * u(i, j, k)) -
                                     u(i - 1, j, k) * (u(i - 1, j, k) + 2.0 * u(i, j, k)));
                    double conv_y = 0.25 / hy *
                                    ((u(i, j, k) + u(i, j + 1, k)) * (v(i - 1, j + 1, k) + v(i, j + 1, k)) -
                                     (u(i, j - 1, k) + u(i, j, k)) * (v(i - 1, j, k) + v(i, j, k)));
                    double conv_z = 0.25 / hz *
                                    ((u(i, j, k) + u(i, j, k + 1)) * (w(i - 1, j, k + 1) + w(i, j, k + 1)) -
                                     (u(i, j, k - 1) + u(i, j, k)) * (w(i - 1, j, k) + w(i, j, k)));
                    double diffuse_x = mu * nu / hx / hx * (u(i + 1, j, k) - 2.0 * u(i, j, k) + u(i - 1, j, k));
                    double diffuse_y = mu * nu / hy / hy * (u(i, j + 1, k) - 2.0 * u(i, j, k) + u(i, j - 1, k));
                    double diffuse_z = mu * nu / hz / hz * (u(i, j, k + 1) - 2.0 * u(i, j, k) + u(i, j, k - 1));

                    double u_old    = u(i, j, k);
                    u_temp(i, j, k) = u(i, j, k) - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);

                    // Debug output at specific points
                    if (i == nx / 2 && j == ny / 2 && k == nz / 2)
                    {
                        std::cout << "[U-CENTER] i=" << i << " j=" << j << " k=" << k
                                  << " u_old=" << std::setprecision(10) << u_old << " u_new=" << u_temp(i, j, k)
                                  << " conv_x=" << conv_x << " conv_y=" << conv_y << " conv_z=" << conv_z
                                  << " diff_x=" << diffuse_x << " diff_y=" << diffuse_y << " diff_z=" << diffuse_z
                                  << " mu=" << mu << " c_avg=" << c_avg << std::endl;
                    }
                    if (i == 1 && j == ny / 2 && k == nz / 2)
                    {
                        std::cout << "[U-XMIN] i=" << i << " j=" << j << " k=" << k
                                  << " u_old=" << std::setprecision(10) << u_old << " u_new=" << u_temp(i, j, k)
                                  << std::endl;
                    }
                    if (i == nx - 2 && j == ny / 2 && k == nz / 2)
                    {
                        std::cout << "[U-XMAX] i=" << i << " j=" << j << " k=" << k
                                  << " u_old=" << std::setprecision(10) << u_old << " u_new=" << u_temp(i, j, k)
                                  << std::endl;
                    }
                }
            }
        }

        // v
        // OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    double c_avg = (c(i, j, k) + c(i, j - 1, k)) / 2.0;
                    double mu    = std::pow(mu1_mu2, std::clamp(c_avg, 0.0, 1.0));

                    double conv_x = 0.25 / hx *
                                    ((v(i, j, k) + v(i + 1, j, k)) * (u(i + 1, j - 1, k) + u(i + 1, j, k)) -
                                     (v(i - 1, j, k) + v(i, j, k)) * (u(i, j - 1, k) + u(i, j, k)));
                    double conv_y = 0.25 / hy *
                                    (v(i, j + 1, k) * (v(i, j + 1, k) + 2.0 * v(i, j, k)) -
                                     v(i, j - 1, k) * (v(i, j - 1, k) + 2.0 * v(i, j, k)));
                    double conv_z = 0.25 / hz *
                                    ((v(i, j, k) + v(i, j, k + 1)) * (w(i, j - 1, k + 1) + w(i, j, k + 1)) -
                                     (v(i, j, k - 1) + v(i, j, k)) * (w(i, j - 1, k) + w(i, j, k)));
                    double diffuse_x = mu * nu / hx / hx * (v(i + 1, j, k) - 2.0 * v(i, j, k) + v(i - 1, j, k));
                    double diffuse_y = mu * nu / hy / hy * (v(i, j + 1, k) - 2.0 * v(i, j, k) + v(i, j - 1, k));
                    double diffuse_z = mu * nu / hz / hz * (v(i, j, k + 1) - 2.0 * v(i, j, k) + v(i, j, k - 1));

                    double v_old    = v(i, j, k);
                    v_temp(i, j, k) = v(i, j, k) - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);

                    // Debug output at specific points
                    if (i == nx / 2 && j == ny / 2 && k == nz / 2)
                    {
                        std::cout << "[V-CENTER] i=" << i << " j=" << j << " k=" << k
                                  << " v_old=" << std::setprecision(10) << v_old << " v_new=" << v_temp(i, j, k)
                                  << " conv_x=" << conv_x << " conv_y=" << conv_y << " conv_z=" << conv_z
                                  << " diff_x=" << diffuse_x << " diff_y=" << diffuse_y << " diff_z=" << diffuse_z
                                  << " mu=" << mu << " c_avg=" << c_avg << std::endl;
                    }
                    if (i == nx / 2 && j == 1 && k == nz / 2)
                    {
                        std::cout << "[V-YMIN] i=" << i << " j=" << j << " k=" << k
                                  << " v_old=" << std::setprecision(10) << v_old << " v_new=" << v_temp(i, j, k)
                                  << std::endl;
                    }
                    if (i == nx / 2 && j == ny - 2 && k == nz / 2)
                    {
                        std::cout << "[V-YMAX] i=" << i << " j=" << j << " k=" << k
                                  << " v_old=" << std::setprecision(10) << v_old << " v_new=" << v_temp(i, j, k)
                                  << std::endl;
                    }
                }
            }
        }

        // w
        // OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    double c_avg = (c(i, j, k) + c(i, j, k - 1)) / 2.0;
                    double mu    = std::pow(mu1_mu2, std::clamp(c_avg, 0.0, 1.0));

                    double conv_x = 0.25 / hx *
                                    ((w(i, j, k) + w(i + 1, j, k)) * (u(i + 1, j, k - 1) + u(i + 1, j, k)) -
                                     (w(i - 1, j, k) + w(i, j, k)) * (u(i, j, k - 1) + u(i, j, k)));
                    double conv_y = 0.25 / hy *
                                    ((w(i, j, k) + w(i, j + 1, k)) * (v(i, j + 1, k - 1) + v(i, j + 1, k)) -
                                     (w(i, j - 1, k) + w(i, j, k)) * (v(i, j, k - 1) + v(i, j, k)));
                    double conv_z = 0.25 / hz *
                                    (w(i, j, k + 1) * (w(i, j, k + 1) + 2.0 * w(i, j, k)) -
                                     w(i, j, k - 1) * (w(i, j, k - 1) + 2.0 * w(i, j, k)));
                    double diffuse_x = mu * nu / hx / hx * (w(i + 1, j, k) - 2.0 * w(i, j, k) + w(i - 1, j, k));
                    double diffuse_y = mu * nu / hy / hy * (w(i, j + 1, k) - 2.0 * w(i, j, k) + w(i, j - 1, k));
                    double diffuse_z = mu * nu / hz / hz * (w(i, j, k + 1) - 2.0 * w(i, j, k) + w(i, j, k - 1));

                    double w_old    = w(i, j, k);
                    w_temp(i, j, k) = w(i, j, k) - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);

                    // Debug output at specific points
                    if (i == nx / 2 && j == ny / 2 && k == nz / 2)
                    {
                        std::cout << "[W-CENTER] i=" << i << " j=" << j << " k=" << k
                                  << " w_old=" << std::setprecision(10) << w_old << " w_new=" << w_temp(i, j, k)
                                  << " conv_x=" << conv_x << " conv_y=" << conv_y << " conv_z=" << conv_z
                                  << " diff_x=" << diffuse_x << " diff_y=" << diffuse_y << " diff_z=" << diffuse_z
                                  << " mu=" << mu << " c_avg=" << c_avg << std::endl;
                    }
                    if (i == nx / 2 && j == ny / 2 && k == 1)
                    {
                        std::cout << "[W-ZMIN] i=" << i << " j=" << j << " k=" << k
                                  << " w_old=" << std::setprecision(10) << w_old << " w_new=" << w_temp(i, j, k)
                                  << std::endl;
                    }
                    if (i == nx / 2 && j == ny / 2 && k == nz - 2)
                    {
                        std::cout << "[W-ZMAX] i=" << i << " j=" << j << " k=" << k
                                  << " w_old=" << std::setprecision(10) << w_old << " w_new=" << w_temp(i, j, k)
                                  << std::endl;
                    }
                }
            }
        }
    }
}

void NSSolver3DNonUniVisc::euler_conv_diff_outer()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];
        field3& c = *c_var->field_map[domain];

        field3& u_temp = *u_temp_field_map[domain];
        field3& v_temp = *v_temp_field_map[domain];
        field3& w_temp = *w_temp_field_map[domain];

        field2& u_xneg_buffer = *u_buffer_map[domain][LocationType::XNegative];
        field2& u_xpos_buffer = *u_buffer_map[domain][LocationType::XPositive];
        field2& u_yneg_buffer = *u_buffer_map[domain][LocationType::YNegative];
        field2& u_ypos_buffer = *u_buffer_map[domain][LocationType::YPositive];
        field2& u_zneg_buffer = *u_buffer_map[domain][LocationType::ZNegative];
        field2& u_zpos_buffer = *u_buffer_map[domain][LocationType::ZPositive];

        field2& v_xneg_buffer = *v_buffer_map[domain][LocationType::XNegative];
        field2& v_xpos_buffer = *v_buffer_map[domain][LocationType::XPositive];
        field2& v_yneg_buffer = *v_buffer_map[domain][LocationType::YNegative];
        field2& v_ypos_buffer = *v_buffer_map[domain][LocationType::YPositive];
        field2& v_zneg_buffer = *v_buffer_map[domain][LocationType::ZNegative];
        field2& v_zpos_buffer = *v_buffer_map[domain][LocationType::ZPositive];

        field2& w_xneg_buffer = *w_buffer_map[domain][LocationType::XNegative];
        field2& w_xpos_buffer = *w_buffer_map[domain][LocationType::XPositive];
        field2& w_yneg_buffer = *w_buffer_map[domain][LocationType::YNegative];
        field2& w_ypos_buffer = *w_buffer_map[domain][LocationType::YPositive];
        field2& w_zneg_buffer = *w_buffer_map[domain][LocationType::ZNegative];
        field2& w_zpos_buffer = *w_buffer_map[domain][LocationType::ZPositive];

        field2& c_xneg_buffer = *c_var->buffer_map[domain][LocationType::XNegative];
        field2& c_yneg_buffer = *c_var->buffer_map[domain][LocationType::YNegative];
        field2& c_zneg_buffer = *c_var->buffer_map[domain][LocationType::ZNegative];

        double* u_corner_along_y = u_corner_y_map[domain];
        double* u_corner_along_z = u_corner_z_map[domain];
        double* v_corner_along_x = v_corner_x_map[domain];
        double* v_corner_along_z = v_corner_z_map[domain];
        double* w_corner_along_x = w_corner_x_map[domain];
        double* w_corner_along_y = w_corner_y_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        auto bound_cal_u = [&](int i, int j, int k) {
            double u_ijk = u(i, j, k);
            double u_im1 = i == 0 ? u_xneg_buffer(j, k) : u(i - 1, j, k);
            double u_ip1 = i == nx - 1 ? u_xpos_buffer(j, k) : u(i + 1, j, k);
            double u_jm1 = j == 0 ? u_yneg_buffer(i, k) : u(i, j - 1, k);
            double u_jp1 = j == ny - 1 ? u_ypos_buffer(i, k) : u(i, j + 1, k);
            double u_km1 = k == 0 ? u_zneg_buffer(i, j) : u(i, j, k - 1);
            double u_kp1 = k == nz - 1 ? u_zpos_buffer(i, j) : u(i, j, k + 1);

            double v_im1_jp1 = i == 0 ? (j == ny - 1 ? v_corner_along_z[k] : v_xneg_buffer(j + 1, k)) :
                                        (j == ny - 1 ? v_ypos_buffer(i - 1, k) : v(i - 1, j + 1, k));
            double v_jp1     = j == ny - 1 ? v_ypos_buffer(i, k) : v(i, j + 1, k);
            double v_im1     = i == 0 ? v_xneg_buffer(j, k) : v(i - 1, j, k);

            double w_im1_kp1 = i == 0 ? (k == nz - 1 ? w_corner_along_y[j] : w_xneg_buffer(j, k + 1)) :
                                        (k == nz - 1 ? w_zpos_buffer(i - 1, j) : w(i - 1, j, k + 1));
            double w_kp1     = k == nz - 1 ? w_zpos_buffer(i, j) : w(i, j, k + 1);
            double w_im1     = i == 0 ? w_xneg_buffer(j, k) : w(i - 1, j, k);

            double c_ijk = c(i, j, k);
            double c_im1 = i == 0 ? c_xneg_buffer(j, k) : c(i - 1, j, k);

            double c_avg = (c_ijk + c_im1) / 2.0;
            double mu    = std::pow(mu1_mu2, std::clamp(c_avg, 0.0, 1.0));

            double conv_x = 0.25 / hx * (u_ip1 * (u_ip1 + 2.0 * u_ijk) - u_im1 * (u_im1 + 2.0 * u_ijk));
            double conv_y =
                0.25 / hy * ((u_ijk + u_jp1) * (v_im1_jp1 + v_jp1) - (u_jm1 + u_ijk) * (v_im1 + v(i, j, k)));
            double conv_z =
                0.25 / hz * ((u_ijk + u_kp1) * (w_im1_kp1 + w_kp1) - (u_km1 + u_ijk) * (w_im1 + w(i, j, k)));
            double diffuse_x = mu * nu / hx / hx * (u_ip1 - 2.0 * u_ijk + u_im1);
            double diffuse_y = mu * nu / hy / hy * (u_jp1 - 2.0 * u_ijk + u_jm1);
            double diffuse_z = mu * nu / hz / hz * (u_kp1 - 2.0 * u_ijk + u_km1);

            u_temp(i, j, k) = u_ijk - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
        };

        auto bound_cal_v = [&](int i, int j, int k) {
            double v_ijk = v(i, j, k);
            double v_im1 = i == 0 ? v_xneg_buffer(j, k) : v(i - 1, j, k);
            double v_ip1 = i == nx - 1 ? v_xpos_buffer(j, k) : v(i + 1, j, k);
            double v_jm1 = j == 0 ? v_yneg_buffer(i, k) : v(i, j - 1, k);
            double v_jp1 = j == ny - 1 ? v_ypos_buffer(i, k) : v(i, j + 1, k);
            double v_km1 = k == 0 ? v_zneg_buffer(i, j) : v(i, j, k - 1);
            double v_kp1 = k == nz - 1 ? v_zpos_buffer(i, j) : v(i, j, k + 1);

            double u_ip1_jm1 = i == nx - 1 ? (j == 0 ? u_corner_along_z[k] : u_xpos_buffer(j - 1, k)) :
                                             (j == 0 ? u_yneg_buffer(i + 1, k) : u(i + 1, j - 1, k));
            double u_ip1     = i == nx - 1 ? u_xpos_buffer(j, k) : u(i + 1, j, k);
            double u_jm1     = j == 0 ? u_yneg_buffer(i, k) : u(i, j - 1, k);

            double w_jm1_kp1 = j == 0 ? (k == nz - 1 ? w_corner_along_x[i] : w_yneg_buffer(i, k + 1)) :
                                        (k == nz - 1 ? w_zpos_buffer(i, j - 1) : w(i, j - 1, k + 1));
            double w_kp1     = k == nz - 1 ? w_zpos_buffer(i, j) : w(i, j, k + 1);
            double w_jm1     = j == 0 ? w_yneg_buffer(i, k) : w(i, j - 1, k);

            double c_ijk = c(i, j, k);
            double c_jm1 = j == 0 ? c_yneg_buffer(i, k) : c(i, j - 1, k);

            double c_avg = (c_ijk + c_jm1) / 2.0;
            double mu    = std::pow(mu1_mu2, std::clamp(c_avg, 0.0, 1.0));

            double conv_x =
                0.25 / hx * ((v_ijk + v_ip1) * (u_ip1_jm1 + u_ip1) - (v_im1 + v_ijk) * (u_jm1 + u(i, j, k)));
            double conv_y = 0.25 / hy * (v_jp1 * (v_jp1 + 2.0 * v_ijk) - v_jm1 * (v_jm1 + 2.0 * v_ijk));
            double conv_z =
                0.25 / hz * ((v_ijk + v_kp1) * (w_jm1_kp1 + w_kp1) - (v_km1 + v_ijk) * (w_jm1 + w(i, j, k)));
            double diffuse_x = mu * nu / hx / hx * (v_ip1 - 2.0 * v_ijk + v_im1);
            double diffuse_y = mu * nu / hy / hy * (v_jp1 - 2.0 * v_ijk + v_jm1);
            double diffuse_z = mu * nu / hz / hz * (v_kp1 - 2.0 * v_ijk + v_km1);

            v_temp(i, j, k) = v_ijk - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
        };

        auto bound_cal_w = [&](int i, int j, int k) {
            double w_ijk = w(i, j, k);
            double w_im1 = i == 0 ? w_xneg_buffer(j, k) : w(i - 1, j, k);
            double w_ip1 = i == nx - 1 ? w_xpos_buffer(j, k) : w(i + 1, j, k);
            double w_jm1 = j == 0 ? w_yneg_buffer(i, k) : w(i, j - 1, k);
            double w_jp1 = j == ny - 1 ? w_ypos_buffer(i, k) : w(i, j + 1, k);
            double w_km1 = k == 0 ? w_zneg_buffer(i, j) : w(i, j, k - 1);
            double w_kp1 = k == nz - 1 ? w_zpos_buffer(i, j) : w(i, j, k + 1);

            double u_ip1_km1 = i == nx - 1 ? (k == 0 ? u_corner_along_y[j] : u_xpos_buffer(j, k - 1)) :
                                             (k == 0 ? u_zneg_buffer(i + 1, j) : u(i + 1, j, k - 1));
            double u_ip1     = i == nx - 1 ? u_xpos_buffer(j, k) : u(i + 1, j, k);
            double u_km1     = k == 0 ? u_zneg_buffer(i, j) : u(i, j, k - 1);

            double v_jp1_km1 = j == ny - 1 ? (k == 0 ? v_corner_along_x[i] : v_ypos_buffer(i, k - 1)) :
                                             (k == 0 ? v_zneg_buffer(i, j + 1) : v(i, j + 1, k - 1));
            double v_jp1     = j == ny - 1 ? v_ypos_buffer(i, k) : v(i, j + 1, k);
            double v_km1     = k == 0 ? v_zneg_buffer(i, j) : v(i, j, k - 1);

            double c_ijk = c(i, j, k);
            double c_km1 = k == 0 ? c_zneg_buffer(i, j) : c(i, j, k - 1);

            double c_avg = (c_ijk + c_km1) / 2.0;
            double mu    = std::pow(mu1_mu2, std::clamp(c_avg, 0.0, 1.0));

            double conv_x =
                0.25 / hx * ((w_ijk + w_ip1) * (u_ip1_km1 + u_ip1) - (w_im1 + w_ijk) * (u_km1 + u(i, j, k)));
            double conv_y =
                0.25 / hy * ((w_ijk + w_jp1) * (v_jp1_km1 + v_jp1) - (w_jm1 + w_ijk) * (v_km1 + v(i, j, k)));
            double conv_z = 0.25 / hz * (w_kp1 * (w_kp1 + 2.0 * w_ijk) - w_km1 * (w_km1 + 2.0 * w_ijk));

            double diffuse_x = mu * nu / hx / hx * (w_ip1 - 2.0 * w_ijk + w_im1);
            double diffuse_y = mu * nu / hy / hy * (w_jp1 - 2.0 * w_ijk + w_jm1);
            double diffuse_z = mu * nu / hz / hz * (w_kp1 - 2.0 * w_ijk + w_km1);

            w_temp(i, j, k) = w_ijk - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
        };

        // OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                if (u_var->boundary_type_map[domain][LocationType::XNegative] == PDEBoundaryType::Adjacented)
                    bound_cal_u(0, j, k);
                bound_cal_u(nx - 1, j, k);
                bound_cal_v(0, j, k);
                bound_cal_v(nx - 1, j, k);
                bound_cal_w(0, j, k);
                bound_cal_w(nx - 1, j, k);
            }
        }

        // OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int k = 0; k < nz; k++)
            {
                bound_cal_u(i, 0, k);
                bound_cal_u(i, ny - 1, k);
                if (v_var->boundary_type_map[domain][LocationType::YNegative] == PDEBoundaryType::Adjacented)
                    bound_cal_v(i, 0, k);
                bound_cal_v(i, ny - 1, k);
                bound_cal_w(i, 0, k);
                bound_cal_w(i, ny - 1, k);
            }
        }

        // OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                bound_cal_u(i, j, 0);
                bound_cal_u(i, j, nz - 1);
                bound_cal_v(i, j, 0);
                bound_cal_v(i, j, nz - 1);
                if (w_var->boundary_type_map[domain][LocationType::ZNegative] == PDEBoundaryType::Adjacented)
                    bound_cal_w(i, j, 0);
                bound_cal_w(i, j, nz - 1);
            }
        }

        swap_field_data(u, u_temp);
        swap_field_data(v, v_temp);
        swap_field_data(w, w_temp);
    }
}