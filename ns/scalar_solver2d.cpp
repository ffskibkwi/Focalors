#include "scalar_solver2d.h"

#include "base/config.h"
#include "boundary_2d_utils.h"

ScalarSolver2D::ScalarSolver2D(Variable2D*          in_u_var,
                               Variable2D*          in_v_var,
                               Variable2D*          in_s_var,
                               double               _nr,
                               DifferenceSchemeType _scheme)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , s_var(in_s_var)
{
    TimeAdvancingConfig& time_cfg    = TimeAdvancingConfig::Get();
    PhysicsConfig&       physics_cfg = PhysicsConfig::Get();

    dt     = time_cfg.dt;
    nr     = _nr;
    scheme = _scheme;

    // check u v p share one geometry
    if (u_var->geometry != v_var->geometry || u_var->geometry != s_var->geometry)
        throw std::runtime_error("ScalarSolver2D: u v p do not share one geometry");

    // geometry double check
    if (u_var->geometry == nullptr)
        throw std::runtime_error("ScalarSolver2D: u->geometry is nrll");
    if (!u_var->geometry->is_checked)
        u_var->geometry->check();
    if (u_var->geometry->tree_root == nullptr || u_var->geometry->tree_map.empty())
        u_var->geometry->solve_prepare();

    domains   = u_var->geometry->domains;
    adjacency = u_var->geometry->adjacency;

    u_field_map = u_var->field_map;
    v_field_map = v_var->field_map;
    s_field_map = s_var->field_map;

    u_buffer_map = u_var->buffer_map;
    v_buffer_map = v_var->buffer_map;
    s_buffer_map = s_var->buffer_map;

    // Construct the temp field for each domain
    for (auto& [domain, field] : s_field_map)
        s_temp_field_map[domain] =
            new field2(field->get_nx(), field->get_ny(), field->get_name() + "_temp");

    // update boundary for first step
    phys_boundary_update();
    nondiag_shared_boundary_update();
}

void ScalarSolver2D::variable_check()
{
    if (u_var->position_type != VariablePositionType::XFace)
        throw std::runtime_error("ScalarSolver2D: u->position_type is not XFace");
    if (v_var->position_type != VariablePositionType::YFace)
        throw std::runtime_error("ScalarSolver2D: v->position_type is not YFace");
    if (s_var->position_type != VariablePositionType::Center)
        throw std::runtime_error("ScalarSolver2D: p->position_type is not Center");
}

void ScalarSolver2D::solve()
{
    switch (scheme)
    {
        case DifferenceSchemeType::Conv_Center2nd_Diff_Center2nd:
            conv_cd2nd_diff_cd2nd_inner();
            conv_cd2nd_diff_cd2nd_outer();
            break;
        case DifferenceSchemeType::Conv_Upwind1st_Diff_Center2nd:
            conv_uw1st_diff_cd2nd_inner();
            conv_uw1st_diff_cd2nd_outer_width1();
            break;
        case DifferenceSchemeType::Conv_QUICK_Diff_Center2nd:
            conv_QUICK_diff_cd2nd_inner();
            conv_uw1st_diff_cd2nd_outer_width2();
            break;
        case DifferenceSchemeType::Conv_TVD_VanLeer_Diff_Center2nd:
            conv_TVD_VanLeer_diff_cd2nd_inner();
            conv_uw1st_diff_cd2nd_outer_width2();
            break;
        default:
            std::cerr << "ScalarSolver2D " << scheme << " is not implemented!" << std::endl;
            break;
    }

    // update boundary at last to ensure other solver get xpos value at boundary
    phys_boundary_update();
    nondiag_shared_boundary_update();
}

void ScalarSolver2D::conv_cd2nd_diff_cd2nd_inner()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& s = *s_field_map[domain];

        field2& s_temp = *s_temp_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                double u_ijk = u(i, j);
                double v_ijk = v(i, j);
                double u_ip1 = u(i + 1, j);
                double v_jp1 = v(i, j + 1);

                double s_ijk = s(i, j);
                double s_im1 = s(i - 1, j);
                double s_ip1 = s(i + 1, j);
                double s_jm1 = s(i, j - 1);
                double s_jp1 = s(i, j + 1);

                double conv_x    = 0.5 / hx * (u_ip1 * (s_ip1 + s_ijk) - u_ijk * (s_im1 + s_ijk));
                double conv_y    = 0.5 / hy * (v_jp1 * (s_jp1 + s_ijk) - v_ijk * (s_jm1 + s_ijk));
                double diffuse_x = nr / hx / hx * (s_ip1 - 2.0 * s_ijk + s_im1);
                double diffuse_y = nr / hy / hy * (s_jp1 - 2.0 * s_ijk + s_jm1);

                s_temp(i, j) = s(i, j) - dt * (conv_x + conv_y - diffuse_x - diffuse_y);
            }
        }
    }
}

void ScalarSolver2D::conv_cd2nd_diff_cd2nd_outer()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& s = *s_field_map[domain];

        field2& s_temp = *s_temp_field_map[domain];

        double* u_xpos_buffer = u_buffer_map[domain][LocationType::XPositive];
        double* v_ypos_buffer = v_buffer_map[domain][LocationType::YPositive];

        double* s_xneg_buffer = s_buffer_map[domain][LocationType::XNegative];
        double* s_xpos_buffer = s_buffer_map[domain][LocationType::XPositive];
        double* s_yneg_buffer = s_buffer_map[domain][LocationType::YNegative];
        double* s_ypos_buffer = s_buffer_map[domain][LocationType::YPositive];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        auto bound_cal_s = [&](int i, int j) {
            double u_ijk = u(i, j);
            double v_ijk = v(i, j);
            double u_ip1 = i == nx - 1 ? u_xpos_buffer[j] : u(i + 1, j);
            double v_jp1 = j == ny - 1 ? v_ypos_buffer[i] : v(i, j + 1);

            double s_ijk = s(i, j);
            double s_im1 = i == 0 ? s_xneg_buffer[j] : s(i - 1, j);
            double s_ip1 = i == nx - 1 ? s_xpos_buffer[j] : s(i + 1, j);
            double s_jm1 = j == 0 ? s_yneg_buffer[i] : s(i, j - 1);
            double s_jp1 = j == ny - 1 ? s_ypos_buffer[i] : s(i, j + 1);

            double conv_x    = 0.5 / hx * (u_ip1 * (s_ip1 + s_ijk) - u_ijk * (s_im1 + s_ijk));
            double conv_y    = 0.5 / hy * (v_jp1 * (s_jp1 + s_ijk) - v_ijk * (s_jm1 + s_ijk));
            double diffuse_x = nr / hx / hx * (s_ip1 - 2.0 * s_ijk + s_im1);
            double diffuse_y = nr / hy / hy * (s_jp1 - 2.0 * s_ijk + s_jm1);

            s_temp(i, j) = s_ijk - dt * (conv_x + conv_y - diffuse_x - diffuse_y);
        };

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny; j++)
        {
            bound_cal_s(0, j);
            bound_cal_s(nx - 1, j);
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            bound_cal_s(i, 0);
            bound_cal_s(i, ny - 1);
        }

        swap_field_data(s, s_temp);
    }
}

void ScalarSolver2D::conv_uw1st_diff_cd2nd_inner()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& s = *s_field_map[domain];

        field2& s_temp = *s_temp_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                double u_ijk = u(i, j);
                double v_ijk = v(i, j);
                double u_ip1 = u(i + 1, j);
                double v_jp1 = v(i, j + 1);

                double s_ijk = s(i, j);
                double s_im1 = s(i - 1, j);
                double s_ip1 = s(i + 1, j);
                double s_jm1 = s(i, j - 1);
                double s_jp1 = s(i, j + 1);

                /**
                 * Convective Term: First-Order Upwind Scheme
                 * Logic: If velocity > 0, transport from upstream (xneg/yneg).
                 * If velocity < 0, transport from downstream (xpos/ypos).
                 * This eliminates oscillations by adding numerical diffusion.
                 */

                double flux_x_xpos = (u_ip1 > 0) ? (u_ip1 * s_ijk) : (u_ip1 * s_ip1);
                double flux_x_xneg = (u_ijk > 0) ? (u_ijk * s_im1) : (u_ijk * s_ijk);
                double conv_x      = (flux_x_xpos - flux_x_xneg) / hx;

                double flux_y_ypos = (v_jp1 > 0) ? (v_jp1 * s_ijk) : (v_jp1 * s_jp1);
                double flux_y_yneg = (v_ijk > 0) ? (v_ijk * s_jm1) : (v_ijk * s_ijk);
                double conv_y      = (flux_y_ypos - flux_y_yneg) / hy;

                /**
                 * Diffusive Term: Second-Order Central Difference
                 * Stable as long as dt <= (Pe * h^2) / 6.
                 */

                double diffuse_x = nr / hx / hx * (s_ip1 - 2.0 * s_ijk + s_im1);
                double diffuse_y = nr / hy / hy * (s_jp1 - 2.0 * s_ijk + s_jm1);

                s_temp(i, j) = s(i, j) - dt * (conv_x + conv_y - diffuse_x - diffuse_y);
            }
        }
    }
}

void ScalarSolver2D::conv_uw1st_diff_cd2nd_outer_width1()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& s = *s_field_map[domain];

        field2& s_temp = *s_temp_field_map[domain];

        double* u_xpos_buffer = u_buffer_map[domain][LocationType::XPositive];
        double* v_ypos_buffer = v_buffer_map[domain][LocationType::YPositive];

        double* s_xneg_buffer = s_buffer_map[domain][LocationType::XNegative];
        double* s_xpos_buffer = s_buffer_map[domain][LocationType::XPositive];
        double* s_yneg_buffer = s_buffer_map[domain][LocationType::YNegative];
        double* s_ypos_buffer = s_buffer_map[domain][LocationType::YPositive];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        auto bound_cal_s = [&](int i, int j) {
            double u_ijk = u(i, j);
            double v_ijk = v(i, j);
            double u_ip1 = i == nx - 1 ? u_xpos_buffer[j] : u(i + 1, j);
            double v_jp1 = j == ny - 1 ? v_ypos_buffer[i] : v(i, j + 1);

            double s_ijk = s(i, j);
            double s_im1 = i == 0 ? s_xneg_buffer[j] : s(i - 1, j);
            double s_ip1 = i == nx - 1 ? s_xpos_buffer[j] : s(i + 1, j);
            double s_jm1 = j == 0 ? s_yneg_buffer[i] : s(i, j - 1);
            double s_jp1 = j == ny - 1 ? s_ypos_buffer[i] : s(i, j + 1);

            /**
             * Convective Term: First-Order Upwind Scheme
             * Logic: If velocity > 0, transport from upstream (xneg/yneg).
             * If velocity < 0, transport from downstream (xpos/ypos).
             * This eliminates oscillations by adding numerical diffusion.
             */

            double flux_x_xpos = (u_ip1 > 0) ? (u_ip1 * s_ijk) : (u_ip1 * s_ip1);
            double flux_x_xneg = (u_ijk > 0) ? (u_ijk * s_im1) : (u_ijk * s_ijk);
            double conv_x      = (flux_x_xpos - flux_x_xneg) / hx;

            double flux_y_ypos = (v_jp1 > 0) ? (v_jp1 * s_ijk) : (v_jp1 * s_jp1);
            double flux_y_yneg = (v_ijk > 0) ? (v_ijk * s_jm1) : (v_ijk * s_ijk);
            double conv_y      = (flux_y_ypos - flux_y_yneg) / hy;

            /**
             * Diffusive Term: Second-Order Central Difference
             * Stable as long as dt <= (Pe * h^2) / 6.
             */

            double diffuse_x = nr / hx / hx * (s_ip1 - 2.0 * s_ijk + s_im1);
            double diffuse_y = nr / hy / hy * (s_jp1 - 2.0 * s_ijk + s_jm1);

            s_temp(i, j) = s_ijk - dt * (conv_x + conv_y - diffuse_x - diffuse_y);
        };

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny; j++)
        {
            bound_cal_s(0, j);
            bound_cal_s(nx - 1, j);
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            bound_cal_s(i, 0);
            bound_cal_s(i, ny - 1);
        }

        swap_field_data(s, s_temp);
    }
}

void ScalarSolver2D::conv_QUICK_diff_cd2nd_inner()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& s = *s_field_map[domain];

        field2& s_temp = *s_temp_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 2; i < nx - 2; i++)
        {
            for (int j = 2; j < ny - 2; j++)
            {
                double u_ijk = u(i, j);
                double v_ijk = v(i, j);
                double u_ip1 = u(i + 1, j);
                double v_jp1 = v(i, j + 1);

                double s_ijk = s(i, j);
                double s_im1 = s(i - 1, j);
                double s_im2 = s(i - 2, j);
                double s_ip1 = s(i + 1, j);
                double s_ip2 = s(i + 2, j);

                double s_jm1 = s(i, j - 1);
                double s_jm2 = s(i, j - 2);
                double s_jp1 = s(i, j + 1);
                double s_jp2 = s(i, j + 2);

                double s_e_face = (u_ip1 > 0) ? (0.75 * s_ijk + 0.375 * s_ip1 - 0.125 * s_im1) :
                                                (0.75 * s_ip1 + 0.375 * s_ijk - 0.125 * s_ip2);
                double s_w_face = (u_ijk > 0) ? (0.75 * s_im1 + 0.375 * s_ijk - 0.125 * s_im2) :
                                                (0.75 * s_ijk + 0.375 * s_im1 - 0.125 * s_ip1);
                double conv_x   = (u_ip1 * s_e_face - u_ijk * s_w_face) / hx;

                double s_n_face = (v_jp1 > 0) ? (0.75 * s_ijk + 0.375 * s_jp1 - 0.125 * s_jm1) :
                                                (0.75 * s_jp1 + 0.375 * s_ijk - 0.125 * s_jp2);
                double s_s_face = (v_ijk > 0) ? (0.75 * s_jm1 + 0.375 * s_ijk - 0.125 * s_jm2) :
                                                (0.75 * s_ijk + 0.375 * s_jm1 - 0.125 * s_jp1);
                double conv_y   = (v_jp1 * s_n_face - v_ijk * s_s_face) / hy;

                /**
                 * Diffusive Term: Second-Order Central Difference
                 * Stable as long as dt <= (Pe * h^2) / 6.
                 */

                double diffuse_x = nr / hx / hx * (s_ip1 - 2.0 * s_ijk + s_im1);
                double diffuse_y = nr / hy / hy * (s_jp1 - 2.0 * s_ijk + s_jm1);

                s_temp(i, j) = s(i, j) - dt * (conv_x + conv_y - diffuse_x - diffuse_y);
            }
        }
    }
}

void ScalarSolver2D::conv_uw1st_diff_cd2nd_outer_width2()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& s = *s_field_map[domain];

        field2& s_temp = *s_temp_field_map[domain];

        double* u_xpos_buffer = u_buffer_map[domain][LocationType::XPositive];
        double* v_ypos_buffer = v_buffer_map[domain][LocationType::YPositive];

        double* s_xneg_buffer = s_buffer_map[domain][LocationType::XNegative];
        double* s_xpos_buffer = s_buffer_map[domain][LocationType::XPositive];
        double* s_yneg_buffer = s_buffer_map[domain][LocationType::YNegative];
        double* s_ypos_buffer = s_buffer_map[domain][LocationType::YPositive];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        auto bound_cal_s = [&](int i, int j) {
            double u_ijk = u(i, j);
            double v_ijk = v(i, j);
            double u_ip1 = i == nx - 1 ? u_xpos_buffer[j] : u(i + 1, j);
            double v_jp1 = j == ny - 1 ? v_ypos_buffer[i] : v(i, j + 1);

            double s_ijk = s(i, j);
            double s_im1 = i == 0 ? s_xneg_buffer[j] : s(i - 1, j);
            double s_ip1 = i == nx - 1 ? s_xpos_buffer[j] : s(i + 1, j);
            double s_jm1 = j == 0 ? s_yneg_buffer[i] : s(i, j - 1);
            double s_jp1 = j == ny - 1 ? s_ypos_buffer[i] : s(i, j + 1);

            /**
             * Convective Term: First-Order Upwind Scheme
             * Logic: If velocity > 0, transport from upstream (xneg/yneg).
             * If velocity < 0, transport from downstream (xpos/ypos).
             * This eliminates oscillations by adding numerical diffusion.
             */

            double flux_x_xpos = (u_ip1 > 0) ? (u_ip1 * s_ijk) : (u_ip1 * s_ip1);
            double flux_x_xneg = (u_ijk > 0) ? (u_ijk * s_im1) : (u_ijk * s_ijk);
            double conv_x      = (flux_x_xpos - flux_x_xneg) / hx;

            double flux_y_ypos = (v_jp1 > 0) ? (v_jp1 * s_ijk) : (v_jp1 * s_jp1);
            double flux_y_yneg = (v_ijk > 0) ? (v_ijk * s_jm1) : (v_ijk * s_ijk);
            double conv_y      = (flux_y_ypos - flux_y_yneg) / hy;

            /**
             * Diffusive Term: Second-Order Central Difference
             * Stable as long as dt <= (Pe * h^2) / 6.
             */

            double diffuse_x = nr / hx / hx * (s_ip1 - 2.0 * s_ijk + s_im1);
            double diffuse_y = nr / hy / hy * (s_jp1 - 2.0 * s_ijk + s_jm1);

            s_temp(i, j) = s_ijk - dt * (conv_x + conv_y - diffuse_x - diffuse_y);
        };

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny; j++)
        {
            bound_cal_s(0, j);
            bound_cal_s(1, j);
            bound_cal_s(nx - 2, j);
            bound_cal_s(nx - 1, j);
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            bound_cal_s(i, 0);
            bound_cal_s(i, 1);
            bound_cal_s(i, ny - 2);
            bound_cal_s(i, ny - 1);
        }

        swap_field_data(s, s_temp);
    }
}

inline double ScalarSolver2D::get_tvd_van_leer(double s_up2, double s_up, double s_down)
{
    double eps         = 1e-15;
    double denominator = s_down - s_up;

    // If gradient is zero, face value is simply the upstream value
    if (std::abs(denominator) < eps)
        return s_up;

    // Successive gradient ratio
    double r = (s_up - s_up2) / denominator;

    // Van Leer Limiter function
    double psi = (r + std::abs(r)) / (1.0 + std::abs(r));

    // Face value: Upwind + Anti-diffusive correction
    return s_up + 0.5 * psi * (s_down - s_up);
}

void ScalarSolver2D::conv_TVD_VanLeer_diff_cd2nd_inner()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& s      = *s_field_map[domain];
        field2& s_temp = *s_temp_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 2; i < nx - 2; i++)
        {
            for (int j = 2; j < ny - 2; j++)
            {
                double s_ijk = s(i, j);

                double u_e    = u(i + 1, j);
                double s_e    = (u_e > 0) ? get_tvd_van_leer(s(i - 1, j), s_ijk, s(i + 1, j)) :
                                            get_tvd_van_leer(s(i + 2, j), s(i + 1, j), s_ijk);
                double u_w    = u(i, j);
                double s_w    = (u_w > 0) ? get_tvd_van_leer(s(i - 2, j), s(i - 1, j), s_ijk) :
                                            get_tvd_van_leer(s(i + 1, j), s_ijk, s(i - 1, j));
                double conv_x = (u_e * s_e - u_w * s_w) / hx;

                double v_n    = v(i, j + 1);
                double s_n    = (v_n > 0) ? get_tvd_van_leer(s(i, j - 1), s_ijk, s(i, j + 1)) :
                                            get_tvd_van_leer(s(i, j + 2), s(i, j + 1), s_ijk);
                double v_s    = v(i, j);
                double s_s    = (v_s > 0) ? get_tvd_van_leer(s(i, j - 2), s(i, j - 1), s_ijk) :
                                            get_tvd_van_leer(s(i, j + 1), s_ijk, s(i, j - 1));
                double conv_y = (v_n * s_n - v_s * s_s) / hy;

                double diffuse = nr * ((s(i + 1, j) - 2.0 * s_ijk + s(i - 1, j)) / (hx * hx) +
                                       (s(i, j + 1) - 2.0 * s_ijk + s(i, j - 1)) / (hy * hy));

                s_temp(i, j) = s_ijk - dt * (conv_x + conv_y - diffuse);
            }
        }
    }
}
