#include "physical_pe_solver3d.h"

PhysicalPESolver3D::PhysicalPESolver3D(Variable3D*            in_u_var,
                                       Variable3D*            in_v_var,
                                       Variable3D*            in_w_var,
                                       Variable3D*            in_p_var,
                                       ConcatPoissonSolver3D* in_p_solver,
                                       double                 in_rho)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , w_var(in_w_var)
    , p_var(in_p_var)
    , p_solver(in_p_solver)
    , rho(in_rho)
{
    auto& domains = u_var->geometry->domains;
    for (auto& domain : domains)
    {
        auto u_field = u_var->field_map[domain];
        auto v_field = v_var->field_map[domain];
        auto w_field = w_var->field_map[domain];

        c_u_map[domain] = new field3(u_field->get_nx(), u_field->get_ny(), u_field->get_nz(), "C_u_" + domain->name);
        c_v_map[domain] = new field3(v_field->get_nx(), v_field->get_ny(), v_field->get_nz(), "C_v_" + domain->name);
        c_w_map[domain] = new field3(w_field->get_nx(), w_field->get_ny(), w_field->get_nz(), "C_w_" + domain->name);

        // debug

        conv_u_x_map[domain] =
            new field3(u_field->get_nx(), u_field->get_ny(), u_field->get_nz(), "C_u_" + domain->name);
        conv_u_y_map[domain] =
            new field3(v_field->get_nx(), v_field->get_ny(), v_field->get_nz(), "C_v_" + domain->name);
        conv_u_z_map[domain] =
            new field3(w_field->get_nx(), w_field->get_ny(), w_field->get_nz(), "C_w_" + domain->name);

        conv_v_x_map[domain] =
            new field3(u_field->get_nx(), u_field->get_ny(), u_field->get_nz(), "C_u_" + domain->name);
        conv_v_y_map[domain] =
            new field3(v_field->get_nx(), v_field->get_ny(), v_field->get_nz(), "C_v_" + domain->name);
        conv_v_z_map[domain] =
            new field3(w_field->get_nx(), w_field->get_ny(), w_field->get_nz(), "C_w_" + domain->name);

        conv_w_x_map[domain] =
            new field3(u_field->get_nx(), u_field->get_ny(), u_field->get_nz(), "C_u_" + domain->name);
        conv_w_y_map[domain] =
            new field3(v_field->get_nx(), v_field->get_ny(), v_field->get_nz(), "C_v_" + domain->name);
        conv_w_z_map[domain] =
            new field3(w_field->get_nx(), w_field->get_ny(), w_field->get_nz(), "C_w_" + domain->name);

        for (auto kv : u_var->buffer_map[domain])
            c_u_buffer_map[domain][kv.first] = new field2(kv.second->get_nx(), kv.second->get_ny());
        for (auto kv : v_var->buffer_map[domain])
            c_v_buffer_map[domain][kv.first] = new field2(kv.second->get_nx(), kv.second->get_ny());
        for (auto kv : w_var->buffer_map[domain])
            c_w_buffer_map[domain][kv.first] = new field2(kv.second->get_nx(), kv.second->get_ny());
    }
}

PhysicalPESolver3D::~PhysicalPESolver3D()
{
    for (auto& [domain, field] : c_u_map)
        delete field;
    for (auto& [domain, field] : c_v_map)
        delete field;
    for (auto& [domain, field] : c_w_map)
        delete field;
    for (auto& [domain, buffer_map] : c_u_buffer_map)
        for (auto kv : buffer_map)
            delete kv.second;
    for (auto& [domain, buffer_map] : c_v_buffer_map)
        for (auto kv : buffer_map)
            delete kv.second;
    for (auto& [domain, buffer_map] : c_w_buffer_map)
        for (auto kv : buffer_map)
            delete kv.second;

    // debug
    for (auto& [domain, field] : conv_u_x_map)
        delete field;
    for (auto& [domain, field] : conv_u_y_map)
        delete field;
    for (auto& [domain, field] : conv_u_z_map)
        delete field;

    for (auto& [domain, field] : conv_v_x_map)
        delete field;
    for (auto& [domain, field] : conv_v_y_map)
        delete field;
    for (auto& [domain, field] : conv_v_z_map)
        delete field;

    for (auto& [domain, field] : conv_w_x_map)
        delete field;
    for (auto& [domain, field] : conv_w_y_map)
        delete field;
    for (auto& [domain, field] : conv_w_z_map)
        delete field;
}

void PhysicalPESolver3D::solve()
{
    calc_conv_inner();
    calc_conv_outer();
    nondiag_shared_boundary_update();
    calc_rhs();
    p_solver->solve();
}

void PhysicalPESolver3D::calc_conv_inner()
{
    auto& domains = u_var->geometry->domains;
    for (auto& domain : domains)
    {
        field3& u = *u_var->field_map[domain];
        field3& v = *v_var->field_map[domain];
        field3& w = *w_var->field_map[domain];

        field3& c_u = *c_u_map[domain];
        field3& c_v = *c_v_map[domain];
        field3& c_w = *c_w_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    double conv_u_x = 0.25 / hx *
                                      (u(i + 1, j, k) * (u(i + 1, j, k) + 2.0 * u(i, j, k)) -
                                       u(i - 1, j, k) * (u(i - 1, j, k) + 2.0 * u(i, j, k)));
                    double conv_u_y = 0.25 / hy *
                                      ((u(i, j, k) + u(i, j + 1, k)) * (v(i - 1, j + 1, k) + v(i, j + 1, k)) -
                                       (u(i, j - 1, k) + u(i, j, k)) * (v(i - 1, j, k) + v(i, j, k)));
                    double conv_u_z = 0.25 / hz *
                                      ((u(i, j, k) + u(i, j, k + 1)) * (w(i - 1, j, k + 1) + w(i, j, k + 1)) -
                                       (u(i, j, k - 1) + u(i, j, k)) * (w(i - 1, j, k) + w(i, j, k)));
                    c_u(i, j, k) = conv_u_x + conv_u_y + conv_u_z;

                    double conv_v_x = 0.25 / hx *
                                      ((v(i, j, k) + v(i + 1, j, k)) * (u(i + 1, j - 1, k) + u(i + 1, j, k)) -
                                       (v(i - 1, j, k) + v(i, j, k)) * (u(i, j - 1, k) + u(i, j, k)));
                    double conv_v_y = 0.25 / hy *
                                      (v(i, j + 1, k) * (v(i, j + 1, k) + 2.0 * v(i, j, k)) -
                                       v(i, j - 1, k) * (v(i, j - 1, k) + 2.0 * v(i, j, k)));
                    double conv_v_z = 0.25 / hz *
                                      ((v(i, j, k) + v(i, j, k + 1)) * (w(i, j - 1, k + 1) + w(i, j, k + 1)) -
                                       (v(i, j, k - 1) + v(i, j, k)) * (w(i, j - 1, k) + w(i, j, k)));
                    c_v(i, j, k) = conv_v_x + conv_v_y + conv_v_z;

                    double conv_w_x = 0.25 / hx *
                                      ((w(i, j, k) + w(i + 1, j, k)) * (u(i + 1, j, k - 1) + u(i + 1, j, k)) -
                                       (w(i - 1, j, k) + w(i, j, k)) * (u(i, j, k - 1) + u(i, j, k)));
                    double conv_w_y = 0.25 / hy *
                                      ((w(i, j, k) + w(i, j + 1, k)) * (v(i, j + 1, k - 1) + v(i, j + 1, k)) -
                                       (w(i, j - 1, k) + w(i, j, k)) * (v(i, j, k - 1) + v(i, j, k)));
                    double conv_w_z = 0.25 / hz *
                                      (w(i, j, k + 1) * (w(i, j, k + 1) + 2.0 * w(i, j, k)) -
                                       w(i, j, k - 1) * (w(i, j, k - 1) + 2.0 * w(i, j, k)));
                    c_w(i, j, k) = conv_w_x + conv_w_y + conv_w_z;

                    // debug
                    (*conv_u_x_map[domain])(i, j, k) = conv_u_x;
                    (*conv_u_y_map[domain])(i, j, k) = conv_u_y;
                    (*conv_u_z_map[domain])(i, j, k) = conv_u_z;

                    (*conv_v_x_map[domain])(i, j, k) = conv_v_x;
                    (*conv_v_y_map[domain])(i, j, k) = conv_v_y;
                    (*conv_v_z_map[domain])(i, j, k) = conv_v_z;

                    (*conv_w_x_map[domain])(i, j, k) = conv_w_x;
                    (*conv_w_y_map[domain])(i, j, k) = conv_w_y;
                    (*conv_w_z_map[domain])(i, j, k) = conv_w_z;
                }
            }
        }
    }
}

void PhysicalPESolver3D::calc_conv_outer()
{
    auto& domains = u_var->geometry->domains;
    for (auto& domain : domains)
    {
        field3& u = *u_var->field_map[domain];
        field3& v = *v_var->field_map[domain];
        field3& w = *w_var->field_map[domain];

        field3& c_u = *c_u_map[domain];
        field3& c_v = *c_v_map[domain];
        field3& c_w = *c_w_map[domain];

        field2& u_left_buffer  = *u_var->buffer_map[domain][LocationType::Left];
        field2& u_right_buffer = *u_var->buffer_map[domain][LocationType::Right];
        field2& u_front_buffer = *u_var->buffer_map[domain][LocationType::Front];
        field2& u_back_buffer  = *u_var->buffer_map[domain][LocationType::Back];
        field2& u_down_buffer  = *u_var->buffer_map[domain][LocationType::Down];
        field2& u_up_buffer    = *u_var->buffer_map[domain][LocationType::Up];

        field2& v_left_buffer  = *v_var->buffer_map[domain][LocationType::Left];
        field2& v_right_buffer = *v_var->buffer_map[domain][LocationType::Right];
        field2& v_front_buffer = *v_var->buffer_map[domain][LocationType::Front];
        field2& v_back_buffer  = *v_var->buffer_map[domain][LocationType::Back];
        field2& v_down_buffer  = *v_var->buffer_map[domain][LocationType::Down];
        field2& v_up_buffer    = *v_var->buffer_map[domain][LocationType::Up];

        field2& w_left_buffer  = *w_var->buffer_map[domain][LocationType::Left];
        field2& w_right_buffer = *w_var->buffer_map[domain][LocationType::Right];
        field2& w_front_buffer = *w_var->buffer_map[domain][LocationType::Front];
        field2& w_back_buffer  = *w_var->buffer_map[domain][LocationType::Back];
        field2& w_down_buffer  = *w_var->buffer_map[domain][LocationType::Down];
        field2& w_up_buffer    = *w_var->buffer_map[domain][LocationType::Up];

        double* u_corner_along_y = u_var->corner_value_map_y[domain];
        double* u_corner_along_z = u_var->corner_value_map_z[domain];
        double* v_corner_along_x = v_var->corner_value_map_x[domain];
        double* v_corner_along_z = v_var->corner_value_map_z[domain];
        double* w_corner_along_x = w_var->corner_value_map_x[domain];
        double* w_corner_along_y = w_var->corner_value_map_y[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        auto bound_cal = [&](int i, int j, int k) {
            double u_ijk = u(i, j, k);
            double u_im1 = i == 0 ? u_left_buffer(j, k) : u(i - 1, j, k);
            double u_ip1 = i == nx - 1 ? u_right_buffer(j, k) : u(i + 1, j, k);
            double u_jm1 = j == 0 ? u_front_buffer(i, k) : u(i, j - 1, k);
            double u_jp1 = j == ny - 1 ? u_back_buffer(i, k) : u(i, j + 1, k);
            double u_km1 = k == 0 ? u_down_buffer(i, j) : u(i, j, k - 1);
            double u_kp1 = k == nz - 1 ? u_up_buffer(i, j) : u(i, j, k + 1);

            double u_ip1_jm1 = i == nx - 1 ? (j == 0 ? u_corner_along_z[k] : u_right_buffer(j - 1, k)) :
                                             (j == 0 ? u_front_buffer(i + 1, k) : u(i + 1, j - 1, k));
            double u_ip1_km1 = i == nx - 1 ? (k == 0 ? u_corner_along_y[j] : u_right_buffer(j, k - 1)) :
                                             (k == 0 ? u_down_buffer(i + 1, j) : u(i + 1, j, k - 1));

            double v_ijk = v(i, j, k);
            double v_im1 = i == 0 ? v_left_buffer(j, k) : v(i - 1, j, k);
            double v_ip1 = i == nx - 1 ? v_right_buffer(j, k) : v(i + 1, j, k);
            double v_jm1 = j == 0 ? v_front_buffer(i, k) : v(i, j - 1, k);
            double v_jp1 = j == ny - 1 ? v_back_buffer(i, k) : v(i, j + 1, k);
            double v_km1 = k == 0 ? v_down_buffer(i, j) : v(i, j, k - 1);
            double v_kp1 = k == nz - 1 ? v_up_buffer(i, j) : v(i, j, k + 1);

            double v_im1_jp1 = i == 0 ? (j == ny - 1 ? v_corner_along_z[k] : v_left_buffer(j + 1, k)) :
                                        (j == ny - 1 ? v_back_buffer(i - 1, k) : v(i - 1, j + 1, k));
            double v_jp1_km1 = j == ny - 1 ? (k == 0 ? v_corner_along_x[i] : v_back_buffer(i, k - 1)) :
                                             (k == 0 ? v_down_buffer(i, j + 1) : v(i, j + 1, k - 1));

            double w_ijk = w(i, j, k);
            double w_im1 = i == 0 ? w_left_buffer(j, k) : w(i - 1, j, k);
            double w_ip1 = i == nx - 1 ? w_right_buffer(j, k) : w(i + 1, j, k);
            double w_jm1 = j == 0 ? w_front_buffer(i, k) : w(i, j - 1, k);
            double w_jp1 = j == ny - 1 ? w_back_buffer(i, k) : w(i, j + 1, k);
            double w_km1 = k == 0 ? w_down_buffer(i, j) : w(i, j, k - 1);
            double w_kp1 = k == nz - 1 ? w_up_buffer(i, j) : w(i, j, k + 1);

            double w_im1_kp1 = i == 0 ? (k == nz - 1 ? w_corner_along_y[j] : w_left_buffer(j, k + 1)) :
                                        (k == nz - 1 ? w_up_buffer(i - 1, j) : w(i - 1, j, k + 1));
            double w_jm1_kp1 = j == 0 ? (k == nz - 1 ? w_corner_along_x[i] : w_front_buffer(i, k + 1)) :
                                        (k == nz - 1 ? w_up_buffer(i, j - 1) : w(i, j - 1, k + 1));

            double conv_u_x = 0.25 / hx * (u_ip1 * (u_ip1 + 2.0 * u_ijk) - u_im1 * (u_im1 + 2.0 * u_ijk));
            double conv_u_y =
                0.25 / hy * ((u_ijk + u_jp1) * (v_im1_jp1 + v_jp1) - (u_jm1 + u_ijk) * (v_im1 + v(i, j, k)));
            double conv_u_z =
                0.25 / hz * ((u_ijk + u_kp1) * (w_im1_kp1 + w_kp1) - (u_km1 + u_ijk) * (w_im1 + w(i, j, k)));
            c_u(i, j, k) = conv_u_x + conv_u_y + conv_u_z;

            double conv_v_x =
                0.25 / hx * ((v_ijk + v_ip1) * (u_ip1_jm1 + u_ip1) - (v_im1 + v_ijk) * (u_jm1 + u(i, j, k)));
            double conv_v_y = 0.25 / hy * (v_jp1 * (v_jp1 + 2.0 * v_ijk) - v_jm1 * (v_jm1 + 2.0 * v_ijk));
            double conv_v_z =
                0.25 / hz * ((v_ijk + v_kp1) * (w_jm1_kp1 + w_kp1) - (v_km1 + v_ijk) * (w_jm1 + w(i, j, k)));
            c_v(i, j, k) = conv_v_x + conv_v_y + conv_v_z;

            double conv_w_x =
                0.25 / hx * ((w_ijk + w_ip1) * (u_ip1_km1 + u_ip1) - (w_im1 + w_ijk) * (u_km1 + u(i, j, k)));
            double conv_w_y =
                0.25 / hy * ((w_ijk + w_jp1) * (v_jp1_km1 + v_jp1) - (w_jm1 + w_ijk) * (v_km1 + v(i, j, k)));
            double conv_w_z = 0.25 / hz * (w_kp1 * (w_kp1 + 2.0 * w_ijk) - w_km1 * (w_km1 + 2.0 * w_ijk));
            c_w(i, j, k)    = conv_w_x + conv_w_y + conv_w_z;

            // debug
            (*conv_u_x_map[domain])(i, j, k) = conv_u_x;
            (*conv_u_y_map[domain])(i, j, k) = conv_u_y;
            (*conv_u_z_map[domain])(i, j, k) = conv_u_z;

            (*conv_v_x_map[domain])(i, j, k) = conv_v_x;
            (*conv_v_y_map[domain])(i, j, k) = conv_v_y;
            (*conv_v_z_map[domain])(i, j, k) = conv_v_z;

            (*conv_w_x_map[domain])(i, j, k) = conv_w_x;
            (*conv_w_y_map[domain])(i, j, k) = conv_w_y;
            (*conv_w_z_map[domain])(i, j, k) = conv_w_z;
        };

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                bound_cal(0, j, k);
                bound_cal(nx - 1, j, k);
            }
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int k = 0; k < nz; k++)
            {
                bound_cal(i, 0, k);
                bound_cal(i, ny - 1, k);
            }
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                bound_cal(i, j, 0);
                bound_cal(i, j, nz - 1);
            }
        }
    }
}

void PhysicalPESolver3D::calc_rhs()
{
    auto& domains = u_var->geometry->domains;
    for (auto& domain : domains)
    {
        // redirect in convenience of using ns code
        field3& u = *c_u_map[domain];
        field3& v = *c_v_map[domain];
        field3& w = *c_w_map[domain];
        field3& p = *p_var->field_map[domain];

        // redirect in convenience of using ns code
        field2& u_buffer_right = *c_u_buffer_map[domain][LocationType::Right];
        field2& v_buffer_back  = *c_v_buffer_map[domain][LocationType::Back];
        field2& w_buffer_up    = *c_w_buffer_map[domain][LocationType::Up];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx - 1; i++)
            for (int j = 0; j < ny - 1; j++)
                for (int k = 0; k < nz - 1; k++)
                    p(i, j, k) = -rho * ((u(i + 1, j, k) - u(i, j, k)) / hx + (v(i, j + 1, k) - v(i, j, k)) / hy +
                                         (w(i, j, k + 1) - w(i, j, k)) / hz);

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx - 1; i++)
            for (int j = 0; j < ny - 1; j++)
                p(i, j, nz - 1) =
                    -rho * ((u(i + 1, j, nz - 1) - u(i, j, nz - 1)) / hx +
                            (v(i, j + 1, nz - 1) - v(i, j, nz - 1)) / hy + (w_buffer_up(i, j) - w(i, j, nz - 1)) / hz);

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx - 1; i++)
            for (int k = 0; k < nz - 1; k++)
                p(i, ny - 1, k) = -rho * ((u(i + 1, ny - 1, k) - u(i, ny - 1, k)) / hx +
                                          (v_buffer_back(i, k) - v(i, ny - 1, k)) / hy +
                                          (w(i, ny - 1, k + 1) - w(i, ny - 1, k)) / hz);

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny - 1; j++)
            for (int k = 0; k < nz - 1; k++)
                p(nx - 1, j, k) = -rho * ((u_buffer_right(j, k) - u(nx - 1, j, k)) / hx +
                                          (v(nx - 1, j + 1, k) - v(nx - 1, j, k)) / hy +
                                          (w(nx - 1, j, k + 1) - w(nx - 1, j, k)) / hz);

        for (int i = 0; i < nx - 1; i++)
            p(i, ny - 1, nz - 1) = -rho * ((u(i + 1, ny - 1, nz - 1) - u(i, ny - 1, nz - 1)) / hx +
                                           (v_buffer_back(i, nz - 1) - v(i, ny - 1, nz - 1)) / hy +
                                           (w_buffer_up(i, ny - 1) - w(i, ny - 1, nz - 1)) / hz);

        for (int j = 0; j < ny - 1; j++)
            p(nx - 1, j, nz - 1) = -rho * ((u_buffer_right(j, nz - 1) - u(nx - 1, j, nz - 1)) / hx +
                                           (v(nx - 1, j + 1, nz - 1) - v(nx - 1, j, nz - 1)) / hy +
                                           (w_buffer_up(nx - 1, j) - w(nx - 1, j, nz - 1)) / hz);

        for (int k = 0; k < nz - 1; k++)
            p(nx - 1, ny - 1, k) = -rho * ((u_buffer_right(ny - 1, k) - u(nx - 1, ny - 1, k)) / hx +
                                           (v_buffer_back(nx - 1, k) - v(nx - 1, ny - 1, k)) / hy +
                                           (w(nx - 1, ny - 1, k + 1) - w(nx - 1, ny - 1, k)) / hz);

        p(nx - 1, ny - 1, nz - 1) = -rho * ((u_buffer_right(ny - 1, nz - 1) - u(nx - 1, ny - 1, nz - 1)) / hx +
                                            (v_buffer_back(nx - 1, nz - 1) - v(nx - 1, ny - 1, nz - 1)) / hy +
                                            (w_buffer_up(nx - 1, ny - 1) - w(nx - 1, ny - 1, nz - 1)) / hz);
    }
}