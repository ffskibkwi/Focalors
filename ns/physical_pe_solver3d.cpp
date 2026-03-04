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

        u_xpos_ypos_corner_map[domain] = new double[u_field->get_nz()];
        u_xpos_zpos_corner_map[domain] = new double[u_field->get_ny()];
        v_xpos_ypos_corner_map[domain] = new double[v_field->get_nz()];
        v_ypos_zpos_corner_map[domain] = new double[v_field->get_nx()];
        w_xpos_zpos_corner_map[domain] = new double[w_field->get_ny()];
        w_ypos_zpos_corner_map[domain] = new double[w_field->get_nx()];

        // debug

        dudx_map[domain] = new field3(u_field->get_nx(), u_field->get_ny(), u_field->get_nz(), "C_u_" + domain->name);
        dudy_map[domain] = new field3(v_field->get_nx(), v_field->get_ny(), v_field->get_nz(), "C_v_" + domain->name);
        dudz_map[domain] = new field3(w_field->get_nx(), w_field->get_ny(), w_field->get_nz(), "C_w_" + domain->name);

        dvdx_map[domain] = new field3(u_field->get_nx(), u_field->get_ny(), u_field->get_nz(), "C_u_" + domain->name);
        dvdy_map[domain] = new field3(v_field->get_nx(), v_field->get_ny(), v_field->get_nz(), "C_v_" + domain->name);
        dvdz_map[domain] = new field3(w_field->get_nx(), w_field->get_ny(), w_field->get_nz(), "C_w_" + domain->name);

        dwdx_map[domain] = new field3(u_field->get_nx(), u_field->get_ny(), u_field->get_nz(), "C_u_" + domain->name);
        dwdy_map[domain] = new field3(v_field->get_nx(), v_field->get_ny(), v_field->get_nz(), "C_v_" + domain->name);
        dwdz_map[domain] = new field3(w_field->get_nx(), w_field->get_ny(), w_field->get_nz(), "C_w_" + domain->name);
    }
}

PhysicalPESolver3D::~PhysicalPESolver3D()
{
    for (auto& [domain, buffer] : u_xpos_ypos_corner_map)
        delete[] buffer;
    for (auto& [domain, buffer] : u_xpos_zpos_corner_map)
        delete[] buffer;
    for (auto& [domain, buffer] : v_xpos_ypos_corner_map)
        delete[] buffer;
    for (auto& [domain, buffer] : v_ypos_zpos_corner_map)
        delete[] buffer;
    for (auto& [domain, buffer] : w_xpos_zpos_corner_map)
        delete[] buffer;
    for (auto& [domain, buffer] : w_ypos_zpos_corner_map)
        delete[] buffer;

    // debug
    for (auto& [domain, field] : dudx_map)
        delete field;
    for (auto& [domain, field] : dudy_map)
        delete field;
    for (auto& [domain, field] : dudz_map)
        delete field;

    for (auto& [domain, field] : dvdx_map)
        delete field;
    for (auto& [domain, field] : dvdy_map)
        delete field;
    for (auto& [domain, field] : dvdz_map)
        delete field;

    for (auto& [domain, field] : dwdx_map)
        delete field;
    for (auto& [domain, field] : dwdy_map)
        delete field;
    for (auto& [domain, field] : dwdz_map)
        delete field;
}

void PhysicalPESolver3D::solve()
{
    diag_shared_boundary_update();
    calc_rhs();
    p_solver->solve();
}

void PhysicalPESolver3D::calc_rhs()
{
    auto& domains = u_var->geometry->domains;
    for (auto& domain : domains)
    {
        // redirect in convenience of using ns code
        field3& u = *u_var->field_map[domain];
        field3& v = *v_var->field_map[domain];
        field3& w = *w_var->field_map[domain];
        field3& p = *p_var->field_map[domain];

        field2& u_xneg_buffer = *u_var->buffer_map[domain][LocationType::XNegative];
        field2& u_xpos_buffer = *u_var->buffer_map[domain][LocationType::XPositive];
        field2& u_yneg_buffer = *u_var->buffer_map[domain][LocationType::YNegative];
        field2& u_ypos_buffer = *u_var->buffer_map[domain][LocationType::YPositive];
        field2& u_zneg_buffer = *u_var->buffer_map[domain][LocationType::ZNegative];
        field2& u_zpos_buffer = *u_var->buffer_map[domain][LocationType::ZPositive];

        field2& v_xneg_buffer = *v_var->buffer_map[domain][LocationType::XNegative];
        field2& v_xpos_buffer = *v_var->buffer_map[domain][LocationType::XPositive];
        field2& v_yneg_buffer = *v_var->buffer_map[domain][LocationType::YNegative];
        field2& v_ypos_buffer = *v_var->buffer_map[domain][LocationType::YPositive];
        field2& v_zneg_buffer = *v_var->buffer_map[domain][LocationType::ZNegative];
        field2& v_zpos_buffer = *v_var->buffer_map[domain][LocationType::ZPositive];

        field2& w_xneg_buffer = *w_var->buffer_map[domain][LocationType::XNegative];
        field2& w_xpos_buffer = *w_var->buffer_map[domain][LocationType::XPositive];
        field2& w_yneg_buffer = *w_var->buffer_map[domain][LocationType::YNegative];
        field2& w_ypos_buffer = *w_var->buffer_map[domain][LocationType::YPositive];
        field2& w_zneg_buffer = *w_var->buffer_map[domain][LocationType::ZNegative];
        field2& w_zpos_buffer = *w_var->buffer_map[domain][LocationType::ZPositive];

        double* u_corner_along_y = u_var->corner_y_map[domain];
        double* u_corner_along_z = u_var->corner_z_map[domain];
        double* v_corner_along_x = v_var->corner_x_map[domain];
        double* v_corner_along_z = v_var->corner_z_map[domain];
        double* w_corner_along_x = w_var->corner_x_map[domain];
        double* w_corner_along_y = w_var->corner_y_map[domain];

        double* u_xpos_ypos_corner = u_xpos_ypos_corner_map[domain];
        double* u_xpos_zpos_corner = u_xpos_zpos_corner_map[domain];
        double* v_xpos_ypos_corner = v_xpos_ypos_corner_map[domain];
        double* v_ypos_zpos_corner = v_ypos_zpos_corner_map[domain];
        double* w_xpos_zpos_corner = w_xpos_zpos_corner_map[domain];
        double* w_ypos_zpos_corner = w_ypos_zpos_corner_map[domain];

        // debug
        field3& dudx = *dudx_map[domain];
        field3& dudy = *dudy_map[domain];
        field3& dudz = *dudz_map[domain];

        field3& dvdx = *dvdx_map[domain];
        field3& dvdy = *dvdy_map[domain];
        field3& dvdz = *dvdz_map[domain];

        field3& dwdx = *dwdx_map[domain];
        field3& dwdy = *dwdy_map[domain];
        field3& dwdz = *dwdz_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int k = 0; k < nz; k++)
                {
                    std::array<std::array<double, 3>, 3> L;

                    double u_ijk = u(i, j, k);
                    double u_im1 = i == 0 ? u_xneg_buffer(j, k) : u(i - 1, j, k);
                    double u_ip1 = i == nx - 1 ? u_xpos_buffer(j, k) : u(i + 1, j, k);
                    double u_jm1 = j == 0 ? u_yneg_buffer(i, k) : u(i, j - 1, k);
                    double u_jp1 = j == ny - 1 ? u_ypos_buffer(i, k) : u(i, j + 1, k);
                    double u_km1 = k == 0 ? u_zneg_buffer(i, j) : u(i, j, k - 1);
                    double u_kp1 = k == nz - 1 ? u_zpos_buffer(i, j) : u(i, j, k + 1);

                    double u_ip1_jm1 = i == nx - 1 ? (j == 0 ? u_corner_along_z[k] : u_xpos_buffer(j - 1, k)) :
                                                     (j == 0 ? u_yneg_buffer(i + 1, k) : u(i + 1, j - 1, k));
                    double u_ip1_jp1 = i == nx - 1 ? (j == ny - 1 ? u_xpos_ypos_corner[k] : u_xpos_buffer(j + 1, k)) :
                                                     (j == ny - 1 ? u_ypos_buffer(i + 1, k) : u(i + 1, j + 1, k));
                    double u_ip1_km1 = i == nx - 1 ? (k == 0 ? u_corner_along_y[j] : u_xpos_buffer(j, k - 1)) :
                                                     (k == 0 ? u_zneg_buffer(i + 1, j) : u(i + 1, j, k - 1));
                    double u_ip1_kp1 = i == nx - 1 ? (k == nz - 1 ? u_xpos_zpos_corner[j] : u_xpos_buffer(j, k + 1)) :
                                                     (k == nz - 1 ? u_zpos_buffer(i + 1, j) : u(i + 1, j, k + 1));

                    // dudy $(u_{i, j+1, k} + u_{i+1, j+1, k}) - (u_{i, j-1, k} + u_{i+1, j-1, k})$
                    // dudz $(u_{i, j, k+1} + u_{i+1, j, k+1}) - (u_{i, j, k-1} + u_{i+1, j, k-1})$

                    L[0][0] = (u_ip1 - u_ijk) / hx;
                    L[0][1] = (u_jp1 + u_ip1_jp1 - u_jm1 - u_ip1_jm1) / 4.0 / hy;
                    L[0][2] = (u_kp1 + u_ip1_kp1 - u_km1 - u_ip1_km1) / 4.0 / hz;

                    // debug
                    dudx(i, j, k) = L[0][0];
                    dudy(i, j, k) = L[0][1];
                    dudz(i, j, k) = L[0][2];

                    double v_ijk = v(i, j, k);
                    double v_im1 = i == 0 ? v_xneg_buffer(j, k) : v(i - 1, j, k);
                    double v_ip1 = i == nx - 1 ? v_xpos_buffer(j, k) : v(i + 1, j, k);
                    double v_jm1 = j == 0 ? v_yneg_buffer(i, k) : v(i, j - 1, k);
                    double v_jp1 = j == ny - 1 ? v_ypos_buffer(i, k) : v(i, j + 1, k);
                    double v_km1 = k == 0 ? v_zneg_buffer(i, j) : v(i, j, k - 1);
                    double v_kp1 = k == nz - 1 ? v_zpos_buffer(i, j) : v(i, j, k + 1);

                    double v_im1_jp1 = i == 0 ? (j == ny - 1 ? v_corner_along_z[k] : v_xneg_buffer(j + 1, k)) :
                                                (j == ny - 1 ? v_ypos_buffer(i - 1, k) : v(i - 1, j + 1, k));
                    double v_ip1_jp1 = i == nx - 1 ? (j == ny - 1 ? v_xpos_ypos_corner[k] : v_xpos_buffer(j + 1, k)) :
                                                     (j == ny - 1 ? v_ypos_buffer(i + 1, k) : v(i + 1, j + 1, k));
                    double v_jp1_km1 = j == ny - 1 ? (k == 0 ? v_corner_along_x[i] : v_ypos_buffer(i, k - 1)) :
                                                     (k == 0 ? v_zneg_buffer(i, j + 1) : v(i, j + 1, k - 1));
                    double v_jp1_kp1 = j == ny - 1 ? (k == nz - 1 ? v_ypos_zpos_corner[i] : v_ypos_buffer(i, k + 1)) :
                                                     (k == nz - 1 ? v_zpos_buffer(i, j + 1) : v(i, j + 1, k + 1));

                    // dvdx $(v_{i+1, j, k} + v_{i+1, j+1, k}) - (v_{i-1, j, k} + v_{i-1, j+1, k})$
                    // dvdz $(v_{i, j, k+1} + v_{i, j+1, k+1}) - (v_{i, j, k-1} + v_{i, j+1, k-1})$

                    L[1][0] = (v_ip1 + v_ip1_jp1 - v_im1 - v_im1_jp1) / 4.0 / hx;
                    L[1][1] = (v_jp1 - v_ijk) / hy;
                    L[1][2] = (v_kp1 + v_jp1_kp1 - v_km1 - v_jp1_km1) / 4.0 / hz;

                    // debug
                    dvdx(i, j, k) = L[1][0];
                    dvdy(i, j, k) = L[1][1];
                    dvdz(i, j, k) = L[1][2];

                    double w_ijk = w(i, j, k);
                    double w_im1 = i == 0 ? w_xneg_buffer(j, k) : w(i - 1, j, k);
                    double w_ip1 = i == nx - 1 ? w_xpos_buffer(j, k) : w(i + 1, j, k);
                    double w_jm1 = j == 0 ? w_yneg_buffer(i, k) : w(i, j - 1, k);
                    double w_jp1 = j == ny - 1 ? w_ypos_buffer(i, k) : w(i, j + 1, k);
                    double w_km1 = k == 0 ? w_zneg_buffer(i, j) : w(i, j, k - 1);
                    double w_kp1 = k == nz - 1 ? w_zpos_buffer(i, j) : w(i, j, k + 1);

                    double w_im1_kp1 = i == 0 ? (k == nz - 1 ? w_corner_along_y[j] : w_xneg_buffer(j, k + 1)) :
                                                (k == nz - 1 ? w_zpos_buffer(i - 1, j) : w(i - 1, j, k + 1));
                    double w_ip1_kp1 = i == nx - 1 ? (k == nz - 1 ? w_xpos_zpos_corner[j] : w_xpos_buffer(j, k + 1)) :
                                                     (k == nz - 1 ? w_zpos_buffer(i + 1, j) : w(i + 1, j, k + 1));
                    double w_jm1_kp1 = j == 0 ? (k == nz - 1 ? w_corner_along_x[i] : w_yneg_buffer(i, k + 1)) :
                                                (k == nz - 1 ? w_zpos_buffer(i, j - 1) : w(i, j - 1, k + 1));
                    double w_jp1_kp1 = j == ny - 1 ? (k == nz - 1 ? w_ypos_zpos_corner[i] : w_ypos_buffer(i, k + 1)) :
                                                     (k == nz - 1 ? w_zpos_buffer(i, j + 1) : w(i, j + 1, k + 1));

                    // dzdx $(w_{i+1, j, k} + w_{i+1, j, k+1}) - (w_{i-1, j, k} + w_{i-1, j, k+1})$
                    // dzdy $(w_{i, j+1, k} + w_{i, j+1, k+1}) - (w_{i, j-1, k} + w_{i, j-1, k+1})$

                    L[2][0] = (w_ip1 + w_ip1_kp1 - w_im1 - w_im1_kp1) / 4.0 / hx;
                    L[2][1] = (w_jp1 + w_jp1_kp1 - w_jm1 - w_jm1_kp1) / 4.0 / hy;
                    L[2][2] = (w_kp1 - w_ijk) / hz;

                    // debug
                    dwdx(i, j, k) = L[2][0];
                    dwdy(i, j, k) = L[2][1];
                    dwdz(i, j, k) = L[2][2];

                    p(i, j, k) = 0.0;
                    for (int m = 0; m < 3; m++)
                        for (int n = 0; n < 3; n++)
                            p(i, j, k) += L[m][n] * L[n][m];
                    p(i, j, k) *= -rho;
                }
            }
        }
    }
}