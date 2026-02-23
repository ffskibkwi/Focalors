#include "ns_solver3d.h"
#include "boundary_3d_utils.h"

ConcatNSSolver3D::ConcatNSSolver3D(Variable3D* in_u_var,
                                   Variable3D* in_v_var,
                                   Variable3D* in_w_var,
                                   Variable3D* in_p_var)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , w_var(in_w_var)
    , p_var(in_p_var)
    , u_corner_value_map_y(u_var->corner_value_map_y)
    , u_corner_value_map_z(u_var->corner_value_map_z)
    , v_corner_value_map_x(v_var->corner_value_map_x)
    , v_corner_value_map_z(v_var->corner_value_map_z)
    , w_corner_value_map_x(w_var->corner_value_map_x)
    , w_corner_value_map_y(w_var->corner_value_map_y)
{
    TimeAdvancingConfig& time_cfg    = TimeAdvancingConfig::Get();
    PhysicsConfig&       physics_cfg = PhysicsConfig::Get();

    dt      = time_cfg.dt;
    num_it  = time_cfg.num_iterations;
    corr_it = time_cfg.corr_iter;

    nu = physics_cfg.nu;

    // check u v p share one geometry
    if (u_var->geometry != v_var->geometry || u_var->geometry != w_var->geometry || u_var->geometry != p_var->geometry)
        throw std::runtime_error("ConcatNSSolver3D: u v w p do not share one geometry");

    // geometry double check
    if (u_var->geometry == nullptr)
        throw std::runtime_error("ConcatNSSolver3D: u->geometry is null");
    if (!u_var->geometry->is_checked)
        u_var->geometry->check();
    if (u_var->geometry->tree_root == nullptr || u_var->geometry->tree_map.empty())
        u_var->geometry->solve_prepare();

    domains   = u_var->geometry->domains;
    adjacency = u_var->geometry->adjacency;

    u_field_map = u_var->field_map;
    v_field_map = v_var->field_map;
    w_field_map = w_var->field_map;
    p_field_map = p_var->field_map;

    u_buffer_map = u_var->buffer_map;
    v_buffer_map = v_var->buffer_map;
    w_buffer_map = w_var->buffer_map;
    p_buffer_map = p_var->buffer_map;

    // Construct the temp field for each domain
    for (auto& [domain, field] : u_field_map)
        u_temp_field_map[domain] =
            new field3(field->get_nx(), field->get_ny(), field->get_nz(), field->get_name() + "_temp");
    for (auto& [domain, field] : v_field_map)
        v_temp_field_map[domain] =
            new field3(field->get_nx(), field->get_ny(), field->get_nz(), field->get_name() + "_temp");
    for (auto& [domain, field] : w_field_map)
        w_temp_field_map[domain] =
            new field3(field->get_nx(), field->get_ny(), field->get_nz(), field->get_name() + "_temp");

    p_solver = new ConcatPoissonSolver3D(p_var);

    // update boundary for first step
    phys_boundary_update();
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();
}

ConcatNSSolver3D::~ConcatNSSolver3D() { delete p_solver; }

void ConcatNSSolver3D::variable_check()
{
    if (u_var->position_type != VariablePositionType::XFace)
        throw std::runtime_error("ConcatNSSolver3D: u->position_type is not XFace");
    if (v_var->position_type != VariablePositionType::YFace)
        throw std::runtime_error("ConcatNSSolver3D: v->position_type is not YFace");
    if (w_var->position_type != VariablePositionType::ZFace)
        throw std::runtime_error("ConcatNSSolver3D: w->position_type is not ZFace");
    if (p_var->position_type != VariablePositionType::Center)
        throw std::runtime_error("ConcatNSSolver3D: p->position_type is not Center");
}

void ConcatNSSolver3D::solve()
{
    // NS
    euler_conv_diff_inner();
    euler_conv_diff_outer();

    // update boundary for divu
    phys_boundary_update();
    nondiag_shared_boundary_update();

    for (int it = 0; it < corr_it; it++)
    {
        // divu
        velocity_div_inner();
        velocity_div_outer();

        // PE
        normalize_pressure();
        p_solver->solve();

        // update buffer for p
        pressure_buffer_update();

        // p grad
        add_pressure_gradient();
    }

    // update boundary
    phys_boundary_update();
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();
}

void ConcatNSSolver3D::euler_conv_diff_inner()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];

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
        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    double conv_x = 0.25 / hx *
                                    (u(i + 1, j, k) * (u(i + 1, j, k) + 2.0 * u(i, j, k)) -
                                     u(i - 1, j, k) * (u(i - 1, j, k) + 2.0 * u(i, j, k)));
                    double conv_y = 0.25 / hy *
                                    ((u(i, j, k) + u(i, j + 1, k)) * (v(i - 1, j + 1, k) + v(i, j + 1, k)) -
                                     (u(i, j - 1, k) + u(i, j, k)) * (v(i - 1, j, k) + v(i, j, k)));
                    double conv_z = 0.25 / hz *
                                    ((u(i, j, k) + u(i, j, k + 1)) * (w(i - 1, j, k + 1) + w(i, j, k + 1)) -
                                     (u(i, j, k - 1) + u(i, j, k)) * (w(i - 1, j, k) + w(i, j, k)));
                    double diffuse_x = nu / hx / hx * (u(i + 1, j, k) - 2.0 * u(i, j, k) + u(i - 1, j, k));
                    double diffuse_y = nu / hy / hy * (u(i, j + 1, k) - 2.0 * u(i, j, k) + u(i, j - 1, k));
                    double diffuse_z = nu / hz / hz * (u(i, j, k + 1) - 2.0 * u(i, j, k) + u(i, j, k - 1));

                    u_temp(i, j, k) = u(i, j, k) - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
                }
            }
        }

        // v
        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    double conv_x = 0.25 / hx *
                                    ((v(i, j, k) + v(i + 1, j, k)) * (u(i + 1, j - 1, k) + u(i + 1, j, k)) -
                                     (v(i - 1, j, k) + v(i, j, k)) * (u(i, j - 1, k) + u(i, j, k)));
                    double conv_y = 0.25 / hy *
                                    (v(i, j + 1, k) * (v(i, j + 1, k) + 2.0 * v(i, j, k)) -
                                     v(i, j - 1, k) * (v(i, j - 1, k) + 2.0 * v(i, j, k)));
                    double conv_z = 0.25 / hz *
                                    ((v(i, j, k) + v(i, j, k + 1)) * (w(i, j - 1, k + 1) + w(i, j, k + 1)) -
                                     (v(i, j, k - 1) + v(i, j, k)) * (w(i, j - 1, k) + w(i, j, k)));
                    double diffuse_x = nu / hx / hx * (v(i + 1, j, k) - 2.0 * v(i, j, k) + v(i - 1, j, k));
                    double diffuse_y = nu / hy / hy * (v(i, j + 1, k) - 2.0 * v(i, j, k) + v(i, j - 1, k));
                    double diffuse_z = nu / hz / hz * (v(i, j, k + 1) - 2.0 * v(i, j, k) + v(i, j, k - 1));

                    v_temp(i, j, k) = v(i, j, k) - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
                }
            }
        }

        // w
        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    double conv_x = 0.25 / hx *
                                    ((w(i, j, k) + w(i + 1, j, k)) * (u(i + 1, j, k - 1) + u(i + 1, j, k)) -
                                     (w(i - 1, j, k) + w(i, j, k)) * (u(i, j, k - 1) + u(i, j, k)));
                    double conv_y = 0.25 / hy *
                                    ((w(i, j, k) + w(i, j + 1, k)) * (v(i, j + 1, k - 1) + v(i, j + 1, k)) -
                                     (w(i, j - 1, k) + w(i, j, k)) * (v(i, j, k - 1) + v(i, j, k)));
                    double conv_z = 0.25 / hz *
                                    (w(i, j, k + 1) * (w(i, j, k + 1) + 2.0 * w(i, j, k)) -
                                     w(i, j, k - 1) * (w(i, j, k - 1) + 2.0 * w(i, j, k)));
                    double diffuse_x = nu / hx / hx * (w(i + 1, j, k) - 2.0 * w(i, j, k) + w(i - 1, j, k));
                    double diffuse_y = nu / hy / hy * (w(i, j + 1, k) - 2.0 * w(i, j, k) + w(i, j - 1, k));
                    double diffuse_z = nu / hz / hz * (w(i, j, k + 1) - 2.0 * w(i, j, k) + w(i, j, k - 1));

                    w_temp(i, j, k) = w(i, j, k) - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
                }
            }
        }
    }
}

void ConcatNSSolver3D::euler_conv_diff_outer()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& p = *p_field_map[domain];

        field3& u_temp = *u_temp_field_map[domain];
        field3& v_temp = *v_temp_field_map[domain];
        field3& w_temp = *w_temp_field_map[domain];

        field2& u_left_buffer  = *u_buffer_map[domain][LocationType::Left];
        field2& u_right_buffer = *u_buffer_map[domain][LocationType::Right];
        field2& u_front_buffer = *u_buffer_map[domain][LocationType::Front];
        field2& u_back_buffer  = *u_buffer_map[domain][LocationType::Back];
        field2& u_down_buffer  = *u_buffer_map[domain][LocationType::Down];
        field2& u_up_buffer    = *u_buffer_map[domain][LocationType::Up];

        field2& v_left_buffer  = *v_buffer_map[domain][LocationType::Left];
        field2& v_right_buffer = *v_buffer_map[domain][LocationType::Right];
        field2& v_front_buffer = *v_buffer_map[domain][LocationType::Front];
        field2& v_back_buffer  = *v_buffer_map[domain][LocationType::Back];
        field2& v_down_buffer  = *v_buffer_map[domain][LocationType::Down];
        field2& v_up_buffer    = *v_buffer_map[domain][LocationType::Up];

        field2& w_left_buffer  = *w_buffer_map[domain][LocationType::Left];
        field2& w_right_buffer = *w_buffer_map[domain][LocationType::Right];
        field2& w_front_buffer = *w_buffer_map[domain][LocationType::Front];
        field2& w_back_buffer  = *w_buffer_map[domain][LocationType::Back];
        field2& w_down_buffer  = *w_buffer_map[domain][LocationType::Down];
        field2& w_up_buffer    = *w_buffer_map[domain][LocationType::Up];

        double* u_corner_along_y = u_corner_value_map_y[domain];
        double* u_corner_along_z = u_corner_value_map_z[domain];
        double* v_corner_along_x = v_corner_value_map_x[domain];
        double* v_corner_along_z = v_corner_value_map_z[domain];
        double* w_corner_along_x = w_corner_value_map_x[domain];
        double* w_corner_along_y = w_corner_value_map_y[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        auto bound_cal_u = [&](int i, int j, int k) {
            double u_ijk = u(i, j, k);
            double u_im1 = i == 0 ? u_left_buffer(j, k) : u(i - 1, j, k);
            double u_ip1 = i == nx - 1 ? u_right_buffer(j, k) : u(i + 1, j, k);
            double u_jm1 = j == 0 ? u_front_buffer(i, k) : u(i, j - 1, k);
            double u_jp1 = j == ny - 1 ? u_back_buffer(i, k) : u(i, j + 1, k);
            double u_km1 = k == 0 ? u_down_buffer(i, j) : u(i, j, k - 1);
            double u_kp1 = k == nz - 1 ? u_up_buffer(i, j) : u(i, j, k + 1);

            double v_im1_jp1 = i == 0 ? (j == ny - 1 ? v_corner_along_z[k] : v_left_buffer(j + 1, k)) :
                                        (j == ny - 1 ? v_back_buffer(i - 1, k) : v(i - 1, j + 1, k));
            double v_jp1     = j == ny - 1 ? v_back_buffer(i, k) : v(i, j + 1, k);
            double v_im1     = i == 0 ? v_left_buffer(j, k) : v(i - 1, j, k);

            double w_im1_kp1 = i == 0 ? (k == nz - 1 ? w_corner_along_y[j] : w_left_buffer(j, k + 1)) :
                                        (k == nz - 1 ? w_up_buffer(i - 1, j) : w(i - 1, j, k + 1));
            double w_kp1     = k == nz - 1 ? w_up_buffer(i, j) : w(i, j, k + 1);
            double w_im1     = i == 0 ? w_left_buffer(j, k) : w(i - 1, j, k);

            double conv_x = 0.25 / hx * (u_ip1 * (u_ip1 + 2.0 * u_ijk) - u_im1 * (u_im1 + 2.0 * u_ijk));
            double conv_y =
                0.25 / hy * ((u_ijk + u_jp1) * (v_im1_jp1 + v_jp1) - (u_jm1 + u_ijk) * (v_im1 + v(i, j, k)));
            double conv_z =
                0.25 / hz * ((u_ijk + u_kp1) * (w_im1_kp1 + w_kp1) - (u_km1 + u_ijk) * (w_im1 + w(i, j, k)));
            double diffuse_x = nu / hx / hx * (u_ip1 - 2.0 * u_ijk + u_im1);
            double diffuse_y = nu / hy / hy * (u_jp1 - 2.0 * u_ijk + u_jm1);
            double diffuse_z = nu / hz / hz * (u_kp1 - 2.0 * u_ijk + u_km1);

            u_temp(i, j, k) = u_ijk - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
        };

        auto bound_cal_v = [&](int i, int j, int k) {
            double v_ijk = v(i, j, k);
            double v_im1 = i == 0 ? v_left_buffer(j, k) : v(i - 1, j, k);
            double v_ip1 = i == nx - 1 ? v_right_buffer(j, k) : v(i + 1, j, k);
            double v_jm1 = j == 0 ? v_front_buffer(i, k) : v(i, j - 1, k);
            double v_jp1 = j == ny - 1 ? v_back_buffer(i, k) : v(i, j + 1, k);
            double v_km1 = k == 0 ? v_down_buffer(i, j) : v(i, j, k - 1);
            double v_kp1 = k == nz - 1 ? v_up_buffer(i, j) : v(i, j, k + 1);

            double u_ip1_jm1 = i == nx - 1 ? (j == 0 ? u_corner_along_z[k] : u_right_buffer(j - 1, k)) :
                                             (j == 0 ? u_front_buffer(i + 1, k) : u(i + 1, j - 1, k));
            double u_ip1     = i == nx - 1 ? u_right_buffer(j, k) : u(i + 1, j, k);
            double u_jm1     = j == 0 ? u_front_buffer(i, k) : u(i, j - 1, k);

            double w_jm1_kp1 = j == 0 ? (k == nz - 1 ? w_corner_along_x[i] : w_front_buffer(i, k + 1)) :
                                        (k == nz - 1 ? w_up_buffer(i, j - 1) : w(i, j - 1, k + 1));
            double w_kp1     = k == nz - 1 ? w_up_buffer(i, j) : w(i, j, k + 1);
            double w_jm1     = j == 0 ? w_front_buffer(i, k) : w(i, j - 1, k);

            double conv_x =
                0.25 / hx * ((v_ijk + v_ip1) * (u_ip1_jm1 + u_ip1) - (v_im1 + v_ijk) * (u_jm1 + u(i, j, k)));
            double conv_y = 0.25 / hy * (v_jp1 * (v_jp1 + 2.0 * v_ijk) - v_jm1 * (v_jm1 + 2.0 * v_ijk));
            double conv_z =
                0.25 / hz * ((v_ijk + v_kp1) * (w_jm1_kp1 + w_kp1) - (v_km1 + v_ijk) * (w_jm1 + w(i, j, k)));
            double diffuse_x = nu / hx / hx * (v_ip1 - 2.0 * v_ijk + v_im1);
            double diffuse_y = nu / hy / hy * (v_jp1 - 2.0 * v_ijk + v_jm1);
            double diffuse_z = nu / hz / hz * (v_kp1 - 2.0 * v_ijk + v_km1);

            v_temp(i, j, k) = v_ijk - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
        };

        auto bound_cal_w = [&](int i, int j, int k) {
            double w_ijk = w(i, j, k);
            double w_im1 = i == 0 ? w_left_buffer(j, k) : w(i - 1, j, k);
            double w_ip1 = i == nx - 1 ? w_right_buffer(j, k) : w(i + 1, j, k);
            double w_jm1 = j == 0 ? w_front_buffer(i, k) : w(i, j - 1, k);
            double w_jp1 = j == ny - 1 ? w_back_buffer(i, k) : w(i, j + 1, k);
            double w_km1 = k == 0 ? w_down_buffer(i, j) : w(i, j, k - 1);
            double w_kp1 = k == nz - 1 ? w_up_buffer(i, j) : w(i, j, k + 1);

            double u_ip1_km1 = i == nx - 1 ? (k == 0 ? u_corner_along_y[j] : u_right_buffer(j, k - 1)) :
                                             (k == 0 ? u_down_buffer(i + 1, j) : u(i + 1, j, k - 1));
            double u_ip1     = i == nx - 1 ? u_right_buffer(j, k) : u(i + 1, j, k);
            double u_km1     = k == 0 ? u_down_buffer(i, j) : u(i, j, k - 1);

            double v_jp1_km1 = j == ny - 1 ? (k == 0 ? v_corner_along_x[i] : v_back_buffer(i, k - 1)) :
                                             (k == 0 ? v_down_buffer(i, j + 1) : v(i, j + 1, k - 1));
            double v_jp1     = j == ny - 1 ? v_back_buffer(i, k) : v(i, j + 1, k);
            double v_km1     = k == 0 ? v_down_buffer(i, j) : v(i, j, k - 1);

            double conv_x =
                0.25 / hx * ((w_ijk + w_ip1) * (u_ip1_km1 + u_ip1) - (w_im1 + w_ijk) * (u_km1 + u(i, j, k)));
            double conv_y =
                0.25 / hy * ((w_ijk + w_jp1) * (v_jp1_km1 + v_jp1) - (w_jm1 + w_ijk) * (v_km1 + v(i, j, k)));
            double conv_z = 0.25 / hz * (w_kp1 * (w_kp1 + 2.0 * w_ijk) - w_km1 * (w_km1 + 2.0 * w_ijk));

            double diffuse_x = nu / hx / hx * (w_ip1 - 2.0 * w_ijk + w_im1);
            double diffuse_y = nu / hy / hy * (w_jp1 - 2.0 * w_ijk + w_jm1);
            double diffuse_z = nu / hz / hz * (w_kp1 - 2.0 * w_ijk + w_km1);

            w_temp(i, j, k) = w_ijk - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
        };

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                if (u_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Adjacented)
                    bound_cal_u(0, j, k);
                bound_cal_u(nx - 1, j, k);
                bound_cal_v(0, j, k);
                bound_cal_v(nx - 1, j, k);
                bound_cal_w(0, j, k);
                bound_cal_w(nx - 1, j, k);
            }
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int k = 0; k < nz; k++)
            {
                bound_cal_u(i, 0, k);
                bound_cal_u(i, ny - 1, k);
                if (v_var->boundary_type_map[domain][LocationType::Front] == PDEBoundaryType::Adjacented)
                    bound_cal_v(i, 0, k);
                bound_cal_v(i, ny - 1, k);
                bound_cal_w(i, 0, k);
                bound_cal_w(i, ny - 1, k);
            }
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                bound_cal_u(i, j, 0);
                bound_cal_u(i, j, nz - 1);
                bound_cal_v(i, j, 0);
                bound_cal_v(i, j, nz - 1);
                if (w_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Adjacented)
                    bound_cal_w(i, j, 0);
                bound_cal_w(i, j, nz - 1);
            }
        }

        swap_field_data(u, u_temp);
        swap_field_data(v, v_temp);
        swap_field_data(w, w_temp);
    }
}