#include "scalar_solver3d.h"

#include "base/config.h"
#include "boundary_3d_utils.h"

ScalarSolver3D::ScalarSolver3D(Variable3D* in_u_var,
                               Variable3D* in_v_var,
                               Variable3D* in_w_var,
                               Variable3D* in_s_var,
                               double      _nr)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , w_var(in_w_var)
    , s_var(in_s_var)
{
    TimeAdvancingConfig& time_cfg    = TimeAdvancingConfig::Get();
    PhysicsConfig&       physics_cfg = PhysicsConfig::Get();

    dt = time_cfg.dt;
    nr = _nr;

    // check u v p share one geometry
    if (u_var->geometry != v_var->geometry || u_var->geometry != w_var->geometry || u_var->geometry != s_var->geometry)
        throw std::runtime_error("ScalarSolver3D: u v w p do not share one geometry");

    // geometry double check
    if (u_var->geometry == nullptr)
        throw std::runtime_error("ScalarSolver3D: u->geometry is nrll");
    if (!u_var->geometry->is_checked)
        u_var->geometry->check();
    if (u_var->geometry->tree_root == nullptr || u_var->geometry->tree_map.empty())
        u_var->geometry->solve_prepare();

    domains   = u_var->geometry->domains;
    adjacency = u_var->geometry->adjacency;

    u_field_map = u_var->field_map;
    v_field_map = v_var->field_map;
    w_field_map = w_var->field_map;
    s_field_map = s_var->field_map;

    u_buffer_map = u_var->buffer_map;
    v_buffer_map = v_var->buffer_map;
    w_buffer_map = w_var->buffer_map;
    s_buffer_map = s_var->buffer_map;

    // Construct the temp field for each domain
    for (auto& [domain, field] : s_field_map)
        s_temp_field_map[domain] =
            new field3(field->get_nx(), field->get_ny(), field->get_nz(), field->get_name() + "_temp");

    // update boundary for first step
    phys_boundary_update();
    nondiag_shared_boundary_update();
}

void ScalarSolver3D::variable_check()
{
    if (u_var->position_type != VariablePositionType::XFace)
        throw std::runtime_error("ScalarSolver3D: u->position_type is not XFace");
    if (v_var->position_type != VariablePositionType::YFace)
        throw std::runtime_error("ScalarSolver3D: v->position_type is not YFace");
    if (w_var->position_type != VariablePositionType::ZFace)
        throw std::runtime_error("ScalarSolver3D: w->position_type is not ZFace");
    if (s_var->position_type != VariablePositionType::Center)
        throw std::runtime_error("ScalarSolver3D: p->position_type is not Center");
}

void ScalarSolver3D::solve()
{
    euler_conv_diff_inner();
    euler_conv_diff_outer();

    phys_boundary_update();
    nondiag_shared_boundary_update();
}

void ScalarSolver3D::euler_conv_diff_inner()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& s = *s_field_map[domain];

        field3& s_temp = *s_temp_field_map[domain];

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
                    double conv_x =
                        0.5 / hx *
                        (u(i + 1, j, k) * (s(i + 1, j, k) + s(i, j, k)) - u(i, j, k) * (s(i - 1, j, k) + s(i, j, k)));
                    double conv_y =
                        0.5 / hy *
                        (v(i, j + 1, k) * (s(i, j, k) + s(i, j + 1, k)) - v(i, j, k) * (s(i, j - 1, k) + s(i, j, k)));
                    double conv_z =
                        0.5 / hz *
                        (w(i, j, k + 1) * (s(i, j, k) + s(i, j, k + 1)) - w(i, j, k) * (s(i, j, k - 1) + s(i, j, k)));
                    double diffuse_x = nr / hx / hx * (s(i + 1, j, k) - 2.0 * s(i, j, k) + s(i - 1, j, k));
                    double diffuse_y = nr / hy / hy * (s(i, j + 1, k) - 2.0 * s(i, j, k) + s(i, j - 1, k));
                    double diffuse_z = nr / hz / hz * (s(i, j, k + 1) - 2.0 * s(i, j, k) + s(i, j, k - 1));

                    s_temp(i, j, k) = s(i, j, k) - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
                }
            }
        }
    }
}

void ScalarSolver3D::euler_conv_diff_outer()
{
    for (auto& domain : domains)
    {
        field3& u = *u_field_map[domain];
        field3& v = *v_field_map[domain];
        field3& w = *w_field_map[domain];
        field3& s = *s_field_map[domain];

        field3& s_temp = *s_temp_field_map[domain];

        field2& u_right_buffer = *u_buffer_map[domain][LocationType::Right];
        field2& v_back_buffer  = *v_buffer_map[domain][LocationType::Back];
        field2& w_up_buffer    = *w_buffer_map[domain][LocationType::Up];

        field2& s_left_buffer  = *s_buffer_map[domain][LocationType::Left];
        field2& s_right_buffer = *s_buffer_map[domain][LocationType::Right];
        field2& s_front_buffer = *s_buffer_map[domain][LocationType::Front];
        field2& s_back_buffer  = *s_buffer_map[domain][LocationType::Back];
        field2& s_down_buffer  = *s_buffer_map[domain][LocationType::Down];
        field2& s_up_buffer    = *s_buffer_map[domain][LocationType::Up];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        int    nz = u.get_nz();
        double hx = domain->hx;
        double hy = domain->hy;
        double hz = domain->hz;

        auto bound_cal_s = [&](int i, int j, int k) {
            double u_ijk = u(i, j, k);
            double v_ijk = v(i, j, k);
            double w_ijk = w(i, j, k);
            double u_ip1 = i == nx - 1 ? u_right_buffer(j, k) : u(i + 1, j, k);
            double v_jp1 = j == ny - 1 ? v_back_buffer(i, k) : v(i, j + 1, k);
            double w_kp1 = k == nz - 1 ? w_up_buffer(i, j) : w(i, j, k + 1);

            double s_ijk = s(i, j, k);
            double s_im1 = i == 0 ? s_left_buffer(j, k) : s(i - 1, j, k);
            double s_ip1 = i == nx - 1 ? s_right_buffer(j, k) : s(i + 1, j, k);
            double s_jm1 = j == 0 ? s_front_buffer(i, k) : s(i, j - 1, k);
            double s_jp1 = k == ny - 1 ? s_back_buffer(i, k) : s(i, j + 1, k);
            double s_km1 = k == 0 ? s_down_buffer(i, j) : s(i, j, k - 1);
            double s_kp1 = k == nz - 1 ? s_up_buffer(i, j) : s(i, j, k + 1);

            double conv_x    = 0.5 / hx * (u_ip1 * (s_ip1 + s_ijk) - u_ijk * (s_im1 + s_ijk));
            double conv_y    = 0.5 / hy * (v_jp1 * (s_ijk + s_jp1) - v_ijk * (s_jm1 + s_ijk));
            double conv_z    = 0.5 / hz * (w_kp1 * (s_ijk + s_kp1) - w_ijk * (s_km1 + s_ijk));
            double diffuse_x = nr / hx / hx * (s_im1 - 2.0 * s_ijk + s_im1);
            double diffuse_y = nr / hy / hy * (s_jp1 - 2.0 * s_ijk + s_jm1);
            double diffuse_z = nr / hz / hz * (s_kp1 - 2.0 * s_ijk + s_km1);

            s_temp(i, j, k) = s_ijk - dt * (conv_x + conv_y + conv_z - diffuse_x - diffuse_y - diffuse_z);
        };

        OPENMP_PARALLEL_FOR()
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                bound_cal_s(0, j, k);
                bound_cal_s(nx - 1, j, k);
            }
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int k = 0; k < nz; k++)
            {
                bound_cal_s(i, 0, k);
                bound_cal_s(i, ny - 1, k);
            }
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                bound_cal_s(i, j, 0);
                bound_cal_s(i, j, nz - 1);
            }
        }

        swap_field_data(s, s_temp);
    }
}