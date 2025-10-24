#include "ns_solver2d.h"

ConcatNSSolver2D::ConcatNSSolver2D(Variable* in_u_var, Variable* in_v_var, Variable* in_p_var, TimeAdvancingConfig* in_time_config, PhysicsConfig* in_physics_config, EnvironmentConfig* in_env_config)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , p_var(in_p_var)
    , time_config(in_time_config)
    , phy_config(in_physics_config)
    , env_config(in_env_config)
    , left_up_corner_map(v_var->left_up_corner_map)
    , right_down_corner_map(u_var->right_down_corner_map)
{
    //config load
    if (in_env_config)
    {
        //Maybe useful in future
    }
    
    dt = time_config->dt;
    num_it = time_config->num_iterations;
    
    nu = phy_config->nu;

    //check u v p share one geometry
    if (u_var->geometry != v_var->geometry || u_var->geometry != p_var->geometry)
        throw std::runtime_error("ConcatNSSolver2D: u v p do not share one geometry");
    
    // geometry double check
    if (u_var->geometry == nullptr)
        throw std::runtime_error("ConcatNSSolver2D: u->geometry is null");
    if (!u_var->geometry->is_checked)
        u_var->geometry->check();
    if (u_var->geometry->tree_root == nullptr || u_var->geometry->tree_map.empty())
        u_var->geometry->solve_prepare();

    domains = u_var->geometry->domains;
    adjacency = u_var->geometry->adjacency;

    u_field_map = u_var->field_map;
    v_field_map = v_var->field_map;
    p_field_map = p_var->field_map;

    u_buffer_map = u_var->buffer_map;
    v_buffer_map = v_var->buffer_map;
    p_buffer_map = p_var->buffer_map;

    //Construct the temp field for each domain
    for (auto &[domain, field] : u_field_map)
        u_temp_field_map[domain] = new field2(field->get_nx(), field->get_ny(), field->get_name() + "_temp");
    for (auto &[domain, field] : v_field_map)
        v_temp_field_map[domain] = new field2(field->get_nx(), field->get_ny(), field->get_name() + "_temp");
}

ConcatNSSolver2D::~ConcatNSSolver2D()
{
    for (auto &[domain, field] : u_field_map)
        delete u_temp_field_map[domain];
    for (auto &[domain, field] : v_field_map)
        delete v_temp_field_map[domain];
}

void ConcatNSSolver2D::variable_check()
{
    if (u_var->position_type != VariablePositionType::XEdge)
        throw std::runtime_error("ConcatNSSolver2D: u->position_type is not XEdge");
    if (v_var->position_type != VariablePositionType::YEdge)
        throw std::runtime_error("ConcatNSSolver2D: v->position_type is not YEdge");
    if (p_var->position_type != VariablePositionType::Center)
        throw std::runtime_error("ConcatNSSolver2D: p->position_type is not Center");
}


void ConcatNSSolver2D::solve()
{
    boundary_init();
    for (int it = 0; it < num_it; it++)
    {
        euler_conv_diff_inner();
        buffer_pass();
        euler_conv_diff_outer();
        boundary_update();
    }
}

void ConcatNSSolver2D::euler_conv_diff_inner()
{
    for (auto &domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];
        field2& p = *p_field_map[domain];

        field2& u_temp = *u_temp_field_map[domain];
        field2& v_temp = *v_temp_field_map[domain];

        int nx = u.get_nx();
        int ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        // u
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double conv_x = u(i + 1, j) * (u(i + 1, j) + 2.0 * u(i, j)) - u(i - 1, j) * (u(i - 1, j) + 2.0 * u(i, j));
                double conv_y = (u(i, j) + u(i, j + 1)) * (v(i - 1, j + 1) + v(i, j + 1)) - (u(i, j - 1) + u(i, j)) * (v(i - 1, j) + v(i, j));
                double diff = (u(i + 1, j) + u(i - 1, j) - 2.0 * u(i, j)) / hx / hx + (u(i, j + 1) + u(i, j - 1) - 2.0 * u(i, j)) / hy / hy;

                u_temp(i, j) = -0.25 / hx * conv_x - 0.25 / hy * conv_y + nu * diff;
            }
        }

        // v
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double conv_x = (v(i, j) + v(i + 1, j)) * (u(i + 1, j - 1) + u(i + 1, j)) - (v(i - 1, j) + v(i, j)) * (u(i, j - 1) + u(i, j));
                double conv_y = v(i, j + 1) * (v(i, j + 1) + 2.0 * v(i, j)) - v(i, j - 1) * (v(i, j - 1) + 2.0 * v(i, j));
                double diff = (v(i + 1, j) + v(i - 1, j) - 2.0 * v(i, j)) / hx / hx + (v(i, j + 1) + v(i, j - 1) - 2.0 * v(i, j)) / hy / hy;

                v_temp(i, j) = -0.25 / hx * conv_x - 0.25 / hy * conv_y + nu * diff;
            }
        }
    }
}

void ConcatNSSolver2D::euler_conv_diff_outer()
{

}