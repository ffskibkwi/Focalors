#include "boundary_2d_utils.h"
#include "mhd_module_2d.h"
#include "ns_solver2d.h"

/** @brief Minimum shear rate threshold to prevent singularity in power-law model */
constexpr double GAMMA_DOT_MIN = 1.0e-4;

void ConcatNSSolver2D::init_nonnewton(Variable2D* in_mu_var,
                                      Variable2D* in_tau_xx_var,
                                      Variable2D* in_tau_yy_var,
                                      Variable2D* in_tau_xy_var)
{
    mu_var     = in_mu_var;
    tau_xx_var = in_tau_xx_var;
    tau_yy_var = in_tau_yy_var;
    tau_xy_var = in_tau_xy_var;

    mu_field_map     = mu_var->field_map;
    tau_xx_field_map = tau_xx_var->field_map;
    tau_yy_field_map = tau_yy_var->field_map;
    tau_xy_field_map = tau_xy_var->field_map;

    tau_xx_buffer_map = tau_xx_var->buffer_map;
    tau_yy_buffer_map = tau_yy_var->buffer_map;
}

void ConcatNSSolver2D::solve_nonnewton()
{
    // 1. Update boundary for NS (Ghost Cells / Buffers)
    phys_boundary_update();
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();

    // 2. Calculate Viscosity (mu) based on current velocity field
    viscosity_update();

    // 3. Calculate Stress Tensor (tau) based on velocity and viscosity
    stress_update();

    // 3.1 Update Stress Buffers (tau_xx, tau_yy)
    stress_buffer_update();

    // 4. Solve Momentum Equation (Predictor Step)
    // Replaces euler_conv_diff_inner/outer with Non-Newtonian versions
    euler_conv_diff_inner_nonnewton();
    euler_conv_diff_outer_nonnewton();

    // MHD: predictor step finished, before div(u) boundary update
    if (phy_config->enable_mhd)
    {
        if (!mhd_module)
            mhd_module =
                std::unique_ptr<MHDModule2D>(new MHDModule2D(u_var, v_var, phy_config, time_config, env_config));
        mhd_module->init();
        mhd_module->solveElectricPotential();
        mhd_module->updateCurrentDensity();
        mhd_module->applyLorentzForce();

        // refresh predicted velocity boundary after applying Lorentz force
        phys_boundary_update();
        nondiag_shared_boundary_update();
    }

    // 5. Update boundary for divu (Prepare for Pressure Projection)
    phys_boundary_update();
    nondiag_shared_boundary_update();

    // 6. Pressure Correction Loop (Projection Method)
    for (int it = 0; it < corr_it; it++)
    {
        // Calculate Velocity Divergence
        velocity_div_inner();
        velocity_div_outer();

        // Solve Poisson Equation for Pressure
        normalize_pressure();
        p_solver->solve();

        // Update Pressure Boundaries
        pressure_buffer_update();

        // Correct Velocity with Pressure Gradient
        add_pressure_gradient();
    }
}

void ConcatNSSolver2D::viscosity_update()
{
    for (auto& domain : domains)
    {
        field2& u  = *u_field_map[domain];
        field2& v  = *v_field_map[domain];
        field2& mu = *mu_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        double* u_left_buffer  = u_buffer_map[domain][LocationType::Left];
        double* u_right_buffer = u_buffer_map[domain][LocationType::Right];
        double* u_down_buffer  = u_buffer_map[domain][LocationType::Down];
        double* u_up_buffer    = u_buffer_map[domain][LocationType::Up];

        double* v_left_buffer  = v_buffer_map[domain][LocationType::Left];
        double* v_right_buffer = v_buffer_map[domain][LocationType::Right];
        double* v_down_buffer  = v_buffer_map[domain][LocationType::Down];
        double* v_up_buffer    = v_buffer_map[domain][LocationType::Up];

        // Helper to get u at (i, j) handling boundaries
        // u is defined at (i, j+0.5) for i in [0, nx], j in [-1, ny]
        auto get_u = [&](int i, int j) -> double {
            return get_u_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       u,
                                       u_left_buffer,
                                       u_right_buffer,
                                       u_down_buffer,
                                       u_up_buffer,
                                       right_down_corner_value_map[domain]);
        };

        // Helper to get v at (i, j) handling boundaries
        // v is defined at (i+0.5, j) for i in [-1, nx], j in [0, ny]
        auto get_v = [&](int i, int j) -> double {
            return get_v_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       v,
                                       v_left_buffer,
                                       v_right_buffer,
                                       v_down_buffer,
                                       v_up_buffer,
                                       left_up_corner_value_map[domain]);
        };

        // Helper lambda for du/dx at (i, j_row) where u is defined
        auto calc_du_dx_row = [&](int i_idx, int j_idx) -> double {
            if (i_idx > 0 && i_idx < nx)
                return (get_u(i_idx + 1, j_idx) - get_u(i_idx - 1, j_idx)) / (2.0 * hx);
            else if (i_idx == 0)
                return (-3 * get_u(0, j_idx) + 4 * get_u(1, j_idx) - get_u(2, j_idx)) /
                       hx; // Forward difference at 2 order accuaracy
            else           // i_idx == nx
                return (3 * get_u(nx, j_idx) - 4 * get_u(nx - 1, j_idx) + get_u(nx - 2, j_idx)) /
                       hx; // Backward difference at 2 order accuaracy
        };

        // Helper lambda for dv/dy at (i_col, j) where v is defined
        auto calc_dv_dy_col = [&](int i_idx, int j_idx) -> double {
            if (j_idx > 0 && j_idx < ny)
                return (get_v(i_idx, j_idx + 1) - get_v(i_idx, j_idx - 1)) / (2.0 * hy);
            else if (j_idx == 0)
                return (-3 * get_v(i_idx, 0) + 4 * get_v(i_idx, 1) - get_v(i_idx, 2)) /
                       hy; // Forward difference at 2 order accuaracy
            else           // j_idx == ny
                return (3 * get_v(i_idx, ny) - 4 * get_v(i_idx, ny - 1) + get_v(i_idx, ny - 2)) /
                       hy; // Backward difference at 2 order accuaracy
        };

        // Calculate viscosity at Nodes (nx+1, ny+1)
        // Exclude (nx, ny) point
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i <= nx; i++)
        {
            int j_end = (i == nx) ? ny : ny + 1;
            for (int j = 0; j < j_end; j++)
            {
                // 1. Calculate du_dy and dv_dx at Node (i, j)
                // Node (i, j) is at x=i*hx, y=j*hy

                // du_dy approx (u(i, j) - u(i, j-1)) / hy
                // u(i, j) is at y=j+0.5, u(i, j-1) is at y=j-0.5. Midpoint is y=j. Correct.
                double val_u_j   = get_u(i, j);     // u(i, j)
                double val_u_jm1 = get_u(i, j - 1); // u(i, j-1)
                double du_dy     = (val_u_j - val_u_jm1) / hy;

                // dv_dx approx (v(i, j) - v(i-1, j)) / hx
                // v(i, j) is at x=i+0.5, v(i-1, j) is at x=i-0.5. Midpoint is x=i. Correct.
                double val_v_i   = get_v(i, j);     // v(i, j)
                double val_v_im1 = get_v(i - 1, j); // v(i-1, j)
                double dv_dx     = (val_v_i - val_v_im1) / hx;

                // TODO 可以在不改变buffer的情况下，主动根据边界类型及其值计算更精确的边界导数，暂时先不考虑
                // 2. Calculate du_dx and dv_dy at Node (i, j)
                // Using central differencing for interior nodes
                // Using one-sided differencing for boundary nodes
                double du_dx = 0.5 * (calc_du_dx_row(i, j) + calc_du_dx_row(i, j - 1));
                double dv_dy = 0.5 * (calc_dv_dy_col(i, j) + calc_dv_dy_col(i - 1, j));

                // 3. Calculate Shear Rate GammaDot
                // gamma_dot = sqrt( 2*(du_dx^2 + dv_dy^2) + (du_dy + dv_dx)^2 )
                double gamma_dot = std::sqrt(2.0 * (du_dx * du_dx + dv_dy * dv_dy) + (du_dy + dv_dx) * (du_dy + dv_dx));

                // 4. Update Viscosity
                double mu_val = phy_config->nu;

                if (phy_config->model_type == 1) // Power Law
                {
                    double k      = phy_config->k;
                    double n      = phy_config->n;
                    double mu_min = phy_config->mu_min;
                    double mu_max = phy_config->mu_max;
                    if (n < 1.0)
                        gamma_dot = std::max(gamma_dot, GAMMA_DOT_MIN); // protect against zero when exponent negative
                    // Avoid division by zero if gamma_dot is 0 and n < 1
                    mu_val = k * std::pow(gamma_dot, n - 1.0);

                    // Enforce limits
                    mu_val = std::max(mu_min, std::min(mu_val, mu_max));
                }
                else if (phy_config->model_type == 2) // Carreau
                {
                    double mu_0   = phy_config->mu_0;
                    double mu_inf = phy_config->mu_inf;
                    double lambda = phy_config->lambda;
                    double n      = phy_config->n;
                    double a      = phy_config->a;

                    mu_val = mu_inf + (mu_0 - mu_inf) * std::pow(1.0 + std::pow(lambda * gamma_dot, a), (n - 1.0) / a);
                }

                mu(i, j) = mu_val;
            }
        }
    }
}

void ConcatNSSolver2D::stress_buffer_update()
{
    for (auto& domain : domains)
    {
        field2& u = *u_field_map[domain];
        field2& v = *v_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        double* u_left_buffer = u_buffer_map[domain][LocationType::Left];
        double* v_left_buffer = v_buffer_map[domain][LocationType::Left];

        double* u_down_buffer = u_buffer_map[domain][LocationType::Down];
        double* v_down_buffer = v_buffer_map[domain][LocationType::Down];

        double* tau_xx_left_buffer = tau_xx_buffer_map[domain][LocationType::Left];
        double* tau_yy_down_buffer = tau_yy_buffer_map[domain][LocationType::Down];

        // Helper to get u at (i, j) handling boundaries
        auto get_u = [&](int i_idx, int j_idx) {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       u,
                                       u_left_buffer,
                                       u_buffer_map[domain][LocationType::Right],
                                       u_down_buffer,
                                       u_buffer_map[domain][LocationType::Up],
                                       right_down_corner_value_map[domain]);
        };

        // Helper to get v at (i, j) handling boundaries
        auto get_v = [&](int i_idx, int j_idx) {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       v,
                                       v_left_buffer,
                                       v_buffer_map[domain][LocationType::Right],
                                       v_down_buffer,
                                       v_buffer_map[domain][LocationType::Up],
                                       left_up_corner_value_map[domain]);
        };

        // 1. Update tau_xx left buffer (at ghost cell -1, j)
        // Left Boundary (i=0 needs tau_xx(-1, j))
        for (int j = 0; j < ny; j++)
        {
            // Calculate tau_xx at ghost cell (-1, j)
            // 1. du/dx at (-0.5, j+0.5)
            // u(0, j) is at x=0, u_left_buffer[j] is at x=-1
            double du_dx_ghost = (u(0, j) - u_left_buffer[j]) / hx;

            // 2. dv/dy at (-0.5, j+0.5)
            // v_left_buffer is at x=-0.5. Need dy.
            // v_left_buffer[j] is at y=j, v_left_buffer[j+1] is at y=j+1
            // Use get_v to handle corner cases safely
            double v_left_j    = get_v(-1, j);
            double v_left_jp1  = get_v(-1, j + 1);
            double dv_dy_ghost = (v_left_jp1 - v_left_j) / hy;

            // 3. du/dy at (-0.5, j+0.5)
            // Average of du/dy at x=-1 (u_left_buffer) and x=0 (u(0, j))
            auto get_du_dy_col = [&](int col_idx) {
                if (j == 0)
                {
                    if (ny >= 3)
                        return (-3.0 * get_u(col_idx, 0) + 4.0 * get_u(col_idx, 1) - get_u(col_idx, 2)) / (2.0 * hy);
                    else
                        return (get_u(col_idx, 1) - get_u(col_idx, 0)) / hy;
                }
                else if (j == ny - 1)
                {
                    if (ny >= 3)
                        return (3.0 * get_u(col_idx, ny - 1) - 4.0 * get_u(col_idx, ny - 2) + get_u(col_idx, ny - 3)) /
                               (2.0 * hy);
                    else
                        return (get_u(col_idx, ny - 1) - get_u(col_idx, ny - 2)) / hy;
                }
                else
                {
                    return (get_u(col_idx, j + 1) - get_u(col_idx, j - 1)) / (2.0 * hy);
                }
            };

            double du_dy_left  = get_du_dy_col(-1);
            double du_dy_0     = get_du_dy_col(0);
            double du_dy_ghost = 0.5 * (du_dy_left + du_dy_0);

            // 4. dv/dx at (-0.5, j+0.5)
            // Need dv/dx at x=-0.5.
            // Points: x=-0.5 (v_left), x=0.5 (v0), x=1.5 (v1)
            auto get_dv_dx_node = [&](int k) {
                double v_l = get_v(-1, k);
                double v_0 = get_v(0, k);
                if (nx >= 2)
                {
                    double v_1 = get_v(1, k);
                    return (-3.0 * v_l + 4.0 * v_0 - v_1) / (2.0 * hx);
                }
                else
                {
                    return (v_0 - v_l) / hx;
                }
            };
            double dv_dx_ghost = 0.5 * (get_dv_dx_node(j) + get_dv_dx_node(j + 1));

            // 5. Gamma Dot
            double gamma_dot = std::sqrt(2.0 * (du_dx_ghost * du_dx_ghost + dv_dy_ghost * dv_dy_ghost) +
                                         (du_dy_ghost + dv_dx_ghost) * (du_dy_ghost + dv_dx_ghost));

            // 6. Viscosity
            double mu_val = phy_config->nu;
            if (phy_config->model_type == 1) // Power Law
            {
                double k      = phy_config->k;
                double n      = phy_config->n;
                double mu_min = phy_config->mu_min;
                double mu_max = phy_config->mu_max;
                if (n < 1.0)
                    gamma_dot = std::max(gamma_dot, GAMMA_DOT_MIN);
                mu_val = k * std::pow(gamma_dot, n - 1.0);
                mu_val = std::max(mu_min, std::min(mu_val, mu_max));
            }
            else if (phy_config->model_type == 2) // Carreau
            {
                double mu_0   = phy_config->mu_0;
                double mu_inf = phy_config->mu_inf;
                double lambda = phy_config->lambda;
                double n      = phy_config->n;
                double a      = phy_config->a;
                mu_val = mu_inf + (mu_0 - mu_inf) * std::pow(1.0 + std::pow(lambda * gamma_dot, a), (n - 1.0) / a);
            }

            tau_xx_left_buffer[j] = 2.0 * mu_val * du_dx_ghost;
        }

        // Down Boundary (j=0 needs tau_yy(i, -1))
        for (int i = 0; i < nx; i++)
        {
            // Calculate tau_yy at ghost cell (i, -1)
            // 1. dv/dy at (i+0.5, -0.5)
            // v(i, 0) is at y=0, v_down_buffer[i] is at y=-1
            double dv_dy_ghost = (v(i, 0) - v_down_buffer[i]) / hy;

            // 2. du/dx at (i+0.5, -0.5)
            // u_down_buffer is at y=-0.5. Need dx.
            // u_down_buffer[i] is at x=i, u_down_buffer[i+1] is at x=i+1
            // Use get_u to handle corner cases safely
            double u_down_i    = get_u(i, -1);
            double u_down_ip1  = get_u(i + 1, -1);
            double du_dx_ghost = (u_down_ip1 - u_down_i) / hx;

            // 3. dv/dx at (i+0.5, -0.5)
            // Average of dv/dx at y=-1 (v_down_buffer) and y=0 (v(i, 0))
            auto get_dv_dx_row = [&](int row_idx) {
                if (i == 0)
                {
                    if (nx >= 3)
                        return (-3.0 * get_v(0, row_idx) + 4.0 * get_v(1, row_idx) - get_v(2, row_idx)) / (2.0 * hx);
                    else
                        return (get_v(1, row_idx) - get_v(0, row_idx)) / hx;
                }
                else if (i == nx - 1)
                {
                    if (nx >= 3)
                        return (3.0 * get_v(nx - 1, row_idx) - 4.0 * get_v(nx - 2, row_idx) + get_v(nx - 3, row_idx)) /
                               (2.0 * hx);
                    else
                        return (get_v(nx - 1, row_idx) - get_v(nx - 2, row_idx)) / hx;
                }
                else
                {
                    return (get_v(i + 1, row_idx) - get_v(i - 1, row_idx)) / (2.0 * hx);
                }
            };

            double dv_dx_down  = get_dv_dx_row(-1);
            double dv_dx_0     = get_dv_dx_row(0);
            double dv_dx_ghost = 0.5 * (dv_dx_down + dv_dx_0);

            // 4. du/dy at (i+0.5, -0.5)
            // Need du/dy at y=-0.5.
            // Points: y=-0.5 (u_down), y=0.5 (u0), y=1.5 (u1)
            auto get_du_dy_node = [&](int k) {
                double u_d = get_u(k, -1);
                double u_0 = get_u(k, 0);
                if (ny >= 2)
                {
                    double u_1 = get_u(k, 1);
                    return (-3.0 * u_d + 4.0 * u_0 - u_1) / (2.0 * hy);
                }
                else
                {
                    return (u_0 - u_d) / hy;
                }
            };
            double du_dy_ghost = 0.5 * (get_du_dy_node(i) + get_du_dy_node(i + 1));

            // 5. Gamma Dot
            double gamma_dot = std::sqrt(2.0 * (du_dx_ghost * du_dx_ghost + dv_dy_ghost * dv_dy_ghost) +
                                         (du_dy_ghost + dv_dx_ghost) * (du_dy_ghost + dv_dx_ghost));

            // 6. Viscosity
            double mu_val = phy_config->nu;
            if (phy_config->model_type == 1) // Power Law
            {
                double k      = phy_config->k;
                double n      = phy_config->n;
                double mu_min = phy_config->mu_min;
                double mu_max = phy_config->mu_max;
                if (n < 1.0)
                    gamma_dot = std::max(gamma_dot, GAMMA_DOT_MIN);
                mu_val = k * std::pow(gamma_dot, n - 1.0);
                mu_val = std::max(mu_min, std::min(mu_val, mu_max));
            }
            else if (phy_config->model_type == 2) // Carreau
            {
                double mu_0   = phy_config->mu_0;
                double mu_inf = phy_config->mu_inf;
                double lambda = phy_config->lambda;
                double n      = phy_config->n;
                double a      = phy_config->a;
                mu_val = mu_inf + (mu_0 - mu_inf) * std::pow(1.0 + std::pow(lambda * gamma_dot, a), (n - 1.0) / a);
            }

            tau_yy_down_buffer[i] = 2.0 * mu_val * dv_dy_ghost;
        }
    }
}

void ConcatNSSolver2D::stress_update()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& mu     = *mu_field_map[domain];
        field2& tau_xx = *tau_xx_field_map[domain];
        field2& tau_yy = *tau_yy_field_map[domain];
        field2& tau_xy = *tau_xy_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        double* u_left_buffer  = u_buffer_map[domain][LocationType::Left];
        double* u_right_buffer = u_buffer_map[domain][LocationType::Right];
        double* u_down_buffer  = u_buffer_map[domain][LocationType::Down];
        double* u_up_buffer    = u_buffer_map[domain][LocationType::Up];

        double* v_left_buffer  = v_buffer_map[domain][LocationType::Left];
        double* v_right_buffer = v_buffer_map[domain][LocationType::Right];
        double* v_down_buffer  = v_buffer_map[domain][LocationType::Down];
        double* v_up_buffer    = v_buffer_map[domain][LocationType::Up];

        // Helper to get u at (i, j) handling boundaries for tau_xy calculation
        auto get_u = [&](int i, int j) -> double {
            return get_u_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       u,
                                       u_left_buffer,
                                       u_right_buffer,
                                       u_down_buffer,
                                       u_up_buffer,
                                       right_down_corner_value_map[domain]);
        };

        // Helper to get v at (i, j) handling boundaries for tau_xy calculation
        auto get_v = [&](int i, int j) -> double {
            return get_v_with_boundary(i,
                                       j,
                                       nx,
                                       ny,
                                       v,
                                       v_left_buffer,
                                       v_right_buffer,
                                       v_down_buffer,
                                       v_up_buffer,
                                       left_up_corner_value_map[domain]);
        };

        // 1. Calculate Normal Stresses (tau_xx, tau_yy) at Centers
        // Loop over cells (i, j)
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Calculate mu at center by averaging 4 corners
                // Corners of cell (i, j) are (i, j), (i+1, j), (i, j+1), (i+1, j+1)
                double mu_cen = (i == nx - 1 && j == ny - 1) ?
                                    0.5 * (mu(i + 1, j) + mu(i, j + 1)) : // Only two corners available
                                    0.25 * (mu(i, j) + mu(i + 1, j) + mu(i, j + 1) + mu(i + 1, j + 1));

                // du/dx at center
                double du_dx = (get_u(i + 1, j) - u(i, j)) / hx;

                // dv/dy at center
                double dv_dy = (get_v(i, j + 1) - v(i, j)) / hy;

                tau_xx(i, j) = 2.0 * mu_cen * du_dx;
                tau_yy(i, j) = 2.0 * mu_cen * dv_dy;
            }
        }

        // 2. Calculate Shear Stress (tau_xy) at Nodes
        // Loop over nodes (i, j) from 0 to nx, 0 to ny
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i <= nx; i++)
        {
            int j_end = (i == nx) ? ny : ny + 1;
            for (int j = 0; j < j_end; j++)
            {
                // mu is already at nodes
                double mu_node = mu(i, j);

                // du/dy at node (i, j)
                // u(i, j) is at y=j+0.5, u(i, j-1) is at y=j-0.5
                double val_u_j   = get_u(i, j);
                double val_u_jm1 = get_u(i, j - 1);
                double du_dy     = (val_u_j - val_u_jm1) / hy;

                // dv/dx at node (i, j)
                // v(i, j) is at x=i+0.5, v(i-1, j) is at x=i-0.5
                double val_v_i   = get_v(i, j);
                double val_v_im1 = get_v(i - 1, j);
                double dv_dx     = (val_v_i - val_v_im1) / hx;

                tau_xy(i, j) = mu_node * (du_dy + dv_dx);
            }
        }
    }
}

void ConcatNSSolver2D::euler_conv_diff_inner_nonnewton()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& p      = *p_field_map[domain];
        field2& tau_xx = *tau_xx_field_map[domain];
        field2& tau_yy = *tau_yy_field_map[domain];
        field2& tau_xy = *tau_xy_field_map[domain];

        field2& u_temp = *u_temp_field_map[domain];
        field2& v_temp = *v_temp_field_map[domain];

        int    nx = u.get_nx();
        int    ny = u.get_ny();
        double hx = domain->hx;
        double hy = domain->hy;

        // u (interior only; boundaries handled in euler_conv_diff_outer)
        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                double conv_x =
                    u(i + 1, j) * (u(i + 1, j) + 2.0 * u(i, j)) - u(i - 1, j) * (u(i - 1, j) + 2.0 * u(i, j));
                double conv_y = (u(i, j) + u(i, j + 1)) * (v(i - 1, j + 1) + v(i, j + 1)) -
                                (u(i, j - 1) + u(i, j)) * (v(i - 1, j) + v(i, j));

                // Non-Newtonian Diffusion: div(tau)
                // For u-momentum: d(tau_xx)/dx + d(tau_xy)/dy
                // u(i, j) is at left face of cell (i, j)
                // tau_xx is at center. d(tau_xx)/dx approx (tau_xx(i, j) - tau_xx(i-1, j)) / hx
                // tau_xy is at node. d(tau_xy)/dy approx (tau_xy(i, j+1) - tau_xy(i, j)) / hy
                // Note: tau_xy(i, j) is at node (i, j) which is bottom-left of cell (i, j)
                //       tau_xy(i, j+1) is at node (i, j+1) which is top-left of cell (i, j)
                //       So this matches the location of u(i, j) which is left face center.

                double diff_x = (tau_xx(i, j) - tau_xx(i - 1, j)) / hx;
                double diff_y = (tau_xy(i, j + 1) - tau_xy(i, j)) / hy;
                double diff   = diff_x + diff_y;

                //? if 1/Re needed in non-newtonian case?
                u_temp(i, j) = u(i, j) - dt * (0.25 / hx * conv_x + 0.25 / hy * conv_y - diff);
            }
        }

        // v
        OPENMP_PARALLEL_FOR()
        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                double conv_x = (v(i, j) + v(i + 1, j)) * (u(i + 1, j - 1) + u(i + 1, j)) -
                                (v(i - 1, j) + v(i, j)) * (u(i, j - 1) + u(i, j));
                double conv_y =
                    v(i, j + 1) * (v(i, j + 1) + 2.0 * v(i, j)) - v(i, j - 1) * (v(i, j - 1) + 2.0 * v(i, j));

                // Non-Newtonian Diffusion: div(tau)
                // For v-momentum: d(tau_xy)/dx + d(tau_yy)/dy
                // v(i, j) is at bottom face of cell (i, j)
                // tau_xy is at node. d(tau_xy)/dx approx (tau_xy(i+1, j) - tau_xy(i, j)) / hx
                // tau_yy is at center. d(tau_yy)/dy approx (tau_yy(i, j) - tau_yy(i, j-1)) / hy
                // Note: tau_xy(i, j) is at node (i, j) which is bottom-left of cell (i, j)
                //       tau_xy(i+1, j) is at node (i+1, j) which is bottom-right of cell (i, j)
                //       So this matches the location of v(i, j) which is bottom face center.

                double diff_x = (tau_xy(i + 1, j) - tau_xy(i, j)) / hx;
                double diff_y = (tau_yy(i, j) - tau_yy(i, j - 1)) / hy;
                double diff   = diff_x + diff_y;

                v_temp(i, j) = v(i, j) - dt * (0.25 / hx * conv_x + 0.25 / hy * conv_y - diff);
            }
        }
    }
}

void ConcatNSSolver2D::euler_conv_diff_outer_nonnewton()
{
    for (auto& domain : domains)
    {
        field2& u      = *u_field_map[domain];
        field2& v      = *v_field_map[domain];
        field2& u_temp = *u_temp_field_map[domain];
        field2& v_temp = *v_temp_field_map[domain];
        field2& tau_xx = *tau_xx_field_map[domain];
        field2& tau_yy = *tau_yy_field_map[domain];
        field2& tau_xy = *tau_xy_field_map[domain];

        double hx = domain->hx;
        double hy = domain->hy;

        double* v_left_buffer  = v_buffer_map[domain][LocationType::Left];
        double* v_right_buffer = v_buffer_map[domain][LocationType::Right];
        double* v_down_buffer  = v_buffer_map[domain][LocationType::Down];
        double* v_up_buffer    = v_buffer_map[domain][LocationType::Up];

        double* u_left_buffer  = u_buffer_map[domain][LocationType::Left];
        double* u_right_buffer = u_buffer_map[domain][LocationType::Right];
        double* u_down_buffer  = u_buffer_map[domain][LocationType::Down];
        double* u_up_buffer    = u_buffer_map[domain][LocationType::Up];

        int nx = domain->get_nx();
        int ny = domain->get_ny();

        // TODO FIX cal tau at boundary issue
        auto bound_cal_u = [&](int i, int j) {
            double u_left  = i == 0 ? u_left_buffer[j] : u(i - 1, j);
            double u_right = i == nx - 1 ? u_right_buffer[j] : u(i + 1, j);
            double u_down  = j == 0 ? u_down_buffer[i] : u(i, j - 1);
            double u_up    = j == ny - 1 ? u_up_buffer[i] : u(i, j + 1);

            double v_left = i == 0 ? v_left_buffer[j] : v(i - 1, j);
            double v_up   = j == ny - 1 ? v_up_buffer[i] : v(i, j + 1);

            double v_left_up = i == 0 ? (j == ny - 1 ? left_up_corner_value_map[domain] : v_left_buffer[j + 1]) :
                                        (j == ny - 1 ? v_up_buffer[i - 1] : v(i - 1, j + 1));

            double u_conv_x = u_right * (u_right + 2.0 * u(i, j)) - u_left * (u_left + 2.0 * u(i, j));
            double u_conv_y = (u(i, j) + u_up) * (v_left_up + v_up) - (u_down + u(i, j)) * (v_left + v(i, j));

            // Non-Newtonian Diffusion
            double diff_x, diff_y;

            // diff_x = (tau_xx(i, j) - tau_xx(i-1, j)) / hx
            // If i=0, tau_xx(-1, j) is needed.
            double t_xx_curr = tau_xx(i, j);
            double t_xx_prev = (i == 0) ? tau_xx_buffer_map[domain][LocationType::Left][j] : tau_xx(i - 1, j);

            diff_x = (t_xx_curr - t_xx_prev) / hx;

            // diff_y = (tau_xy(i, j+1) - tau_xy(i, j)) / hy
            // tau_xy indices are valid.
            diff_y = (tau_xy(i, j + 1) - tau_xy(i, j)) / hy;

            double diff = diff_x + diff_y;

            u_temp(i, j) = u(i, j) - dt * (0.25 / hx * u_conv_x + 0.25 / hy * u_conv_y - diff);
        };

        auto bound_cal_v = [&](int i, int j) {
            double v_left  = i == 0 ? v_left_buffer[j] : v(i - 1, j);
            double v_right = i == nx - 1 ? v_right_buffer[j] : v(i + 1, j);
            double v_down  = j == 0 ? v_down_buffer[i] : v(i, j - 1);
            double v_up    = j == ny - 1 ? v_up_buffer[i] : v(i, j + 1);

            double u_right = i == nx - 1 ? u_right_buffer[j] : u(i + 1, j);
            double u_down  = j == 0 ? u_down_buffer[i] : u(i, j - 1);

            double u_right_down = j == 0 ? (i == nx - 1 ? right_down_corner_value_map[domain] : u_down_buffer[i + 1]) :
                                           (i == nx - 1 ? u_right_buffer[j - 1] : u(i + 1, j - 1));

            double v_conv_x = (v(i, j) + v_right) * (u_right_down + u_right) - (v_left + v(i, j)) * (u_down + u(i, j));
            double v_conv_y = v_up * (v_up + 2.0 * v(i, j)) - v_down * (v_down + 2.0 * v(i, j));

            // Non-Newtonian Diffusion
            double diff_x, diff_y;

            // diff_x = (tau_xy(i+1, j) - tau_xy(i, j)) / hx
            // tau_xy indices are valid.
            diff_x = (tau_xy(i + 1, j) - tau_xy(i, j)) / hx;

            // diff_y = (tau_yy(i, j) - tau_yy(i, j-1)) / hy
            // If j=0, tau_yy(i, -1) is needed.
            double t_yy_curr = tau_yy(i, j);
            double t_yy_prev = (j == 0) ? tau_yy_buffer_map[domain][LocationType::Down][i] : tau_yy(i, j - 1);
            diff_y           = (t_yy_curr - t_yy_prev) / hy;

            double diff = diff_x + diff_y;

            v_temp(i, j) = v(i, j) - dt * (0.25 / hx * v_conv_x + 0.25 / hy * v_conv_y - diff);
        };

        // Left
        for (int j = 0; j < ny; j++)
        {
            if (u_var->boundary_type_map[domain][LocationType::Left] == PDEBoundaryType::Adjacented)
                bound_cal_u(0, j);
            bound_cal_v(0, j);
        }

        // Right
        for (int j = 0; j < ny; j++)
        {
            bound_cal_u(nx - 1, j);
            bound_cal_v(nx - 1, j);
        }

        // Down
        for (int i = 0; i < nx; i++)
        {
            bound_cal_u(i, 0);
            if (v_var->boundary_type_map[domain][LocationType::Down] == PDEBoundaryType::Adjacented)
                bound_cal_v(i, 0);
        }

        // Up
        for (int i = 0; i < nx; i++)
        {
            bound_cal_u(i, ny - 1);
            bound_cal_v(i, ny - 1);
        }

        // Swap data pointers: u <-> u_temp, v <-> v_temp
        swap_field_data(u, u_temp);
        swap_field_data(v, v_temp);
    }
}