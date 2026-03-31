#include "physical_pe_solver2d.h"

#include "boundary_2d_utils.h"

PhysicalPESolver2D::PhysicalPESolver2D(Variable2D*            in_u_var,
                                       Variable2D*            in_v_var,
                                       Variable2D*            in_p_var,
                                       ConcatPoissonSolver2D* in_p_solver,
                                       double                 in_rho)
    : u_var(in_u_var)
    , v_var(in_v_var)
    , p_var(in_p_var)
    , p_solver(in_p_solver)
    , rho(in_rho)
{
    if (u_var->geometry != v_var->geometry || u_var->geometry != p_var->geometry)
        throw std::runtime_error("PhysicalPESolver2D: u v p do not share one geometry");
    if (u_var->geometry == nullptr)
        throw std::runtime_error("PhysicalPESolver2D: u->geometry is null");

    if (!u_var->geometry->is_checked)
        u_var->geometry->check();
    if (u_var->geometry->tree_root == nullptr || u_var->geometry->tree_map.empty())
        u_var->geometry->solve_prepare();

    domains   = u_var->geometry->domains;
    adjacency = u_var->geometry->adjacency;

    for (auto* domain : domains)
    {
        auto* u_field = u_var->field_map[domain];
        auto* v_field = v_var->field_map[domain];

        u_xpos_ypos_corner_map[domain] = 0.0;
        v_xpos_ypos_corner_map[domain] = 0.0;

        dudx_map[domain] = new field2(u_field->get_nx(), u_field->get_ny(), "dudx_" + domain->name);
        dudy_map[domain] = new field2(u_field->get_nx(), u_field->get_ny(), "dudy_" + domain->name);
        dvdx_map[domain] = new field2(v_field->get_nx(), v_field->get_ny(), "dvdx_" + domain->name);
        dvdy_map[domain] = new field2(v_field->get_nx(), v_field->get_ny(), "dvdy_" + domain->name);
    }
}

PhysicalPESolver2D::~PhysicalPESolver2D()
{
    for (auto& [domain, field] : dudx_map)
        delete field;
    for (auto& [domain, field] : dudy_map)
        delete field;
    for (auto& [domain, field] : dvdx_map)
        delete field;
    for (auto& [domain, field] : dvdy_map)
        delete field;
}

void PhysicalPESolver2D::solve()
{
    phys_boundary_update();
    // Unlike the NS pressure-correction path, PPE assembles rhs directly from
    // shared-face buffers and corner ghosts, so the full phys/nondiag/diag
    // chain must finish before calc_rhs().
    nondiag_shared_boundary_update();
    diag_shared_boundary_update();
    calc_rhs();

    if (isAllNeumannBoundary(*p_var))
        normalizeRhsForNeumannBc(*p_var, domains, p_var->field_map);

    p_solver->solve();
}

void PhysicalPESolver2D::calc_rhs()
{
    for (auto* domain : domains)
    {
        field2& u = *u_var->field_map[domain];
        field2& v = *v_var->field_map[domain];
        field2& p = *p_var->field_map[domain];

        double* u_xpos_buffer = u_var->buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer = u_var->buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer = u_var->buffer_map[domain][LocationType::YPositive];

        double* v_xneg_buffer = v_var->buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer = v_var->buffer_map[domain][LocationType::XPositive];
        double* v_ypos_buffer = v_var->buffer_map[domain][LocationType::YPositive];

        double u_xpos_yneg_corner = u_var->xpos_yneg_corner_map[domain];
        double u_xpos_ypos_corner = u_xpos_ypos_corner_map[domain];
        double v_xneg_ypos_corner = v_var->xneg_ypos_corner_map[domain];
        double v_xpos_ypos_corner = v_xpos_ypos_corner_map[domain];

        field2& dudx = *dudx_map[domain];
        field2& dudy = *dudy_map[domain];
        field2& dvdx = *dvdx_map[domain];
        field2& dvdy = *dvdy_map[domain];

        const int    nx = u.get_nx();
        const int    ny = u.get_ny();
        const double hx = domain->hx;
        const double hy = domain->hy;

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                const double u_ij  = u(i, j);
                const double u_ip1 = i == nx - 1 ? u_xpos_buffer[j] : u(i + 1, j);
                const double u_jm1 = j == 0 ? u_yneg_buffer[i] : u(i, j - 1);
                const double u_jp1 = j == ny - 1 ? u_ypos_buffer[i] : u(i, j + 1);

                const double u_ip1_jm1 = i == nx - 1 ? (j == 0 ? u_xpos_yneg_corner : u_xpos_buffer[j - 1]) :
                                                       (j == 0 ? u_yneg_buffer[i + 1] : u(i + 1, j - 1));
                const double u_ip1_jp1 = i == nx - 1 ? (j == ny - 1 ? u_xpos_ypos_corner : u_xpos_buffer[j + 1]) :
                                                       (j == ny - 1 ? u_ypos_buffer[i + 1] : u(i + 1, j + 1));

                dudx(i, j) = (u_ip1 - u_ij) / hx;
                dudy(i, j) = (u_jp1 + u_ip1_jp1 - u_jm1 - u_ip1_jm1) / (4.0 * hy);

                const double v_ij  = v(i, j);
                const double v_im1 = i == 0 ? v_xneg_buffer[j] : v(i - 1, j);
                const double v_ip1 = i == nx - 1 ? v_xpos_buffer[j] : v(i + 1, j);
                const double v_jp1 = j == ny - 1 ? v_ypos_buffer[i] : v(i, j + 1);

                const double v_im1_jp1 = i == 0 ? (j == ny - 1 ? v_xneg_ypos_corner : v_xneg_buffer[j + 1]) :
                                                  (j == ny - 1 ? v_ypos_buffer[i - 1] : v(i - 1, j + 1));
                const double v_ip1_jp1 = i == nx - 1 ? (j == ny - 1 ? v_xpos_ypos_corner : v_xpos_buffer[j + 1]) :
                                                       (j == ny - 1 ? v_ypos_buffer[i + 1] : v(i + 1, j + 1));

                dvdx(i, j) = (v_ip1 + v_ip1_jp1 - v_im1 - v_im1_jp1) / (4.0 * hx);
                dvdy(i, j) = (v_jp1 - v_ij) / hy;

                p(i, j) = -rho * (dudx(i, j) * dudx(i, j) + 2.0 * dudy(i, j) * dvdx(i, j) + dvdy(i, j) * dvdy(i, j));
            }
        }
    }
}
