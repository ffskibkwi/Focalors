#include "mhd_module_2d_yee.h"

#include "base/parallel/omp/enable_openmp.h"
#include "boundary_2d_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

MHDModule2DYee::MHDModule2DYee(Variable2D* in_u_var, Variable2D* in_v_var)
    : m_uVar(in_u_var)
    , m_vVar(in_v_var)
{
    if (m_uVar == nullptr || m_vVar == nullptr)
        throw std::runtime_error("MHDModule2D: u/v variable2d is null");
    if (m_uVar->geometry == nullptr || m_vVar->geometry == nullptr)
        throw std::runtime_error("MHDModule2D: u/v geometry is null");
    if (m_uVar->geometry != m_vVar->geometry)
        throw std::runtime_error("MHDModule2D: u/v do not share one geometry");

    Geometry2D* geo = m_uVar->geometry;
    if (!geo->is_checked)
        geo->check();
    if (geo->tree_root == nullptr || geo->tree_map.empty())
        geo->solve_prepare();

    m_domains   = geo->domains;
    m_adjacency = geo->adjacency;

    m_uFieldMap = m_uVar->field_map;
    m_vFieldMap = m_vVar->field_map;
}

MHDModule2DYee::~MHDModule2DYee() = default;

void MHDModule2DYee::init(Variable2D* phi_var)
{
    if (m_initialized)
        return;

    Geometry2D* geo = m_uVar->geometry;

    bool externalPhi = (phi_var != nullptr);

    if (externalPhi)
    {
        // Use externally provided phi variable2d (non-owning)
        m_phiVar = phi_var;
    }
    else
    {
        // Create internal phi variable2d
        m_phiOwned = std::unique_ptr<Variable2D>(new Variable2D("phi"));
        m_phiOwned->set_geometry(*geo);
        m_phiVar = m_phiOwned.get();
    }

    m_jxVar = std::unique_ptr<Variable2D>(new Variable2D("jx"));
    m_jyVar = std::unique_ptr<Variable2D>(new Variable2D("jy"));
    m_jzVar = std::unique_ptr<Variable2D>(new Variable2D("jz"));

    m_jxVar->set_geometry(*geo);
    m_jyVar->set_geometry(*geo);
    m_jzVar->set_geometry(*geo);

    for (auto& domain : m_domains)
    {
        const int nx = domain->get_nx();
        const int ny = domain->get_ny();

        m_jxFieldStorage[domain] = std::unique_ptr<field2>(new field2(nx, ny, "jx_" + domain->name));
        m_jyFieldStorage[domain] = std::unique_ptr<field2>(new field2(nx, ny, "jy_" + domain->name));
        m_jzFieldStorage[domain] = std::unique_ptr<field2>(new field2(nx, ny, "jz_" + domain->name));

        m_jxVar->set_y_edge_field(domain, *m_jxFieldStorage[domain]);
        m_jyVar->set_x_edge_field(domain, *m_jyFieldStorage[domain]);
        m_jzVar->set_center_field(domain, *m_jzFieldStorage[domain]);

        if (externalPhi)
        {
            // External phi: fields already exist in the passed Variable2D, just reference them
        }
        else
        {
            m_phiFieldStorage[domain] = std::unique_ptr<field2>(new field2(nx, ny, "phi_" + domain->name));
            m_phiVar->set_inner_field(domain, *m_phiFieldStorage[domain]);

            for (LocationType loc :
                 {LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive})
            {
                if (m_adjacency.count(domain) && m_adjacency[domain].count(loc))
                {
                    m_phiVar->set_boundary_type(domain, loc, PDEBoundaryType::Adjacented);
                }
                else
                {
                    m_phiVar->set_boundary_type(domain, loc, PDEBoundaryType::Neumann);
                }
            }

            m_phiFieldStorage[domain]->clear(0.0);
        }

        // Ensure phi owns 4-side buffers for Yee node-based stencil usage.
        // set_inner_field() already allocates all four; for external phi we repair missing sides.
        auto& phi_buffer_map = m_phiVar->buffer_map[domain];
        if (!phi_buffer_map.count(LocationType::XNegative))
            phi_buffer_map[LocationType::XNegative] = new double[ny];
        if (!phi_buffer_map.count(LocationType::XPositive))
            phi_buffer_map[LocationType::XPositive] = new double[ny];
        if (!phi_buffer_map.count(LocationType::YNegative))
            phi_buffer_map[LocationType::YNegative] = new double[nx];
        if (!phi_buffer_map.count(LocationType::YPositive))
            phi_buffer_map[LocationType::YPositive] = new double[nx];

        m_jxFieldStorage[domain]->clear(0.0);
        m_jyFieldStorage[domain]->clear(0.0);
        m_jzFieldStorage[domain]->clear(0.0);
    }

    m_phiFieldMap = m_phiVar->field_map;
    m_jxFieldMap  = m_jxVar->field_map;
    m_jyFieldMap  = m_jyVar->field_map;
    m_jzFieldMap  = m_jzVar->field_map;

    m_phiSolver = std::unique_ptr<ConcatPoissonSolver2D>(new ConcatPoissonSolver2D(m_phiVar));

    PhysicsConfig&       phyConfig  = PhysicsConfig::Get();
    TimeAdvancingConfig& timeConfig = TimeAdvancingConfig::Get();

    m_Bx = phyConfig.Bx;
    m_By = phyConfig.By;
    m_Bz = phyConfig.Bz;
    m_Ha = phyConfig.Ha;
    m_Re = phyConfig.Re;
    m_dt = timeConfig.dt;
    if (m_Re != 0.0)
        m_lorentzCoef = m_dt * (m_Ha * m_Ha) / m_Re;
    else
        m_lorentzCoef = 0.0;

    m_initialized = true;
}

void MHDModule2DYee::solveElectricPotential()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2DYee::solveElectricPotential(): module not initialized");

    if (m_Bz == 0.0)
    {
        for (auto& domain : m_domains)
        {
            m_phiFieldMap[domain]->clear(0.0);
        }
        buffer_update_phi();
        return;
    }

    for (auto& domain : m_domains)
    {
        field2& u   = *m_uFieldMap[domain];
        field2& v   = *m_vFieldMap[domain];
        field2& rhs = *m_phiFieldMap[domain];

        const int    nx = domain->get_nx();
        const int    ny = domain->get_ny();
        const double hx = domain->hx;
        const double hy = domain->hy;

        double* u_xneg_buffer          = m_uVar->buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer          = m_uVar->buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer          = m_uVar->buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer          = m_uVar->buffer_map[domain][LocationType::YPositive];
        double* v_xneg_buffer          = m_vVar->buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer          = m_vVar->buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer          = m_vVar->buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer          = m_vVar->buffer_map[domain][LocationType::YPositive];
        double  xpos_yneg_corner_value = m_uVar->xpos_yneg_corner_value_map[domain];
        double  xneg_ypos_corner_value = m_vVar->xneg_ypos_corner_value_map[domain];

        auto get_u = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       u,
                                       u_xneg_buffer,
                                       u_xpos_buffer,
                                       u_yneg_buffer,
                                       u_ypos_buffer,
                                       xpos_yneg_corner_value);
        };
        auto get_v = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       v,
                                       v_xneg_buffer,
                                       v_xpos_buffer,
                                       v_yneg_buffer,
                                       v_ypos_buffer,
                                       xneg_ypos_corner_value);
        };

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Yee node formulation:
                // RHS = div(u x B) = Bz * (dv/dx - du/dy), evaluated at phi nodes.
                const double dv_dx = (get_v(i, j) - get_v(i - 1, j)) / hx;
                const double du_dy = (get_u(i, j) - get_u(i, j - 1)) / hy;

                rhs(i, j) = m_Bz * (dv_dx - du_dy) * hx * hy;
            }
        }
    }

    if (isAllNeumannBoundary(*m_phiVar))
    {
        normalizeRhsForNeumannBc(*m_phiVar, m_domains, m_phiFieldMap);
    }

    m_phiSolver->solve();
    buffer_update_phi();
}

void MHDModule2DYee::updateCurrentDensity()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2DYee::updateCurrentDensity(): module not initialized");

    if (m_Bz == 0.0)
    {
        for (auto& domain : m_domains)
        {
            m_jxFieldMap[domain]->clear(0.0);
            m_jyFieldMap[domain]->clear(0.0);
            m_jzFieldMap[domain]->clear(0.0);
        }
        buffer_update_j();
        return;
    }

    for (auto& domain : m_domains)
    {
        field2& u   = *m_uFieldMap[domain];
        field2& v   = *m_vFieldMap[domain];
        field2& phi = *m_phiFieldMap[domain];

        field2& jx = *m_jxFieldMap[domain];
        field2& jy = *m_jyFieldMap[domain];
        field2& jz = *m_jzFieldMap[domain];

        const int    nx = domain->get_nx();
        const int    ny = domain->get_ny();
        const double hx = domain->hx;
        const double hy = domain->hy;

        double* phi_xneg_buffer = m_phiVar->buffer_map[domain][LocationType::XNegative];
        double* phi_xpos_buffer = m_phiVar->buffer_map[domain][LocationType::XPositive];
        double* phi_yneg_buffer = m_phiVar->buffer_map[domain][LocationType::YNegative];
        double* phi_ypos_buffer = m_phiVar->buffer_map[domain][LocationType::YPositive];

        double* u_xneg_buffer          = m_uVar->buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer          = m_uVar->buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer          = m_uVar->buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer          = m_uVar->buffer_map[domain][LocationType::YPositive];
        double* v_xneg_buffer          = m_vVar->buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer          = m_vVar->buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer          = m_vVar->buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer          = m_vVar->buffer_map[domain][LocationType::YPositive];
        double  xpos_yneg_corner_value = m_uVar->xpos_yneg_corner_value_map[domain];
        double  xneg_ypos_corner_value = m_vVar->xneg_ypos_corner_value_map[domain];

        auto get_phi = [&](int i_idx, int j_idx) -> double {
            if (i_idx >= 0 && i_idx < nx && j_idx >= 0 && j_idx < ny)
                return phi(i_idx, j_idx);
            if (i_idx < 0)
                return phi_xneg_buffer[j_idx];
            if (i_idx >= nx)
                return phi_xpos_buffer[j_idx];
            if (j_idx < 0)
                return phi_yneg_buffer[i_idx];
            return phi_ypos_buffer[i_idx];
        };
        auto get_u = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       u,
                                       u_xneg_buffer,
                                       u_xpos_buffer,
                                       u_yneg_buffer,
                                       u_ypos_buffer,
                                       xpos_yneg_corner_value);
        };
        auto get_v = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       v,
                                       v_xneg_buffer,
                                       v_xpos_buffer,
                                       v_yneg_buffer,
                                       v_ypos_buffer,
                                       xneg_ypos_corner_value);
        };

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Jx at v-position (vertical face center):
                // Jx = -dphi/dx + v*Bz
                const double dphi_dx = (get_phi(i + 1, j) - get_phi(i, j)) / hx;
                jx(i, j)             = -dphi_dx + get_v(i, j) * m_Bz;

                // Jy at u-position (horizontal face center):
                // Jy = -dphi/dy - u*Bz
                const double dphi_dy = (get_phi(i, j + 1) - get_phi(i, j)) / hy;
                jy(i, j)             = -dphi_dy - get_u(i, j) * m_Bz;

                // Jz at center from face velocities around center.
                const double u_c = 0.5 * (get_u(i, j) + get_u(i + 1, j));
                const double v_c = 0.5 * (get_v(i, j) + get_v(i, j + 1));
                jz(i, j)         = u_c * m_By - v_c * m_Bx;
            }
        }
    }

    buffer_update_j();
}

void MHDModule2DYee::applyLorentzForce()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2DYee::applyLorentzForce(): module not initialized");

    if (m_Re == 0.0)
        throw std::runtime_error("MHDModule2DYee::applyLorentzForce(): Re is zero");

    if (m_lorentzCoef == 0.0 || m_Bz == 0.0)
        return;

    for (auto& domain : m_domains)
    {
        field2& u = *m_uFieldMap[domain];
        field2& v = *m_vFieldMap[domain];

        field2& jx = *m_jxFieldMap[domain];
        field2& jy = *m_jyFieldMap[domain];
        field2& jz = *m_jzFieldMap[domain];

        const int nx = domain->get_nx();
        const int ny = domain->get_ny();

        double* jz_xneg_buffer = m_jzVar->buffer_map[domain][LocationType::XNegative];
        double* jz_yneg_buffer = m_jzVar->buffer_map[domain][LocationType::YNegative];

        auto get_jz = [&](int i_idx, int j_idx) -> double {
            if (i_idx < 0)
            {
                int jj = (j_idx < 0) ? 0 : (j_idx >= ny ? ny - 1 : j_idx);
                return jz_xneg_buffer[jj];
            }
            if (j_idx < 0)
            {
                int ii = (i_idx < 0) ? 0 : (i_idx >= nx ? nx - 1 : i_idx);
                return jz_yneg_buffer[ii];
            }
            int ii = (i_idx < 0) ? 0 : (i_idx >= nx ? nx - 1 : i_idx);
            int jj = (j_idx < 0) ? 0 : (j_idx >= ny ? ny - 1 : j_idx);
            return jz(ii, jj);
        };

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                const double jz_on_u = 0.5 * (get_jz(i - 1, j) + get_jz(i, j));
                const double Fx_u    = jy(i, j) * m_Bz - jz_on_u * m_By;
                u(i, j) += m_lorentzCoef * Fx_u;
            }
        }

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                const double jz_on_v = 0.5 * (get_jz(i, j - 1) + get_jz(i, j));
                const double Fy_v    = jz_on_v * m_Bx - jx(i, j) * m_Bz;
                v(i, j) += m_lorentzCoef * Fy_v;
            }
        }
    }
}

void MHDModule2DYee::buffer_update_phi()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2DYee::buffer_update_phi(): module not initialized");

    for (auto& domain : m_domains)
    {
        field2& phi = *m_phiFieldMap[domain];

        const int    nx = domain->get_nx();
        const int    ny = domain->get_ny();
        const double hx = domain->hx;
        const double hy = domain->hy;

        auto& type_map = m_phiVar->boundary_type_map[domain];
        auto& has_map  = m_phiVar->has_boundary_value_map[domain];
        auto& val_map  = m_phiVar->boundary_value_map[domain];

        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::XNegative))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::XNegative];
            PDEBoundaryType type = type_map[LocationType::XNegative];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::XNegative))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::XNegative];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    const int        adj_nx     = adj_phi.get_nx();
                    copy_x_to_buffer(buf, adj_phi, adj_nx - 1);
                    continue;
                }
            }

            if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::XNegative] ? val_map[LocationType::XNegative] : nullptr);
                mirror_x_to_buffer(buf, phi, 0, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::XNegative] ? val_map[LocationType::XNegative] : nullptr);
                neumann_x_to_buffer(buf, phi, 0, q_ptr, 0.0, hx, -1.0);
            }
            else
            {
                copy_x_to_buffer(buf, phi, 0);
            }
        }

        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::XPositive))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::XPositive];
            PDEBoundaryType type = type_map[LocationType::XPositive];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::XPositive))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::XPositive];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    copy_x_to_buffer(buf, adj_phi, 0);
                }
            }
            else if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::XPositive] ? val_map[LocationType::XPositive] : nullptr);
                mirror_x_to_buffer(buf, phi, nx - 1, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::XPositive] ? val_map[LocationType::XPositive] : nullptr);
                neumann_x_to_buffer(buf, phi, nx - 1, q_ptr, 0.0, hx, +1.0);
            }
            else
            {
                copy_x_to_buffer(buf, phi, nx - 1);
            }
        }

        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::YNegative))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::YNegative];
            PDEBoundaryType type = type_map[LocationType::YNegative];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::YNegative))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::YNegative];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    const int        adj_ny     = adj_phi.get_ny();
                    copy_y_to_buffer(buf, adj_phi, adj_ny - 1);
                    continue;
                }
            }

            if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::YNegative] ? val_map[LocationType::YNegative] : nullptr);
                mirror_y_to_buffer(buf, phi, 0, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::YNegative] ? val_map[LocationType::YNegative] : nullptr);
                neumann_y_to_buffer(buf, phi, 0, q_ptr, 0.0, hy, -1.0);
            }
            else
            {
                copy_y_to_buffer(buf, phi, 0);
            }
        }

        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::YPositive))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::YPositive];
            PDEBoundaryType type = type_map[LocationType::YPositive];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::YPositive))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::YPositive];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    copy_y_to_buffer(buf, adj_phi, 0);
                }
            }
            else if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::YPositive] ? val_map[LocationType::YPositive] : nullptr);
                mirror_y_to_buffer(buf, phi, ny - 1, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::YPositive] ? val_map[LocationType::YPositive] : nullptr);
                neumann_y_to_buffer(buf, phi, ny - 1, q_ptr, 0.0, hy, +1.0);
            }
            else
            {
                copy_y_to_buffer(buf, phi, ny - 1);
            }
        }
    }
}

void MHDModule2DYee::buffer_update_j()
{
    for (auto& domain : m_domains)
    {
        field2& jx = *m_jxFieldMap[domain];
        field2& jy = *m_jyFieldMap[domain];

        const int nx = domain->get_nx();
        const int ny = domain->get_ny();

        auto& v_type_map = m_vVar->boundary_type_map[domain];
        auto& u_type_map = m_uVar->boundary_type_map[domain];

        auto fill_buffer = [&](Variable2D* var, field2& f, LocationType loc, bool use_neumann) {
            if (!var->buffer_map.count(domain) || !var->buffer_map[domain].count(loc))
                return;
            double* buf = var->buffer_map[domain][loc];

            if (use_neumann)
            {
                if (loc == LocationType::XNegative || loc == LocationType::XPositive)
                {
                    assign_val_to_buffer(buf, ny, nullptr, 0.0);
                }
                else
                {
                    assign_val_to_buffer(buf, nx, nullptr, 0.0);
                }
                return;
            }

            auto adj_it = m_adjacency.find(domain);
            if (adj_it != m_adjacency.end() && adj_it->second.count(loc))
            {
                Domain2DUniform* adj_domain = adj_it->second[loc];
                field2&          adj_f      = *var->field_map[adj_domain];
                if (loc == LocationType::XNegative)
                {
                    const int adj_nx = adj_f.get_nx();
                    copy_x_to_buffer(buf, adj_f, adj_nx - 1);
                }
                else if (loc == LocationType::XPositive)
                {
                    copy_x_to_buffer(buf, adj_f, 0);
                }
                else if (loc == LocationType::YNegative)
                {
                    const int adj_ny = adj_f.get_ny();
                    copy_y_to_buffer(buf, adj_f, adj_ny - 1);
                }
                else
                {
                    copy_y_to_buffer(buf, adj_f, 0);
                }
                return;
            }

            if (loc == LocationType::XNegative)
                copy_x_to_buffer(buf, f, 0);
            else if (loc == LocationType::XPositive)
                copy_x_to_buffer(buf, f, nx - 1);
            else if (loc == LocationType::YNegative)
                copy_y_to_buffer(buf, f, 0);
            else
                copy_y_to_buffer(buf, f, ny - 1);
        };

        const bool v_xneg_neumann = v_type_map[LocationType::XNegative] == PDEBoundaryType::Neumann;
        const bool v_xpos_neumann = v_type_map[LocationType::XPositive] == PDEBoundaryType::Neumann;
        const bool v_yneg_neumann = v_type_map[LocationType::YNegative] == PDEBoundaryType::Neumann;
        const bool v_ypos_neumann = v_type_map[LocationType::YPositive] == PDEBoundaryType::Neumann;

        const bool u_xneg_neumann = u_type_map[LocationType::XNegative] == PDEBoundaryType::Neumann;
        const bool u_xpos_neumann = u_type_map[LocationType::XPositive] == PDEBoundaryType::Neumann;
        const bool u_yneg_neumann = u_type_map[LocationType::YNegative] == PDEBoundaryType::Neumann;
        const bool u_ypos_neumann = u_type_map[LocationType::YPositive] == PDEBoundaryType::Neumann;

        fill_buffer(m_jxVar.get(), jx, LocationType::XNegative, v_xneg_neumann);
        fill_buffer(m_jxVar.get(), jx, LocationType::XPositive, v_xpos_neumann);
        fill_buffer(m_jxVar.get(), jx, LocationType::YNegative, v_yneg_neumann);
        fill_buffer(m_jxVar.get(), jx, LocationType::YPositive, v_ypos_neumann);

        fill_buffer(m_jyVar.get(), jy, LocationType::XNegative, u_xneg_neumann);
        fill_buffer(m_jyVar.get(), jy, LocationType::XPositive, u_xpos_neumann);
        fill_buffer(m_jyVar.get(), jy, LocationType::YNegative, u_yneg_neumann);
        fill_buffer(m_jyVar.get(), jy, LocationType::YPositive, u_ypos_neumann);
    }
}
