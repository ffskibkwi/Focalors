#include "mhd_module_2d.h"

#include "base/parallel/omp/enable_openmp.h"
#include "boundary_2d_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

MHDModule2D::MHDModule2D(Variable2D*          in_u_var,
                         Variable2D*          in_v_var,
                         PhysicsConfig*       in_phy_config,
                         TimeAdvancingConfig* in_time_config,
                         EnvironmentConfig*   in_env_config)
    : m_uVar(in_u_var)
    , m_vVar(in_v_var)
    , m_phyConfig(in_phy_config)
    , m_timeConfig(in_time_config)
    , m_envConfig(in_env_config)
{
    if (m_uVar == nullptr || m_vVar == nullptr)
        throw std::runtime_error("MHDModule2D: u/v variable is null");
    if (m_uVar->geometry == nullptr || m_vVar->geometry == nullptr)
        throw std::runtime_error("MHDModule2D: u/v geometry is null");
    if (m_uVar->geometry != m_vVar->geometry)
        throw std::runtime_error("MHDModule2D: u/v do not share one geometry");

    if (m_phyConfig == nullptr || m_timeConfig == nullptr)
        throw std::runtime_error("MHDModule2D: config pointer is null");

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

MHDModule2D::~MHDModule2D() = default;

void MHDModule2D::init(Variable2D* phi_var)
{
    if (m_initialized)
        return;

    Geometry2D* geo = m_uVar->geometry;

    // Determine if phi is externally provided (with boundary conditions already configured)
    bool externalPhi = (phi_var != nullptr);

    if (externalPhi)
    {
        // Use externally provided phi variable (Case has configured boundary conditions)
        m_phiVar.reset(phi_var);
    }
    else
    {
        // Create internal phi variable with default Neumann boundary conditions
        m_phiVar = std::unique_ptr<Variable2D>(new Variable2D("phi"));
        m_phiVar->set_geometry(*geo);
    }

    m_jxVar = std::unique_ptr<Variable2D>(new Variable2D("jx"));
    m_jyVar = std::unique_ptr<Variable2D>(new Variable2D("jy"));
    m_jzVar = std::unique_ptr<Variable2D>(new Variable2D("jz"));

    m_jxVar->set_geometry(*geo);
    m_jyVar->set_geometry(*geo);
    m_jzVar->set_geometry(*geo);

    for (auto& domain : m_domains)
    {
        // Allocate center fields for jx, jy, jz (always internal)
        m_jxFieldStorage[domain] =
            std::unique_ptr<field2>(new field2(domain->get_nx(), domain->get_ny(), "jx_" + domain->name));
        m_jyFieldStorage[domain] =
            std::unique_ptr<field2>(new field2(domain->get_nx(), domain->get_ny(), "jy_" + domain->name));
        m_jzFieldStorage[domain] =
            std::unique_ptr<field2>(new field2(domain->get_nx(), domain->get_ny(), "jz_" + domain->name));

        m_jxVar->set_center_field(domain, *m_jxFieldStorage[domain]);
        m_jyVar->set_center_field(domain, *m_jyFieldStorage[domain]);
        m_jzVar->set_center_field(domain, *m_jzFieldStorage[domain]);

        if (externalPhi)
        {
            // External phi: fields already exist in the passed Variable2D, just reference them
            // No need to allocate storage or set boundary conditions
        }
        else
        {
            // Internal phi: allocate storage and set default boundary conditions
            m_phiFieldStorage[domain] =
                std::unique_ptr<field2>(new field2(domain->get_nx(), domain->get_ny(), "phi_" + domain->name));
            m_phiVar->set_center_field(domain, *m_phiFieldStorage[domain]);

            // Electric insulation (recommended): physical boundaries use Neumann=0.
            // For adjacented faces, keep Adjacented type.
            for (LocationType loc : {LocationType::Left, LocationType::Right, LocationType::Down, LocationType::Up})
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

            // Zero initialize phi field
            m_phiFieldStorage[domain]->clear(0.0);
        }

        // Zero initialize jx, jy, jz fields
        m_jxFieldStorage[domain]->clear(0.0);
        m_jyFieldStorage[domain]->clear(0.0);
        m_jzFieldStorage[domain]->clear(0.0);
    }

    m_phiFieldMap = m_phiVar->field_map;
    m_jxFieldMap  = m_jxVar->field_map;
    m_jyFieldMap  = m_jyVar->field_map;
    m_jzFieldMap  = m_jzVar->field_map;

    m_phiSolver = std::unique_ptr<ConcatPoissonSolver2D>(new ConcatPoissonSolver2D(m_phiVar.get(), m_envConfig));

    // Cache constant parameters (avoid repeated config reads in each step).
    m_Bx = m_phyConfig->Bx;
    m_By = m_phyConfig->By;
    m_Bz = m_phyConfig->Bz;
    m_Ha = m_phyConfig->Ha;
    m_Re = m_phyConfig->Re;
    m_dt = m_timeConfig->dt;
    if (m_Re != 0.0)
        m_lorentzCoef = m_dt * (m_Ha * m_Ha) / m_Re;
    else
        m_lorentzCoef = 0.0;

    m_initialized = true;
}

void MHDModule2D::solveElectricPotential()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::solveElectricPotential(): module not initialized");

    // U=(u,v,0) B=(Bx,By,Bz)
    // U×B=(vB_z,-uB_z,uB_y-vB_x)
    // In 2D simulations, all variables (including velocity u and v) are assumed to be uniform in the z direction. Then,
    // ∂z∂(⋅)​= 0 For constant external Bz: div(u×B) = Bz * (dv/dx - du/dy)
    if (m_Bz == 0.0)
    {
        // RHS is identically 0 in this 2D formulation; choose gauge phi=0 and skip solve.
        for (auto& domain : m_domains)
        {
            m_phiFieldMap[domain]->clear(0.0);
        }
        buffer_update();
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

        // Conservative discretization on MAC grid (cell-centered vorticity approximation)
        // dv/dx at center (i,j): (v(i,j)-v(i-1,j))/hx
        // du/dy at center (i,j): (u(i,j)-u(i,j-1))/hy
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double v_im1 = (i == 0) ? v(i, j) : v(i - 1, j);
                double u_jm1 = (j == 0) ? u(i, j) : u(i, j - 1);

                // Compute RHS = div(u×B) = Bz*(dv/dx - du/dy) * hx * hy (scaled by cell area for Poisson solver)
                rhs(i, j) = m_Bz * ((v(i, j) - v_im1) * hy - (u(i, j) - u_jm1) * hx);
            }
        }
    }

    // For pure Neumann/Periodic Poisson problems, RHS must satisfy the global solvability condition.
    if (isAllNeumannBoundary(*m_phiVar))
    {
        normalizeRhsForNeumannBc(*m_phiVar, m_domains, m_phiFieldMap);
    }

    // Solve Poisson equation; phi overwrites RHS in phi_var->field_map
    m_phiSolver->solve();
    buffer_update();
}

void MHDModule2D::updateCurrentDensity()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::updateCurrentDensity(): module not initialized");

    // `phi` is solved at centers; make sure its ghost/buffer is coherent before taking gradients.
    buffer_update();

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

        double* phi_left_buffer = m_phiVar->buffer_map[domain][LocationType::Left];
        double* phi_down_buffer = m_phiVar->buffer_map[domain][LocationType::Down];

        auto& type_map = m_phiVar->boundary_type_map[domain];
        auto& has_map  = m_phiVar->has_boundary_value_map[domain];
        auto& val_map  = m_phiVar->boundary_value_map[domain];

        const PDEBoundaryType right_type = type_map[LocationType::Right];
        const PDEBoundaryType up_type    = type_map[LocationType::Up];

        double* right_val_ptr = (has_map[LocationType::Right] ? val_map[LocationType::Right] : nullptr);
        double* up_val_ptr    = (has_map[LocationType::Up] ? val_map[LocationType::Up] : nullptr);

        auto get_phi = [&](int i_idx, int j_idx) -> double {
            // Adjacented faces on Right/Up: query neighbour center value directly.
            if (i_idx >= nx && right_type == PDEBoundaryType::Adjacented)
            {
                auto it_dom = m_adjacency.find(domain);
                if (it_dom != m_adjacency.end() && it_dom->second.count(LocationType::Right))
                {
                    Domain2DUniform* adj_domain = it_dom->second[LocationType::Right];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    return adj_phi(0, j_idx);
                }
            }
            if (j_idx >= ny && up_type == PDEBoundaryType::Adjacented)
            {
                auto it_dom = m_adjacency.find(domain);
                if (it_dom != m_adjacency.end() && it_dom->second.count(LocationType::Up))
                {
                    Domain2DUniform* adj_domain = it_dom->second[LocationType::Up];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    return adj_phi(i_idx, 0);
                }
            }

            return get_scalar_with_boundary(i_idx,
                                            j_idx,
                                            nx,
                                            ny,
                                            phi,
                                            phi_left_buffer,
                                            phi_down_buffer,
                                            hx,
                                            hy,
                                            right_type,
                                            right_val_ptr,
                                            0.0,
                                            up_type,
                                            up_val_ptr,
                                            0.0);
        };

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Center velocity from edges
                double u_c = 0.5 * (u(i, j) + (i + 1 < nx ? u(i + 1, j) : u(i, j)));
                double v_c = 0.5 * (v(i, j) + (j + 1 < ny ? v(i, j + 1) : v(i, j)));

                // grad(phi) at center using ghost-aware central differences
                double dphi_dx = (get_phi(i + 1, j) - get_phi(i - 1, j)) / (2.0 * hx);
                double dphi_dy = (get_phi(i, j + 1) - get_phi(i, j - 1)) / (2.0 * hy);

                // u×B at center: (v*Bz - w*By, w*Bx - u*Bz, u*By - v*Bx)
                // 2D: w=0
                double uxB_x = v_c * m_Bz;
                double uxB_y = -u_c * m_Bz;
                double uxB_z = u_c * m_By - v_c * m_Bx;

                jx(i, j) = -dphi_dx + uxB_x;
                jy(i, j) = -dphi_dy + uxB_y;
                jz(i, j) = uxB_z;
            }
        }
    }
}

void MHDModule2D::applyLorentzForce()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::applyLorentzForce(): module not initialized");

    if (m_Re == 0.0)
        throw std::runtime_error("MHDModule2D::applyLorentzForce(): Re is zero");

    if (m_lorentzCoef == 0.0)
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

        // Center force F = J×B
        field2 fx_center(nx, ny, "fx_center");
        field2 fy_center(nx, ny, "fy_center");

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double Fx = jy(i, j) * m_Bz - jz(i, j) * m_By;
                double Fy = jz(i, j) * m_Bx - jx(i, j) * m_Bz;

                fx_center(i, j) = Fx;
                fy_center(i, j) = Fy;
            }
        }

        // Interpolate center force to edges and add to predicted velocity.
        // NOTE: At this point in NS predictor step, u/v have been swapped to u_temp/v_temp already.
        // We directly add to current u/v fields.

        // u on XFace: use Fx at adjacent centers (i-1, j) and (i, j)
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double Fx_l = (i - 1 >= 0) ? fx_center(i - 1, j) : fx_center(i, j);
                double Fx_r = fx_center(i, j);
                double Fx_u = 0.5 * (Fx_l + Fx_r);
                u(i, j) += m_lorentzCoef * Fx_u;
            }
        }

        // v on YFace: use Fy at adjacent centers (i, j-1) and (i, j)
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double Fy_d = (j - 1 >= 0) ? fy_center(i, j - 1) : fy_center(i, j);
                double Fy_u = fy_center(i, j);
                double Fy_v = 0.5 * (Fy_d + Fy_u);
                v(i, j) += m_lorentzCoef * Fy_v;
            }
        }
    }
}

void MHDModule2D::buffer_update()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::buffer_update(): module not initialized");

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

        // Center variable: buffer ownership is Left/Down only.
        // We still compute physical-boundary ghost values for these owned buffers.

        // Left buffer: phi(-1, j)
        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::Left))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::Left];
            PDEBoundaryType type = type_map[LocationType::Left];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Left))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Left];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    const int        adj_nx     = adj_phi.get_nx();
                    copy_x_to_buffer(buf, adj_phi, adj_nx - 1);
                    continue;
                }
            }

            if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::Left] ? val_map[LocationType::Left] : nullptr);
                mirror_x_to_buffer(buf, phi, 0, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::Left] ? val_map[LocationType::Left] : nullptr);
                neumann_x_to_buffer(buf, phi, 0, q_ptr, 0.0, hx, -1.0);
            }
            else
            {
                // Default fallback: zero-gradient.
                copy_x_to_buffer(buf, phi, 0);
            }
        }

        // Down buffer: phi(i, -1)
        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::Down))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::Down];
            PDEBoundaryType type = type_map[LocationType::Down];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Down))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Down];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    const int        adj_ny     = adj_phi.get_ny();
                    copy_y_to_buffer(buf, adj_phi, adj_ny - 1);
                    continue;
                }
            }

            if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::Down] ? val_map[LocationType::Down] : nullptr);
                mirror_y_to_buffer(buf, phi, 0, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::Down] ? val_map[LocationType::Down] : nullptr);
                neumann_y_to_buffer(buf, phi, 0, q_ptr, 0.0, hy, -1.0);
            }
            else
            {
                copy_y_to_buffer(buf, phi, 0);
            }
        }
    }
}
