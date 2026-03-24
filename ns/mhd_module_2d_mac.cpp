#include "mhd_module_2d_mac.h"

#include "base/parallel/omp/enable_openmp.h"
#include "boundary_2d_utils.h"
#include <cmath>
#include <stdexcept>

MHDModule2D::MHDModule2D(Variable2D* in_u_var, Variable2D* in_v_var)
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
        // Use externally provided phi variable2d (Case has configured boundary conditions)
        m_phiVar = phi_var;
    }
    else
    {
        // Create internal phi variable2d with default Neumann boundary conditions
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
        // Allocate fields for jx/jy on edges and jz at center (always internal)
        m_jxFieldStorage[domain] =
            std::unique_ptr<field2>(new field2(domain->get_nx(), domain->get_ny(), "jx_" + domain->name));
        m_jyFieldStorage[domain] =
            std::unique_ptr<field2>(new field2(domain->get_nx(), domain->get_ny(), "jy_" + domain->name));
        m_jzFieldStorage[domain] =
            std::unique_ptr<field2>(new field2(domain->get_nx(), domain->get_ny(), "jz_" + domain->name));

        m_jxVar->set_x_edge_field(domain, *m_jxFieldStorage[domain]);
        m_jyVar->set_y_edge_field(domain, *m_jyFieldStorage[domain]);
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

            // Electric insulation (recommended): physical boundaries use Neumann.
            // For adjacented faces, keep Adjacented type.
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

    m_phiSolver = std::unique_ptr<ConcatPoissonSolver2D>(new ConcatPoissonSolver2D(m_phiVar));

    // Cache constant parameters (avoid repeated config reads in each step).
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

void MHDModule2D::setTimeStep(double in_dt)
{
    m_dt = in_dt;
    if (m_Re != 0.0)
        m_lorentzCoef = m_dt * (m_Ha * m_Ha) / m_Re;
    else
        m_lorentzCoef = 0.0;
}

void MHDModule2D::solveElectricPotential()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::solveElectricPotential(): module not initialized");

    // U=(u,v,w) B=(Bx,By,Bz)
    // 1. General: UxB = (vBz-wBy, wBx-uBz, uBy-vBx)
    //    div(UxB) = ∂x(vBz-wBy) + ∂y(wBx-uBz) + ∂z(uBy-vBx)
    //
    // 2. 2D Simplification (w=0, ∂/∂z=0, constant Bz):
    //    div(UxB) = ∂x(vBz) + ∂y(-uBz) + 0
    //             = Bz * (∂v/∂x - ∂u/∂y)
    if (m_Bz == 0.0)
    {
        // RHS is identically 0 in this 2D formulation; choose gauge phi=0 and skip solve.
        for (auto& domain : m_domains)
        {
            m_phiFieldMap[domain]->clear(0.0);
        }
        phys_boundary_update_phi();
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

        double* v_xneg_buffer    = m_vVar->buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer    = m_vVar->buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer    = m_vVar->buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer    = m_vVar->buffer_map[domain][LocationType::YPositive];
        double* u_xneg_buffer    = m_uVar->buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer    = m_uVar->buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer    = m_uVar->buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer    = m_uVar->buffer_map[domain][LocationType::YPositive];
        double  xpos_yneg_corner = m_vVar->xpos_yneg_corner_map[domain];
        double  xneg_ypos_corner = m_uVar->xneg_ypos_corner_map[domain];

        // Helper to get u at (i, j) handling boundaries
        auto get_u = [&](int i, int j) -> double {
            return get_u_with_boundary(
                i, j, nx, ny, u, u_xneg_buffer, u_xpos_buffer, u_yneg_buffer, u_ypos_buffer, xpos_yneg_corner);
        };
        // Helper to get v at (i, j) handling boundaries
        auto get_v = [&](int i, int j) -> double {
            return get_v_with_boundary(
                i, j, nx, ny, v, v_xneg_buffer, v_xpos_buffer, v_yneg_buffer, v_ypos_buffer, xneg_ypos_corner);
        };
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Interpolate velocity to cell center before taking derivatives.
                double v_c_y_ip1 = 0.5 * (get_v(i + 1, j) + get_v(i + 1, j + 1));
                double v_c_y_im1 = 0.5 * (get_v(i - 1, j) + get_v(i - 1, j + 1));
                double u_c_x_jp1 = 0.5 * (get_u(i, j + 1) + get_u(i + 1, j + 1));
                double u_c_x_jm1 = 0.5 * (get_u(i, j - 1) + get_u(i + 1, j - 1));

                double dv_dx = (v_c_y_ip1 - v_c_y_im1) / (2.0 * hx);
                double du_dy = (u_c_x_jp1 - u_c_x_jm1) / (2.0 * hy);

                // Compute RHS = div(u×B) = Bz*(dv/dx - du/dy) * hx * hy (scaled by cell area for Poisson solver)
                rhs(i, j) = m_Bz * (dv_dx - du_dy) * hx * hy;
            }
        }
    }

    phys_boundary_update_phi();

    // For pure Neumann/Periodic Poisson problems, RHS must satisfy the global solvability condition.
    if (isAllNeumannBoundary(*m_phiVar))
    {
        normalizeRhsForNeumannBc(*m_phiVar, m_domains, m_phiFieldMap);
    }

    // Solve Poisson equation; phi overwrites RHS in phi_var->field_map
    m_phiSolver->solve();
    phys_boundary_update_phi();
    buffer_update_phi();
}

void MHDModule2D::updateCurrentDensity()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::updateCurrentDensity(): module not initialized");

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

        auto&   phi_type_map    = m_phiVar->boundary_type_map[domain];
        double* phi_xneg_buffer = m_phiVar->buffer_map[domain][LocationType::XNegative];
        double* phi_yneg_buffer = m_phiVar->buffer_map[domain][LocationType::YNegative];

        double* u_xneg_buffer    = m_uVar->buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer    = m_uVar->buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer    = m_uVar->buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer    = m_uVar->buffer_map[domain][LocationType::YPositive];
        double* v_xneg_buffer    = m_vVar->buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer    = m_vVar->buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer    = m_vVar->buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer    = m_vVar->buffer_map[domain][LocationType::YPositive];
        double  xpos_yneg_corner = m_vVar->xpos_yneg_corner_map[domain];
        double  xneg_ypos_corner = m_uVar->xneg_ypos_corner_map[domain];

        auto get_u = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(
                i_idx, j_idx, nx, ny, u, u_xneg_buffer, u_xpos_buffer, u_yneg_buffer, u_ypos_buffer, xpos_yneg_corner);
        };
        auto get_v = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(
                i_idx, j_idx, nx, ny, v, v_xneg_buffer, v_xpos_buffer, v_yneg_buffer, v_ypos_buffer, xneg_ypos_corner);
        };

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Jx at XFace (collocated with u)
                double dphi_dx = 0.0;
                if (i == 0 && phi_type_map[LocationType::XNegative] == PDEBoundaryType::Neumann)
                {
                    dphi_dx = phi_xneg_buffer[j];
                }
                else if (i == 0 && phi_type_map[LocationType::XNegative] == PDEBoundaryType::Dirichlet)
                {
                    dphi_dx = 2.0 * (phi(i, j) - phi_xneg_buffer[j]) / hx;
                }
                else
                {
                    double phi_im1 = (i == 0) ? phi_xneg_buffer[j] : phi(i - 1, j);
                    dphi_dx        = (phi(i, j) - phi_im1) / hx;
                }
                double v_on_u = 0.25 * (get_v(i - 1, j) + get_v(i, j) + get_v(i - 1, j + 1) + get_v(i, j + 1));
                jx(i, j)      = -dphi_dx + v_on_u * m_Bz;

                // Jy at YFace (collocated with v)
                double dphi_dy = 0.0;
                if (j == 0 && phi_type_map[LocationType::YNegative] == PDEBoundaryType::Neumann)
                {
                    dphi_dy = phi_yneg_buffer[i];
                }
                else if (j == 0 && phi_type_map[LocationType::YNegative] == PDEBoundaryType::Dirichlet)
                {
                    dphi_dy = 2.0 * (phi(i, j) - phi_yneg_buffer[i]) / hy;
                }
                else
                {
                    double phi_jm1 = (j == 0) ? phi_yneg_buffer[i] : phi(i, j - 1);
                    dphi_dy        = (phi(i, j) - phi_jm1) / hy;
                }
                double u_on_v = 0.25 * (get_u(i, j - 1) + get_u(i + 1, j - 1) + get_u(i, j) + get_u(i + 1, j));
                jy(i, j)      = -dphi_dy - u_on_v * m_Bz;

                // Jz at center
                double u_c = 0.5 * (get_u(i, j) + get_u(i + 1, j));
                double v_c = 0.5 * (get_v(i, j) + get_v(i, j + 1));
                jz(i, j)   = u_c * m_By - v_c * m_Bx;
            }
        }
    }

    buffer_update_j();
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

        double* jx_xneg_buffer = m_jxVar->buffer_map[domain][LocationType::XNegative];
        double* jx_xpos_buffer = m_jxVar->buffer_map[domain][LocationType::XPositive];
        double* jx_yneg_buffer = m_jxVar->buffer_map[domain][LocationType::YNegative];
        double* jx_ypos_buffer = m_jxVar->buffer_map[domain][LocationType::YPositive];

        double* jy_xneg_buffer = m_jyVar->buffer_map[domain][LocationType::XNegative];
        double* jy_xpos_buffer = m_jyVar->buffer_map[domain][LocationType::XPositive];
        double* jy_yneg_buffer = m_jyVar->buffer_map[domain][LocationType::YNegative];
        double* jy_ypos_buffer = m_jyVar->buffer_map[domain][LocationType::YPositive];

        double* jz_xneg_buffer = m_jzVar->buffer_map[domain][LocationType::XNegative];
        double* jz_yneg_buffer = m_jzVar->buffer_map[domain][LocationType::YNegative];

        const double jx_xpos_yneg_corner = m_jxVar->xpos_yneg_corner_map[domain];
        const double jy_xneg_ypos_corner = m_jyVar->xneg_ypos_corner_map[domain];

        auto get_jx = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       jx,
                                       jx_xneg_buffer,
                                       jx_xpos_buffer,
                                       jx_yneg_buffer,
                                       jx_ypos_buffer,
                                       jx_xpos_yneg_corner);
        };
        auto get_jy = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       jy,
                                       jy_xneg_buffer,
                                       jy_xpos_buffer,
                                       jy_yneg_buffer,
                                       jy_ypos_buffer,
                                       jy_xneg_ypos_corner);
        };
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

        // Interpolate J to edges and add Lorentz force to predicted velocity.
        // NOTE: At this point in NS predictor step, u/v have been swapped to u_temp/v_temp already.
        // We directly add to current u/v fields.

        // u on XEdge: Fx = Jy*Bz - Jz*By at XFace
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double jy_on_u = 0.25 * (get_jy(i - 1, j) + get_jy(i, j) + get_jy(i - 1, j + 1) + get_jy(i, j + 1));
                double jz_on_u = 0.5 * (get_jz(i - 1, j) + get_jz(i, j));
                double Fx_u    = jy_on_u * m_Bz - jz_on_u * m_By;
                u(i, j) += m_lorentzCoef * Fx_u;
            }
        }

        // v on YEdge: Fy = Jz*Bx - Jx*Bz at YFace
        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                double jx_on_v = 0.25 * (get_jx(i, j - 1) + get_jx(i + 1, j - 1) + get_jx(i, j) + get_jx(i + 1, j));
                double jz_on_v = 0.5 * (get_jz(i, j - 1) + get_jz(i, j));
                double Fy_v    = jz_on_v * m_Bx - jx_on_v * m_Bz;
                v(i, j) += m_lorentzCoef * Fy_v;
            }
        }
    }
}

void MHDModule2D::buffer_update_j()
{
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
        double* phi_yneg_buffer = m_phiVar->buffer_map[domain][LocationType::YNegative];

        double* u_xneg_buffer    = m_uVar->buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer    = m_uVar->buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer    = m_uVar->buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer    = m_uVar->buffer_map[domain][LocationType::YPositive];
        double* v_xneg_buffer    = m_vVar->buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer    = m_vVar->buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer    = m_vVar->buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer    = m_vVar->buffer_map[domain][LocationType::YPositive];
        double  xpos_yneg_corner = m_vVar->xpos_yneg_corner_map[domain];
        double  xneg_ypos_corner = m_uVar->xneg_ypos_corner_map[domain];

        auto& u_type_map   = m_uVar->boundary_type_map[domain];
        auto& v_type_map   = m_vVar->boundary_type_map[domain];
        auto& phi_type_map = m_phiVar->boundary_type_map[domain];

        double* phi_xpos_buffer = m_phiVar->buffer_map[domain][LocationType::XPositive];
        double* phi_ypos_buffer = m_phiVar->buffer_map[domain][LocationType::YPositive];

        auto get_u = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(
                i_idx, j_idx, nx, ny, u, u_xneg_buffer, u_xpos_buffer, u_yneg_buffer, u_ypos_buffer, xpos_yneg_corner);
        };
        auto get_v = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(
                i_idx, j_idx, nx, ny, v, v_xneg_buffer, v_xpos_buffer, v_yneg_buffer, v_ypos_buffer, xneg_ypos_corner);
        };
        auto nearest_phi_row = [&](int j_face) -> int {
            if (j_face < 0)
                return 0;
            if (j_face >= ny)
                return ny - 1;
            return j_face;
        };
        auto nearest_phi_col = [&](int i_face) -> int {
            if (i_face < 0)
                return 0;
            if (i_face >= nx)
                return nx - 1;
            return i_face;
        };
        auto phi_aux_xneg = [&](int j_aux) -> double {
            // Raw phi buffers on physical Dirichlet/Neumann sides store BC data
            // (value or normal derivative), not a ghost-center scalar.
            // Reconstruct the left ghost-center scalar before taking tangential
            // differences on jy_xneg_buffer or other outer auxiliary lines.
            const PDEBoundaryType bc = phi_type_map[LocationType::XNegative];
            if (bc == PDEBoundaryType::Adjacented || bc == PDEBoundaryType::Periodic)
                return phi_xneg_buffer[j_aux];
            if (bc == PDEBoundaryType::Dirichlet)
                return 2.0 * phi_xneg_buffer[j_aux] - phi(0, j_aux);
            if (bc == PDEBoundaryType::Neumann)
                return phi(0, j_aux) - phi_xneg_buffer[j_aux] * hx;
            return phi_xneg_buffer[j_aux];
        };
        auto phi_aux_xpos = [&](int j_aux) -> double {
            const PDEBoundaryType bc = phi_type_map[LocationType::XPositive];
            if (bc == PDEBoundaryType::Adjacented || bc == PDEBoundaryType::Periodic)
                return phi_xpos_buffer[j_aux];
            if (bc == PDEBoundaryType::Dirichlet)
                return 2.0 * phi_xpos_buffer[j_aux] - phi(nx - 1, j_aux);
            if (bc == PDEBoundaryType::Neumann)
                return phi(nx - 1, j_aux) + phi_xpos_buffer[j_aux] * hx;
            return phi_xpos_buffer[j_aux];
        };
        auto phi_aux_yneg = [&](int i_aux) -> double {
            // Symmetric to phi_aux_xneg(): rebuild the bottom ghost-center scalar
            // before using phi on jx_yneg_buffer or other outer auxiliary lines.
            const PDEBoundaryType bc = phi_type_map[LocationType::YNegative];
            if (bc == PDEBoundaryType::Adjacented || bc == PDEBoundaryType::Periodic)
                return phi_yneg_buffer[i_aux];
            if (bc == PDEBoundaryType::Dirichlet)
                return 2.0 * phi_yneg_buffer[i_aux] - phi(i_aux, 0);
            if (bc == PDEBoundaryType::Neumann)
                return phi(i_aux, 0) - phi_yneg_buffer[i_aux] * hy;
            return phi_yneg_buffer[i_aux];
        };
        auto phi_aux_ypos = [&](int i_aux) -> double {
            const PDEBoundaryType bc = phi_type_map[LocationType::YPositive];
            if (bc == PDEBoundaryType::Adjacented || bc == PDEBoundaryType::Periodic)
                return phi_ypos_buffer[i_aux];
            if (bc == PDEBoundaryType::Dirichlet)
                return 2.0 * phi_ypos_buffer[i_aux] - phi(i_aux, ny - 1);
            if (bc == PDEBoundaryType::Neumann)
                return phi(i_aux, ny - 1) + phi_ypos_buffer[i_aux] * hy;
            return phi_ypos_buffer[i_aux];
        };
        auto get_phi_aux = [&](int i_idx, int j_idx) -> double {
            if (i_idx >= 0 && i_idx < nx && j_idx >= 0 && j_idx < ny)
                return phi(i_idx, j_idx);
            if (i_idx < 0)
                return phi_aux_xneg(nearest_phi_row(j_idx));
            if (i_idx >= nx)
                return phi_aux_xpos(nearest_phi_row(j_idx));
            if (j_idx < 0)
                return phi_aux_yneg(nearest_phi_col(i_idx));
            return phi_aux_ypos(nearest_phi_col(i_idx));
        };
        auto u_aux_xneg = [&](int j_aux) -> double {
            // Left-side auxiliary value used by:
            //   1) jy_xneg_buffer[0];
            //   2) the left-top jy corner.
            // Case 1 only happens when XNegative is not Adjacented.
            // Case 2 can still happen when XNegative is Adjacented but YPositive is physical,
            // so keep the Adjacented branch for that mixed corner case.
            // On a physical/Periodic XNegative side, the boundary-maintained value is stored
            // in field column i=0, not in u_xneg_buffer.
            if (u_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
                return u_xneg_buffer[j_aux];
            return u(0, j_aux);
        };
        auto v_aux_yneg = [&](int i_aux) -> double {
            // Bottom-side auxiliary value used by:
            //   1) jx_yneg_buffer[0];
            //   2) the right-bottom jx corner.
            // Case 1 only happens when YNegative is not Adjacented.
            // Case 2 can still happen when YNegative is Adjacented but XPositive is physical,
            // so keep the Adjacented branch for that mixed corner case.
            // On a physical/Periodic YNegative side, the boundary-maintained value is stored
            // in field row j=0, not in v_yneg_buffer.
            if (v_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
                return v_yneg_buffer[i_aux];
            return v(i_aux, 0);
        };
        auto calc_v_on_u = [&](int i_face, int j_face) -> double {
            // Only three jx boundary points are actually read later by applyLorentzForce():
            //   jx_yneg_buffer[0]      -> calc_jx_face(0,  -1)
            //   jx_xpos_buffer[ny - 1] -> calc_jx_face(nx, ny - 1)
            //   jx_xpos_yneg_corner    -> calc_jx_face(nx, -1)
            // These are the actual outer auxiliary points queried by applyLorentzForce().
            // Use a direct diagonal average of those two auxiliary values. When one side
            // does not maintain a physical buffer (notably v on YNegative), supplement the
            // needed auxiliary point from the boundary-maintained field value first.
            // Other outer endpoints such as jx_xneg_buffer[ny - 1] / jx_ypos_buffer[0]
            // are not queried by applyLorentzForce(), so keep their generic auxiliary
            // semantics instead of imposing an extra closure.
            if (i_face == 0 && j_face == -1)
                return 0.5 * (v_xneg_buffer[0] + v_aux_yneg(0));
            if (i_face == nx && j_face == ny - 1)
                return 0.5 * (v_xpos_buffer[ny - 1] + v_ypos_buffer[nx - 1]);
            if (i_face == nx && j_face == -1)
                return 0.5 * (v_xpos_buffer[0] + v_aux_yneg(nx - 1));
            return 0.25 * (get_v(i_face - 1, j_face) + get_v(i_face, j_face) + get_v(i_face - 1, j_face + 1) +
                           get_v(i_face, j_face + 1));
        };
        auto calc_u_on_v = [&](int i_face, int j_face) -> double {
            // Symmetric to calc_v_on_u(): the two jy dual-boundary endpoints plus the
            // left-top jy corner use a direct diagonal average of the two actual
            // auxiliary values. When XNegative has no physical buffer for u, supplement
            // that outer auxiliary point from the maintained field value.
            // The remaining outer endpoints stay generic.
            if (i_face == -1 && j_face == 0)
                return 0.5 * (u_aux_xneg(0) + u_yneg_buffer[0]);
            if (i_face == nx - 1 && j_face == ny)
                return 0.5 * (u_xpos_buffer[ny - 1] + u_ypos_buffer[nx - 1]);
            if (i_face == -1 && j_face == ny)
                return 0.5 * (u_aux_xneg(ny - 1) + u_ypos_buffer[0]);
            return 0.25 * (get_u(i_face, j_face - 1) + get_u(i_face + 1, j_face - 1) + get_u(i_face, j_face) +
                           get_u(i_face + 1, j_face));
        };
        auto calc_jx_face = [&](int i_face, int j_face) -> double {
            const int    j_bound = nearest_phi_row(j_face);
            const double v_on_u  = calc_v_on_u(i_face, j_face);
            double       dphi_dx = 0.0;
            if (i_face == 0 && j_face < 0 && phi_type_map[LocationType::XNegative] == PDEBoundaryType::Neumann)
            {
                // jx_yneg_buffer[0] needs an x-directed phi gradient. On an XNegative
                // Neumann side, use that normal-flux relation directly at this dual-
                // boundary point instead of rebuilding a symmetric phi corner value.
                dphi_dx = m_Bz * v_on_u;
            }
            else if (i_face == 0 && j_face < 0 && phi_type_map[LocationType::XNegative] == PDEBoundaryType::Dirichlet)
            {
                dphi_dx = 2.0 * (phi_aux_yneg(0) - phi_xneg_buffer[0]) / hx;
            }
            else if (i_face == nx && j_face < 0 && phi_type_map[LocationType::XPositive] == PDEBoundaryType::Neumann)
            {
                // Symmetric right-bottom jx corner: still use the x-directed flux
                // relation directly, with v already closed on the two outer auxiliary
                // points.
                dphi_dx = m_Bz * v_on_u;
            }
            else if (i_face == nx && j_face < 0 && phi_type_map[LocationType::XPositive] == PDEBoundaryType::Dirichlet)
            {
                dphi_dx = 2.0 * (phi_xpos_buffer[0] - phi_aux_yneg(nx - 1)) / hx;
            }
            else if (i_face == 0 && phi_type_map[LocationType::XNegative] == PDEBoundaryType::Neumann)
            {
                dphi_dx = phi_xneg_buffer[j_bound];
            }
            else if (i_face == 0 && phi_type_map[LocationType::XNegative] == PDEBoundaryType::Dirichlet)
            {
                dphi_dx = 2.0 * (phi(0, j_bound) - phi_xneg_buffer[j_bound]) / hx;
            }
            else if (i_face == nx && phi_type_map[LocationType::XPositive] == PDEBoundaryType::Neumann)
            {
                dphi_dx = phi_xpos_buffer[j_bound];
            }
            else if (i_face == nx && phi_type_map[LocationType::XPositive] == PDEBoundaryType::Dirichlet)
            {
                dphi_dx = 2.0 * (phi_xpos_buffer[j_bound] - phi(nx - 1, j_bound)) / hx;
            }
            else
            {
                dphi_dx = (get_phi_aux(i_face, j_face) - get_phi_aux(i_face - 1, j_face)) / hx;
            }
            return -dphi_dx + v_on_u * m_Bz;
        };
        auto calc_jy_face = [&](int i_face, int j_face) -> double {
            const int    i_bound = nearest_phi_col(i_face);
            const double u_on_v  = calc_u_on_v(i_face, j_face);
            double       dphi_dy = 0.0;
            if (j_face == 0 && i_face < 0 && phi_type_map[LocationType::YNegative] == PDEBoundaryType::Neumann)
            {
                dphi_dy = -m_Bz * u_on_v;
            }
            else if (j_face == 0 && i_face < 0 && phi_type_map[LocationType::YNegative] == PDEBoundaryType::Dirichlet)
            {
                dphi_dy = 2.0 * (phi_aux_xneg(0) - phi_yneg_buffer[0]) / hy;
            }
            else if (j_face == ny && i_face < 0 && phi_type_map[LocationType::YPositive] == PDEBoundaryType::Neumann)
            {
                dphi_dy = -m_Bz * u_on_v;
            }
            else if (j_face == ny && i_face < 0 && phi_type_map[LocationType::YPositive] == PDEBoundaryType::Dirichlet)
            {
                dphi_dy = 2.0 * (phi_ypos_buffer[0] - phi_aux_xneg(ny - 1)) / hy;
            }
            else if (j_face == 0 && phi_type_map[LocationType::YNegative] == PDEBoundaryType::Neumann)
            {
                dphi_dy = phi_yneg_buffer[i_bound];
            }
            else if (j_face == 0 && phi_type_map[LocationType::YNegative] == PDEBoundaryType::Dirichlet)
            {
                dphi_dy = 2.0 * (phi(i_bound, 0) - phi_yneg_buffer[i_bound]) / hy;
            }
            else if (j_face == ny && phi_type_map[LocationType::YPositive] == PDEBoundaryType::Neumann)
            {
                dphi_dy = phi_ypos_buffer[i_bound];
            }
            else if (j_face == ny && phi_type_map[LocationType::YPositive] == PDEBoundaryType::Dirichlet)
            {
                dphi_dy = 2.0 * (phi_ypos_buffer[i_bound] - phi(i_bound, ny - 1)) / hy;
            }
            else
            {
                dphi_dy = (get_phi_aux(i_face, j_face) - get_phi_aux(i_face, j_face - 1)) / hy;
            }
            return -dphi_dy - u_on_v * m_Bz;
        };

        auto adj_it = m_adjacency.find(domain);

        auto update_jx_xpos_buffer = [&]() {
            if (!m_jxVar->buffer_map.count(domain) || !m_jxVar->buffer_map[domain].count(LocationType::XPositive))
                return;
            double* buf = m_jxVar->buffer_map[domain][LocationType::XPositive];
            if (u_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::XPositive))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::XPositive];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    copy_x_to_buffer(buf, adj_jx, 0);
                }
                return;
            }
            for (int j = 0; j < ny; ++j)
                buf[j] = calc_jx_face(nx, j);
        };
        auto update_jx_xneg_buffer = [&]() {
            if (!m_jxVar->buffer_map.count(domain) || !m_jxVar->buffer_map[domain].count(LocationType::XNegative))
                return;
            double* buf = m_jxVar->buffer_map[domain][LocationType::XNegative];
            if (u_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::XNegative))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::XNegative];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    const int        adj_nx     = adj_jx.get_nx();
                    copy_x_to_buffer(buf, adj_jx, adj_nx - 1);
                }
                return;
            }
            for (int j = 0; j < ny; ++j)
                buf[j] = calc_jx_face(-1, j);
        };
        auto update_jx_yneg_buffer = [&]() {
            if (!m_jxVar->buffer_map.count(domain) || !m_jxVar->buffer_map[domain].count(LocationType::YNegative))
                return;
            double* buf = m_jxVar->buffer_map[domain][LocationType::YNegative];
            if (u_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::YNegative))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::YNegative];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    const int        adj_ny     = adj_jx.get_ny();
                    copy_y_to_buffer(buf, adj_jx, adj_ny - 1);
                }
                return;
            }
            for (int i = 0; i < nx; ++i)
                buf[i] = calc_jx_face(i, -1);
        };
        auto update_jx_ypos_buffer = [&]() {
            if (!m_jxVar->buffer_map.count(domain) || !m_jxVar->buffer_map[domain].count(LocationType::YPositive))
                return;
            double* buf = m_jxVar->buffer_map[domain][LocationType::YPositive];
            if (u_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::YPositive))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::YPositive];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    copy_y_to_buffer(buf, adj_jx, 0);
                }
                return;
            }
            for (int i = 0; i < nx; ++i)
                buf[i] = calc_jx_face(i, ny);
        };
        auto update_jy_ypos_buffer = [&]() {
            if (!m_jyVar->buffer_map.count(domain) || !m_jyVar->buffer_map[domain].count(LocationType::YPositive))
                return;
            double* buf = m_jyVar->buffer_map[domain][LocationType::YPositive];
            if (v_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::YPositive))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::YPositive];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    copy_y_to_buffer(buf, adj_jy, 0);
                }
                return;
            }
            for (int i = 0; i < nx; ++i)
                buf[i] = calc_jy_face(i, ny);
        };
        auto update_jy_yneg_buffer = [&]() {
            if (!m_jyVar->buffer_map.count(domain) || !m_jyVar->buffer_map[domain].count(LocationType::YNegative))
                return;
            double* buf = m_jyVar->buffer_map[domain][LocationType::YNegative];
            if (v_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::YNegative))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::YNegative];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    const int        adj_ny     = adj_jy.get_ny();
                    copy_y_to_buffer(buf, adj_jy, adj_ny - 1);
                }
                return;
            }
            for (int i = 0; i < nx; ++i)
                buf[i] = calc_jy_face(i, -1);
        };
        auto update_jy_xneg_buffer = [&]() {
            if (!m_jyVar->buffer_map.count(domain) || !m_jyVar->buffer_map[domain].count(LocationType::XNegative))
                return;
            double* buf = m_jyVar->buffer_map[domain][LocationType::XNegative];
            if (v_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::XNegative))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::XNegative];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    const int        adj_nx     = adj_jy.get_nx();
                    copy_x_to_buffer(buf, adj_jy, adj_nx - 1);
                }
                return;
            }
            for (int j = 0; j < ny; ++j)
                buf[j] = calc_jy_face(-1, j);
        };
        auto update_jy_xpos_buffer = [&]() {
            if (!m_jyVar->buffer_map.count(domain) || !m_jyVar->buffer_map[domain].count(LocationType::XPositive))
                return;
            double* buf = m_jyVar->buffer_map[domain][LocationType::XPositive];
            if (v_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented)
            {
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::XPositive))
                {
                    Domain2DUniform* adj_domain = adj_it->second[LocationType::XPositive];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    copy_x_to_buffer(buf, adj_jy, 0);
                }
                return;
            }
            for (int j = 0; j < ny; ++j)
                buf[j] = calc_jy_face(nx, j);
        };

        // ---- jx/jy buffers consumed by applyLorentzForce() ----
        // Current MAC Lorentz interpolation reads:
        //   jx: XPositive, YNegative, xpos_yneg corner;
        //   jy: XNegative, YPositive, xneg_ypos corner.
        // Keep these physically relevant paths together so the closure used by the
        // force term is easy to inspect.
        update_jx_xpos_buffer();
        update_jx_yneg_buffer();
        update_jy_xneg_buffer();
        update_jy_ypos_buffer();

        // ---- compatibility-only jx/jy buffers ----
        // These outer lines are kept populated because Variable2D still exposes the
        // full four-sided buffer layout, but applyLorentzForce() does not currently
        // consume them. Leave them in a separate block so they do not obscure the
        // main MHD closure paths above.
        update_jx_xneg_buffer();
        update_jx_ypos_buffer();
        update_jy_yneg_buffer();
        update_jy_xpos_buffer();

        // Corner values for jx/jy (follow u/v diagonal adjacency logic)
        bool jx_corner_set = false;
        bool jy_corner_set = false;
        if (adj_it != m_adjacency.end())
        {
            if (u_type_map[LocationType::YNegative] == PDEBoundaryType::Adjacented &&
                adj_it->second.count(LocationType::YNegative))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::YNegative];
                auto             adj_buf_it = m_jxVar->buffer_map.find(adj_domain);
                if (adj_buf_it != m_jxVar->buffer_map.end() && adj_buf_it->second.count(LocationType::XPositive))
                {
                    double* adj_xpos                      = adj_buf_it->second[LocationType::XPositive];
                    int     adj_ny                        = adj_domain->get_ny();
                    m_jxVar->xpos_yneg_corner_map[domain] = adj_xpos[adj_ny - 1];
                    jx_corner_set                         = true;
                }
            }
            if (u_type_map[LocationType::XPositive] == PDEBoundaryType::Adjacented &&
                adj_it->second.count(LocationType::XPositive))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::XPositive];
                auto&            adj_types  = m_uVar->boundary_type_map[adj_domain];
                if (adj_types[LocationType::YNegative] == PDEBoundaryType::Adjacented &&
                    m_adjacency.count(adj_domain) && m_adjacency.at(adj_domain).count(LocationType::YNegative))
                {
                    Domain2DUniform* diag_domain          = m_adjacency.at(adj_domain).at(LocationType::YNegative);
                    field2&          diag_jx              = *m_jxFieldMap[diag_domain];
                    m_jxVar->xpos_yneg_corner_map[domain] = diag_jx(0, diag_domain->get_ny() - 1);
                    jx_corner_set                         = true;
                }
            }

            if (v_type_map[LocationType::XNegative] == PDEBoundaryType::Adjacented &&
                adj_it->second.count(LocationType::XNegative))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::XNegative];
                auto             adj_buf_it = m_jyVar->buffer_map.find(adj_domain);
                if (adj_buf_it != m_jyVar->buffer_map.end() && adj_buf_it->second.count(LocationType::YPositive))
                {
                    double* adj_ypos                      = adj_buf_it->second[LocationType::YPositive];
                    int     adj_nx                        = adj_domain->get_nx();
                    m_jyVar->xneg_ypos_corner_map[domain] = adj_ypos[adj_nx - 1];
                    jy_corner_set                         = true;
                }
            }
            if (v_type_map[LocationType::YPositive] == PDEBoundaryType::Adjacented &&
                adj_it->second.count(LocationType::YPositive))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::YPositive];
                auto&            adj_types  = m_vVar->boundary_type_map[adj_domain];
                if (adj_types[LocationType::XNegative] == PDEBoundaryType::Adjacented &&
                    m_adjacency.count(adj_domain) && m_adjacency.at(adj_domain).count(LocationType::XNegative))
                {
                    Domain2DUniform* diag_domain          = m_adjacency.at(adj_domain).at(LocationType::XNegative);
                    field2&          diag_jy              = *m_jyFieldMap[diag_domain];
                    m_jyVar->xneg_ypos_corner_map[domain] = diag_jy(diag_domain->get_nx() - 1, 0);
                    jy_corner_set                         = true;
                }
            }
        }

        if (!jx_corner_set)
        {
            const bool xpos_physical = u_type_map[LocationType::XPositive] != PDEBoundaryType::Adjacented;
            const bool yneg_physical = u_type_map[LocationType::YNegative] != PDEBoundaryType::Adjacented;
            if (xpos_physical || yneg_physical)
            {
                m_jxVar->xpos_yneg_corner_map[domain] = calc_jx_face(nx, -1);
            }
        }

        if (!jy_corner_set)
        {
            const bool xneg_physical = v_type_map[LocationType::XNegative] != PDEBoundaryType::Adjacented;
            const bool up_physical   = v_type_map[LocationType::YPositive] != PDEBoundaryType::Adjacented;
            if (xneg_physical || up_physical)
            {
                m_jyVar->xneg_ypos_corner_map[domain] = calc_jy_face(-1, ny);
            }
        }

        // ---- jz (center) ghost buffers ----
        // applyLorentzForce() queries jz at i<0 (XNegative) / j<0 (YNegative) when interpolating to faces.
        // These buffers must be populated; otherwise uninitialized reads can produce NaNs.
        if (m_jzVar && m_jzVar->buffer_map.count(domain))
        {
            auto& jz_buf_map = m_jzVar->buffer_map[domain];
            auto  adj_it     = m_adjacency.find(domain);

            // XNegative buffer: jz(-1, j)
            if (jz_buf_map.count(LocationType::XNegative))
            {
                double* buf = jz_buf_map[LocationType::XNegative];
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::XNegative))
                {
                    Domain2DUniform* adj_domain = adj_it->second.at(LocationType::XNegative);
                    field2&          adj_jz     = *m_jzFieldMap[adj_domain];
                    const int        adj_nx     = adj_jz.get_nx();
                    copy_x_to_buffer(buf, adj_jz, adj_nx - 1);
                }
                else
                {
                    // Physical boundary fallback: zero-gradient
                    copy_x_to_buffer(buf, jz, 0);
                }
            }

            // YNegative buffer: jz(i, -1)
            if (jz_buf_map.count(LocationType::YNegative))
            {
                double* buf = jz_buf_map[LocationType::YNegative];
                if (adj_it != m_adjacency.end() && adj_it->second.count(LocationType::YNegative))
                {
                    Domain2DUniform* adj_domain = adj_it->second.at(LocationType::YNegative);
                    field2&          adj_jz     = *m_jzFieldMap[adj_domain];
                    const int        adj_ny     = adj_jz.get_ny();
                    copy_y_to_buffer(buf, adj_jz, adj_ny - 1);
                }
                else
                {
                    // Physical boundary fallback: zero-gradient
                    copy_y_to_buffer(buf, jz, 0);
                }
            }
        }
    }
}

void MHDModule2D::phys_boundary_update_phi()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::phys_boundary_update_phi(): module not initialized");

    for (auto& domain : m_domains)
    {
        field2& u   = *m_uFieldMap[domain];
        field2& v   = *m_vFieldMap[domain];
        field2& phi = *m_phiFieldMap[domain];

        const int nx = domain->get_nx();
        const int ny = domain->get_ny();

        auto& type_map   = m_phiVar->boundary_type_map[domain];
        auto& has_map    = m_phiVar->has_boundary_value_map[domain];
        auto& val_map    = m_phiVar->boundary_value_map[domain];
        auto& buffer_map = m_phiVar->buffer_map[domain];

        double* u_xneg_buffer    = m_uVar->buffer_map[domain][LocationType::XNegative];
        double* u_xpos_buffer    = m_uVar->buffer_map[domain][LocationType::XPositive];
        double* u_yneg_buffer    = m_uVar->buffer_map[domain][LocationType::YNegative];
        double* u_ypos_buffer    = m_uVar->buffer_map[domain][LocationType::YPositive];
        double* v_xneg_buffer    = m_vVar->buffer_map[domain][LocationType::XNegative];
        double* v_xpos_buffer    = m_vVar->buffer_map[domain][LocationType::XPositive];
        double* v_yneg_buffer    = m_vVar->buffer_map[domain][LocationType::YNegative];
        double* v_ypos_buffer    = m_vVar->buffer_map[domain][LocationType::YPositive];
        double  xpos_yneg_corner = m_vVar->xpos_yneg_corner_map[domain];
        double  xneg_ypos_corner = m_uVar->xneg_ypos_corner_map[domain];

        auto get_u = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(
                i_idx, j_idx, nx, ny, u, u_xneg_buffer, u_xpos_buffer, u_yneg_buffer, u_ypos_buffer, xpos_yneg_corner);
        };
        auto get_v = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(
                i_idx, j_idx, nx, ny, v, v_xneg_buffer, v_xpos_buffer, v_yneg_buffer, v_ypos_buffer, xneg_ypos_corner);
        };
        auto get_bound_val = [&](LocationType loc, int idx) -> double {
            auto has_it = has_map.find(loc);
            auto val_it = val_map.find(loc);
            if (has_it != has_map.end() && has_it->second && val_it != val_map.end() && val_it->second != nullptr)
            {
                return val_it->second[idx];
            }
            return 0.0;
        };

        for (auto& [loc, type] : type_map)
        {
            if (type == PDEBoundaryType::Adjacented || !buffer_map.count(loc))
                continue;

            switch (loc)
            {
                case LocationType::XNegative:
                    if (type == PDEBoundaryType::Dirichlet)
                    {
                        for (int j = 0; j < ny; ++j)
                            buffer_map[loc][j] = get_bound_val(loc, j);
                    }
                    else if (type == PDEBoundaryType::Neumann)
                    {
                        for (int j = 0; j < ny; ++j)
                        {
                            const double v_on_u =
                                0.25 * (get_v(-1, j) + get_v(0, j) + get_v(-1, j + 1) + get_v(0, j + 1));
                            buffer_map[loc][j] = v_on_u * m_Bz;
                        }
                    }
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], phi, nx - 1);
                    else
                        copy_x_to_buffer(buffer_map[loc], phi, 0);
                    break;
                case LocationType::XPositive:
                    if (type == PDEBoundaryType::Dirichlet)
                    {
                        for (int j = 0; j < ny; ++j)
                            buffer_map[loc][j] = get_bound_val(loc, j);
                    }
                    else if (type == PDEBoundaryType::Neumann)
                    {
                        for (int j = 0; j < ny; ++j)
                        {
                            const double v_on_u =
                                0.25 * (get_v(nx - 1, j) + get_v(nx, j) + get_v(nx - 1, j + 1) + get_v(nx, j + 1));
                            buffer_map[loc][j] = v_on_u * m_Bz;
                        }
                    }
                    else if (type == PDEBoundaryType::Periodic)
                        copy_x_to_buffer(buffer_map[loc], phi, 0);
                    else
                        copy_x_to_buffer(buffer_map[loc], phi, nx - 1);
                    break;
                case LocationType::YNegative:
                    if (type == PDEBoundaryType::Dirichlet)
                    {
                        for (int i = 0; i < nx; ++i)
                            buffer_map[loc][i] = get_bound_val(loc, i);
                    }
                    else if (type == PDEBoundaryType::Neumann)
                    {
                        for (int i = 0; i < nx; ++i)
                        {
                            const double u_on_v =
                                0.25 * (get_u(i, -1) + get_u(i + 1, -1) + get_u(i, 0) + get_u(i + 1, 0));
                            buffer_map[loc][i] = -u_on_v * m_Bz;
                        }
                    }
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], phi, ny - 1);
                    else
                        copy_y_to_buffer(buffer_map[loc], phi, 0);
                    break;
                case LocationType::YPositive:
                    if (type == PDEBoundaryType::Dirichlet)
                    {
                        for (int i = 0; i < nx; ++i)
                            buffer_map[loc][i] = get_bound_val(loc, i);
                    }
                    else if (type == PDEBoundaryType::Neumann)
                    {
                        for (int i = 0; i < nx; ++i)
                        {
                            const double u_on_v =
                                0.25 * (get_u(i, ny - 1) + get_u(i + 1, ny - 1) + get_u(i, ny) + get_u(i + 1, ny));
                            buffer_map[loc][i] = -u_on_v * m_Bz;
                        }
                    }
                    else if (type == PDEBoundaryType::Periodic)
                        copy_y_to_buffer(buffer_map[loc], phi, 0);
                    else
                        copy_y_to_buffer(buffer_map[loc], phi, ny - 1);
                    break;
                default:
                    throw std::runtime_error("MHDModule2D::phys_boundary_update_phi(): invalid location type");
            }
        }
    }
}

void MHDModule2D::buffer_update_phi()
{
    if (!m_initialized)
        throw std::runtime_error("MHDModule2D::buffer_update_phi(): module not initialized");

    for (auto& domain : m_domains)
    {
        auto& type_map   = m_phiVar->boundary_type_map[domain];
        auto& buffer_map = m_phiVar->buffer_map[domain];
        auto  adj_dom_it = m_adjacency.find(domain);

        for (auto& [loc, type] : type_map)
        {
            if (type != PDEBoundaryType::Adjacented || !buffer_map.count(loc) || adj_dom_it == m_adjacency.end() ||
                !adj_dom_it->second.count(loc))
            {
                continue;
            }

            Domain2DUniform* adj_domain = adj_dom_it->second[loc];
            field2&          adj_phi    = *m_phiFieldMap[adj_domain];
            const int        adj_nx     = adj_phi.get_nx();
            const int        adj_ny     = adj_phi.get_ny();

            switch (loc)
            {
                case LocationType::XNegative:
                    copy_x_to_buffer(buffer_map[loc], adj_phi, adj_nx - 1);
                    break;
                case LocationType::XPositive:
                    copy_x_to_buffer(buffer_map[loc], adj_phi, 0);
                    break;
                case LocationType::YNegative:
                    copy_y_to_buffer(buffer_map[loc], adj_phi, adj_ny - 1);
                    break;
                case LocationType::YPositive:
                    copy_y_to_buffer(buffer_map[loc], adj_phi, 0);
                    break;
                default:
                    throw std::runtime_error("MHDModule2D::buffer_update_phi(): invalid location type");
            }
        }
    }
}
