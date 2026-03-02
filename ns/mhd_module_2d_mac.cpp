#include "mhd_module_2d_mac.h"

#include "base/parallel/omp/enable_openmp.h"
#include "boundary_2d_utils.h"
#include <algorithm>
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

        // Ensure phi has Right/Up buffers for MHD-specific usage.
        auto& phi_buffer_map = m_phiVar->buffer_map[domain];
        if (!phi_buffer_map.count(LocationType::Right))
        {
            phi_buffer_map[LocationType::Right] = new double[domain->get_ny()];
            std::fill_n(phi_buffer_map[LocationType::Right], static_cast<std::size_t>(domain->get_ny()), 0.0);
        }
        if (!phi_buffer_map.count(LocationType::Up))
        {
            phi_buffer_map[LocationType::Up] = new double[domain->get_nx()];
            std::fill_n(phi_buffer_map[LocationType::Up], static_cast<std::size_t>(domain->get_nx()), 0.0);
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

        double* v_left_buffer           = m_vVar->buffer_map[domain][LocationType::Left];
        double* v_right_buffer          = m_vVar->buffer_map[domain][LocationType::Right];
        double* v_down_buffer           = m_vVar->buffer_map[domain][LocationType::Down];
        double* v_up_buffer             = m_vVar->buffer_map[domain][LocationType::Up];
        double* u_left_buffer           = m_uVar->buffer_map[domain][LocationType::Left];
        double* u_right_buffer          = m_uVar->buffer_map[domain][LocationType::Right];
        double* u_down_buffer           = m_uVar->buffer_map[domain][LocationType::Down];
        double* u_up_buffer             = m_uVar->buffer_map[domain][LocationType::Up];
        double  right_down_corner_value = m_vVar->right_down_corner_value_map[domain];
        double  left_up_corner_value    = m_uVar->left_up_corner_value_map[domain];

        // Helper to get u at (i, j) handling boundaries
        auto get_u = [&](int i, int j) -> double {
            return get_u_with_boundary(
                i, j, nx, ny, u, u_left_buffer, u_right_buffer, u_down_buffer, u_up_buffer, right_down_corner_value);
        };
        // Helper to get v at (i, j) handling boundaries
        auto get_v = [&](int i, int j) -> double {
            return get_v_with_boundary(
                i, j, nx, ny, v, v_left_buffer, v_right_buffer, v_down_buffer, v_up_buffer, left_up_corner_value);
        };
        // // Helper lambda for du/dx at (i, j_row) where u is defined
        // auto calc_du_dx_row = [&](int i_idx, int j_idx) -> double {
        //     if (i_idx > 0 && i_idx < nx)
        //         return (get_u(i_idx + 1, j_idx) - get_u(i_idx - 1, j_idx)) / (2.0 * hx);
        //     else if (i_idx == 0)
        //         return (-3 * get_u(0, j_idx) + 4 * get_u(1, j_idx) - get_u(2, j_idx)) /
        //                hx; // Forward difference at 2 order accuaracy
        //     else           // i_idx == nx
        //         return (3 * get_u(nx, j_idx) - 4 * get_u(nx - 1, j_idx) + get_u(nx - 2, j_idx)) /
        //                hx; // Backward difference at 2 order accuaracy
        // };
        // auto calc_dv_dx_row = [&](int i_idx, int j_idx) -> double {
        //     if (i_idx > 0 && i_idx < nx)
        //         return (get_v(i_idx + 1, j_idx) - get_v(i_idx - 1, j_idx)) / (2.0 * hx);
        //     else if (i_idx == 0)
        //         return (-3 * get_v(0, j_idx) + 4 * get_v(1, j_idx) - get_v(2, j_idx)) /
        //                hx; // Forward difference at 2 order accuaracy
        //     else           // i_idx == nx
        //         return (3 * get_v(nx, j_idx) - 4 * get_v(nx - 1, j_idx) + get_v(nx - 2, j_idx)) /
        //                hx; // Backward difference at 2 order accuaracy
        // };
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

    // For pure Neumann/Periodic Poisson problems, RHS must satisfy the global solvability condition.
    if (isAllNeumannBoundary(*m_phiVar))
    {
        normalizeRhsForNeumannBc(*m_phiVar, m_domains, m_phiFieldMap);
    }

    // Solve Poisson equation; phi overwrites RHS in phi_var->field_map
    m_phiSolver->solve();
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

        double* phi_left_buffer = m_phiVar->buffer_map[domain][LocationType::Left];
        double* phi_down_buffer = m_phiVar->buffer_map[domain][LocationType::Down];

        double* u_left_buffer           = m_uVar->buffer_map[domain][LocationType::Left];
        double* u_right_buffer          = m_uVar->buffer_map[domain][LocationType::Right];
        double* u_down_buffer           = m_uVar->buffer_map[domain][LocationType::Down];
        double* u_up_buffer             = m_uVar->buffer_map[domain][LocationType::Up];
        double* v_left_buffer           = m_vVar->buffer_map[domain][LocationType::Left];
        double* v_right_buffer          = m_vVar->buffer_map[domain][LocationType::Right];
        double* v_down_buffer           = m_vVar->buffer_map[domain][LocationType::Down];
        double* v_up_buffer             = m_vVar->buffer_map[domain][LocationType::Up];
        double  right_down_corner_value = m_vVar->right_down_corner_value_map[domain];
        double  left_up_corner_value    = m_uVar->left_up_corner_value_map[domain];

        auto get_u = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       u,
                                       u_left_buffer,
                                       u_right_buffer,
                                       u_down_buffer,
                                       u_up_buffer,
                                       right_down_corner_value);
        };
        auto get_v = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       v,
                                       v_left_buffer,
                                       v_right_buffer,
                                       v_down_buffer,
                                       v_up_buffer,
                                       left_up_corner_value);
        };

        OPENMP_PARALLEL_FOR()
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                // Jx at XFace (collocated with u)
                double phi_im1 = (i == 0) ? phi_left_buffer[j] : phi(i - 1, j);
                double dphi_dx = (phi(i, j) - phi_im1) / hx;
                double v_on_u  = 0.25 * (get_v(i - 1, j) + get_v(i, j) + get_v(i - 1, j + 1) + get_v(i, j + 1));
                jx(i, j)       = -dphi_dx + v_on_u * m_Bz;

                // Jy at YFace (collocated with v)
                double phi_jm1 = (j == 0) ? phi_down_buffer[i] : phi(i, j - 1);
                double dphi_dy = (phi(i, j) - phi_jm1) / hy;
                double u_on_v  = 0.25 * (get_u(i, j - 1) + get_u(i + 1, j - 1) + get_u(i, j) + get_u(i + 1, j));
                jy(i, j)       = -dphi_dy - u_on_v * m_Bz;

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

        double* jx_left_buffer  = m_jxVar->buffer_map[domain][LocationType::Left];
        double* jx_right_buffer = m_jxVar->buffer_map[domain][LocationType::Right];
        double* jx_down_buffer  = m_jxVar->buffer_map[domain][LocationType::Down];
        double* jx_up_buffer    = m_jxVar->buffer_map[domain][LocationType::Up];

        double* jy_left_buffer  = m_jyVar->buffer_map[domain][LocationType::Left];
        double* jy_right_buffer = m_jyVar->buffer_map[domain][LocationType::Right];
        double* jy_down_buffer  = m_jyVar->buffer_map[domain][LocationType::Down];
        double* jy_up_buffer    = m_jyVar->buffer_map[domain][LocationType::Up];

        double* jz_left_buffer = m_jzVar->buffer_map[domain][LocationType::Left];
        double* jz_down_buffer = m_jzVar->buffer_map[domain][LocationType::Down];

        const double jx_right_down_corner = m_jxVar->right_down_corner_value_map[domain];
        const double jy_left_up_corner    = m_jyVar->left_up_corner_value_map[domain];

        auto get_jx = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       jx,
                                       jx_left_buffer,
                                       jx_right_buffer,
                                       jx_down_buffer,
                                       jx_up_buffer,
                                       jx_right_down_corner);
        };
        auto get_jy = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       jy,
                                       jy_left_buffer,
                                       jy_right_buffer,
                                       jy_down_buffer,
                                       jy_up_buffer,
                                       jy_left_up_corner);
        };
        auto get_jz = [&](int i_idx, int j_idx) -> double {
            if (i_idx < 0)
            {
                int jj = (j_idx < 0) ? 0 : (j_idx >= ny ? ny - 1 : j_idx);
                return jz_left_buffer[jj];
            }
            if (j_idx < 0)
            {
                int ii = (i_idx < 0) ? 0 : (i_idx >= nx ? nx - 1 : i_idx);
                return jz_down_buffer[ii];
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

        const int    nx = domain->get_nx();
        const int    ny = domain->get_ny();
        const double hx = domain->hx;
        const double hy = domain->hy;

        double* phi_left_buffer = m_phiVar->buffer_map[domain][LocationType::Left];
        double* phi_down_buffer = m_phiVar->buffer_map[domain][LocationType::Down];

        double* u_left_buffer           = m_uVar->buffer_map[domain][LocationType::Left];
        double* u_right_buffer          = m_uVar->buffer_map[domain][LocationType::Right];
        double* u_down_buffer           = m_uVar->buffer_map[domain][LocationType::Down];
        double* u_up_buffer             = m_uVar->buffer_map[domain][LocationType::Up];
        double* v_left_buffer           = m_vVar->buffer_map[domain][LocationType::Left];
        double* v_right_buffer          = m_vVar->buffer_map[domain][LocationType::Right];
        double* v_down_buffer           = m_vVar->buffer_map[domain][LocationType::Down];
        double* v_up_buffer             = m_vVar->buffer_map[domain][LocationType::Up];
        double  right_down_corner_value = m_vVar->right_down_corner_value_map[domain];
        double  left_up_corner_value    = m_uVar->left_up_corner_value_map[domain];

        auto& u_type_map = m_uVar->boundary_type_map[domain];
        auto& v_type_map = m_vVar->boundary_type_map[domain];

        auto& phi_type_map = m_phiVar->boundary_type_map[domain];
        auto& phi_has_map  = m_phiVar->has_boundary_value_map[domain];
        auto& phi_val_map  = m_phiVar->boundary_value_map[domain];

        auto is_zero_neumann = [&](LocationType loc, int len) -> bool {
            if (phi_type_map[loc] != PDEBoundaryType::Neumann)
                return false;
            if (!phi_has_map[loc])
                return true;
            double* val = phi_val_map[loc];
            if (val == nullptr)
                return true;
            for (int i = 0; i < len; ++i)
            {
                if (std::abs(val[i]) > 1e-12)
                    return false;
            }
            return true;
        };

        double* phi_right_buffer = m_phiVar->buffer_map[domain][LocationType::Right];
        double* phi_up_buffer    = m_phiVar->buffer_map[domain][LocationType::Up];

        auto get_u = [&](int i_idx, int j_idx) -> double {
            return get_u_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       u,
                                       u_left_buffer,
                                       u_right_buffer,
                                       u_down_buffer,
                                       u_up_buffer,
                                       right_down_corner_value);
        };
        auto get_v = [&](int i_idx, int j_idx) -> double {
            return get_v_with_boundary(i_idx,
                                       j_idx,
                                       nx,
                                       ny,
                                       v,
                                       v_left_buffer,
                                       v_right_buffer,
                                       v_down_buffer,
                                       v_up_buffer,
                                       left_up_corner_value);
        };
        auto get_phi = [&](int i_idx, int j_idx) -> double {
            if (i_idx >= 0 && i_idx < nx && j_idx >= 0 && j_idx < ny)
                return phi(i_idx, j_idx);
            if (i_idx < 0)
                return phi_left_buffer[j_idx];
            if (i_idx >= nx)
                return phi_right_buffer[j_idx];
            if (j_idx < 0)
                return phi_down_buffer[i_idx];
            return phi_up_buffer[i_idx];
        };
        auto calc_jx_face = [&](int i_face, int j_face) -> double {
            double dphi_dx = (get_phi(i_face, j_face) - get_phi(i_face - 1, j_face)) / hx;
            double v_on_u  = 0.25 * (get_v(i_face - 1, j_face) + get_v(i_face, j_face) + get_v(i_face - 1, j_face + 1) +
                                    get_v(i_face, j_face + 1));
            return -dphi_dx + v_on_u * m_Bz;
        };
        auto calc_jy_face = [&](int i_face, int j_face) -> double {
            double dphi_dy = (get_phi(i_face, j_face) - get_phi(i_face, j_face - 1)) / hy;
            double u_on_v  = 0.25 * (get_u(i_face, j_face - 1) + get_u(i_face + 1, j_face - 1) + get_u(i_face, j_face) +
                                    get_u(i_face + 1, j_face));
            return -dphi_dy - u_on_v * m_Bz;
        };

        // Right buffer for jx (i = nx)
        if (m_jxVar->buffer_map.count(domain) && m_jxVar->buffer_map[domain].count(LocationType::Right))
        {
            double* buf = m_jxVar->buffer_map[domain][LocationType::Right];
            if (u_type_map[LocationType::Right] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Right))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Right];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    copy_x_to_buffer(buf, adj_jx, 0);
                }
            }
            else if (is_zero_neumann(LocationType::Right, ny))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, ny, nullptr, 0.0);
            }
            else
            {
                for (int j = 0; j < ny; ++j)
                    buf[j] = calc_jx_face(nx, j);
            }
        }

        // Left buffer for jx (i = -1)
        if (m_jxVar->buffer_map.count(domain) && m_jxVar->buffer_map[domain].count(LocationType::Left))
        {
            double* buf = m_jxVar->buffer_map[domain][LocationType::Left];
            if (u_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Left))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Left];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    const int        adj_nx     = adj_jx.get_nx();
                    copy_x_to_buffer(buf, adj_jx, adj_nx - 1);
                }
            }
            else if (is_zero_neumann(LocationType::Left, ny))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, ny, nullptr, 0.0);
            }
            else
            {
                for (int j = 0; j < ny; ++j)
                    buf[j] = calc_jx_face(-1, j);
            }
        }

        // Down buffer for jx (j = -1)
        if (m_jxVar->buffer_map.count(domain) && m_jxVar->buffer_map[domain].count(LocationType::Down))
        {
            double* buf = m_jxVar->buffer_map[domain][LocationType::Down];
            if (u_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Down))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Down];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    const int        adj_ny     = adj_jx.get_ny();
                    copy_y_to_buffer(buf, adj_jx, adj_ny - 1);
                }
            }
            else if (is_zero_neumann(LocationType::Down, nx))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, nx, nullptr, 0.0);
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                    buf[i] = calc_jx_face(i, -1);
            }
        }

        // Up buffer for jx (j = ny)
        if (m_jxVar->buffer_map.count(domain) && m_jxVar->buffer_map[domain].count(LocationType::Up))
        {
            double* buf = m_jxVar->buffer_map[domain][LocationType::Up];
            if (u_type_map[LocationType::Up] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Up))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Up];
                    field2&          adj_jx     = *m_jxFieldMap[adj_domain];
                    copy_y_to_buffer(buf, adj_jx, 0);
                }
            }
            else if (is_zero_neumann(LocationType::Up, nx))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, nx, nullptr, 0.0);
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                    buf[i] = calc_jx_face(i, ny);
            }
        }

        // Up buffer for jy (j = ny)
        if (m_jyVar->buffer_map.count(domain) && m_jyVar->buffer_map[domain].count(LocationType::Up))
        {
            double* buf = m_jyVar->buffer_map[domain][LocationType::Up];
            if (v_type_map[LocationType::Up] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Up))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Up];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    copy_y_to_buffer(buf, adj_jy, 0);
                }
            }
            else if (is_zero_neumann(LocationType::Up, nx))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, nx, nullptr, 0.0);
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                    buf[i] = calc_jy_face(i, ny);
            }
        }

        // Down buffer for jy (j = -1)
        if (m_jyVar->buffer_map.count(domain) && m_jyVar->buffer_map[domain].count(LocationType::Down))
        {
            double* buf = m_jyVar->buffer_map[domain][LocationType::Down];
            if (v_type_map[LocationType::Down] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Down))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Down];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    const int        adj_ny     = adj_jy.get_ny();
                    copy_y_to_buffer(buf, adj_jy, adj_ny - 1);
                }
            }
            else if (is_zero_neumann(LocationType::Down, nx))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, nx, nullptr, 0.0);
            }
            else
            {
                for (int i = 0; i < nx; ++i)
                    buf[i] = calc_jy_face(i, -1);
            }
        }

        // Left buffer for jy (i = -1)
        if (m_jyVar->buffer_map.count(domain) && m_jyVar->buffer_map[domain].count(LocationType::Left))
        {
            double* buf = m_jyVar->buffer_map[domain][LocationType::Left];
            if (v_type_map[LocationType::Left] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Left))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Left];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    const int        adj_nx     = adj_jy.get_nx();
                    copy_x_to_buffer(buf, adj_jy, adj_nx - 1);
                }
            }
            else if (is_zero_neumann(LocationType::Left, ny))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, ny, nullptr, 0.0);
            }
            else
            {
                for (int j = 0; j < ny; ++j)
                    buf[j] = calc_jy_face(-1, j);
            }
        }

        // Right buffer for jy (i = nx)
        if (m_jyVar->buffer_map.count(domain) && m_jyVar->buffer_map[domain].count(LocationType::Right))
        {
            double* buf = m_jyVar->buffer_map[domain][LocationType::Right];
            if (v_type_map[LocationType::Right] == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Right))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Right];
                    field2&          adj_jy     = *m_jyFieldMap[adj_domain];
                    copy_x_to_buffer(buf, adj_jy, 0);
                }
            }
            else if (is_zero_neumann(LocationType::Right, ny))
            {
                // Insulating wall: enforce Jn = 0
                assign_val_to_buffer(buf, ny, nullptr, 0.0);
            }
            else
            {
                for (int j = 0; j < ny; ++j)
                    buf[j] = calc_jy_face(nx, j);
            }
        }

        // Corner values for jx/jy (follow u/v diagonal adjacency logic)
        bool jx_corner_set = false;
        bool jy_corner_set = false;
        auto adj_it        = m_adjacency.find(domain);
        if (adj_it != m_adjacency.end())
        {
            if (u_type_map[LocationType::Down] == PDEBoundaryType::Adjacented &&
                adj_it->second.count(LocationType::Down))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::Down];
                auto             adj_buf_it = m_jxVar->buffer_map.find(adj_domain);
                if (adj_buf_it != m_jxVar->buffer_map.end() && adj_buf_it->second.count(LocationType::Right))
                {
                    double* adj_right                            = adj_buf_it->second[LocationType::Right];
                    int     adj_ny                               = adj_domain->get_ny();
                    m_jxVar->right_down_corner_value_map[domain] = adj_right[adj_ny - 1];
                    jx_corner_set                                = true;
                }
            }
            if (u_type_map[LocationType::Right] == PDEBoundaryType::Adjacented &&
                adj_it->second.count(LocationType::Right))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::Right];
                auto&            adj_types  = m_uVar->boundary_type_map[adj_domain];
                if (adj_types[LocationType::Down] == PDEBoundaryType::Adjacented && m_adjacency.count(adj_domain) &&
                    m_adjacency.at(adj_domain).count(LocationType::Down))
                {
                    Domain2DUniform* diag_domain                 = m_adjacency.at(adj_domain).at(LocationType::Down);
                    field2&          diag_jx                     = *m_jxFieldMap[diag_domain];
                    m_jxVar->right_down_corner_value_map[domain] = diag_jx(0, diag_domain->get_ny() - 1);
                    jx_corner_set                                = true;
                }
            }

            if (v_type_map[LocationType::Left] == PDEBoundaryType::Adjacented &&
                adj_it->second.count(LocationType::Left))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::Left];
                auto             adj_buf_it = m_jyVar->buffer_map.find(adj_domain);
                if (adj_buf_it != m_jyVar->buffer_map.end() && adj_buf_it->second.count(LocationType::Up))
                {
                    double* adj_up                            = adj_buf_it->second[LocationType::Up];
                    int     adj_nx                            = adj_domain->get_nx();
                    m_jyVar->left_up_corner_value_map[domain] = adj_up[adj_nx - 1];
                    jy_corner_set                             = true;
                }
            }
            if (v_type_map[LocationType::Up] == PDEBoundaryType::Adjacented && adj_it->second.count(LocationType::Up))
            {
                Domain2DUniform* adj_domain = adj_it->second[LocationType::Up];
                auto&            adj_types  = m_vVar->boundary_type_map[adj_domain];
                if (adj_types[LocationType::Left] == PDEBoundaryType::Adjacented && m_adjacency.count(adj_domain) &&
                    m_adjacency.at(adj_domain).count(LocationType::Left))
                {
                    Domain2DUniform* diag_domain              = m_adjacency.at(adj_domain).at(LocationType::Left);
                    field2&          diag_jy                  = *m_jyFieldMap[diag_domain];
                    m_jyVar->left_up_corner_value_map[domain] = diag_jy(diag_domain->get_nx() - 1, 0);
                    jy_corner_set                             = true;
                }
            }
        }

        if (!jx_corner_set)
        {
            const bool right_physical = u_type_map[LocationType::Right] != PDEBoundaryType::Adjacented;
            const bool down_physical  = u_type_map[LocationType::Down] != PDEBoundaryType::Adjacented;
            if (right_physical || down_physical)
            {
                if (is_zero_neumann(LocationType::Right, ny) || is_zero_neumann(LocationType::Down, nx))
                {
                    m_jxVar->right_down_corner_value_map[domain] = 0.0;
                }
                else
                {
                    double dphi_dx = 0.0;
                    if (nx >= 3)
                    {
                        dphi_dx = (2.0 * phi(nx - 1, 0) - 3.0 * phi(nx - 2, 0) + phi(nx - 3, 0)) / hx;
                    }
                    else if (nx >= 2)
                    {
                        dphi_dx = (phi(nx - 1, 0) - phi(nx - 2, 0)) / hx;
                    }

                    const double v_down                          = (nx > 0) ? v_down_buffer[nx - 1] : 0.0;
                    const double v_right                         = (ny > 0) ? v_right_buffer[0] : 0.0;
                    const double v_on_u                          = 0.5 * (v_down + v_right);
                    m_jxVar->right_down_corner_value_map[domain] = -dphi_dx + v_on_u * m_Bz;
                }
            }
        }

        if (!jy_corner_set)
        {
            const bool left_physical = v_type_map[LocationType::Left] != PDEBoundaryType::Adjacented;
            const bool up_physical   = v_type_map[LocationType::Up] != PDEBoundaryType::Adjacented;
            if (left_physical || up_physical)
            {
                if (is_zero_neumann(LocationType::Left, ny) || is_zero_neumann(LocationType::Up, nx))
                {
                    m_jyVar->left_up_corner_value_map[domain] = 0.0;
                }
                else
                {
                    double dphi_dy = 0.0;
                    if (ny >= 3)
                    {
                        dphi_dy = (2.0 * phi(0, ny - 1) - 3.0 * phi(0, ny - 2) + phi(0, ny - 3)) / hy;
                    }
                    else if (ny >= 2)
                    {
                        dphi_dy = (phi(0, ny - 1) - phi(0, ny - 2)) / hy;
                    }

                    const double u_left                       = (ny > 0) ? u_left_buffer[ny - 1] : 0.0;
                    const double u_up                         = (nx > 0) ? u_up_buffer[0] : 0.0;
                    const double u_on_v                       = 0.5 * (u_left + u_up);
                    m_jyVar->left_up_corner_value_map[domain] = -dphi_dy - u_on_v * m_Bz;
                }
            }
        }
    }
}

void MHDModule2D::buffer_update_phi()
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

        // Center variable2d: buffer ownership is Left/Down only.
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

        // Right buffer: phi(nx, j)
        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::Right))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::Right];
            PDEBoundaryType type = type_map[LocationType::Right];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Right))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Right];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    copy_x_to_buffer(buf, adj_phi, 0);
                }
            }
            else if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::Right] ? val_map[LocationType::Right] : nullptr);
                mirror_x_to_buffer(buf, phi, nx - 1, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::Right] ? val_map[LocationType::Right] : nullptr);
                neumann_x_to_buffer(buf, phi, nx - 1, q_ptr, 0.0, hx, +1.0);
            }
            else
            {
                copy_x_to_buffer(buf, phi, nx - 1);
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

        // Up buffer: phi(i, ny)
        if (m_phiVar->buffer_map.count(domain) && m_phiVar->buffer_map[domain].count(LocationType::Up))
        {
            double*         buf  = m_phiVar->buffer_map[domain][LocationType::Up];
            PDEBoundaryType type = type_map[LocationType::Up];

            if (type == PDEBoundaryType::Adjacented)
            {
                auto adj_dom_it = m_adjacency.find(domain);
                if (adj_dom_it != m_adjacency.end() && adj_dom_it->second.count(LocationType::Up))
                {
                    Domain2DUniform* adj_domain = adj_dom_it->second[LocationType::Up];
                    field2&          adj_phi    = *m_phiFieldMap[adj_domain];
                    copy_y_to_buffer(buf, adj_phi, 0);
                }
            }
            else if (type == PDEBoundaryType::Dirichlet)
            {
                double* g_ptr = (has_map[LocationType::Up] ? val_map[LocationType::Up] : nullptr);
                mirror_y_to_buffer(buf, phi, ny - 1, g_ptr, 0.0);
            }
            else if (type == PDEBoundaryType::Neumann)
            {
                double* q_ptr = (has_map[LocationType::Up] ? val_map[LocationType::Up] : nullptr);
                neumann_y_to_buffer(buf, phi, ny - 1, q_ptr, 0.0, hy, +1.0);
            }
            else
            {
                copy_y_to_buffer(buf, phi, ny - 1);
            }
        }
    }
}
