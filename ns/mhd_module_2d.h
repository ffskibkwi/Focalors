#pragma once

#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/config.h"
#include "pe/concat/concat_solver2d.h"

#include <memory>
#include <unordered_map>
#include <vector>

// Forward declaration
class ConcatNSSolver2D;

/**
 * @brief MHD module for 2D quasi-static/induced electric potential method
 * 
 * Implements the following algorithm:
 * 1. Solve Poisson equation for electric potential phi: div(u×B) = RHS
 * 2. Update current density: J = -∇phi + u×B
 * 3. Apply Lorentz force: F = J×B to predicted velocity
 */
class MHDModule2D
{
public:
    MHDModule2D(Variable*            in_u_var,
                Variable*            in_v_var,
                PhysicsConfig*       in_phy_config,
                TimeAdvancingConfig* in_time_config,
                EnvironmentConfig*   in_env_config = nullptr);
    ~MHDModule2D();

    /**
     * @brief Initialize MHD fields and boundary conditions
     *
     * Allocates phi, jx, jy, jz fields (all at cell centers)
     *
     * @param phi_var Optional pre-configured phi variable (with boundary conditions).
     *                If nullptr, creates internal phi variable with default Neumann BCs.
     */
    void init(Variable* phi_var = nullptr);

    /**
     * @brief Solve Poisson equation for electric potential
     * 
     * Computes RHS = div(u×B) = Bz*(dv/dx - du/dy) using conservative discretization
     * Solves for phi (solution unique up to a constant due to Neumann BC)
     */
    void solveElectricPotential();

    /**
     * @brief Update current density based on Ohm's law
     * 
     * Computes J = -∇phi + u×B at cell centers
     * - jx = -dphi/dx + (u×B)_x
     * - jy = -dphi/dy + (u×B)_y  
     * - jz = -dphi/dz + (u×B)_z = u*By - v*Bx (2D case)
     */
    void updateCurrentDensity();

    /**
     * @brief Apply Lorentz force to predicted velocity
     * 
     * Adds dt*(Ha^2/Re)*(J×B) to the predicted velocity fields (u_temp, v_temp)
     * Forces computed at centers are interpolated to edges (MAC grid staggering)
     */
    void applyLorentzForce();

private:
    void buffer_update();

    // Reference to NS velocity variables (for geometry and fields)
    Variable* m_uVar;
    Variable* m_vVar;

    // MHD-specific variables (all at cell centers)
    std::unique_ptr<Variable> m_phiVar; // Electric potential
    std::unique_ptr<Variable> m_jxVar;  // Current density X
    std::unique_ptr<Variable> m_jyVar;  // Current density Y
    std::unique_ptr<Variable> m_jzVar;  // Current density Z

    // Poisson solver for electric potential
    std::unique_ptr<ConcatPoissonSolver2D> m_phiSolver;

    // Configuration references
    PhysicsConfig*       m_phyConfig;
    TimeAdvancingConfig* m_timeConfig;
    EnvironmentConfig*   m_envConfig;

    // Cached parameters
    std::vector<Domain2DUniform*> m_domains;
    std::unordered_map<Domain2DUniform*, std::unordered_map<LocationType, Domain2DUniform*>> m_adjacency;

    double m_Bx          = 0.0;
    double m_By          = 0.0;
    double m_Bz          = 0.0;
    double m_Ha          = 0.0;
    double m_Re          = 0.0;
    double m_dt          = 0.0;
    double m_lorentzCoef = 0.0;

    // Field storage
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_phiFieldStorage;
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_jxFieldStorage;
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_jyFieldStorage;
    std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> m_jzFieldStorage;

    // Field maps for quick access
    std::unordered_map<Domain2DUniform*, field2*> m_uFieldMap, m_vFieldMap;
    std::unordered_map<Domain2DUniform*, field2*> m_phiFieldMap, m_jxFieldMap, m_jyFieldMap, m_jzFieldMap;

    bool m_initialized = false;
};
