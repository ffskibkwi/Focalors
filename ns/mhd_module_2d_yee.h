#pragma once

#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
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
 * @brief MHD module for 2D quasi-static/induced electric potential method (Yee Grid)
 *
 * Variable2D Layout (Yee):
 * - phi: Node (i, j), stored as inner field with 4-side buffers
 * - u:   X-Face (i+1/2, j)
 * - v:   Y-Face (i, j+1/2)
 *
 * Derived Fields:
 * - Jx:  Y-Face (collocated with v)
 * - Jy:  X-Face (collocated with u)
 * - Jz:  Center (optional for 2D)
 */
class MHDModule2DYee
{
public:
    MHDModule2DYee(Variable2D*          in_u_var,
                   Variable2D*          in_v_var,
                   PhysicsConfig*       in_phy_config,
                   TimeAdvancingConfig* in_time_config,
                   EnvironmentConfig*   in_env_config = nullptr);
    ~MHDModule2DYee();

    void init(Variable2D* phi_var = nullptr);

    void solveElectricPotential();
    void updateCurrentDensity();
    void applyLorentzForce();

private:
    void buffer_update_phi();
    void buffer_update_j();

    // Reference to NS velocity variable2ds
    Variable2D* m_uVar;
    Variable2D* m_vVar;

    // MHD-specific variable2ds
    Variable2D*                 m_phiVar   = nullptr; // Electric potential (Node, non-owning)
    std::unique_ptr<Variable2D> m_phiOwned = nullptr; // Owned phi when created internally
    std::unique_ptr<Variable2D> m_jxVar;              // Current density X (Y-Face)
    std::unique_ptr<Variable2D> m_jyVar;              // Current density Y (X-Face)
    std::unique_ptr<Variable2D> m_jzVar;              // Current density Z (Center)

    // Poisson solver for electric potential
    std::unique_ptr<ConcatPoissonSolver2D> m_phiSolver;

    // Configuration references
    PhysicsConfig*       m_phyConfig;
    TimeAdvancingConfig* m_timeConfig;
    EnvironmentConfig*   m_envConfig;

    // Cached parameters
    std::vector<Domain2DUniform*>                                                            m_domains;
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
