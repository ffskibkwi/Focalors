#pragma once

#include "base/domain/variable2d.h"
#include "shape2d.h"

#include <vector>
#include <unordered_map>

/**
 * @brief Velocity Fixer for Solid Points in 2D (Staggered Grid)
 *
 * This class identifies grid points inside immersed solid shapes and provides
 * a method to set their velocity to zero. This is used to prevent concentration
 * penetration into solids due to convective transport at high Reynolds numbers.
 *
 * Staggered grid layout:
 *   - u: XFace, located at (i, j+0.5), size nx*ny
 *   - v: YFace, located at (i+0.5, j), size nx*ny
 *
 * For each stored velocity component, we check if its physical location is inside
 * any shape. Only the field values are set to zero (not buffer values).
 *
 * Usage:
 *   1. Create the fixer with velocity pointers (u, v)
 *   2. Add shapes using add_shape()
 *   3. Call build() to identify solid points in all domains
 *   4. Call apply() before computing scalar RHS to zero out velocity in solids
 */
class SolidVelocityFixer2D
{
public:
    /**
     * @brief Constructor
     * @param u Velocity in x-direction (XFace)
     * @param v Velocity in y-direction (YFace)
     */
    SolidVelocityFixer2D(Variable2D* u, Variable2D* v);

    /**
     * @brief Add a shape to the solver
     * @param shape Pointer to the shape
     */
    void add_shape(Shape2D* shape);

    /**
     * @brief Build the solid point list for all shapes and domains
     * Must be called after adding all shapes and before applying
     */
    void build();

    /**
     * @brief Set velocity to zero for all solid points
     * Should be called before computing scalar field RHS
     */
    void apply();

    /**
     * @brief Get the number of solid u points for a domain
     */
    size_t get_num_solid_u_points(Domain2DUniform* domain) const;

    /**
     * @brief Get the number of solid v points for a domain
     */
    size_t get_num_solid_v_points(Domain2DUniform* domain) const;

    /**
     * @brief Check if a domain has solid points
     */
    bool has_solid_points(Domain2DUniform* domain) const;

    /**
     * @brief Get the solid u point indices for a domain
     */
    const std::vector<std::pair<int, int>>& get_solid_u_points(Domain2DUniform* domain) const;

    /**
     * @brief Get the solid v point indices for a domain
     */
    const std::vector<std::pair<int, int>>& get_solid_v_points(Domain2DUniform* domain) const;

private:
    /**
     * @brief Check if a grid point is inside any shape
     */
    bool is_inside_any_shape(double x, double y) const;

    Variable2D* u;
    Variable2D* v;

    std::vector<Shape2D*> shapes;

    // Solid points for u (XFace): domain -> vector of (i, j) indices
    std::unordered_map<Domain2DUniform*, std::vector<std::pair<int, int>>> solid_u_points_map;

    // Solid points for v (YFace): domain -> vector of (i, j) indices
    std::unordered_map<Domain2DUniform*, std::vector<std::pair<int, int>>> solid_v_points_map;
};
