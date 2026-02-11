#pragma once

#include "base/domain/variable3d.h"
#include <map>
#include <string>
#include <vector>

#include <vtkDoubleArray.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>

class VTKWriter
{
public:
    VTKWriter() = default;

    /**
     * @brief Adds a scalar variable to be mapped to VTK Cell Data.
     */
    void add_scalar_as_cell_data(Variable3D* var);

    /**
     * @brief Synthesizes X, Y, and Z face-centered variables into a 3D vector.
     */
    void add_vector_as_cell_data(Variable3D* vx, Variable3D* vy, Variable3D* vz, const std::string& vector_name);

    /**
     * @brief Pre-calculates grid points and validates Tree/Geometry consistency.
     * Call this once before entering the time-loop.
     */
    void validate();

    /**
     * @brief Writes current field data to a .vtm file using cached geometry.
     */
    void write(const std::string& filename);

private:
    std::vector<Variable3D*> scalar_variables;

    struct VectorGroup
    {
        Variable3D *vx, *vy, *vz;
        std::string name;
    };
    std::vector<VectorGroup> vector_groups;

    // Cache structure to store pre-calculated geometry for each domain
    struct DomainCache
    {
        vtkSmartPointer<vtkPoints> points;
        int                        extent[6];
    };
    std::map<Domain3DUniform*, DomainCache> geometry_cache;
    bool                                    is_validated = false;

    double get_interpolated_value(Variable3D* var, Domain3DUniform* s, int i, int j, int k);
};