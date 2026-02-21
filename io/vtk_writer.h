#pragma once

#include "base/domain/variable3d.h"
#include <map>
#include <string>
#include <vector>

#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

class VTKWriter
{
public:
    VTKWriter() = default;

    /**
     * Adds a scalar variable to be mapped to VTK Cell Data.
     */
    void add_scalar_as_cell_data(Variable3D* var);

    /**
     * Synthesizes X, Y, and Z face-centered variables into a 3D vector.
     */
    void add_vector_as_cell_data(Variable3D* vx, Variable3D* vy, Variable3D* vz, const std::string& vector_name);

    /**
     * Pre-calculates the global unstructured mesh topology and points.
     * This organizes multiple domains into a single contiguous grid.
     */
    void validate();

    /**
     * Writes current field data to a single .vtu file.
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

    /**
     * Stores global grid data to avoid re-generating topology every time-step.
     */
    vtkSmartPointer<vtkUnstructuredGrid> global_grid;

    /**
     * Maps each domain to its starting point ID in the global points array.
     */
    std::map<Domain3DUniform*, vtkIdType> domain_point_offsets;

    bool is_validated = false;

    double get_interpolated_value(Variable3D* var, Domain3DUniform* s, int i, int j, int k);
};