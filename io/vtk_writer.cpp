#include "vtk_writer.h"

#include "common.h"

#include <stdexcept>
#include <vtkCellData.h>
#include <vtkHexahedron.h>
#include <vtkXMLUnstructuredGridWriter.h>

void VTKWriter::add_scalar_as_cell_data(Variable3D* var)
{
    if (var)
        scalar_variables.push_back(var);
    is_validated = false;
}

void VTKWriter::add_vector_as_cell_data(Variable3D* vx, Variable3D* vy, Variable3D* vz, const std::string& vector_name)
{
    if (vx && vy && vz)
    {
        vector_groups.push_back({vx, vy, vz, vector_name});
    }
    is_validated = false;
}

void VTKWriter::validate()
{
    if (scalar_variables.empty() && vector_groups.empty())
        return;

    Variable3D* ref = scalar_variables.empty() ? vector_groups[0].vx : scalar_variables[0];
    Geometry3D* geo = ref->geometry;

    global_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToDouble();

    domain_point_offsets.clear();

    /**
     * Generate all points for all domains in a single global points array.
     */
    for (auto* domain : geo->domains)
    {
        domain_point_offsets[domain] = points->GetNumberOfPoints();
        for (int k = 0; k <= domain->nz; ++k)
        {
            for (int j = 0; j <= domain->ny; ++j)
            {
                for (int i = 0; i <= domain->nx; ++i)
                {
                    points->InsertNextPoint(domain->get_offset_x() + i * domain->get_hx(),
                                            domain->get_offset_y() + j * domain->get_hy(),
                                            domain->get_offset_z() + k * domain->get_hz());
                }
            }
        }
    }
    global_grid->SetPoints(points);

    /**
     * Construct hexahedral cells for every domain using global point IDs.
     */
    for (auto* domain : geo->domains)
    {
        vtkIdType offset = domain_point_offsets[domain];
        int       nxp    = domain->nx + 1;
        int       nyp    = domain->ny + 1;

        for (int k = 0; k < domain->nz; ++k)
        {
            for (int j = 0; j < domain->ny; ++j)
            {
                for (int i = 0; i < domain->nx; ++i)
                {
                    vtkIdType ids[8];
                    /**
                     * Map local grid indices to global point IDs based on VTK Hexahedron convention.
                     */
                    ids[0] = offset + (k * nyp * nxp + j * nxp + i);
                    ids[1] = offset + (k * nyp * nxp + j * nxp + (i + 1));
                    ids[2] = offset + (k * nyp * nxp + (j + 1) * nxp + (i + 1));
                    ids[3] = offset + (k * nyp * nxp + (j + 1) * nxp + i);
                    ids[4] = offset + ((k + 1) * nyp * nxp + j * nxp + i);
                    ids[5] = offset + ((k + 1) * nyp * nxp + j * nxp + (i + 1));
                    ids[6] = offset + ((k + 1) * nyp * nxp + (j + 1) * nxp + (i + 1));
                    ids[7] = offset + ((k + 1) * nyp * nxp + (j + 1) * nxp + i);

                    global_grid->InsertNextCell(VTK_HEXAHEDRON, 8, ids);
                }
            }
        }
    }

    is_validated = true;
}

double VTKWriter::get_interpolated_value(Variable3D* var, Domain3DUniform* s, int i, int j, int k)
{
    field3* f = var->field_map.at(s);
    switch (var->position_type)
    {
        case VariablePositionType::Center:
            return (*f)(i, j, k);
        case VariablePositionType::XFace: {
            double v_curr = (*f)(i, j, k);
            double v_next =
                (i < s->nx - 1) ? (*f)(i + 1, j, k) : var->buffer_map.at(s).at(LocationType::Right)->operator()(j, k);
            return 0.5 * (v_curr + v_next);
        }
        case VariablePositionType::YFace: {
            double v_curr = (*f)(i, j, k);
            double v_next =
                (j < s->ny - 1) ? (*f)(i, j + 1, k) : var->buffer_map.at(s).at(LocationType::Back)->operator()(i, k);
            return 0.5 * (v_curr + v_next);
        }
        case VariablePositionType::ZFace: {
            double v_curr = (*f)(i, j, k);
            double v_next =
                (k < s->nz - 1) ? (*f)(i, j, k + 1) : var->buffer_map.at(s).at(LocationType::Up)->operator()(i, j);
            return 0.5 * (v_curr + v_next);
        }
        default:
            return 0.0;
    }
}

void VTKWriter::write(const std::string& filename)
{
    if (!is_validated)
    {
        throw std::runtime_error("VTKWriter: You must call validate() before write().");
    }

    fs::path path(filename);
    fs::path dir = path.parent_path();
    IO::create_directory(dir);

    /**
     * Clear existing cell data from previous time steps before adding new values.
     */
    global_grid->GetCellData()->Initialize();

    /**
     * Populate scalar arrays across all combined domains.
     */
    for (auto* var : scalar_variables)
    {
        auto arr = vtkSmartPointer<vtkDoubleArray>::New();
        arr->SetName(var->name.c_str());
        for (auto const& [domain, offset] : domain_point_offsets)
        {
            for (int k = 0; k < domain->nz; ++k)
                for (int j = 0; j < domain->ny; ++j)
                    for (int i = 0; i < domain->nx; ++i)
                        arr->InsertNextValue(get_interpolated_value(var, domain, i, j, k));
        }
        global_grid->GetCellData()->AddArray(arr);
    }

    /**
     * Populate vector arrays across all combined domains.
     */
    for (auto& group : vector_groups)
    {
        auto vec = vtkSmartPointer<vtkDoubleArray>::New();
        vec->SetName(group.name.c_str());
        vec->SetNumberOfComponents(3);
        for (auto const& [domain, offset] : domain_point_offsets)
        {
            for (int k = 0; k < domain->nz; ++k)
            {
                for (int j = 0; j < domain->ny; ++j)
                {
                    for (int i = 0; i < domain->nx; ++i)
                    {
                        double val_x = get_interpolated_value(group.vx, domain, i, j, k);
                        double val_y = get_interpolated_value(group.vy, domain, i, j, k);
                        double val_z = get_interpolated_value(group.vz, domain, i, j, k);
                        vec->InsertNextTuple3(val_x, val_y, val_z);
                    }
                }
            }
        }
        global_grid->GetCellData()->AddArray(vec);
    }

    /**
     * Write the consolidated grid to a .vtu file.
     */
    auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetFileName((filename + ".vtu").c_str());
    writer->SetInputData(global_grid);
    writer->SetDataModeToBinary();
    writer->SetCompressorTypeToNone();
    writer->Write();
}