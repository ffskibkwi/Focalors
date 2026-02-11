#include "vtk_writer.h"

#include <stdexcept>
#include <vtkCellData.h>
#include <vtkXMLMultiBlockDataWriter.h>

void VTKWriter::add_scalar_as_cell_data(Variable3D* var)
{
    if (var)
        scalar_variables.push_back(var);
    is_validated = false; // Reset validation if variables change
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

    // 1. Validate Consistency (Geometry/Tree check)
    Variable3D* ref = scalar_variables.empty() ? vector_groups[0].vx : scalar_variables[0];
    Geometry3D* geo = ref->geometry;

    auto check_geo = [&](Variable3D* v) {
        if (!v || v->geometry != geo)
            throw std::runtime_error("VTKWriter: Variable geometry/tree mismatch detected during validate().");
    };

    for (auto* v : scalar_variables)
        check_geo(v);
    for (auto& g : vector_groups)
    {
        check_geo(g.vx);
        check_geo(g.vy);
        check_geo(g.vz);
    }

    // 2. Pre-calculate Points for each domain
    geometry_cache.clear();
    for (auto* domain : geo->domains)
    {
        DomainCache cache;
        // Set extent
        cache.extent[0] = 0;
        cache.extent[1] = domain->nx;
        cache.extent[2] = 0;
        cache.extent[3] = domain->ny;
        cache.extent[4] = 0;
        cache.extent[5] = domain->nz;

        // Generate points
        cache.points = vtkSmartPointer<vtkPoints>::New();
        cache.points->SetDataTypeToDouble();
        cache.points->Allocate((domain->nx + 1) * (domain->ny + 1) * (domain->nz + 1));

        for (int k = 0; k <= domain->nz; ++k)
        {
            for (int j = 0; j <= domain->ny; ++j)
            {
                for (int i = 0; i <= domain->nx; ++i)
                {
                    cache.points->InsertNextPoint(domain->get_offset_x() + i * domain->get_hx(),
                                                  domain->get_offset_y() + j * domain->get_hy(),
                                                  domain->get_offset_z() + k * domain->get_hz());
                }
            }
        }
        geometry_cache[domain] = cache;
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

    auto multi_block = vtkSmartPointer<vtkMultiBlockDataSet>::New();
    int  block_idx   = 0;

    // Use the reference geometry from the cache keys
    for (auto& [domain, cache] : geometry_cache)
    {
        auto grid = vtkSmartPointer<vtkStructuredGrid>::New();
        grid->SetExtent(cache.extent);
        grid->SetPoints(cache.points);

        // Process Scalar Fields
        for (auto* var : scalar_variables)
        {
            auto arr = vtkSmartPointer<vtkDoubleArray>::New();
            arr->SetName(var->name.c_str());
            arr->SetNumberOfTuples(domain->nx * domain->ny * domain->nz);
            for (int k = 0; k < domain->nz; ++k)
                for (int j = 0; j < domain->ny; ++j)
                    for (int i = 0; i < domain->nx; ++i)
                        arr->SetTuple1(k * domain->ny * domain->nx + j * domain->nx + i,
                                       get_interpolated_value(var, domain, i, j, k));
            grid->GetCellData()->AddArray(arr);
        }

        // Process Vector Fields
        for (auto& group : vector_groups)
        {
            auto vec = vtkSmartPointer<vtkDoubleArray>::New();
            vec->SetName(group.name.c_str());
            vec->SetNumberOfComponents(3);
            vec->SetNumberOfTuples(domain->nx * domain->ny * domain->nz);
            for (int k = 0; k < domain->nz; ++k)
            {
                for (int j = 0; j < domain->ny; ++j)
                {
                    for (int i = 0; i < domain->nx; ++i)
                    {
                        double val_x = get_interpolated_value(group.vx, domain, i, j, k);
                        double val_y = get_interpolated_value(group.vy, domain, i, j, k);
                        double val_z = get_interpolated_value(group.vz, domain, i, j, k);
                        vec->SetTuple3(k * domain->ny * domain->nx + j * domain->nx + i, val_x, val_y, val_z);
                    }
                }
            }
            grid->GetCellData()->AddArray(vec);
        }
        multi_block->SetBlock(block_idx++, grid);
    }

    auto vtm_writer = vtkSmartPointer<vtkXMLMultiBlockDataWriter>::New();
    vtm_writer->SetFileName((filename + ".vtm").c_str());
    vtm_writer->SetInputData(multi_block);
    vtm_writer->SetDataModeToAppended();
    vtm_writer->SetCompressorTypeToNone();
    vtm_writer->Write();
}