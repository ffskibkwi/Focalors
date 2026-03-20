#include "boundary_3d_utils.h"
#include <stdexcept>
#include <utility>

void assign_x(field3& f, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = val(j, k);
            }
        }
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = val_default;
            }
        }
    }
}

void assign_y(field3& f, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = val(i, k);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = val_default;
            }
        }
    }
}

void assign_z(field3& f, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = val(i, j);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = val_default;
            }
        }
    }
}

void copy_x(field3& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            f(dest, j, k) = f(src, j, k);
        }
    }
}

void copy_y(field3& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            f(i, dest, k) = f(i, src, k);
        }
    }
}

void copy_z(field3& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            f(i, j, dest) = f(i, j, src);
        }
    }
}

void mirror_x(field3& f, int src, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = 2.0 * val(j, k) - f(src, j, k);
            }
        }
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(dest, j, k) = 2.0 * val_default - f(src, j, k);
            }
        }
    }
}

void mirror_y(field3& f, int src, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = 2.0 * val(i, k) - f(i, src, k);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                f(i, dest, k) = 2.0 * val_default - f(i, src, k);
            }
        }
    }
}

void mirror_z(field3& f, int src, int dest, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = 2.0 * val(i, j) - f(i, j, src);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                f(i, j, dest) = 2.0 * val_default - f(i, j, src);
            }
        }
    }
}

void assign_val_to_buffer(field2& buffer, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < val.get_nx(); i++)
        {
            for (int j = 0; j < val.get_ny(); j++)
            {
                buffer(i, j) = val(i, j);
            }
        }
    }
    else
    {
        for (int i = 0; i < buffer.get_nx(); i++)
        {
            for (int j = 0; j < buffer.get_ny(); j++)
            {
                buffer(i, j) = val_default;
            }
        }
    }
}

void copy_x_to_buffer(field2& buffer, field3& f, int src)
{
    for (int j = 0; j < f.get_ny(); j++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            buffer(j, k) = f(src, j, k);
        }
    }
}

void copy_y_to_buffer(field2& buffer, field3& f, int src)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int k = 0; k < f.get_nz(); k++)
        {
            buffer(i, k) = f(i, src, k);
        }
    }
}

void copy_z_to_buffer(field2& buffer, field3& f, int src)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            buffer(i, j) = f(i, j, src);
        }
    }
}

void copy_x_to_buffer(double* buffer, field3& f, int src_y, int src_z)
{
    for (int i = 0; i < f.get_nx(); i++)
    {
        buffer[i] = f(i, src_y, src_z);
    }
}

void copy_y_to_buffer(double* buffer, field3& f, int src_x, int src_z)
{
    for (int j = 0; j < f.get_ny(); j++)
    {
        buffer[j] = f(src_x, j, src_z);
    }
}

void copy_z_to_buffer(double* buffer, field3& f, int src_x, int src_y)
{
    for (int k = 0; k < f.get_nz(); k++)
    {
        buffer[k] = f(src_x, src_y, k);
    }
}

void mirror_x_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(j, k) = 2.0 * val(j, k) - f(src, j, k);
            }
        }
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(j, k) = 2.0 * val_default - f(src, j, k);
            }
        }
    }
}

void mirror_y_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(i, k) = 2.0 * val(i, k) - f(i, src, k);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int k = 0; k < f.get_nz(); k++)
            {
                buffer(i, k) = 2.0 * val_default - f(i, src, k);
            }
        }
    }
}

void mirror_z_to_buffer(field2& buffer, field3& f, int src, field2* val_ptr, double val_default)
{
    if (val_ptr != nullptr)
    {
        field2& val = *val_ptr;
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                buffer(i, j) = 2.0 * val(i, j) - f(i, j, src);
            }
        }
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
        {
            for (int j = 0; j < f.get_ny(); j++)
            {
                buffer(i, j) = 2.0 * val_default - f(i, j, src);
            }
        }
    }
}

void swap_field_data(field3& a, field3& b)
{
    if (a.get_size_n() != b.get_size_n())
        throw std::runtime_error("swap_field_data(field3): size mismatch");
    std::swap(a.value, b.value);
}

bool isAllNeumannBoundary(const Variable3D& var)
{
    if (var.geometry == nullptr)
        return false;

    for (auto* domain : var.geometry->domains)
    {
        const auto domIt = var.boundary_type_map.find(domain);
        if (domIt == var.boundary_type_map.end())
            return false;

        const auto& locMap = domIt->second;
        for (const auto loc : kBoundaryLocations3D)
        {
            PDEBoundaryType type = PDEBoundaryType::Null;
            if (const auto locIt = locMap.find(loc); locIt != locMap.end())
                type = locIt->second;

            if (type == PDEBoundaryType::Adjacented)
                continue;
            if (type == PDEBoundaryType::Neumann || type == PDEBoundaryType::Periodic)
                continue;
            return false;
        }
    }

    return true;
}

double normalizeRhsForNeumannBc(const Variable3D&                                    var,
                                const std::vector<Domain3DUniform*>&                 domains,
                                const std::unordered_map<Domain3DUniform*, field3*>& fieldMap)
{
    auto requireField = [&](Domain3DUniform* domain) -> field3& {
        const auto it = fieldMap.find(domain);
        if (it == fieldMap.end() || it->second == nullptr)
            throw std::runtime_error("normalizeRhsForNeumannBc(3D): fieldMap missing entry for domain");
        return *it->second;
    };

    double totalSum  = 0.0;
    size_t totalSize = 0;
    for (auto* domain : domains)
    {
        field3& f = requireField(domain);
        totalSum += f.sum();
        totalSize += static_cast<size_t>(f.get_size_n());
    }

    if (totalSize == 0)
        return 0.0;

    auto sumBoundaryBuffer = [](const field2& boundaryBuffer) -> double {
        double sum = 0.0;
        for (int i = 0; i < boundaryBuffer.get_nx(); ++i)
        {
            for (int j = 0; j < boundaryBuffer.get_ny(); ++j)
                sum += boundaryBuffer(i, j);
        }
        return sum;
    };

    // For pure Neumann Poisson problems, the solvability condition is:
    //   sum(RHS_with_bc) == 0 (discrete orthogonality to constant nullspace).
    // RHS_with_bc means the RHS after ConcatPoissonSolver3D::boundary_assembly()
    // adds Neumann flux terms. We compensate this by including the same boundary
    // contributions in the global mean.
    double bcSum = 0.0;
    for (auto* domain : domains)
    {
        const double hx = domain->get_hx();
        const double hy = domain->get_hy();
        const double hz = domain->get_hz();

        const auto& typeMap = var.boundary_type_map.at(domain);

        const auto  domBufIt  = var.buffer_map.find(domain);
        const auto* bufMapPtr = domBufIt == var.buffer_map.end() ? nullptr : &domBufIt->second;

        auto requireBoundaryBuffer = [&](LocationType loc) -> const field2& {
            if (bufMapPtr == nullptr)
                throw std::runtime_error("normalizeRhsForNeumannBc(3D): buffer_map missing for domain");
            const auto it = bufMapPtr->find(loc);
            if (it == bufMapPtr->end() || it->second == nullptr)
                throw std::runtime_error("normalizeRhsForNeumannBc(3D): buffer_map missing entry for Neumann bc");
            return *it->second;
        };

        const auto xnegType = typeMap.at(LocationType::XNegative);
        const auto xposType = typeMap.at(LocationType::XPositive);
        const auto ynegType = typeMap.at(LocationType::YNegative);
        const auto yposType = typeMap.at(LocationType::YPositive);
        const auto znegType = typeMap.at(LocationType::ZNegative);
        const auto zposType = typeMap.at(LocationType::ZPositive);

        // Neumann contributions follow ConcatPoissonSolver3D::boundary_assembly() sign conventions.
        if (xnegType == PDEBoundaryType::Neumann)
            bcSum += sumBoundaryBuffer(requireBoundaryBuffer(LocationType::XNegative)) / hx;
        if (xposType == PDEBoundaryType::Neumann)
            bcSum -= sumBoundaryBuffer(requireBoundaryBuffer(LocationType::XPositive)) / hx;
        if (ynegType == PDEBoundaryType::Neumann)
            bcSum += sumBoundaryBuffer(requireBoundaryBuffer(LocationType::YNegative)) / hy;
        if (yposType == PDEBoundaryType::Neumann)
            bcSum -= sumBoundaryBuffer(requireBoundaryBuffer(LocationType::YPositive)) / hy;
        if (znegType == PDEBoundaryType::Neumann)
            bcSum += sumBoundaryBuffer(requireBoundaryBuffer(LocationType::ZNegative)) / hz;
        if (zposType == PDEBoundaryType::Neumann)
            bcSum -= sumBoundaryBuffer(requireBoundaryBuffer(LocationType::ZPositive)) / hz;
    }

    const double mean = (totalSum + bcSum) / static_cast<double>(totalSize);

    for (auto* domain : domains)
    {
        field3& f  = requireField(domain);
        int     nx = f.get_nx();
        int     ny = f.get_ny();
        int     nz = f.get_nz();
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int k = 0; k < nz; ++k)
                    f(i, j, k) -= mean;
            }
        }
    }

    return mean;
}
