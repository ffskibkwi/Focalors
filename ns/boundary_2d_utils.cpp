#include "boundary_2d_utils.h"
#include <stdexcept>
#include <utility>

void assign_x(field2& f, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = val_ptr[j];
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = val_default;
    }
}

void assign_y(field2& f, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = val_ptr[i];
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = val_default;
    }
}

void copy_x(field2& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
        f(dest, j) = f(src, j);
}

void copy_y(field2& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
        f(i, dest) = f(i, src);
}

void mirror_x(field2& f, int src, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = 2.0 * val_ptr[j] - f(src, j);
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
            f(dest, j) = 2.0 * val_default - f(src, j);
    }
}

void mirror_y(field2& f, int src, int dest, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = 2.0 * val_ptr[i] - f(i, src);
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
            f(i, dest) = 2.0 * val_default - f(i, src);
    }
}

void assign_val_to_buffer(double* buffer, int length, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < length; i++)
            buffer[i] = val_ptr[i];
    }
    else
    {
        for (int i = 0; i < length; i++)
            buffer[i] = val_default;
    }
}

void copy_x_to_buffer(double* buffer, field2& f, int src)
{
    for (int j = 0; j < f.get_ny(); j++)
        buffer[j] = f(src, j);
}

void copy_y_to_buffer(double* buffer, field2& f, int src)
{
    for (int i = 0; i < f.get_nx(); i++)
        buffer[i] = f(i, src);
}

void copy_src_x_to_buffer_x(field2& buffer, field2& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
        buffer(dest, j) = f(src, j);
}

void copy_src_y_to_buffer_y(field2& buffer, field2& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
        buffer(i, dest) = f(i, src);
}

void copy_src_x_to_buffer_y(field2& buffer, field2& f, int src, int dest)
{
    for (int j = 0; j < f.get_ny(); j++)
        buffer(j, dest) = f(src, j);
}

void copy_src_y_to_buffer_x(field2& buffer, field2& f, int src, int dest)
{
    for (int i = 0; i < f.get_nx(); i++)
        buffer(dest, i) = f(i, src);
}

void mirror_x_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int j = 0; j < f.get_ny(); j++)
            buffer[j] = 2.0 * val_ptr[j] - f(src, j);
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
            buffer[j] = 2.0 * val_default - f(src, j);
    }
}

void mirror_y_to_buffer(double* buffer, field2& f, int src, double* val_ptr, double val_default)
{
    if (val_ptr)
    {
        for (int i = 0; i < f.get_nx(); i++)
            buffer[i] = 2.0 * val_ptr[i] - f(i, src);
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
            buffer[i] = 2.0 * val_default - f(i, src);
    }
}

void neumann_x_to_buffer(double* buffer, field2& f, int src, double* q_ptr, double q_default, double hx, double sign)
{
    if (q_ptr)
    {
        for (int j = 0; j < f.get_ny(); j++)
            buffer[j] = f(src, j) + sign * q_ptr[j] * hx;
    }
    else
    {
        for (int j = 0; j < f.get_ny(); j++)
            buffer[j] = f(src, j) + sign * q_default * hx;
    }
}

void neumann_y_to_buffer(double* buffer, field2& f, int src, double* q_ptr, double q_default, double hy, double sign)
{
    if (q_ptr)
    {
        for (int i = 0; i < f.get_nx(); i++)
            buffer[i] = f(i, src) + sign * q_ptr[i] * hy;
    }
    else
    {
        for (int i = 0; i < f.get_nx(); i++)
            buffer[i] = f(i, src) + sign * q_default * hy;
    }
}

// Swap underlying data pointers for two same-typed fields.
// Requires identical storage size; only swaps the data pointer without touching shape/metadata.
void swap_field_data(field2& a, field2& b)
{
    if (a.get_size_n() != b.get_size_n())
        throw std::runtime_error("swap_field_data(field2): size mismatch");
    std::swap(a.value, b.value);
}

bool isAllNeumannBoundary(const Variable& var)
{
    if (var.geometry == nullptr)
        return false;

    const LocationType locs[4] = {LocationType::Left, LocationType::Right, LocationType::Down, LocationType::Up};

    for (auto* domain : var.geometry->domains)
    {
        const auto dom_it = var.boundary_type_map.find(domain);
        if (dom_it == var.boundary_type_map.end())
            return false;

        const auto& loc_map = dom_it->second;
        for (const auto loc : locs)
        {
            PDEBoundaryType type = PDEBoundaryType::Null;
            if (const auto loc_it = loc_map.find(loc); loc_it != loc_map.end())
                type = loc_it->second;

            if (type == PDEBoundaryType::Adjacented)
                continue;
            if (type == PDEBoundaryType::Neumann || type == PDEBoundaryType::Periodic)
                continue;
            return false;
        }
    }

    return true;
}

double normalizeRhsForNeumannBc(const Variable&                                      var,
                                const std::vector<Domain2DUniform*>&                 domains,
                                const std::unordered_map<Domain2DUniform*, field2*>& fieldMap)
{
    auto requireField = [&](Domain2DUniform* domain) -> field2& {
        const auto it = fieldMap.find(domain);
        if (it == fieldMap.end() || it->second == nullptr)
            throw std::runtime_error("normalizeRhsForNeumannBc: fieldMap missing entry for domain");
        return *it->second;
    };

    double total_sum  = 0.0;
    size_t total_size = 0;
    for (auto* domain : domains)
    {
        field2& f = requireField(domain);
        total_sum += f.sum();
        total_size += static_cast<size_t>(f.get_size_n());
    }

    if (total_size == 0)
        return 0.0;

    // For pure Neumann Poisson problems, the solvability condition is:
    //   sum(RHS_with_bc) == 0 (discrete orthogonality to constant nullspace).
    // RHS_with_bc means the RHS after PoissonSolver2D::boundary_assembly() adds Neumann flux terms.
    // We compensate this by including the same boundary contributions in the global mean.
    double bc_sum = 0.0;
    for (auto* domain : domains)
    {
        const int    nx = domain->get_nx();
        const int    ny = domain->get_ny();
        const double hx = domain->get_hx();
        const double hy = domain->get_hy();

        const auto& type_map = var.boundary_type_map.at(domain);

        const auto  dom_has_it  = var.has_boundary_value_map.find(domain);
        const auto* has_map_ptr = dom_has_it == var.has_boundary_value_map.end() ? nullptr : &dom_has_it->second;

        const auto  dom_val_it  = var.boundary_value_map.find(domain);
        const auto* val_map_ptr = dom_val_it == var.boundary_value_map.end() ? nullptr : &dom_val_it->second;

        auto hasBoundaryValue = [&](LocationType loc) -> bool {
            if (has_map_ptr == nullptr)
                return false;
            const auto it = has_map_ptr->find(loc);
            return it != has_map_ptr->end() && it->second;
        };

        auto requireBoundaryValuePtr = [&](LocationType loc) -> const double* {
            if (val_map_ptr == nullptr)
                throw std::runtime_error("normalizeRhsForNeumannBc: boundary_value_map missing for domain");
            const auto it = val_map_ptr->find(loc);
            if (it == val_map_ptr->end() || it->second == nullptr)
                throw std::runtime_error("normalizeRhsForNeumannBc: boundary_value_map missing entry for Neumann bc");
            return it->second;
        };

        auto sum_or_default = [](const double* arr, int n, double def_val) -> double {
            if (arr == nullptr)
                return def_val * static_cast<double>(n);
            double s = 0.0;
            for (int i = 0; i < n; ++i)
                s += arr[i];
            return s;
        };

        const auto left_type  = type_map.at(LocationType::Left);
        const auto right_type = type_map.at(LocationType::Right);
        const auto down_type  = type_map.at(LocationType::Down);
        const auto up_type    = type_map.at(LocationType::Up);

        // Neumann contributions follow PoissonSolver2D::boundary_assembly() sign conventions.
        if (left_type == PDEBoundaryType::Neumann && hasBoundaryValue(LocationType::Left))
        {
            bc_sum += sum_or_default(requireBoundaryValuePtr(LocationType::Left), ny, 0.0) / hx;
        }
        if (right_type == PDEBoundaryType::Neumann && hasBoundaryValue(LocationType::Right))
        {
            bc_sum -= sum_or_default(requireBoundaryValuePtr(LocationType::Right), ny, 0.0) / hx;
        }
        if (down_type == PDEBoundaryType::Neumann && hasBoundaryValue(LocationType::Down))
        {
            bc_sum += sum_or_default(requireBoundaryValuePtr(LocationType::Down), nx, 0.0) / hy;
        }
        if (up_type == PDEBoundaryType::Neumann && hasBoundaryValue(LocationType::Up))
        {
            bc_sum -= sum_or_default(requireBoundaryValuePtr(LocationType::Up), nx, 0.0) / hy;
        }
    }

    const double mean = (total_sum + bc_sum) / static_cast<double>(total_size);

    for (auto* domain : domains)
    {
        field2& f  = requireField(domain);
        int     nx = f.get_nx();
        int     ny = f.get_ny();
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                f(i, j) -= mean;
    }

    return mean;
}

double get_u_with_boundary(int           i,
                           int           j,
                           int           nx,
                           int           ny,
                           const field2& u,
                           double*       u_left_buffer,
                           double*       u_right_buffer,
                           double*       u_down_buffer,
                           double*       u_up_buffer,
                           double        right_down_corner_value)
{
    if (i >= 0 && i < nx && j >= 0 && j < ny)
        return u(i, j);
    if (j < 0)
        return (i >= nx) ? right_down_corner_value : u_down_buffer[i];
    if (j >= ny)
        return u_up_buffer[i];
    if (i >= nx)
        return u_right_buffer[j];
    // i < 0
    return u_left_buffer[j];
}

double get_v_with_boundary(int           i,
                           int           j,
                           int           nx,
                           int           ny,
                           const field2& v,
                           double*       v_left_buffer,
                           double*       v_right_buffer,
                           double*       v_down_buffer,
                           double*       v_up_buffer,
                           double        left_up_corner_value)
{
    if (i >= 0 && i < nx && j >= 0 && j < ny)
        return v(i, j);
    if (i < 0)
        return (j >= ny) ? left_up_corner_value : v_left_buffer[j];
    if (i >= nx)
        return v_right_buffer[j];
    if (j >= ny)
        return v_up_buffer[i];
    // j < 0
    return v_down_buffer[i];
}

double get_scalar_with_boundary(int             i,
                                int             j,
                                int             nx,
                                int             ny,
                                const field2&   f,
                                double*         left_buffer,
                                double*         down_buffer,
                                double          hx,
                                double          hy,
                                PDEBoundaryType right_bc_type,
                                double*         right_bc_val,
                                double          right_bc_default,
                                PDEBoundaryType up_bc_type,
                                double*         up_bc_val,
                                double          up_bc_default)
{
    if (i >= 0 && i < nx && j >= 0 && j < ny)
        return f(i, j);

    auto clamp_i = [nx](int idx) -> int {
        if (idx < 0)
            return 0;
        if (idx >= nx)
            return nx - 1;
        return idx;
    };
    auto clamp_j = [ny](int idx) -> int {
        if (idx < 0)
            return 0;
        if (idx >= ny)
            return ny - 1;
        return idx;
    };

    if (i < 0)
    {
        const int jj = clamp_j(j);
        return left_buffer ? left_buffer[jj] : f(0, jj);
    }
    if (j < 0)
    {
        const int ii = clamp_i(i);
        return down_buffer ? down_buffer[ii] : f(ii, 0);
    }

    if (i >= nx)
    {
        const int jj = clamp_j(j);
        if (right_bc_type == PDEBoundaryType::Dirichlet)
        {
            const double g = right_bc_val ? right_bc_val[jj] : right_bc_default;
            return 2.0 * g - f(nx - 1, jj);
        }
        if (right_bc_type == PDEBoundaryType::Neumann)
        {
            const double q = right_bc_val ? right_bc_val[jj] : right_bc_default;
            return f(nx - 1, jj) + q * hx;
        }
        return f(nx - 1, jj);
    }

    // j >= ny
    const int ii = clamp_i(i);
    if (up_bc_type == PDEBoundaryType::Dirichlet)
    {
        const double g = up_bc_val ? up_bc_val[ii] : up_bc_default;
        return 2.0 * g - f(ii, ny - 1);
    }
    if (up_bc_type == PDEBoundaryType::Neumann)
    {
        const double q = up_bc_val ? up_bc_val[ii] : up_bc_default;
        return f(ii, ny - 1) + q * hy;
    }
    return f(ii, ny - 1);
}
