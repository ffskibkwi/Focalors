#include "ib_solver_2d_mirror_point.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

IBSolver2D_MirrorPoint::IBSolver2D_MirrorPoint(Variable2D* in_var, PDEBoundaryType in_boundary_type, double in_boundary_value)
    : var(in_var)
    , boundary_type(in_boundary_type)
    , boundary_value(in_boundary_value)
{
    if (var == nullptr)
        throw std::runtime_error("IBSolver2D_MirrorPoint: variable is nullptr");
    if (var->geometry == nullptr)
        throw std::runtime_error("IBSolver2D_MirrorPoint: variable has no geometry");
}

void IBSolver2D_MirrorPoint::add_shape(Shape2D* shape)
{
    shapes.push_back(shape);
}

void IBSolver2D_MirrorPoint::build()
{
    if (shapes.empty())
        throw std::runtime_error("IBSolver2D_MirrorPoint: no shapes added");

    auto& geometry = *var->geometry;

    // First, build interior point map for each domain using unordered_map for deduplication
    // Key: (domain, i, j) -> value: bool (true if interior point)
    // We'll process domain by domain, but template can cross domain boundaries

    for (auto* domain : geometry.domains)
    {
        auto* field = var->field_map.at(domain);
        int   nx    = field->get_nx();
        int   ny    = field->get_ny();

        // Use unordered_map for deduplication: key = (i << 32) | j
        std::unordered_map<long long, std::pair<int, int>> interior_map;

        // Iterate over all grid points in this domain
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                auto [x, y] = get_physical_location(domain, i, j);

                // Skip if this point is inside any shape
                if (is_inside_any_shape(x, y))
                    continue;

                // This is an exterior point, check its template neighbors
                // Template can cross domain boundaries, so we need to check neighbor domains
                // MUST check ALL template points - cannot skip once one interior point is found
                for (int dj = -template_radius; dj <= template_radius; ++dj)
                {
                    for (int di = -template_radius; di <= template_radius; ++di)
                    {
                        if (di == 0 && dj == 0)
                            continue;

                        int ni = i + di;
                        int nj = j + dj;

                        // Determine which domain the template point belongs to
                        Domain2DUniform* target_domain = domain;
                        int              ti = ni;
                        int              tj = nj;

                        // Check if template point is within current domain
                        if (ni < 0)
                        {
                            // Need to check XNegative neighbor
                            auto adj_it = geometry.adjacency.find(domain);
                            if (adj_it != geometry.adjacency.end())
                            {
                                auto neg_it = adj_it->second.find(LocationType::XNegative);
                                if (neg_it != adj_it->second.end())
                                {
                                    target_domain = neg_it->second;
                                    // Adjust index for neighbor domain
                                    auto* neighbor_field = var->field_map.at(target_domain);
                                    int   neighbor_nx   = neighbor_field->get_nx();
                                    ti = ni + neighbor_nx;
                                }
                            }
                        }
                        else if (ni >= nx)
                        {
                            // Need to check XPositive neighbor
                            auto adj_it = geometry.adjacency.find(domain);
                            if (adj_it != geometry.adjacency.end())
                            {
                                auto pos_it = adj_it->second.find(LocationType::XPositive);
                                if (pos_it != adj_it->second.end())
                                {
                                    target_domain = pos_it->second;
                                    ti = ni - nx;
                                }
                            }
                        }

                        if (nj < 0)
                        {
                            // Need to check YNegative neighbor
                            auto adj_it = geometry.adjacency.find(domain);
                            if (adj_it != geometry.adjacency.end())
                            {
                                auto neg_it = adj_it->second.find(LocationType::YNegative);
                                if (neg_it != adj_it->second.end())
                                {
                                    target_domain = neg_it->second;
                                    auto* neighbor_field = var->field_map.at(target_domain);
                                    int   neighbor_ny   = neighbor_field->get_ny();
                                    tj = nj + neighbor_ny;
                                }
                            }
                        }
                        else if (nj >= ny)
                        {
                            // Need to check YPositive neighbor
                            auto adj_it = geometry.adjacency.find(domain);
                            if (adj_it != geometry.adjacency.end())
                            {
                                auto pos_it = adj_it->second.find(LocationType::YPositive);
                                if (pos_it != adj_it->second.end())
                                {
                                    target_domain = pos_it->second;
                                    tj = nj - ny;
                                }
                            }
                        }

                        // Skip if target domain doesn't have field (shouldn't happen for valid geometry)
                        if (var->field_map.find(target_domain) == var->field_map.end())
                            continue;

                        // Get physical location of template point in target domain
                        auto [tx, ty] = get_physical_location(target_domain, ti, tj);

                        // If template point is inside any shape, mark current point as interior
                        // Must check ALL template points - add to set if found, continue checking others
                        if (is_inside_any_shape(tx, ty))
                        {
                            long long key = (static_cast<long long>(i) << 32) | static_cast<unsigned int>(j);
                            interior_map[key] = {i, j};
                            // Continue checking other template points - we need to check ALL of them
                        }
                    }
                }
            }
        }

        // Convert map to vector
        auto& interior_points = interior_points_map[domain];
        interior_points.reserve(interior_map.size());
        for (auto& kv : interior_map)
        {
            interior_points.emplace_back(kv.second.first, kv.second.second);
        }

        // Build mirror info for each interior point
        auto& mirror_infos = mirror_info_map[domain];
        mirror_infos.reserve(interior_points.size());

        for (const auto& [i, j] : interior_points)
        {
            auto [x, y] = get_physical_location(domain, i, j);

            double mirror_x, mirror_y, delta_l;
            find_mirror_point(x, y, mirror_x, mirror_y, delta_l);

            mirror_infos.push_back({i, j, mirror_x, mirror_y, delta_l});
        }
    }
}

void IBSolver2D_MirrorPoint::apply()
{
    for (auto* domain : var->geometry->domains)
    {
        auto* field       = var->field_map.at(domain);
        auto& mirror_list = mirror_info_map[domain];

        for (const auto& mirror : mirror_list)
        {
            int   i       = mirror.i;
            int   j       = mirror.j;
            double phi    = (*field)(i, j);

            // Interpolate mirror point value
            double phi_mirror = interpolate_mirror_value(domain, mirror.mirror_x, mirror.mirror_y);

            double new_phi;

            if (boundary_type == PDEBoundaryType::Dirichlet)
            {
                // Dirichlet: phi + phi_mirror = 2 * BC
                new_phi = 2.0 * boundary_value - phi_mirror;
            }
            else if (boundary_type == PDEBoundaryType::Neumann)
            {
                // Neumann: (phi_mirror - phi) / delta_l = BC
                // => phi_mirror = phi + BC * delta_l
                // => phi = phi_mirror - BC * delta_l
                new_phi = phi_mirror - boundary_value * mirror.delta_l;
            }
            else
            {
                throw std::runtime_error("IBSolver2D_MirrorPoint: unsupported boundary type");
            }

            (*field)(i, j) = new_phi;
        }
    }
}

size_t IBSolver2D_MirrorPoint::get_num_interior_points(Domain2DUniform* domain) const
{
    auto it = interior_points_map.find(domain);
    if (it == interior_points_map.end())
        return 0;
    return it->second.size();
}

bool IBSolver2D_MirrorPoint::has_interior_points(Domain2DUniform* domain) const
{
    return interior_points_map.find(domain) != interior_points_map.end();
}

const std::vector<std::pair<int, int>>& IBSolver2D_MirrorPoint::get_interior_points(Domain2DUniform* domain) const
{
    static std::vector<std::pair<int, int>> empty;
    auto it = interior_points_map.find(domain);
    if (it == interior_points_map.end())
        return empty;
    return it->second;
}

std::pair<double, double> IBSolver2D_MirrorPoint::get_physical_location(Domain2DUniform* domain, int i, int j) const
{
    double x, y;

    switch (var->position_type)
    {
        case VariablePositionType::Center:
            // Cell center: x = offset_x + (i + 0.5) * hx
            x = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
            y = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
            break;
        case VariablePositionType::XFace:
            // X-face center: x = offset_x + i * hx
            x = domain->get_offset_x() + static_cast<double>(i) * domain->get_hx();
            y = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
            break;
        case VariablePositionType::YFace:
            // Y-face center: x = offset_x + (i + 0.5) * hx
            x = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
            y = domain->get_offset_y() + static_cast<double>(j) * domain->get_hy();
            break;
        case VariablePositionType::Corner:
            // Node: x = offset_x + i * hx
            x = domain->get_offset_x() + static_cast<double>(i) * domain->get_hx();
            y = domain->get_offset_y() + static_cast<double>(j) * domain->get_hy();
            break;
        default:
            // Default to cell center
            x = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * domain->get_hx();
            y = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * domain->get_hy();
            break;
    }

    return {x, y};
}

bool IBSolver2D_MirrorPoint::is_inside_any_shape(double x, double y) const
{
    for (const auto* shape : shapes)
    {
        if (shape->is_inside(x, y))
            return true;
    }
    return false;
}

void IBSolver2D_MirrorPoint::find_mirror_point(double x, double y, double& mirror_x, double& mirror_y, double& delta_l) const
{
    // Find the closest point on the shape surface
    // For multiple shapes, use the closest one
    double min_dist = 1e100;
    double closest_x = 0, closest_y = 0;
    double normal_x = 0, normal_y = 0;

    for (const auto* shape : shapes)
    {
        if (!shape->is_inside(x, y))
            continue;

        auto [surf_x, surf_y] = shape->get_closest_point(x, y);

        // Vector from interior point to surface
        double dx = surf_x - x;
        double dy = surf_y - y;
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist < min_dist)
        {
            min_dist = dist;
            closest_x = surf_x;
            closest_y = surf_y;

            if (dist > 1e-14)
            {
                normal_x = dx / dist;
                normal_y = dy / dist;
            }
            else
            {
                // Interior point is at the center, use radial direction
                normal_x = 1.0;
                normal_y = 0.0;
            }
        }
    }

    if (min_dist >= 1e100)
    {
        // Should not happen if called correctly
        mirror_x = x;
        mirror_y = y;
        delta_l = 0.0;
        return;
    }

    // Mirror point is on the other side of the boundary
    // Extend from the closest point along the outward normal
    // The interior point is inside, so we go in the direction from surface to interior
    // Actually, we want to go from interior point away from the surface
    mirror_x = x + normal_x * min_dist;
    mirror_y = y + normal_y * min_dist;
    delta_l = min_dist;
}

double IBSolver2D_MirrorPoint::interpolate_mirror_value(Domain2DUniform* domain, double mirror_x, double mirror_y) const
{
    auto* field = var->field_map.at(domain);
    int   nx    = field->get_nx();
    int   ny    = field->get_ny();

    // Get domain physical range
    double hx = domain->get_hx();
    double hy = domain->get_hy();
    double ox = domain->get_offset_x();
    double oy = domain->get_offset_y();
    double x_max = ox + nx * hx;
    double y_max = oy + ny * hy;

    // Determine which domain to use for interpolation
    // The mirror point may be in a neighbor domain
    Domain2DUniform* target_domain = domain;
    double           target_mirror_x = mirror_x;
    double           target_mirror_y = mirror_y;

    // Check if mirror point is outside current domain and find neighbor domain
    bool is_outside = false;
    if (mirror_x < ox || mirror_x > x_max || mirror_y < oy || mirror_y > y_max)
    {
        is_outside = true;
    }

    if (is_outside)
    {
        // Find the appropriate neighbor domain based on mirror point position
        auto adj_it = var->geometry->adjacency.find(domain);
        if (adj_it != var->geometry->adjacency.end())
        {
            // Determine which direction(s) the mirror point is outside
            bool outside_x_neg = mirror_x < ox;
            bool outside_x_pos = mirror_x > x_max;
            bool outside_y_neg = mirror_y < oy;
            bool outside_y_pos = mirror_y > y_max;

            // Try to find neighbor domain
            Domain2DUniform* found_domain = nullptr;

            // Check corner cases first (both x and y outside)
            if (outside_x_neg && outside_y_neg)
            {
                auto corner_it = adj_it->second.find(LocationType::XNegative);
                if (corner_it != adj_it->second.end())
                {
                    Domain2DUniform* x_neg_domain = corner_it->second;
                    auto y_it = var->geometry->adjacency.find(x_neg_domain);
                    if (y_it != var->geometry->adjacency.end())
                    {
                        auto corner_y_it = y_it->second.find(LocationType::YNegative);
                        if (corner_y_it != y_it->second.end())
                        {
                            found_domain = corner_y_it->second;
                        }
                    }
                }
            }
            else if (outside_x_neg && outside_y_pos)
            {
                auto corner_it = adj_it->second.find(LocationType::XNegative);
                if (corner_it != adj_it->second.end())
                {
                    Domain2DUniform* x_neg_domain = corner_it->second;
                    auto y_it = var->geometry->adjacency.find(x_neg_domain);
                    if (y_it != var->geometry->adjacency.end())
                    {
                        auto corner_y_it = y_it->second.find(LocationType::YPositive);
                        if (corner_y_it != y_it->second.end())
                        {
                            found_domain = corner_y_it->second;
                        }
                    }
                }
            }
            else if (outside_x_pos && outside_y_neg)
            {
                auto corner_it = adj_it->second.find(LocationType::XPositive);
                if (corner_it != adj_it->second.end())
                {
                    Domain2DUniform* x_pos_domain = corner_it->second;
                    auto y_it = var->geometry->adjacency.find(x_pos_domain);
                    if (y_it != var->geometry->adjacency.end())
                    {
                        auto corner_y_it = y_it->second.find(LocationType::YNegative);
                        if (corner_y_it != y_it->second.end())
                        {
                            found_domain = corner_y_it->second;
                        }
                    }
                }
            }
            else if (outside_x_pos && outside_y_pos)
            {
                auto corner_it = adj_it->second.find(LocationType::XPositive);
                if (corner_it != adj_it->second.end())
                {
                    Domain2DUniform* x_pos_domain = corner_it->second;
                    auto y_it = var->geometry->adjacency.find(x_pos_domain);
                    if (y_it != var->geometry->adjacency.end())
                    {
                        auto corner_y_it = y_it->second.find(LocationType::YPositive);
                        if (corner_y_it != y_it->second.end())
                        {
                            found_domain = corner_y_it->second;
                        }
                    }
                }
            }
            // Then check edge cases
            else if (outside_x_neg)
            {
                auto dir_it = adj_it->second.find(LocationType::XNegative);
                if (dir_it != adj_it->second.end())
                    found_domain = dir_it->second;
            }
            else if (outside_x_pos)
            {
                auto dir_it = adj_it->second.find(LocationType::XPositive);
                if (dir_it != adj_it->second.end())
                    found_domain = dir_it->second;
            }
            else if (outside_y_neg)
            {
                auto dir_it = adj_it->second.find(LocationType::YNegative);
                if (dir_it != adj_it->second.end())
                    found_domain = dir_it->second;
            }
            else if (outside_y_pos)
            {
                auto dir_it = adj_it->second.find(LocationType::YPositive);
                if (dir_it != adj_it->second.end())
                    found_domain = dir_it->second;
            }

            if (found_domain != nullptr && var->field_map.find(found_domain) != var->field_map.end())
            {
                target_domain = found_domain;
                // Update field pointer and domain parameters
                field = var->field_map.at(target_domain);
                nx = field->get_nx();
                ny = field->get_ny();
                hx = target_domain->get_hx();
                hy = target_domain->get_hy();
                ox = target_domain->get_offset_x();
                oy = target_domain->get_offset_y();
            }
        }
    }

    // Convert physical coordinates to grid indices based on position type
    double gi, gj;
    switch (var->position_type)
    {
        case VariablePositionType::Center:
            gi = (target_mirror_x - ox) / hx - 0.5;
            gj = (target_mirror_y - oy) / hy - 0.5;
            break;
        case VariablePositionType::XFace:
            gi = (target_mirror_x - ox) / hx;
            gj = (target_mirror_y - oy) / hy - 0.5;
            break;
        case VariablePositionType::YFace:
            gi = (target_mirror_x - ox) / hx - 0.5;
            gj = (target_mirror_y - oy) / hy;
            break;
        case VariablePositionType::Corner:
            gi = (target_mirror_x - ox) / hx;
            gj = (target_mirror_y - oy) / hy;
            break;
        default:
            gi = (target_mirror_x - ox) / hx - 0.5;
            gj = (target_mirror_y - oy) / hy - 0.5;
            break;
    }

    int i0 = static_cast<int>(std::floor(gi));
    int j0 = static_cast<int>(std::floor(gj));
    int i1 = i0 + 1;
    int j1 = j0 + 1;

    double sx = gi - static_cast<double>(i0);
    double sy = gj - static_cast<double>(j0);

    // Helper lambda to get value with boundary handling for target domain
    auto get_val = [&](int i, int j) -> double {
        // Check if within target domain
        if (i >= 0 && i < nx && j >= 0 && j < ny)
        {
            return (*field)(i, j);
        }

        // Need to handle out-of-bounds - use buffer if available
        auto buffer_it = var->buffer_map.find(target_domain);
        if (buffer_it != var->buffer_map.end())
        {
            const auto& buffer = buffer_it->second;

            // X negative buffer
            if (i < 0 && j >= 0 && j < ny)
            {
                auto buf_it = buffer.find(LocationType::XNegative);
                if (buf_it != buffer.end())
                    return buf_it->second[j];
            }
            // X positive buffer
            if (i >= nx && j >= 0 && j < ny)
            {
                auto buf_it = buffer.find(LocationType::XPositive);
                if (buf_it != buffer.end())
                    return buf_it->second[j];
            }
            // Y negative buffer
            if (j < 0 && i >= 0 && i < nx)
            {
                auto buf_it = buffer.find(LocationType::YNegative);
                if (buf_it != buffer.end())
                    return buf_it->second[i];
            }
            // Y positive buffer
            if (j >= ny && i >= 0 && i < nx)
            {
                auto buf_it = buffer.find(LocationType::YPositive);
                if (buf_it != buffer.end())
                    return buf_it->second[i];
            }
        }

        // Fall back to single-side extrapolation (clamp to boundary)
        if (i < 0)
            i = 0;
        if (i >= nx)
            i = nx - 1;
        if (j < 0)
            j = 0;
        if (j >= ny)
            j = ny - 1;

        return (*field)(i, j);
    };

    // Get the four corner values for bilinear interpolation
    double v00 = get_val(i0, j0);
    double v10 = get_val(i1, j0);
    double v01 = get_val(i0, j1);
    double v11 = get_val(i1, j1);

    // Bilinear interpolation
    double result = (1.0 - sx) * (1.0 - sy) * v00 + sx * (1.0 - sy) * v10 + (1.0 - sx) * sy * v01 + sx * sy * v11;

    return result;
}

double IBSolver2D_MirrorPoint::get_field_value(Domain2DUniform* domain, int i, int j) const
{
    auto* field = var->field_map.at(domain);
    int   nx    = field->get_nx();
    int   ny    = field->get_ny();

    if (i >= 0 && i < nx && j >= 0 && j < ny)
    {
        return (*field)(i, j);
    }

    // Use buffer if available
    auto buffer_it = var->buffer_map.find(domain);
    if (buffer_it != var->buffer_map.end())
    {
        const auto& buffer = buffer_it->second;

        if (i < 0 && j >= 0 && j < ny)
        {
            auto buf_it = buffer.find(LocationType::XNegative);
            if (buf_it != buffer.end())
                return buf_it->second[j];
        }
        if (i >= nx && j >= 0 && j < ny)
        {
            auto buf_it = buffer.find(LocationType::XPositive);
            if (buf_it != buffer.end())
                return buf_it->second[j];
        }
        if (j < 0 && i >= 0 && i < nx)
        {
            auto buf_it = buffer.find(LocationType::YNegative);
            if (buf_it != buffer.end())
                return buf_it->second[i];
        }
        if (j >= ny && i >= 0 && i < nx)
        {
            auto buf_it = buffer.find(LocationType::YPositive);
            if (buf_it != buffer.end())
                return buf_it->second[i];
        }
    }

    // Fallback to boundary value
    return boundary_value;
}
