#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "ns/physical_pe_solver2d.h"
#include "pe/concat/concat_solver2d.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    struct ErrorStats
    {
        double l2   = 0.0;
        double linf = 0.0;
    };

    struct MaxErrorInfo
    {
        std::string domain_name;
        int         i    = -1;
        int         j    = -1;
        double      x    = 0.0;
        double      y    = 0.0;
        double      diff = 0.0;
    };

    struct BoundarySnapshot
    {
        double u_yneg_0 = 0.0;
        double u_yneg_1 = 0.0;
        double v_xneg_0 = 0.0;
        double v_xneg_1 = 0.0;
        double u_corner = 0.0;
        double v_corner = 0.0;
    };

    struct UpperRightCornerSnapshot
    {
        double u_corner = 0.0;
        double v_corner = 0.0;
    };

    constexpr double kA   = 1.0;
    constexpr double kB   = 0.5;
    constexpr double kC   = 0.25;
    constexpr double kRho = 2.0;
    constexpr double kPx  = 0.3;
    constexpr double kPy  = -0.2;
    constexpr double kP0  = 0.1;

    double exact_u(double x, double y) { return kA * x + kB * y; }

    double exact_v(double x, double y) { return kC * x - kA * y; }

    double exact_p(double x, double y)
    {
        const double coeff = -0.5 * kRho * (kA * kA + kB * kC);
        return coeff * (x * x + y * y) + kPx * x + kPy * y + kP0;
    }

    double exact_rhs() { return -kRho * (kA * kA + 2.0 * kB * kC + kA * kA); }

    double exact_u_xpos_ypos_corner(Domain2DUniform* domain)
    {
        const double x = domain->get_offset_x() + static_cast<double>(domain->get_nx()) * domain->get_hx();
        const double y = domain->get_offset_y() + (static_cast<double>(domain->get_ny()) + 0.5) * domain->get_hy();
        return exact_u(x, y);
    }

    double exact_v_xpos_ypos_corner(Domain2DUniform* domain)
    {
        const double x = domain->get_offset_x() + (static_cast<double>(domain->get_nx()) + 0.5) * domain->get_hx();
        const double y = domain->get_offset_y() + static_cast<double>(domain->get_ny()) * domain->get_hy();
        return exact_v(x, y);
    }

    bool is_adjacented(const Geometry2D& geo, Domain2DUniform* domain, LocationType loc)
    {
        return geo.adjacency.count(domain) && geo.adjacency.at(domain).count(loc);
    }

    void set_exact_boundary(Variable2D& var, Domain2DUniform* domain, LocationType loc, double (*func)(double, double))
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(domain, loc, func);
    }

    void set_exact_poisson_dirichlet_buffer(Variable2D&      var,
                                            Domain2DUniform* domain,
                                            LocationType     loc,
                                            double (*func)(double, double))
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Dirichlet);
        var.set_buffer_value(domain, loc, func);
    }

    ErrorStats accumulate_field_error(const Variable2D&                                    var,
                                      const std::unordered_map<Domain2DUniform*, field2*>& field_map,
                                      double (*exact)(double, double))
    {
        ErrorStats stats;
        double     sum_sq = 0.0;
        double     count  = 0.0;

        for (auto* domain : var.geometry->domains)
        {
            const field2& field = *field_map.at(domain);
            const double  hx    = domain->get_hx();
            const double  hy    = domain->get_hy();
            const double  ox    = domain->get_offset_x();
            const double  oy    = domain->get_offset_y();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    double x = ox;
                    double y = oy;

                    switch (var.position_type)
                    {
                        case VariablePositionType::XFace:
                            x += static_cast<double>(i) * hx;
                            y += (static_cast<double>(j) + 0.5) * hy;
                            break;
                        case VariablePositionType::YFace:
                            x += (static_cast<double>(i) + 0.5) * hx;
                            y += static_cast<double>(j) * hy;
                            break;
                        case VariablePositionType::Center:
                            x += (static_cast<double>(i) + 0.5) * hx;
                            y += (static_cast<double>(j) + 0.5) * hy;
                            break;
                        default:
                            throw std::runtime_error("Unsupported variable position type");
                    }

                    const double diff = field(i, j) - exact(x, y);
                    sum_sq += diff * diff;
                    stats.linf = std::max(stats.linf, std::abs(diff));
                    count += 1.0;
                }
            }
        }

        stats.l2 = count > 0.0 ? std::sqrt(sum_sq / count) : 0.0;
        return stats;
    }

    MaxErrorInfo find_max_field_error(const Variable2D&                                    var,
                                      const std::unordered_map<Domain2DUniform*, field2*>& field_map,
                                      double (*exact)(double, double))
    {
        MaxErrorInfo info;

        for (auto* domain : var.geometry->domains)
        {
            const field2& field = *field_map.at(domain);
            const double  hx    = domain->get_hx();
            const double  hy    = domain->get_hy();
            const double  ox    = domain->get_offset_x();
            const double  oy    = domain->get_offset_y();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    double x = ox;
                    double y = oy;

                    switch (var.position_type)
                    {
                        case VariablePositionType::XFace:
                            x += static_cast<double>(i) * hx;
                            y += (static_cast<double>(j) + 0.5) * hy;
                            break;
                        case VariablePositionType::YFace:
                            x += (static_cast<double>(i) + 0.5) * hx;
                            y += static_cast<double>(j) * hy;
                            break;
                        case VariablePositionType::Center:
                            x += (static_cast<double>(i) + 0.5) * hx;
                            y += (static_cast<double>(j) + 0.5) * hy;
                            break;
                        default:
                            throw std::runtime_error("Unsupported variable position type");
                    }

                    const double diff = field(i, j) - exact(x, y);
                    if (std::abs(diff) <= std::abs(info.diff))
                        continue;

                    info.domain_name = domain->name;
                    info.i           = i;
                    info.j           = j;
                    info.x           = x;
                    info.y           = y;
                    info.diff        = diff;
                }
            }
        }

        return info;
    }

    ErrorStats accumulate_constant_error(const std::unordered_map<Domain2DUniform*, field2*>& field_map, double exact)
    {
        ErrorStats stats;
        double     sum_sq = 0.0;
        double     count  = 0.0;

        for (const auto& [domain, field_ptr] : field_map)
        {
            const field2& field = *field_ptr;
            (void)domain;
            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    const double diff = field(i, j) - exact;
                    sum_sq += diff * diff;
                    stats.linf = std::max(stats.linf, std::abs(diff));
                    count += 1.0;
                }
            }
        }

        stats.l2 = count > 0.0 ? std::sqrt(sum_sq / count) : 0.0;
        return stats;
    }

    MaxErrorInfo find_max_constant_error(const std::unordered_map<Domain2DUniform*, field2*>& field_map, double exact)
    {
        MaxErrorInfo info;

        for (const auto& [domain, field_ptr] : field_map)
        {
            const field2& field = *field_ptr;
            const double  hx    = domain->get_hx();
            const double  hy    = domain->get_hy();
            const double  ox    = domain->get_offset_x();
            const double  oy    = domain->get_offset_y();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    const double diff = field(i, j) - exact;
                    if (std::abs(diff) <= std::abs(info.diff))
                        continue;

                    info.domain_name = domain->name;
                    info.i           = i;
                    info.j           = j;
                    info.x           = ox + (static_cast<double>(i) + 0.5) * hx;
                    info.y           = oy + (static_cast<double>(j) + 0.5) * hy;
                    info.diff        = diff;
                }
            }
        }

        return info;
    }

    void fill_exact_velocity_fields(Variable2D& u, Variable2D& v)
    {
        for (auto* domain : u.geometry->domains)
        {
            field2&      u_field = *u.field_map[domain];
            field2&      v_field = *v.field_map[domain];
            const double hx      = domain->get_hx();
            const double hy      = domain->get_hy();
            const double ox      = domain->get_offset_x();
            const double oy      = domain->get_offset_y();

            for (int i = 0; i < u_field.get_nx(); ++i)
            {
                for (int j = 0; j < u_field.get_ny(); ++j)
                {
                    const double x = ox + static_cast<double>(i) * hx;
                    const double y = oy + (static_cast<double>(j) + 0.5) * hy;
                    u_field(i, j)  = exact_u(x, y);
                }
            }

            for (int i = 0; i < v_field.get_nx(); ++i)
            {
                for (int j = 0; j < v_field.get_ny(); ++j)
                {
                    const double x = ox + (static_cast<double>(i) + 0.5) * hx;
                    const double y = oy + static_cast<double>(j) * hy;
                    v_field(i, j)  = exact_v(x, y);
                }
            }
        }
    }

    BoundarySnapshot capture_a2_boundary_snapshot(Variable2D& u, Variable2D& v, Domain2DUniform* domain)
    {
        BoundarySnapshot snapshot;
        const double*    u_yneg = u.buffer_map[domain][LocationType::YNegative];
        const double*    v_xneg = v.buffer_map[domain][LocationType::XNegative];

        snapshot.u_yneg_0 = u_yneg[0];
        snapshot.u_yneg_1 = u_yneg[1];
        snapshot.v_xneg_0 = v_xneg[0];
        snapshot.v_xneg_1 = v_xneg[1];
        snapshot.u_corner = u.xpos_yneg_corner_map[domain];
        snapshot.v_corner = v.xneg_ypos_corner_map[domain];
        return snapshot;
    }

    UpperRightCornerSnapshot capture_upper_right_corner_snapshot(const PhysicalPESolver2D& solver, Domain2DUniform* domain)
    {
        UpperRightCornerSnapshot snapshot;
        snapshot.u_corner = solver.u_xpos_ypos_corner_map.at(domain);
        snapshot.v_corner = solver.v_xpos_ypos_corner_map.at(domain);
        return snapshot;
    }
} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: validation_2d_cross_slot_physical_pressure_mms rank" << std::endl;
        return 1;
    }

    const int rank = std::stoi(argv[1]);
    if (rank < 4)
    {
        std::cerr << "rank must be >= 4" << std::endl;
        return 1;
    }

    EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
    env_cfg.showCurrentStep    = false;
    env_cfg.showGmresRes       = false;
    env_cfg.debugOutputDir     = "./result/cross_slot_physical_pressure_mms/rank" + std::to_string(rank);

    Geometry2D geo;

    Domain2DUniform A2(rank, rank, 1.0, 1.0, "A2");
    Domain2DUniform A1("A1");
    Domain2DUniform A3("A3");
    Domain2DUniform A4("A4");
    Domain2DUniform A5("A5");

    A1.set_nx(rank);
    A1.set_lx(1.0);
    A3.set_nx(rank);
    A3.set_lx(1.0);
    A4.set_ny(rank);
    A4.set_ly(1.0);
    A5.set_ny(rank);
    A5.set_ly(1.0);

    geo.add_domain({&A1, &A2, &A3, &A4, &A5});
    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);
    geo.connect(&A2, LocationType::YPositive, &A5);
    geo.axis(&A2, LocationType::XNegative);
    geo.axis(&A2, LocationType::YNegative);
    geo.global_move_x(-0.5);
    geo.global_move_y(-0.5);

    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    field2 u_A1, u_A2, u_A3, u_A4, u_A5;
    field2 v_A1, v_A2, v_A3, v_A4, v_A5;
    field2 p_A1, p_A2, p_A3, p_A4, p_A5;

    u.set_x_edge_field(&A1, u_A1);
    u.set_x_edge_field(&A2, u_A2);
    u.set_x_edge_field(&A3, u_A3);
    u.set_x_edge_field(&A4, u_A4);
    u.set_x_edge_field(&A5, u_A5);

    v.set_y_edge_field(&A1, v_A1);
    v.set_y_edge_field(&A2, v_A2);
    v.set_y_edge_field(&A3, v_A3);
    v.set_y_edge_field(&A4, v_A4);
    v.set_y_edge_field(&A5, v_A5);

    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);
    p.set_center_field(&A5, p_A5);

    const std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};
    const std::vector<LocationType>     dirs    = {
        LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive};

    for (auto* domain : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(geo, domain, loc))
                continue;

            set_exact_boundary(u, domain, loc, exact_u);
            set_exact_boundary(v, domain, loc, exact_v);
            set_exact_poisson_dirichlet_buffer(p, domain, loc, exact_p);
        }
    }

    fill_exact_velocity_fields(u, v);
    for (auto* domain : domains)
        p.field_map[domain]->clear(0.0);

    ConcatPoissonSolver2D p_solver(&p);
    p_solver.set_parameter(40, 1.0e-12, 200);
    PhysicalPESolver2D ppe_solver(&u, &v, &p, &p_solver, kRho);

    ppe_solver.phys_boundary_update();
    const UpperRightCornerSnapshot a3_after_phys = capture_upper_right_corner_snapshot(ppe_solver, &A3);
    ppe_solver.nondiag_shared_boundary_update();
    const BoundarySnapshot after_nondiag = capture_a2_boundary_snapshot(u, v, &A2);
    const UpperRightCornerSnapshot a2_after_nondiag = capture_upper_right_corner_snapshot(ppe_solver, &A2);
    ppe_solver.diag_shared_boundary_update();
    const BoundarySnapshot after_diag = capture_a2_boundary_snapshot(u, v, &A2);
    const UpperRightCornerSnapshot a1_after_diag = capture_upper_right_corner_snapshot(ppe_solver, &A1);
    ppe_solver.calc_rhs();
    const ErrorStats   rhs_error = accumulate_constant_error(p.field_map, exact_rhs());
    const MaxErrorInfo rhs_max   = find_max_constant_error(p.field_map, exact_rhs());

    for (auto* domain : domains)
        p.field_map[domain]->clear(0.0);

    ppe_solver.solve();

    const ErrorStats   dudx_error = accumulate_constant_error(ppe_solver.dudx_map, kA);
    const ErrorStats   dudy_error = accumulate_constant_error(ppe_solver.dudy_map, kB);
    const ErrorStats   dvdx_error = accumulate_constant_error(ppe_solver.dvdx_map, kC);
    const ErrorStats   dvdy_error = accumulate_constant_error(ppe_solver.dvdy_map, -kA);
    const ErrorStats   p_error    = accumulate_field_error(p, p.field_map, exact_p);
    const MaxErrorInfo dudy_max   = find_max_constant_error(ppe_solver.dudy_map, kB);
    const MaxErrorInfo dvdx_max   = find_max_constant_error(ppe_solver.dvdx_map, kC);
    const MaxErrorInfo p_max      = find_max_field_error(p, p.field_map, exact_p);

    const double a3_phys_u_expected = exact_u_xpos_ypos_corner(&A3);
    const double a3_phys_v_expected = exact_v_xpos_ypos_corner(&A3);
    const double a2_nondiag_u_expected = exact_u_xpos_ypos_corner(&A2);
    const double a2_nondiag_v_expected = exact_v_xpos_ypos_corner(&A2);
    const double a1_diag_u_expected = exact_u_xpos_ypos_corner(&A1);
    const double a1_diag_v_expected = exact_v_xpos_ypos_corner(&A1);

    const double a3_phys_u_diff = a3_after_phys.u_corner - a3_phys_u_expected;
    const double a3_phys_v_diff = a3_after_phys.v_corner - a3_phys_v_expected;
    const double a2_nondiag_u_diff = a2_after_nondiag.u_corner - a2_nondiag_u_expected;
    const double a2_nondiag_v_diff = a2_after_nondiag.v_corner - a2_nondiag_v_expected;
    const double a1_diag_u_diff = a1_after_diag.u_corner - a1_diag_u_expected;
    const double a1_diag_v_diff = a1_after_diag.v_corner - a1_diag_v_expected;

    std::cout << std::setprecision(16);
    std::cout << "ppe_rhs_error: l2 = " << rhs_error.l2 << ", linf = " << rhs_error.linf << std::endl;
    std::cout << "dudx_error: l2 = " << dudx_error.l2 << ", linf = " << dudx_error.linf << std::endl;
    std::cout << "dudy_error: l2 = " << dudy_error.l2 << ", linf = " << dudy_error.linf << std::endl;
    std::cout << "dvdx_error: l2 = " << dvdx_error.l2 << ", linf = " << dvdx_error.linf << std::endl;
    std::cout << "dvdy_error: l2 = " << dvdy_error.l2 << ", linf = " << dvdy_error.linf << std::endl;
    std::cout << "ppe_solution_error: l2 = " << p_error.l2 << ", linf = " << p_error.linf << std::endl;
    std::cout << "dudy_max: domain = " << dudy_max.domain_name << ", i = " << dudy_max.i << ", j = " << dudy_max.j
              << ", x = " << dudy_max.x << ", y = " << dudy_max.y << ", diff = " << dudy_max.diff << std::endl;
    std::cout << "dvdx_max: domain = " << dvdx_max.domain_name << ", i = " << dvdx_max.i << ", j = " << dvdx_max.j
              << ", x = " << dvdx_max.x << ", y = " << dvdx_max.y << ", diff = " << dvdx_max.diff << std::endl;
    std::cout << "rhs_max: domain = " << rhs_max.domain_name << ", i = " << rhs_max.i << ", j = " << rhs_max.j
              << ", x = " << rhs_max.x << ", y = " << rhs_max.y << ", diff = " << rhs_max.diff << std::endl;
    std::cout << "p_max: domain = " << p_max.domain_name << ", i = " << p_max.i << ", j = " << p_max.j
              << ", x = " << p_max.x << ", y = " << p_max.y << ", diff = " << p_max.diff << std::endl;
    std::cout << "A3_right_top_after_phys: "
              << "u_corner = " << a3_after_phys.u_corner << ", "
              << "u_expected = " << a3_phys_u_expected << ", "
              << "u_diff = " << a3_phys_u_diff << ", "
              << "v_corner = " << a3_after_phys.v_corner << ", "
              << "v_expected = " << a3_phys_v_expected << ", "
              << "v_diff = " << a3_phys_v_diff << std::endl;
    std::cout << "A2_right_top_after_nondiag: "
              << "u_corner = " << a2_after_nondiag.u_corner << ", "
              << "u_expected = " << a2_nondiag_u_expected << ", "
              << "u_diff = " << a2_nondiag_u_diff << ", "
              << "v_corner = " << a2_after_nondiag.v_corner << ", "
              << "v_expected = " << a2_nondiag_v_expected << ", "
              << "v_diff = " << a2_nondiag_v_diff << std::endl;
    std::cout << "A1_right_top_after_diag: "
              << "u_corner = " << a1_after_diag.u_corner << ", "
              << "u_expected = " << a1_diag_u_expected << ", "
              << "u_diff = " << a1_diag_u_diff << ", "
              << "v_corner = " << a1_after_diag.v_corner << ", "
              << "v_expected = " << a1_diag_v_expected << ", "
              << "v_diff = " << a1_diag_v_diff << std::endl;

    const double tol   = 1.0e-12;
    const double p_tol = 1.0e-9;
    const bool right_top_corner_pass = std::abs(a3_phys_u_diff) < tol && std::abs(a3_phys_v_diff) < tol &&
                                       std::abs(a2_nondiag_u_diff) < tol && std::abs(a2_nondiag_v_diff) < tol &&
                                       std::abs(a1_diag_u_diff) < tol && std::abs(a1_diag_v_diff) < tol;
    const bool pass = rhs_error.linf < tol && dudx_error.linf < tol && dudy_error.linf < tol && dvdx_error.linf < tol &&
                      dvdy_error.linf < tol && p_error.linf < p_tol && right_top_corner_pass;

    if (!pass)
    {
        std::cout << "A2_after_nondiag: "
                  << "u_yneg[0] = " << after_nondiag.u_yneg_0 << ", "
                  << "u_yneg[1] = " << after_nondiag.u_yneg_1 << ", "
                  << "v_xneg[0] = " << after_nondiag.v_xneg_0 << ", "
                  << "v_xneg[1] = " << after_nondiag.v_xneg_1 << ", "
                  << "u_corner = " << after_nondiag.u_corner << ", "
                  << "v_corner = " << after_nondiag.v_corner << std::endl;
        std::cout << "A2_after_diag: "
                  << "u_yneg[0] = " << after_diag.u_yneg_0 << ", "
                  << "u_yneg[1] = " << after_diag.u_yneg_1 << ", "
                  << "v_xneg[0] = " << after_diag.v_xneg_0 << ", "
                  << "v_xneg[1] = " << after_diag.v_xneg_1 << ", "
                  << "u_corner = " << after_diag.u_corner << ", "
                  << "v_corner = " << after_diag.v_corner << std::endl;
    }

    return pass ? 0 : 2;
}
