#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field2.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "ns/physical_pe_solver3d.h"
#include "pe/concat/concat_solver3d.h"

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
        int         k    = -1;
        double      x    = 0.0;
        double      y    = 0.0;
        double      z    = 0.0;
        double      diff = 0.0;
    };

    struct XPositiveFaceSnapshot
    {
        double u = 0.0;
        double v = 0.0;
        double w = 0.0;
    };

    struct PositiveEdgeSnapshot
    {
        double u_xy = 0.0;
        double v_xy = 0.0;
        double u_xz = 0.0;
        double w_xz = 0.0;
        double v_yz = 0.0;
        double w_yz = 0.0;
    };

    constexpr double kA   = 1.0;
    constexpr double kB   = 0.5;
    constexpr double kC   = -0.25;
    constexpr double kD   = 0.75;
    constexpr double kE   = -0.6;
    constexpr double kF   = 0.4;
    constexpr double kG   = -0.3;
    constexpr double kH   = 0.2;
    constexpr double kI   = -(kA + kE);
    constexpr double kRho = 2.0;
    constexpr double kPx  = 0.3;
    constexpr double kPy  = -0.2;
    constexpr double kPz  = 0.1;
    constexpr double kP0  = 0.05;

    double exact_u(double x, double y, double z) { return kA * x + kB * y + kC * z; }

    double exact_v(double x, double y, double z) { return kD * x + kE * y + kF * z; }

    double exact_w(double x, double y, double z) { return kG * x + kH * y + kI * z; }

    double exact_rhs()
    {
        const double trace_l2 =
            kA * kA + kE * kE + kI * kI + 2.0 * kB * kD + 2.0 * kC * kG + 2.0 * kF * kH;
        return -kRho * trace_l2;
    }

    double exact_p(double x, double y, double z)
    {
        const double alpha = exact_rhs() / 6.0;
        return alpha * (x * x + y * y + z * z) + kPx * x + kPy * y + kPz * z + kP0;
    }

    bool is_adjacented(const Geometry3D& geo, Domain3DUniform* domain, LocationType loc)
    {
        return geo.adjacency.count(domain) && geo.adjacency.at(domain).count(loc);
    }

    ErrorStats accumulate_constant_error(const std::unordered_map<Domain3DUniform*, field3*>& field_map, double exact)
    {
        ErrorStats stats;
        double     sum_sq = 0.0;
        double     count  = 0.0;

        for (const auto& [domain, field_ptr] : field_map)
        {
            const field3& field = *field_ptr;
            (void)domain;

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    for (int k = 0; k < field.get_nz(); ++k)
                    {
                        const double diff = field(i, j, k) - exact;
                        sum_sq += diff * diff;
                        stats.linf = std::max(stats.linf, std::abs(diff));
                        count += 1.0;
                    }
                }
            }
        }

        stats.l2 = count > 0.0 ? std::sqrt(sum_sq / count) : 0.0;
        return stats;
    }

    MaxErrorInfo find_max_constant_error(const std::unordered_map<Domain3DUniform*, field3*>& field_map, double exact)
    {
        MaxErrorInfo info;

        for (const auto& [domain, field_ptr] : field_map)
        {
            const field3& field = *field_ptr;
            const double  hx    = domain->get_hx();
            const double  hy    = domain->get_hy();
            const double  hz    = domain->get_hz();
            const double  ox    = domain->get_offset_x();
            const double  oy    = domain->get_offset_y();
            const double  oz    = domain->get_offset_z();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    for (int k = 0; k < field.get_nz(); ++k)
                    {
                        const double diff = field(i, j, k) - exact;
                        if (std::abs(diff) <= std::abs(info.diff))
                            continue;

                        info.domain_name = domain->name;
                        info.i           = i;
                        info.j           = j;
                        info.k           = k;
                        info.x           = ox + (static_cast<double>(i) + 0.5) * hx;
                        info.y           = oy + (static_cast<double>(j) + 0.5) * hy;
                        info.z           = oz + (static_cast<double>(k) + 0.5) * hz;
                        info.diff        = diff;
                    }
                }
            }
        }

        return info;
    }

    ErrorStats accumulate_field_error(const Variable3D&                                    var,
                                      const std::unordered_map<Domain3DUniform*, field3*>& field_map,
                                      double (*exact)(double, double, double))
    {
        ErrorStats stats;
        double     sum_sq = 0.0;
        double     count  = 0.0;

        for (auto* domain : var.geometry->domains)
        {
            const field3& field = *field_map.at(domain);
            const double  hx    = domain->get_hx();
            const double  hy    = domain->get_hy();
            const double  hz    = domain->get_hz();
            const double  ox    = domain->get_offset_x();
            const double  oy    = domain->get_offset_y();
            const double  oz    = domain->get_offset_z();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    for (int k = 0; k < field.get_nz(); ++k)
                    {
                        double x = ox;
                        double y = oy;
                        double z = oz;

                        switch (var.position_type)
                        {
                            case VariablePositionType::XFace:
                                x += static_cast<double>(i) * hx;
                                y += (static_cast<double>(j) + 0.5) * hy;
                                z += (static_cast<double>(k) + 0.5) * hz;
                                break;
                            case VariablePositionType::YFace:
                                x += (static_cast<double>(i) + 0.5) * hx;
                                y += static_cast<double>(j) * hy;
                                z += (static_cast<double>(k) + 0.5) * hz;
                                break;
                            case VariablePositionType::ZFace:
                                x += (static_cast<double>(i) + 0.5) * hx;
                                y += (static_cast<double>(j) + 0.5) * hy;
                                z += static_cast<double>(k) * hz;
                                break;
                            case VariablePositionType::Center:
                                x += (static_cast<double>(i) + 0.5) * hx;
                                y += (static_cast<double>(j) + 0.5) * hy;
                                z += (static_cast<double>(k) + 0.5) * hz;
                                break;
                            default:
                                throw std::runtime_error("Unsupported variable position type");
                        }

                        const double diff = field(i, j, k) - exact(x, y, z);
                        sum_sq += diff * diff;
                        stats.linf = std::max(stats.linf, std::abs(diff));
                        count += 1.0;
                    }
                }
            }
        }

        stats.l2 = count > 0.0 ? std::sqrt(sum_sq / count) : 0.0;
        return stats;
    }

    MaxErrorInfo find_max_field_error(const Variable3D&                                    var,
                                      const std::unordered_map<Domain3DUniform*, field3*>& field_map,
                                      double (*exact)(double, double, double))
    {
        MaxErrorInfo info;

        for (auto* domain : var.geometry->domains)
        {
            const field3& field = *field_map.at(domain);
            const double  hx    = domain->get_hx();
            const double  hy    = domain->get_hy();
            const double  hz    = domain->get_hz();
            const double  ox    = domain->get_offset_x();
            const double  oy    = domain->get_offset_y();
            const double  oz    = domain->get_offset_z();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    for (int k = 0; k < field.get_nz(); ++k)
                    {
                        double x = ox;
                        double y = oy;
                        double z = oz;

                        switch (var.position_type)
                        {
                            case VariablePositionType::XFace:
                                x += static_cast<double>(i) * hx;
                                y += (static_cast<double>(j) + 0.5) * hy;
                                z += (static_cast<double>(k) + 0.5) * hz;
                                break;
                            case VariablePositionType::YFace:
                                x += (static_cast<double>(i) + 0.5) * hx;
                                y += static_cast<double>(j) * hy;
                                z += (static_cast<double>(k) + 0.5) * hz;
                                break;
                            case VariablePositionType::ZFace:
                                x += (static_cast<double>(i) + 0.5) * hx;
                                y += (static_cast<double>(j) + 0.5) * hy;
                                z += static_cast<double>(k) * hz;
                                break;
                            case VariablePositionType::Center:
                                x += (static_cast<double>(i) + 0.5) * hx;
                                y += (static_cast<double>(j) + 0.5) * hy;
                                z += (static_cast<double>(k) + 0.5) * hz;
                                break;
                            default:
                                throw std::runtime_error("Unsupported variable position type");
                        }

                        const double diff = field(i, j, k) - exact(x, y, z);
                        if (std::abs(diff) <= std::abs(info.diff))
                            continue;

                        info.domain_name = domain->name;
                        info.i           = i;
                        info.j           = j;
                        info.k           = k;
                        info.x           = x;
                        info.y           = y;
                        info.z           = z;
                        info.diff        = diff;
                    }
                }
            }
        }

        return info;
    }

    void fill_exact_velocity_fields(Variable3D& u, Variable3D& v, Variable3D& w)
    {
        for (auto* domain : u.geometry->domains)
        {
            field3&      u_field = *u.field_map[domain];
            field3&      v_field = *v.field_map[domain];
            field3&      w_field = *w.field_map[domain];
            const double hx      = domain->get_hx();
            const double hy      = domain->get_hy();
            const double hz      = domain->get_hz();
            const double ox      = domain->get_offset_x();
            const double oy      = domain->get_offset_y();
            const double oz      = domain->get_offset_z();

            for (int i = 0; i < u_field.get_nx(); ++i)
            {
                for (int j = 0; j < u_field.get_ny(); ++j)
                {
                    for (int k = 0; k < u_field.get_nz(); ++k)
                    {
                        const double x = ox + static_cast<double>(i) * hx;
                        const double y = oy + (static_cast<double>(j) + 0.5) * hy;
                        const double z = oz + (static_cast<double>(k) + 0.5) * hz;
                        u_field(i, j, k) = exact_u(x, y, z);
                    }
                }
            }

            for (int i = 0; i < v_field.get_nx(); ++i)
            {
                for (int j = 0; j < v_field.get_ny(); ++j)
                {
                    for (int k = 0; k < v_field.get_nz(); ++k)
                    {
                        const double x = ox + (static_cast<double>(i) + 0.5) * hx;
                        const double y = oy + static_cast<double>(j) * hy;
                        const double z = oz + (static_cast<double>(k) + 0.5) * hz;
                        v_field(i, j, k) = exact_v(x, y, z);
                    }
                }
            }

            for (int i = 0; i < w_field.get_nx(); ++i)
            {
                for (int j = 0; j < w_field.get_ny(); ++j)
                {
                    for (int k = 0; k < w_field.get_nz(); ++k)
                    {
                        const double x = ox + (static_cast<double>(i) + 0.5) * hx;
                        const double y = oy + (static_cast<double>(j) + 0.5) * hy;
                        const double z = oz + static_cast<double>(k) * hz;
                        w_field(i, j, k) = exact_w(x, y, z);
                    }
                }
            }
        }
    }

    void fill_exact_base_corner_maps(Variable3D& u, Variable3D& v, Variable3D& w)
    {
        for (auto* domain : u.geometry->domains)
        {
            const double ox = domain->get_offset_x();
            const double oy = domain->get_offset_y();
            const double oz = domain->get_offset_z();
            const double hx = domain->get_hx();
            const double hy = domain->get_hy();
            const double hz = domain->get_hz();
            const int    nx = domain->get_nx();
            const int    ny = domain->get_ny();
            const int    nz = domain->get_nz();

            const double x_pos = ox + static_cast<double>(nx) * hx;
            const double y_pos = oy + static_cast<double>(ny) * hy;
            const double z_pos = oz + static_cast<double>(nz) * hz;
            const double x_neg = ox - 0.5 * hx;
            const double y_neg = oy - 0.5 * hy;
            const double z_neg = oz - 0.5 * hz;

            for (int j = 0; j < ny; ++j)
            {
                const double y = oy + (static_cast<double>(j) + 0.5) * hy;
                u.corner_y_map[domain][j] = exact_u(x_pos, y, z_neg);
                w.corner_y_map[domain][j] = exact_w(x_neg, y, z_pos);
            }

            for (int k = 0; k < nz; ++k)
            {
                const double z = oz + (static_cast<double>(k) + 0.5) * hz;
                u.corner_z_map[domain][k] = exact_u(x_pos, y_neg, z);
                v.corner_z_map[domain][k] = exact_v(x_neg, y_pos, z);
            }

            for (int i = 0; i < nx; ++i)
            {
                const double x = ox + (static_cast<double>(i) + 0.5) * hx;
                v.corner_x_map[domain][i] = exact_v(x, y_pos, z_neg);
                w.corner_x_map[domain][i] = exact_w(x, y_neg, z_pos);
            }
        }
    }

    XPositiveFaceSnapshot capture_xpos_face_snapshot(Variable3D& u, Variable3D& v, Variable3D& w, Domain3DUniform* domain)
    {
        XPositiveFaceSnapshot snapshot;
        const field2&         u_xpos = *u.buffer_map[domain][LocationType::XPositive];
        const field2&         v_xpos = *v.buffer_map[domain][LocationType::XPositive];
        const field2&         w_xpos = *w.buffer_map[domain][LocationType::XPositive];

        snapshot.u = u_xpos(0, 0);
        snapshot.v = v_xpos(0, 0);
        snapshot.w = w_xpos(0, 0);
        return snapshot;
    }

    PositiveEdgeSnapshot capture_positive_edge_snapshot(const PhysicalPESolver3D& solver, Domain3DUniform* domain)
    {
        PositiveEdgeSnapshot snapshot;
        snapshot.u_xy = solver.u_xpos_ypos_corner_map.at(domain)[0];
        snapshot.v_xy = solver.v_xpos_ypos_corner_map.at(domain)[0];
        snapshot.u_xz = solver.u_xpos_zpos_corner_map.at(domain)[0];
        snapshot.w_xz = solver.w_xpos_zpos_corner_map.at(domain)[0];
        snapshot.v_yz = solver.v_ypos_zpos_corner_map.at(domain)[0];
        snapshot.w_yz = solver.w_ypos_zpos_corner_map.at(domain)[0];
        return snapshot;
    }

    double exact_u_xpos_buffer_00(Domain3DUniform* domain)
    {
        const double x = domain->get_offset_x() + static_cast<double>(domain->get_nx()) * domain->get_hx();
        const double y = domain->get_offset_y() + 0.5 * domain->get_hy();
        const double z = domain->get_offset_z() + 0.5 * domain->get_hz();
        return exact_u(x, y, z);
    }

    double exact_v_xpos_buffer_00(Domain3DUniform* domain)
    {
        const double x = domain->get_offset_x() + (static_cast<double>(domain->get_nx()) + 0.5) * domain->get_hx();
        const double y = domain->get_offset_y();
        const double z = domain->get_offset_z() + 0.5 * domain->get_hz();
        return exact_v(x, y, z);
    }

    double exact_w_xpos_buffer_00(Domain3DUniform* domain)
    {
        const double x = domain->get_offset_x() + (static_cast<double>(domain->get_nx()) + 0.5) * domain->get_hx();
        const double y = domain->get_offset_y() + 0.5 * domain->get_hy();
        const double z = domain->get_offset_z();
        return exact_w(x, y, z);
    }

    PositiveEdgeSnapshot exact_positive_edge_snapshot(Domain3DUniform* domain)
    {
        PositiveEdgeSnapshot snapshot;

        const double ox = domain->get_offset_x();
        const double oy = domain->get_offset_y();
        const double oz = domain->get_offset_z();
        const double hx = domain->get_hx();
        const double hy = domain->get_hy();
        const double hz = domain->get_hz();
        const int    nx = domain->get_nx();
        const int    ny = domain->get_ny();
        const int    nz = domain->get_nz();

        snapshot.u_xy = exact_u(ox + static_cast<double>(nx) * hx,
                                oy + (static_cast<double>(ny) + 0.5) * hy,
                                oz + 0.5 * hz);
        snapshot.v_xy = exact_v(ox + (static_cast<double>(nx) + 0.5) * hx,
                                oy + static_cast<double>(ny) * hy,
                                oz + 0.5 * hz);

        snapshot.u_xz = exact_u(ox + static_cast<double>(nx) * hx,
                                oy + 0.5 * hy,
                                oz + (static_cast<double>(nz) + 0.5) * hz);
        snapshot.w_xz = exact_w(ox + (static_cast<double>(nx) + 0.5) * hx,
                                oy + 0.5 * hy,
                                oz + static_cast<double>(nz) * hz);

        snapshot.v_yz = exact_v(ox + 0.5 * hx,
                                oy + static_cast<double>(ny) * hy,
                                oz + (static_cast<double>(nz) + 0.5) * hz);
        snapshot.w_yz = exact_w(ox + 0.5 * hx,
                                oy + (static_cast<double>(ny) + 0.5) * hy,
                                oz + static_cast<double>(nz) * hz);
        return snapshot;
    }
} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: validation_3d_cross_slot_physical_pressure_mms rank" << std::endl;
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
    env_cfg.debugOutputDir     = "./result/cross_slot_physical_pressure_mms_3d/rank" + std::to_string(rank);

    Geometry3D geo;

    Domain3DUniform A2(rank, rank, rank, 1.0, 1.0, 1.0, "A2");
    Domain3DUniform A1(rank, rank, rank, 1.0, 1.0, 1.0, "A1");
    Domain3DUniform A3(rank, rank, rank, 1.0, 1.0, 1.0, "A3");
    Domain3DUniform A4(rank, rank, rank, 1.0, 1.0, 1.0, "A4");
    Domain3DUniform A5(rank, rank, rank, 1.0, 1.0, 1.0, "A5");

    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);
    geo.connect(&A2, LocationType::YPositive, &A5);
    geo.axis(&A2, LocationType::XNegative);
    geo.axis(&A2, LocationType::YNegative);
    geo.axis(&A2, LocationType::ZNegative);
    geo.global_move_x(-0.5);
    geo.global_move_y(-0.5);
    geo.global_move_z(-0.5);

    Variable3D u("u"), v("v"), w("w"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    w.set_geometry(geo);
    p.set_geometry(geo);

    field3 u_A1, u_A2, u_A3, u_A4, u_A5;
    field3 v_A1, v_A2, v_A3, v_A4, v_A5;
    field3 w_A1, w_A2, w_A3, w_A4, w_A5;
    field3 p_A1, p_A2, p_A3, p_A4, p_A5;

    u.set_x_face_center_field(&A1, u_A1);
    u.set_x_face_center_field(&A2, u_A2);
    u.set_x_face_center_field(&A3, u_A3);
    u.set_x_face_center_field(&A4, u_A4);
    u.set_x_face_center_field(&A5, u_A5);

    v.set_y_face_center_field(&A1, v_A1);
    v.set_y_face_center_field(&A2, v_A2);
    v.set_y_face_center_field(&A3, v_A3);
    v.set_y_face_center_field(&A4, v_A4);
    v.set_y_face_center_field(&A5, v_A5);

    w.set_z_face_center_field(&A1, w_A1);
    w.set_z_face_center_field(&A2, w_A2);
    w.set_z_face_center_field(&A3, w_A3);
    w.set_z_face_center_field(&A4, w_A4);
    w.set_z_face_center_field(&A5, w_A5);

    p.set_center_field(&A1, p_A1);
    p.set_center_field(&A2, p_A2);
    p.set_center_field(&A3, p_A3);
    p.set_center_field(&A4, p_A4);
    p.set_center_field(&A5, p_A5);

    const std::vector<Domain3DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};

    u.set_boundary_type(PDEBoundaryType::Dirichlet);
    u.set_boundary(exact_u);
    u.set_buffer(exact_u);

    v.set_boundary_type(PDEBoundaryType::Dirichlet);
    v.set_boundary(exact_v);
    v.set_buffer(exact_v);

    w.set_boundary_type(PDEBoundaryType::Dirichlet);
    w.set_boundary(exact_w);
    w.set_buffer(exact_w);

    p.set_boundary_type(PDEBoundaryType::Dirichlet);
    p.set_buffer(exact_p);

    fill_exact_velocity_fields(u, v, w);
    fill_exact_base_corner_maps(u, v, w);

    for (auto* domain : domains)
        p.field_map[domain]->clear(0.0);

    ConcatPoissonSolver3D p_solver(&p);
    p_solver.set_parameter(40, 1.0e-12, 200);
    PhysicalPESolver3D ppe_solver(&u, &v, &w, &p, &p_solver, kRho);

    ppe_solver.phys_boundary_update();
    const PositiveEdgeSnapshot a3_after_phys = capture_positive_edge_snapshot(ppe_solver, &A3);
    ppe_solver.nondiag_shared_boundary_update();
    const XPositiveFaceSnapshot a2_xpos_after_nondiag = capture_xpos_face_snapshot(u, v, w, &A2);
    ppe_solver.diag_shared_boundary_update();
    const PositiveEdgeSnapshot a1_after_diag = capture_positive_edge_snapshot(ppe_solver, &A1);
    ppe_solver.calc_rhs();

    const ErrorStats   rhs_error = accumulate_constant_error(p.field_map, exact_rhs());
    const MaxErrorInfo rhs_max   = find_max_constant_error(p.field_map, exact_rhs());

    for (auto* domain : domains)
        p.field_map[domain]->clear(0.0);

    ppe_solver.solve();

    const ErrorStats dudx_error = accumulate_constant_error(ppe_solver.dudx_map, kA);
    const ErrorStats dudy_error = accumulate_constant_error(ppe_solver.dudy_map, kB);
    const ErrorStats dudz_error = accumulate_constant_error(ppe_solver.dudz_map, kC);
    const ErrorStats dvdx_error = accumulate_constant_error(ppe_solver.dvdx_map, kD);
    const ErrorStats dvdy_error = accumulate_constant_error(ppe_solver.dvdy_map, kE);
    const ErrorStats dvdz_error = accumulate_constant_error(ppe_solver.dvdz_map, kF);
    const ErrorStats dwdx_error = accumulate_constant_error(ppe_solver.dwdx_map, kG);
    const ErrorStats dwdy_error = accumulate_constant_error(ppe_solver.dwdy_map, kH);
    const ErrorStats dwdz_error = accumulate_constant_error(ppe_solver.dwdz_map, kI);
    const ErrorStats p_error    = accumulate_field_error(p, p.field_map, exact_p);

    const MaxErrorInfo dudy_max = find_max_constant_error(ppe_solver.dudy_map, kB);
    const MaxErrorInfo dudz_max = find_max_constant_error(ppe_solver.dudz_map, kC);
    const MaxErrorInfo dvdx_max = find_max_constant_error(ppe_solver.dvdx_map, kD);
    const MaxErrorInfo dvdz_max = find_max_constant_error(ppe_solver.dvdz_map, kF);
    const MaxErrorInfo dwdx_max = find_max_constant_error(ppe_solver.dwdx_map, kG);
    const MaxErrorInfo dwdy_max = find_max_constant_error(ppe_solver.dwdy_map, kH);
    const MaxErrorInfo p_max    = find_max_field_error(p, p.field_map, exact_p);

    const PositiveEdgeSnapshot a3_phys_expected = exact_positive_edge_snapshot(&A3);
    const PositiveEdgeSnapshot a1_diag_expected = exact_positive_edge_snapshot(&A1);

    const double a2_u_xpos_expected = exact_u_xpos_buffer_00(&A2);
    const double a2_v_xpos_expected = exact_v_xpos_buffer_00(&A2);
    const double a2_w_xpos_expected = exact_w_xpos_buffer_00(&A2);

    const double a3_u_xy_diff = a3_after_phys.u_xy - a3_phys_expected.u_xy;
    const double a3_v_xy_diff = a3_after_phys.v_xy - a3_phys_expected.v_xy;
    const double a3_u_xz_diff = a3_after_phys.u_xz - a3_phys_expected.u_xz;
    const double a3_w_xz_diff = a3_after_phys.w_xz - a3_phys_expected.w_xz;
    const double a3_v_yz_diff = a3_after_phys.v_yz - a3_phys_expected.v_yz;
    const double a3_w_yz_diff = a3_after_phys.w_yz - a3_phys_expected.w_yz;

    const double a2_u_xpos_diff = a2_xpos_after_nondiag.u - a2_u_xpos_expected;
    const double a2_v_xpos_diff = a2_xpos_after_nondiag.v - a2_v_xpos_expected;
    const double a2_w_xpos_diff = a2_xpos_after_nondiag.w - a2_w_xpos_expected;

    const double a1_u_xy_diff = a1_after_diag.u_xy - a1_diag_expected.u_xy;
    const double a1_v_xy_diff = a1_after_diag.v_xy - a1_diag_expected.v_xy;

    std::cout << std::setprecision(16);
    std::cout << "ppe_rhs_error: l2 = " << rhs_error.l2 << ", linf = " << rhs_error.linf << std::endl;
    std::cout << "dudx_error: l2 = " << dudx_error.l2 << ", linf = " << dudx_error.linf << std::endl;
    std::cout << "dudy_error: l2 = " << dudy_error.l2 << ", linf = " << dudy_error.linf << std::endl;
    std::cout << "dudz_error: l2 = " << dudz_error.l2 << ", linf = " << dudz_error.linf << std::endl;
    std::cout << "dvdx_error: l2 = " << dvdx_error.l2 << ", linf = " << dvdx_error.linf << std::endl;
    std::cout << "dvdy_error: l2 = " << dvdy_error.l2 << ", linf = " << dvdy_error.linf << std::endl;
    std::cout << "dvdz_error: l2 = " << dvdz_error.l2 << ", linf = " << dvdz_error.linf << std::endl;
    std::cout << "dwdx_error: l2 = " << dwdx_error.l2 << ", linf = " << dwdx_error.linf << std::endl;
    std::cout << "dwdy_error: l2 = " << dwdy_error.l2 << ", linf = " << dwdy_error.linf << std::endl;
    std::cout << "dwdz_error: l2 = " << dwdz_error.l2 << ", linf = " << dwdz_error.linf << std::endl;
    std::cout << "ppe_solution_error: l2 = " << p_error.l2 << ", linf = " << p_error.linf << std::endl;

    std::cout << "rhs_max: domain = " << rhs_max.domain_name << ", i = " << rhs_max.i << ", j = " << rhs_max.j
              << ", k = " << rhs_max.k << ", x = " << rhs_max.x << ", y = " << rhs_max.y << ", z = " << rhs_max.z
              << ", diff = " << rhs_max.diff << std::endl;
    std::cout << "dudy_max: domain = " << dudy_max.domain_name << ", i = " << dudy_max.i << ", j = " << dudy_max.j
              << ", k = " << dudy_max.k << ", x = " << dudy_max.x << ", y = " << dudy_max.y << ", z = " << dudy_max.z
              << ", diff = " << dudy_max.diff << std::endl;
    std::cout << "dudz_max: domain = " << dudz_max.domain_name << ", i = " << dudz_max.i << ", j = " << dudz_max.j
              << ", k = " << dudz_max.k << ", x = " << dudz_max.x << ", y = " << dudz_max.y << ", z = " << dudz_max.z
              << ", diff = " << dudz_max.diff << std::endl;
    std::cout << "dvdx_max: domain = " << dvdx_max.domain_name << ", i = " << dvdx_max.i << ", j = " << dvdx_max.j
              << ", k = " << dvdx_max.k << ", x = " << dvdx_max.x << ", y = " << dvdx_max.y << ", z = " << dvdx_max.z
              << ", diff = " << dvdx_max.diff << std::endl;
    std::cout << "dvdz_max: domain = " << dvdz_max.domain_name << ", i = " << dvdz_max.i << ", j = " << dvdz_max.j
              << ", k = " << dvdz_max.k << ", x = " << dvdz_max.x << ", y = " << dvdz_max.y << ", z = " << dvdz_max.z
              << ", diff = " << dvdz_max.diff << std::endl;
    std::cout << "dwdx_max: domain = " << dwdx_max.domain_name << ", i = " << dwdx_max.i << ", j = " << dwdx_max.j
              << ", k = " << dwdx_max.k << ", x = " << dwdx_max.x << ", y = " << dwdx_max.y << ", z = " << dwdx_max.z
              << ", diff = " << dwdx_max.diff << std::endl;
    std::cout << "dwdy_max: domain = " << dwdy_max.domain_name << ", i = " << dwdy_max.i << ", j = " << dwdy_max.j
              << ", k = " << dwdy_max.k << ", x = " << dwdy_max.x << ", y = " << dwdy_max.y << ", z = " << dwdy_max.z
              << ", diff = " << dwdy_max.diff << std::endl;
    std::cout << "p_max: domain = " << p_max.domain_name << ", i = " << p_max.i << ", j = " << p_max.j
              << ", k = " << p_max.k << ", x = " << p_max.x << ", y = " << p_max.y << ", z = " << p_max.z
              << ", diff = " << p_max.diff << std::endl;

    std::cout << "A3_positive_edges_after_phys: "
              << "u_xy = " << a3_after_phys.u_xy << ", u_xy_expected = " << a3_phys_expected.u_xy
              << ", u_xy_diff = " << a3_u_xy_diff << ", "
              << "v_xy = " << a3_after_phys.v_xy << ", v_xy_expected = " << a3_phys_expected.v_xy
              << ", v_xy_diff = " << a3_v_xy_diff << ", "
              << "u_xz = " << a3_after_phys.u_xz << ", u_xz_expected = " << a3_phys_expected.u_xz
              << ", u_xz_diff = " << a3_u_xz_diff << ", "
              << "w_xz = " << a3_after_phys.w_xz << ", w_xz_expected = " << a3_phys_expected.w_xz
              << ", w_xz_diff = " << a3_w_xz_diff << ", "
              << "v_yz = " << a3_after_phys.v_yz << ", v_yz_expected = " << a3_phys_expected.v_yz
              << ", v_yz_diff = " << a3_v_yz_diff << ", "
              << "w_yz = " << a3_after_phys.w_yz << ", w_yz_expected = " << a3_phys_expected.w_yz
              << ", w_yz_diff = " << a3_w_yz_diff << std::endl;

    std::cout << "A2_xpos_face_after_nondiag: "
              << "u = " << a2_xpos_after_nondiag.u << ", u_expected = " << a2_u_xpos_expected
              << ", u_diff = " << a2_u_xpos_diff << ", "
              << "v = " << a2_xpos_after_nondiag.v << ", v_expected = " << a2_v_xpos_expected
              << ", v_diff = " << a2_v_xpos_diff << ", "
              << "w = " << a2_xpos_after_nondiag.w << ", w_expected = " << a2_w_xpos_expected
              << ", w_diff = " << a2_w_xpos_diff << std::endl;

    std::cout << "A1_xy_edge_after_diag: "
              << "u_xy = " << a1_after_diag.u_xy << ", u_xy_expected = " << a1_diag_expected.u_xy
              << ", u_xy_diff = " << a1_u_xy_diff << ", "
              << "v_xy = " << a1_after_diag.v_xy << ", v_xy_expected = " << a1_diag_expected.v_xy
              << ", v_xy_diff = " << a1_v_xy_diff << std::endl;

    const double tol   = 1.0e-12;
    const double p_tol = 1.0e-9;
    const bool   pass  = rhs_error.linf < tol && dudx_error.linf < tol && dudy_error.linf < tol &&
                       dudz_error.linf < tol && dvdx_error.linf < tol && dvdy_error.linf < tol &&
                       dvdz_error.linf < tol && dwdx_error.linf < tol && dwdy_error.linf < tol &&
                       dwdz_error.linf < tol && p_error.linf < p_tol;

    return pass ? 0 : 1;
}
