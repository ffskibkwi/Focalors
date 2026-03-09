#include "base/config.h"
#include "base/domain/domain3d.h"
#include "base/domain/geometry3d.h"
#include "base/domain/variable3d.h"
#include "base/field/field3.h"
#include "base/location_boundary.h"
#include "ibm/ib_solver_3d.h"
#include "ibm/particles_coordinate_map_3d.h"
#include "poisson_base/base/math/compare.h"

#include <cmath>
#include <iostream>

// 初始化三维速度场：u = sin(x), v = cos(y), w = sin(z)
static void init_velocity_sin(Variable3D& u, Variable3D& v, Variable3D& w)
{
    u.set_value([](double x, double /*y*/, double /*z*/) { return std::sin(x); });
    v.set_value([](double /*x*/, double y, double /*z*/) { return std::cos(y); });
    w.set_value([](double /*x*/, double /*y*/, double z) { return std::sin(z); });
}

// 采样并输出几个点的速度值，用于验证 IBM solver 工作正常
static void sample_and_print_velocity(const Variable3D& u, const Variable3D& v, const Variable3D& w, const std::string& label)
{
    std::cout << "[" << label << "] Velocity samples:\n";
    std::vector<std::tuple<double, double, double>> sample_points = {
        {0.5, 0.5, 0.5},    // Center
        {0.3, 0.5, 0.5},    // Left of center
        {0.7, 0.5, 0.5},    // Right of center
        {0.5, 0.3, 0.5},    // Below center
        {0.5, 0.7, 0.5},    // Above center
        {0.5, 0.5, 0.3},    // Front of center
        {0.5, 0.5, 0.7},    // Back of center
        {0.25, 0.25, 0.25}, // Corner
        {0.75, 0.75, 0.75}, // Opposite corner
    };

    for (const auto& [x, y, z] : sample_points)
    {
        double u_val = 0.0, v_val = 0.0, w_val = 0.0;
        bool u_ok = sample_u_at(u, x, y, z, u_val);
        bool v_ok = sample_v_at(v, x, y, z, v_val);
        bool w_ok = sample_w_at(w, x, y, z, w_val);
        std::cout << "    (" << x << "," << y << "," << z << ") u=" << (u_ok ? std::to_string(u_val) : "N/A")
                  << " v=" << (v_ok ? std::to_string(v_val) : "N/A")
                  << " w=" << (w_ok ? std::to_string(w_val) : "N/A") << "\n";
    }
}

// 在给定物理坐标 (x,y,z) 上，从 Variable3D 中采样 u/v/w（面心变量）
static bool sample_u_at(const Variable3D& u, double x, double y, double z, double& value)
{
    Geometry3D* geo = u.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();

        if (x < ox || x > ox + lx || y < oy || y > oy + ly || z < oz || z > oz + lz)
            continue;

        auto&  f  = *u.field_map.at(d);
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // u: x-face centered, x = ox + i*hx, y = oy + (j+0.5)*hy, z = oz + (k+0.5)*hz
        int i = static_cast<int>(std::round((x - ox) / hx));
        int j = static_cast<int>(std::round((y - oy) / hy - 0.5));
        int k = static_cast<int>(std::round((z - oz) / hz - 0.5));

        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            return true;
        }
    }
    return false;
}

static bool sample_v_at(const Variable3D& v, double x, double y, double z, double& value)
{
    Geometry3D* geo = v.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();

        if (x < ox || x > ox + lx || y < oy || y > oy + ly || z < oz || z > oz + lz)
            continue;

        auto&  f  = *v.field_map.at(d);
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // v: y-face centered, x = ox + (i+0.5)*hx, y = oy + j*hy, z = oz + (k+0.5)*hz
        int i = static_cast<int>(std::round((x - ox) / hx - 0.5));
        int j = static_cast<int>(std::round((y - oy) / hy));
        int k = static_cast<int>(std::round((z - oz) / hz - 0.5));

        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            return true;
        }
    }
    return false;
}

static bool sample_w_at(const Variable3D& w, double x, double y, double z, double& value)
{
    Geometry3D* geo = w.geometry;
    for (auto* d : geo->domains)
    {
        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double lx = d->get_lx();
        double ly = d->get_ly();
        double lz = d->get_lz();

        if (x < ox || x > ox + lx || y < oy || y > oy + ly || z < oz || z > oz + lz)
            continue;

        auto&  f  = *w.field_map.at(d);
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // w: z-face centered, x = ox + (i+0.5)*hx, y = oy + (j+0.5)*hy, z = oz + k*hz
        int i = static_cast<int>(std::round((x - ox) / hx - 0.5));
        int j = static_cast<int>(std::round((y - oy) / hy - 0.5));
        int k = static_cast<int>(std::round((z - oz) / hz));

        if (i >= 0 && i < f.get_nx() && j >= 0 && j < f.get_ny() && k >= 0 && k < f.get_nz())
        {
            value = f(i, j, k);
            return true;
        }
    }
    return false;
}

// 真正对比：单域 (u_single,v_single,w_single) 与 多域 (u_multi,v_multi,w_multi)
static void compare_velocity_fields_phys(const Variable3D& u_single,
                                         const Variable3D& v_single,
                                         const Variable3D& w_single,
                                         const Variable3D& u_multi,
                                         const Variable3D& v_multi,
                                         const Variable3D& w_multi,
                                         double            abs_eps,
                                         double            rel_eps)
{
    std::cout << "[IBM 3D] Comparing velocities between single-domain and two-domain setups (by physical coords)...\n";

    Geometry3D* geo_single = u_single.geometry;

    for (auto* d : geo_single->domains)
    {
        auto& fu = *u_single.field_map.at(d);
        auto& fv = *v_single.field_map.at(d);
        auto& fw = *w_single.field_map.at(d);

        double ox = d->get_offset_x();
        double oy = d->get_offset_y();
        double oz = d->get_offset_z();
        double hx = d->get_hx();
        double hy = d->get_hy();
        double hz = d->get_hz();

        // u: x-face centered
        for (int i = 0; i < fu.get_nx(); ++i)
        {
            for (int j = 0; j < fu.get_ny(); ++j)
            {
                for (int k = 0; k < fu.get_nz(); ++k)
                {
                    double x = ox + i * hx;
                    double y = oy + (j + 0.5) * hy;
                    double z = oz + (k + 0.5) * hz;

                    double a = fu(i, j, k);
                    double b = 0.0;
                    if (!sample_u_at(u_multi, x, y, z, b) || !approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                    {
                        std::cout << "[u] mismatch at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << ", multi=" << b << "\n";
                    }
                }
            }
        }

        // v: y-face centered
        for (int i = 0; i < fv.get_nx(); ++i)
        {
            for (int j = 0; j < fv.get_ny(); ++j)
            {
                for (int k = 0; k < fv.get_nz(); ++k)
                {
                    double x = ox + (i + 0.5) * hx;
                    double y = oy + j * hy;
                    double z = oz + (k + 0.5) * hz;

                    double a = fv(i, j, k);
                    double b = 0.0;
                    if (!sample_v_at(v_multi, x, y, z, b) || !approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                    {
                        std::cout << "[v] mismatch at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << ", multi=" << b << "\n";
                    }
                }
            }
        }

        // w: z-face centered
        for (int i = 0; i < fw.get_nx(); ++i)
        {
            for (int j = 0; j < fw.get_ny(); ++j)
            {
                for (int k = 0; k < fw.get_nz(); ++k)
                {
                    double x = ox + (i + 0.5) * hx;
                    double y = oy + (j + 0.5) * hy;
                    double z = oz + k * hz;

                    double a = fw(i, j, k);
                    double b = 0.0;
                    if (!sample_w_at(w_multi, x, y, z, b) || !approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                    {
                        std::cout << "[w] mismatch at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << ", multi=" << b << "\n";
                    }
                }
            }
        }
    }
}

int main(int /*argc*/, char* /*argv*/[])
{
    constexpr int    NX_TOTAL = 8;
    constexpr int    NY_TOTAL = 8;
    constexpr int    NZ_TOTAL = 8;
    constexpr double LX       = 1.0;
    constexpr double LY       = 1.0;
    constexpr double LZ       = 1.0;

    const double hx = LX / NX_TOTAL;
    const double hy = LY / NY_TOTAL;
    const double hz = LZ / NZ_TOTAL;

    // ------------------ 情况 1：单个 domain ------------------
    Geometry3D geo_single;

    Domain3DUniform d_single(NX_TOTAL, NY_TOTAL, NZ_TOTAL, LX, LY, LZ, "Single3D");
    geo_single.add_domain(&d_single);
    geo_single.set_global_spatial_step(hx, hy, hz);

    Variable3D u_single("u_single"), v_single("v_single"), w_single("w_single");
    u_single.set_geometry(geo_single);
    v_single.set_geometry(geo_single);
    w_single.set_geometry(geo_single);

    field3 u_single_f, v_single_f, w_single_f;
    u_single.set_x_face_center_field(&d_single, u_single_f);
    v_single.set_y_face_center_field(&d_single, v_single_f);
    w_single.set_z_face_center_field(&d_single, w_single_f);

    init_velocity_sin(u_single, v_single, w_single);

    // IBM 粒子：在立方体中间放一个球
    const double cx = 0.5 * LX;
    const double cy = 0.5 * LY;
    const double cz = 0.5 * LZ;
    const double r  = 0.15 * LX;

    PCoordMap3D coord_map_single;
    coord_map_single.add_sphere(400, r, cx, cy, cz);
    coord_map_single.generate_map(&geo_single);

    auto coord_map_single_raw = coord_map_single.get_map();

    ImmersedBoundarySolver3D ibm_single(&u_single, &v_single, &w_single, coord_map_single_raw);
    ibm_single.set_parameters(coord_map_single.get_h(), hx);

    // Sample before IBM solve
    sample_and_print_velocity(u_single, v_single, w_single, "Single Domain (Before IBM)");

    ibm_single.solve();

    // Sample after IBM solve
    sample_and_print_velocity(u_single, v_single, w_single, "Single Domain (After IBM)");

    std::cout << "[DEBUG] Single domain u field sample at (0.5,0.0625,0.0625):\n";
    double test_val;
    if (sample_u_at(u_single, 0.5, 0.0625, 0.0625, test_val))
    {
        std::cout << "    Value = " << test_val << "\n";
    }
    else
    {
        std::cout << "    FAILED to sample!\n";
    }

    // ------------------ 情况 2：两个 domain 拼接（沿 x 方向）------------------
    Geometry3D geo_multi;

    Domain3DUniform d_left(NX_TOTAL / 2, NY_TOTAL, NZ_TOTAL, LX / 2.0, LY, LZ, "Left3D");
    Domain3DUniform d_right(NX_TOTAL / 2, NY_TOTAL, NZ_TOTAL, LX / 2.0, LY, LZ, "Right3D");

    geo_multi.add_domain(&d_left);
    geo_multi.add_domain(&d_right);
    geo_multi.connect(&d_left, LocationType::XPositive, &d_right);
    geo_multi.set_global_spatial_step(hx, hy, hz);

    // Set the axis to automatically compute offsets
    geo_multi.axis(&d_left, LocationType::XNegative);

    Variable3D u_multi("u_multi"), v_multi("v_multi"), w_multi("w_multi");
    u_multi.set_geometry(geo_multi);
    v_multi.set_geometry(geo_multi);
    w_multi.set_geometry(geo_multi);

    field3 u_left_f, u_right_f;
    field3 v_left_f, v_right_f;
    field3 w_left_f, w_right_f;

    u_multi.set_x_face_center_field(&d_left, u_left_f);
    u_multi.set_x_face_center_field(&d_right, u_right_f);
    v_multi.set_y_face_center_field(&d_left, v_left_f);
    v_multi.set_y_face_center_field(&d_right, v_right_f);
    w_multi.set_z_face_center_field(&d_left, w_left_f);
    w_multi.set_z_face_center_field(&d_right, w_right_f);

    init_velocity_sin(u_multi, v_multi, w_multi);

    PCoordMap3D coord_map_multi;
    coord_map_multi.add_sphere(400, r, cx, cy, cz);
    coord_map_multi.generate_map(&geo_multi);

    auto coord_map_multi_raw = coord_map_multi.get_map();

    ImmersedBoundarySolver3D ibm_multi(&u_multi, &v_multi, &w_multi, coord_map_multi_raw);
    ibm_multi.set_parameters(coord_map_multi.get_h(), hx);

    // Sample before IBM solve
    sample_and_print_velocity(u_multi, v_multi, w_multi, "Multi Domain (Before IBM)");

    ibm_multi.solve();

    // Sample after IBM solve
    sample_and_print_velocity(u_multi, v_multi, w_multi, "Multi Domain (After IBM)");

    // 按物理坐标真正对比单域结果和双域结果
    compare_velocity_fields_phys(u_single, v_single, w_single, u_multi, v_multi, w_multi, 1e-10, 1e-8);

    std::cout << "[IBM 3D] Validation finished.\n";

    return 0;
}
