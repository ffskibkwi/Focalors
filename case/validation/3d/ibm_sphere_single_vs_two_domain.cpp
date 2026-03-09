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

        // Debug output for boundary cases
        static int debug_counter = 0;
        if (debug_counter < 10 && (i < 0 || i >= f.get_nx() || j < 0 || j >= f.get_ny() || k < 0 || k >= f.get_nz()))
        {
            std::cout << "[DEBUG sample_u_at] Point (" << x << "," << y << "," << z << ") in domain '" << d->get_name()
                      << "'\n";
            std::cout << "    Computed indices: i=" << i << ", j=" << j << ", k=" << k << "\n";
            std::cout << "    Field size: nx=" << f.get_nx() << ", ny=" << f.get_ny() << ", nz=" << f.get_nz() << "\n";
            std::cout << "    Domain bounds: x=[" << ox << "," << ox + lx << "], y=[" << oy << "," << oy + ly
                      << "], z=[" << oz << "," << oz + lz << "]\n";
            std::cout << "    Grid spacing: hx=" << hx << ", hy=" << hy << ", hz=" << hz << "\n";
            debug_counter++;
        }

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

    int mismatch_count = 0;
    int total_checks   = 0;

    Geometry3D* geo_single = u_single.geometry;
    Geometry3D* geo_multi  = u_multi.geometry;

    // Debug: print domain info
    std::cout << "[DEBUG] Single domain count: " << geo_single->domains.size() << "\n";
    for (auto* d : geo_single->domains)
    {
        std::cout << "[DEBUG] Single domain '" << d->get_name() << "': offset=(" << d->get_offset_x() << ","
                  << d->get_offset_y() << "," << d->get_offset_z() << "), size=(" << d->get_nx() << "," << d->get_ny()
                  << "," << d->get_nz() << ")\n";
    }

    std::cout << "[DEBUG] Multi domain count: " << geo_multi->domains.size() << "\n";
    for (auto* d : geo_multi->domains)
    {
        std::cout << "[DEBUG] Multi domain '" << d->get_name() << "': offset=(" << d->get_offset_x() << ","
                  << d->get_offset_y() << "," << d->get_offset_z() << "), size=(" << d->get_nx() << "," << d->get_ny()
                  << "," << d->get_nz() << ")\n";
    }

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
                    total_checks++;
                    if (!sample_u_at(u_multi, x, y, z, b))
                    {
                        mismatch_count++;
                        std::cout << "[u] CANNOT SAMPLE at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << "\n";

                        // Debug: find which domain should contain this point
                        bool found = false;
                        for (auto* md : geo_multi->domains)
                        {
                            double mox = md->get_offset_x();
                            double moy = md->get_offset_y();
                            double moz = md->get_offset_z();
                            double mlx = md->get_lx();
                            double mly = md->get_ly();
                            double mlz = md->get_lz();

                            if (x >= mox && x <= mox + mlx && y >= moy && y <= moy + mly && z >= moz && z <= moz + mlz)
                            {
                                std::cout << "    [DEBUG] Point is in multi-domain '" << md->get_name() << "'\n";
                                std::cout << "    [DEBUG] Domain bounds: x=[" << mox << "," << mox + mlx << "], y=["
                                          << moy << "," << moy + mly << "], z=[" << moz << "," << moz + mlz << "]\n";

                                auto&  mf  = *u_multi.field_map.at(md);
                                double mhx = md->get_hx();
                                double mhy = md->get_hy();
                                double mhz = md->get_hz();

                                int mi = static_cast<int>(std::round((x - mox) / mhx));
                                int mj = static_cast<int>(std::round((y - moy) / mhy - 0.5));
                                int mk = static_cast<int>(std::round((z - moz) / mhz - 0.5));

                                std::cout << "    [DEBUG] Computed indices: i=" << mi << ", j=" << mj << ", k=" << mk
                                          << "\n";
                                std::cout << "    [DEBUG] Field size: nx=" << mf.get_nx() << ", ny=" << mf.get_ny()
                                          << ", nz=" << mf.get_nz() << "\n";
                                found = true;
                                break;
                            }
                        }
                        if (!found)
                        {
                            std::cout << "    [DEBUG] Point NOT in any multi-domain!\n";
                        }
                    }
                    else if (!approximatelyEqualAbsRel(a, b, abs_eps, rel_eps))
                    {
                        mismatch_count++;
                        std::cout << "[u] VALUE MISMATCH at phys(" << x << "," << y << "," << z << ") single=" << a
                                  << ", multi=" << b << ", diff=" << (a - b) << "\n";
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

    std::cout << "[IBM 3D] Total checks: " << total_checks << ", Mismatches: " << mismatch_count << "\n";
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

    ibm_single.solve();

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

    std::cout << "[DEBUG] Multi-domain field values before IB:\n";
    for (auto* d : geo_multi.domains)
    {
        auto& f = *u_multi.field_map.at(d);
        std::cout << "    Domain '" << d->get_name() << "': sample values\n";
        for (int k = 0; k < std::min(3, f.get_nz()); ++k)
        {
            for (int j = 0; j < std::min(3, f.get_ny()); ++j)
            {
                for (int i = 0; i < std::min(3, f.get_nx()); ++i)
                {
                    std::cout << "        (" << i << "," << j << "," << k << ") = " << f(i, j, k) << "\n";
                }
            }
        }
    }

    PCoordMap3D coord_map_multi;
    coord_map_multi.add_sphere(400, r, cx, cy, cz);
    coord_map_multi.generate_map(&geo_multi);

    auto coord_map_multi_raw = coord_map_multi.get_map();

    ImmersedBoundarySolver3D ibm_multi(&u_multi, &v_multi, &w_multi, coord_map_multi_raw);
    ibm_multi.set_parameters(coord_map_multi.get_h(), hx);

    ibm_multi.solve();

    std::cout << "[DEBUG] Multi domain u field sample at (0.5,0.0625,0.0625):\n";
    double test_val_multi;
    if (sample_u_at(u_multi, 0.5, 0.0625, 0.0625, test_val_multi))
    {
        std::cout << "    Value = " << test_val_multi << "\n";
    }
    else
    {
        std::cout << "    FAILED to sample!\n";
    }

    // 按物理坐标真正对比单域结果和双域结果
    compare_velocity_fields_phys(u_single, v_single, w_single, u_multi, v_multi, w_multi, 1e-10, 1e-8);

    std::cout << "[IBM 3D] Validation finished.\n";

    return 0;
}
