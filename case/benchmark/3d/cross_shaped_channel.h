#pragma once

#include "base/config.h"
#include "base/location_boundary.h"
#include "io/case_base.hpp"

#include <array>
#include <string>

/**
 * @brief Case class for Cross-Shaped Channel 3D simulation.
 *
 * Manages parameters for the 3D cross-shaped channel geometry and flow conditions.
 * Keeps interface-compatible placeholders for future Non-Newtonian and MHD modules.
 */
class CrossShapedChannel3DCase : public CaseBase
{
public:
    static constexpr double POWERLAW_ETA_C = 0.00604;
    static constexpr double CARREAU_ETA_C  = 0.00596;
    static constexpr double POWERLAW_N      = 0.708;
    static constexpr double CARREAU_N       = 0.3568;

    CrossShapedChannel3DCase(int argc, char* argv[])
        : CaseBase(argc, argv)
    {}

    void read_paras() override
    {
        CaseBase::read_paras();

        // Grid and Geometry Parameters
        IO::read_number(para_map, "h", h);
        IO::read_number(para_map, "lx_2", lx_2);
        IO::read_number(para_map, "ly_2", ly_2);
        IO::read_number(para_map, "lz_2", lz_2);

        IO::read_number(para_map, "lx_1", lx_1);
        if (!IO::read_number(para_map, "ly_1", ly_1))
            ly_1 = ly_2;
        if (!IO::read_number(para_map, "lz_1", lz_1))
            lz_1 = lz_2;

        IO::read_number(para_map, "lx_3", lx_3);
        if (!IO::read_number(para_map, "ly_3", ly_3))
            ly_3 = ly_2;
        if (!IO::read_number(para_map, "lz_3", lz_3))
            lz_3 = lz_2;

        if (!IO::read_number(para_map, "lx_4", lx_4))
            lx_4 = lx_2;
        IO::read_number(para_map, "ly_4", ly_4);
        if (!IO::read_number(para_map, "lz_4", lz_4))
            lz_4 = lz_2;

        if (!IO::read_number(para_map, "lx_5", lx_5))
            lx_5 = lx_2;
        IO::read_number(para_map, "ly_5", ly_5);
        if (!IO::read_number(para_map, "lz_5", lz_5))
            lz_5 = lz_2;

        // Physics Parameters
        IO::read_number(para_map, "Re", Re);
        IO::read_number(para_map, "U0", U0);

        // Time Stepping
        IO::read_number(para_map, "dt_factor", dt_factor);
        IO::read_number(para_map, "T_total", T_total);
        IO::read_number(para_map, "pv_output_step", pv_output_step);
        IO::read_number(para_map, "vtk_output_step", vtk_output_step);

        // Solver Parameters
        IO::read_number(para_map, "gmres_m", gmres_m);
        IO::read_number(para_map, "gmres_tol", gmres_tol);
        IO::read_number(para_map, "gmres_max_iter", gmres_max_iter);

        // Non-Newtonian extension slots
        IO::read_number(para_map, "model_type", model_type);
        const bool has_n_index = IO::read_number(para_map, "n_index", n_index);
        if (!has_n_index)
        {
            n_index = (model_type == 2) ? CARREAU_N : POWERLAW_N;
        }
        IO::read_number(para_map, "mu_0", mu_0);
        IO::read_number(para_map, "mu_inf", mu_inf);
        IO::read_number(para_map, "lambda", lambda);
        IO::read_number(para_map, "a", a);

        // Non-Newtonian parameters actually used by config setters
        IO::read_number(para_map, "k_pl", k_pl);
        IO::read_number(para_map, "mu_min_pl", mu_min_pl);
        IO::read_number(para_map, "mu_max_pl", mu_max_pl);
        if (!IO::read_number(para_map, "mu_ref", mu_ref))
        {
            mu_ref = (model_type == 2) ? CARREAU_ETA_C : POWERLAW_ETA_C;
        }
        IO::read_bool(para_map, "use_dimensionless_viscosity", use_dimensionless_viscosity);

        // MHD extension slots
        IO::read_number(para_map, "Ha", Ha);
        IO::read_number(para_map, "Bx", Bx);
        IO::read_number(para_map, "By", By);
        IO::read_number(para_map, "Bz", Bz);
        IO::read_bool(para_map, "enable_mhd", enable_mhd);
    }

    void init_nonnewton_config(PhysicsConfig& physics_cfg) const
    {
        physics_cfg.set_model_type(model_type);
        if (model_type == 1)
        {
            physics_cfg.set_power_law_dimensionless(
                k_pl, n_index, Re, mu_ref, use_dimensionless_viscosity, mu_min_pl, mu_max_pl);
        }
        else if (model_type == 2)
        {
            physics_cfg.set_carreau_dimensionless(
                mu_0, mu_inf, a, lambda, n_index, Re, mu_ref, use_dimensionless_viscosity);
        }
    }

    void init_mhd_config(PhysicsConfig& physics_cfg) const
    {
        physics_cfg.set_enable_mhd(enable_mhd);
        physics_cfg.set_Ha(Ha);
        physics_cfg.set_magnetic_field(Bx, By, Bz);
    }

    bool record_paras() override
    {
        if (!CaseBase::record_paras())
            return false;

        paras_record.record("h", h)
            .record("lx_2", lx_2)
            .record("ly_2", ly_2)
            .record("lz_2", lz_2)
            .record("lx_1", lx_1)
            .record("ly_1", ly_1)
            .record("lz_1", lz_1)
            .record("lx_3", lx_3)
            .record("ly_3", ly_3)
            .record("lz_3", lz_3)
            .record("lx_4", lx_4)
            .record("ly_4", ly_4)
            .record("lz_4", lz_4)
            .record("lx_5", lx_5)
            .record("ly_5", ly_5)
            .record("lz_5", lz_5)
            .record("Re", Re)
            .record("U0", U0)
            .record("dt_factor", dt_factor)
            .record("T_total", T_total)
            .record("pv_output_step", pv_output_step)
            .record("vtk_output_step", vtk_output_step)
            .record("gmres_m", gmres_m)
            .record("gmres_tol", gmres_tol)
            .record("gmres_max_iter", gmres_max_iter)
            .record("model_type", model_type)
            .record("n_index", n_index)
            .record("mu_0", mu_0)
            .record("mu_inf", mu_inf)
            .record("lambda", lambda)
            .record("a", a)
            .record("k_pl", k_pl)
            .record("mu_min_pl", mu_min_pl)
            .record("mu_max_pl", mu_max_pl)
            .record("mu_ref", mu_ref)
            .record("use_dimensionless_viscosity", use_dimensionless_viscosity ? 1 : 0)
            .record("Ha", Ha)
            .record("Bx", Bx)
            .record("By", By)
            .record("Bz", Bz)
            .record("enable_mhd", enable_mhd ? 1 : 0);

        return true;
    }

public:
    struct DomainTopology3D
    {
        std::string center       = "A2";
        std::string inlet_left   = "A1";
        std::string inlet_right  = "A3";
        std::string outlet_front = "A4";
        std::string outlet_back  = "A5";

        LocationType link_a2_a1 = LocationType::Left;
        LocationType link_a2_a3 = LocationType::Right;
        LocationType link_a2_a4 = LocationType::Front;
        LocationType link_a2_a5 = LocationType::Back;

        LocationType outlet_a4_loc = LocationType::Front;
        LocationType outlet_a5_loc = LocationType::Back;
    };

    struct MultiPhysicsSlots3D
    {
        bool reserve_nonnewton_slots = true;
        bool reserve_mhd_slots       = true;

        std::string                mu_name      = "mu";
        std::array<std::string, 6> stress_names = {"tau_xx", "tau_yy", "tau_zz", "tau_xy", "tau_xz", "tau_yz"};

        std::string                phi_name              = "phi";
        std::array<std::string, 3> current_density_names = {"jx", "jy", "jz"};
    };

    // Grid spacing
    double h = 0.02;

    // Geometry dimensions (A2 is center block)
    double lx_2 = 1.0;
    double ly_2 = 1.0;
    double lz_2 = 1.0;

    // A1 (left arm)
    double lx_1 = 8.0;
    double ly_1 = ly_2;
    double lz_1 = lz_2;

    // A3 (right arm)
    double lx_3 = 8.0;
    double ly_3 = ly_2;
    double lz_3 = lz_2;

    // A4 (front arm)
    double lx_4 = lx_2;
    double ly_4 = 8.0;
    double lz_4 = lz_2;

    // A5 (back arm)
    double lx_5 = lx_2;
    double ly_5 = 8.0;
    double lz_5 = lz_2;

    // Physics
    double Re = 100.0;
    double U0 = 1.0;

    // Time stepping
    double dt_factor       = 0.05; // dt = dt_factor * h
    double T_total         = 100.0;
    int    pv_output_step  = 0; // reserved for compatibility
    int    vtk_output_step = 200;

    // Solver settings
    int    gmres_m        = 30;
    double gmres_tol      = 1.e-6;
    int    gmres_max_iter = 1000;

    // Non-Newtonian extension slots
    int    model_type = 1;       // 0: Newtonian, 1: Power Law (paper default), 2: Carreau
    double n_index    = POWERLAW_N; // Power-law index (paper)
    double mu_0       = 0.056;   // Zero-shear viscosity
    double mu_inf     = 0.00345; // Infinite-shear viscosity
    double lambda     = 3.313;   // Relaxation time (paper Bird-Carreau)
    double a          = 2.0;     // Carreau model parameter
    double k_pl       = 0.017;   // Power-law consistency index (paper)
    double mu_min_pl  = 0.00345;
    double mu_max_pl  = 0.056;
    double mu_ref         = POWERLAW_ETA_C;
    bool   use_dimensionless_viscosity = true;

    // MHD extension slots
    bool   enable_mhd = false;
    double Ha         = 10.0;
    double Bx         = 0.0;
    double By         = 1.0;
    double Bz         = 0.0;

    DomainTopology3D    topology;
    MultiPhysicsSlots3D multiphysics_slots;
};
