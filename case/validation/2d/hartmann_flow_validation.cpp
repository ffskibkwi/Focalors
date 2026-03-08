#include "hartmann_flow_validation.h"

#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "io/csv_writer_2d.h"
#include "ns/ns_solver2d.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
    constexpr double kDiffusionDtSafety = 0.20;
    constexpr double kMagneticDtSafety  = 0.50;
    constexpr double kSmall             = 1.0e-12;

    struct TimeStepSelection
    {
        double convective_dt      = 0.0;
        double diffusion_dt_limit = std::numeric_limits<double>::infinity();
        double magnetic_dt_limit  = std::numeric_limits<double>::infinity();
        double accuracy_dt_limit  = std::numeric_limits<double>::infinity();
        double selected_dt        = 0.0;
        double viscosity_upper    = 0.0;
        double magnetic_factor_sq = 0.0;
    };

    struct ErrorNorms
    {
        double l1_abs   = std::numeric_limits<double>::quiet_NaN();
        double l1_rel   = std::numeric_limits<double>::quiet_NaN();
        double l2_abs   = std::numeric_limits<double>::quiet_NaN();
        double l2_rel   = std::numeric_limits<double>::quiet_NaN();
        double linf_abs = std::numeric_limits<double>::quiet_NaN();
        double linf_rel = std::numeric_limits<double>::quiet_NaN();
    };

    struct WeightedErrorAccumulator
    {
        double l1_error = 0.0;
        double l1_ref   = 0.0;
        double l2_error = 0.0;
        double l2_ref   = 0.0;
        double linf_err = 0.0;
        double linf_ref = 0.0;

        void add(double numerical, double analytical, double weight)
        {
            const double error = numerical - analytical;
            l1_error += std::abs(error) * weight;
            l1_ref += std::abs(analytical) * weight;
            l2_error += error * error * weight;
            l2_ref += analytical * analytical * weight;
            linf_err = std::max(linf_err, std::abs(error));
            linf_ref = std::max(linf_ref, std::abs(analytical));
        }

        ErrorNorms finalize() const
        {
            ErrorNorms norms;
            norms.l1_abs   = l1_error;
            norms.l1_rel   = l1_error / std::max(l1_ref, kSmall);
            norms.l2_abs   = std::sqrt(l2_error);
            norms.l2_rel   = std::sqrt(l2_error / std::max(l2_ref, kSmall));
            norms.linf_abs = linf_err;
            norms.linf_rel = linf_err / std::max(linf_ref, kSmall);
            return norms;
        }
    };

    struct ProfileComparison
    {
        std::vector<double> y;
        std::vector<double> y_over_h;
        std::vector<double> numerical;
        std::vector<double> analytical;
        std::vector<double> abs_error;
        std::vector<double> rel_error_pct;
        ErrorNorms          norms;
        double              u_center_numerical  = std::numeric_limits<double>::quiet_NaN();
        double              u_center_analytical = std::numeric_limits<double>::quiet_NaN();
        double              u_bulk_numerical    = std::numeric_limits<double>::quiet_NaN();
        double              u_bulk_analytical   = std::numeric_limits<double>::quiet_NaN();
    };

    struct ProbeFace
    {
        Domain2DUniform* domain = nullptr;
        int              i      = 0;
        double           x      = 0.0;
    };

    struct ProbeSelection
    {
        double                 x_probe = 0.0;
        double                 x_min   = 0.0;
        double                 x_max   = 0.0;
        std::vector<ProbeFace> faces;
    };

    struct DomainBundle
    {
        Geometry2D                                    geometry;
        std::vector<std::unique_ptr<Domain2DUniform>> holders;
        std::vector<Domain2DUniform*>                 domains;
        Domain2DUniform*                              left_domain  = nullptr;
        Domain2DUniform*                              right_domain = nullptr;
        int                                           nx_left      = 0;
        int                                           nx_right     = 0;
        double                                        hx           = 0.0;
        double                                        hy           = 0.0;
    };

    struct SolverState
    {
        Variable2D u_var;
        Variable2D v_var;
        Variable2D p_var;

        std::unique_ptr<Variable2D> phi_var;

        std::unique_ptr<Variable2D> mu_var;
        std::unique_ptr<Variable2D> tau_xx_var;
        std::unique_ptr<Variable2D> tau_yy_var;
        std::unique_ptr<Variable2D> tau_xy_var;

        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> u_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> v_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> p_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> phi_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> mu_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> tau_xx_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> tau_yy_fields;
        std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>> tau_xy_fields;

        explicit SolverState(Geometry2D& geometry)
            : u_var("u")
            , v_var("v")
            , p_var("p")
        {
            u_var.set_geometry(geometry);
            v_var.set_geometry(geometry);
            p_var.set_geometry(geometry);
        }
    };

    struct RunSummary
    {
        int        final_step              = 0;
        double     final_residual          = std::numeric_limits<double>::infinity();
        double     steady_u_step           = std::numeric_limits<double>::infinity();
        double     steady_v_step           = std::numeric_limits<double>::infinity();
        double     steady_bulk             = std::numeric_limits<double>::quiet_NaN();
        double     steady_bulk_change_rel  = std::numeric_limits<double>::infinity();
        double     steady_u_max_abs        = std::numeric_limits<double>::quiet_NaN();
        double     steady_u_max_change_rel = std::numeric_limits<double>::infinity();
        double     steady_v_max_abs        = std::numeric_limits<double>::quiet_NaN();
        double     steady_v_max_change_rel = std::numeric_limits<double>::infinity();
        double     steady_v_to_u_ratio     = std::numeric_limits<double>::infinity();
        double     steady_indicator        = std::numeric_limits<double>::infinity();
        bool       steady_reached          = false;
        ErrorNorms profile_norms;
        ErrorNorms field_norms;
    };

    struct VelocityAmplitudeStats
    {
        double u_bulk    = 0.0;
        double u_max_abs = 0.0;
        double v_max_abs = 0.0;
    };

    struct SteadyMetricHistory
    {
        double prev_u_bulk    = 0.0;
        double prev_u_max_abs = 0.0;
        double prev_v_max_abs = 0.0;
    };

    struct SteadyMetrics
    {
        double u_step_rel        = std::numeric_limits<double>::infinity();
        double v_step_rel        = std::numeric_limits<double>::infinity();
        double u_bulk            = std::numeric_limits<double>::quiet_NaN();
        double u_bulk_change_rel = std::numeric_limits<double>::infinity();
        double u_max_abs         = std::numeric_limits<double>::quiet_NaN();
        double u_max_change_rel  = std::numeric_limits<double>::infinity();
        double v_max_abs         = std::numeric_limits<double>::quiet_NaN();
        double v_max_change_rel  = std::numeric_limits<double>::infinity();
        double v_to_u_ratio      = std::numeric_limits<double>::infinity();
        double indicator         = std::numeric_limits<double>::infinity();
    };

    double classical_hartmann_analytical(const HartmannFlowValidation2DCase& case_param, double y_centered)
    {
        const double half_height = case_param.half_height;
        if (std::abs(y_centered) > half_height + 1.0e-12)
            return 0.0;

        const double forcing = -case_param.Re * case_param.dp_dx;
        const double ha_eff  = case_param.getEffectiveHartmann();

        if (ha_eff <= 1.0e-10)
        {
            return 0.5 * forcing * (half_height * half_height - y_centered * y_centered);
        }

        const double coeff = forcing / (ha_eff * ha_eff);
        return coeff * (1.0 - std::cosh(ha_eff * y_centered) / std::max(std::cosh(ha_eff * half_height), 1.0 + kSmall));
    }

    double analytical_velocity(const HartmannFlowValidation2DCase& case_param, double y_centered)
    {
        if (!case_param.hasHartmannAnalyticalReference())
            return std::numeric_limits<double>::quiet_NaN();

        return classical_hartmann_analytical(case_param, y_centered);
    }

    double initial_corner_viscosity(const HartmannFlowValidation2DCase& case_param)
    {
        if (case_param.model_type == 1)
            return case_param.k_pl;
        if (case_param.model_type == 2)
            return case_param.mu_0;
        if (case_param.model_type == 3)
            return case_param.casson_mu;
        return 1.0 / std::max(case_param.Re, kSmall);
    }

    TimeStepSelection select_time_step(const HartmannFlowValidation2DCase& case_param, const PhysicsConfig& physics_cfg)
    {
        TimeStepSelection selection;
        const double      h_min =
            std::min(case_param.getLx() / std::max(case_param.nx, 1), case_param.getLy() / std::max(case_param.ny, 1));

        selection.convective_dt   = case_param.dt_factor * h_min;
        selection.viscosity_upper = (case_param.model_type == 0) ?
                                        physics_cfg.nu :
                                        std::max(physics_cfg.mu_max, initial_corner_viscosity(case_param));

        if (selection.viscosity_upper > 0.0)
        {
            selection.diffusion_dt_limit =
                kDiffusionDtSafety * h_min * h_min / std::max(selection.viscosity_upper, kSmall);
        }

        selection.magnetic_factor_sq =
            case_param.Bx * case_param.Bx + case_param.By * case_param.By + case_param.Bz * case_param.Bz;
        if (std::abs(case_param.Ha) > 0.0 && selection.magnetic_factor_sq > 0.0)
        {
            selection.magnetic_dt_limit =
                kMagneticDtSafety * case_param.Re / (case_param.Ha * case_param.Ha * selection.magnetic_factor_sq);
        }

        if (case_param.use_accuracy_dt_limit && case_param.dt_accuracy_factor > 0.0)
            selection.accuracy_dt_limit = case_param.dt_accuracy_factor * h_min * h_min;

        selection.selected_dt = std::min(
            selection.convective_dt,
            std::min(selection.diffusion_dt_limit, std::min(selection.magnetic_dt_limit, selection.accuracy_dt_limit)));

        if (!std::isfinite(selection.selected_dt) || selection.selected_dt <= 0.0)
            selection.selected_dt = selection.convective_dt;

        return selection;
    }

    DomainBundle build_domains(const HartmannFlowValidation2DCase& case_param)
    {
        DomainBundle bundle;

        const double lx = case_param.getLx();
        const double ly = case_param.getLy();

        if (case_param.nx < 4 || case_param.ny < 4)
            throw std::runtime_error("Invalid grid: nx and ny must both be >= 4.");

        if (!case_param.split_domain)
        {
            bundle.holders.emplace_back(new Domain2DUniform(case_param.nx, case_param.ny, lx, ly, "D0"));
            bundle.domains.push_back(bundle.holders.back().get());
            bundle.left_domain  = bundle.domains.front();
            bundle.right_domain = bundle.domains.front();
            bundle.nx_left      = case_param.nx;
            bundle.nx_right     = case_param.nx;
            bundle.geometry.add_domain(bundle.domains.front());
        }
        else
        {
            bundle.nx_left  = case_param.nx / 2;
            bundle.nx_right = case_param.nx - bundle.nx_left;

            if (bundle.nx_left < 2 || bundle.nx_right < 2)
                throw std::runtime_error("Split-domain mode requires at least 2 cells in each sub-domain.");

            const double hx       = lx / static_cast<double>(case_param.nx);
            const double lx_left  = hx * static_cast<double>(bundle.nx_left);
            const double lx_right = hx * static_cast<double>(bundle.nx_right);

            bundle.holders.emplace_back(new Domain2DUniform(bundle.nx_left, case_param.ny, lx_left, ly, "D0"));
            bundle.holders.emplace_back(new Domain2DUniform(bundle.nx_right, case_param.ny, lx_right, ly, "D1"));

            bundle.domains.push_back(bundle.holders[0].get());
            bundle.domains.push_back(bundle.holders[1].get());
            bundle.left_domain  = bundle.domains.front();
            bundle.right_domain = bundle.domains.back();

            bundle.geometry.connect(bundle.left_domain, LocationType::XPositive, bundle.right_domain);
        }

        bundle.geometry.axis(bundle.left_domain, LocationType::XNegative);
        bundle.geometry.axis(bundle.left_domain, LocationType::YNegative);
        bundle.geometry.check();
        bundle.geometry.solve_prepare();

        bundle.hx = bundle.left_domain->get_hx();
        bundle.hy = bundle.left_domain->get_hy();
        return bundle;
    }

    void add_field_for_all_domains(Variable2D&                                                    var,
                                   const std::vector<Domain2DUniform*>&                           domains,
                                   std::unordered_map<Domain2DUniform*, std::unique_ptr<field2>>& storages,
                                   VariablePositionType                                           pos)
    {
        for (auto* domain : domains)
        {
            storages[domain] = std::unique_ptr<field2>(new field2(var.name + "_" + domain->name));

            if (pos == VariablePositionType::XFace)
                var.set_x_edge_field(domain, *storages[domain]);
            else if (pos == VariablePositionType::YFace)
                var.set_y_edge_field(domain, *storages[domain]);
            else if (pos == VariablePositionType::Center)
                var.set_center_field(domain, *storages[domain]);
            else if (pos == VariablePositionType::Corner)
                var.set_corner_field(domain, *storages[domain]);
            else
                throw std::runtime_error("Unsupported field position.");
        }
    }

    bool needs_external_phi(const HartmannFlowValidation2DCase& case_param)
    {
        return std::abs(case_param.Ha) > 0.0 && std::abs(case_param.Bz) > 0.0;
    }

    bool is_nonnewtonian(const HartmannFlowValidation2DCase& case_param) { return case_param.model_type != 0; }

    void build_state(const HartmannFlowValidation2DCase& case_param, DomainBundle& bundle, SolverState& state)
    {
        add_field_for_all_domains(state.u_var, bundle.domains, state.u_fields, VariablePositionType::XFace);
        add_field_for_all_domains(state.v_var, bundle.domains, state.v_fields, VariablePositionType::YFace);
        add_field_for_all_domains(state.p_var, bundle.domains, state.p_fields, VariablePositionType::Center);

        if (needs_external_phi(case_param))
        {
            state.phi_var = std::make_unique<Variable2D>("phi");
            state.phi_var->set_geometry(bundle.geometry);
            add_field_for_all_domains(*state.phi_var, bundle.domains, state.phi_fields, VariablePositionType::Center);
            state.phi_var->set_boundary_type(PDEBoundaryType::Neumann);
        }

        if (is_nonnewtonian(case_param))
        {
            state.mu_var     = std::make_unique<Variable2D>("mu");
            state.tau_xx_var = std::make_unique<Variable2D>("tau_xx");
            state.tau_yy_var = std::make_unique<Variable2D>("tau_yy");
            state.tau_xy_var = std::make_unique<Variable2D>("tau_xy");

            state.mu_var->set_geometry(bundle.geometry);
            state.tau_xx_var->set_geometry(bundle.geometry);
            state.tau_yy_var->set_geometry(bundle.geometry);
            state.tau_xy_var->set_geometry(bundle.geometry);

            add_field_for_all_domains(*state.mu_var, bundle.domains, state.mu_fields, VariablePositionType::Corner);
            add_field_for_all_domains(
                *state.tau_xx_var, bundle.domains, state.tau_xx_fields, VariablePositionType::Center);
            add_field_for_all_domains(
                *state.tau_yy_var, bundle.domains, state.tau_yy_fields, VariablePositionType::Center);
            add_field_for_all_domains(
                *state.tau_xy_var, bundle.domains, state.tau_xy_fields, VariablePositionType::Corner);

            state.mu_var->set_boundary_type(PDEBoundaryType::Neumann);
            state.tau_xy_var->set_boundary_type(PDEBoundaryType::Neumann);

            const double mu_init = initial_corner_viscosity(case_param);
            for (auto* domain : bundle.domains)
                state.mu_var->field_map[domain]->clear(mu_init);
        }
    }

    void set_dirichlet(Variable2D& var, Domain2DUniform* domain, LocationType loc, double value)
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(domain, loc, value);
    }

    void set_neumann(Variable2D& var, Domain2DUniform* domain, LocationType loc, double value = 0.0)
    {
        var.set_boundary_type(domain, loc, PDEBoundaryType::Neumann);
        var.set_boundary_value(domain, loc, value);
    }

    void setup_boundary_conditions([[maybe_unused]] const HartmannFlowValidation2DCase& case_param,
                                   const DomainBundle&                                  bundle,
                                   Variable2D&                                          u,
                                   Variable2D&                                          v,
                                   Variable2D&                                          p,
                                   Variable2D*                                          phi)
    {
        for (auto* domain : bundle.domains)
        {
            set_dirichlet(u, domain, LocationType::YNegative, 0.0);
            set_dirichlet(u, domain, LocationType::YPositive, 0.0);
            set_dirichlet(v, domain, LocationType::YNegative, 0.0);
            set_dirichlet(v, domain, LocationType::YPositive, 0.0);
            set_neumann(p, domain, LocationType::YNegative, 0.0);
            set_neumann(p, domain, LocationType::YPositive, 0.0);

            if (phi != nullptr)
            {
                set_neumann(*phi, domain, LocationType::YNegative, 0.0);
                set_neumann(*phi, domain, LocationType::YPositive, 0.0);
            }
        }

        set_neumann(u, bundle.left_domain, LocationType::XNegative, 0.0);
        set_neumann(u, bundle.right_domain, LocationType::XPositive, 0.0);
        set_dirichlet(v, bundle.left_domain, LocationType::XNegative, 0.0);
        set_dirichlet(v, bundle.right_domain, LocationType::XPositive, 0.0);
        set_neumann(p, bundle.left_domain, LocationType::XNegative, 0.0);
        set_neumann(p, bundle.right_domain, LocationType::XPositive, 0.0);

        if (phi != nullptr)
        {
            set_neumann(*phi, bundle.left_domain, LocationType::XNegative, 0.0);
            set_neumann(*phi, bundle.right_domain, LocationType::XPositive, 0.0);
            phi->set_boundary_type(PDEBoundaryType::Neumann);
        }
    }

    void
    initialize_fields(const HartmannFlowValidation2DCase& case_param, const DomainBundle& bundle, SolverState& state)
    {
        const bool analytic_init =
            case_param.use_analytical_initialization && case_param.hasHartmannAnalyticalReference();

        for (auto* domain : bundle.domains)
        {
            field2& u_field = *state.u_var.field_map[domain];
            field2& v_field = *state.v_var.field_map[domain];
            field2& p_field = *state.p_var.field_map[domain];

            for (int i = 0; i < u_field.get_nx(); ++i)
            {
                for (int j = 0; j < u_field.get_ny(); ++j)
                {
                    const double y = domain->get_offset_y() + (j + 0.5) * domain->get_hy() - case_param.half_height;
                    u_field(i, j)  = analytic_init ? analytical_velocity(case_param, y) : 0.0;
                }
            }

            v_field.clear(0.0);
            p_field.clear(0.0);

            if (state.phi_var)
                state.phi_var->field_map[domain]->clear(0.0);

            if (state.tau_xx_var)
            {
                state.tau_xx_var->field_map[domain]->clear(0.0);
                state.tau_yy_var->field_map[domain]->clear(0.0);
                state.tau_xy_var->field_map[domain]->clear(0.0);
            }
        }
    }

    void apply_streamwise_pressure_gradient_force(const HartmannFlowValidation2DCase& case_param, SolverState& state)
    {
        const double delta_u = TimeAdvancingConfig::Get().dt * (-case_param.dp_dx);
        if (std::abs(delta_u) <= kSmall)
            return;

        for (auto* domain : state.u_var.geometry->domains)
        {
            field2& u_field = *state.u_var.field_map[domain];
            for (int i = 0; i < u_field.get_nx(); ++i)
            {
                for (int j = 0; j < u_field.get_ny(); ++j)
                    u_field(i, j) += delta_u;
            }
        }
    }

    double compute_velocity_residual(Variable2D& var, std::unordered_map<Domain2DUniform*, field2>& prev_map)
    {
        double total_diff_sq = 0.0;
        double total_norm_sq = 0.0;

        for (auto* domain : var.geometry->domains)
        {
            field2& curr = *var.field_map[domain];
            field2& prev = prev_map[domain];
            field2  diff = curr - prev;
            total_diff_sq += diff.squared_sum();
            total_norm_sq += curr.squared_sum();
        }

        if (total_norm_sq > 1.0e-14)
            return std::sqrt(total_diff_sq / total_norm_sq);

        return std::sqrt(total_diff_sq);
    }

    void update_prev_velocity(Variable2D& var, std::unordered_map<Domain2DUniform*, field2>& prev_map)
    {
        for (auto* domain : var.geometry->domains)
        {
            field2& curr = *var.field_map[domain];
            field2& prev = prev_map[domain];

            for (int i = 0; i < curr.get_nx(); ++i)
            {
                for (int j = 0; j < curr.get_ny(); ++j)
                    prev(i, j) = curr(i, j);
            }
        }
    }

    double relative_change(double current, double previous)
    {
        return std::abs(current - previous) / std::max(std::abs(current), kSmall);
    }

    VelocityAmplitudeStats measure_velocity_amplitudes(Variable2D& u_var, Variable2D& v_var)
    {
        VelocityAmplitudeStats stats;
        double                 weighted_u_sum = 0.0;
        double                 total_weight   = 0.0;

        for (auto* domain : u_var.geometry->domains)
        {
            field2&      u_field = *u_var.field_map[domain];
            field2&      v_field = *v_var.field_map[domain];
            const double hx      = domain->get_hx();
            const double hy      = domain->get_hy();
            const double weight  = hx * hy;

            for (int i = 0; i < u_field.get_nx(); ++i)
            {
                for (int j = 0; j < u_field.get_ny(); ++j)
                {
                    const double u_val = u_field(i, j);
                    const double v_val = v_field(i, j);
                    stats.u_max_abs    = std::max(stats.u_max_abs, std::abs(u_val));
                    stats.v_max_abs    = std::max(stats.v_max_abs, std::abs(v_val));
                    weighted_u_sum += u_val * weight;
                    total_weight += weight;
                }
            }
        }

        stats.u_bulk = weighted_u_sum / std::max(total_weight, kSmall);
        return stats;
    }

    SteadyMetrics evaluate_steady_metrics(const HartmannFlowValidation2DCase&           case_param,
                                          Variable2D&                                   u_var,
                                          Variable2D&                                   v_var,
                                          std::unordered_map<Domain2DUniform*, field2>& prev_u_map,
                                          std::unordered_map<Domain2DUniform*, field2>& prev_v_map,
                                          const SteadyMetricHistory&                    history)
    {
        SteadyMetrics                metrics;
        const VelocityAmplitudeStats amplitudes = measure_velocity_amplitudes(u_var, v_var);

        metrics.u_step_rel        = compute_velocity_residual(u_var, prev_u_map);
        metrics.v_step_rel        = compute_velocity_residual(v_var, prev_v_map);
        metrics.u_bulk            = amplitudes.u_bulk;
        metrics.u_bulk_change_rel = relative_change(amplitudes.u_bulk, history.prev_u_bulk);
        metrics.u_max_abs         = amplitudes.u_max_abs;
        metrics.u_max_change_rel  = relative_change(amplitudes.u_max_abs, history.prev_u_max_abs);
        metrics.v_max_abs         = amplitudes.v_max_abs;
        metrics.v_max_change_rel  = relative_change(amplitudes.v_max_abs, history.prev_v_max_abs);
        metrics.v_to_u_ratio      = amplitudes.v_max_abs / std::max(amplitudes.u_max_abs, kSmall);

        metrics.indicator = std::max({metrics.u_step_rel / std::max(case_param.steady_u_tol, kSmall),
                                      metrics.u_bulk_change_rel / std::max(case_param.steady_bulk_tol, kSmall),
                                      metrics.u_max_change_rel / std::max(case_param.steady_peak_tol, kSmall),
                                      metrics.v_max_change_rel / std::max(case_param.steady_v_peak_tol, kSmall)});
        return metrics;
    }

    void update_steady_history(const SteadyMetrics& metrics, SteadyMetricHistory& history)
    {
        history.prev_u_bulk    = metrics.u_bulk;
        history.prev_u_max_abs = metrics.u_max_abs;
        history.prev_v_max_abs = metrics.v_max_abs;
    }

    bool is_steady_converged(const HartmannFlowValidation2DCase& case_param, const SteadyMetrics& metrics)
    {
        return metrics.u_step_rel < case_param.steady_u_tol && metrics.u_bulk_change_rel < case_param.steady_bulk_tol &&
               metrics.u_max_change_rel < case_param.steady_peak_tol &&
               metrics.v_max_change_rel < case_param.steady_v_peak_tol;
    }

    std::string build_steady_failure_message(const HartmannFlowValidation2DCase& case_param,
                                             int                                  final_step,
                                             const SteadyMetrics&                 metrics)
    {
        std::ostringstream oss;
        oss << "Hartmann validation did not reach steady state within the configured time window. final_step="
            << final_step << ", T_total=" << case_param.T_total << ", steady_indicator=" << metrics.indicator
            << ", u_step=" << metrics.u_step_rel << ", bulk_change=" << metrics.u_bulk_change_rel
            << ", u_peak_change=" << metrics.u_max_change_rel << ", v_peak_change=" << metrics.v_max_change_rel
            << ", v_to_u_ratio=" << metrics.v_to_u_ratio;
        return oss.str();
    }

    ProbeSelection select_probe_faces(const HartmannFlowValidation2DCase& case_param, const DomainBundle& bundle)
    {
        ProbeSelection selection;
        const double   lx         = case_param.getLx();
        const double   half_width = std::max(case_param.x_window_half_width_over_h, 0.0) * case_param.half_height;
        const double   x_probe    = (case_param.x_probe_over_h < 0.0) ?
                                        (0.5 * lx) :
                                        std::min(std::max(case_param.x_probe_over_h * case_param.half_height, 0.0), lx);

        selection.x_probe = x_probe;
        selection.x_min   = std::max(0.0, selection.x_probe - half_width);
        selection.x_max   = std::min(lx, selection.x_probe + half_width);

        ProbeFace nearest_face;
        double    nearest_distance = std::numeric_limits<double>::infinity();

        for (auto* domain : bundle.domains)
        {
            const double hx = domain->get_hx();
            for (int i = 0; i < domain->get_nx(); ++i)
            {
                const double x_face   = domain->get_offset_x() + i * hx;
                const double distance = std::abs(x_face - selection.x_probe);
                if (distance < nearest_distance)
                {
                    nearest_distance = distance;
                    nearest_face     = {domain, i, x_face};
                }

                if (x_face >= selection.x_min - 1.0e-12 && x_face <= selection.x_max + 1.0e-12)
                    selection.faces.push_back({domain, i, x_face});
            }
        }

        if (selection.faces.empty())
        {
            selection.faces.push_back(nearest_face);
            selection.x_min = nearest_face.x;
            selection.x_max = nearest_face.x;
        }

        return selection;
    }

    ProfileComparison build_probe_profile(const HartmannFlowValidation2DCase& case_param,
                                          const DomainBundle&                 bundle,
                                          const ProbeSelection&               selection,
                                          Variable2D&                         u_var)
    {
        ProfileComparison comparison;
        const int         ny             = bundle.left_domain->get_ny();
        const double      hy             = bundle.left_domain->get_hy();
        const bool        has_analytical = case_param.hasHartmannAnalyticalReference();

        comparison.y.resize(static_cast<std::size_t>(ny));
        comparison.y_over_h.resize(static_cast<std::size_t>(ny));
        comparison.numerical.resize(static_cast<std::size_t>(ny));
        comparison.analytical.resize(static_cast<std::size_t>(ny), std::numeric_limits<double>::quiet_NaN());
        comparison.abs_error.resize(static_cast<std::size_t>(ny), std::numeric_limits<double>::quiet_NaN());
        comparison.rel_error_pct.resize(static_cast<std::size_t>(ny), std::numeric_limits<double>::quiet_NaN());

        WeightedErrorAccumulator accumulator;
        double                   integral_num = 0.0;
        double                   integral_ana = 0.0;
        int                      center_j     = 0;
        double                   min_abs_y    = std::numeric_limits<double>::infinity();

        for (int j = 0; j < ny; ++j)
        {
            double sum_u = 0.0;
            int    count = 0;

            for (const ProbeFace& face : selection.faces)
            {
                field2& u_field = *u_var.field_map[face.domain];
                sum_u += u_field(face.i, j);
                ++count;
            }

            const double y_centered = bundle.left_domain->get_offset_y() + (j + 0.5) * hy - case_param.half_height;
            const double u_num      = (count > 0) ? (sum_u / static_cast<double>(count)) : 0.0;

            comparison.y[static_cast<std::size_t>(j)]         = y_centered;
            comparison.y_over_h[static_cast<std::size_t>(j)]  = y_centered / std::max(case_param.half_height, kSmall);
            comparison.numerical[static_cast<std::size_t>(j)] = u_num;

            if (std::abs(y_centered) < min_abs_y)
            {
                min_abs_y = std::abs(y_centered);
                center_j  = j;
            }

            integral_num += u_num * hy;

            if (has_analytical)
            {
                const double u_ana = analytical_velocity(case_param, y_centered);
                const double err   = u_num - u_ana;

                comparison.analytical[static_cast<std::size_t>(j)]    = u_ana;
                comparison.abs_error[static_cast<std::size_t>(j)]     = std::abs(err);
                comparison.rel_error_pct[static_cast<std::size_t>(j)] = (std::abs(u_ana) > 1.0e-10) ?
                                                                            (100.0 * std::abs(err) / std::abs(u_ana)) :
                                                                            std::numeric_limits<double>::quiet_NaN();

                accumulator.add(u_num, u_ana, hy);
                integral_ana += u_ana * hy;
            }
        }

        if (has_analytical)
        {
            comparison.norms               = accumulator.finalize();
            comparison.u_center_analytical = comparison.analytical[static_cast<std::size_t>(center_j)];
            comparison.u_bulk_analytical   = integral_ana / std::max(case_param.getLy(), kSmall);
        }

        comparison.u_center_numerical = comparison.numerical[static_cast<std::size_t>(center_j)];
        comparison.u_bulk_numerical   = integral_num / std::max(case_param.getLy(), kSmall);
        return comparison;
    }

    ErrorNorms compare_full_field(const HartmannFlowValidation2DCase& case_param, Variable2D& u_var)
    {
        if (!case_param.hasHartmannAnalyticalReference())
            return {};

        WeightedErrorAccumulator accumulator;

        for (auto* domain : u_var.geometry->domains)
        {
            field2&      u_field = *u_var.field_map[domain];
            const double hx      = domain->get_hx();
            const double hy      = domain->get_hy();
            const double weight  = hx * hy;

            for (int i = 0; i < u_field.get_nx(); ++i)
                for (int j = 0; j < u_field.get_ny(); ++j)
                {
                    const double y_centered = domain->get_offset_y() + (j + 0.5) * hy - case_param.half_height;
                    accumulator.add(u_field(i, j), analytical_velocity(case_param, y_centered), weight);
                }
        }

        return accumulator.finalize();
    }

    void write_profile_csv(const std::string& root_dir, const ProfileComparison& comparison)
    {
        std::ofstream out(root_dir + "/profile_x_probe.csv");
        out << std::setprecision(16);
        out << "y,y_over_h,u_numerical,u_analytical,abs_error,rel_error_pct\n";

        for (std::size_t idx = 0; idx < comparison.y.size(); ++idx)
        {
            out << comparison.y[idx] << "," << comparison.y_over_h[idx] << "," << comparison.numerical[idx] << ","
                << comparison.analytical[idx] << "," << comparison.abs_error[idx] << ","
                << comparison.rel_error_pct[idx] << "\n";
        }
    }

    void write_summary_csv(const HartmannFlowValidation2DCase& case_param,
                           const DomainBundle&                 bundle,
                           const ProbeSelection&               selection,
                           const TimeStepSelection&            dt_selection,
                           const RunSummary&                   summary,
                           const ProfileComparison&            profile)
    {
        std::ofstream out(case_param.root_dir + "/verification_summary.csv");
        out << std::setprecision(16);
        out << "topology,split_domain,num_domains,nx_total,ny,nx_left,nx_right,hx,hy,Re,Ha,Ha_effective,dp_dx,"
               "Bx,By,Bz,model_type,n_index,has_analytical_reference,x_probe,x_window_min,x_window_max,"
               "probe_sample_count,final_step,final_residual,steady_u_step,steady_v_step,steady_bulk,"
               "steady_bulk_change_rel,steady_u_max_abs,steady_u_max_change_rel,steady_v_max_abs,"
               "steady_v_max_change_rel,steady_v_to_u_ratio,steady_indicator,steady_reached,dt,dt_convective,dt_diffusion_limit,dt_magnetic_limit,"
               "dt_accuracy_limit,u_center_numerical,u_center_analytical,u_bulk_numerical,u_bulk_analytical,"
               "profile_l1_abs,profile_l1_rel,profile_l2_abs,profile_l2_rel,profile_linf_abs,profile_linf_rel,"
               "field_l1_abs,field_l1_rel,field_l2_abs,field_l2_rel,field_linf_abs,field_linf_rel\n";

        out << (case_param.split_domain ? "split" : "single") << "," << (case_param.split_domain ? 1 : 0) << ","
            << bundle.domains.size() << "," << case_param.nx << "," << case_param.ny << "," << bundle.nx_left << ","
            << bundle.nx_right << "," << bundle.hx << "," << bundle.hy << "," << case_param.Re << "," << case_param.Ha
            << "," << case_param.getEffectiveHartmann() << "," << case_param.dp_dx << "," << case_param.Bx << ","
            << case_param.By << "," << case_param.Bz << "," << case_param.model_type << "," << case_param.n_index << ","
            << (case_param.hasHartmannAnalyticalReference() ? 1 : 0) << "," << selection.x_probe << ","
            << selection.x_min << "," << selection.x_max << "," << selection.faces.size() << "," << summary.final_step
            << "," << summary.final_residual << "," << summary.steady_u_step << "," << summary.steady_v_step << ","
            << summary.steady_bulk << "," << summary.steady_bulk_change_rel << "," << summary.steady_u_max_abs << ","
            << summary.steady_u_max_change_rel << "," << summary.steady_v_max_abs << ","
            << summary.steady_v_max_change_rel << "," << summary.steady_v_to_u_ratio << ","
            << summary.steady_indicator << "," << (summary.steady_reached ? 1 : 0) << ","
            << TimeAdvancingConfig::Get().dt << ","
            << dt_selection.convective_dt << "," << dt_selection.diffusion_dt_limit << ","
            << dt_selection.magnetic_dt_limit << "," << dt_selection.accuracy_dt_limit << ","
            << profile.u_center_numerical << "," << profile.u_center_analytical << "," << profile.u_bulk_numerical
            << "," << profile.u_bulk_analytical << "," << summary.profile_norms.l1_abs << ","
            << summary.profile_norms.l1_rel << "," << summary.profile_norms.l2_abs << ","
            << summary.profile_norms.l2_rel << "," << summary.profile_norms.linf_abs << ","
            << summary.profile_norms.linf_rel << "," << summary.field_norms.l1_abs << "," << summary.field_norms.l1_rel
            << "," << summary.field_norms.l2_abs << "," << summary.field_norms.l2_rel << ","
            << summary.field_norms.linf_abs << "," << summary.field_norms.linf_rel << "\n";
    }

    void write_final_fields(const HartmannFlowValidation2DCase& case_param, int final_step, SolverState& state)
    {
        IO::write_csv(state.u_var, case_param.root_dir + "/final/u_" + std::to_string(final_step));
        IO::write_csv(state.v_var, case_param.root_dir + "/final/v_" + std::to_string(final_step));
        IO::write_csv(state.p_var, case_param.root_dir + "/final/p_" + std::to_string(final_step));

        if (state.mu_var)
            IO::write_csv(*state.mu_var, case_param.root_dir + "/final/mu_" + std::to_string(final_step));

        if (state.phi_var)
            IO::write_csv(*state.phi_var, case_param.root_dir + "/final/phi_" + std::to_string(final_step));
    }
} // namespace

int main(int argc, char* argv[])
{
    try
    {
        HartmannFlowValidation2DCase case_param(argc, argv);
        case_param.read_paras();
        case_param.record_paras();

        EnvironmentConfig& env_cfg = EnvironmentConfig::Get();
        env_cfg.showCurrentStep    = false;
        env_cfg.showGmresRes       = false;

        PhysicsConfig& physics_cfg = PhysicsConfig::Get();
        physics_cfg.set_Re(case_param.Re);
        physics_cfg.set_model_type(case_param.model_type);

        if (case_param.model_type == 1)
        {
            physics_cfg.set_power_law_dimensionless(case_param.k_pl,
                                                    case_param.n_index,
                                                    case_param.Re,
                                                    case_param.mu_ref,
                                                    case_param.use_dimensionless_viscosity,
                                                    case_param.mu_min_pl,
                                                    case_param.mu_max_pl);
        }
        else if (case_param.model_type == 2)
        {
            physics_cfg.set_carreau_dimensionless(case_param.mu_0,
                                                  case_param.mu_inf,
                                                  case_param.a,
                                                  case_param.lambda,
                                                  case_param.n_index,
                                                  case_param.Re,
                                                  case_param.mu_ref,
                                                  case_param.use_dimensionless_viscosity,
                                                  case_param.mu_min_pl,
                                                  case_param.mu_max_pl);
        }
        else if (case_param.model_type == 3)
        {
            physics_cfg.set_casson_dimensionless(case_param.casson_mu,
                                                 case_param.casson_tau0,
                                                 case_param.Re,
                                                 case_param.mu_ref,
                                                 case_param.use_dimensionless_viscosity,
                                                 case_param.mu_min_pl,
                                                 case_param.mu_max_pl);
        }

        const bool enable_mhd =
            std::abs(case_param.Ha) > 0.0 &&
            (case_param.Bx * case_param.Bx + case_param.By * case_param.By + case_param.Bz * case_param.Bz > 0.0);
        physics_cfg.set_enable_mhd(enable_mhd);
        physics_cfg.set_Ha(case_param.Ha);
        physics_cfg.set_magnetic_field(case_param.Bx, case_param.By, case_param.Bz);

        const TimeStepSelection dt_selection = select_time_step(case_param, physics_cfg);

        TimeAdvancingConfig& time_cfg = TimeAdvancingConfig::Get();
        time_cfg.set_dt(dt_selection.selected_dt);
        time_cfg.set_t_max(case_param.T_total);
        time_cfg.set_corr_iter(case_param.corr_iter);
        if (time_cfg.num_iterations < 1)
            time_cfg.set_num_iterations(1);

        case_param.paras_record.record("dt", time_cfg.dt)
            .record("dt_convective", dt_selection.convective_dt)
            .record("dt_diffusion_limit", dt_selection.diffusion_dt_limit)
            .record("dt_magnetic_limit", dt_selection.magnetic_dt_limit)
            .record("dt_accuracy_limit", dt_selection.accuracy_dt_limit)
            .record("viscosity_upper_bound", dt_selection.viscosity_upper)
            .record("magnetic_factor_sq", dt_selection.magnetic_factor_sq)
            .record("pressure_drive_realization", std::string("equivalent_uniform_source_from_dpdx"))
            .record("has_analytical_reference", case_param.hasHartmannAnalyticalReference() ? 1 : 0)
            .record("Ha_effective", case_param.getEffectiveHartmann());

        DomainBundle bundle = build_domains(case_param);
        SolverState  state(bundle.geometry);
        build_state(case_param, bundle, state);
        setup_boundary_conditions(case_param, bundle, state.u_var, state.v_var, state.p_var, state.phi_var.get());
        initialize_fields(case_param, bundle, state);

        ConcatPoissonSolver2D p_solver(&state.p_var);
        ConcatNSSolver2D      ns_solver(&state.u_var, &state.v_var, &state.p_var, &p_solver);
        ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

        if (is_nonnewtonian(case_param))
        {
            ns_solver.init_nonnewton(state.mu_var.get(),
                                     state.tau_xx_var.get(),
                                     state.tau_yy_var.get(),
                                     state.tau_xy_var.get(),
                                     state.phi_var.get());
        }
        else if (enable_mhd)
        {
            ns_solver.init_mhd(state.phi_var.get());
        }

        std::unordered_map<Domain2DUniform*, field2> prev_u_map;
        std::unordered_map<Domain2DUniform*, field2> prev_v_map;
        for (auto* domain : bundle.domains)
        {
            prev_u_map[domain].init(state.u_var.field_map[domain]->get_nx(),
                                    state.u_var.field_map[domain]->get_ny(),
                                    "prev_u_" + domain->name);
            prev_v_map[domain].init(state.v_var.field_map[domain]->get_nx(),
                                    state.v_var.field_map[domain]->get_ny(),
                                    "prev_v_" + domain->name);
        }
        update_prev_velocity(state.u_var, prev_u_map);
        update_prev_velocity(state.v_var, prev_v_map);

        SteadyMetricHistory steady_history;
        {
            const VelocityAmplitudeStats initial_amplitudes = measure_velocity_amplitudes(state.u_var, state.v_var);
            steady_history.prev_u_bulk                      = initial_amplitudes.u_bulk;
            steady_history.prev_u_max_abs                   = initial_amplitudes.u_max_abs;
            steady_history.prev_v_max_abs                   = initial_amplitudes.v_max_abs;
        }

        std::cout << "Hartmann flow validation case" << std::endl;
        std::cout << "  Topology: " << (case_param.split_domain ? "split" : "single") << std::endl;
        std::cout << "  Grid: nx=" << case_param.nx << ", ny=" << case_param.ny << std::endl;
        std::cout << "  Geometry: Lx=" << case_param.getLx() << ", Ly=" << case_param.getLy()
                  << ", half_height=" << case_param.half_height << std::endl;
        std::cout << "  Physics: Re=" << case_param.Re << ", Ha=" << case_param.Ha
                  << ", Ha_effective=" << case_param.getEffectiveHartmann() << ", dp_dx=" << case_param.dp_dx << ", B=("
                  << case_param.Bx << "," << case_param.By << "," << case_param.Bz << ")" << std::endl;
        std::cout << "  Driving: equivalent uniform source reconstructed from physical dp/dx" << std::endl;
        std::cout << "  Model type: " << case_param.model_type << std::endl;
        std::cout << "  Time step: dt=" << time_cfg.dt << ", iterations=" << time_cfg.num_iterations
                  << ", corr_iter=" << time_cfg.corr_iter << std::endl;
        std::cout << "  Analytical reference: " << (case_param.hasHartmannAnalyticalReference() ? "yes" : "no")
                  << std::endl;

        int           final_step     = time_cfg.num_iterations;
        int           converged_hits = 0;
        double        final_residual = std::numeric_limits<double>::infinity();
        SteadyMetrics final_metrics;
        bool          steady_reached = false;
        const int     print_step = std::max(1, time_cfg.num_iterations / 10);

        for (int step = 1; step <= time_cfg.num_iterations; ++step)
        {
            if (is_nonnewtonian(case_param))
                ns_solver.solve_nonnewton();
            else
                ns_solver.solve();

            apply_streamwise_pressure_gradient_force(case_param, state);
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();

            const SteadyMetrics metrics =
                evaluate_steady_metrics(case_param, state.u_var, state.v_var, prev_u_map, prev_v_map, steady_history);
            final_metrics  = metrics;
            final_residual = metrics.indicator;

            if (is_steady_converged(case_param, metrics))
                ++converged_hits;
            else
                converged_hits = 0;

            if (step <= 10 || step % print_step == 0)
            {
                std::cout << "  step=" << step << " steady(u_step,v_step,bulk,u_peak,v_peak,v/u,indicator)=("
                          << metrics.u_step_rel << "," << metrics.v_step_rel << "," << metrics.u_bulk_change_rel << ","
                          << metrics.u_max_change_rel << "," << metrics.v_max_change_rel << "," << metrics.v_to_u_ratio
                          << "," << metrics.indicator << "), hits=" << converged_hits << "/" << case_param.converged_hits
                          << std::endl;
            }

            if (converged_hits >= case_param.converged_hits)
            {
                final_step = step;
                steady_reached = true;
                std::cout << "Converged at step=" << final_step << " with steady_indicator=" << final_residual
                          << std::endl;
                break;
            }

            update_prev_velocity(state.u_var, prev_u_map);
            update_prev_velocity(state.v_var, prev_v_map);
            update_steady_history(metrics, steady_history);

            if (case_param.pv_output_step > 0 && step % case_param.pv_output_step == 0)
            {
                IO::write_csv(state.u_var, case_param.root_dir + "/u/u_" + std::to_string(step));
                IO::write_csv(state.v_var, case_param.root_dir + "/v/v_" + std::to_string(step));
                IO::write_csv(state.p_var, case_param.root_dir + "/p/p_" + std::to_string(step));
                if (state.mu_var)
                    IO::write_csv(*state.mu_var, case_param.root_dir + "/mu/mu_" + std::to_string(step));
            }
        }

        ns_solver.phys_boundary_update();
        ns_solver.nondiag_shared_boundary_update();
        ns_solver.diag_shared_boundary_update();

        if (!steady_reached)
        {
            if (case_param.require_steady_exit)
                throw std::runtime_error(build_steady_failure_message(case_param, final_step, final_metrics));

            std::cerr << "[WARN] " << build_steady_failure_message(case_param, final_step, final_metrics) << std::endl;
        }

        if (final_metrics.v_to_u_ratio > case_param.steady_v_ratio_tol)
        {
            std::cout << "[WARN] Final v/u ratio exceeds diagnostic threshold: " << final_metrics.v_to_u_ratio
                      << " > " << case_param.steady_v_ratio_tol << std::endl;
        }

        write_final_fields(case_param, final_step, state);

        const ProbeSelection probe_selection = select_probe_faces(case_param, bundle);
        ProfileComparison    profile         = build_probe_profile(case_param, bundle, probe_selection, state.u_var);
        ErrorNorms           field_norm      = compare_full_field(case_param, state.u_var);

        RunSummary summary;
        summary.final_step              = final_step;
        summary.final_residual          = final_residual;
        summary.steady_u_step           = final_metrics.u_step_rel;
        summary.steady_v_step           = final_metrics.v_step_rel;
        summary.steady_bulk             = final_metrics.u_bulk;
        summary.steady_bulk_change_rel  = final_metrics.u_bulk_change_rel;
        summary.steady_u_max_abs        = final_metrics.u_max_abs;
        summary.steady_u_max_change_rel = final_metrics.u_max_change_rel;
        summary.steady_v_max_abs        = final_metrics.v_max_abs;
        summary.steady_v_max_change_rel = final_metrics.v_max_change_rel;
        summary.steady_v_to_u_ratio     = final_metrics.v_to_u_ratio;
        summary.steady_indicator        = final_metrics.indicator;
        summary.steady_reached          = steady_reached;
        summary.profile_norms           = profile.norms;
        summary.field_norms             = field_norm;

        write_profile_csv(case_param.root_dir, profile);
        write_summary_csv(case_param, bundle, probe_selection, dt_selection, summary, profile);

        std::cout << "Finished. profile saved to " << case_param.root_dir + "/profile_x_probe.csv" << std::endl;
        if (case_param.hasHartmannAnalyticalReference())
        {
            std::cout << "  Profile L2 relative error = " << summary.profile_norms.l2_rel << std::endl;
            std::cout << "  Full-field L2 relative error = " << summary.field_norms.l2_rel << std::endl;
        }
        else
        {
            std::cout << "  Analytical error norms skipped for current parameter combination." << std::endl;
        }

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return -1;
    }
}
