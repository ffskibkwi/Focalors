#include "base/config.h"
#include "base/domain/domain2d.h"
#include "base/domain/geometry2d.h"
#include "base/domain/variable2d.h"
#include "base/field/field2.h"
#include "base/location_boundary.h"
#include "cross_shaped_channel.h"
#include "io/common.h"
#include "io/csv_writer_2d.h"
#include "ns/mhd_module_2d_mac.h"
#include "ns/ns_solver2d.h"
#include "ns/scalar_solver2d.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr double EXPLICIT_DIFFUSION_DT_FACTOR = 0.20;
    constexpr double MAGNETIC_DT_FACTOR           = 0.50;
    constexpr double SMALL_NUMBER                 = 1.0e-12;

    double scale_viscosity_to_solver_units(double viscosity_value, const PhysicsConfig& physics_cfg)
    {
        if (physics_cfg.model_type == 0 || !physics_cfg.use_dimensionless_viscosity)
            return viscosity_value;

        const double solver_scale = physics_cfg.mu_ref * physics_cfg.Re;
        if (solver_scale <= 0.0)
            return viscosity_value;

        return viscosity_value / solver_scale;
    }

    struct TimeStepSelection
    {
        double convective_dt                   = 0.0;
        double diffusion_dt_limit              = std::numeric_limits<double>::infinity();
        double magnetic_dt_limit               = std::numeric_limits<double>::infinity();
        double selected_dt                     = 0.0;
        double viscosity_upper_bound_raw       = 0.0;
        double viscosity_upper_bound_effective = 0.0;
        double magnetic_factor_sq              = 0.0;
        bool   diffusion_limited               = false;
        bool   magnetic_limited                = false;
    };

    struct TimeStepSchedule
    {
        double base_dt        = 0.0;
        double startup_dt     = 0.0;
        double startup_t_end  = 0.0;
        bool   has_startup_dt = false;
        double initial_dt() const { return has_startup_dt ? startup_dt : base_dt; }
    };

    TimeStepSelection select_time_step(double h, double dt_factor, const PhysicsConfig& physics_cfg)
    {
        TimeStepSelection selection;

        selection.convective_dt             = dt_factor * h;
        selection.viscosity_upper_bound_raw = physics_cfg.model_type == 0 ? physics_cfg.nu : physics_cfg.mu_max;
        selection.viscosity_upper_bound_effective =
            scale_viscosity_to_solver_units(selection.viscosity_upper_bound_raw, physics_cfg);

        if (selection.viscosity_upper_bound_effective > 0.0)
        {
            selection.diffusion_dt_limit =
                EXPLICIT_DIFFUSION_DT_FACTOR * h * h / selection.viscosity_upper_bound_effective;
        }

        selection.magnetic_factor_sq = physics_cfg.Bx * physics_cfg.Bx + physics_cfg.By * physics_cfg.By +
                                       physics_cfg.Bz * physics_cfg.Bz;
        if (std::abs(physics_cfg.Ha) > 0.0 && selection.magnetic_factor_sq > 0.0 && physics_cfg.Re > 0.0)
        {
            selection.magnetic_dt_limit =
                MAGNETIC_DT_FACTOR * physics_cfg.Re / (physics_cfg.Ha * physics_cfg.Ha * selection.magnetic_factor_sq);
        }

        selection.selected_dt =
            std::min(selection.convective_dt, std::min(selection.diffusion_dt_limit, selection.magnetic_dt_limit));
        selection.diffusion_limited = selection.diffusion_dt_limit < selection.convective_dt &&
                                      selection.diffusion_dt_limit <= selection.magnetic_dt_limit;
        selection.magnetic_limited = selection.magnetic_dt_limit < selection.convective_dt &&
                                     selection.magnetic_dt_limit < selection.diffusion_dt_limit;

        return selection;
    }

    double compute_step_dt(double current_time, double total_time, const TimeStepSchedule& schedule)
    {
        const double eps = 128.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, total_time);
        if (current_time >= total_time - eps)
            return 0.0;

        const bool   in_startup_phase = schedule.has_startup_dt && current_time < schedule.startup_t_end - eps;
        const double phase_dt         = in_startup_phase ? schedule.startup_dt : schedule.base_dt;
        const double phase_end        = in_startup_phase ? schedule.startup_t_end : total_time;
        const double phase_remain     = std::max(0.0, phase_end - current_time);
        const double total_remain     = std::max(0.0, total_time - current_time);

        return std::min(phase_dt, std::min(phase_remain, total_remain));
    }

    int estimate_num_steps(double total_time, const TimeStepSchedule& schedule)
    {
        int    step         = 0;
        double current_time = 0.0;

        while (true)
        {
            const double dt_step = compute_step_dt(current_time, total_time, schedule);
            if (dt_step <= 0.0)
                break;

            current_time += dt_step;
            ++step;
        }

        return step;
    }

    std::string to_lower(std::string text)
    {
        std::transform(text.begin(),
                       text.end(),
                       text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    DifferenceSchemeType parse_scalar_scheme(const std::string& scheme_name)
    {
        const std::string key = to_lower(scheme_name);
        if (key == "cd2" || key == "center" || key == "central" || key == "conv_center2nd_diff_center2nd")
            return DifferenceSchemeType::Conv_Center2nd_Diff_Center2nd;
        if (key == "uw1" || key == "upwind" || key == "upwind1st" || key == "conv_upwind1st_diff_center2nd")
            return DifferenceSchemeType::Conv_Upwind1st_Diff_Center2nd;
        if (key == "quick" || key == "conv_quick_diff_center2nd")
            return DifferenceSchemeType::Conv_QUICK_Diff_Center2nd;
        if (key == "tvd" || key == "vanleer" || key == "tvd_vanleer" || key == "conv_tvd_vanleer_diff_center2nd")
            return DifferenceSchemeType::Conv_TVD_VanLeer_Diff_Center2nd;

        throw std::runtime_error("Unknown scalar scheme: " + scheme_name);
    }

    class CrossShapedChannel2DStabilityCase : public CrossShapedChannel2DCase
    {
    public:
        CrossShapedChannel2DStabilityCase(int argc, char* argv[])
            : CrossShapedChannel2DCase(argc, argv)
        {}

        void read_paras() override
        {
            CrossShapedChannel2DCase::read_paras();
            IO::read_number(para_map, "perturb_eps", perturb_eps);
            IO::read_number(para_map, "perturb_sigma_over_d", perturb_sigma_over_d);
            IO::read_number(para_map, "perturb_center_x_over_d", perturb_center_x_over_d);
            IO::read_number(para_map, "perturb_center_y_over_d", perturb_center_y_over_d);
            IO::read_number(para_map, "perturb_stream_sign", perturb_stream_sign);
            IO::read_number(para_map, "diagnostic_window_half_width_over_d", diagnostic_window_half_width_over_d);
            IO::read_number(para_map, "stagnation_window_half_width_over_d", stagnation_window_half_width_over_d);
            IO::read_number(para_map, "history_output_step", history_output_step);

            if (perturb_stream_sign == 0)
                perturb_stream_sign = 1;
            if (history_output_step <= 0)
                history_output_step = 1;
        }

        bool record_paras() override
        {
            if (!CrossShapedChannel2DCase::record_paras())
                return false;

            paras_record.record("perturb_eps", perturb_eps)
                .record("perturb_sigma_over_d", perturb_sigma_over_d)
                .record("perturb_center_x_over_d", perturb_center_x_over_d)
                .record("perturb_center_y_over_d", perturb_center_y_over_d)
                .record("perturb_stream_sign", perturb_stream_sign)
                .record("diagnostic_window_half_width_over_d", diagnostic_window_half_width_over_d)
                .record("stagnation_window_half_width_over_d", stagnation_window_half_width_over_d)
                .record("history_output_step", history_output_step);

            return true;
        }

        double perturb_eps                        = 1.0e-6;
        double perturb_sigma_over_d               = 0.25;
        double perturb_center_x_over_d            = 0.0;
        double perturb_center_y_over_d            = 0.1;
        int    perturb_stream_sign                = 1;
        double diagnostic_window_half_width_over_d = 1.0;
        double stagnation_window_half_width_over_d = 0.5;
        int    history_output_step                 = 50;
    };

    struct StabilityMetrics
    {
        double energy_upper       = 0.0;
        double energy_lower       = 0.0;
        double energy_left        = 0.0;
        double energy_right       = 0.0;
        double energy_total       = 0.0;
        double asymmetry_ud_energy = std::numeric_limits<double>::quiet_NaN();
        double asymmetry_lr_energy = std::numeric_limits<double>::quiet_NaN();
        double stagnation_x_over_d = std::numeric_limits<double>::quiet_NaN();
        double stagnation_y_over_d = std::numeric_limits<double>::quiet_NaN();
        double stagnation_r_over_d = std::numeric_limits<double>::quiet_NaN();
        double stagnation_speed    = std::numeric_limits<double>::quiet_NaN();
        double max_speed_window    = std::numeric_limits<double>::quiet_NaN();
        bool   has_window          = false;
        bool   has_stagnation      = false;
    };

    struct MaxAbsTracker
    {
        double abs_value = -1.0;
        int    step      = 0;
        double time      = 0.0;
        bool   valid     = false;

        void update(double value, int in_step, double in_time)
        {
            if (!std::isfinite(value))
                return;

            const double abs_value_candidate = std::abs(value);
            if (!valid || abs_value_candidate > abs_value)
            {
                abs_value = abs_value_candidate;
                step      = in_step;
                time      = in_time;
                valid     = true;
            }
        }
    };

    struct MaxTracker
    {
        double value = -1.0;
        int    step  = 0;
        double time  = 0.0;
        bool   valid = false;

        void update(double candidate, int in_step, double in_time)
        {
            if (!std::isfinite(candidate))
                return;

            if (!valid || candidate > value)
            {
                value = candidate;
                step  = in_step;
                time  = in_time;
                valid = true;
            }
        }
    };

    struct StabilitySummaryAccumulator
    {
        StabilityMetrics last_metrics;
        bool             has_last_metrics   = false;
        int              history_samples    = 0;
        int              last_recorded_step = -1;
        double           last_recorded_time = 0.0;
        MaxAbsTracker    asymmetry_ud_abs_max;
        MaxAbsTracker    asymmetry_lr_abs_max;
        MaxTracker       stagnation_r_max;

        void update(const StabilityMetrics& metrics, int step, double time)
        {
            last_metrics       = metrics;
            has_last_metrics   = true;
            last_recorded_step = step;
            last_recorded_time = time;
            ++history_samples;
            asymmetry_ud_abs_max.update(metrics.asymmetry_ud_energy, step, time);
            asymmetry_lr_abs_max.update(metrics.asymmetry_lr_energy, step, time);
            stagnation_r_max.update(metrics.stagnation_r_over_d, step, time);
        }
    };

    double field_value_clamped(const field2& field, int i, int j)
    {
        const int ci = std::max(0, std::min(i, field.get_nx() - 1));
        const int cj = std::max(0, std::min(j, field.get_ny() - 1));
        return field(ci, cj);
    }

    double sample_u_center(const field2& u_field, int nx, int ny, int i, int j)
    {
        const int cj = std::max(0, std::min(j, std::min(ny - 1, u_field.get_ny() - 1)));
        if (u_field.get_nx() >= nx + 1)
            return 0.5 * (field_value_clamped(u_field, i, cj) + field_value_clamped(u_field, i + 1, cj));

        return 0.5 * (field_value_clamped(u_field, i, cj) + field_value_clamped(u_field, i + 1, cj));
    }

    double sample_v_center(const field2& v_field, int nx, int ny, int i, int j)
    {
        const int ci = std::max(0, std::min(i, std::min(nx - 1, v_field.get_nx() - 1)));
        if (v_field.get_ny() >= ny + 1)
            return 0.5 * (field_value_clamped(v_field, ci, j) + field_value_clamped(v_field, ci, j + 1));

        return 0.5 * (field_value_clamped(v_field, ci, j) + field_value_clamped(v_field, ci, j + 1));
    }

    void apply_local_streamfunction_perturbation(const CrossShapedChannel2DStabilityCase& case_param,
                                                 Variable2D&                              u_var,
                                                 Variable2D&                              v_var,
                                                 double                                   domain_width,
                                                 double                                   center_x,
                                                 double                                   center_y)
    {
        if (std::abs(case_param.perturb_eps) <= 0.0)
            return;

        const double sigma = case_param.perturb_sigma_over_d * domain_width;
        if (sigma <= 0.0)
            throw std::runtime_error("perturb_sigma_over_d must be > 0");

        const double xc = case_param.perturb_center_x_over_d * domain_width;
        const double yc = case_param.perturb_center_y_over_d * domain_width;
        const double amplitude =
            static_cast<double>(case_param.perturb_stream_sign >= 0 ? 1 : -1) * case_param.perturb_eps * domain_width;
        const double sigma_sq = sigma * sigma;

        for (auto& pair : u_var.field_map)
        {
            Domain2DUniform* domain = pair.first;
            field2&          field  = *pair.second;
            const double     hx     = domain->get_hx();
            const double     hy     = domain->get_hy();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    const double x_rel = domain->get_offset_x() + static_cast<double>(i) * hx - center_x;
                    const double y_rel = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * hy - center_y;
                    const double dx    = x_rel - xc;
                    const double dy    = y_rel - yc;
                    const double r_sq  = dx * dx + dy * dy;
                    const double psi   = amplitude * std::exp(-0.5 * r_sq / sigma_sq);
                    field(i, j) += psi * (-dy / sigma_sq);
                }
            }
        }

        for (auto& pair : v_var.field_map)
        {
            Domain2DUniform* domain = pair.first;
            field2&          field  = *pair.second;
            const double     hx     = domain->get_hx();
            const double     hy     = domain->get_hy();

            for (int i = 0; i < field.get_nx(); ++i)
            {
                for (int j = 0; j < field.get_ny(); ++j)
                {
                    const double x_rel = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * hx - center_x;
                    const double y_rel = domain->get_offset_y() + static_cast<double>(j) * hy - center_y;
                    const double dx    = x_rel - xc;
                    const double dy    = y_rel - yc;
                    const double r_sq  = dx * dx + dy * dy;
                    const double psi   = amplitude * std::exp(-0.5 * r_sq / sigma_sq);
                    field(i, j) += psi * (dx / sigma_sq);
                }
            }
        }
    }

    StabilityMetrics compute_stability_metrics(const CrossShapedChannel2DStabilityCase& case_param,
                                               Variable2D&                              u_var,
                                               Variable2D&                              v_var,
                                               double                                   domain_width,
                                               double                                   center_x,
                                               double                                   center_y)
    {
        StabilityMetrics metrics;

        const double diag_half_width = case_param.diagnostic_window_half_width_over_d * domain_width;
        const double stag_half_width = case_param.stagnation_window_half_width_over_d * domain_width;
        double       best_stagnation_speed_sq = std::numeric_limits<double>::infinity();
        double       best_stagnation_radius_sq = std::numeric_limits<double>::infinity();
        double       max_speed_window         = 0.0;

        for (auto* domain : u_var.geometry->domains)
        {
            field2&      u_field = *u_var.field_map[domain];
            field2&      v_field = *v_var.field_map[domain];
            const int    nx      = domain->get_nx();
            const int    ny      = domain->get_ny();
            const double hx      = domain->get_hx();
            const double hy      = domain->get_hy();
            const double area    = hx * hy;

            for (int i = 0; i < nx; ++i)
            {
                const double x_center = domain->get_offset_x() + (static_cast<double>(i) + 0.5) * hx - center_x;

                for (int j = 0; j < ny; ++j)
                {
                    const double y_center = domain->get_offset_y() + (static_cast<double>(j) + 0.5) * hy - center_y;
                    const double u_center = sample_u_center(u_field, nx, ny, i, j);
                    const double v_center = sample_v_center(v_field, nx, ny, i, j);
                    const double speed_sq = u_center * u_center + v_center * v_center;
                    const double speed    = std::sqrt(speed_sq);

                    if (std::abs(x_center) <= diag_half_width && std::abs(y_center) <= diag_half_width)
                    {
                        metrics.has_window  = true;
                        metrics.energy_total += speed_sq * area;
                        if (y_center >= 0.0)
                            metrics.energy_upper += speed_sq * area;
                        else
                            metrics.energy_lower += speed_sq * area;

                        if (x_center >= 0.0)
                            metrics.energy_right += speed_sq * area;
                        else
                            metrics.energy_left += speed_sq * area;

                        max_speed_window = std::max(max_speed_window, speed);
                    }

                    if (std::abs(x_center) <= stag_half_width && std::abs(y_center) <= stag_half_width)
                    {
                        const double radius_sq = x_center * x_center + y_center * y_center;
                        const bool   better_speed =
                            speed_sq < best_stagnation_speed_sq - SMALL_NUMBER;
                        const bool speed_tied = std::abs(speed_sq - best_stagnation_speed_sq) <= SMALL_NUMBER;

                        if (!metrics.has_stagnation || better_speed ||
                            (speed_tied && radius_sq < best_stagnation_radius_sq))
                        {
                            best_stagnation_speed_sq = speed_sq;
                            best_stagnation_radius_sq = radius_sq;
                            metrics.stagnation_x_over_d = x_center / domain_width;
                            metrics.stagnation_y_over_d = y_center / domain_width;
                            metrics.stagnation_r_over_d = std::sqrt(radius_sq) / domain_width;
                            metrics.stagnation_speed = speed;
                            metrics.has_stagnation   = true;
                        }
                    }
                }
            }
        }

        if (metrics.has_window)
        {
            const double ud_denom = std::max(metrics.energy_upper + metrics.energy_lower, SMALL_NUMBER);
            const double lr_denom = std::max(metrics.energy_right + metrics.energy_left, SMALL_NUMBER);
            metrics.asymmetry_ud_energy = (metrics.energy_upper - metrics.energy_lower) / ud_denom;
            metrics.asymmetry_lr_energy = (metrics.energy_right - metrics.energy_left) / lr_denom;
            metrics.max_speed_window    = max_speed_window;
        }

        return metrics;
    }

    void write_history_header(std::ofstream& history_out)
    {
        history_out << "step,time,dt,energy_upper,energy_lower,energy_left,energy_right,energy_total,"
                       "asymmetry_ud_energy,asymmetry_lr_energy,stagnation_x_over_d,stagnation_y_over_d,"
                       "stagnation_r_over_d,stagnation_speed,max_speed_window\n";
    }

    void write_history_row(std::ofstream&             history_out,
                           int                        step,
                           double                     time,
                           double                     dt,
                           const StabilityMetrics& metrics)
    {
        history_out << step << "," << time << "," << dt << "," << metrics.energy_upper << "," << metrics.energy_lower
                    << "," << metrics.energy_left << "," << metrics.energy_right << "," << metrics.energy_total << ","
                    << metrics.asymmetry_ud_energy << "," << metrics.asymmetry_lr_energy << ","
                    << metrics.stagnation_x_over_d << "," << metrics.stagnation_y_over_d << ","
                    << metrics.stagnation_r_over_d << "," << metrics.stagnation_speed << ","
                    << metrics.max_speed_window << "\n";
    }

    double tracked_abs_value(const MaxAbsTracker& tracker)
    {
        return tracker.valid ? tracker.abs_value : std::numeric_limits<double>::quiet_NaN();
    }

    int tracked_step(const MaxAbsTracker& tracker) { return tracker.valid ? tracker.step : -1; }

    double tracked_time(const MaxAbsTracker& tracker)
    {
        return tracker.valid ? tracker.time : std::numeric_limits<double>::quiet_NaN();
    }

    double tracked_value(const MaxTracker& tracker)
    {
        return tracker.valid ? tracker.value : std::numeric_limits<double>::quiet_NaN();
    }

    int tracked_step(const MaxTracker& tracker) { return tracker.valid ? tracker.step : -1; }

    double tracked_time(const MaxTracker& tracker)
    {
        return tracker.valid ? tracker.time : std::numeric_limits<double>::quiet_NaN();
    }

    void write_summary_csv(const CrossShapedChannel2DStabilityCase& case_param,
                           const std::string&                       run_status,
                           bool                                     diverged,
                           int                                      final_step,
                           double                                   final_time,
                           double                                   final_dt,
                           int                                      estimated_total_steps,
                           const StabilitySummaryAccumulator&       summary_acc)
    {
        std::ofstream out(case_param.root_dir + "/stability_summary.csv");
        if (!out.is_open())
            throw std::runtime_error("Failed to open stability_summary.csv for writing.");

        out << std::setprecision(16);
        out << "run_status,diverged,final_step,final_time,final_dt,estimated_total_steps,history_samples,"
               "perturb_eps,perturb_sigma_over_d,perturb_center_x_over_d,perturb_center_y_over_d,"
               "diagnostic_window_half_width_over_d,stagnation_window_half_width_over_d,"
               "asymmetry_ud_energy_final,asymmetry_ud_energy_abs_max,asymmetry_ud_energy_abs_max_step,"
               "asymmetry_ud_energy_abs_max_time,asymmetry_lr_energy_final,asymmetry_lr_energy_abs_max,"
               "asymmetry_lr_energy_abs_max_step,asymmetry_lr_energy_abs_max_time,stagnation_x_over_d_final,"
               "stagnation_y_over_d_final,stagnation_r_over_d_final,stagnation_r_over_d_max,"
               "stagnation_r_over_d_max_step,stagnation_r_over_d_max_time,stagnation_speed_final\n";

        const StabilityMetrics final_metrics =
            summary_acc.has_last_metrics ? summary_acc.last_metrics : StabilityMetrics {};

        out << run_status << "," << (diverged ? 1 : 0) << "," << final_step << "," << final_time << "," << final_dt
            << "," << estimated_total_steps << "," << summary_acc.history_samples << "," << case_param.perturb_eps
            << "," << case_param.perturb_sigma_over_d << "," << case_param.perturb_center_x_over_d << ","
            << case_param.perturb_center_y_over_d << "," << case_param.diagnostic_window_half_width_over_d << ","
            << case_param.stagnation_window_half_width_over_d << "," << final_metrics.asymmetry_ud_energy << ","
            << tracked_abs_value(summary_acc.asymmetry_ud_abs_max) << ","
            << tracked_step(summary_acc.asymmetry_ud_abs_max) << ","
            << tracked_time(summary_acc.asymmetry_ud_abs_max) << "," << final_metrics.asymmetry_lr_energy << ","
            << tracked_abs_value(summary_acc.asymmetry_lr_abs_max) << ","
            << tracked_step(summary_acc.asymmetry_lr_abs_max) << ","
            << tracked_time(summary_acc.asymmetry_lr_abs_max) << "," << final_metrics.stagnation_x_over_d << ","
            << final_metrics.stagnation_y_over_d << "," << final_metrics.stagnation_r_over_d << ","
            << tracked_value(summary_acc.stagnation_r_max) << "," << tracked_step(summary_acc.stagnation_r_max) << ","
            << tracked_time(summary_acc.stagnation_r_max) << "," << final_metrics.stagnation_speed << "\n";
    }

    bool representative_value_is_finite(const field2& field)
    {
        if (field.get_nx() <= 0 || field.get_ny() <= 0)
            return false;

        const int i = std::min(1, field.get_nx() - 1);
        const int j = std::min(1, field.get_ny() - 1);
        return std::isfinite(field(i, j));
    }
} // namespace

int main(int argc, char* argv[])
{
    CrossShapedChannel2DStabilityCase case_param(argc, argv);
    case_param.read_paras();

    double      Sc                 = 0.0;
    std::string scalar_scheme_name = "quick";
    IO::read_number(case_param.para_map, "Sc", Sc);
    IO::read_string(case_param.para_map, "scalar_scheme", scalar_scheme_name);
    const bool                 enable_scalar_transport = Sc > 0.0;
    const double               Pe                      = enable_scalar_transport ? case_param.Re * Sc : 0.0;
    const double               nr                      = enable_scalar_transport ? 1.0 / Pe : 0.0;
    const DifferenceSchemeType scalar_scheme          = parse_scalar_scheme(scalar_scheme_name);
    if (enable_scalar_transport && Pe <= 0.0)
        throw std::runtime_error("Scalar transport requires Re * Sc > 0");

    Geometry2D geo;
    double     h = case_param.h;

    EnvironmentConfig& env_cfg       = EnvironmentConfig::Get();
    env_cfg.showGmresRes             = false;
    env_cfg.showCurrentStep          = false;
    TimeAdvancingConfig& time_cfg    = TimeAdvancingConfig::Get();
    PhysicsConfig&       physics_cfg = PhysicsConfig::Get();
    physics_cfg.set_Re(case_param.Re);

    const bool enable_mhd = (std::abs(case_param.Ha) > 0.0);
    physics_cfg.set_enable_mhd(enable_mhd);
    physics_cfg.Ha = case_param.Ha;
    physics_cfg.set_magnetic_field(case_param.Bx, case_param.By, case_param.Bz);

    std::cout << "MHD Parameters:" << std::endl;
    std::cout << "  enable_mhd: " << enable_mhd << std::endl;
    std::cout << "  Ha: " << case_param.Ha << std::endl;
    std::cout << "  Bx: " << case_param.Bx << std::endl;
    std::cout << "  By: " << case_param.By << std::endl;
    std::cout << "  Bz: " << case_param.Bz << std::endl;
    std::cout << "Scalar Transport Parameters:" << std::endl;
    std::cout << "  enable_scalar_transport: " << enable_scalar_transport << std::endl;
    std::cout << "  Sc: " << Sc << std::endl;
    if (enable_scalar_transport)
    {
        std::cout << "  Pe: " << Pe << std::endl;
        std::cout << "  nr: " << nr << std::endl;
    }
    std::cout << "  scalar_scheme: " << scalar_scheme << std::endl;

    physics_cfg.set_model_type(case_param.model_type);
    physics_cfg.set_gamma_ref(case_param.gamma_ref);
    if (case_param.model_type == 1)
    {
        physics_cfg.set_power_law_dimensionless(case_param.k_pl,
                                                case_param.n_index,
                                                case_param.Re,
                                                case_param.mu_ref,
                                                case_param.use_dimensionless_viscosity,
                                                case_param.mu_min_pl,
                                                case_param.mu_max_pl);

        std::cout << "Configuring Power Law Model (Dimensionless):" << std::endl;
        std::cout << "  k_pl: " << case_param.k_pl << std::endl;
        std::cout << "  Re: " << case_param.Re << std::endl;
        std::cout << "  mu_ref: " << case_param.mu_ref << std::endl;
        std::cout << "  gamma_ref: " << case_param.gamma_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
        std::cout << "  n: " << case_param.n_index << std::endl;
        std::cout << "  mu_min_pl: " << case_param.mu_min_pl << std::endl;
        std::cout << "  mu_max_pl: " << case_param.mu_max_pl << std::endl;
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
        std::cout << "Configuring Carreau Model (Dimensionless):" << std::endl;
        std::cout << "  mu_0: " << case_param.mu_0 << std::endl;
        std::cout << "  mu_inf: " << case_param.mu_inf << std::endl;
        std::cout << "  lambda: " << case_param.lambda << std::endl;
        std::cout << "  Re: " << case_param.Re << std::endl;
        std::cout << "  mu_ref: " << case_param.mu_ref << std::endl;
        std::cout << "  gamma_ref: " << case_param.gamma_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
        std::cout << "  a: " << case_param.a << std::endl;
        std::cout << "  n: " << case_param.n_index << std::endl;
        std::cout << "  mu_min_pl: " << case_param.mu_min_pl << std::endl;
        std::cout << "  mu_max_pl: " << case_param.mu_max_pl << std::endl;
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
        std::cout << "Configuring Casson Model (Dimensionless):" << std::endl;
        std::cout << "  casson_mu: " << case_param.casson_mu << std::endl;
        std::cout << "  casson_tau0: " << case_param.casson_tau0 << std::endl;
        std::cout << "  Re: " << case_param.Re << std::endl;
        std::cout << "  mu_ref: " << case_param.mu_ref << std::endl;
        std::cout << "  gamma_ref: " << case_param.gamma_ref << std::endl;
        std::cout << "  use_dimensionless_viscosity: " << case_param.use_dimensionless_viscosity << std::endl;
    }
    else
    {
        std::cout << "Configuring Newtonian Model." << std::endl;
    }

    const TimeStepSelection base_time_step_selection = select_time_step(h, case_param.dt_factor, physics_cfg);

    const bool        has_requested_startup_dt = case_param.startup_dt_factor > 0.0 && case_param.startup_t_end > 0.0;
    TimeStepSelection startup_time_step_selection;
    if (has_requested_startup_dt)
        startup_time_step_selection = select_time_step(h, case_param.startup_dt_factor, physics_cfg);

    TimeStepSchedule time_step_schedule;
    time_step_schedule.base_dt = base_time_step_selection.selected_dt;
    if (has_requested_startup_dt)
    {
        time_step_schedule.startup_dt    = startup_time_step_selection.selected_dt;
        time_step_schedule.startup_t_end = std::min(case_param.startup_t_end, case_param.T_total);
        time_step_schedule.has_startup_dt =
            time_step_schedule.startup_t_end > 0.0 && time_step_schedule.startup_dt < time_step_schedule.base_dt;
    }

    const int estimated_total_steps = estimate_num_steps(case_param.T_total, time_step_schedule);

    time_cfg.dt             = time_step_schedule.initial_dt();
    time_cfg.t_max          = case_param.T_total;
    time_cfg.num_iterations = estimated_total_steps;

    std::cout << "Time Step Selection:" << std::endl;
    std::cout << "  base_convective_dt: " << base_time_step_selection.convective_dt << std::endl;
    std::cout << "  base_diffusion_dt_limit: " << base_time_step_selection.diffusion_dt_limit << std::endl;
    std::cout << "  base_magnetic_dt_limit: " << base_time_step_selection.magnetic_dt_limit << std::endl;
    std::cout << "  viscosity_upper_bound_raw: " << base_time_step_selection.viscosity_upper_bound_raw << std::endl;
    std::cout << "  viscosity_upper_bound_effective: " << base_time_step_selection.viscosity_upper_bound_effective
              << std::endl;
    std::cout << "  magnetic_factor_sq: " << base_time_step_selection.magnetic_factor_sq << std::endl;
    std::cout << "  base_selected_dt: " << base_time_step_selection.selected_dt << std::endl;
    std::cout << "  initial_dt: " << time_cfg.dt << std::endl;
    std::cout << "  base_diffusion_limited: " << std::boolalpha << base_time_step_selection.diffusion_limited
              << std::noboolalpha << std::endl;
    std::cout << "  base_magnetic_limited: " << std::boolalpha << base_time_step_selection.magnetic_limited
              << std::noboolalpha << std::endl;
    if (has_requested_startup_dt)
    {
        std::cout << "  startup_convective_dt: " << startup_time_step_selection.convective_dt << std::endl;
        std::cout << "  startup_diffusion_dt_limit: " << startup_time_step_selection.diffusion_dt_limit << std::endl;
        std::cout << "  startup_magnetic_dt_limit: " << startup_time_step_selection.magnetic_dt_limit << std::endl;
        std::cout << "  startup_selected_dt: " << startup_time_step_selection.selected_dt << std::endl;
        std::cout << "  startup_t_end: " << time_step_schedule.startup_t_end << std::endl;
        std::cout << "  startup_active: " << std::boolalpha << time_step_schedule.has_startup_dt << std::noboolalpha
                  << std::endl;
        std::cout << "  startup_magnetic_limited: " << std::boolalpha << startup_time_step_selection.magnetic_limited
                  << std::noboolalpha << std::endl;
    }
    std::cout << "  estimated_total_steps: " << estimated_total_steps << std::endl;

    if (estimated_total_steps <= 0)
    {
        std::cerr << "No valid time step schedule produced for T_total=" << case_param.T_total << std::endl;
        return -1;
    }

    const int history_output_step = std::max(1, case_param.history_output_step);
    const int pv_output_step =
        case_param.pv_output_step > 0 ? case_param.pv_output_step : std::max(1, estimated_total_steps / 10);
    const int final_step_to_save = case_param.step_to_save > 0 ? case_param.step_to_save : estimated_total_steps;

    case_param.max_step     = estimated_total_steps;
    case_param.step_to_save = final_step_to_save;

    const bool should_record_paras = case_param.record_paras();
    if (should_record_paras)
    {
        case_param.paras_record.record("mhd_grid", std::string("mac"))
            .record("dt", time_cfg.dt)
            .record("dt_base", base_time_step_selection.selected_dt)
            .record("dt_base_convective", base_time_step_selection.convective_dt)
            .record("dt_base_diffusion_limit", base_time_step_selection.diffusion_dt_limit)
            .record("dt_base_magnetic_limit", base_time_step_selection.magnetic_dt_limit)
            .record("dt_startup", has_requested_startup_dt ? startup_time_step_selection.selected_dt : 0.0)
            .record("dt_startup_convective", has_requested_startup_dt ? startup_time_step_selection.convective_dt : 0.0)
            .record("dt_startup_diffusion_limit",
                    has_requested_startup_dt ? startup_time_step_selection.diffusion_dt_limit : 0.0)
            .record("dt_startup_magnetic_limit",
                    has_requested_startup_dt ? startup_time_step_selection.magnetic_dt_limit : 0.0)
            .record("dt_startup_active", time_step_schedule.has_startup_dt ? 1 : 0)
            .record("dt_startup_t_end", time_step_schedule.startup_t_end)
            .record("estimated_total_steps", estimated_total_steps)
            .record("viscosity_upper_bound_raw", base_time_step_selection.viscosity_upper_bound_raw)
            .record("viscosity_upper_bound_effective", base_time_step_selection.viscosity_upper_bound_effective)
            .record("viscosity_upper_bound", base_time_step_selection.viscosity_upper_bound_effective)
            .record("magnetic_factor_sq", base_time_step_selection.magnetic_factor_sq)
            .record("dt_diffusion_limited", base_time_step_selection.diffusion_limited ? 1 : 0)
            .record("dt_magnetic_limited", base_time_step_selection.magnetic_limited ? 1 : 0)
            .record("dt_startup_diffusion_limited",
                    has_requested_startup_dt && startup_time_step_selection.diffusion_limited ? 1 : 0)
            .record("dt_startup_magnetic_limited",
                    has_requested_startup_dt && startup_time_step_selection.magnetic_limited ? 1 : 0)
            .record("scalar_transport_enabled", enable_scalar_transport ? 1 : 0)
            .record("Sc", Sc)
            .record("Pe", Pe)
            .record("nr", nr)
            .record("scalar_scheme", scalar_scheme_name);
    }

    const double lx2 = case_param.lx_2;
    const double ly2 = case_param.ly_2;
    const double lx1 = case_param.lx_1;
    const double lx3 = case_param.lx_3;
    const double ly4 = case_param.ly_4;
    const double ly5 = case_param.ly_5;

    const int nx2 = static_cast<int>(lx2 / h);
    const int ny2 = static_cast<int>(ly2 / h);
    const int nx1 = static_cast<int>(lx1 / h);
    const int nx3 = static_cast<int>(lx3 / h);
    const int ny4 = static_cast<int>(ly4 / h);
    const int ny5 = static_cast<int>(ly5 / h);

    const int    ny1 = ny2;
    const int    ny3 = ny2;
    const int    nx4 = nx2;
    const int    nx5 = nx2;
    const double ly1 = ly2;
    const double ly3 = ly2;
    const double lx4 = lx2;
    const double lx5 = lx2;

    std::cout << "Construct cross-shaped channel geometry (MHD + Non-Newtonian + Stability)..." << std::endl;
    std::cout << "Domain A2: " << nx2 << " x " << ny2 << " (" << lx2 << " x " << ly2 << ")" << std::endl;
    std::cout << "Domain A1: " << nx1 << " x " << ny1 << " (" << lx1 << " x " << ly1 << ")" << std::endl;
    std::cout << "Domain A3: " << nx3 << " x " << ny3 << " (" << lx3 << " x " << ly3 << ")" << std::endl;
    std::cout << "Domain A4: " << nx4 << " x " << ny4 << " (" << lx4 << " x " << ly4 << ")" << std::endl;
    std::cout << "Domain A5: " << nx5 << " x " << ny5 << " (" << lx5 << " x " << ly5 << ")" << std::endl;

    const int total_grid = nx1 * ny1 + nx2 * ny2 + nx3 * ny3 + nx4 * ny4 + nx5 * ny5;
    std::cout << "Total grid points: " << total_grid << std::endl;

    Domain2DUniform A2(nx2, ny2, lx2, ly2, "A2");
    Domain2DUniform A1(nx1, ny1, lx1, ly1, "A1");
    Domain2DUniform A3(nx3, ny3, lx3, ly3, "A3");
    Domain2DUniform A4(nx4, ny4, lx4, ly4, "A4");
    Domain2DUniform A5(nx5, ny5, lx5, ly5, "A5");

    geo.connect(&A2, LocationType::XNegative, &A1);
    geo.connect(&A2, LocationType::XPositive, &A3);
    geo.connect(&A2, LocationType::YNegative, &A4);
    geo.connect(&A2, LocationType::YPositive, &A5);
    geo.axis(&A2, LocationType::XNegative);
    geo.axis(&A2, LocationType::YNegative);
    geo.check();
    geo.solve_prepare();

    const double reference_domain_width = lx2;
    const double geometry_center_x      = 0.5 * lx2;
    const double geometry_center_y      = 0.5 * ly2;

    Variable2D u("u"), v("v"), p("p");
    u.set_geometry(geo);
    v.set_geometry(geo);
    p.set_geometry(geo);

    Variable2D phi("phi");
    if (enable_mhd)
        phi.set_geometry(geo);

    Variable2D c("c");
    if (enable_scalar_transport)
        c.set_geometry(geo);

    Variable2D mu("mu"), tau_xx("tau_xx"), tau_yy("tau_yy"), tau_xy("tau_xy");
    mu.set_geometry(geo);
    tau_xx.set_geometry(geo);
    tau_yy.set_geometry(geo);
    tau_xy.set_geometry(geo);

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

    field2 phi_A1("phi_A1"), phi_A2("phi_A2"), phi_A3("phi_A3"), phi_A4("phi_A4"), phi_A5("phi_A5");
    if (enable_mhd)
    {
        phi.set_center_field(&A1, phi_A1);
        phi.set_center_field(&A2, phi_A2);
        phi.set_center_field(&A3, phi_A3);
        phi.set_center_field(&A4, phi_A4);
        phi.set_center_field(&A5, phi_A5);
    }

    field2 c_A1("c_A1"), c_A2("c_A2"), c_A3("c_A3"), c_A4("c_A4"), c_A5("c_A5");
    if (enable_scalar_transport)
    {
        c.set_center_field(&A1, c_A1);
        c.set_center_field(&A2, c_A2);
        c.set_center_field(&A3, c_A3);
        c.set_center_field(&A4, c_A4);
        c.set_center_field(&A5, c_A5);
    }

    field2 mu_A1("mu_A1"), mu_A2("mu_A2"), mu_A3("mu_A3"), mu_A4("mu_A4"), mu_A5("mu_A5");
    field2 txx_A1("txx_A1"), txx_A2("txx_A2"), txx_A3("txx_A3"), txx_A4("txx_A4"), txx_A5("txx_A5");
    field2 tyy_A1("tyy_A1"), tyy_A2("tyy_A2"), tyy_A3("tyy_A3"), tyy_A4("tyy_A4"), tyy_A5("tyy_A5");
    field2 txy_A1("txy_A1"), txy_A2("txy_A2"), txy_A3("txy_A3"), txy_A4("txy_A4"), txy_A5("txy_A5");

    mu.set_corner_field(&A1, mu_A1);
    mu.set_corner_field(&A2, mu_A2);
    mu.set_corner_field(&A3, mu_A3);
    mu.set_corner_field(&A4, mu_A4);
    mu.set_corner_field(&A5, mu_A5);
    tau_xx.set_center_field(&A1, txx_A1);
    tau_xx.set_center_field(&A2, txx_A2);
    tau_xx.set_center_field(&A3, txx_A3);
    tau_xx.set_center_field(&A4, txx_A4);
    tau_xx.set_center_field(&A5, txx_A5);
    tau_yy.set_center_field(&A1, tyy_A1);
    tau_yy.set_center_field(&A2, tyy_A2);
    tau_yy.set_center_field(&A3, tyy_A3);
    tau_yy.set_center_field(&A4, tyy_A4);
    tau_yy.set_center_field(&A5, tyy_A5);
    tau_xy.set_corner_field(&A1, txy_A1);
    tau_xy.set_corner_field(&A2, txy_A2);
    tau_xy.set_corner_field(&A3, txy_A3);
    tau_xy.set_corner_field(&A4, txy_A4);
    tau_xy.set_corner_field(&A5, txy_A5);

    auto set_dirichlet_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Dirichlet);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto set_neumann_zero = [](Variable2D& var, Domain2DUniform* d, LocationType loc) {
        var.set_boundary_type(d, loc, PDEBoundaryType::Neumann);
        var.set_boundary_value(d, loc, 0.0);
    };
    auto is_adjacented = [&](Domain2DUniform* d, LocationType loc) {
        return geo.adjacency.count(d) && geo.adjacency[d].count(loc);
    };

    std::vector<Domain2DUniform*> domains = {&A1, &A2, &A3, &A4, &A5};
    std::vector<LocationType>     dirs    = {
        LocationType::XNegative, LocationType::XPositive, LocationType::YNegative, LocationType::YPositive};

    for (auto* d : domains)
    {
        for (auto loc : dirs)
        {
            if (is_adjacented(d, loc))
                continue;

            set_dirichlet_zero(u, d, loc);
            set_dirichlet_zero(v, d, loc);
            set_neumann_zero(p, d, loc);
            if (enable_scalar_transport)
                set_neumann_zero(c, d, loc);
        }
    }

    set_dirichlet_zero(p, &A4, LocationType::YNegative);
    set_dirichlet_zero(p, &A5, LocationType::YPositive);

    if (enable_mhd)
    {
        for (auto* d : domains)
        {
            for (auto loc : dirs)
            {
                if (is_adjacented(d, loc))
                    continue;
                set_neumann_zero(phi, d, loc);
            }
        }

        phi.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A1, LocationType::XNegative, 0.0);
        phi.has_boundary_value_map[&A1][LocationType::XNegative] = true;
        phi.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A3, LocationType::XPositive, 0.0);
        phi.has_boundary_value_map[&A3][LocationType::XPositive] = true;
        phi.set_boundary_type(&A4, LocationType::YNegative, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A4, LocationType::YNegative, 0.0);
        phi.has_boundary_value_map[&A4][LocationType::YNegative] = true;
        phi.set_boundary_type(&A5, LocationType::YPositive, PDEBoundaryType::Dirichlet);
        phi.set_boundary_value(&A5, LocationType::YPositive, 0.0);
        phi.has_boundary_value_map[&A5][LocationType::YPositive] = true;
    }

    if (enable_scalar_transport)
    {
        set_dirichlet_zero(c, &A1, LocationType::XNegative);
        c.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
        c.set_boundary_value(&A3, LocationType::XPositive, 1.0);
        c.has_boundary_value_map[&A3][LocationType::XPositive] = true;
        set_neumann_zero(c, &A4, LocationType::YNegative);
        set_neumann_zero(c, &A5, LocationType::YPositive);
    }

    u.set_boundary_type(&A1, LocationType::XNegative, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A1, LocationType::XNegative, 0.0);
    u.has_boundary_value_map[&A1][LocationType::XNegative] = true;
    set_dirichlet_zero(v, &A1, LocationType::XNegative);
    for (int j = 0; j < u_A1.get_ny(); ++j)
        u.boundary_value_map[&A1][LocationType::XNegative][j] = 1.0;

    u.set_boundary_type(&A3, LocationType::XPositive, PDEBoundaryType::Dirichlet);
    u.set_boundary_value(&A3, LocationType::XPositive, 0.0);
    u.has_boundary_value_map[&A3][LocationType::XPositive] = true;
    set_dirichlet_zero(v, &A3, LocationType::XPositive);
    for (int j = 0; j < u_A3.get_ny(); ++j)
        u.boundary_value_map[&A3][LocationType::XPositive][j] = -1.0;

    set_neumann_zero(u, &A4, LocationType::YNegative);
    set_neumann_zero(v, &A4, LocationType::YNegative);
    set_neumann_zero(u, &A5, LocationType::YPositive);
    set_neumann_zero(v, &A5, LocationType::YPositive);

    ConcatPoissonSolver2D p_solver(&p);
    ConcatNSSolver2D      ns_solver(&u, &v, &p, &p_solver);
    ns_solver.init_nonnewton(&mu, &tau_xx, &tau_yy, &tau_xy, enable_mhd ? &phi : nullptr);
    std::unique_ptr<ScalarSolver2D> scalar_solver;
    if (enable_scalar_transport)
        scalar_solver = std::make_unique<ScalarSolver2D>(&u, &v, &c, nr, scalar_scheme);

    ns_solver.p_solver->set_parameter(case_param.gmres_m, case_param.gmres_tol, case_param.gmres_max_iter);

    apply_local_streamfunction_perturbation(case_param, u, v, reference_domain_width, geometry_center_x, geometry_center_y);
    ns_solver.phys_boundary_update();
    ns_solver.nondiag_shared_boundary_update();
    ns_solver.diag_shared_boundary_update();

    std::ofstream stability_history(case_param.root_dir + "/stability_history.csv");
    if (!stability_history.is_open())
        throw std::runtime_error("Failed to open stability_history.csv for writing.");
    stability_history << std::setprecision(16);
    write_history_header(stability_history);

    StabilitySummaryAccumulator summary_accumulator;
    const StabilityMetrics      initial_metrics =
        compute_stability_metrics(case_param, u, v, reference_domain_width, geometry_center_x, geometry_center_y);
    write_history_row(stability_history, 0, 0.0, time_cfg.dt, initial_metrics);
    stability_history.flush();
    summary_accumulator.update(initial_metrics, 0, 0.0);

    std::cout << "Starting MHD + Non-Newtonian stability simulation..." << std::endl;

    double current_time = 0.0;
    double last_dt      = time_cfg.dt;
    int    step         = 0;
    bool   diverged     = false;

    while (step < estimated_total_steps)
    {
        const double dt_step = compute_step_dt(current_time, case_param.T_total, time_step_schedule);
        if (dt_step <= 0.0)
            break;

        ++step;
        time_cfg.dt = dt_step;
        last_dt     = dt_step;
        ns_solver.setTimeStep(dt_step);
        if (scalar_solver)
            scalar_solver->setTimeStep(dt_step);

        if (step % 200 == 0)
        {
            env_cfg.showGmresRes = true;
            std::cout << "step: " << step << "/" << estimated_total_steps << ", t = " << current_time
                      << ", dt = " << dt_step << std::endl;
        }
        else
        {
            env_cfg.showGmresRes = (step <= 5);
        }

        {
            Timer step_timer("step_time", TimeRecordType::None, step % 200 == 0);
            ns_solver.solve_nonnewton();
            if (scalar_solver)
                scalar_solver->solve();
        }

        current_time += dt_step;

        if (step % pv_output_step == 0)
        {
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
            IO::write_csv(u, case_param.root_dir + "/u/u_" + std::to_string(step));
            IO::write_csv(v, case_param.root_dir + "/v/v_" + std::to_string(step));
            IO::write_csv(p, case_param.root_dir + "/p/p_" + std::to_string(step));
            IO::write_csv(mu, case_param.root_dir + "/mu/mu_" + std::to_string(step));
            if (enable_mhd)
                IO::write_csv(phi, case_param.root_dir + "/phi/phi_" + std::to_string(step));
            if (enable_scalar_transport)
                IO::write_csv(c, case_param.root_dir + "/c/c_" + std::to_string(step));
        }

        const bool velocity_finite = representative_value_is_finite(u_A1) && representative_value_is_finite(v_A1);
        const bool scalar_finite   = !enable_scalar_transport || representative_value_is_finite(c_A1);
        if (!velocity_finite || !scalar_finite)
        {
            diverged = true;
            std::cout << "=== DIVERGENCE ===" << std::endl;
            break;
        }

        if (step % history_output_step == 0)
        {
            ns_solver.phys_boundary_update();
            ns_solver.nondiag_shared_boundary_update();
            ns_solver.diag_shared_boundary_update();
            const StabilityMetrics metrics =
                compute_stability_metrics(case_param, u, v, reference_domain_width, geometry_center_x, geometry_center_y);
            write_history_row(stability_history, step, current_time, dt_step, metrics);
            stability_history.flush();
            summary_accumulator.update(metrics, step, current_time);
        }
    }

    std::string run_status = diverged ? "diverged" : "finished";

    if (!diverged)
    {
        ns_solver.phys_boundary_update();
        ns_solver.nondiag_shared_boundary_update();
        ns_solver.diag_shared_boundary_update();

        if (summary_accumulator.last_recorded_step != step)
        {
            const StabilityMetrics final_metrics =
                compute_stability_metrics(case_param, u, v, reference_domain_width, geometry_center_x, geometry_center_y);
            write_history_row(stability_history, step, current_time, last_dt, final_metrics);
            stability_history.flush();
            summary_accumulator.update(final_metrics, step, current_time);
        }

        const int runtime_final_step = case_param.step_to_save > 0 ? case_param.step_to_save : step;
        IO::write_csv(u, case_param.root_dir + "/final/u_" + std::to_string(runtime_final_step));
        IO::write_csv(v, case_param.root_dir + "/final/v_" + std::to_string(runtime_final_step));
        IO::write_csv(p, case_param.root_dir + "/final/p_" + std::to_string(runtime_final_step));
        IO::write_csv(mu, case_param.root_dir + "/final/mu_" + std::to_string(runtime_final_step));
        if (enable_mhd)
            IO::write_csv(phi, case_param.root_dir + "/final/phi_" + std::to_string(runtime_final_step));
        if (enable_scalar_transport)
            IO::write_csv(c, case_param.root_dir + "/final/c_" + std::to_string(runtime_final_step));
    }

    write_summary_csv(
        case_param, run_status, diverged, step, current_time, last_dt, estimated_total_steps, summary_accumulator);
    std::cout << "Simulation finished with run_status=" << run_status << std::endl;

    return diverged ? -1 : 0;
}
