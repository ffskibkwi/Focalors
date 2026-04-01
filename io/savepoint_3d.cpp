#include "savepoint_3d.h"
#include "common.h" // Assuming IO::create_directory is defined here
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace
{
    /**
     * @brief Helper to convert types to string using overloaded operator<<
     */
    template<typename T>
    std::string to_string_custom(T value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    /**
     * @brief Binary write for field3 (3D data)
     */
    void write_field_binary(field3* f, const std::string& path)
    {
        if (!f)
            return;
        std::ofstream os(path, std::ios::binary);
        if (!os.is_open())
            return;

        // Write dimensions as header for validation during read
        int dims[3] = {f->get_nx(), f->get_ny(), f->get_nz()};
        os.write(reinterpret_cast<const char*>(dims), 3 * sizeof(int));

        // Write raw data buffer
        os.write(reinterpret_cast<const char*>(f->get_ptr(0, 0, 0)), f->get_size_n() * sizeof(double));
        os.close();
    }

    /**
     * @brief Binary read for field3 (3D data)
     */
    void read_field_binary(field3* f, const std::string& path)
    {
        if (!f)
            return;
        std::ifstream is(path, std::ios::binary);
        if (!is.is_open())
            return;

        int dims[3];
        is.read(reinterpret_cast<char*>(dims), 3 * sizeof(int));

        // Safety check: ensure dimensions match the current allocated field
        if (dims[0] == f->get_nx() && dims[1] == f->get_ny() && dims[2] == f->get_nz())
        {
            is.read(reinterpret_cast<char*>(f->get_ptr(0, 0, 0)), f->get_size_n() * sizeof(double));
        }
        is.close();
    }

    /**
     * @brief Binary write for field2 (2D data used in buffers)
     */
    void write_buffer_binary(field2* f, const std::string& path)
    {
        if (!f)
            return;
        std::ofstream os(path, std::ios::binary);
        if (!os.is_open())
            return;
        os.write(reinterpret_cast<const char*>(f->get_ptr(0, 0)), f->get_size_n() * sizeof(double));
        os.close();
    }

    /**
     * @brief Binary read for field2 (2D data used in buffers)
     */
    void read_buffer_binary(field2* f, const std::string& path)
    {
        if (!f)
            return;
        std::ifstream is(path, std::ios::binary);
        if (!is.is_open())
            return;
        is.read(reinterpret_cast<char*>(f->get_ptr(0, 0)), f->get_size_n() * sizeof(double));
        is.close();
    }
} // namespace

void write_savepoint(const Variable3D& var, const std::string& filename)
{
    // Traverse every domain associated with the Variable
    for (auto const& [domain, field] : var.field_map)
    {
        // Construct directory: filename + domain name
        std::string domain_path = filename + "/" + domain->name;
        fs::create_directories(domain_path);

        // 1. Save Main Field: filename/domain_name/field_name
        std::string field_file = domain_path + "/field.bin";
        write_field_binary(field, field_file);

        // 2. Save Buffers: filename/domain_name/location_type_string
        if (var.buffer_map.count(domain))
        {
            for (auto const& [loc, f2] : var.buffer_map.at(domain))
            {
                // Uses the overloaded operator<< for LocationType
                std::string loc_name    = to_string_custom(loc);
                std::string buffer_file = domain_path + "/" + loc_name + ".bin";
                write_buffer_binary(f2, buffer_file);
            }
        }

        // 3. Save Corners: filename/domain_name/corner_pos (Hardcoded)
        auto save_corner = [&](double* ptr, const std::string& suffix, size_t count) {
            if (ptr)
            {
                std::ofstream os(domain_path + "/" + suffix + ".bin", std::ios::binary);
                os.write(reinterpret_cast<const char*>(ptr), count * sizeof(double));
            }
        };

        if (var.corner_x_map.count(domain))
            save_corner(var.corner_x_map.at(domain), "corner_x", domain->nx + 1);
        if (var.corner_y_map.count(domain))
            save_corner(var.corner_y_map.at(domain), "corner_y", domain->ny + 1);
        if (var.corner_z_map.count(domain))
            save_corner(var.corner_z_map.at(domain), "corner_z", domain->nz + 1);
    }
}

void read_savepoint(const Variable3D& var, const std::string& filename)
{
    for (auto const& [domain, field] : var.field_map)
    {
        std::string domain_path = filename + "/" + domain->name;

        // 1. Read Main Field
        read_field_binary(field, domain_path + "/field.bin");

        // 2. Read Buffers
        if (var.buffer_map.count(domain))
        {
            for (auto const& [loc, f2] : var.buffer_map.at(domain))
            {
                std::string loc_name = to_string_custom(loc);
                read_buffer_binary(f2, domain_path + "/" + loc_name + ".bin");
            }
        }

        // 3. Read Corners
        auto load_corner = [&](double* ptr, const std::string& suffix, size_t count) {
            if (ptr)
            {
                std::ifstream is(domain_path + "/" + suffix + ".bin", std::ios::binary);
                if (is.is_open())
                {
                    is.read(reinterpret_cast<char*>(ptr), count * sizeof(double));
                }
            }
        };

        if (var.corner_x_map.count(domain))
            load_corner(var.corner_x_map.at(domain), "corner_x", domain->nx + 1);
        if (var.corner_y_map.count(domain))
            load_corner(var.corner_y_map.at(domain), "corner_y", domain->ny + 1);
        if (var.corner_z_map.count(domain))
            load_corner(var.corner_z_map.at(domain), "corner_z", domain->nz + 1);
    }
}