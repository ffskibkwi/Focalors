#include "common.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>

namespace IO
{
    void create_directory(const std::string& path)
    {
        fs::path fs_path(path);

        try
        {
            bool is_exist   = fs::exists(fs_path);
            bool is_created = fs::create_directories(fs_path);
        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Filesystem error creating directory '" << path << "': " << e.what() << "\n"
                      << "Path1: " << e.path1() << "\n";
            if (!e.path2().empty())
            {
                std::cerr << "Path2: " << e.path2() << "\n";
            }
            std::cerr << "Error code: " << e.code() << std::endl;
        }
        catch (const std::bad_alloc& e)
        {
            std::cerr << "Memory allocation failed while creating directory '" << path << "': " << e.what()
                      << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Unexpected error creating directory '" << path << "': " << e.what() << std::endl;
        }
    }

    std::string create_timestamp()
    {
        std::stringstream ss;

        auto now          = std::chrono::system_clock::now();
        auto now_time_t   = std::chrono::system_clock::to_time_t(now);
        auto local_time   = *std::localtime(&now_time_t);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        ss << local_time.tm_year + 1900 << std::setw(2) << std::setfill('0') << local_time.tm_mon + 1 << std::setw(2)
           << std::setfill('0') << local_time.tm_mday << "_" << std::setw(2) << std::setfill('0') << local_time.tm_hour
           << std::setw(2) << std::setfill('0') << local_time.tm_min << std::setw(2) << std::setfill('0')
           << local_time.tm_sec << "_" << std::setw(3) << std::setfill('0') << milliseconds.count();

        return ss.str();
    }
} // namespace IO