#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if defined(__has_include)
#    if __has_include(<filesystem>)
#        include <filesystem>
namespace fs = std::filesystem;
#    elif __has_include(<experimental/filesystem>)
#        include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#    else
#        error "No filesystem support"
#    endif
#elif defined(_MSC_VER)
#    if _MSC_VER >= 1914 // Visual Studio 2017 version 15.7 or later
#        include <filesystem>
namespace fs = std::filesystem;
#    else
#        include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#    endif
#elif defined(__GNUC__)
#    if __GNUC__ >= 8
#        include <filesystem>
namespace fs = std::filesystem;
#    else
#        include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#    endif
#elif defined(__clang__)
#    if __clang_major__ >= 9
#        include <filesystem>
namespace fs = std::filesystem;
#    else
#        include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#    endif
#else
#    error "Compiler not supported"
#endif

namespace IO
{
    void create_directory(const std::string& filepath);

    std::string create_timestamp();
} // namespace IO