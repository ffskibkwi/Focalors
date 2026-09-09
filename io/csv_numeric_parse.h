#pragma once

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace IO
{
    namespace detail
    {
        // Kept separate so CSV numeric readback has a small regression seam.
        inline double parse_csv_double(const std::string& token)
        {
            errno                    = 0;
            char*        end         = nullptr;
            const double value       = std::strtod(token.c_str(), &end);
            const int    parse_errno = errno;
            if (end == token.c_str())
                throw std::invalid_argument("CSV token contains no number");
            while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end)))
                ++end;
            if (*end != '\0' || !std::isfinite(value))
                throw std::invalid_argument("CSV token is not a complete finite number");

            // strtod can set ERANGE for a representable, nonzero subnormal.
            // Preserve that result; never clamp it to zero. A zero underflow,
            // overflow (including finite saturation), or other error still fails.
            if (parse_errno == ERANGE && !(value != 0.0 && std::abs(value) <= std::numeric_limits<double>::min()))
                throw std::out_of_range("CSV number is not representable as a finite nonzero double");
            if (parse_errno != 0 && parse_errno != ERANGE)
                throw std::invalid_argument("CSV numeric conversion failed");
            return value;
        }
    } // namespace detail
} // namespace IO
