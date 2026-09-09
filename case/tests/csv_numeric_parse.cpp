#include "io/csv_numeric_parse.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main()
{
    int                            failures = 0;
    const std::vector<std::string> valid    = {"0",
                                               "-0",
                                               "1.25",
                                               "-3.5e12",
                                               "  1.25\r",
                                               "2.68772e-321",
                                               "-2.68772e-321",
                                               "4.9406564584124654e-324",
                                               "1e-309",
                                               "1.7976931348623157e308"};
    for (const auto& token : valid)
    {
        const double expected = std::strtod(token.c_str(), nullptr);
        try
        {
            const double actual = IO::detail::parse_csv_double(token);
            if (actual != expected || std::signbit(actual) != std::signbit(expected))
            {
                ++failures;
                std::cerr << "FAIL changed valid value: " << token << '\n';
            }
        }
        catch (const std::exception& error)
        {
            ++failures;
            std::cerr << "FAIL rejected valid value: " << token << " (" << error.what() << ")\n";
        }
    }
    for (const std::string token :
         {"", "  ", "junk", "1junk", "1,2", "nan", "inf", "-inf", "1e309", "-1e309", "1e-9999", "-1e-9999"})
    {
        bool rejected = false;
        try
        {
            (void)IO::detail::parse_csv_double(token);
        }
        catch (const std::exception&)
        {
            rejected = true;
        }
        if (!rejected)
        {
            ++failures;
            std::cerr << "FAIL accepted invalid/unrepresentable value: " << token << '\n';
        }
    }
    std::cout << "CSV numeric parsing: " << (failures == 0 ? "PASS" : "FAIL") << "; failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
