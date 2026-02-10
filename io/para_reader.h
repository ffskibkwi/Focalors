#pragma once

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace IO
{
    std::unordered_map<std::string, std::string> paras_to_map(int argc, char* argv[]);

    template<typename T>
    typename std::enable_if<std::is_same<T, int>::value || std::is_same<T, double>::value ||
                                std::is_same<T, float>::value,
                            bool>::type
    read_number(const std::unordered_map<std::string, std::string>& para_map, const std::string& token, T& value_ref)
    {
        for (const auto& kv : para_map)
        {
            if (kv.first == token)
            {
                const std::string& value_str = kv.second;

                size_t index = value_str.find("/");
                if (index != std::string::npos)
                {
                    std::string numerator_str   = value_str.substr(0, index);
                    std::string denominator_str = value_str.substr(index + 1);

                    try
                    {
                        double numerator   = std::stod(numerator_str);
                        double denominator = std::stod(denominator_str);
                        if (denominator == 0)
                        {
                            std::cerr << "Error: Division by zero." << std::endl;
                            return false;
                        }

                        value_ref = static_cast<T>(numerator / denominator);
                        return true;
                    }
                    catch (const std::invalid_argument& e)
                    {
                        std::cerr << "Error: Invalid argument, not a number." << std::endl;
                        return false;
                    }
                    catch (const std::out_of_range& e)
                    {
                        std::cerr << "Error: Argument out of range." << std::endl;
                        return false;
                    }
                }

                try
                {
                    value_ref = static_cast<T>(std::stod(value_str));
                    return true;
                }
                catch (const std::invalid_argument& e)
                {
                    std::cerr << "Error: Invalid argument, not a number." << std::endl;
                    return false;
                }
                catch (const std::out_of_range& e)
                {
                    std::cerr << "Error: Argument out of range." << std::endl;
                    return false;
                }
            }
        }

        return false;
    }

    bool read_string(const std::unordered_map<std::string, std::string>& para_map,
                     const std::string&                                  token,
                     std::string&                                        value_ref);

    bool
    read_bool(const std::unordered_map<std::string, std::string>& para_map, const std::string& token, bool& value_ref);
} // namespace IO
