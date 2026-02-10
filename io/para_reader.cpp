#include "para_reader.h"

#include <algorithm> // For std::transform
#include <cctype>    // For std::tolower

namespace IO
{
    std::unordered_map<std::string, std::string> paras_to_map(int argc, char* argv[])
    {
        std::unordered_map<std::string, std::string> para_map(argc);
        for (int i = 0; i < argc; ++i)
        {
            std::string line(argv[i]);
            int         index = line.find("=");
            if (index != std::string::npos)
            {
                para_map[line.substr(0, index)] = line.substr(index + 1);
            }
        }

        return para_map;
    }

    bool read_string(const std::unordered_map<std::string, std::string>& para_map,
                     const std::string&                                  token,
                     std::string&                                        value_ref)
    {
        for (const auto& kv : para_map)
        {
            if (kv.first == token)
            {
                value_ref = kv.second;
                return true;
            }
        }

        return false;
    }

    bool
    read_bool(const std::unordered_map<std::string, std::string>& para_map, const std::string& token, bool& value_ref)
    {
        std::string value_str;
        if (read_string(para_map, token, value_str))
        {
            // to lower case
            std::transform(
                value_str.begin(), value_str.end(), value_str.begin(), [](unsigned char c) { return std::tolower(c); });

            if (value_str == "true" || value_str == "1")
            {
                value_ref = true;
            }
            else if (value_str == "false" || value_str == "0")
            {
                value_ref = false;
            }
            else
            {
                std::cerr << "Error: Invalid boolean value for token '" << token << "'." << std::endl;
                return false;
            }
            return true;
        }
        return false;
    }
} // namespace IO
