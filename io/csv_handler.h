#pragma once

#include "common.h"

#include "non_copyable.h"

#include <fstream>

class CSVHandler : public NonCopyable
{
public:
    CSVHandler() {}

    CSVHandler(const std::string& filename) { init(filename); }

    void init(const std::string& filename)
    {
        if (is_open)
            return;

        is_open = true;

        m_filename = filename;

        fs::path path(m_filename);
        fs::path dir = path.parent_path();
        IO::create_directory(dir);

        stream.open(m_filename + ".csv", std::ios::app);

        if (!stream.is_open())
        {
            std::cerr << "Failed to open file: " << m_filename + ".csv" << std::endl;
            return;
        }
    }

    ~CSVHandler()
    {
        is_open = false;

        stream.close();

        if (stream.fail())
        {
            std::cout << "Writing to file failed: " << m_filename + ".csv" << std::endl;
        }
    }

    bool          is_open = false;
    std::ofstream stream;

private:
    std::string m_filename;
};