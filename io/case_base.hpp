#pragma once

#include "base/parallel/mpi/mpi_misc.h"
#include "instrumentor/profiler.h"
#include "io/common.h"
#include "io/para_reader.h"
#include "io/paras_record.hpp"
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <signal.h>
#include <string>
#include <unistd.h>

void got_signal(int signum)
{
    PROFILE_END_SESSION();
    exit(signum);
}

class CaseBase
{
public:
    CaseBase(int argc, char* argv[])
        : m_argc(argc)
        , m_argv(argv)
    {
        program_name    = argv[0];
        std::size_t pos = program_name.rfind('/');
        if (pos != std::string::npos)
            program_name = program_name.substr(pos + 1);

        para_map  = IO::paras_to_map(argc, argv);
        timestamp = IO::create_timestamp();

        max_threads = omp_get_max_threads();

        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = got_signal;
        sigfillset(&sa.sa_mask);
        sigaction(SIGINT, &sa, NULL);

        is_mpi = MPIUtils::is_mpi();

        if (is_mpi)
        {
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        }

        root_dir = "result/" + program_name + "/" + timestamp;
        IO::read_string(para_map, "root_dir", root_dir);
        if (is_mpi)
        {
            MPIUtils::bcast_string(root_dir);
        }
        MPIUtils::do_once(is_mpi, mpi_rank, [&]() { std::cout << "Root directory: " << root_dir << std::endl; });

        savepoint_root_to_write = root_dir + "/savepoint";

        MPIUtils::do_once(is_mpi, mpi_rank, [&]() { IO::create_directory(root_dir); });

        paras_record = std::move(IO::ParasRecord(root_dir, IO::SimInfoOutputFormat::Matlab));

        if (para_map.find("is_writing_enabled") != para_map.end())
        {
            is_writing_enabled = para_map["is_writing_enabled"] == "true";
        }
        if (is_writing_enabled)
        {
            ProfilerSingleton::Get().EnableWritingProfile();
        }

        MPIUtils::do_once(is_mpi, mpi_rank, [&]() {
            paras_record.record("savepoint_root_to_write", savepoint_root_to_write);
            PROFILE_BEGIN_SESSION(program_name,
                                  root_dir + "/Profile-" + program_name + "-threads" + std::to_string(max_threads) +
                                      ".json");
        });
    }

    ~CaseBase()
    {
        MPIUtils::do_once(is_mpi, mpi_rank, [&]() { PROFILE_END_SESSION(); });
    }

    virtual void read_paras()
    {
        IO::read_string(para_map, "savepoint_root_to_read", savepoint_root_to_read);
        IO::read_number(para_map, "step_to_read", step_to_read);
        IO::read_number(para_map, "step_to_save", step_to_save);
        IO::read_number(para_map, "max_step", max_step);

        step_to_save = step_to_save == -1 ? max_step : step_to_save;
    }

    virtual bool record_paras()
    {
        if (is_mpi && mpi_rank != 0)
            return false;

        paras_record.record("max_threads", max_threads)
            .record("savepoint_root_to_read", savepoint_root_to_read)
            .record("step_to_read", step_to_read)
            .record("max_step", max_step)
            .record("step_to_save", step_to_save)
            .record("mpi_size", mpi_size);

        return true;
    }

    int    m_argc;
    char** m_argv;

    std::string                                  program_name;
    std::string                                  timestamp;
    std::string                                  root_dir;
    int                                          max_threads;
    std::unordered_map<std::string, std::string> para_map;

    IO::ParasRecord paras_record;

    int mpi_rank = 0;
    int mpi_size = 1;

    bool is_mpi             = false;
    bool is_writing_enabled = false;

    int         max_step                = 1;
    int         step_to_read            = 0;
    int         step_to_save            = -1;
    std::string savepoint_root_to_read  = "invalid";
    std::string savepoint_root_to_write = "invalid";
};