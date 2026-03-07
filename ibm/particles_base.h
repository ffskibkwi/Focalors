#pragma once

#include "base/non_copyable.h"
#include "particles_macro.h"

#include <vector>

struct ParticlesBase : public NonCopyable
{
    int max_n = 0;
    int cur_n = 0;

    ParticlesBase() {}

    virtual ~ParticlesBase()
    {
        for (auto& property : properties)
        {
            delete[] property;
        }
    }

    ParticlesBase(ParticlesBase&& rhs) noexcept { swap(*this, rhs); }

    ParticlesBase& operator=(ParticlesBase&& rhs) noexcept
    {
        if (this != &rhs)
        {
            swap(*this, rhs);
        }

        return *this;
    }

    const std::vector<double*> get_properties() const { return properties; }

    friend void swap(ParticlesBase& lhs, ParticlesBase& rhs);

protected:
    // To support polymorphic traversal
    std::vector<double*> properties;
};