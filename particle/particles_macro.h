#pragma once

#define DECLARE_PROPERTY(pname)         \
    int     pname##_idx = 0;            \
    double* get_##pname()               \
    {                                   \
        return properties[pname##_idx]; \
    }

#define INITIALIZE_PROPERTY(pname)     \
    double* pname = new double[max_n]; \
    pname##_idx   = properties.size(); \
    for (int i = 0; i < max_n; ++i)    \
    {                                  \
        pname[i] = 0.0;                \
    }                                  \
    properties.push_back(pname);

#define EXPOSE_PROPERTY(p, pname) double* pname = p->get_##pname();
