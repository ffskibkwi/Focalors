#pragma once

#define INITIALIZE_PROPERTY(collection, new_property, number) \
    new_property = new double[number];                        \
    for (int i = 0; i < number; ++i)                          \
    {                                                         \
        new_property[i] = 0.0;                                \
    }                                                         \
    collection.push_back(new_property)
