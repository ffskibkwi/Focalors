#pragma once

#include <cassert>
#include <iostream>
#include <string>

#define ASSERT1(cond)                      assert(cond)
#define ASSERT2(cond, msg)                 assert((cond) || !fprintf(stderr, "%s\n", msg))
#define GET_ASSERT_IMPL(_1, _2, NAME, ...) NAME
#define ASSERT(...)                        GET_ASSERT_IMPL(__VA_ARGS__, ASSERT2, ASSERT1)(__VA_ARGS__)