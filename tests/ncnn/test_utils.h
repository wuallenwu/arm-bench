// test_utils.h — shared test infrastructure for ncnn kernel tests
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
#include <string>
#include <numeric>

#include "starter/ncnn/ncnn_helpers.h"  // ncnn::Mat + read_mat (for expect_mat_near)

// ─── Assertion macros ────────────────────────────────────────────────────────

#define ASSERT_EQ(a, b)                                                        \
    do {                                                                        \
        if ((a) != (b)) {                                                       \
            fprintf(stderr, "  FAIL [%s:%d]  %s == %s  (%d != %d)\n",         \
                    __FILE__, __LINE__, #a, #b, (int)(a), (int)(b));           \
            g_failed++;                                                         \
        }                                                                       \
    } while (0)

#define ASSERT_TRUE(cond)                                                       \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "  FAIL [%s:%d]  %s is false\n",                  \
                    __FILE__, __LINE__, #cond);                                 \
            g_failed++;                                                         \
        }                                                                       \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                  \
    do {                                                                        \
        float _a = (float)(a), _b = (float)(b), _t = (float)(tol);            \
        if (fabsf(_a - _b) > _t) {                                             \
            fprintf(stderr, "  FAIL [%s:%d]  |%s - %s| = %.6f > %.6f\n",      \
                    __FILE__, __LINE__, #a, #b, fabsf(_a - _b), _t);          \
            g_failed++;                                                         \
        }                                                                       \
    } while (0)

#define ASSERT_VEC_NEAR(got, ref, n, tol)                                       \
    do {                                                                        \
        for (int _i = 0; _i < (int)(n); ++_i) {                               \
            float _g = (float)(got)[_i], _r = (float)(ref)[_i];               \
            if (fabsf(_g - _r) > (float)(tol)) {                              \
                fprintf(stderr, "  FAIL [%s:%d]  vec[%d]: got %.6f  ref %.6f\n",\
                        __FILE__, __LINE__, _i, _g, _r);                       \
                g_failed++;                                                     \
                break;                                                          \
            }                                                                   \
        }                                                                       \
    } while (0)

// ─── Test runner ─────────────────────────────────────────────────────────────

static int g_failed = 0;
static int g_passed = 0;

#define RUN_TEST(fn)                                                            \
    do {                                                                        \
        int _before = g_failed;                                                 \
        fn();                                                                   \
        if (g_failed == _before) {                                             \
            printf("  PASS  %s\n", #fn);                                        \
            g_passed++;                                                         \
        } else {                                                                \
            printf("  FAIL  %s\n", #fn);                                        \
        }                                                                       \
    } while (0)

inline void print_summary(const char* suite) {
    int total = g_passed + g_failed;
    printf("\n[%s]  %d / %d passed\n", suite, g_passed, total);
}

// ─── Output comparison: ncnn::Mat (candidate/baseline) vs ncnn::Mat (reference) ─

// Returns true if outputs match within tol. Bumps g_failed on mismatch.
// Both arguments handled via read_mat (from ncnn_helpers.h), which supports
// dims 1/2/3/4.
static inline bool expect_mat_near(const ncnn::Mat& got, const ncnn::Mat& ref, float tol = 1e-3f)
{
    if (got.empty()) {
        fprintf(stderr, "  FAIL  expect_mat_near: got is empty (kernel failed earlier)\n");
        return false;
    }
    std::vector<float> got_flat, ref_flat;
    read_mat(got, got_flat);
    read_mat(ref, ref_flat);
    if (got_flat.size() != ref_flat.size()) { //if there is a size mismatch, g_failed accumulate 2
        fprintf(stderr, "  FAIL  expect_mat_near: size %zu != ref %zu  (got w=%d h=%d c=%d  ref w=%d h=%d c=%d)\n",
                got_flat.size(), ref_flat.size(),
                got.w, got.h, got.c, ref.w, ref.h, ref.c);
        g_failed++;
        return false;
    }
    const int before = g_failed;
    ASSERT_VEC_NEAR(got_flat, ref_flat.data(), (int)got_flat.size(), tol);
    return g_failed == before;
}

// Fuse: call run_fn(args) + ref_fn(args) with identical arg list, then compare.
#define EXPECT_MATCH(run_fn, ref_fn, ...)                                       \
    do {                                                                        \
        ncnn::Mat _got = run_fn(__VA_ARGS__);                                   \
        ncnn::Mat _ref = ref_fn(__VA_ARGS__);                                   \
        ASSERT_TRUE(expect_mat_near(_got, _ref));                               \
    } while (0)
