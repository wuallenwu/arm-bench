#include "test_utils.h"
#include "starter/ncnn/convolution.h"
// CANDIDATE_TESTCASE_START
// ── Convolution (base) ────────────────────────────────────────────
void test_conv_base_1x1_s1() {
    ASSERT_TRUE(run_conv2d(1, 1, 4, 4, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d(3, 4, 5, 5, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d(8, 8, 7, 7, 1, 1, 1, 1, 0, 0));
}

void test_conv_base_3x3_s1() {
    ASSERT_TRUE(run_conv2d(1, 1, 5, 5, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d(3, 4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_conv2d(4, 8, 10, 10, 3, 3, 1, 1, 1, 1));
}

void test_conv_base_3x3_s2() {
    ASSERT_TRUE(run_conv2d(1, 1, 8, 8, 3, 3, 2, 2, 0, 0));
    ASSERT_TRUE(run_conv2d(4, 8, 12, 12, 3, 3, 2, 2, 1, 1));
}

void test_conv_base_5x5() {
    ASSERT_TRUE(run_conv2d(3, 4, 10, 10, 5, 5, 1, 1, 2, 2));
}

void test_conv_base_dilation() {
    ASSERT_TRUE(run_conv2d(2, 4, 8, 8, 3, 3, 1, 1, 0, 0, 2, 2));
}

void test_conv_base_bias() {
    ASSERT_TRUE(run_conv2d(4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}
// CANDIDATE_TESTCASE_END

// BASELINE_TESTCASE_START
// ── Convolution_arm ───────────────────────────────────────────────
void test_conv_arm_1x1_s1() {
    ASSERT_TRUE(run_conv2d_arm(1, 1, 4, 4, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(3, 4, 5, 5, 1, 1, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(8, 8, 7, 7, 1, 1, 1, 1, 0, 0));
}

void test_conv_arm_3x3_s1() {
    ASSERT_TRUE(run_conv2d_arm(1, 1, 5, 5, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(3, 4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_conv2d_arm(4, 8, 10, 10, 3, 3, 1, 1, 1, 1));
}

void test_conv_arm_3x3_s2() {
    ASSERT_TRUE(run_conv2d_arm(1, 1, 8, 8, 3, 3, 2, 2, 0, 0));
    ASSERT_TRUE(run_conv2d_arm(4, 8, 12, 12, 3, 3, 2, 2, 1, 1));
}

void test_conv_arm_5x5() {
    ASSERT_TRUE(run_conv2d_arm(3, 4, 10, 10, 5, 5, 1, 1, 2, 2));
}

void test_conv_arm_bias() {
    ASSERT_TRUE(run_conv2d_arm(4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}