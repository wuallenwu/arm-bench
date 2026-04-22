#include "test_utils.h"
#include "starter/ncnn/baseline/convolution_arm.h"

// BASELINE_TESTCASE_START
// ── Convolution_arm ───────────────────────────────────────────────
void test_conv_arm_1x1_s1() {
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 1, 1, 4, 4, 1, 1, 1, 1, 0, 0);
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 3, 4, 5, 5, 1, 1, 1, 1, 0, 0);
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 8, 8, 7, 7, 1, 1, 1, 1, 0, 0);
}

void test_conv_arm_3x3_s1() {
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 1, 1, 5, 5, 3, 3, 1, 1, 0, 0);
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 3, 4, 8, 8, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 4, 8, 10, 10, 3, 3, 1, 1, 1, 1);
}

void test_conv_arm_3x3_s2() {
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 1, 1, 8, 8, 3, 3, 2, 2, 0, 0);
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 4, 8, 12, 12, 3, 3, 2, 2, 1, 1);
}

void test_conv_arm_5x5() {
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 3, 4, 10, 10, 5, 5, 1, 1, 2, 2);
}

void test_conv_arm_bias() {
    EXPECT_MATCH(run_conv2d_arm, run_ref_conv2d, 4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
}
