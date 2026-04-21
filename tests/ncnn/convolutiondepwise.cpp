#include "test_utils.h"
#include "starter/ncnn/convolutiondepthwise.h"
void test_dw_base_3x3() {
    ASSERT_TRUE(run_depthwise_conv2d(2, 6, 6, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_depthwise_conv2d(4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_depthwise_conv2d(8, 12, 12, 3, 3, 2, 2, 0, 0));
}

void test_dw_base_5x5() {
    ASSERT_TRUE(run_depthwise_conv2d(4, 8, 8, 5, 5, 1, 1, 2, 2));
}

void test_dw_base_bias() {
    ASSERT_TRUE(run_depthwise_conv2d(4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}
// CANDIDATE_TESTCASE_END

// BASELINE_TESTCASE_START
// ── ConvolutionDepthWise_arm ──────────────────────────────────────
void test_dw_arm_3x3() {
    ASSERT_TRUE(run_depthwise_conv2d_arm(2, 6, 6, 3, 3, 1, 1, 0, 0));
    ASSERT_TRUE(run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1));
    ASSERT_TRUE(run_depthwise_conv2d_arm(8, 12, 12, 3, 3, 2, 2, 0, 0));
}

void test_dw_arm_5x5() {
    ASSERT_TRUE(run_depthwise_conv2d_arm(4, 8, 8, 5, 5, 1, 1, 2, 2));
}

void test_dw_arm_bias() {
    ASSERT_TRUE(run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true));
}