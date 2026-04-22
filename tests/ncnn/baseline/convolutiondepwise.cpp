#include "test_utils.h"
#include "starter/ncnn/baseline/convolutiondepthwise_arm.h"

// BASELINE_TESTCASE_START
// ── ConvolutionDepthWise_arm ──────────────────────────────────────
void test_dw_arm_3x3() {
    EXPECT_MATCH(run_depthwise_conv2d_arm, run_ref_depthwise_conv2d, 2, 6, 6, 3, 3, 1, 1, 0, 0);
    EXPECT_MATCH(run_depthwise_conv2d_arm, run_ref_depthwise_conv2d, 4, 8, 8, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d_arm, run_ref_depthwise_conv2d, 8, 12, 12, 3, 3, 2, 2, 0, 0);
}

void test_dw_arm_5x5() {
    EXPECT_MATCH(run_depthwise_conv2d_arm, run_ref_depthwise_conv2d, 4, 8, 8, 5, 5, 1, 1, 2, 2);
}

void test_dw_arm_bias() {
    EXPECT_MATCH(run_depthwise_conv2d_arm, run_ref_depthwise_conv2d, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
}