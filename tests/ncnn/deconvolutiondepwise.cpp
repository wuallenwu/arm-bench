#include "test_utils.h"
#include "starter/ncnn/deconvolutiondepthwise.h"
void test_dw_deconv_base_2x2_s2() {
    ASSERT_TRUE(run_depthwise_deconv2d(2, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_depthwise_deconv2d(4, 4, 4, 2, 2, 2, 2));
}

void test_dw_deconv_base_3x3_s1() {
    ASSERT_TRUE(run_depthwise_deconv2d(2, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_depthwise_deconv2d(4, 5, 5, 3, 3, 1, 1));
}
// CANDIDATE_TESTCASE_END

// BASELINE_TESTCASE_START
// ── DeconvolutionDepthWise_arm ────────────────────────────────────
void test_dw_deconv_arm_2x2_s2() {
    ASSERT_TRUE(run_depthwise_deconv2d_arm(2, 3, 3, 2, 2, 2, 2));
    ASSERT_TRUE(run_depthwise_deconv2d_arm(4, 4, 4, 2, 2, 2, 2));
}

void test_dw_deconv_arm_3x3_s1() {
    ASSERT_TRUE(run_depthwise_deconv2d_arm(2, 4, 4, 3, 3, 1, 1));
    ASSERT_TRUE(run_depthwise_deconv2d_arm(4, 5, 5, 3, 3, 1, 1));
}