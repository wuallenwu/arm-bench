#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolution_arm.h"

// BASELINE_TESTCASE_START
// ── Deconvolution_arm ─────────────────────────────────────────────
void test_deconv_arm_2x2_s2() {
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 1, 1, 3, 3, 2, 2, 2, 2);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 2, 4, 4, 4, 2, 2, 2, 2);
}

void test_deconv_arm_3x3_s1() {
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 1, 1, 4, 4, 3, 3, 1, 1);
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 3, 4, 5, 5, 3, 3, 1, 1);
}

void test_deconv_arm_bias() {
    EXPECT_MATCH(run_deconv2d_arm, run_ref_deconv2d, 2, 4, 4, 4, 3, 3, 1, 1, true);
}
// BASELINE_TESTCASE_END