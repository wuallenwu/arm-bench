#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolutiondepthwise_arm.h"

// BASELINE_TESTCASE_START
// ── DeconvolutionDepthWise_arm ────────────────────────────────────
void perf_dw_deconv_arm_2x2_s2() {
    run_depthwise_deconv2d_arm(2, 3, 3, 2, 2, 2, 2);
    run_depthwise_deconv2d_arm(4, 4, 4, 2, 2, 2, 2);
}

void perf_dw_deconv_arm_3x3_s1() {
    run_depthwise_deconv2d_arm(2, 4, 4, 3, 3, 1, 1);
    run_depthwise_deconv2d_arm(4, 5, 5, 3, 3, 1, 1);
}
