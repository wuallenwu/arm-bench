#include "test_utils.h"
#include "starter/ncnn/baseline/convolutiondepthwise_arm.h"

// BASELINE_TESTCASE_START
// ── ConvolutionDepthWise_arm ──────────────────────────────────────
void perf_dw_arm_3x3() {
    run_depthwise_conv2d_arm(2, 6, 6, 3, 3, 1, 1, 0, 0);
    run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1);
    run_depthwise_conv2d_arm(8, 12, 12, 3, 3, 2, 2, 0, 0);
}

void perf_dw_arm_5x5() {
    run_depthwise_conv2d_arm(4, 8, 8, 5, 5, 1, 1, 2, 2);
}

void perf_dw_arm_bias() {
    run_depthwise_conv2d_arm(4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
}