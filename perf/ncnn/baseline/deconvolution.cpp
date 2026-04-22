#include "test_utils.h"
#include "starter/ncnn/baseline/deconvolution_arm.h"

// BASELINE_TESTCASE_START
// ── Deconvolution_arm ─────────────────────────────────────────────
void perf_deconv_arm_2x2_s2() {
    run_deconv2d_arm(1, 1, 3, 3, 2, 2, 2, 2);
    run_deconv2d_arm(2, 4, 4, 4, 2, 2, 2, 2);
}

void perf_deconv_arm_3x3_s1() {
    run_deconv2d_arm(1, 1, 4, 4, 3, 3, 1, 1);
    run_deconv2d_arm(3, 4, 5, 5, 3, 3, 1, 1);
}

void perf_deconv_arm_bias() {
    run_deconv2d_arm(2, 4, 4, 4, 3, 3, 1, 1, true);
}
// BASELINE_TESTCASE_END