#include "starter/ncnn/baseline/convolution_arm.h"

// BASELINE_TESTCASE_START
// ── Convolution_arm ───────────────────────────────────────────────
void perf_conv_arm_1x1_s1() {
    run_conv2d_arm(1, 1, 4, 4, 1, 1, 1, 1, 0, 0);
    run_conv2d_arm(3, 4, 5, 5, 1, 1, 1, 1, 0, 0);
    run_conv2d_arm(8, 8, 7, 7, 1, 1, 1, 1, 0, 0);
}

void perf_conv_arm_3x3_s1() {
    run_conv2d_arm(1, 1, 5, 5, 3, 3, 1, 1, 0, 0);
    run_conv2d_arm(3, 4, 8, 8, 3, 3, 1, 1, 1, 1);
    run_conv2d_arm(4, 8, 10, 10, 3, 3, 1, 1, 1, 1);
}

void perf_conv_arm_3x3_s2() {
    run_conv2d_arm(1, 1, 8, 8, 3, 3, 2, 2, 0, 0);
    run_conv2d_arm(4, 8, 12, 12, 3, 3, 2, 2, 1, 1);
}

void perf_conv_arm_5x5() {
    run_conv2d_arm(3, 4, 10, 10, 5, 5, 1, 1, 2, 2);
}

void test_conv_arm_bias() {
    run_conv2d_arm(4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
}
