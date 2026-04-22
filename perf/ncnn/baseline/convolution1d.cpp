#include "starter/ncnn/baseline/convolution1d_arm.h"

// BASELINE_TESTCASE_START
void perf_conv1d_arm_k3() {
    run_conv1d_arm(2, 4, 8, 3, 1, 0);
    run_conv1d_arm(4, 8, 16, 3, 1, 1);
    run_conv1d_arm(8, 4, 12, 3, 2, 0);
}

void perf_conv1d_arm_k1() {
    run_conv1d_arm(3, 4, 8, 1, 1, 0);
}

void perf_conv1d_arm_bias() {
    run_conv1d_arm(4, 8, 8, 3, 1, 1, 1, true);
}
