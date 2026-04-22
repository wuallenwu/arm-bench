#include "test_utils.h"
#include "starter/ncnn/baseline/convolution1d_arm.h"

// BASELINE_TESTCASE_START
void test_conv1d_arm_k3() {
    EXPECT_MATCH(run_conv1d_arm, run_ref_conv1d, 2, 4, 8, 3, 1, 0);
    EXPECT_MATCH(run_conv1d_arm, run_ref_conv1d, 4, 8, 16, 3, 1, 1);
    EXPECT_MATCH(run_conv1d_arm, run_ref_conv1d, 8, 4, 12, 3, 2, 0);
}

void test_conv1d_arm_k1() {
    EXPECT_MATCH(run_conv1d_arm, run_ref_conv1d, 3, 4, 8, 1, 1, 0);
}

void test_conv1d_arm_bias() {
    EXPECT_MATCH(run_conv1d_arm, run_ref_conv1d, 4, 8, 8, 3, 1, 1, 1, true);
}
