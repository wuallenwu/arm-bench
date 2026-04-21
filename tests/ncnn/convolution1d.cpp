#include "test_utils.h"
#include "starter/ncnn/convolution1d.h"
void test_conv1d_base_k3() {
    ASSERT_TRUE(run_conv1d(2, 4, 8, 3, 1, 0));
    ASSERT_TRUE(run_conv1d(4, 8, 16, 3, 1, 1));
    ASSERT_TRUE(run_conv1d(8, 4, 12, 3, 2, 0));
}

void test_conv1d_base_k1() {
    ASSERT_TRUE(run_conv1d(3, 4, 8, 1, 1, 0));
}

void test_conv1d_base_bias() {
    ASSERT_TRUE(run_conv1d(4, 8, 8, 3, 1, 1, 1, true));
}
// CANDIDATE_TESTCASE_END

// BASELINE_TESTCASE_START
void test_conv1d_arm_k3() {
    ASSERT_TRUE(run_conv1d_arm(2, 4, 8, 3, 1, 0));
    ASSERT_TRUE(run_conv1d_arm(4, 8, 16, 3, 1, 1));
    ASSERT_TRUE(run_conv1d_arm(8, 4, 12, 3, 2, 0));
}

void test_conv1d_arm_k1() {
    ASSERT_TRUE(run_conv1d_arm(3, 4, 8, 1, 1, 0));
}

void test_conv1d_arm_bias() {
    ASSERT_TRUE(run_conv1d_arm(4, 8, 8, 3, 1, 1, 1, true));
}