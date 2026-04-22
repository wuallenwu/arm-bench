#include "test_utils.h"
#include "starter/ncnn/candidate/convolution1d.h"
// CANDIDATE_TESTCASE_START
void test_conv1d_base_k3() {
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 2, 4, 8, 3, 1, 0);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 4, 8, 16, 3, 1, 1);
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 8, 4, 12, 3, 2, 0);
}

void test_conv1d_base_k1() {
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 3, 4, 8, 1, 1, 0);
}

void test_conv1d_base_bias() {
    EXPECT_MATCH(run_conv1d, run_ref_conv1d, 4, 8, 8, 3, 1, 1, 1, true);
}
// CANDIDATE_TESTCASE_END
