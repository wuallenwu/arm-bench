#include "test_utils.h"
#include "starter/ncnn/candidate/convolutiondepthwise.h"
// CANDIDATE_TESTCASE_START
void test_dw_base_3x3() {
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 2, 6, 6, 3, 3, 1, 1, 0, 0);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 4, 8, 8, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 8, 12, 12, 3, 3, 2, 2, 0, 0);
}

void test_dw_base_5x5() {
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 4, 8, 8, 5, 5, 1, 1, 2, 2);
}

void test_dw_base_bias() {
    EXPECT_MATCH(run_depthwise_conv2d, run_ref_depthwise_conv2d, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
}
// CANDIDATE_TESTCASE_END
