#include "test_utils.h"
#include "starter/ncnn/candidate/deconvolutiondepthwise.h"
// CANDIDATE_TESTCASE_START
void test_dw_deconv_base_2x2_s2() {
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d, 2, 3, 3, 2, 2, 2, 2);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d, 4, 4, 4, 2, 2, 2, 2);
}

void test_dw_deconv_base_3x3_s1() {
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d, 2, 4, 4, 3, 3, 1, 1);
    EXPECT_MATCH(run_depthwise_deconv2d, run_ref_depthwise_deconv2d, 4, 5, 5, 3, 3, 1, 1);
}
// CANDIDATE_TESTCASE_END
