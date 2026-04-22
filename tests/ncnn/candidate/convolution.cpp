#include "test_utils.h"
#include "starter/ncnn/candidate/convolution.h"
// CANDIDATE_TESTCASE_START
// ── Convolution (base) ────────────────────────────────────────────
void test_conv_base_1x1_s1() {
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 1, 1, 4, 4, 1, 1, 1, 1, 0, 0);
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 3, 4, 5, 5, 1, 1, 1, 1, 0, 0);
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 8, 8, 7, 7, 1, 1, 1, 1, 0, 0);
}

void test_conv_base_3x3_s1() {
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 1, 1, 5, 5, 3, 3, 1, 1, 0, 0);
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 3, 4, 8, 8, 3, 3, 1, 1, 1, 1);
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 4, 8, 10, 10, 3, 3, 1, 1, 1, 1);
}

void test_conv_base_3x3_s2() {
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 1, 1, 8, 8, 3, 3, 2, 2, 0, 0);
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 4, 8, 12, 12, 3, 3, 2, 2, 1, 1);
}

void test_conv_base_5x5() {
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 3, 4, 10, 10, 5, 5, 1, 1, 2, 2);
}

void test_conv_base_dilation() {
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 2, 4, 8, 8, 3, 3, 1, 1, 0, 0, 2, 2);
}

void test_conv_base_bias() {
    EXPECT_MATCH(run_conv2d, run_ref_conv2d, 4, 8, 8, 8, 3, 3, 1, 1, 1, 1, 1, 1, true);
}
// CANDIDATE_TESTCASE_END
