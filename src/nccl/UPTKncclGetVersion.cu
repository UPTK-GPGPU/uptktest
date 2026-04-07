
#include "UPTK_nccl_utils.h"

// ✅ 测试用例
TEST(uptk_nccl, UPTKncclGetVersion) {
    int version = 0;

    NCCL_CHECK(UPTKncclGetVersion(&version));

    ASSERT_GT(version, 0);
    EXPECT_EQ(UPTK_NCCL_VERSION(UPTK_NCCL_MAJOR, UPTK_NCCL_MINOR, UPTK_NCCL_PATCH), version);

    int major = version / 1000;
    int minor = (version % 1000) / 100;
    int patch = version % 100;

    std::cout << "UPTK NCCL Version: "
              << version << " ("
              << major << "." << minor << "." << patch << ")"
              << std::endl;

    // 简单断言（防止异常版本）
    ASSERT_GE(major, 0);
    ASSERT_GE(minor, 0);
    ASSERT_GE(patch, 0);
}