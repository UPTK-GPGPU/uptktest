#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclGetUniqueId) {
    UPTKncclUniqueId id;

    NCCL_CHECK(UPTKncclGetUniqueId(&id));

    // ✅ 基本校验：不能全 0
    bool all_zero = true;
    for (size_t i = 0; i < sizeof(id.internal); i++) {
        if (id.internal[i] != 0) {
            all_zero = false;
            break;
        }
    }

    ASSERT_FALSE(all_zero) << "UniqueId should not be all zeros";
}