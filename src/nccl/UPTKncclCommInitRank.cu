#include "UPTK_nccl_utils.h"

TEST(uptk_nccl, UPTKncclCommInitRank) {
    const int nranks = 1;
    const int rank   = 0;

    UPTKncclUniqueId id;
    UPTKncclComm_t comm;

    // 获取通信ID
    NCCL_CHECK(UPTKncclGetUniqueId(&id));

    // 初始化通信域
    NCCL_CHECK(UPTKncclCommInitRank(&comm, nranks, id, rank));

    // 简单校验
    int out_rank = -1;
    NCCL_CHECK(UPTKncclCommUserRank(comm, &out_rank));
    ASSERT_EQ(out_rank, rank);

    int count = -1;
    NCCL_CHECK(UPTKncclCommCount(comm, &count));
    ASSERT_EQ(count, nranks);

    // 销毁
    NCCL_CHECK(UPTKncclCommDestroy(comm));
}