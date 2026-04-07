#include "UPTK_nccl_utils.h"

TEST(UPTKNCCLTest, UPTKncclCommAbort) {
    UPTKncclComm_t comm;
    UPTKncclUniqueId id;

    NCCL_CHECK(UPTKncclGetUniqueId(&id));
    NCCL_CHECK(UPTKncclCommInitRank(&comm, 1, id, 0));

    NCCL_CHECK(UPTKncclCommAbort(comm));

}