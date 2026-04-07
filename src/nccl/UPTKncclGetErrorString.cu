#include "UPTK_nccl_utils.h"
TEST(uptk_nccl, UPTKncclGetErrorString)
{
    UPTKncclResult_t errors[] =
        {
            UPTKncclSuccess,
            UPTKncclUnhandledCudaError,
            UPTKncclSystemError,
            UPTKncclInternalError,
            UPTKncclInvalidArgument,
            UPTKncclInvalidUsage,
            UPTKncclRemoteError,
            UPTKncclInProgress,
        };

    for (auto err : errors)
    {
        const char *str = UPTKncclGetErrorString(err);

        ASSERT_NE(str, nullptr);
        printf("Error %d -> %s\n", err, str);
    }
}
