#include <gtest/gtest.h>
#include <UPTK_nccl.h>
#include <UPTK_runtime.h>
#include <UPTK_runtime_api.h>
#include <vector>

#define NCCL_CHECK(cmd)                                                          \
    do                                                                           \
    {                                                                            \
        UPTKncclResult_t err = cmd;                                              \
        if (err != UPTKncclSuccess)                                              \
        {                                                                        \
            std::printf("UPTK NCCL error core: %d ,error string: %s at %s:%d\n", \
                        err, UPTKncclGetErrorString(err), __FILE__, __LINE__);   \
            throw std::runtime_error("UPTK error");                              \
        }                                                                        \
    } while (0)

#define UPTK_CHECK(cmd)                                                       \
    do                                                                        \
    {                                                                         \
        UPTKError_t err = (cmd);                                              \
        if (err != UPTKSuccess)                                               \
        {                                                                     \
            std::printf("UPTK error %d at %s:%d\n", err, __FILE__, __LINE__); \
            throw std::runtime_error("UPTK error");                           \
        }                                                                     \
    } while (0)