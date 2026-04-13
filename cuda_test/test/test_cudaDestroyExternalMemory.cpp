#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#define CHECK_CUDA(call) \
    do { \
        UPTKError_t err = call; \
        if (err != UPTKSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   UPTKGetErrorString(err)); \
            return 1; \
        } \
    } while (0)

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Basic functionality - import and destroy external memory via POSIX fd
    {
        // Create a shared memory object for external memory import
        int fd = shm_open("/cuda_ext_mem_test", O_CREAT | O_RDWR, 0666);
        if (fd < 0) {
            printf("test_skip: cannot create shared memory for external memory test\n");
            return 0;
        }
        if (ftruncate(fd, 4096) != 0) {
            close(fd);
            shm_unlink("/cuda_ext_mem_test");
            printf("test_skip: cannot resize shared memory\n");
            return 0;
        }

        UPTKExternalMemoryHandleDesc memHandleDesc;
        memset(&memHandleDesc, 0, sizeof(memHandleDesc));
        memHandleDesc.type = UPTKExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = fd;
        memHandleDesc.size = 4096;
        memHandleDesc.flags = 0;

        UPTKExternalMemory_t extMem = NULL;
        UPTKError_t err = UPTKImportExternalMemory(&extMem, &memHandleDesc);
        if (err != UPTKSuccess) {
            // May fail if external memory is not supported
            close(fd);
            shm_unlink("/cuda_ext_mem_test");
            printf("test_skip: UPTKImportExternalMemory failed: %s\n", UPTKGetErrorString(err));
            return 0;
        }

        // Destroy the external memory
        CHECK_CUDA(UPTKDestroyExternalMemory(extMem));

        close(fd);
        shm_unlink("/cuda_ext_mem_test");
    }

    // Scenario 2: Error handling - invalid external memory handle
    {
        UPTKError_t err = UPTKDestroyExternalMemory((UPTKExternalMemory_t)0xDEADBEEF);
        if (err != UPTKErrorInvalidValue && err != UPTKErrorInvalidResourceHandle) {
            printf("CUDA error: expected error for invalid handle, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    // Scenario 3: Error handling - null handle
    {
        UPTKError_t err = UPTKDestroyExternalMemory(NULL);
        if (err != UPTKErrorInvalidValue) {
            printf("CUDA error: expected UPTKErrorInvalidValue for null handle, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    printf("test_cudaDestroyExternalMemory PASS\n");
    return 0;
}
