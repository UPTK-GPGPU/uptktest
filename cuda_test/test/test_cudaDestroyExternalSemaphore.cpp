#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <semaphore.h>

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

    // Scenario 1: Basic functionality - import and destroy external semaphore via POSIX fd
    /*{
        // Create a named semaphore for external semaphore import
        sem_t *sem = sem_open("/cuda_ext_sem_test", O_CREAT, 0666, 0);
        if (sem == SEM_FAILED) {
            printf("test_skip: cannot create semaphore for external semaphore test\n");
            return 0;
        }

        // Get fd from semaphore (not directly possible, so we skip this test)
        sem_close(sem);
        sem_unlink("/cuda_ext_sem_test");

        // Use a simpler approach: just test error handling since external semaphore
        // import requires specific OS support
        printf("test_skip: external semaphore import requires specific OS support\n");
        return 0;
    }*/

    // Scenario 2: Error handling - invalid external semaphore handle
    /*{
        UPTKError_t err = UPTKDestroyExternalSemaphore((UPTKExternalSemaphore_t)0xDEADBEEF);
        if (err != UPTKErrorInvalidValue && err != UPTKErrorInvalidResourceHandle) {
            printf("CUDA error: expected error for invalid handle, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }*/

    // Scenario 3: Error handling - null handle
    {
        UPTKError_t err = UPTKDestroyExternalSemaphore(NULL);
        if (err != UPTKErrorInvalidValue) {
            printf("CUDA error: expected UPTKErrorInvalidValue for null handle, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    printf("test_cudaDestroyExternalSemaphore PASS\n");
    return 0;
}
