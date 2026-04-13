#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>
#include <string.h>

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

    // Scenario 1: Basic functionality - choose device with default properties
    {
        struct UPTKDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.major = 1;
        prop.minor = 0;

        int device = -1;
        UPTKError_t err = UPTKChooseDevice(&device, &prop);
        // May succeed if a matching device exists, or fail with UPTKErrorNoDevice
        if (err != UPTKSuccess && err != UPTKErrorNoDevice) {
            printf("CUDA error: unexpected error from UPTKChooseDevice: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
        if (err == UPTKSuccess && device < 0) {
            printf("CUDA error: device should be >= 0 on success\n");
            return 1;
        }
    }

    // Scenario 2: Choose device with specific compute capability
    {
        struct UPTKDeviceProp prop;
        memset(&prop, 0, sizeof(prop));

        // Get actual device properties to create a match
        struct UPTKDeviceProp actualProp;
        CHECK_CUDA(UPTKGetDeviceProperties(&actualProp, 0));

        prop.major = actualProp.major;
        prop.minor = actualProp.minor;

        int device = -1;
        CHECK_CUDA(UPTKChooseDevice(&device, &prop));
        if (device < 0) {
            printf("CUDA error: device should be >= 0\n");
            return 1;
        }
    }

    // Scenario 3: Error handling - null device pointer
    {
        struct UPTKDeviceProp prop;
        memset(&prop, 0, sizeof(prop));
        prop.major = 1;
        prop.minor = 0;

        UPTKError_t err = UPTKChooseDevice(NULL, &prop);
        if (err != UPTKErrorInvalidValue) {
            printf("CUDA error: expected UPTKErrorInvalidValue for null device, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    // Scenario 4: Error handling - null property pointer
    {
        int device = -1;
        UPTKError_t err = UPTKChooseDevice(&device, NULL);
        if (err != UPTKErrorInvalidValue) {
            printf("CUDA error: expected UPTKErrorInvalidValue for null prop, got: %s\n",
                   UPTKGetErrorString(err));
            return 1;
        }
    }

    printf("test_cudaChooseDevice PASS\n");
    return 0;
}
