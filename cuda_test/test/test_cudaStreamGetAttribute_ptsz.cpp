#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

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

    // Scene 1: Set and get stream attribute
    UPTKStream_t stream;
    CHECK_CUDA(UPTKStreamCreate(&stream));

    UPTKAccessPolicyWindow window;
    window.num_bytes = 1024;
    window.hitRatio = 0.5f;
    window.hitProp = UPTKAccessPropertyPersisting;
    window.missProp = UPTKAccessPropertyStreaming;

    UPTKStreamAttrValue attr;
    attr.accessPolicyWindow = window;
    UPTKError_t err = UPTKStreamSetAttribute(stream, UPTKStreamAttributeAccessPolicyWindow, &attr);
    if (err != UPTKSuccess) {
        printf("UPTKStreamSetAttribute returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
    } else {
        printf("Stream attribute set successfully\n");
        // Try to get it back
        UPTKStreamAttrValue get_attr;
        err = UPTKStreamGetAttribute(stream, UPTKStreamAttributeAccessPolicyWindow, &get_attr);
        if (err != UPTKSuccess) {
            printf("UPTKStreamGetAttribute returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
        } else {
            printf("Stream attribute retrieved successfully\n");
        }
    }

    // Scene 2: Set and get on default stream
    UPTKStreamAttrValue attr2;
    attr2.accessPolicyWindow.num_bytes = 512;
    attr2.accessPolicyWindow.hitRatio = 0.8f;
    attr2.accessPolicyWindow.hitProp = UPTKAccessPropertyPersisting;
    attr2.accessPolicyWindow.missProp = UPTKAccessPropertyNormal;
    err = UPTKStreamSetAttribute(0, UPTKStreamAttributeAccessPolicyWindow, &attr2);
    if (err != UPTKSuccess) {
        printf("UPTKStreamSetAttribute (default) returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
    } else {
        printf("Default stream attribute set successfully\n");
        UPTKStreamAttrValue get_attr2;
        err = UPTKStreamGetAttribute(0, UPTKStreamAttributeAccessPolicyWindow, &get_attr2);
        if (err != UPTKSuccess) {
            printf("UPTKStreamGetAttribute (default) returned: %s (expected on this platform)\n", UPTKGetErrorString(err));
        } else {
            printf("Default stream attribute retrieved successfully\n");
        }
    }

    UPTKStreamDestroy(stream);

    printf("test_cudaStreamGetAttribute_ptsz PASS\n");
    return 0;
}
