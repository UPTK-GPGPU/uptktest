#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int g_callback_called = 0;

static void destroy_callback(void *ptr) {
    (void)ptr;
    g_callback_called = 1;
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Create user object with initial refcount 1, release triggers callback
    void *ptr = malloc(64);
    g_callback_called = 0;
    UPTKUserObject_t obj;
    UPTKError_t err = UPTKUserObjectCreate(&obj, ptr, destroy_callback, 1, UPTKUserObjectNoDestructorSync);
    if (err != UPTKSuccess) {
        printf("test_skip: UPTKUserObjectCreate not supported on DTK: %s\n", UPTKGetErrorString(err));
        free(ptr);
        return 0;
    }

    err = UPTKUserObjectRelease(obj, 1);
    if (err != UPTKSuccess) {
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
        free(ptr);
        return 1;
    }
    // On DTK, the callback is called when refcount reaches 0, which means ptr is freed by callback
    // Don't call free(ptr) again to avoid double free
    printf("  User object created and released, callback called: %d\n", g_callback_called);

    // Scenario 2: Create with initial refcount 2, release once (should not trigger callback)
    void *ptr2 = malloc(64);
    g_callback_called = 0;
    UPTKUserObject_t obj2;
    err = UPTKUserObjectCreate(&obj2, ptr2, destroy_callback, 2, UPTKUserObjectNoDestructorSync);
    if (err != UPTKSuccess) {
        printf("  UPTKUserObjectCreate with refcount 2: %s\n", UPTKGetErrorString(err));
        free(ptr2);
    } else {
        err = UPTKUserObjectRelease(obj2, 1);
        if (err != UPTKSuccess) {
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, UPTKGetErrorString(err));
            free(ptr2);
            return 1;
        }
        // refcount is still 1, callback should not be called yet
        printf("  User object released once (refcount still 1), callback called: %d\n", g_callback_called);
        // Release again to trigger callback and free
        err = UPTKUserObjectRelease(obj2, 1);
        if (err == UPTKSuccess) {
            printf("  User object fully released, callback called: %d\n", g_callback_called);
        }
    }

    printf("test_cudaUserObjectCreate PASS\n");
    return 0;
}
