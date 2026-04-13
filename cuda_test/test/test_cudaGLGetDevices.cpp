#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    UPTKError_t err = UPTKGetDeviceCount(&deviceCount);
    if (err != UPTKSuccess || deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scene 1: UPTKGLGetDevices requires OpenGL interop support.
    // This API is only available when CUDA is built with OpenGL interop
    // and requires the cuda_gl_interop.h header which is not present
    // in the current environment. The function is used to get a list of
    // CUDA devices that can interoperate with OpenGL.
    // Without GL interop headers and a GL context, this cannot be tested.
    printf("test_skip: UPTKGLGetDevices requires OpenGL interop headers and context, not available in this environment\n");
    return 0;
}
