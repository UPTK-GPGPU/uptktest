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

void hostFunc(void *userData) {
    int *val = (int *)userData;
    *val = 99;
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Get host node params
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int userData = 42;
        UPTKHostNodeParams params = {};
        params.fn = hostFunc;
        params.userData = &userData;

        UPTKGraphNode_t hostNode;
        CHECK_CUDA(UPTKGraphAddHostNode(&hostNode, graph, NULL, 0, &params));

        UPTKHostNodeParams getParams = {};
        CHECK_CUDA(UPTKGraphHostNodeGetParams(hostNode, &getParams));

        if (getParams.fn != hostFunc) {
            printf("Verification failed: function pointer mismatch\n");
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }
        if (getParams.userData != &userData) {
            printf("Verification failed: userData pointer mismatch\n");
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Get params from host node with different userData
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int userData = 123;
        UPTKHostNodeParams params = {};
        params.fn = hostFunc;
        params.userData = &userData;

        UPTKGraphNode_t hostNode;
        CHECK_CUDA(UPTKGraphAddHostNode(&hostNode, graph, NULL, 0, &params));

        UPTKHostNodeParams getParams = {};
        CHECK_CUDA(UPTKGraphHostNodeGetParams(hostNode, &getParams));

        if (*(int *)getParams.userData != 123) {
            printf("Verification failed: userData value mismatch\n");
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphHostNodeGetParams PASS\n");
    return 0;
}
