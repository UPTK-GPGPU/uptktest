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

void hostFuncA(void *userData) {
    int *val = (int *)userData;
    *val = 1;
}

void hostFuncB(void *userData) {
    int *val = (int *)userData;
    *val = 2;
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Set host node params and verify via get
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int userData = 0;
        UPTKHostNodeParams params = {};
        params.fn = hostFuncA;
        params.userData = &userData;

        UPTKGraphNode_t hostNode;
        CHECK_CUDA(UPTKGraphAddHostNode(&hostNode, graph, NULL, 0, &params));

        int newUserData = 0;
        UPTKHostNodeParams newParams = {};
        newParams.fn = hostFuncB;
        newParams.userData = &newUserData;

        CHECK_CUDA(UPTKGraphHostNodeSetParams(hostNode, &newParams));

        UPTKHostNodeParams getParams = {};
        CHECK_CUDA(UPTKGraphHostNodeGetParams(hostNode, &getParams));

        if (getParams.fn != hostFuncB) {
            printf("Verification failed: function not updated\n");
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }
        if (getParams.userData != &newUserData) {
            printf("Verification failed: userData not updated\n");
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Set host node params and launch graph
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int userData = 0;
        UPTKHostNodeParams params = {};
        params.fn = hostFuncA;
        params.userData = &userData;

        UPTKGraphNode_t hostNode;
        CHECK_CUDA(UPTKGraphAddHostNode(&hostNode, graph, NULL, 0, &params));

        int newUserData = 0;
        UPTKHostNodeParams newParams = {};
        newParams.fn = hostFuncB;
        newParams.userData = &newUserData;

        CHECK_CUDA(UPTKGraphHostNodeSetParams(hostNode, &newParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));
        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        if (newUserData != 2) {
            printf("Verification failed: host function did not execute correctly\n");
            CHECK_CUDA(UPTKGraphExecDestroy(exec));
            CHECK_CUDA(UPTKGraphDestroy(graph));
            return 1;
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphHostNodeSetParams PASS\n");
    return 0;
}
