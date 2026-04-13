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

static volatile int hostFlag = 0;

void hostFunc(void* userData) {
    int* flag = (int*)userData;
    *flag = 1;
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Set host node params on exec
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int userData = 0;
        UPTKHostNodeParams hostParams = {};
        hostParams.fn = hostFunc;
        hostParams.userData = &userData;

        UPTKGraphNode_t hostNode;
        CHECK_CUDA(UPTKGraphAddHostNode(&hostNode, graph, NULL, 0, &hostParams));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        int newUserData = 0;
        UPTKHostNodeParams newHostParams = {};
        newHostParams.fn = hostFunc;
        newHostParams.userData = &newUserData;

        CHECK_CUDA(UPTKGraphExecHostNodeSetParams(exec, hostNode, &newHostParams));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        if (newUserData != 1) {
            printf("Verification failed: host function did not execute\n");
            return 1;
        }

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Set different host function params
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        int userData1 = 0;
        UPTKHostNodeParams params1 = {};
        params1.fn = hostFunc;
        params1.userData = &userData1;

        UPTKGraphNode_t hostNode;
        CHECK_CUDA(UPTKGraphAddHostNode(&hostNode, graph, NULL, 0, &params1));

        UPTKGraphExec_t exec;
        CHECK_CUDA(UPTKGraphInstantiate(&exec, graph, NULL, NULL, 0));

        int userData2 = 0;
        UPTKHostNodeParams params2 = {};
        params2.fn = hostFunc;
        params2.userData = &userData2;

        CHECK_CUDA(UPTKGraphExecHostNodeSetParams(exec, hostNode, &params2));

        CHECK_CUDA(UPTKGraphLaunch(exec, 0));
        CHECK_CUDA(UPTKStreamSynchronize(0));

        CHECK_CUDA(UPTKGraphExecDestroy(exec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphExecHostNodeSetParams PASS\n");
    return 0;
}
