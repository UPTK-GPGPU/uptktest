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

void hostFunction(void *userData) {
    int *val = (int *)userData;
    *val = 42;
}

int main() {
    int deviceCount;
    UPTKGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("test_skip: no CUDA device available\n");
        return 0;
    }
    UPTKSetDevice(0);

    // Scenario 1: Add host node to a graph with no dependencies
    UPTKGraph_t graph = NULL;
    CHECK_CUDA(UPTKGraphCreate(&graph, 0));

    int userData = 0;
    UPTKHostNodeParams hostParams = {};
    hostParams.fn = hostFunction;
    hostParams.userData = &userData;

    UPTKGraphNode_t hostNode = NULL;
    CHECK_CUDA(UPTKGraphAddHostNode(&hostNode, graph, NULL, 0, &hostParams));
    if (hostNode == NULL) {
        printf("CUDA error: UPTKGraphAddHostNode returned NULL node\n");
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 2: Add host node with dependency
    UPTKGraphNode_t emptyNode = NULL;
    CHECK_CUDA(UPTKGraphAddEmptyNode(&emptyNode, graph, NULL, 0));

    int userData2 = 0;
    UPTKHostNodeParams hostParams2 = {};
    hostParams2.fn = hostFunction;
    hostParams2.userData = &userData2;

    UPTKGraphNode_t hostNode2 = NULL;
    CHECK_CUDA(UPTKGraphAddHostNode(&hostNode2, graph, &emptyNode, 1, &hostParams2));
    if (hostNode2 == NULL) {
        printf("CUDA error: UPTKGraphAddHostNode with dependency returned NULL\n");
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 3: Verify node type
    UPTKGraphNodeType nodeType;
    CHECK_CUDA(UPTKGraphNodeGetType(hostNode, &nodeType));
    if (nodeType != UPTKGraphNodeTypeHost) {
        printf("CUDA error: expected Host node type, got %d\n", nodeType);
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    // Scenario 4: Instantiate and launch the graph to verify host node execution
    UPTKGraphExec_t graphExec = NULL;
    CHECK_CUDA(UPTKGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    CHECK_CUDA(UPTKGraphLaunch(graphExec, 0));
    CHECK_CUDA(UPTKStreamSynchronize(0));

    if (userData != 42) {
        printf("CUDA error: host node did not execute correctly, userData=%d\n", userData);
        CHECK_CUDA(UPTKGraphExecDestroy(graphExec));
        CHECK_CUDA(UPTKGraphDestroy(graph));
        return 1;
    }

    CHECK_CUDA(UPTKGraphExecDestroy(graphExec));
    CHECK_CUDA(UPTKGraphDestroy(graph));

    printf("test_cudaGraphAddHostNode PASS\n");
    return 0;
}
