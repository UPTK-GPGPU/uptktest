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

    // Scenario 1: Get event from event record node
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKEvent_t event;
        CHECK_CUDA(UPTKEventCreate(&event));

        UPTKGraphNode_t recordNode;
        CHECK_CUDA(UPTKGraphAddEventRecordNode(&recordNode, graph, NULL, 0, event));

        UPTKEvent_t retrievedEvent;
        CHECK_CUDA(UPTKGraphEventRecordNodeGetEvent(recordNode, &retrievedEvent));

        if (retrievedEvent != event) {
            printf("Verification failed: retrieved event does not match original\n");
            return 1;
        }

        CHECK_CUDA(UPTKEventDestroy(event));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    // Scenario 2: Get event after setting a different event
    {
        UPTKGraph_t graph;
        CHECK_CUDA(UPTKGraphCreate(&graph, 0));

        UPTKEvent_t event1, event2;
        CHECK_CUDA(UPTKEventCreate(&event1));
        CHECK_CUDA(UPTKEventCreate(&event2));

        UPTKGraphNode_t recordNode;
        CHECK_CUDA(UPTKGraphAddEventRecordNode(&recordNode, graph, NULL, 0, event1));

        CHECK_CUDA(UPTKGraphEventRecordNodeSetEvent(recordNode, event2));

        UPTKEvent_t retrievedEvent;
        CHECK_CUDA(UPTKGraphEventRecordNodeGetEvent(recordNode, &retrievedEvent));

        if (retrievedEvent != event2) {
            printf("Verification failed: retrieved event should be event2\n");
            return 1;
        }

        CHECK_CUDA(UPTKEventDestroy(event1));
        CHECK_CUDA(UPTKEventDestroy(event2));
        CHECK_CUDA(UPTKGraphDestroy(graph));
    }

    printf("test_cudaGraphEventRecordNodeGetEvent PASS\n");
    return 0;
}
