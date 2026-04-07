/*
** test_file: hipGraphTest_1.cpp
** brief:     hipGraph stress test and SPACE_KERNEL = 1 (no kernel operation)
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <gtest/gtest.h>

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define CUDACHECK(error)                                                                            \
    {                                                                                              \
        UPTKError_t localError = error;                                                             \
        if ((localError != UPTKSuccess) && (localError != UPTKErrorPeerAccessAlreadyEnabled)) {      \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, UPTKGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }

#define TESTLOOP                    10
#define TESTSIZE_OF_2_POWER_MIN     8
#define TESTSIZE_OF_2_POWER_MAX     18
#define BLOCK_SIZE_MIN              64
#define BLOCK_SIZE_MAX              512
#define GRID_SIZE_MIN               1
#define GRID_SIZE_MAX               64
#define GRAPH_KERNELS_NUM_MIN       6
#define GRAPH_KERNELS_NUM_MAX       17
//#define GRAPH_KERNELS_NUM_MAX       16777216   // 16777216 for space kernel, 10240 for vectoradd
#define SPACE_KERNEL                1

double logtime[10];

static double getRealTime() {
    struct timeval tp = {0, 0};
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec * 1000000 + tp.tv_usec);
}

__global__ void add(float *d_x, float *d_y, float *d_z, size_t N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
#if (SPACE_KERNEL == 0)
        d_z[i] = d_x[i] + d_y[i];
#endif
    }
}

bool hipSerialKernelLaunch_1(float *inputVec_h, float *inputVec_2_h,
                             float *inputVec_d, float *inputVec_2_d,
                             float *outputVec_d,
                             float *result_h,
                             size_t inputSize, 
                             size_t kernelNum, size_t blockSize, size_t gridSize) {
    //return true;
    double startTime = 0;
    double stopTime = 0;

    UPTKStream_t streamSerial;
    CUDACHECK(UPTKStreamCreate(&streamSerial));

    CUDACHECK(UPTKMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, UPTKMemcpyDefault, streamSerial));
    CUDACHECK(UPTKMemcpyAsync(inputVec_2_d, inputVec_2_h, sizeof(float) * inputSize, UPTKMemcpyDefault, streamSerial));
    CUDACHECK(UPTKMemsetAsync(outputVec_d, 0, sizeof(float) * inputSize, streamSerial));
    CUDACHECK(UPTKMemsetAsync(result_h, 0, sizeof(float) * inputSize, streamSerial));

    startTime = getRealTime();
    for (size_t i = 0; i < kernelNum; i++) {
        //hipLaunchKernelGGL(add, gridSize, blockSize, 0, streamSerial , inputVec_d, inputVec_2_d, outputVec_d, inputSize);
        add<<<gridSize, blockSize, 0, streamSerial >>> (inputVec_d, inputVec_2_d, outputVec_d, inputSize);

    }
    double time1 = getRealTime();
    CUDACHECK(UPTKStreamSynchronize(streamSerial));
    stopTime = getRealTime();
    
    CUDACHECK(UPTKMemcpy(result_h, outputVec_d, sizeof(float) * inputSize, UPTKMemcpyDefault));
    // printf("Serial kernel launch %d times cost: %f \n", GRAPH_KERNELS_NUM, stopTime - startTime);
    logtime[0] = time1 - startTime;
    logtime[1] = stopTime - startTime;
    CUDACHECK(UPTKStreamDestroy(streamSerial));


#if (SPACE_KERNEL == 0)
    for (int i = 0; i < gridSize*blockSize; i++) {
        if (result_h[i] != 3.0f) {
            printf("in func: %s Final result_h[%d] != 3.0f error \n", __func__, i);
            return false;
        }
    }
#endif
    // printf("in func: %s success \n", __func__);
    return true;
}

bool hipGraphNodeKernelLaunch_1(float *inputVec_h, float *inputVec_2_h,
                                float *inputVec_d, float *inputVec_2_d,
                                float *outputVec_d, 
                                float *result_h,
                                size_t inputSize, 
                                size_t kernelNum, size_t blockSize, size_t gridSize) {

    //return true;
    double startTime = 0;
    double stopTime = 0;

    UPTKGraph_t graph;
    UPTKStream_t streamForGraph;
    UPTKGraphNode_t kernelNode;
    UPTKKernelNodeParams kernelNodeParams = {0};
    CUDACHECK(UPTKStreamCreate(&streamForGraph));

    CUDACHECK(UPTKMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, UPTKMemcpyHostToDevice, streamForGraph));
    CUDACHECK(UPTKMemcpyAsync(inputVec_2_d, inputVec_2_h, sizeof(float) * inputSize, UPTKMemcpyHostToDevice, streamForGraph));
    //CUDACHECK(UPTKMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, streamForGraph));
    CUDACHECK(UPTKMemset(result_h, 0, sizeof(float) * inputSize));

    startTime = getRealTime();
    CUDACHECK(UPTKGraphCreate(&graph, 0));
    stopTime = getRealTime();
    logtime[2] = stopTime - startTime;

    void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&inputVec_2_d, (void *)&outputVec_d, &inputSize};
    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    startTime = getRealTime();
    for (int i = 0; i < kernelNum; i++) {
        kernelNodeParams.func = (void *) add;
        kernelNodeParams.gridDim = dim3(gridSize, 1, 1);
        kernelNodeParams.blockDim = dim3(blockSize, 1, 1);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = kernelArgs;
        kernelNodeParams.extra = NULL;
        CUDACHECK(UPTKGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelNodeParams));
    }
    stopTime = getRealTime();
    logtime[3] = stopTime - startTime;

    UPTKGraphExec_t graphExec;
    //UPTKGraphNode_t nodes = NULL;
    //size_t numNodes = 0;
    //CUDACHECK(UPTKGraphGetNodes(graph, &nodes, &numNodes));
    // printf("\nNum of nodes in the graph created using hipGraphsManual API = %zu\n", numNodes);
    //CUDACHECK(UPTKGraphGetRootNodes(graph, &nodes, &numNodes));
    // printf("Num of root nodes in the graph created using hipGraphsManual API = %zu\n", numNodes);

    startTime = getRealTime();
    CUDACHECK(UPTKGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    stopTime = getRealTime();
    logtime[4] = stopTime - startTime;

    startTime = getRealTime();
    CUDACHECK(UPTKGraphLaunch(graphExec, streamForGraph));  //execute graph
    double time2 = getRealTime();
    logtime[5] = time2 - startTime;
    CUDACHECK(UPTKStreamSynchronize(streamForGraph));
    stopTime = getRealTime();
    logtime[6] = stopTime - startTime;

    // printf("UPTKGraphLaunch over\n");
    UPTKDeviceSynchronize();
    CUDACHECK(UPTKMemcpy(result_h, outputVec_d, sizeof(float) * inputSize, UPTKMemcpyDeviceToHost));

    startTime = getRealTime();
    CUDACHECK(UPTKGraphExecDestroy(graphExec));
    stopTime = getRealTime();
    logtime[7] = stopTime - startTime;

    startTime = getRealTime();
    CUDACHECK(UPTKGraphDestroy(graph));
    stopTime = getRealTime();
    logtime[8] = stopTime - startTime;


    CUDACHECK(UPTKStreamDestroy(streamForGraph));
#if (SPACE_KERNEL == 0)
    for (int i = 0; i < gridSize*blockSize; i++) {
        if (result_h[i] != 3.0f) {
            printf("in func: %s result_h[%d] != 3.0f error \n", __func__, i);
            return false;
        }
    }
#endif
    // printf("[%s] graph success \n", __func__);
    return true;
}

bool hipGraphCaptureKernelLaunch_1(float *inputVec_h, float *inputVec_2_h,
                                   float *inputVec_d, float *inputVec_2_d,
                                   float *outputVec_d, float *result_h,
                                   size_t inputSize, 
                                   size_t kernelNum) {
    return true;
}

bool graphTest_1() {

    float *inputVec_d = NULL;
    float *inputVec_2_d = NULL;
    float *inputVec_h = NULL;
    float *inputVec_2_h = NULL;
    float *outputVec_d = NULL;
    float *result_h = NULL;

    float dispatch_totaltime = 0;
    float launch_totaltime = 0;

    printf("loop size kernelNum grid block Serial_dispatch Serial_launch GraphCreate NodeAdd Instance dispatch Launch ExecDestory Destory Graph_dispatch_T Graph_launch_T\n");

    for(unsigned int loopnum = 0; loopnum < TESTLOOP; loopnum++) {

        size_t sizeStart = 0x1u << TESTSIZE_OF_2_POWER_MIN;
        size_t sizeEnd   = 0x1u <<TESTSIZE_OF_2_POWER_MAX;
        size_t size = 0;

        size_t kernelNumStart = 0x1u << GRAPH_KERNELS_NUM_MIN;
        size_t kernelNumEnd = 0x1u << GRAPH_KERNELS_NUM_MAX;
        size_t kernelNum = 0;

        size_t gridSizeStart = GRID_SIZE_MIN;
        size_t gridSizeEnd = GRID_SIZE_MAX;
        size_t gridSize = 0;

        size_t blockSizeStart = BLOCK_SIZE_MIN;
        size_t blockSizeEnd = BLOCK_SIZE_MAX;
        size_t blockSize = 0;

        UPTKSetDevice(0);

        for (size = sizeStart; size <= sizeEnd; size = size*2 ){

            for (kernelNum = kernelNumStart; kernelNum <= kernelNumEnd; kernelNum *= 2) {

                for (gridSize = gridSizeStart; gridSize <= gridSizeEnd; gridSize += 63) {

                     for (blockSize = blockSizeStart; blockSize <= blockSizeEnd; blockSize *= 2) {
 
                        CUDACHECK(UPTKMallocHost(&inputVec_h, sizeof(float) * size, UPTKHostAllocDefault));
                        CUDACHECK(UPTKMallocHost(&inputVec_2_h, sizeof(float) * size, UPTKHostAllocDefault));
                        CUDACHECK(UPTKMallocHost(&result_h, sizeof(float) * size, UPTKHostAllocDefault));

                        for (int i = 0; i < size; i++) {
                            inputVec_h[i] = 1.0f;
                            inputVec_2_h[i] = 2.0f;
                            result_h[i] = 0.0f;
                        }

                        CUDACHECK(UPTKMalloc(&inputVec_d, sizeof(float) * size));
                        CUDACHECK(UPTKMalloc(&inputVec_2_d, sizeof(float) * size));
                        CUDACHECK(UPTKMalloc(&outputVec_d, sizeof(float) * size));

                        bool statusSerial = hipSerialKernelLaunch_1(inputVec_h, inputVec_2_h, inputVec_d, inputVec_2_d, outputVec_d,
                                                                    result_h, size, kernelNum, blockSize, gridSize);


                        bool statusNode = hipGraphNodeKernelLaunch_1(inputVec_h, inputVec_2_h, inputVec_d, inputVec_2_d, outputVec_d,
                                                                     result_h, size, kernelNum, blockSize, gridSize);

                        //bool status3 = hipGraphCaptureKernelLaunch_1(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

                        CUDACHECK(UPTKFree(inputVec_d));
                        CUDACHECK(UPTKFree(inputVec_2_d));
                        CUDACHECK(UPTKFree(outputVec_d));
                        CUDACHECK(UPTKFreeHost(inputVec_h));
                        CUDACHECK(UPTKFreeHost(inputVec_2_h));
                        CUDACHECK(UPTKFreeHost(result_h));

                        if (!statusSerial) {
                            printf("Failed during hip without graph\n");
                            exit(1);
                        }
                        if (!statusNode) {
                            printf("Failed during hip graph Kernels \n");
                            exit(1);
                        }
                        inputVec_d = NULL;
                        inputVec_2_d = NULL;
                        inputVec_h = NULL;
                        inputVec_2_h = NULL;
                        outputVec_d = NULL;
                        result_h = NULL;

                        dispatch_totaltime = logtime[2] + logtime[3] + logtime[4] + logtime[5] + logtime[7] + logtime[8];
                        launch_totaltime = logtime[2] + logtime[3] + logtime[4] + logtime[6] + logtime[7] + logtime[8];
                        printf("%d %zu %zu %zu %zu %f %f %f %f %f %f %f %f %f %f %f\n",
                                loopnum, size, kernelNum, gridSize, blockSize,
                                logtime[0], logtime[1], logtime[2], logtime[3], logtime[4], logtime[5], logtime[6], logtime[7], logtime[8],
                                dispatch_totaltime, launch_totaltime);


                    }

                }
                
            }

        }
    }
    return true;
}

TEST(cudaGraph,cudaGraphTest_1) {
    graphTest_1();
}
