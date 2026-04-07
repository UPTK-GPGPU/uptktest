/*
** test_file: hipGraphTest.cpp
** brief:     hipGraph function/api test
*/

#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>

//#define GRAPH_COPY_SYNC 0
#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 1

#define HIP_CHECK(stat)                                                        \
    ({                                                                          \
        UPTKError_t restat = stat;                                               \
        if(restat != UPTKSuccess)                                                 \
        {                                                                      \
            printf("Error: hip error in line %d,stat:%d\n",__LINE__,restat) ;      \
        }                                                                      \
    })

static double getRealTime() {
    struct timeval tp = {0, 0};
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec * 1000000 + tp.tv_usec);
}

__global__ void reduce(float* d_in, double* d_out, size_t inputSize, size_t outputSize){
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        int blkx = blockIdx.x;
        d_out[blockIdx.x] = d_in[myId];
    }
}
__global__ void reduceFinal(double* d_in, double* d_out, size_t inputSize){
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        *d_out = d_in[myId];
    }
}

void init_input(float* a, size_t size){
    for(size_t i = 0; i < size; i++) a[i] = (rand() & 0xff) / (float)RAND_MAX;
}

bool hipGraphManual(float* inputVec_h, float* inputVec_d, double* outputVec_d, double* result_d,
                    size_t inputSize, size_t numOfBlocks){
    UPTKGraph_t graph;
    UPTKStream_t streamForGraph;
    std::vector<UPTKGraphNode_t> nodeDependencies;
    UPTKGraphNode_t memcpyNode, kernelNode, memsetNode;
    double *result_h = NULL;
    HIP_CHECK(UPTKMallocHost(&result_h, sizeof(double), UPTKHostAllocDefault));

    HIP_CHECK(UPTKStreamCreate(&streamForGraph));

    //double start = getRealTime();

    UPTKKernelNodeParams kernelNodeParams = {0};
    UPTKMemsetParams memsetParams = {0};

    HIP_CHECK(UPTKGraphCreate(&graph, 0));
    HIP_CHECK(UPTKGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0, inputVec_d, inputVec_h, 
                                      sizeof(float)*inputSize, UPTKMemcpyHostToDevice));

    memsetParams.dst = (void*)outputVec_d;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = numOfBlocks * 2;
    memsetParams.height = 1;
    HIP_CHECK(UPTKGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

    nodeDependencies.push_back(memcpyNode);
    nodeDependencies.push_back(memsetNode);

    void* kernelArgs[4] = {(void*)&inputVec_d, (void*)&outputVec_d, &inputSize, &numOfBlocks};
    kernelNodeParams.func = (void*)reduce;
    kernelNodeParams.gridDim = dim3(inputSize / THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;
    kernelNodeParams.kernelParams = (void**)kernelArgs;
    kernelNodeParams.extra = NULL;
    HIP_CHECK(UPTKGraphAddKernelNode(&kernelNode, graph, 
                                    nodeDependencies.data(), nodeDependencies.size(),
                                    &kernelNodeParams));

    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = result_d;
    memsetParams.value = 1;
    memsetParams.elementSize = sizeof(double);
    memsetParams.width = 1;
    memsetParams.height = 1;
    HIP_CHECK(UPTKGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));
    nodeDependencies.push_back(memsetNode);

    memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
    kernelNodeParams.func = (void*)reduceFinal;
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    kernelNodeParams.extra = NULL;
    kernelNodeParams.sharedMemBytes = 0;
    void* kernelArgs2[3] = {(void*)&outputVec_d, (void*)&result_d, &numOfBlocks};
    kernelNodeParams.kernelParams = kernelArgs2;
    HIP_CHECK(UPTKGraphAddKernelNode(&kernelNode, graph,
                                    nodeDependencies.data(), nodeDependencies.size(),
                                    &kernelNodeParams));
    
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);
    HIP_CHECK(UPTKGraphAddMemcpyNode1D(&memcpyNode, graph,
                                      nodeDependencies.data(), nodeDependencies.size(),
                                      result_h, result_d, sizeof(double),
                                      UPTKMemcpyDeviceToHost));
    
    nodeDependencies.clear();
    nodeDependencies.push_back(memcpyNode);
    UPTKGraphNode_t hostNode;
    UPTKGraphExec_t graphExec;
    UPTKGraphNode_t nodes[5] = {};
    size_t numNodes = 5;
    HIP_CHECK(UPTKGraphGetNodes(graph, nodes, &numNodes));
    HIP_CHECK(UPTKGraphGetRootNodes(graph, nodes, &numNodes));

    HIP_CHECK(UPTKGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    

    for(int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++){
        HIP_CHECK(UPTKGraphLaunch(graphExec, streamForGraph));
    }

    HIP_CHECK(UPTKStreamSynchronize(streamForGraph));
    
    HIP_CHECK(UPTKGraphExecDestroy(graphExec));
    HIP_CHECK(UPTKGraphDestroy(graph));
    HIP_CHECK(UPTKStreamDestroy(streamForGraph));

    double result_h_cpu = 0.0;
    for(int i = 0; i < inputSize; i++){
        result_h_cpu += inputVec_h[i];
    }
    if(result_h_cpu != *result_h){
        printf("hipGraphManual faild Final reduced sum = %lf,  get form device result: %lf \n", result_h_cpu, *result_h);
        return false;
    }
    printf("hipGraphManual API Success over result_h_cpu: %lf, get form device result: %lf \n", result_h_cpu, *result_h);
    return true;
}

TEST(cudaGraph,cudaGraphTest){
    std::size_t size = 1 << 12;
    std::size_t maxBlocks = 512;
    UPTKError_t ret = UPTKSuccess;

    float *inputVec_d = nullptr, *inputVec_h = nullptr;
    double *outputVec_d = nullptr, *result_d = nullptr;

#if (defined GRAPH_COPY_SYNC) && (GRAPH_COPY_SYNC == 1)
    inputVec_h = (float*)malloc(sizeof(float) * size);
#else
    ret = UPTKMallocHost(&inputVec_h, sizeof(float) * size, UPTKHostAllocDefault);
    EXPECT_EQ(ret, UPTKSuccess);
#endif

    UPTKSetDevice(0);

    ret = UPTKMalloc(&inputVec_d, sizeof(float) * size);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMalloc(&outputVec_d, sizeof(double) *maxBlocks);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMalloc(&result_d, sizeof(double));

    init_input(inputVec_h, size);

    bool status = hipGraphManual(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
    EXPECT_EQ(status, true);

#if (defined GRAPH_COPY_SYNC) && (GRAPH_COPY_SYNC == 1)
    free(inputVec_h);
#else
    ret = UPTKFreeHost(inputVec_h);
    EXPECT_EQ(ret, UPTKSuccess);
#endif
    
    ret = UPTKFree(inputVec_d);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKFree(outputVec_d);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKFree(result_d);
    EXPECT_EQ(ret, UPTKSuccess);
}