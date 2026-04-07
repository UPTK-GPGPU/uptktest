#include <iostream>
#include <gtest/gtest.h>
#include <sys/time.h>
// hip header file
#include "cuda.h"

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)


__global__ void MatrixAddition_03(float* a, float *b, float*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < NUM)  
    {  
        c[threadID] = (a[threadID] + b[threadID]) / 2;  
    }
}

TEST(hipPerformanceEvent,hipEventRecord_Performance_15){
    cudaError_t ret = cudaSuccess;
    //HOST 
    float* Host_in;
    float* Host_out;

    //DEVICE
    float* Device_in;
    float* Device_out;
    float* result;

    int streamSize = NUM;

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    //cudaStream_t stream1;
    cudaStream_t stream2;
    //ret = cudaStreamCreate(&stream1);
    //EXPECT_EQ(ret, cudaSuccess);
    ret = cudaStreamCreate(&stream2);
    EXPECT_EQ(ret, cudaSuccess);

    //create event
    cudaEvent_t  event;  
    cudaEventCreate(&event);

    float eventMs = 1.0f;
	int i;
    int errors;
	
    std::cout << "Device name " << devProp.name << std::endl;
	
    Host_in = (float*)malloc(NUM * sizeof(float));
    Host_out = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Host_in[i] = (float)i * 10.0f;
    }
	
	// allocate the memory on the device side
    cudaMalloc((void**)&Device_in, NUM * sizeof(float));
    cudaMalloc((void**)&Device_out, NUM * sizeof(float));
    cudaMalloc((void**)&result, NUM * sizeof(float));

    for (i = 0; i < NUM; i++) {
        Device_out[i] = (float)i * 10.0f;
    }

    // Memory transfer from host to device by using stream1
    cudaMemcpyAsync(Device_in, Host_in, NUM * sizeof(float), cudaMemcpyHostToDevice,NULL);

    // Record the start event
    cudaEventRecord(event, NULL);
    
    //Memory transfer from device to Host copy by using stream2
	cudaMemcpyAsync(Host_out, Device_out, NUM * sizeof(float), cudaMemcpyDeviceToHost,stream2);

    // wait for event in stream1
    cudaStreamWaitEvent(stream2,event,0);
    
    //hipLaunchKernelGGL(MatrixAddition_03, streamSize/WIDTH , WIDTH, 0, stream2 , Device_in, Device_out, result);
    MatrixAddition_03<<<streamSize/WIDTH , WIDTH, 0, stream2 >>>(Device_in, Device_out, result);
    free(Host_in);
    free(Host_out);

    cudaFree(Device_in);
    cudaFree(Device_out);
    cudaFree(result);
	
    //ret = cudaStreamDestroy(stream1);
    //EXPECT_EQ(ret, cudaSuccess);
	ret = cudaStreamDestroy(stream2);
    EXPECT_EQ(ret, cudaSuccess);

}
