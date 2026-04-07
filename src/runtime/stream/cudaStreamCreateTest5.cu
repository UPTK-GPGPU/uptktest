#include <iostream>
#include <gtest/gtest.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define nStreams 4

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrixTranspose_9(float* out, float* in, const int width) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}


// CPU implementation of matrix transpose
void matrixTransposeCPUReference_9(float* output, float* input, const unsigned int width) {
    for (unsigned int j = 0; j < width; j++) {
        for (unsigned int i = 0; i < width; i++) {
            output[i * width + j] = input[j * width + i];
        }
    }
}

TEST(cudaStream, cudaStreamCreateTest5) {
    int i;
    int errors;
	
    const int streamSize = NUM/nStreams;
    const int streamBytes = streamSize * sizeof(float);
    UPTKError_t ret = UPTKSuccess;
    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    UPTKDeviceProp devProp;
    UPTKGetDeviceProperties(&devProp, 0);

    UPTKStream_t pstreams[nStreams];
    for(i = 0 ;i < nStreams; i++){
	ret = UPTKStreamCreate(&pstreams[i]);
        EXPECT_EQ(ret, UPTKSuccess);
    }
    UPTKEvent_t start, stop;
    UPTKEventCreate(&start);
    UPTKEventCreate(&stop);
    float eventMs = 1.0f;

    std::cout << "Device name " << devProp.name << std::endl;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));
    //UPTKMalloc((void**)&Matrix, NUM * sizeof(float));
    // UPTKMalloc((void**)&TransposeMatrix, NUM * sizeof(float));
    cpuTransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    UPTKMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    UPTKMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));
    
    // Record the start event
    UPTKEventRecord(start, NULL);

    // Memory transfer from host to device
	for(i = 0; i<nStreams;i++){
	int offset = i * streamSize;
	UPTKMemcpyAsync(&gpuMatrix[offset], &Matrix[offset], streamBytes, UPTKMemcpyHostToDevice, pstreams[i]);
	
    // Lauching kernel from host
    //hipLaunchKernelGGL(matrixTranspose_9, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
    //    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, pstreams[i] , 0, gpuTransposeMatrix, gpuMatrix, WIDTH);
	matrixTranspose_9<<<dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, pstreams[i]>>>(gpuTransposeMatrix, gpuMatrix, WIDTH);
	UPTKMemcpyAsync(&TransposeMatrix[offset], &gpuMatrix[offset], streamBytes, UPTKMemcpyDeviceToHost, pstreams[i]);
	}
	for(i = 0; i<nStreams;i++){
		UPTKStreamSynchronize(pstreams[i]);
	}
    // Record the stop event
    UPTKEventRecord(stop, NULL);
    UPTKEventSynchronize(stop);

    UPTKEventElapsedTime(&eventMs, start, stop);

    printf("UPTKMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);


    // CPU MatrixTranspose computation
    matrixTransposeCPUReference_9(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors = 0;
    double eps = 1.0E-6;
    for (i = 0; i < NUM; i++) {
        if (std::abs(TransposeMatrix[i] - Matrix[i]) > eps) {
            errors++;
        }
    }
    if (errors != 0) {
        EXPECT_EQ(errors, 0);
        printf("FAILED: %d errors\n", errors);
    }
    else {
        printf("PASSED!\n");
    }
    for(i = 0;i < nStreams ;i++){
		
        ret = UPTKStreamDestroy(pstreams[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
     }
    // free the resources on device side
    UPTKFree(gpuMatrix);
    UPTKFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

}
