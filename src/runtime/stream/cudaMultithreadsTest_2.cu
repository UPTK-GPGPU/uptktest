
#include <stdio.h>
#include <thread>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

#define num_stream 8

const int N = 1 << 20;



__global__ void kernel_XXXX(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

void *launch_kernel_XXXX(void *dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    
    UPTKStream_t mystream;
    UPTKStreamCreate(&mystream);

    for(int j = 0; j < 1000 ; j++){
        ///hipLaunchKernelGGL(kernel_XXXX, 1, 64, 0, mystream, data, N);
        kernel_XXXX<<<1, 64, 0, mystream>>> (data, N);

    }
    UPTKStreamSynchronize(mystream);

    return NULL;
}

TEST(cudaStream, cudaMultithreads_2){
    const int num_threads = 8;

    pthread_t threads[num_threads];




    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, launch_kernel_XXXX, 0)) {
            fprintf(stderr, "Error creating threadn");
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
        }
    }

    UPTKDeviceReset();
    fprintf(stderr, "Pass\n");

}