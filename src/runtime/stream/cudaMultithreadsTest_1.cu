
#include <stdio.h>
#include <thread>
#include <gtest/gtest.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


const int N = 1 << 20;

__global__ void kernel_1(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

void *launch_kernel_1(void *dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    
    for(int i = 0; i<1000 ; i++){
        ///hipLaunchKernelGGL(kernel_1, 1, 64, 0, 0, data, N);
        kernel_1<<<1, 64>>> (data, N);

    }
    UPTKStreamSynchronize(0);

    return NULL;
}

TEST(cudaStream, cudaMultithreads_1){
    const int num_threads = 8;

    pthread_t threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, launch_kernel_1, 0)) {
            fprintf(stderr, "Error creating threadn");
            //return 1;
        }
    }

    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
            //return 2;
        }
    }

    UPTKDeviceReset();
    fprintf(stderr, "Pass\n");
    //return 0;
}