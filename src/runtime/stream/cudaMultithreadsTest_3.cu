
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

UPTKStream_t mystream[num_stream];


__global__ void kernel_2(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

void *launch_kernel_2(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        //hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[0], data, N);
        kernel_2<<<1, 64 , 0 , mystream[0]>>> (data, N);

    }
    UPTKStreamSynchronize(mystream[0]);

    return NULL;
}

void *launch_kernel_2_1(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[1]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        ///hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[1], data, N);
        kernel_2<<<1, 64 , 0 , mystream[1]>>> (data, N);

    }
    UPTKStreamSynchronize(mystream[1]);

    return NULL;
}
void *launch_kernel_2_2(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[2]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        ///hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[2], data, N);
        kernel_2<<<1, 64 , 0 , mystream[2]>>> (data, N);

    }
    UPTKStreamSynchronize(mystream[2]);

    return NULL;
}
void *launch_kernel_2_3(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[3]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        ///hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[3], data, N);
        kernel_2<<<1, 64 , 0 , mystream[3]>>> (data, N);

    }
    UPTKStreamSynchronize(mystream[3]);

    return NULL;
}
void *launch_kernel_2_4(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[4]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        //hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[4], data, N);
        kernel_2<<<1, 64 , 0 , mystream[4]>>>(data, N);

    }
    UPTKStreamSynchronize(mystream[4]);

    return NULL;
}
void *launch_kernel_2_5(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[5]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        //hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[5], data, N);
        kernel_2<<<1, 64 , 0 , mystream[5]>>> (data, N);

    }
    UPTKStreamSynchronize(mystream[5]);

    return NULL;
}
void *launch_kernel_2_6(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[6]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        ///hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[6], data, N);
        kernel_2<<<1, 64 , 0 , mystream[6]>>> (data, N);

    }
    UPTKStreamSynchronize(mystream[6]);

    return NULL;
}

void *launch_kernel_2_7(void* dummy)
{
    float *data;
    UPTKMalloc(&data, N * sizeof(float));
    UPTKError_t ret = UPTKSuccess;
    ret = UPTKStreamCreate(&mystream[7]);
    EXPECT_EQ(ret, UPTKSuccess);
    for(int j = 0; j < 1000 ; j++){
        //hipLaunchKernelGGL(kernel_2, 1, 64 , 0 , mystream[7], data, N);
        kernel_2<<<1, 64 , 0 , mystream[7]>>>(data, N);

    }
    UPTKStreamSynchronize(mystream[7]);

    return NULL;
}

TEST(cudaStream, cudaMultithreads_3){
    const int num_threads = 8;

    pthread_t threads[num_threads];



    pthread_create(&threads[0], NULL, launch_kernel_2, 0);
    pthread_create(&threads[1], NULL, launch_kernel_2_1, 0);
    pthread_create(&threads[2], NULL, launch_kernel_2_2, 0);
    pthread_create(&threads[3], NULL, launch_kernel_2_3, 0);
    pthread_create(&threads[4], NULL, launch_kernel_2_4, 0);
    pthread_create(&threads[5], NULL, launch_kernel_2_5, 0);
    pthread_create(&threads[6], NULL, launch_kernel_2_6, 0);
    pthread_create(&threads[7], NULL, launch_kernel_2_7, 0);


    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
        }
    }

    UPTKDeviceReset();
    fprintf(stderr, "Pass\n");

}