
#include <iostream>
#include <gtest/gtest.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


#define WIDTH 1024

#define NUM (WIDTH * WIDTH)
#define nStreams 2



void MatrixCPUAddition_01(int* a, int *b, int*c) {
    for(int i=0;i<NUM;i++) 
    {  
        c[i] = (a[i] + b[i]) / 2;  
    }
}

__global__ void MatrixAddition_01(int* a, int *b, int*c) {
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < NUM)  
    {  
        c[threadID] = (a[threadID] + b[threadID]) / 2;  
    }
}

TEST(cudaStream, cudaStreamCreateTest9)
{
    UPTKError_t ret = UPTKSuccess;

    int *host_a, *host_b, *host_c, *result;  
    int *dev_a, *dev_b, *dev_c;  
    int *dev_a1, *dev_b1, *dev_c1;

    int streamSize = NUM/nStreams;
    int streamBytes = streamSize * sizeof(int);

    host_a = (int*)malloc(NUM * sizeof(int));
    host_b = (int*)malloc(NUM * sizeof(int));
    host_c = (int*)malloc(NUM * sizeof(int));
    result = (int*)malloc(NUM * sizeof(int));

    UPTKMalloc((void**)&dev_a, streamSize * sizeof(int));
    UPTKMalloc((void**)&dev_a1, streamSize * sizeof(int));

    UPTKMalloc((void**)&dev_b, streamSize * sizeof(int));
    UPTKMalloc((void**)&dev_b1, streamSize * sizeof(int));

    UPTKMalloc((void**)&dev_c, NUM * sizeof(int));
    UPTKMalloc((void**)&dev_c1, NUM * sizeof(int));

    for (int i = 0; i < NUM; i++)  
    {  
        host_a[i] = i;  
        host_b[i] = NUM - i;  
    }

    UPTKStream_t pstreams[nStreams];
    for(int i = 0 ;i < nStreams; i++){
	    ret = UPTKStreamCreate(&pstreams[i]);
        EXPECT_EQ(ret, UPTKSuccess);
    }

    UPTKMemcpyAsync(dev_a, &host_a[0] , streamBytes ,UPTKMemcpyHostToDevice,pstreams[0]);
    UPTKMemcpyAsync(dev_b, &host_b[0] , streamBytes ,UPTKMemcpyHostToDevice,pstreams[0]);
    ///hipLaunchKernelGGL(MatrixAddition_01, streamSize/WIDTH , WIDTH, 0, pstreams[0] , dev_a, dev_b, dev_c);
    MatrixAddition_01<<<streamSize/WIDTH , WIDTH, 0, pstreams[0]>>>(dev_a, dev_b, dev_c);

    UPTKMemcpyAsync(&host_c[0] , dev_c, streamBytes, UPTKMemcpyDeviceToHost , pstreams[0]);


    UPTKMemcpyAsync(dev_a1, &host_a[streamSize] , streamBytes, UPTKMemcpyHostToDevice, pstreams[1]);
    UPTKMemcpyAsync(dev_b1, &host_b[streamSize] , streamBytes, UPTKMemcpyHostToDevice, pstreams[1]);
    ///hipLaunchKernelGGL(MatrixAddition_01, streamSize/WIDTH , WIDTH, 0, pstreams[1] , dev_a1, dev_b1, dev_c1);
     MatrixAddition_01<<<streamSize/WIDTH , WIDTH, 0, pstreams[1]>>>(dev_a1, dev_b1, dev_c1);

    UPTKMemcpyAsync(&host_c[streamSize] , dev_c1, streamBytes, UPTKMemcpyDeviceToHost , pstreams[1]);


    MatrixCPUAddition_01(host_a,host_b,result);
    
    // verify the results
    int errors = 0;
    double eps = 1.0E-6;
    for (int i = 0; i < NUM; i++) {
        if (std::abs(host_c[i] - result[i]) > eps) {
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

    //free stream and mem
    free(host_a);
    free(host_b);
    free(host_c);

    UPTKFree(dev_a);
    UPTKFree(dev_b);
    UPTKFree(dev_c);

    UPTKFree(dev_a1);
    UPTKFree(dev_b1);
    UPTKFree(dev_c1);

    for(int i = 0;i < nStreams ;i++){
		
	    ret = UPTKStreamDestroy(pstreams[i]);
        EXPECT_EQ(ret, UPTKSuccess);
    }


}