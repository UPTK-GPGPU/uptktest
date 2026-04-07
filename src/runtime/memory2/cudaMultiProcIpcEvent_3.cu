#include <sys/time.h>
#include <stdio.h>
#include <gtest/gtest.h>
#include <cuda.h>
#include "MultiProcess.h"
namespace{
#define N 1
#define threadsPerBlock 1
#define iterations 1
#define num_process 1
#define debug_process 0

void initArrays(float* a, float* b, float* c,const int num){
    for(int i=0;i<N;i++){
			a[i]=1;
			b[i]=3;
			c[i]=0;
    }
}



__global__ void VectorADD(float *a, float *b,float *c,const int num){
    
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < N)  
    {  
        c[threadID] = a[threadID] + b[threadID];  
    }
}

TEST(cudaMemory,cudaMultiProcIpcEvent_3) {
  
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;

    UPTKError_t ret = UPTKSuccess;

    size_t Nbytes = N * sizeof(float);

    A_h = (float *)malloc(Nbytes);
    B_h = (float *)malloc(Nbytes);
    C_h = (float *)malloc(Nbytes);


    ret = UPTKMalloc(&A_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMalloc(&B_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMalloc(&C_d, Nbytes);
    EXPECT_EQ(ret, UPTKSuccess);
  

    UPTKEvent_t start, stop;
    ret =UPTKEventCreate(&start);
    EXPECT_EQ(ret, UPTKSuccess);
    ret =UPTKEventCreate(&stop);
    EXPECT_EQ(ret, UPTKSuccess);

    MultiProcess<UPTKIpcEventHandle_t>* mProcess = new MultiProcess<UPTKIpcEventHandle_t>(num_process);
    mProcess->CreateShmem();   //像是链表？
    printf("num of process  =  %d\n",num_process);
    pid_t pid = mProcess->SpawnProcess(debug_process);
    
    // Parent Process
    if (pid != 0) {

        unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > 1024) blocks = 1024;
        if (blocks == 0) blocks = 1;

        printf("N=%d (A+B+C= %6.1f MB total) blocks=%u threadsPerBlock=%u iterations=%d\n", N,
                ((double)3 * N * sizeof(float)) / 1024 / 1024, blocks, threadsPerBlock, iterations);
        printf("iterations=%d\n", iterations);


        initArrays(A_h, B_h, C_h, N);

        ret=UPTKEventCreateWithFlags(&start, UPTKEventDisableTiming|UPTKEventInterprocess);
        EXPECT_EQ(ret, UPTKSuccess);
	    ret=UPTKEventCreateWithFlags(&stop, UPTKEventDisableTiming|UPTKEventInterprocess);
	    EXPECT_EQ(ret, UPTKSuccess);

        ret=UPTKMemcpy(A_d, A_h, Nbytes, UPTKMemcpyHostToDevice);
        EXPECT_EQ(ret, UPTKSuccess);
	    ret=UPTKMemcpy(B_d, B_h, Nbytes, UPTKMemcpyHostToDevice);
	    EXPECT_EQ(ret, UPTKSuccess);

        for (int i = 0; i < iterations; i++) {
    
            ret=UPTKEventRecord(start, NULL);
	        EXPECT_EQ(ret, UPTKSuccess);
            //hipLaunchKernelGGL(VectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
            //                    A_d, B_d, C_d, N);
            VectorADD<<<dim3(blocks), dim3(threadsPerBlock)>>>(
                                A_d, B_d, C_d, N);
    
            ret=UPTKEventRecord(stop, NULL);
            EXPECT_EQ(ret, UPTKSuccess);
	        ret=UPTKEventSynchronize(stop);
            EXPECT_EQ(ret, UPTKSuccess);
	        ret=UPTKEventQuery(stop);
            EXPECT_EQ(ret, UPTKSuccess);

            float eventMs = 1.0f;
            ret=UPTKEventElapsedTime(&eventMs, start, stop);
            EXPECT_EQ(ret, UPTKSuccess); //figure out later !!!!!!!!!
            printf("kernel_time (UPTKEventElapsedTime) =%6.3fms\n", eventMs);
    
        }
        UPTKIpcEventHandle_t ipc_handle;
        ret = UPTKIpcGetEventHandle(&ipc_handle, start);
        EXPECT_EQ(ret, UPTKSuccess); 

        mProcess->WriteHandleToShmem(ipc_handle);
        mProcess->WaitTillAllChildReads();
    } else{

        UPTKEvent_t ipc_event;
        UPTKIpcEventHandle_t ipc_handle;
        mProcess->ReadHandleFromShmem(ipc_handle);
        ret = UPTKIpcOpenEventHandle(&ipc_event, ipc_handle);
        EXPECT_EQ(ret, UPTKSuccess); 

        ret = UPTKEventSynchronize(ipc_event);
        EXPECT_EQ(ret, UPTKSuccess); 
        ret = UPTKEventDestroy(ipc_event);
        EXPECT_EQ(ret, UPTKSuccess); 
        mProcess->NotifyParentDone();

    }


    if (pid != 0) {
        ret=UPTKMemcpy(C_h, C_d, Nbytes, UPTKMemcpyDeviceToHost);
	    EXPECT_EQ(ret, UPTKSuccess);
    //检查结果可以省略；
	//printf("check:\n");
    //CudaTest::checkVectorADD(A_h, B_h, C_h, N, true);

        ret=UPTKEventDestroy(start);
	    EXPECT_EQ(ret, UPTKSuccess);
        ret=UPTKEventDestroy(stop);
	    EXPECT_EQ(ret, UPTKSuccess);
        delete mProcess;
  }



    free(A_h);
    free(B_h);
    free(C_h);

    UPTKFree(A_d);
    UPTKFree(B_d);
    UPTKFree(C_d);

}
}