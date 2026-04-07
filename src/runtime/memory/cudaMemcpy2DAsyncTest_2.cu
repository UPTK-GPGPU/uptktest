#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


void rocblas_set_vector_async(int64_t n,
                              int64_t  elem_size,
                              const void* x_h,
                              int64_t incx,
                              void*       y_d,
                              int64_t incy,
                              UPTKStream_t stream)
{
    if(incx == 1 && incy == 1) // contiguous host vector -> contiguous device vector
    {
        UPTKMemcpyAsync(y_d, x_h, elem_size * n, UPTKMemcpyHostToDevice, stream);
    }
    else // either non-contiguous host vector or non-contiguous device vector
    {
        // pretend data is 2D to compensate for non unit increments
        UPTKMemcpy2DAsync(y_d,
                         elem_size * incy,
                         x_h,
                         elem_size * incx,
                         elem_size,
                         n,
                         UPTKMemcpyHostToDevice,
                         stream);
    }
}

void rocblas_get_vector_async(int64_t n,
                              int64_t elem_size,
                              const void* x_d,
                              int64_t incx,
                              void*       y_h,
                              int64_t incy,
                              UPTKStream_t stream)
{
    if(incx == 1 && incy == 1) // congiguous device vector -> congiguous host vector
    {
        UPTKMemcpyAsync(y_h, x_d, elem_size * n, UPTKMemcpyDeviceToHost, stream);
    }
    else // either device or host vector is non-contiguous
    {
        // pretend data is 2D to compensate for non unit increments
        UPTKMemcpy2DAsync(y_h,
                         elem_size * incy,
                         x_d,
                         elem_size * incx,
                         elem_size,
                         n,
                         UPTKMemcpyDeviceToHost,
                         stream);
    }
};

TEST(cudaMemory,cudaMemcpy2DAsyncTest_2){
    UPTKError_t ret = UPTKSuccess;


   	//UPTKStream_t stream1;
	//UPTKStreamCreate(&stream1);
		
	size_t N = 10;
	size_t Nbytes = N * sizeof(float);
	int M = N;
	int incx = 1;
	int incy = 2;
	int incb = 1;

	// Create HIP device buffer
	float *dx;
	UPTKMalloc(&dx, Nbytes);
	
	// Initialize data
	float *hx;
	UPTKMallocHost(&hx, Nbytes, UPTKHostAllocDefault);
	for(size_t i = 0; i < N; i++)
	{
		hx[i] = 1.0;
	}	
		
	float *hy;
	UPTKMallocHost(&hy, 2*Nbytes, UPTKHostAllocDefault);
	for(size_t i = 0; i < 2 * N; i++)
	{
		hy[i] = 2.0;
	}	
	// Copy data to device
	rocblas_set_vector_async(M, sizeof(float), hx, incx, dx, incb, nullptr);
        //UPTKStreamSynchronize();
        //for(size_t i = 0; i < N; i++)
        //{
	//    hx[i] = dx[i];
        //}
	rocblas_get_vector_async(M, sizeof(float), dx, incb, hy, incy, nullptr);
        UPTKStreamSynchronize(NULL);
        for(size_t i = 0; i < N; i++)
        {
            //EXPECT_EQ(hx[i], dx[i])<<"index:"<<i;
            EXPECT_EQ(hx[i], hy[i*2])<<"index:"<<i;
            //printf("hy:%f, index:%d\n",hy[i],i);
        }

	//UPTKStreamSynchronize(stream1);
	
	//printf("success!\n");
    EXPECT_EQ(ret, UPTKSuccess); 
}
