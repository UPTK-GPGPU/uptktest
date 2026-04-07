#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later)).
 */
inline void enableP2P (int ngpus)
{
    for( int i = 0; i < ngpus; i++ )
    {
        UPTKSetDevice(i);

        for(int j = 0; j < ngpus; j++)
        {
            if(i == j) continue;

            int peer_access_available = 0;
            UPTKDeviceCanAccessPeer(&peer_access_available, i, j);

            if (peer_access_available)
            {
                UPTKDeviceEnablePeerAccess(j, 0);
                printf("> GPU%d enabled direct access to GPU%d\n", i, j);
            }
            else
            {
                printf("(%d, %d)\n", i, j );
            }
        }
    }
}

inline void disableP2P (int ngpus)
{
    for( int i = 0; i < ngpus; i++ )
    {
        UPTKSetDevice(i);

        for(int j = 0; j < ngpus; j++)
        {
            if( i == j ) continue;

            int peer_access_available = 0;
            UPTKDeviceCanAccessPeer( &peer_access_available, i, j );

            if( peer_access_available )
            {
                UPTKDeviceDisablePeerAccess(j);
                printf("> GPU%d disabled direct access to GPU%d\n", i, j);
            }
        }
    }
}

void initialData2(float *ip, int size)
{
    for(int i = 0; i < size; i++)
    {
        ip[i] = (float)rand() / (float)RAND_MAX;
    }
}

bool checkResult2(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            //break;
            return false;
        }
    }

    if (match) printf("Arrays match.\n\n");
    return true;
}

TEST(cudaMemory, cudaMemcpyAsyncTest3)
{
    int ngpus;
    UPTKError_t ret = UPTKSuccess;
    // check device count
    ret = UPTKGetDeviceCount(&ngpus);
    EXPECT_EQ(ret, UPTKSuccess);  
    printf("> cuda-capable device count: %i\n", ngpus);

    if (ngpus > 1) {
        enableP2P(ngpus);
    } else {
       EXPECT_EQ(ret, UPTKSuccess);
       return ;
    }

    // Allocate buffers
    int iSize = 1024;
    const size_t iBytes = iSize * sizeof(float);

    float **d_src = (float **)malloc(sizeof(float*) * ngpus);
    if (!d_src)
    {
        EXPECT_EQ(0, 1);
    }
    float **d_rcv = (float **)malloc(sizeof(float*) * ngpus);
    if (!d_rcv)
    {
        EXPECT_EQ(0, 1);
    }
    float **h_src = (float **)malloc(sizeof(float*) * ngpus);
    if (!h_src)
    {
        EXPECT_EQ(0, 1);
    }
    float **h_rcv = (float **)malloc(sizeof(float*) * ngpus);
    if (!h_rcv)
    {
        EXPECT_EQ(0, 1);
    }
    UPTKStream_t *stream = (UPTKStream_t *)malloc(sizeof(UPTKstream) * ngpus);
    if (!stream)
    {
        EXPECT_EQ(0, 1);
    }

    // Create cuda event handles
    UPTKEvent_t start, stop,start2, stop2;
    ret = UPTKSetDevice(0);
    EXPECT_EQ(ret, UPTKSuccess);  
    ret = UPTKEventCreate(&start);
    EXPECT_EQ(ret, UPTKSuccess);  
    ret = UPTKEventCreate(&stop);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKSetDevice(1);
    EXPECT_EQ(ret, UPTKSuccess);  
    ret = UPTKEventCreate(&start2);
    EXPECT_EQ(ret, UPTKSuccess);  
    ret = UPTKEventCreate(&stop2);
    EXPECT_EQ(ret, UPTKSuccess);  

    for (int i = 0; i < ngpus; i++)
    {
        ret = UPTKSetDevice(i);
        EXPECT_EQ(ret, UPTKSuccess);  
        ret = UPTKMalloc(&d_src[i], iBytes);
        EXPECT_EQ(ret, UPTKSuccess);  
        ret = UPTKMalloc(&d_rcv[i], iBytes);
        EXPECT_EQ(ret, UPTKSuccess);  
        ret = UPTKMallocHost((void **) &h_src[i], iBytes);
        EXPECT_EQ(ret, UPTKSuccess);  
        ret = UPTKMallocHost((void **) &h_rcv[i], iBytes);
        EXPECT_EQ(ret, UPTKSuccess);

        ret = UPTKStreamCreate(&stream[i]);
        EXPECT_EQ(ret, UPTKSuccess);  
    }

    for (int i = 0; i < ngpus; i++)
    {
        initialData2(h_src[i], iSize);
    }
    //  bidirectional asynchronous gmem copy
    ret = UPTKSetDevice(0);
    EXPECT_EQ(ret, UPTKSuccess);
    
    ret = UPTKEventRecord(start, stream[0]);
    EXPECT_EQ(ret, UPTKSuccess);

    ret = UPTKMemcpyAsync(d_src[0], h_src[0], iBytes,UPTKMemcpyHostToDevice, stream[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpyAsync(d_src[1], d_src[0], iBytes,UPTKMemcpyDeviceToDevice, stream[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpyAsync(h_rcv[0], d_src[1], iBytes,UPTKMemcpyDeviceToHost, stream[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventRecord(stop, stream[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventSynchronize(stop);
    EXPECT_EQ(ret, UPTKSuccess);

    float elapsed_time_ms = 0.0f;
    ret = UPTKEventElapsedTime(&elapsed_time_ms, start, stop );
    EXPECT_EQ(ret, UPTKSuccess);

    elapsed_time_ms /= 100.0f;
    printf("Ping-pong bidirectional UPTKMemcpyAsync:\t %8.2fms \n",
           elapsed_time_ms);
    //checkResult2(h_src[0], h_rcv[0], iSize);
    EXPECT_EQ(checkResult2(h_src[0], h_rcv[0], iSize), 1);

    ret = UPTKSetDevice(1);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventRecord(start2, stream[1]);
    EXPECT_EQ(ret, UPTKSuccess);

    printf("host %5.2f  \n", h_src[1][0]);
    ret = UPTKMemcpyAsync(d_rcv[1], h_src[1], iBytes,UPTKMemcpyHostToDevice, stream[1]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpyAsync(d_rcv[0], d_rcv[1], iBytes,UPTKMemcpyDeviceToDevice, stream[1]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpyAsync(h_rcv[1], d_rcv[0], iBytes,UPTKMemcpyDeviceToHost, stream[1]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventRecord(stop2, stream[1]);
    EXPECT_EQ(ret, UPTKSuccess);    

    ret = UPTKEventSynchronize(stop2);
    EXPECT_EQ(ret, UPTKSuccess);
    float elapsed_time_ms2 = 0.0f;
    ret = UPTKEventElapsedTime(&elapsed_time_ms2, start2, stop2);
    EXPECT_EQ(ret, UPTKSuccess);    
    elapsed_time_ms2 /= 100.0f;
    printf("Ping-pong bidirectional cudaMemcpyAsync2:\t %8.2fms \n",
           elapsed_time_ms2);
           
    EXPECT_EQ(checkResult2(h_src[1], d_rcv[1], iSize), 1);

    // free
    ret = UPTKSetDevice(0);
    EXPECT_EQ(ret, UPTKSuccess);
    disableP2P(ngpus);
    
    ret = UPTKEventDestroy(start);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKEventDestroy(stop);
    EXPECT_EQ(ret, UPTKSuccess);
    for (int i = 0; i < ngpus; i++)
    {
        ret = UPTKSetDevice(i);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKFreeHost(h_src[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
        ret = UPTKFreeHost(h_rcv[i]);
        EXPECT_EQ(ret, UPTKSuccess); 
        ret = UPTKFree(d_src[i]);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKFree(d_rcv[i]);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKStreamDestroy(stream[i]);
        EXPECT_EQ(ret, UPTKSuccess);
        ret = UPTKDeviceReset();
        EXPECT_EQ(ret, UPTKSuccess);
    }

    ret = UPTKSetDevice(0);
    EXPECT_EQ(ret, UPTKSuccess);

    free(h_src);
    free(h_rcv);
    free(d_src);
    free(d_rcv);
}
