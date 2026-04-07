#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaMemory,cudaPerformanceUnpinD2H_default_1){
    int numDevice = 0;
    char *buffer = NULL;
    char *hostBuffer = NULL;
    char *hostBufferB = NULL;
    UPTKError_t ret = UPTKSuccess;
    UPTKEvent_t start, stop;
    ret = UPTKEventCreate(&start);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKEventCreate(&stop);
    EXPECT_EQ(ret, UPTKSuccess); 

    long long diff = 0;
    int num = 1;

    char **tempbuffer = NULL;

    int bufferSize = 1;//2的30次方
    //int bufferSize = 2147483648;//2的31次方

    hostBuffer = (char *)malloc(bufferSize);
    ASSERT_TRUE(hostBuffer!=NULL);

    for(int i=0; i<bufferSize; i++)
    {
	    hostBuffer[i]   = i;
    }

    hostBufferB = (char *)malloc(bufferSize);
    ASSERT_TRUE(hostBufferB!=NULL);
    
    tempbuffer = (char **)malloc(sizeof(char *) * num);
    for (int i = 0; i<num; i++)
    {
       ret = UPTKMalloc(&tempbuffer[i], bufferSize);
       EXPECT_EQ(ret, UPTKSuccess);
    }

    //UPTKPointerAttributes attribs;
    //memset(&attribs, 0, sizeof(UPTKPointerAttributes));

    //ret = UPTKPointerGetAttributes(&attribs,(void *)tempbuffer[i]);
    //EXPECT_EQ(ret, UPTKSuccess);
    //EXPECT_EQ((char *)attribs.devicePointer,buffer);

    UPTKStream_t pstream;
    ret = UPTKStreamCreate(&pstream);
    EXPECT_EQ(ret, UPTKSuccess);


    ret = UPTKMemcpyAsync(tempbuffer[0], hostBuffer, bufferSize, UPTKMemcpyDefault, pstream);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpyAsync(hostBufferB, tempbuffer[0], bufferSize, UPTKMemcpyDefault,pstream);
    EXPECT_EQ(ret, UPTKSuccess);
    //std::cout << "buffer=" << (void *)buffer << std::endl;
    
    UPTKStreamSynchronize(pstream);

    ret = UPTKEventRecord(start, 0);
    EXPECT_EQ(ret, UPTKSuccess); 

    //ret = UPTKMemcpyAsync(tempbuffer[i], hostBuffer, bufferSize, UPTKMemcpyHostToDevice);
    ret = UPTKMemcpyAsync(hostBufferB, tempbuffer[0], bufferSize, UPTKMemcpyDefault,pstream);
    EXPECT_EQ(ret, UPTKSuccess);


    ret = UPTKEventRecord(stop, 0);
    EXPECT_EQ(ret, UPTKSuccess); 
    ret = UPTKEventSynchronize(stop);
    EXPECT_EQ(ret, UPTKSuccess); 

    float execution_time;
    ret = UPTKEventElapsedTime(&execution_time, start, stop);
    EXPECT_EQ(ret, UPTKSuccess); 

    std::cout << "The cost time :" << (execution_time)*1000/num<<"(us), size: "<<bufferSize<< std::endl;
    
    for(int i=0; i<bufferSize; i++)
    {    
        EXPECT_EQ(hostBuffer[i], hostBufferB[i]);
    }
    
    free(hostBuffer);
    free(hostBufferB);
    
    for (int i = 0; i<num; i++)
    {
        ret = UPTKFree(tempbuffer[i]);
        EXPECT_EQ(ret, UPTKSuccess);	
    }
    free(tempbuffer);
}

