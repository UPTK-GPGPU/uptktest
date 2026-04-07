#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


TEST(cudaMemory,cudaPerformanceUnpinD2H_default){
    int numDevice = 0;
    char *buffer = NULL;
    char *hostBuffer = NULL;
    char *hostBufferB = NULL;
    struct timeval start;
    struct timeval end;
    long long diff = 0;
    int num = 1;

    char **tempbuffer = NULL;

    int bufferSize = 1;//2的30次方
    //int bufferSize = 2147483648;//2的31次方
    UPTKError_t ret = UPTKSuccess;

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

    ret = UPTKMemcpy(tempbuffer[0], hostBuffer, bufferSize, UPTKMemcpyDefault);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpy(hostBufferB, tempbuffer[0], bufferSize, UPTKMemcpyDefault);
    EXPECT_EQ(ret, UPTKSuccess);
    //std::cout << "buffer=" << (void *)buffer << std::endl;
    for(int i=0; i< num;i++)
    {
        //ret = UPTKMemcpy(tempbuffer[i], hostBuffer, bufferSize, UPTKMemcpyHostToDevice);
        gettimeofday(&start,NULL);
        ret = UPTKMemcpy(hostBufferB, tempbuffer[i], bufferSize, UPTKMemcpyDefault);
        EXPECT_EQ(ret, UPTKSuccess);
        gettimeofday(&end,NULL);
        long long diff2 = ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec));
        //std::cout << "The cost time :" << diff2<<"(us), size: "<<bufferSize<<",index: "<<i<< std::endl;
        diff += diff2;
    }
    std::cout << "The cost time :" << (double)((double)diff/(double)num)<<"(us), size: "<<bufferSize<< std::endl;
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

