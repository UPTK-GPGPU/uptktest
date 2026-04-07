#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

TEST(cudaDevice,cudaDeviceGetByPCIBusIdTest){
    char pciBusId[13];//定义一个长为13的字符串
    int deviceCount = 0;
    UPTKError_t ret1 = UPTKSuccess;
    UPTKError_t ret2 = UPTKSuccess;
    UPTKError_t ret3 = UPTKSuccess;
    UPTKGetDeviceCount(&deviceCount);
    EXPECT_TRUE(deviceCount != 0);
    for (int i = 0; i < deviceCount; i++) {
        int pciBusID = -1;
        int pciDeviceID = -1;
        int pciDomainID = -1;
        int tempPciBusId = -1;
        int tempDeviceId = -1;
        ret1 = UPTKDeviceGetPCIBusId(&pciBusId[0], 13, i);//返回设备的一种总线标准总线标识字符串，重载以采取int设备标识。
        //pciBusId=0000:3d:00.0, 16进制3d为61 ，pciDomainID：pciBusID：pciDeviceID（0,61,0）
        sscanf(pciBusId, "%04x:%02x:%02x", &pciDomainID, &pciBusID, &pciDeviceID);
        ret2 = UPTKDeviceGetAttribute(&tempPciBusId, UPTKDevAttrPciBusId, i);
        EXPECT_EQ(ret1, UPTKSuccess);
        EXPECT_EQ(ret2, UPTKSuccess);
        EXPECT_EQ(pciBusID , tempPciBusId);
       
        ret3 = UPTKDeviceGetByPCIBusId(&tempDeviceId, pciBusId);
        EXPECT_EQ(ret3, UPTKSuccess);
        EXPECT_EQ(tempDeviceId, i);
    }
}