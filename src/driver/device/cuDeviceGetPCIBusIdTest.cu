#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

TEST(cuDevice,cuDeviceGetPCIBusIdTest){
    char pciBusId[13];//定义一个长为13的字符串
    int deviceCount = 0;
    CUresult ret1 = CUDA_SUCCESS;
    CUresult ret2 = CUDA_SUCCESS;
    cudaGetDeviceCount(&deviceCount);
    EXPECT_TRUE(deviceCount != 0);
    for (int i = 0; i < deviceCount; i++) {
        int pciBusID = -1;
        int pciDeviceID = -1;
        int pciDomainID = -1;
        int tempPciBusId = -1;
        ret1 = cuDeviceGetPCIBusId(&pciBusId[0], 13, (CUdevice)i);//返回设备的一种总线标准总线标识字符串，重载以采取int设备标识。
        //pciBusId=0000:3d:00.0, 16进制3d为61 ，pciDomainID：pciBusID：pciDeviceID（0,61,0）
        sscanf(pciBusId, "%04x:%02x:%02x", &pciDomainID, &pciBusID, &pciDeviceID);
        ret2 = cuDeviceGetAttribute(&tempPciBusId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, (CUdevice)i);
        EXPECT_EQ(ret1, CUDA_SUCCESS);
        EXPECT_EQ(ret2, CUDA_SUCCESS);
        EXPECT_EQ(pciBusID , tempPciBusId);
        //cudaDeviceAttributePciBusId=21
        // pciBusID=61
        // tempPciBusId=61
        // pciBusID=136
        // tempPciBusId=136
    }
}