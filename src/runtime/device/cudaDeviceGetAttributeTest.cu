#include <stdio.h>  
#include <gtest/gtest.h>   
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>


UPTKError_t test_hipDeviceGetAttribute(int deviceId, UPTKDeviceAttr attr,
                                      int expectedValue = -1) {
    int value = 0;
    //std::cout << "Test UPDeviceGetAttribute attribute " << attr;
    //if (expectedValue != -1) {
        //std::cout << " expected value " << expectedValue;
    //}
    UPTKError_t e = UPTKDeviceGetAttribute(&value, attr, deviceId);
    //std::cout << " actual value " << value << std::endl;
    if ((expectedValue != -1) && value != expectedValue) {
        std::cout << "Test UPDeviceGetAttribute attribute " << attr;
        std::cout << " expected value " << expectedValue;
        std::cout << " actual value " << value << std::endl;
        std::cout << "fail" << std::endl;
        return UPTKErrorInvalidValue;
    }
    return UPTKSuccess;
}

UPTKError_t test_hipDeviceGetHdpAddress(int deviceId, UPTKDeviceAttr attr,
                                       uint32_t* expectedValue = (uint32_t*)0xdeadbeef) {
    uint32_t* value = 0;
    //std::cout << "Test hipDeviceGetHdpAddress attribute " << attr;
    //if (expectedValue != (uint32_t*)0xdeadbeef) {
    //    std::cout << " expected value " << expectedValue;
    //}
    UPTKError_t e = UPTKDeviceGetAttribute((int*) &value, attr, deviceId);
    //std::cout << " actual value " << value << std::endl;
    if ((expectedValue != (uint32_t*)0xdeadbeef) && value != expectedValue) {
        std::cout << "Test hipDeviceGetHdpAddress attribute " << attr;
        std::cout << " expected value " << expectedValue;
        std::cout << " actual value " << value << std::endl;
        std::cout << "fail" << std::endl;
        return UPTKErrorInvalidValue;
    }
    return UPTKSuccess;
}

TEST(cudaDevice, cudaDeviceGetAttributeTest) {
    UPTKError_t ret = UPTKSuccess;
    int deviceId;
    ret = UPTKGetDevice(&deviceId);
    EXPECT_EQ(ret, UPTKSuccess);
    UPTKDeviceProp props={0};
    ret = UPTKGetDeviceProperties(&props, deviceId);
    EXPECT_EQ(ret, UPTKSuccess);    
    //printf("info: running on device #%d %s\n", deviceId, props.name);
   
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxThreadsPerBlock,
                                     props.maxThreadsPerBlock);  
    EXPECT_EQ(ret, UPTKSuccess);                              
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxBlockDimX,
                                     props.maxThreadsDim[0]);
    EXPECT_EQ(ret, UPTKSuccess);                                  
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxBlockDimY,
                                     props.maxThreadsDim[1]); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxBlockDimZ,
                                     props.maxThreadsDim[2]); 
    EXPECT_EQ(ret, UPTKSuccess);                                 
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxGridDimX, props.maxGridSize[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxGridDimY, props.maxGridSize[1]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxGridDimZ, props.maxGridSize[2]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxSharedMemoryPerBlock,
                                     props.sharedMemPerBlock);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrTotalConstantMemory,
                                     props.totalConstMem);  
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrWarpSize, props.warpSize);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrClockRate, props.clockRate);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrGlobalMemoryBusWidth,
                                     props.memoryBusWidth);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMultiProcessorCount,
                                     props.multiProcessorCount);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrIsMultiGpuBoard,
                                     props.isMultiGpuBoard); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrComputeMode, props.computeMode); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrL2CacheSize, props.l2CacheSize);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxThreadsPerMultiProcessor,
                                     props.maxThreadsPerMultiProcessor);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrComputeCapabilityMajor,
                                     props.major); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrComputeCapabilityMinor,
                                     props.minor); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrConcurrentKernels,
                                     props.concurrentKernels); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrPciBusId, props.pciBusID); 
    EXPECT_EQ(ret, UPTKSuccess);    
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrPciDeviceId, props.pciDeviceID); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxSharedMemoryPerMultiprocessor,
                                     props.sharedMemPerMultiprocessor); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrIntegrated, props.integrated); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxTexture1DWidth, props.maxTexture1D);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxTexture2DWidth, props.maxTexture2D[0]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxTexture2DHeight, props.maxTexture2D[1]); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxTexture3DWidth, props.maxTexture3D[0]); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxTexture3DHeight, props.maxTexture3D[1]); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxTexture3DDepth, props.maxTexture3D[2]);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrCooperativeLaunch, props.cooperativeLaunch); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrCooperativeMultiDeviceLaunch, props.cooperativeMultiDeviceLaunch); 
    EXPECT_EQ(ret, UPTKSuccess);
// #ifndef __HIP_PLATFORM_NVCC__
//     ret = test_hipDeviceGetHdpAddress(deviceId, cudaDevAttrHdpMemFlushCntl, props.hdpMemFlushCntl); 
//     EXPECT_EQ(ret, UPTKSuccess);
//     ret = test_hipDeviceGetHdpAddress(deviceId, cudaDevAttrHdpRegFlushCntl, props.hdpRegFlushCntl); 
//     EXPECT_EQ(ret, UPTKSuccess);  
// #endif
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrMaxPitch, props.memPitch); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrTextureAlignment, props.textureAlignment);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrKernelExecTimeout, props.kernelExecTimeoutEnabled); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrCanMapHostMemory, props.canMapHostMemory); 
    EXPECT_EQ(ret, UPTKSuccess);
    ret = test_hipDeviceGetAttribute(deviceId, UPTKDevAttrEccEnabled, props.ECCEnabled); 
    
};




