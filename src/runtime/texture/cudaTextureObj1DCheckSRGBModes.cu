/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <gtest/gtest.h>
#include <gtest/test_common.h>
#include "texture_helper.h"

template <bool normalizedCoords>
__global__ void tex1DRGBAKernel(float4 *outputData, UPTKTextureObject_t textureObject,
                            int width, float offsetX)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    outputData[x] = tex1D<float4>(textureObject,
                                  normalizedCoords ? (x + offsetX) / width : x + offsetX);
}

__global__ void tex1DRGBAKernelFetch(float4 *outputData, UPTKTextureObject_t textureObject, float offsetX)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    outputData[x] = tex1Dfetch<float4>(textureObject, int(x + offsetX));
}

template <UPTKTextureAddressMode addressMode, UPTKTextureFilterMode filterMode, UPTKResourceType resType,
          bool normalizedCoords, bool sRGB = false>
static void runTest(const int width, const float offsetX = 0)
{
    constexpr float uCharMax = UCHAR_MAX;
    unsigned int size = width * sizeof(uchar4);
    uchar4 *hData = (uchar4 *)malloc(size);
    memset(hData, 0, size);
    for (int j = 0; j < width; j++)
    {
        hData[j].x = static_cast<unsigned char>(j);
        hData[j].y = static_cast<unsigned char>(j + 10);
        hData[j].z = static_cast<unsigned char>(j + 20);
        hData[j].w = static_cast<unsigned char>(j + 30);
    }

    UPTKChannelFormatDesc channelDesc = UPTKCreateChannelDesc<uchar4>();
    uchar4 *cudaBuff = nullptr;
    UPTKArray *UPTKArray = nullptr;
    UPTKResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    if (resType == UPTKResourceTypeArray)
    {
        CUDACHECK(UPTKMallocArray(&UPTKArray, &channelDesc, width));
        CUDACHECK(
            UPTKMemcpy2DToArray(UPTKArray, 0, 0, hData, size, size, 1,
                               UPTKMemcpyHostToDevice));
        resDesc.resType = UPTKResourceTypeArray; // Will call tex1D in kernel
        resDesc.res.array.array = UPTKArray;
    }
    else if (resType == UPTKResourceTypeLinear)
    {
        if (normalizedCoords || filterMode == UPTKFilterModeLinear || addressMode == UPTKAddressModeWrap || addressMode == UPTKAddressModeMirror)
        {
            free(hData);
            FAIL()<<"One or more unexpected parameters for UPTKResourceTypeLinear";
        }
        CUDACHECK(UPTKMalloc((void **)&cudaBuff, size));
        CUDACHECK(UPTKMemcpy(cudaBuff, hData, size, UPTKMemcpyHostToDevice));
        resDesc.resType = UPTKResourceTypeLinear; // Will call tex1Dfetch in kernel
        resDesc.res.linear.devPtr = cudaBuff;
        resDesc.res.linear.sizeInBytes = size;
        resDesc.res.linear.desc = channelDesc;
    }
    else
        FAIL() << "Unexpected resource type " << resType;

    // Specify texture object parameters
    UPTKTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = addressMode;
    texDesc.addressMode[1] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.readMode = UPTKReadModeNormalizedFloat;
    texDesc.normalizedCoords = normalizedCoords;
    texDesc.sRGB = sRGB ? 1 : 0;

    // Create texture object
    UPTKTextureObject_t textureObject = 0;
    auto ret = UPTKCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);
// #if HT_AMD
    if (ret == UPTKErrorInvalidValue && resType == UPTKResourceTypeLinear)
    {
        free(hData);
        CUDACHECK(UPTKFree(cudaBuff));
        // CudaTest::HIP_SKIP_TEST("sRGB is not supported for UPTKResourceTypeLinear type on AMD devices");
        SUCCEED() << "sRGB is not supported for UPTKResourceTypeLinear type on AMD devices";
        return;
    }
// #endif
    CUDACHECK(ret);

    float4 *dData = nullptr;
    size = width * sizeof(float4);
    CUDACHECK(UPTKMalloc((void **)&dData, size));

    dim3 dimBlock(16, 1, 1);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, 1, 1);

    if (resType == UPTKResourceTypeArray)
    {
        // cudaLaunchKernelGGL(tex1DRGBAKernel<normalizedCoords>, dimGrid, dimBlock,
        //                    0, 0, dData, textureObject, width, offsetX);
        tex1DRGBAKernel<normalizedCoords><<<dimGrid, dimBlock>>>(dData, textureObject, width, offsetX);
        CUDACHECK(UPTKGetLastError());
    }
    else
    {
        // cudaLaunchKernelGGL(tex1DRGBAKernelFetch, dimGrid, dimBlock,
        //                    0, 0, dData, textureObject, offsetX);
        tex1DRGBAKernelFetch<<<dimGrid, dimBlock>>>(dData, textureObject, offsetX);
        CUDACHECK(UPTKGetLastError());
    }

    CUDACHECK(UPTKDeviceSynchronize());
    size = width * sizeof(float4);
    float4 *hInputData = (float4 *)malloc(size);  // CPU expected values
    float4 *hOutputData = (float4 *)malloc(size); // GPU output values
    memset(hInputData, 0, size);
    memset(hOutputData, 0, size);

    for (int j = 0; j < width; j++)
    {
        hInputData[j].x = hData[j].x / uCharMax;
        hInputData[j].y = hData[j].y / uCharMax;
        hInputData[j].z = hData[j].z / uCharMax;
        hInputData[j].w = hData[j].w / uCharMax;
    }

    CUDACHECK(UPTKMemcpy(hOutputData, dData, size, UPTKMemcpyDeviceToHost));

    bool result = true;

    for (int j = 0; j < width; j++)
    {
        float4 cpuExpected =
            getExpectedValue<float4, addressMode, filterMode, sRGB>(width, offsetX + j, hInputData);
        float4 gpuOutput = hOutputData[j];
        if (sRGB)
        {
            // CTS will map to sRGP before comparison, so we do so
            cpuExpected = cudaSRGBMap(cpuExpected);
            gpuOutput = cudaSRGBMap(gpuOutput);
        }
        // Convert from [0, 1] back to [0, 255]
        gpuOutput *= uCharMax;
        cpuExpected *= uCharMax;
        if (!cudaTextureSamplingVerify<float4, filterMode, sRGB>(gpuOutput,
                                                                cpuExpected))
        {
            WARN(
                "Mismatch at (" << offsetX + j << ") GPU output : "
                                << gpuOutput.x << ", " << gpuOutput.y << ", " << gpuOutput.z << ", " << gpuOutput.w << ", "
                                << " CPU expected: "
                                << cpuExpected.x << ", " << cpuExpected.y << ", " << cpuExpected.z << ", " << cpuExpected.w << "\n");
            result = false;
            goto line1;
        }
    }

line1:
    CUDACHECK(UPTKDestroyTextureObject(textureObject));
    CUDACHECK(UPTKFree(dData));
    if (UPTKArray)
        CUDACHECK(UPTKFreeArray(UPTKArray));
    if (cudaBuff)
        CUDACHECK(UPTKFree(cudaBuff));
    free(hData);
    free(hOutputData);
    free(hInputData);
    REQUIRE(result);
}

TEST(cutexture, cudaTextureObj1DCheckSRGBModes)
{

    //SECTION("RGBA 1D UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, false>(255, -3.9);
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, false>(255, 4.4);
    }

    //SECTION("RGBA 1D UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, false>(255, -8.5);
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, false>(255, 12.5);
    }

    //SECTION("RGBA 1D UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, false>(255, -0.41);
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, false>(255, 4);
    }

#if HT_AMD
    // nvidia RTX2070 has problem in this mode
    //SECTION("RGBA 1D UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, false>(255, 0);
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, false>(255, 12.1);
    }
#endif

    //SECTION("RGBA 1D UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, true>(255, -3.1);
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, true>(255, 4.2);
    }

    //SECTION("RGBA 1D UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, true>(255, -8.15);
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, true>(255, 12.35);
    }

    //SECTION("RGBA 1D UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, true>(255, -3.1);
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, true>(255, 4.2);
    }

#if HT_AMD
    // nvidia RTX2070 has problem in this mode
    //SECTION("RGBA 1D UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, true>(255, 0);
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, true>(255, -6.7);
    }
#endif



    //SECTION("SRGBA 1D UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, false, true>(255, -3.9);
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, false, true>(255, 4.4);
    }

    //SECTION("SRGBA 1D UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, false, true>(255, -8.5);
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, false, true>(255, 12.5);
    }

    //SECTION("SRGBA 1D UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, false, true>(255, -0.4);
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, false, true>(255, 4);
    }

#if HT_AMD
    // nvidia RTX2070 has problem in this mode
    //SECTION("SRGBA 1D UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, regularCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, false, true>(255, 0);
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, false, true>(255, 12.5);
    }
#endif

    //SECTION("SRGBA 1D UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, true, true>(255, -1.3);
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeArray, true, true>(255, 4.1);
    }

    //SECTION("SRGBA 1D UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, true, true>(255, -8.5);
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeArray, true, true>(255, 12.5);
    }

    //SECTION("SRGBA 1D UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, true, true>(255, -3);
        runTest<UPTKAddressModeClamp, UPTKFilterModeLinear, UPTKResourceTypeArray, true, true>(255, 4);
    }
#if HT_AMD
    // nvidia RTX2070 has problem in this mode
    //SECTION("SRGBA 1D UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, normalizedCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, true, true>(255, 0);
        runTest<UPTKAddressModeBorder, UPTKFilterModeLinear, UPTKResourceTypeArray, true, true>(255, 12.35);
    }
#endif



    // //SECTION("RGBA 1D UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeLinear, regularCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeLinear, false, false>(255);
    }

    // //SECTION("RGBA 1D UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeLinear, regularCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeLinear, false, false>(255);
    }

    // //SECTION("SRGBA 1D UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeLinear, regularCoords")
    {
        runTest<UPTKAddressModeClamp, UPTKFilterModePoint, UPTKResourceTypeLinear, false, true>(255);
    }

    // //SECTION("SRGBA 1D UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeLinear, regularCoords")
    {
        runTest<UPTKAddressModeBorder, UPTKFilterModePoint, UPTKResourceTypeLinear, false, true>(255);
    }
}
