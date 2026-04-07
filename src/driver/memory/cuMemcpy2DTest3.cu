/*
@BUG #33861 cuMemcpy2D func_error - DCUToolkit - Runtime
@Add Case For HIP-TEST
*/
#include <stdio.h>
#include <gtest/gtest.h>
#include <gtest/test_common.h>
#include <stddef.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

size_t N = 4 * 1024 * 1024;
char memsetval = 0x42;
int memsetD32val = 0xDEADBEEF;
short memsetD16val = 0xDEAD;
char memsetD8val = 0xDE;
int iterations = 1;
unsigned blocksPerCU = 6;  // to hide latency
unsigned threadsPerBlock = 256;
int p_gpuDevice = 0;
unsigned p_verbose = 0;
int p_tests = -1; /*which tests to run. Interpretation is left to each test.  default:all*/
#ifdef _WIN64
const char* HIP_VISIBLE_DEVICES_STR = "HIP_VISIBLE_DEVICES=";
const char* CUDA_VISIBLE_DEVICES_STR = "CUDA_VISIBLE_DEVICES=";
const char* PATH_SEPERATOR_STR = "\\";
#else
const char* HIP_VISIBLE_DEVICES_STR = "HIP_VISIBLE_DEVICES";
const char* CUDA_VISIBLE_DEVICES_STR = "CUDA_VISIBLE_DEVICES";
const char* PATH_SEPERATOR_STR = "/";
#endif
namespace CudaTest {


double elapsed_time(long long startTimeUs, long long stopTimeUs) {
    return ((double)(stopTimeUs - startTimeUs)) / ((double)(1000));
}


int parseSize(const char* str, size_t* output) {
    char* next;
    *output = strtoull(str, &next, 0);
    int l = strlen(str);
    if (l) {
        char c = str[l - 1];  // last char.
        if ((c == 'k') || (c == 'K')) {
            *output *= 1024;
        }
        if ((c == 'm') || (c == 'M')) {
            *output *= (1024 * 1024);
        }
        if ((c == 'g') || (c == 'G')) {
            *output *= (1024 * 1024 * 1024);
        }
    }
    return 1;
}


int parseUInt(const char* str, unsigned int* output) {
    char* next;
    *output = strtoul(str, &next, 0);
    return !strlen(next);
}


int parseInt(const char* str, int* output) {
    char* next;
    *output = strtol(str, &next, 0);
    return !strlen(next);
}


int parseStandardArguments(int argc, char* argv[], bool failOnUndefinedArg) {
    int extraArgs = 1;
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];

        if (!strcmp(arg, " ")) {
            // skip NULL args.
        } else if (!strcmp(arg, "--N") || (!strcmp(arg, "-N"))) {
            if (++i >= argc || !CudaTest::parseSize(argv[i], &N)) {
                failed("Bad N size argument");
            }
        } else if (!strcmp(arg, "--threadsPerBlock")) {
            if (++i >= argc || !CudaTest::parseUInt(argv[i], &threadsPerBlock)) {
                failed("Bad threadsPerBlock argument");
            }
        } else if (!strcmp(arg, "--blocksPerCU")) {
            if (++i >= argc || !CudaTest::parseUInt(argv[i], &blocksPerCU)) {
                failed("Bad blocksPerCU argument");
            }
        } else if (!strcmp(arg, "--memsetval")) {
            int ex;
            if (++i >= argc || !CudaTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetval argument");
            }
            memsetval = ex;
        } else if (!strcmp(arg, "--memsetD32val")) {
            int ex;
            if (++i >= argc || !CudaTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetD32val argument");
            }
            memsetD32val = ex;
        } else if (!strcmp(arg, "--memsetD16val")) {
            int ex;
            if (++i >= argc || !CudaTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetD16val argument");
            }
            memsetD16val = ex;
        } else if (!strcmp(arg, "--memsetD8val")) {
            int ex;
            if (++i >= argc || !CudaTest::parseInt(argv[i], &ex)) {
                failed("Bad memsetD8val argument");
            }
            memsetD8val = ex;
        } else if (!strcmp(arg, "--iterations") || (!strcmp(arg, "-i"))) {
            if (++i >= argc || !CudaTest::parseInt(argv[i], &iterations)) {
                failed("Bad iterations argument");
            }

        } else if (!strcmp(arg, "--gpu") || (!strcmp(arg, "-gpuDevice")) || (!strcmp(arg, "-g"))) {
            if (++i >= argc || !CudaTest::parseInt(argv[i], &p_gpuDevice)) {
                failed("Bad gpuDevice argument");
            }

        } else if (!strcmp(arg, "--verbose") || (!strcmp(arg, "-v"))) {
            if (++i >= argc || !CudaTest::parseUInt(argv[i], &p_verbose)) {
                failed("Bad verbose argument");
            }
        } else if (!strcmp(arg, "--tests") || (!strcmp(arg, "-t"))) {
            if (++i >= argc || !CudaTest::parseInt(argv[i], &p_tests)) {
                failed("Bad tests argument");
            }

        } else {
            if (failOnUndefinedArg) {
                failed("Bad argument '%s'", arg);
            } else {
                argv[extraArgs++] = argv[i];
            }
        }
    };

    return extraArgs;
}


unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N) {
    int device;
    CUDACHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDACHECK(cudaGetDeviceProperties(&props, device));

    unsigned blocks = props.multiProcessorCount * blocksPerCU;
    if (blocks * threadsPerBlock > N) {
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    }

    return blocks;
}


}  // namespace CudaTest
inline void initMemCpyParam2D(CUDA_MEMCPY2D &ins, const size_t dpitch,
                            const size_t spitch, const size_t width,
                            const size_t height, CUmemorytype dstType,
                            CUmemorytype srcType) {
  ins.srcXInBytes=0;
  ins.srcY=0;
  ins.srcPitch=spitch;
  ins.dstXInBytes=0;
  ins.dstY=0;
  ins.dstPitch=dpitch;
  ins.WidthInBytes=width;
  ins.Height=height;
  ins.dstMemoryType= dstType;
  ins.srcMemoryType= srcType;
}

TEST(cuMemory,cuMemcpy2Dtest3){
    size_t numW = 512 * 3;
    size_t numH = 512;
    bool usePinnedHost = 1;
    size_t width = numW;
    size_t sizeElements = width * numH;


    char *A_d, *B_d, *C_d, *D_d;
    char *A_h, *B_h, *C_h;

    size_t pitch_A, pitch_B, pitch_C, pitch_D;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<char>();
    CudaTest::initArrays2DPitch<char>(&A_d, &B_d, &C_d, &pitch_A, &pitch_B, &pitch_C, numW, numH);
    CudaTest::initArraysForHost<char>(&A_h, &B_h, &C_h, numW * numH, usePinnedHost);
    unsigned blocks = CudaTest::setNumBlocks(blocksPerCU, threadsPerBlock, numW * numH);

    CUDACHECK(cudaMemcpy2D(A_d, pitch_A, A_h, width, width, numH, cudaMemcpyHostToDevice));
    CUDA_MEMCPY2D ins;
    initMemCpyParam2D(ins,pitch_B,width,width,numH,CU_MEMORYTYPE_DEVICE,CU_MEMORYTYPE_HOST);
    ins.dstDevice = (CUdeviceptr)B_d;
    ins.srcHost   = B_h;
    CUDA_DRIVER_CHECK(cuMemcpy2D(&ins));

    //testdevicetodevice  B_d to C_d
    initMemCpyParam2D(ins,pitch_C,pitch_B,width,numH,CU_MEMORYTYPE_DEVICE,CU_MEMORYTYPE_DEVICE);
    ins.dstDevice = (CUdeviceptr)C_d;
    ins.srcDevice  = (CUdeviceptr)B_d;
    CUDA_DRIVER_CHECK(cuMemcpy2D(&ins));


    //hipLaunchKernelGGL(CudaTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d, B_d, C_d,
    //                pitch_C * numH);
    CudaTest::vectorADD<<<dim3(blocks), dim3(threadsPerBlock)>>>(A_d, B_d, C_d,
                    pitch_C * numH);
    CUDACHECK(cudaMemcpy2D(C_h, width, C_d, pitch_C, width, numH, cudaMemcpyDeviceToHost));

    initMemCpyParam2D(ins,width,pitch_C,width,numH,CU_MEMORYTYPE_HOST,CU_MEMORYTYPE_DEVICE);
    ins.srcDevice = (CUdeviceptr)C_d;
    ins.dstHost   = C_h;
    CUDA_DRIVER_CHECK(cuMemcpy2D(&ins));


    //test copy 2d part 512x512 to { 110, 134, 100, 242 }
    CUDACHECK(cudaMallocPitch((void**)&D_d, &pitch_D, 512 * 3, 512));

    ins.srcXInBytes = 110 * 3;
    ins.srcY = 134;
    ins.srcPitch=pitch_C;
    ins.srcMemoryType= CU_MEMORYTYPE_DEVICE;
    ins.srcDevice = (CUdeviceptr)C_d;

    ins.dstXInBytes=101*3;
    ins.dstY=13;
    ins.dstPitch=pitch_D;
    ins.dstMemoryType= CU_MEMORYTYPE_DEVICE;
    ins.dstDevice = (CUdeviceptr)D_d;

    ins.WidthInBytes=100 * 3;
    ins.Height=242;

    CUDA_DRIVER_CHECK(cuMemcpy2D(&ins));
    char *C_d_off = C_d + 134 * pitch_C + 110 * 3;
    ins.srcXInBytes = 0;
    ins.srcY = 0;
    ins.srcPitch=pitch_C;
    ins.srcMemoryType= CU_MEMORYTYPE_DEVICE;
    ins.srcDevice = (CUdeviceptr)C_d_off;

    ins.dstXInBytes=0;
    ins.dstY=0;
    ins.dstPitch=pitch_D;
    ins.dstMemoryType= CU_MEMORYTYPE_DEVICE;
    ins.dstDevice = (CUdeviceptr)D_d;

    ins.WidthInBytes=100 * 3;
    ins.Height=242;

    CUDA_DRIVER_CHECK(cuMemcpy2D(&ins));
    CUDACHECK(cudaDeviceSynchronize());
    CudaTest::checkVectorADD(A_h, B_h, C_h, numW * numH);
    CudaTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, usePinnedHost);
}