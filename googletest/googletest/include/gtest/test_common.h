/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * File is intended to C and CPP compliant hence any CPP specic changes
 * should be added into CPP section
 *
 */

#ifdef __cplusplus
    #include <iostream>
    #include <iomanip>
    #if __CUDACC__
        #include <sys/time.h>
    #else
        #include <chrono>
    #endif
#endif

// ************************ GCC section **************************
#include <stddef.h>




#define HC __attribute__((hc))


#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define passed()                                                                                   \
    printf("%sPASSED!%s\n", KGRN, KNRM);                                                           \
    exit(0);

#define failed(...)                                                                                \
    printf("%serror: ", KRED);                                                                     \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("error: TEST FAILED\n%s", KNRM);                                                        \
    abort();

#define warn(...)                                                                                  \
    printf("%swarn: ", KYEL);                                                                      \
    printf(__VA_ARGS__);                                                                           \
    printf("\n");                                                                                  \
    printf("warn: TEST WARNING\n%s", KNRM);

#define CUDA_PRINT_STATUS(status)                                                                   \
    std::cout << cudaGetErrorName(status) << " at line: " << __LINE__ << std::endl;

#define CUDACHECK(error)                                                                            \
    {                                                                                              \
        cudaError_t localError = error;                                                             \
        if ((localError != cudaSuccess) && (localError != cudaErrorPeerAccessAlreadyEnabled)) {      \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, cudaGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                          \
    }

#define CUDA_DRIVER_CHECK(error)                                                                            \
    {                                                                                              \
        CUresult localError = error;                                                               \
        const char* ptr_string;                                                                    \
        cuGetErrorString(localError, &ptr_string);                                                            \
        if ((localError != CUDA_SUCCESS) && (localError != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)) {                                                         \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, ptr_string,                    \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("API returned error code.");                                                    \
        }                                                                                         \
    }

#define NVRTC_CHECK(error)                                                                            \
    {                                                                                              \
        nvrtcResult localError = error;                                                             \
        if (localError != NVRTC_SUCCESS) {      \
            printf("%serror: '%s'(%d) from %s at %s:%d%s\n", KRED, nvrtcGetErrorString(localError),  \
                   localError, #error, __FILE__, __LINE__, KNRM);                                  \
            failed("NVRTC API returned error code.");                                                    \
        }                                                                                          \
    }
    
#define CUDAASSERT(condition)                                                                       \
    if (!(condition)) {                                                                            \
        failed("%sassertion %s at %s:%d%s \n", KRED, #condition, __FILE__, __LINE__, KNRM);        \
    }


#define CUDACHECK_API(API_CALL, EXPECTED_ERROR)                                                     \
    {                                                                                              \
        cudaError_t _e = (API_CALL);                                                                \
        if (_e != (EXPECTED_ERROR)) {                                                              \
            failed("%sAPI '%s' returned %d(%s) but test expected %d(%s) at %s:%d%s \n", KRED,      \
                   #API_CALL, _e, cudaGetErrorName(_e), EXPECTED_ERROR,                             \
                   cudaGetErrorName(EXPECTED_ERROR), __FILE__, __LINE__, KNRM);                     \
        }                                                                                          \
    }

#ifdef _WIN64
#include <tchar.h>
#define aligned_alloc(x,y) _aligned_malloc(y,x)
#define aligned_free(x) _aligned_free(x)
#define popen(x,y) _popen(x,y)
#define pclose(x) _pclose(x)
#define setenv(x,y,z) _putenv_s(x,y)
#define unsetenv _putenv
#else
#define aligned_free(x) free(x)
#endif

// standard command-line variables:
extern size_t N;
extern char memsetval;
extern int memsetD32val;
extern short memsetD16val;
extern char memsetD8val;
extern int iterations;
extern unsigned blocksPerCU;
extern unsigned threadsPerBlock;
extern int p_gpuDevice;
extern unsigned p_verbose;
extern int p_tests;
extern const char* HIP_VISIBLE_DEVICES_STR;
extern const char* CUDA_VISIBLE_DEVICES_STR;
extern const char* PATH_SEPERATOR_STR;

// ********************* CPP section *********************
#ifdef __cplusplus

#ifdef __HIP_PLATFORM_HCC
#define TYPENAME(T) typeid(T).name()
#else
#define TYPENAME(T) "?"
#endif

namespace CudaTest {

// Returns the current system time in microseconds
inline long long get_time() {
#if __CUDACC__
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch()
        /std::chrono::microseconds(1);
#endif
}

double elapsed_time(long long startTimeUs, long long stopTimeUs);

int parseSize(const char* str, size_t* output);
int parseUInt(const char* str, unsigned int* output);
int parseInt(const char* str, int* output);
int parseStandardArguments(int argc, char* argv[], bool failOnUndefinedArg);

unsigned setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N);

template<typename T> // pointer type
void checkArray(T hData, T hOutputData, size_t width, size_t height,size_t depth)
{
   for (int i = 0; i < depth; i++) {
      for (int j = 0; j < height; j++) {
          for (int k = 0; k < width; k++) {
              int offset = i*width*height + j*width + k;
              if (hData[offset] != hOutputData[offset]) {
                  std::cerr << '[' << i << ',' << j << ',' << k << "]:" << hData[offset] << "----" << hOutputData[offset]<<"  ";
                  failed("mistmatch at:%d %d %d",i,j,k);
              }
          }
       }
   }
}


template <typename T>
bool check_res(T* check_mem, int test_range) {
    int set_size = 1;
    int offset = 1;
    while(true){
        for(int i = 0; i < set_size; i++)
        {
            if(check_mem[i + offset] != 1)
            {
                printf("check error at %d value mast be 1\n", i + offset);
                return false;
            }
        }

        offset += set_size + 1;
        set_size++;
        if(offset + set_size > test_range) {
            for(int i = offset; i < test_range; i++)
            {
                 if(check_mem[offset - 1] == 1)
                 {
                      printf("check error at %d value mast be 0 at 2\n", offset - 1);
                      return false;
                 }
            }
            return true;
        }
        if(check_mem[offset - 1] == 1)
        {
             printf("check error at %d value mast be 0\n", offset - 1);
             return false;
        }
    }
}


template<typename T> 
void checkArray(T input, T output, size_t height, size_t width)
{
    for(int i=0; i<height; i++ ){
        for(int j=0; j<width; j++ ){
            int offset = i*width + j;
            if( input[offset] !=  output[offset] ){
                 std::cerr << '[' << i << ',' << j << ',' << "]:" << input[offset] << "----" << output[offset]<<"  ";
                 failed("mistmatch at:%d %d",i,j);
            }
        }
    }
}


template <typename T>
__global__ void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < NELEM; i += stride) {
        C_d[i] = A_d[i] + B_d[i];
    }
}


template <typename T>
__global__ void vectorADDReverse(const T* A_d, const T* B_d, T* C_d,
                                 size_t NELEM) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
        C_d[i] = A_d[i] + B_d[i];
    }
}


template <typename T>
__global__ void addCount(const T* A_d, T* C_d, size_t NELEM, int count) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    // Deliberately do this in an inefficient way to increase kernel runtime
    for (int i = 0; i < count; i++) {
        for (size_t i = offset; i < NELEM; i += stride) {
            C_d[i] = A_d[i] + (T)count;
        }
    }
}


template <typename T>
__global__ void addCountReverse(const T* A_d, T* C_d, int64_t NELEM, int count) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    // Deliberately do this in an inefficient way to increase kernel runtime
    for (int i = 0; i < count; i++) {
        for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
            C_d[i] = A_d[i] + (T)count;
        }
    }
}


template <typename T>
__global__ void memsetReverse(T* C_d, T val, int64_t NELEM) {
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (int64_t i = NELEM - stride + offset; i >= 0; i -= stride) {
        C_d[i] = val;
    }
}


template <typename T>
void setDefaultData(size_t numElements, T* A_h, T* B_h, T* C_h) {
    // Initialize the host data:
    for (size_t i = 0; i < numElements; i++) {
        if (A_h) (A_h)[i] = 3.146f + i;  // Pi
        if (B_h) (B_h)[i] = 1.618f + i;  // Phi
        if (C_h) (C_h)[i] = 0.0f + i;
    }
}


template <typename T>
void initArraysForHost(T** A_h, T** B_h, T** C_h, size_t N, bool usePinnedHost = false) {
    size_t Nbytes = N * sizeof(T);

    if (usePinnedHost) {
        if (A_h) {
            CUDACHECK(cudaMallocHost((void**)A_h, Nbytes));
        }
        if (B_h) {
            CUDACHECK(cudaMallocHost((void**)B_h, Nbytes));
        }
        if (C_h) {
            CUDACHECK(cudaMallocHost((void**)C_h, Nbytes));
        }
    } else {
        if (A_h) {
            *A_h = (T*)malloc(Nbytes);
            CUDAASSERT(*A_h != NULL);
        }

        if (B_h) {
            *B_h = (T*)malloc(Nbytes);
            CUDAASSERT(*B_h != NULL);
        }

        if (C_h) {
            *C_h = (T*)malloc(Nbytes);
            CUDAASSERT(*C_h != NULL);
        }
    }

    setDefaultData(N, A_h ? *A_h : NULL, B_h ? *B_h : NULL, C_h ? *C_h : NULL);
}


template <typename T>
void initArrays(T** A_d, T** B_d, T** C_d, T** A_h, T** B_h, T** C_h, size_t N,
                bool usePinnedHost = false) {
    size_t Nbytes = N * sizeof(T);

    if (A_d) {
        CUDACHECK(cudaMalloc(A_d, Nbytes));
    }
    if (B_d) {
        CUDACHECK(cudaMalloc(B_d, Nbytes));
    }
    if (C_d) {
        CUDACHECK(cudaMalloc(C_d, Nbytes));
    }

    initArraysForHost(A_h, B_h, C_h, N, usePinnedHost);
}


template <typename T>
void freeArraysForHost(T* A_h, T* B_h, T* C_h, bool usePinnedHost) {
    if (usePinnedHost) {
        if (A_h) {
            CUDACHECK(cudaFreeHost(A_h));
        }
        if (B_h) {
            CUDACHECK(cudaFreeHost(B_h));
        }
        if (C_h) {
            CUDACHECK(cudaFreeHost(C_h));
        }
    } else {
        if (A_h) {
            free(A_h);
        }
        if (B_h) {
            free(B_h);
        }
        if (C_h) {
            free(C_h);
        }
    }
}

template <typename T>
void freeArrays(T* A_d, T* B_d, T* C_d, T* A_h, T* B_h, T* C_h, bool usePinnedHost) {
    if (A_d) {
        CUDACHECK(cudaFree(A_d));
    }
    if (B_d) {
        CUDACHECK(cudaFree(B_d));
    }
    if (C_d) {
        CUDACHECK(cudaFree(C_d));
    }

    freeArraysForHost(A_h, B_h, C_h, usePinnedHost);
}

//#if defined(__HIP_PLATFORM_HCC__)
template <typename T>
void initArrays2DPitch(T** A_d, T** B_d, T** C_d, size_t* pitch_A, size_t* pitch_B, size_t* pitch_C,
                       size_t numW, size_t numH) {
    if (A_d) {
        CUDACHECK(cudaMallocPitch((void**)A_d, pitch_A, numW * sizeof(T), numH));
    }
    if (B_d) {
        CUDACHECK(cudaMallocPitch((void**)B_d, pitch_B, numW * sizeof(T), numH));
    }
    if (C_d) {
        CUDACHECK(cudaMallocPitch((void**)C_d, pitch_C, numW * sizeof(T), numH));
    }

    CUDAASSERT(*pitch_A == *pitch_B);
    CUDAASSERT(*pitch_A == *pitch_C)
}

inline void initCUDAArrays(cudaArray** A_d, cudaArray** B_d, cudaArray** C_d,
                          const cudaChannelFormatDesc* desc, const size_t numW, const size_t numH,
                          const unsigned int flags) {
    if (A_d) {
        CUDACHECK(cudaMallocArray(A_d, desc, numW, numH, flags));
    }
    if (B_d) {
        CUDACHECK(cudaMallocArray(B_d, desc, numW, numH, flags));
    }
    if (C_d) {
        CUDACHECK(cudaMallocArray(C_d, desc, numW, numH, flags));
    }
}
//#endif

// Assumes C_h contains vector add of A_h + B_h
// Calls the test "failed" macro if a mismatch is detected.
template <typename T>
size_t checkVectorADD(T* A_h, T* B_h, T* result_H, size_t N, bool expectMatch = true,
                      bool reportMismatch = true) {
    size_t mismatchCount = 0;
    size_t firstMismatch = 0;
    size_t mismatchesToPrint = 10;
    for (size_t i = 0; i < N; i++) {
        T expected = A_h[i] + B_h[i];
        if (result_H[i] != expected) {
            if (mismatchCount == 0) {
                firstMismatch = i;
            }
            mismatchCount++;
            if ((mismatchCount <= mismatchesToPrint) && expectMatch) {
                std::cout << std::fixed << std::setprecision(32);
                std::cout << "At " << i << std::endl;
                std::cout << "  Computed:" << result_H[i] << std::endl;
                std::cout << "  Expected:" << expected << std::endl;
            }
        }
    }

    if (reportMismatch) {
        if (expectMatch) {
            if (mismatchCount) {
                failed("%zu mismatches ; first at index:%zu\n", mismatchCount, firstMismatch);
            }
        } else {
            if (mismatchCount == 0) {
                failed("expected mismatches but did not detect any!");
            }
        }
    }

    return mismatchCount;
}


// Assumes C_h contains vector add of A_h + B_h
// Calls the test "failed" macro if a mismatch is detected.
template <typename T>
void checkTest(T* expected_H, T* result_H, size_t N, bool expectMatch = true) {
    size_t mismatchCount = 0;
    size_t firstMismatch = 0;
    size_t mismatchesToPrint = 10;
    for (size_t i = 0; i < N; i++) {
        if (result_H[i] != expected_H[i]) {
            if (mismatchCount == 0) {
                firstMismatch = i;
            }
            mismatchCount++;
            if ((mismatchCount <= mismatchesToPrint) && expectMatch) {
                std::cout << std::fixed << std::setprecision(32);
                std::cout << "At " << i << std::endl;
                std::cout << "  Computed:" << result_H[i] << std::endl;
                std::cout << "  Expected:" << expected_H[i] << std::endl;
            }
        }
    }

    if (expectMatch) {
        if (mismatchCount) {
            fprintf(stderr, "%zu mismatches ; first at index:%zu\n", mismatchCount, firstMismatch);
            // failed("%zu mismatches ; first at index:%zu\n", mismatchCount, firstMismatch);
        }
    } else {
        if (mismatchCount == 0) {
            failed("expected mismatches but did not detect any!");
        }
    }
}


//---
struct Pinned {
    static const bool isPinned = true;
    static const char* str() { return "Pinned"; };

    static void* Alloc(size_t sizeBytes) {
        void* p;
        CUDACHECK(cudaMallocHost((void**)&p, sizeBytes));
        return p;
    };
};


//---
struct Unpinned {
    static const bool isPinned = false;
    static const char* str() { return "Unpinned"; };

    static void* Alloc(size_t sizeBytes) {
        void* p = malloc(sizeBytes);
        CUDAASSERT(p);
        return p;
    };
};


struct Memcpy {
    static const char* str() { return "Memcpy"; };
};

struct MemcpyAsync {
    static const char* str() { return "MemcpyAsync"; };
};


template <typename C>
struct MemTraits;


template <>
struct MemTraits<Memcpy> {
    static void Copy(void* dest, const void* src, size_t sizeBytes, cudaMemcpyKind kind,
                     cudaStream_t stream) {
        CUDACHECK(cudaMemcpy(dest, src, sizeBytes, kind));
    }
};


template <>
struct MemTraits<MemcpyAsync> {
    static void Copy(void* dest, const void* src, size_t sizeBytes, cudaMemcpyKind kind,
                     cudaStream_t stream) {
        CUDACHECK(cudaMemcpyAsync(dest, src, sizeBytes, kind, stream));
    }
};

};  // namespace CudaTest
#endif //__cplusplus
