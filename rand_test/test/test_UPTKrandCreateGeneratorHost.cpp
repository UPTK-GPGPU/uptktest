/*
 * Auto-generated smoke test for UPTKrandCreateGeneratorHost (rand_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rand_test/generate_rand_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_rand.h>
#include <stdint.h>
#include <stdio.h>


static UPTKrandStatus_t rand_bind_gpu(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return (UPTKrandStatus_t)1;
    if (cudaSetDevice(0) != cudaSuccess)
        return (UPTKrandStatus_t)1;
    return UPTKRAND_STATUS_SUCCESS;
}

static UPTKrandStatus_t rand_setup_core(
    UPTKrandGenerator_t *generator,
    void **dev_scratch,
    UPTKStream_t *stream_id,
    int create_generator)
{
    UPTKrandStatus_t st = rand_bind_gpu();
    if (st != UPTKRAND_STATUS_SUCCESS) return st;

    if (cudaMalloc(dev_scratch, 65536) != cudaSuccess)
        return (UPTKrandStatus_t)1;

    if (cudaStreamCreate(stream_id) != cudaSuccess) {
        cudaFree(*dev_scratch);
        *dev_scratch = nullptr;
        return (UPTKrandStatus_t)1;
    }

    *generator = (UPTKrandGenerator_t)(uintptr_t)0;

    if (create_generator) {
        st = UPTKrandCreateGenerator(generator, UPTKRAND_RNG_PSEUDO_DEFAULT);
        if (st != UPTKRAND_STATUS_SUCCESS) return st;
        st = UPTKrandSetStream(*generator, *stream_id);
        if (st != UPTKRAND_STATUS_SUCCESS) return st;
    }

    return UPTKRAND_STATUS_SUCCESS;
}

static void rand_teardown_core(
    UPTKrandGenerator_t generator,
    void *dev_scratch,
    UPTKStream_t stream_id)
{
    if (generator)
        UPTKrandDestroyGenerator(generator);
    if (dev_scratch)
        cudaFree(dev_scratch);
    if (stream_id)
        cudaStreamDestroy(stream_id);
}


int main(void)
{
    UPTKrandGenerator_t generator{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
    UPTKrandStatus_t err;

    err = rand_bind_gpu();
    if (err != UPTKRAND_STATUS_SUCCESS) {
        printf("test_skip: UPTKrandCreateGeneratorHost bind failed\n");
        return 0;
    }
    if (cudaMalloc(&dev_scratch, 65536) != cudaSuccess || cudaStreamCreate(&stream_id) != cudaSuccess) {
        printf("test_skip: UPTKrandCreateGeneratorHost cuda alloc/stream\n");
        return 0;
    }

    err = UPTKrandCreateGeneratorHost(&generator, (UPTKrandRngType_t)UPTKRAND_RNG_PSEUDO_DEFAULT);
    printf("UPTKrandCreateGeneratorHost -> %d\n", (int)err);
    if (generator)
        UPTKrandDestroyGenerator(generator);
    cudaFree(dev_scratch);
    cudaStreamDestroy(stream_id);
    printf("test_UPTKrandCreateGeneratorHost PASS\n");
    return 0;
}
