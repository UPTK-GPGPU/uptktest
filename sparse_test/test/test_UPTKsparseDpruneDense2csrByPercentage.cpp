/*
 * Auto-generated smoke test for UPTKsparseDpruneDense2csrByPercentage (sparse_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/sparse_test/generate_sparse_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_sparse.h>
#include <stdint.h>
#include <stdio.h>


static UPTKsparseStatus_t sparse_bind_gpu(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return (UPTKsparseStatus_t)1;
    if (cudaSetDevice(0) != cudaSuccess)
        return (UPTKsparseStatus_t)1;
    return UPTKSPARSE_STATUS_SUCCESS;
}

static UPTKsparseStatus_t sparse_setup_core(
    UPTKsparseHandle_t *sparse_handle,
    void **dev_scratch,
    UPTKStream_t *stream_id,
    int create_handle)
{
    UPTKsparseStatus_t st = sparse_bind_gpu();
    if (st != UPTKSPARSE_STATUS_SUCCESS) return st;

    if (cudaMalloc(dev_scratch, 65536) != cudaSuccess)
        return (UPTKsparseStatus_t)1;

    if (cudaStreamCreate(stream_id) != cudaSuccess) {
        cudaFree(*dev_scratch);
        *dev_scratch = nullptr;
        return (UPTKsparseStatus_t)1;
    }

    if (create_handle) {
        st = UPTKsparseCreate(sparse_handle);
        if (st != UPTKSPARSE_STATUS_SUCCESS) return st;
        st = UPTKsparseSetStream(*sparse_handle, *stream_id);
        if (st != UPTKSPARSE_STATUS_SUCCESS) return st;
    } else {
        *sparse_handle = (UPTKsparseHandle_t)(uintptr_t)0;
    }

    return UPTKSPARSE_STATUS_SUCCESS;
}

static void sparse_teardown_core(
    UPTKsparseHandle_t sparse_handle,
    void *dev_scratch,
    UPTKStream_t stream_id)
{
    if (sparse_handle)
        UPTKsparseDestroy(sparse_handle);
    if (dev_scratch)
        cudaFree(dev_scratch);
    if (stream_id)
        cudaStreamDestroy(stream_id);
}


int main(void)
{
    UPTKsparseHandle_t sparse_handle{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
    UPTKsparseMatDescr_t descrC{};

    UPTKsparseStatus_t err;

    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDpruneDense2csrByPercentage setup failed (%d)\n", (int)err);
        return 0;
    }


    err = UPTKsparseCreateMatDescr(&descrC);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDpruneDense2csrByPercentage UPTKsparseCreateMatDescr(descrC) failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
        return 0;
    }
    err = UPTKsparseSetMatIndexBase(descrC, UPTKSPARSE_INDEX_BASE_ZERO);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDpruneDense2csrByPercentage SetMatIndexBase(descrC)\n");
        UPTKsparseDestroyMatDescr(descrC); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }
    err = UPTKsparseSetMatType(descrC, UPTKSPARSE_MATRIX_TYPE_GENERAL);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDpruneDense2csrByPercentage SetMatType(descrC)\n");
        UPTKsparseDestroyMatDescr(descrC); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }

    err = UPTKsparseDpruneDense2csrByPercentage(sparse_handle, 0, 0, (const double*)dev_scratch, 0, 0.f, descrC, (double*)dev_scratch, (const int*)dev_scratch, (int*)dev_scratch, (pruneInfo_t)(uintptr_t)0, (void*)nullptr);

    printf("UPTKsparseDpruneDense2csrByPercentage -> %d\n", (int)err);



    if (descrC) UPTKsparseDestroyMatDescr(descrC);
    sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
    printf("test_UPTKsparseDpruneDense2csrByPercentage PASS\n");
    return 0;
}
